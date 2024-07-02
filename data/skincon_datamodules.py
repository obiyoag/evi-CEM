import torch
import pickle
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torchvision.transforms as transforms


from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from models.evi_clm import Evi_CLM
from utils import _convert_image_to_rgb


class SkinConDataset(Dataset):
    def __init__(self, data_dir, data_frame, concept_num, transform):
        super().__init__()
        self.data_dir = data_dir
        self.data_frame = data_frame
        self.concept_num = concept_num
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, index):
        sample = self.data_frame.iloc[index, :]

        image = Image.open(f"{self.data_dir}/raw_data/{sample.id}")
        image = self.transform(image)

        concept = torch.FloatTensor(list(sample.iloc[2 : self.concept_num + 2]))
        soft_concept = torch.FloatTensor(list(sample.iloc[self.concept_num + 2 :]))

        return image, sample.label, concept, soft_concept, index


class SkinConDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size, train_with_c_gt, concept_weight):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.train_with_c_gt = train_with_c_gt
        self.concept_weight = concept_weight

        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]  # imagenet norm
        self.aug_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=32 / 255, saturation=(0.5, 1.5)),
                transforms.RandomHorizontalFlip(),
                transforms.CenterCrop(size=224),
                _convert_image_to_rgb,
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        self.noaug_transform = transforms.Compose(
            [
                transforms.Resize(size=256),
                transforms.CenterCrop(size=224),
                _convert_image_to_rgb,
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

    def prepare_data(self):
        df = pd.read_csv(f"{self.data_dir}/meta_data/clip_skincon.csv")

        df_w_gt, df_wo_gt = df[df["Abscess"].notna()], df[df["Abscess"].isna()]

        # only consider concepts with at least 50 images
        concept_stat = df_w_gt.iloc[:, 2:50].sum(axis=0)
        selected_concepts = list(concept_stat[concept_stat > 50].index)
        clip_selected_concepts = [f"clip_{concept}" for concept in selected_concepts]

        self.concept_num = len(selected_concepts)

        df_w_gt = df_w_gt[["id", "label"] + selected_concepts + clip_selected_concepts]
        df_wo_gt = df_wo_gt[["id", "label"] + selected_concepts + clip_selected_concepts]
        df_wo_gt[selected_concepts] = np.where(df_wo_gt[clip_selected_concepts] > 0.5, 1.0, 0.0)

        self.df_w_gt = df_w_gt
        self.df_wo_gt = df_wo_gt

    @property
    def concept_list(self):
        return list(self.df_w_gt.columns)[2 : self.concept_num + 2]

    @property
    def imbalance_weight(self):
        count = self.df_w_gt[self.concept_list].sum().values
        weight = torch.tensor(len(self.df_w_gt) / count)
        if not self.concept_weight:
            weight = torch.ones_like(weight)
        return weight

    def setup(self, stage):
        train_val_df, test_df = train_test_split(self.df_w_gt, test_size=0.2, random_state=42)
        train_df, val_df = train_test_split(train_val_df, test_size=0.25, random_state=42)

        if self.train_with_c_gt:
            self.train_dataset = SkinConDataset(self.data_dir, train_df, self.concept_num, self.aug_transform)
        else:
            self.train_dataset = SkinConDataset(self.data_dir, self.df_wo_gt, self.concept_num, self.aug_transform)

        self.val_dataset = SkinConDataset(self.data_dir, val_df, self.concept_num, self.noaug_transform)
        self.test_dataset = SkinConDataset(self.data_dir, test_df, self.concept_num, self.noaug_transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, self.batch_size, True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, self.batch_size, False, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, self.batch_size, False, num_workers=4)


class CAVDataModule(SkinConDataModule):
    def __init__(self, data_dir, batch_size, uncertain_concepts, sample_num):
        super().__init__(data_dir, batch_size, train_with_c_gt=None, concept_weight=None)
        self.sample_num = sample_num
        self.uncertain_concepts = uncertain_concepts

    def setup(self, stage=None):
        train_val_df, _ = train_test_split(self.df_w_gt, test_size=0.2, random_state=42)
        df = train_val_df.iloc[:, : 2 + self.concept_num]

        concept_loaders = {}
        for concept in [self.concept_list[i] for i in self.uncertain_concepts]:
            pos_df = df[df[concept] == 1].sample(n=self.sample_num, random_state=42)
            neg_df = df[df[concept] == 0].sample(n=self.sample_num, random_state=42)
            pos_ds = SkinConDataset(self.data_dir, pos_df, self.concept_num, self.noaug_transform)
            neg_ds = SkinConDataset(self.data_dir, neg_df, self.concept_num, self.noaug_transform)
            pos_loader = DataLoader(pos_ds, batch_size=self.batch_size, shuffle=False, num_workers=4)
            neg_loader = DataLoader(neg_ds, batch_size=self.batch_size, shuffle=False, num_workers=4)
            concept_loaders[concept] = {"pos": pos_loader, "neg": neg_loader}

        self.concept_loaders = concept_loaders


class RectifiedDataModule(SkinConDataModule):
    def __init__(self, data_dir, batch_size, train_with_c_gt, concept_weight, pretrain, cav_path):
        super().__init__(data_dir, batch_size, train_with_c_gt, concept_weight)

        self.backbone = Evi_CLM.load_from_checkpoint(pretrain).pre_concept_model.cuda()
        self.backbone.eval()

        self.cav_dict = pickle.load(open(cav_path, "rb"))

    def prepare_data(self):
        super().prepare_data()
        ds = SkinConDataset(self.data_dir, self.df_wo_gt, self.concept_num, self.noaug_transform)

        with torch.no_grad():
            img_act = []
            for image, _, _, _, _ in tqdm(DataLoader(ds, self.batch_size, False, num_workers=6)):
                image = image.cuda()
                img_act.append(self.backbone(image).squeeze().detach().cpu())
            img_act = torch.cat(img_act).numpy().T

        for concept, cav in self.cav_dict.items():
            margins = (cav["vector"] @ img_act  + cav["intercept"]) / cav["norm"]
            cav_pred = np.where(margins > 0, 1.0, 0.0).squeeze()
            self.df_wo_gt[f"clip_{concept}"] = cav_pred * self.df_wo_gt[f"clip_{concept}"]
