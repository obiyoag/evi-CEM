import os
import clip
import torch
import pandas as pd

from tqdm import tqdm
from PIL import Image


concept_term_dict = {
    "Abscess": ["abscess", "swollen", "pus-filled lump"],
    "Acuminate": ["acuminate"],
    "Atrophy": ["atrophy", "atrophic"],  # 萎缩
    "Black": ["black", "black color"],  # 黑色
    "Blue": ["blue", "blue color"],
    "Brown(Hyperpigmentation)": ["brown(hyperpigmentation)", "hyperpigmented", "hyperpigmentation"],  # 褐色（色素增加）
    "Bulla": ["bulla", "bullae", "blister"],  # 水/气泡
    "Burrow": ["burrow", "scabies"],
    "Comedo": ["comedo", "whitehead", "blackhead"],
    "Crust": ["crust", "dried crust"],  # 痂皮
    "Cyst": ["cyst"],
    "Dome-shaped": ["dome-shaped", "like a dome"],  # 圆顶形
    "Erosion": ["erosion", "erosive", "breakdown of the outer layers", "impetigo"],  # 糜烂
    "Erythema": ["erythema", "redness", "erythematous"],  # 红斑
    "Excoriation": ["excoriation"],
    "Exophytic/Fungating": ["exophytic and fungating", "exophytic", "fungating"],
    "Exudate": ["exudate"],  # 渗出物
    "Fissure": ["fissure", "dry and cracked skin"],
    "Flat topped": ["flat topped"],
    "Friable": ["friable"],  # 易碎的
    "Gray": ["gray"],
    "Induration": ["induration", "edema", "oedema"],
    "Lichenification": ["lichenification", "thickened and leathery"],
    "Macule": ["macule", "freckle", "macular", "lentigo"],
    "Nodule": ["nodule", "nodular", "cyst"],  # 结节
    "Papule": ["papule", "papular"],  # 丘疹
    "Patch": ["patch", "hyperpigmented", "melasma", "vitiligo"],  # 斑
    "Pedunculated": ["pedunculated"],
    "Pigmented": ["pigmented"],
    "Plaque": ["plaque", "dermatitis", "psoriasis"],  # 斑块
    "Poikiloderma": ["poikiloderma", "sun aging"],
    "Purple": ["purple", "purple color"],  # 紫色
    "Purpura/Petechiae": ["purpura and petechiae", "purpura", "petechiae"],
    "Pustule": ["pustule"],  # 脓包
    "Salmon": ["salmon", "salmon patch"],
    "Scale": ["scale", "flaky and scaly", "scaly", "hyperkeratosis"],  # 鳞屑
    "Scar": ["scar", "keloid scars", "hypertrophic scars", "contractures scars", "acne scars"],  # 伤疤
    "Sclerosis": ["sclerosis", "scleroderma", "crest syndrome"],
    "Telangiectasia": ["telangiectasia", "dilated or broken blood vessels"],  # 毛细血管扩张
    "Translucent": ["translucent", "this bump is translucent"],
    "Ulcer": ["ulcer", "ulcerated"],  # 溃疡
    "Umbilicated": ["umbilicated"],
    "Vesicle": ["vesicle", "fluid-containing"],
    "Warty/Papillomatous": ["warty and papillomatous", "warty", "papillomatous"],
    "Wheal": ["wheal", "urticaria"],
    "White(Hypopigmentation)": ["white(hypopigmentation)", "hypopigmentation"],  # 白色（色素减少）
    "Xerosis": ["xerosis", "dry skin", "abnormally dry skin"],
    "Yellow": ["yellow", "yellow color"],  # 黄色
}


def similarity(image_embed, concept_embed_dict, ref_embed, concept_list, temp=0.01):
    ref_sim = (ref_embed @ image_embed.T).mean(dim=1)

    sim_score_list = []
    for concept_name in concept_list:
        concept_embed = concept_embed_dict[concept_name]
        concept_sim = (concept_embed @ image_embed.T).mean(dim=1)
        sim_score = torch.softmax(torch.stack([concept_sim, ref_sim]) / temp, dim=0)[0, :].mean()
        sim_score_list.append(sim_score)
    sim_score = torch.stack(sim_score_list)
    return sim_score


@torch.no_grad()
def get_concept_embed(model, concept_list, device):
    prompt_templates = [
        "This is a skin image of {}",
        "This is a dermatology image of {}",
        "This is an image of {}",
        "This is a {} skin image",
        "This is a {} dermatology image",
        "This is an {} image",
    ]

    concept_embed_dict = {}
    for concept in concept_list:
        concept_terms = concept_term_dict[concept]
        concept_prompts = [[template.format(term) for term in concept_terms] for template in prompt_templates]
        tokenized_concept_prompts = [clip.tokenize(concept_prompt) for concept_prompt in concept_prompts]
        concept_embed = torch.stack([model.encode_text(prompt.to(device)) for prompt in tokenized_concept_prompts])
        concept_embed = concept_embed / concept_embed.norm(dim=2, keepdim=True)
        concept_embed_dict.update({concept: concept_embed})

    return concept_embed_dict


@torch.no_grad()
def get_ref_embed(model, device):
    ref_prompts = [
        ["This is a skin image"],
        ["This is a dermatology image"],
        ["This is an image"],
        ["This is a skin image"],
        ["This is a dermatology image"],
        ["This is an image"],
    ]

    tokenized_ref_prompts = [clip.tokenize(prompt_ref) for prompt_ref in ref_prompts]
    ref_embed = torch.stack([model.encode_text(prompt.to(device)) for prompt in tokenized_ref_prompts])
    ref_embed = ref_embed / ref_embed.norm(dim=2, keepdim=True)

    return ref_embed


if __name__ == "__main__":
    model, preprocessor = clip.load("ViT-L/14", device="cuda", jit=False)
    model.load_state_dict(torch.hub.load_state_dict_from_url("https://aimslab.cs.washington.edu/MONET/weight_clip.pt", map_location="cuda"))
    model.eval()

    # load meta data
    skincon_df = pd.read_csv("data/meta_data/skincon.csv", index_col=0)
    data_df = pd.read_csv("data/meta_data/fitzpatrick17k.csv")

    # filter out the samples which can not be downloaded
    image_names = os.listdir("data/raw_data")
    data_df["md5hash"] = data_df["md5hash"] + ".jpg"
    data_df = data_df.rename(columns={"md5hash": "ImageID"})
    data_df = data_df[data_df["ImageID"].isin(image_names)]

    # filter out images with poor quality
    skincon_df = skincon_df[skincon_df["Do not consider this image"] != 1]
    skincon_df = skincon_df.drop("Do not consider this image", axis=1)

    # merge task label in the concept dataframe
    df = pd.merge(data_df[["ImageID", "three_partition_label"]], skincon_df, on="ImageID", how="outer")
    df = df.dropna(subset=["three_partition_label"])

    concept_list = list(df.columns)[2:]

    concept_embed_dict = get_concept_embed(model, concept_list, "cuda")
    ref_embed = get_ref_embed(model, "cuda")

    meta_label_list = []
    wrong_image_ids = []
    for idx in tqdm(range(len(df))):
        sample = df.iloc[idx, :]

        image_name = sample.ImageID
        meta_label = [image_name]

        if sample.three_partition_label == "non-neoplastic":
            meta_label.append(0)
        elif sample.three_partition_label == "benign":
            meta_label.append(1)
        elif sample.three_partition_label == "malignant":
            meta_label.append(2)

        meta_label.extend(list(sample.iloc[2:]))

        image = Image.open(f"data/raw_data/{image_name}")
        image = preprocessor(image).unsqueeze(0).cuda()

        with torch.no_grad():
            image_embed = model.encode_image(image)
        image_embed = image_embed / image_embed.norm(dim=1, keepdim=True)
        sim_score = similarity(image_embed, concept_embed_dict, ref_embed, concept_list)
        meta_label.extend(list(sim_score.cpu().tolist()))

        meta_label_list.append(meta_label)

    for wrong_image_id in wrong_image_ids:
        df = df.drop(df[df["ImageID"] == wrong_image_id].index)

    clip_concepts = [f"clip_{concept}" for concept in concept_list]
    columns = ["id", "label"] + concept_list + clip_concepts
    new_df = pd.DataFrame(meta_label_list, columns=columns)
    new_df.to_csv(f"data/meta_data/clip_skincon.csv", index=False)
