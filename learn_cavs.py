import torch
import pickle
import numpy as np
import pytorch_lightning as pl

from sklearn.svm import SVC

from configs import get_config
from models import construct_model
from data import construct_data_module
from utils import init_logger, construct_callbacks


@torch.no_grad()
def get_embeddings(loader, model):
    activations = None
    for image, _, _, _, _ in loader:
        image = image.cuda()
        batch_act = model(image).squeeze().detach().cpu().numpy()
        if activations is None:
            activations = batch_act
        else:
            activations = np.concatenate([activations, batch_act], axis=0)
    return activations


def get_cav(X_train, y_train, C):
    svm = SVC(C=C, kernel="linear")
    svm.fit(X_train, y_train)
    cav_info = {"vector": svm.coef_, "intercept": svm.intercept_, "norm": np.linalg.norm(svm.coef_)}
    return cav_info


def learn_concept_bank(pos_loader, neg_loader, backbone, reg):
    pos_act = get_embeddings(pos_loader, backbone)
    neg_act = get_embeddings(neg_loader, backbone)

    X_train = np.concatenate([pos_act, neg_act], axis=0)
    y_train = np.concatenate([np.ones(pos_act.shape[0]), np.zeros(neg_act.shape[0])], axis=0)
    concept_info = get_cav(X_train, y_train, reg)

    return concept_info


if __name__ == "__main__":
    config = get_config()
    pl.seed_everything(config["seed"])
    data_module = construct_data_module(config)
    clm_model = construct_model(config, data_module.imbalance_weight)
    logger = init_logger(config)
    callbacks = construct_callbacks(config)

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=[config["device"]],
        max_epochs=config["max_epochs"],
        check_val_every_n_epoch=config["val_every_n_epochs"],
        log_every_n_steps=5,
        logger=logger,
        callbacks=callbacks,
    )
    trainer.fit(clm_model, datamodule=data_module)

    # filter out misaligned concepts
    batch_results = trainer.predict(clm_model, data_module.val_dataloader())
    alpha = torch.cat(list(map(lambda x: x[0], batch_results)))
    beta = torch.cat(list(map(lambda x: x[1], batch_results)))
    uncertainty = (2 / (alpha + beta)).mean(dim=0)
    uncertain_concepts = torch.nonzero(uncertainty > config["misaligned_threshold"]).squeeze().tolist()
    print(uncertain_concepts)

    # learn cavs for mialigned concepts
    backbone = clm_model.pre_concept_model.cuda()
    backbone.eval()

    config["dataset"] = "cav"
    config["uncertain_concepts"] = uncertain_concepts
    data_module = construct_data_module(config)

    print("Concepts to rectify:")
    cav_dict = {}
    for concept_name, loaders in data_module.concept_loaders.items():
        pos_loader, neg_loader = loaders["pos"], loaders["neg"]
        cav_info = learn_concept_bank(pos_loader, neg_loader, backbone, config["svm_reg"])
        cav_dict[concept_name] = cav_info
        print(concept_name)

    cav_path = f"{config['ckpt_save_dir']}/cavs.pkl"
    with open(cav_path, "wb") as f:
        pickle.dump(cav_dict, f)
