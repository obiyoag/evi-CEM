import torch
from .evi_clm import Evi_CLM
from .evi_cem import Evidential_CEM


def construct_model(config, concept_weight):
    if config["arch"] == "evi_cem":
        model = Evidential_CEM(
            n_concepts=len(concept_weight),
            n_tasks=config["num_classes"],
            emb_size=config["emb_size"],
            interven_prob=config["interven_prob"],
            embedding_activation=config["embedding_activation"],
            concept_loss_weight=config["concept_loss_weight"],
            c_extractor_arch=config["c_extractor_arch"],
            learning_rate=config["lr"],
            weight_decay=config["weight_decay"],
            train_with_c_gt=config["train_with_c_gt"],
            concept_weight=concept_weight,
            optimizer=config["optimizer"],
        )
    elif config["arch"] == "evi_clm":
        model = Evi_CLM(
            n_concepts=len(concept_weight),
            emb_size=config["emb_size"],
            embedding_activation=config["embedding_activation"],
            c_extractor_arch=config["c_extractor_arch"],
            learning_rate=config["lr"],
            weight_decay=config["weight_decay"],
            train_with_c_gt=config["train_with_c_gt"],
            concept_weight=concept_weight,
            optimizer=config["optimizer"],
        )
    else:
        raise NotImplementedError

    if config["pretrain"] is not None:
        state_dict = torch.load(config["pretrain"])["state_dict"]
        if "cem" in config["arch"]:
            state_dict.update(
                {
                    "c2y_model.0.weight": model.state_dict()["c2y_model.0.weight"],
                    "c2y_model.0.bias": model.state_dict()["c2y_model.0.bias"],
                }
            )
        model.load_state_dict(state_dict)

    return model
