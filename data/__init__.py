from .skincon_datamodules import SkinConDataModule, CAVDataModule, RectifiedDataModule


def construct_data_module(config):
    if config["dataset"] == "skincon":
        data_module = SkinConDataModule(
            data_dir=config["data_dir"],
            batch_size=config["batch_size"],
            train_with_c_gt=config["train_with_c_gt"],
            concept_weight=config["concept_weight"],
        )
    elif config["dataset"] == "cav":
        data_module = CAVDataModule(
            data_dir=config["data_dir"],
            batch_size=config["batch_size"],
            uncertain_concepts=config["uncertain_concepts"],
            sample_num=config["sample_num"],
        )
    elif config["dataset"] == "rectified":
        data_module = RectifiedDataModule(
            data_dir=config["data_dir"],
            batch_size=config["batch_size"],
            train_with_c_gt=config["train_with_c_gt"],
            concept_weight=config["concept_weight"],
            pretrain=config["pretrain"],
            cav_path=config["cav_path"],
        )
    else:
        raise NotImplementedError

    data_module.prepare_data()
    data_module.setup(stage=None)

    return data_module
