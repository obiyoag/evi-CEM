import torch
import numpy as np
import sklearn.metrics

from pytorch_lightning.callbacks import ModelCheckpoint


def init_logger(config):
    if config["logger"] == "wandb":
        import wandb
        from pytorch_lightning.loggers import WandbLogger

        wandb.init(project=config["project_name"], name=config["exp_name"], config=config)
        logger = WandbLogger(name=config["project_name"])
    else:
        logger = False
    return logger


def construct_callbacks(config):
    callbacks = []
    if config["ckpt_saving"]:
        callbacks.append(
            ModelCheckpoint(
                dirpath=config["ckpt_save_dir"],
                filename="best_auc",
                monitor=config["ckpt_save_monitor"],
                mode=config["ckpt_save_mode"],
                save_weights_only=config["save_weights_only"],
                save_last=True,
            )
        )
    return callbacks


def _convert_image_to_rgb(image):
    return image.convert("RGB")


def compute_bin_accuracy(c_pred, y_pred, c_true, y_true):
    c_pred = c_pred.cpu().detach().numpy() > 0.5
    y_probs = y_pred.cpu().detach().numpy()
    y_pred = y_probs > 0.5
    c_true = c_true.cpu().detach().numpy()
    y_true = y_true.cpu().detach().numpy()
    c_accuracy = c_auc = c_f1 = 0
    num_seen = 0
    for i in range(c_true.shape[-1]):
        true_vars = c_true[:, i]
        indices = np.logical_or(true_vars == 1, true_vars == 0).astype(bool)
        if not np.any(indices):
            continue
        num_seen += 1
        true_vars = true_vars[indices]
        pred_vars = c_pred[:, i][indices]
        c_accuracy += sklearn.metrics.accuracy_score(true_vars, pred_vars)
        if len(np.unique(true_vars)) == 1:
            c_auc += sklearn.metrics.accuracy_score(true_vars, pred_vars)
        else:
            c_auc += sklearn.metrics.roc_auc_score(true_vars, pred_vars)
        c_f1 += sklearn.metrics.f1_score(true_vars, pred_vars, average="macro")
    num_seen = num_seen if num_seen else 1
    c_accuracy = c_accuracy / num_seen
    c_auc = c_auc / num_seen
    c_f1 = c_f1 / num_seen
    y_accuracy = sklearn.metrics.accuracy_score(y_true, y_pred)
    if len(np.unique(y_true)) == 1:
        y_auc = sklearn.metrics.accuracy_score(y_true, y_pred)
    else:
        y_auc = sklearn.metrics.roc_auc_score(y_true, y_probs)
    y_f1 = sklearn.metrics.f1_score(y_true, y_pred)
    return (c_accuracy, c_auc, c_f1), (y_accuracy, y_auc, y_f1)


def compute_accuracy(c_pred, y_pred, c_true, y_true):
    if (len(y_pred.shape) < 2) or (y_pred.shape[-1] == 1):
        return compute_bin_accuracy(c_pred, y_pred, c_true, y_true)
    c_pred = (c_pred.cpu().detach().numpy() >= 0.5).astype(np.int32)
    # Doing the following transformation for when labels are not fully certain
    c_true = (c_true.cpu().detach().numpy() > 0.5).astype(np.int32)
    y_probs = torch.nn.Softmax(dim=-1)(y_pred).cpu().detach()
    # used_classes = np.unique(y_true.cpu().detach())
    # y_probs = y_probs[:, sorted(list(used_classes))]
    y_pred = y_pred.argmax(dim=-1).cpu().detach()
    y_true = y_true.cpu().detach()

    c_accuracy = c_auc = c_f1 = 0
    for i in range(c_true.shape[-1]):
        true_vars = c_true[:, i]
        pred_vars = c_pred[:, i]
        c_accuracy += sklearn.metrics.accuracy_score(true_vars, pred_vars) / c_true.shape[-1]

        if len(np.unique(true_vars)) == 1:
            c_auc += np.mean(true_vars == pred_vars) / c_true.shape[-1]
        else:
            c_auc += sklearn.metrics.roc_auc_score(true_vars, pred_vars) / c_true.shape[-1]
        c_f1 += sklearn.metrics.f1_score(true_vars, pred_vars, average="macro") / c_true.shape[-1]

    y_accuracy = sklearn.metrics.accuracy_score(y_true, y_pred)
    try:
        y_auc = sklearn.metrics.roc_auc_score(y_true, y_probs, multi_class="ovo")
    except Exception as e:
        y_auc = 0.0
    try:
        y_f1 = sklearn.metrics.f1_score(y_true, y_pred, average="macro")
    except:
        y_f1 = 0.0
    return (c_accuracy, c_auc, c_f1), (y_accuracy, y_auc, y_f1)


def compute_concept_metric(c_prob, c_true, mean=True):
    c_prob = c_prob.cpu().detach().numpy()
    c_pred = (c_prob >= 0.5).astype(np.int32)
    # Doing the following transformation for when labels are not fully certain
    c_true = (c_true.cpu().detach().numpy() > 0.5).astype(np.int32)

    c_acc, c_auc, c_f1 = [], [], []
    for i in range(c_true.shape[-1]):
        true_vars = c_true[:, i]
        prob_vars = c_prob[:, i]
        pred_vars = c_pred[:, i]
        c_acc.append(sklearn.metrics.accuracy_score(true_vars, pred_vars))
        if len(np.unique(true_vars)) == 1:
            c_auc.append(np.mean(true_vars == pred_vars))
        else:
            c_auc.append(sklearn.metrics.roc_auc_score(true_vars, prob_vars))
        c_f1.append(sklearn.metrics.f1_score(true_vars, pred_vars, average="macro"))

    c_acc, c_auc, c_f1 = torch.tensor(c_acc), torch.tensor(c_auc), torch.tensor(c_f1)
    if mean:
        return c_acc.mean(), c_auc.mean(), c_f1.mean()
    else:
        return c_acc, c_auc, c_f1


def compute_task_metric(y_logit, y_true):
    y_prob = torch.nn.Softmax(dim=-1)(y_logit).cpu().detach()
    y_pred = y_logit.argmax(dim=-1).cpu().detach()
    y_true = y_true.cpu().detach()

    try:
        y_auc = sklearn.metrics.roc_auc_score(y_true, y_prob, multi_class="ovo")
    except Exception as e:
        y_auc = 1.0
    try:
        y_f1 = sklearn.metrics.f1_score(y_true, y_pred, average="macro")
    except:
        y_f1 = 1.0
    y_acc = sklearn.metrics.accuracy_score(y_true, y_pred)

    return y_acc, y_auc, y_f1


def compute_metric(c_prob, y_logit, c_true, y_true):
    c_acc, c_auc, c_f1 = compute_concept_metric(c_prob, c_true)
    y_acc, y_auc, y_f1 = compute_task_metric(y_logit, y_true)
    return (c_acc, c_auc, c_f1), (y_acc, y_auc, y_f1)
