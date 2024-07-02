import torch
import pytorch_lightning as pl

from torchvision.models import resnet34, ResNet34_Weights

from utils import compute_concept_metric


class ConceptLearningModel(pl.LightningModule):
    def __init__(
        self,
        n_concepts,
        emb_size=16,
        embedding_activation="leakyrelu",
        c_extractor_arch="resnet34",
        optimizer="adam",
        learning_rate=0.01,
        weight_decay=4e-05,
        momentum=0.9,
        train_with_c_gt=False,
        concept_weight=None,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.n_concepts = n_concepts
        self.emb_size = emb_size

        self.optimizer_name = optimizer
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.train_with_c_gt = train_with_c_gt
        self.concept_weight = concept_weight.cuda()

        if c_extractor_arch == "resnet34":
            self.pre_concept_model = resnet34(weights=ResNet34_Weights.DEFAULT)
            backbone_embed_size = list(self.pre_concept_model.modules())[-1].in_features
            self.pre_concept_model.fc = torch.nn.Identity()
        else:
            raise NotImplementedError

        if embedding_activation == "sigmoid":
            embed_act = torch.nn.Sigmoid()
        elif embedding_activation == "leakyrelu":
            embed_act = torch.nn.LeakyReLU()
        elif embedding_activation == "relu":
            embed_act = torch.nn.ReLU()
        elif embedding_activation is None:
            embed_act = torch.nn.Identity()

        self.concept_context_generators = torch.nn.ModuleList()
        self.concept_prob_generator = torch.nn.Linear(2 * emb_size, 1)
        torch.nn.init.xavier_uniform_(self.concept_prob_generator.weight)
        torch.nn.init.constant_(self.concept_prob_generator.bias, 0.0)

        for _ in range(n_concepts):
            concept_context_generator = torch.nn.Sequential(
                torch.nn.Linear(backbone_embed_size, 2 * emb_size), embed_act
            )
            torch.nn.init.xavier_uniform_(concept_context_generator[0].weight)
            torch.nn.init.constant_(concept_context_generator[0].bias, 0.0)
            self.concept_context_generators.append(concept_context_generator)

        self.loss_concept = torch.nn.BCELoss(weight=concept_weight)

    def forward(self, x):
        pre_c = self.pre_concept_model(x)
        c_probs = []

        for context_gen in self.concept_context_generators:
            context = context_gen(pre_c)
            c_probs.append(torch.sigmoid(self.concept_prob_generator(context)))

        c_probs = torch.cat(c_probs, axis=-1)
        return c_probs

    def _run_step(self, batch):
        x, _, c, soft_c, _ = batch
        c_probs = self.forward(x)

        if self.train_with_c_gt:
            loss = self.loss_concept(c_probs, c)
        else:
            loss = self.loss_concept(c_probs, soft_c)

        loss_scalar = loss.detach()

        c_acc, c_auc, c_f1 = compute_concept_metric(c_probs, c)
        result = {"c_acc": c_acc, "c_auc": c_auc, "c_f1": c_f1, "loss": loss_scalar}
        return loss, result

    def training_step(self, batch, batch_idx):
        loss, result = self._run_step(batch, train=True)
        for name, val in result.items():
            if "loss" in name:
                self.log(f"train/loss/{name}", val)
            else:
                self.log(f"train/metric/{name}", val)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        _, result = self._run_step(batch, train=False)
        for name, val in result.items():
            if "loss" in name:
                self.log(f"val/loss/{name}", val)
            else:
                self.log(f"val/metric/{name}", val)
        result = {"val_" + key: val for key, val in result.items()}
        return result

    def test_step(self, batch, batch_idx):
        _, result = self._run_step(batch, train=False)
        for name, val in result.items():
            self.log(f"test/{name}", val, prog_bar=True)
        return result["loss"]

    def predict_step(self, batch, batch_idx):
        x, _, _, _, _ = batch
        c_probs = self.forward(x, train=False)
        return c_probs

    def configure_optimizers(self):
        if self.optimizer_name.lower() == "adam":
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, self.parameters()),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
        elif self.optimizer_name.lower() == "adamw":
            optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, self.parameters()),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
        elif self.optimizer_name.lower() == "sgd":
            optimizer = torch.optim.SGD(
                filter(lambda p: p.requires_grad, self.parameters()),
                lr=self.learning_rate,
                momentum=self.momentum,
                weight_decay=self.weight_decay,
            )
        else:
            raise NotImplementedError
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
            "monitor": "train/loss/loss",
        }
