import torch

from utils import compute_metric
from models.evi_clm import Evi_CLM


class Evidential_CEM(Evi_CLM):
    def __init__(
        self,
        n_concepts,
        n_tasks,
        emb_size=16,
        embedding_activation="leakyrelu",
        c_extractor_arch="resnet34",
        optimizer="adam",
        learning_rate=0.01,
        weight_decay=4e-05,
        momentum=0.9,
        train_with_c_gt=False,
        concept_weight=None,
        interven_prob=0.25,
        concept_loss_weight=1,
    ):
        super().__init__(
            n_concepts,
            emb_size,
            embedding_activation,
            c_extractor_arch,
            optimizer,
            learning_rate,
            weight_decay,
            momentum,
            train_with_c_gt,
            concept_weight,
        )
        self.interven_prob = interven_prob
        self.concept_loss_weight = concept_loss_weight
        self.c2y_model = torch.nn.Sequential(*[torch.nn.Linear(n_concepts * emb_size, n_tasks)])
        self.loss_task = torch.nn.CrossEntropyLoss()

        self.save_hyperparameters()

    def _after_interventions(self, prob, c_true):
        # we will probabilistically intervene in some concepts
        mask = torch.bernoulli(torch.ones(self.n_concepts) * self.interven_prob)
        interven_idxs = torch.tile(mask, (c_true.shape[0], 1)).float().to(prob.device)
        return prob * (1 - interven_idxs) + interven_idxs * c_true

    def forward(self, x, c, train=False):
        pre_c = self.pre_concept_model(x)
        contexts, alpha, beta = [], [], []

        for context_gen in self.concept_context_generators:
            context = context_gen(pre_c)
            contexts.append(context)
            alpha.append(torch.relu(self.alpha_gen(context)) + 1)
            beta.append(torch.relu(self.beta_gen(context)) + 1)

        alpha, beta = torch.stack(alpha, dim=1).squeeze(), torch.stack(beta, dim=1).squeeze()
        c_probs = alpha / (alpha + beta)
        contexts = torch.stack(contexts, dim=1)

        if train and self.interven_prob != 0:
            c_hard = torch.where(c > 0.5, 1.0, 0.0)
            c_probs_mix = self._after_interventions(c_probs, c_true=c_hard)
        else:
            c_probs_mix = c_probs

        contexts_pos = contexts[:, :, : self.emb_size]
        contexts_neg = contexts[:, :, self.emb_size :]
        c_pred = contexts_pos * c_probs_mix.unsqueeze(-1) + contexts_neg * (1 - c_probs_mix.unsqueeze(-1))
        c_pred = c_pred.view((-1, self.emb_size * self.n_concepts))

        y = self.c2y_model(c_pred)

        return (alpha, beta), y

    def _run_step(self, batch, train):
        x, y, c, soft_c, _ = batch
        gamma, y_logits = self.forward(x, c, train)

        task_loss = self.loss_task(y_logits, y)
        task_loss_scalar = task_loss.detach()

        if self.concept_loss_weight != 0:

            if self.train_with_c_gt:
                concept_loss = self.loss_concept(gamma, c)
                kl_loss = self.kl_loss(gamma, c)
            else:
                concept_loss = self.loss_concept(gamma, soft_c)
                kl_loss = self.kl_loss(gamma, torch.where(soft_c > 0.5, 1.0, 0.0))

            concept_loss_scalar = concept_loss.detach()
            kl_loss_scalar = kl_loss.detach()

            lambda_ = min(1.0, self.current_epoch / 10)
            loss = self.concept_loss_weight * (concept_loss + lambda_ * kl_loss) + task_loss
        else:
            loss = task_loss
            concept_loss_scalar = 0.0

        c_probs = gamma[0] / (gamma[0] + gamma[1])
        (c_acc, c_auc, c_f1), (y_acc, y_auc, y_f1) = compute_metric(c_probs, y_logits, c, y)
        result = {
            "c_acc": c_acc,
            "c_auc": c_auc,
            "c_f1": c_f1,
            "y_acc": y_acc,
            "y_auc": y_auc,
            "y_f1": y_f1,
            "concept_loss": concept_loss_scalar,
            "kl_loss": kl_loss_scalar,
            "task_loss": task_loss_scalar,
            "loss": loss.detach(),
            "avg_c_y_acc": (c_acc + y_acc) / 2,
            "avg_c_y_auc": (c_auc + y_auc) / 2,
            "avg_c_y_f1": (c_f1 + y_f1) / 2,
        }
        return loss, result
