import torch

from utils import compute_concept_metric
from models.clm import ConceptLearningModel


class Evi_CLM(ConceptLearningModel):
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

        del self.concept_prob_generator
        self.alpha_gen = torch.nn.Linear(2 * emb_size, 1)
        self.beta_gen = torch.nn.Linear(2 * emb_size, 1)
        torch.nn.init.normal_(self.alpha_gen.weight, mean=0, std=1.0)
        torch.nn.init.normal_(self.beta_gen.weight, mean=0, std=1.0)

    def forward(self, x, train=None):
        pre_c = self.pre_concept_model(x)
        alpha, beta = [], []

        for context_gen in self.concept_context_generators:
            context = context_gen(pre_c)
            alpha.append(torch.relu(self.alpha_gen(context)) + 1)
            beta.append(torch.relu(self.beta_gen(context)) + 1)

        alpha, beta = torch.stack(alpha, dim=1).squeeze(), torch.stack(beta, dim=1).squeeze()
        return (alpha, beta)

    def _run_step(self, batch, train=None):
        x, _, c, soft_c, _ = batch
        gamma = self.forward(x)

        if self.train_with_c_gt:
            concept_loss = self.loss_concept(gamma, c)
            kl_loss = self.kl_loss(gamma, c)
        else:
            concept_loss = self.loss_concept(gamma, soft_c)
            kl_loss = self.kl_loss(gamma, torch.where(soft_c > 0.5, 1.0, 0.0))

        lambda_ = min(1.0, self.current_epoch / 10)
        loss = concept_loss + lambda_ * kl_loss

        loss_scalar = loss.detach()
        concept_loss_scalar = concept_loss.detach()
        kl_loss_scalar = kl_loss.detach()

        c_probs = gamma[0] / (gamma[0] + gamma[1])
        c_acc, c_auc, c_f1 = compute_concept_metric(c_probs, c)
        result = {
            "c_acc": c_acc,
            "c_auc": c_auc,
            "c_f1": c_f1,
            "loss": loss_scalar,
            "concept_loss": concept_loss_scalar,
            "kl_loss": kl_loss_scalar,
        }
        return loss, result

    def loss_concept(self, gamma, concept):
        alpha, beta = gamma
        evi_loss = torch.digamma(alpha + beta) - concept * torch.digamma(alpha) - (1 - concept) * torch.digamma(beta)
        evi_loss = torch.mean(self.concept_weight * evi_loss)
        return evi_loss

    def kl_loss(self, gamma, concept):
        alpha, beta = gamma
        alpha_beta_mixure = torch.where(concept.bool(), beta, alpha)
        kl_term = torch.log(alpha_beta_mixure + 1e-7) + (1 - alpha_beta_mixure) / (alpha_beta_mixure + 1e-7)
        kl_loss = torch.mean(self.concept_weight * kl_term)
        return kl_loss
