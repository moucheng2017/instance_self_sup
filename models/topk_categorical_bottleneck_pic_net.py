import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def soft_categorical_assignments(logits):
    """Deterministic soft categorical bottleneck assignments."""
    return F.softmax(logits, dim=1)


def column_normalize_assignments(assignments, eps=1e-6):
    """One-shot batch-column normalization for [B, C] latent assignments.

    Active columns are rescaled to equal total mass while preserving the batch's
    total assignment mass. Empty columns remain zero; this is deliberately not
    iterative Sinkhorn normalization.
    """
    column_mass = assignments.sum(dim=0, keepdim=True)
    active = column_mass.gt(eps)
    if not bool(active.any()):
        return assignments
    total_mass = assignments.sum().detach()
    target_mass = total_mass / active.float().sum().clamp_min(1.0)
    scale = torch.where(active, target_mass / column_mass.clamp_min(eps), torch.zeros_like(column_mass))
    return assignments * scale


def categorical_kl_to_uniform(probs, eps=1e-8):
    num_classes = probs.shape[-1]
    probs = probs.clamp_min(eps)
    return (probs * (probs.log() + math.log(float(num_classes)))).sum(dim=-1)


def categorical_entropy(probs, eps=1e-8):
    probs = probs.clamp_min(eps)
    return -(probs * probs.log()).sum(dim=-1)


class TopKCategoricalBottleneckPICNet(nn.Module):
    """PIC-style instance prediction through a soft categorical bottleneck.

    Flow:
      image -> backbone feature [B,D]
            -> latent assigner [D,C] -> q(c|x)
            -> soft categorical assignments [B,C]
            -> optional one-shot column normalization
            -> decoder/MLP -> instance logits [B,N]
    """

    def __init__(
        self,
        num_classes,
        backbone=None,
        num_latent_classes=100,
        decoder_hidden_dim=None,
        balance_weight=1.0,
        entropy_weight=0.0,
        target_entropy=None,
        column_normalize=True,
        normalize_features=True,
        normalize_assigner=True,
    ):
        super().__init__()
        if backbone is None:
            raise ValueError("backbone must be provided explicitly.")
        if num_classes is None or int(num_classes) < 1:
            raise ValueError("TopKCategoricalBottleneckPICNet requires at least 1 pseudo class.")

        self.backbone = backbone
        self.num_classes = int(num_classes)
        self.num_latent_classes = int(num_latent_classes)
        self.balance_weight = float(balance_weight)
        self.entropy_weight = float(entropy_weight)
        self.target_entropy = None if target_entropy is None else float(target_entropy)
        self.column_normalize = bool(column_normalize)
        self.normalize_features = bool(normalize_features)
        self.normalize_assigner = bool(normalize_assigner)

        if self.num_latent_classes < 1:
            raise ValueError("num_latent_classes must be positive.")
        if self.balance_weight < 0 or self.entropy_weight < 0:
            raise ValueError("regularization weights must be non-negative.")
        if self.entropy_weight > 0 and self.target_entropy is None:
            raise ValueError("target_entropy must be set when entropy_weight > 0.")

        self.latent_assigner = nn.Linear(backbone.output_dim, self.num_latent_classes, bias=False)
        hidden_dim = 0 if decoder_hidden_dim is None else int(decoder_hidden_dim)
        if hidden_dim > 0:
            self.decoder = nn.Sequential(
                nn.Linear(self.num_latent_classes, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, self.num_classes),
            )
        else:
            self.decoder = nn.Linear(self.num_latent_classes, self.num_classes)

    def _latent_logits(self, features):
        if self.normalize_features:
            features = F.normalize(features, dim=1)
        if self.normalize_assigner:
            weight = F.normalize(self.latent_assigner.weight, dim=1)
            return features @ weight.t()
        return self.latent_assigner(features)

    def forward(self, images, pseudo_labels):
        if pseudo_labels.ndim != 1:
            raise ValueError("pseudo_labels must have shape [batch_size].")

        features = self.backbone(images)
        latent_logits = self._latent_logits(features)
        assignments = soft_categorical_assignments(latent_logits)
        decoder_input = column_normalize_assignments(assignments) if self.column_normalize else assignments
        instance_logits = self.decoder(decoder_input)
        labels = pseudo_labels.long()

        loss_instance = F.cross_entropy(instance_logits, labels)
        mean_probs = assignments.mean(dim=0)
        loss_balance = categorical_kl_to_uniform(mean_probs)
        entropy = categorical_entropy(assignments)
        if self.target_entropy is None:
            loss_entropy = entropy.new_zeros([])
        else:
            loss_entropy = (entropy - self.target_entropy).pow(2).mean()
        loss = (
            loss_instance
            + self.balance_weight * loss_balance
            + self.entropy_weight * loss_entropy
        )

        with torch.no_grad():
            predictions = instance_logits.argmax(dim=1)
            acc = predictions.eq(labels).float().mean()
            latent_perplexity = categorical_entropy(mean_probs).exp()
            mean_entropy = entropy.mean()

        return {
            "loss": loss,
            "loss_instance": loss_instance.detach(),
            "loss_balance": loss_balance.detach(),
            "loss_entropy": loss_entropy.detach(),
            "acc": acc,
            "latent_perplexity": latent_perplexity,
            "latent_entropy": mean_entropy.detach(),
        }
