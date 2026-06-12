import torch
import torch.nn as nn
import torch.nn.functional as F


class AugmentationBinaryClassifier(nn.Module):
    def __init__(self, backbone=None):
        super().__init__()
        if backbone is None:
            raise ValueError("backbone must be provided explicitly.")
        self.backbone = backbone
        self.classifier = nn.Linear(backbone.output_dim, 1)

    def forward(self, images, pseudo_labels):
        features = self.backbone(images)
        logits = self.classifier(features).squeeze(-1)
        pseudo_labels = pseudo_labels.float()
        loss = F.binary_cross_entropy_with_logits(logits, pseudo_labels)
        probabilities = torch.sigmoid(logits)
        predictions = (probabilities >= 0.5).float()
        accuracy = (predictions == pseudo_labels).float().mean()
        return {
            "loss": loss,
            "binary_acc": accuracy,
            "positive_rate": pseudo_labels.mean(),
            "prob_mean": probabilities.mean(),
        }
