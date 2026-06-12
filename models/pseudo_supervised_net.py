import torch.nn as nn
import torch.nn.functional as F


class PseudoSupervisedNet(nn.Module):
    def __init__(self, num_classes, backbone=None):
        super().__init__()
        if backbone is None:
            raise ValueError("backbone must be provided explicitly.")
        if num_classes is None or int(num_classes) < 1:
            raise ValueError("PseudoSupervisedNet requires at least 1 pseudo class.")

        self.backbone = backbone
        self.num_classes = int(num_classes)
        self.classifier = nn.Linear(backbone.output_dim, self.num_classes)

    def forward(self, images, pseudo_labels):
        if pseudo_labels.ndim != 1:
            raise ValueError("pseudo_labels must have shape [batch_size].")

        features = self.backbone(images)
        logits = self.classifier(features)
        labels = pseudo_labels.long()

        loss = F.cross_entropy(logits, labels)
        predictions = logits.argmax(dim=1)
        acc = (predictions == labels).float().mean()

        return {
            "loss": loss,
            "acc": acc,
        }
