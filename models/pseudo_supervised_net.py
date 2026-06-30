import torch.nn as nn
import torch.nn.functional as F
from models.simsiam import prediction_MLP, projection_MLP


class PseudoSupervisedNet(nn.Module):
    def __init__(
        self,
        num_classes,
        backbone,
        l2_normalize=False,
        cosine_softmax=False,
        cosine_scale=16.0,
        projector_dim=512,
    ):
        super().__init__()
        if num_classes is None or int(num_classes) < 1:
            raise ValueError("PseudoSupervisedNet requires at least 1 pseudo class.")
        if float(cosine_scale) <= 0:
            raise ValueError("cosine_scale must be positive.")

        self.backbone = backbone
        self.num_classes = int(num_classes)
        self.l2_norm = bool(l2_normalize)
        self.cosine_softmax = bool(cosine_softmax)
        self.cosine_scale = float(cosine_scale)
        self.proj = projection_MLP(backbone.output_dim, hidden_dim=projector_dim, out_dim=projector_dim)

        pred_hidden = max(projector_dim // 4, 1)
        if self.cosine_softmax:
            self.pred = prediction_MLP(projector_dim, hidden_dim=pred_hidden, out_dim=projector_dim)
            self.classifier = nn.Linear(projector_dim, self.num_classes, bias=False)
        else:
            self.pred = prediction_MLP(projector_dim, hidden_dim=pred_hidden, out_dim=self.num_classes)
            self.classifier = None

    def forward(self, images, pseudo_labels):
        if pseudo_labels.ndim != 1:
            raise ValueError("pseudo_labels must have shape [batch_size].")

        features = self.backbone(images)
        if self.l2_norm:
            features = F.normalize(features, dim=1)

        embeddings_or_logits = self.pred(self.proj(features))
        labels = pseudo_labels.long()

        if self.cosine_softmax:
            embeddings = F.normalize(embeddings_or_logits, dim=1)
            class_weights = F.normalize(self.classifier.weight, dim=1)
            logits = self.cosine_scale * F.linear(embeddings, class_weights)
        else:
            logits = embeddings_or_logits

        loss = F.cross_entropy(logits, labels)
        predictions = logits.argmax(dim=1)
        acc = (predictions == labels).float().mean()

        return {
            "loss": loss,
            "acc": acc,
        }
