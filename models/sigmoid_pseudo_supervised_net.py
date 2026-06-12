import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SigmoidPseudoSupervisedNet(nn.Module):
    """Pairwise sigmoid pseudo-supervision over image-index labels.

    A flat sigmoid head would require one logit per dataset image. Instead, this
    model encodes pseudo-label IDs through a learnable embedding table and scores
    image/ID pairs in the current mini-batch. Matching image-index pairs are
    positives; all non-matching pairs in the batch are negatives.
    """

    def __init__(
        self,
        num_classes,
        backbone=None,
        embedding_dim=None,
        normalize=True,
        init_temperature=10.0,
        init_bias=-10.0,
    ):
        super().__init__()
        if backbone is None:
            raise ValueError("backbone must be provided explicitly.")
        if num_classes is None or int(num_classes) < 1:
            raise ValueError("SigmoidPseudoSupervisedNet requires at least 1 pseudo class.")

        self.backbone = backbone
        self.num_classes = int(num_classes)
        self.embedding_dim = int(embedding_dim or backbone.output_dim)
        self.normalize = bool(normalize)

        if self.embedding_dim == backbone.output_dim:
            self.image_projector = nn.Identity()
        else:
            self.image_projector = nn.Linear(backbone.output_dim, self.embedding_dim)

        self.label_embeddings = nn.Embedding(self.num_classes, self.embedding_dim)
        nn.init.normal_(self.label_embeddings.weight, std=0.02)

        if init_temperature <= 0:
            raise ValueError("init_temperature must be positive.")
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(float(init_temperature)))
        self.logit_bias = nn.Parameter(torch.ones([]) * float(init_bias))

    def forward(self, images, pseudo_labels):
        if pseudo_labels.ndim != 1:
            raise ValueError("pseudo_labels must have shape [batch_size].")

        labels = pseudo_labels.long()
        image_embeddings = self.image_projector(self.backbone(images))
        label_embeddings = self.label_embeddings(labels)

        if self.normalize:
            image_embeddings = F.normalize(image_embeddings, dim=1)
            label_embeddings = F.normalize(label_embeddings, dim=1)

        logits = image_embeddings @ label_embeddings.T
        logits = logits * self.logit_scale.exp() + self.logit_bias

        positive_mask = labels[:, None].eq(labels[None, :])
        signed_targets = torch.where(
            positive_mask,
            torch.ones_like(logits),
            -torch.ones_like(logits),
        )
        # Normalize per pair (B² pairs total) so loss magnitude is independent
        # of batch size. Matches the normalization in HierarchicalBalancedVMF-
        # SelfLabelingNet._sigmoid_regularization.
        loss = -F.logsigmoid(signed_targets * logits).mean()

        predictions = labels[logits.argmax(dim=1)]
        acc = predictions.eq(labels).float().mean()

        positive_logits = logits[positive_mask]
        negative_logits = logits[~positive_mask]
        acc_pos = positive_logits.gt(0).float().mean()
        acc_neg = negative_logits.lt(0).float().mean() if negative_logits.numel() else logits.new_tensor(1.0)

        return {
            "loss": loss,
            "acc": acc,
            "acc_pos": acc_pos,
            "acc_neg": acc_neg,
            "logit_scale": self.logit_scale.exp().detach(),
            "logit_bias": self.logit_bias.detach(),
        }
