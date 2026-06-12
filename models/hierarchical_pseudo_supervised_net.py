import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class HierarchicalPseudoSupervisedNet(nn.Module):
    """Hierarchical two-level softmax for large-scale pseudo-supervised learning.

    Instead of a single N-way linear head (which becomes prohibitively large
    for the whole dataset), pseudo-labels are factorised into a (group, item)
    pair using a balanced two-level tree:

        num_groups = ceil(sqrt(N))
        group_size = ceil(N / num_groups)

    Two lightweight classification heads are trained jointly:
        coarse_head : Linear(d, num_groups)  – which group?
        fine_head   : Linear(d, group_size)  – which item within the group?

    The cross-entropy losses are additive, which is equivalent to maximising
    the log-likelihood of the factorised probability P(image) = P_coarse × P_fine.

    Parameter savings (d=512):
        Flat      N=100k : 100,000 × 512 ≈ 51.2 M params
        Hier.     N=100k :    (317+316) × 512 ≈  0.3 M params  (~157× smaller)
    """

    def __init__(self, num_classes, backbone=None, num_groups=None):
        super().__init__()
        if backbone is None:
            raise ValueError("backbone must be provided explicitly.")
        if num_classes is None or int(num_classes) < 1:
            raise ValueError(
                "HierarchicalPseudoSupervisedNet requires at least 1 pseudo class."
            )

        self.backbone = backbone
        self.num_classes = int(num_classes)
        if num_groups is not None:
            if int(num_groups) < 1 or int(num_groups) >= self.num_classes:
                raise ValueError(
                    f"num_groups must be between 1 and num_classes-1, got {num_groups}."
                )
            self.num_groups = int(num_groups)
        else:
            self.num_groups = math.ceil(math.sqrt(self.num_classes))
        self.group_size = math.ceil(self.num_classes / self.num_groups)

        self.coarse_head = nn.Linear(backbone.output_dim, self.num_groups)
        self.fine_head = nn.Linear(backbone.output_dim, self.group_size)

    def forward(self, images, pseudo_labels):
        if pseudo_labels.ndim != 1:
            raise ValueError("pseudo_labels must have shape [batch_size].")

        features = self.backbone(images)

        # Decompose flat index into (group, within-group) coordinates.
        coarse_labels = (pseudo_labels // self.group_size).long()
        fine_labels = (pseudo_labels % self.group_size).long()

        coarse_logits = self.coarse_head(features)
        fine_logits = self.fine_head(features)

        loss_coarse = F.cross_entropy(coarse_logits, coarse_labels)
        loss_fine = F.cross_entropy(fine_logits, fine_labels)
        loss = loss_coarse + loss_fine

        coarse_acc = (coarse_logits.argmax(1) == coarse_labels).float().mean()
        fine_acc = (fine_logits.argmax(1) == fine_labels).float().mean()
        # Joint accuracy: the network correctly predicted *both* levels (exact image ID).
        joint_acc = (
            (coarse_logits.argmax(1) == coarse_labels)
            & (fine_logits.argmax(1) == fine_labels)
        ).float().mean()

        return {
            "loss": loss,
            "loss_coarse": loss_coarse,
            "loss_fine": loss_fine,
            "acc_coarse": coarse_acc,
            "acc_fine": fine_acc,
            "acc": joint_acc,
        }
