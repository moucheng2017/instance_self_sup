import torch
import torch.nn as nn
import torch.nn.functional as F


def build_balanced_memberships(num_instances, num_latent_classes, membership_size, seed=0):
    """Build a balanced [num_instances, membership_size] latent-label code.

    This is a controlled diagnostic target matrix: every instance belongs to
    exactly `membership_size` latent classes, and latent classes are used as
    evenly as possible.  It is intentionally fixed, so rank K is the controlled
    variable rather than a moving self-labeling target.
    """
    num_instances = int(num_instances)
    num_latent_classes = int(num_latent_classes)
    membership_size = int(membership_size)
    if num_instances < 1:
        raise ValueError("num_instances must be positive.")
    if num_latent_classes < 1:
        raise ValueError("num_latent_classes must be positive.")
    if membership_size < 1:
        raise ValueError("membership_size must be positive.")
    if membership_size > num_latent_classes:
        raise ValueError("membership_size cannot exceed num_latent_classes.")

    generator = torch.Generator().manual_seed(int(seed))
    total_slots = num_instances * membership_size
    repeats = (total_slots + num_latent_classes - 1) // num_latent_classes
    slots = torch.arange(num_latent_classes, dtype=torch.long).repeat(repeats)[:total_slots]
    slots = slots[torch.randperm(total_slots, generator=generator)]
    memberships = slots.view(num_instances, membership_size)

    # Avoid duplicate latent labels within a row.  For the default K >> m this
    # rarely triggers; the loop keeps the helper robust for small K sweeps.
    for row in range(num_instances):
        seen = set()
        for col in range(membership_size):
            value = int(memberships[row, col].item())
            if value not in seen:
                seen.add(value)
                continue
            replacement = int(torch.randint(num_latent_classes, (1,), generator=generator).item())
            while replacement in seen:
                replacement = int(torch.randint(num_latent_classes, (1,), generator=generator).item())
            memberships[row, col] = replacement
            seen.add(replacement)
    return memberships


class LowRankMultitargetPseudoSupervisedNet(nn.Module):
    """Low-rank / multi-target image-index supervision.

    Instead of predicting N independent image IDs, each image index owns a fixed
    multi-hot code over K latent labels.  K controls target rank/granularity;
    membership_size controls how many latent labels an instance must belong to.
    """

    def __init__(
        self,
        num_classes,
        backbone=None,
        num_latent_classes=100,
        membership_size=5,
        membership_seed=0,
        loss_mode="uniform_multi_ce",
    ):
        super().__init__()
        if backbone is None:
            raise ValueError("backbone must be provided explicitly.")
        if num_classes is None or int(num_classes) < 1:
            raise ValueError("LowRankMultitargetPseudoSupervisedNet requires at least 1 pseudo class.")

        self.backbone = backbone
        self.num_classes = int(num_classes)
        self.num_latent_classes = int(num_latent_classes)
        self.membership_size = int(membership_size)
        self.loss_mode = str(loss_mode)
        if self.num_latent_classes < 1:
            raise ValueError("num_latent_classes must be positive.")
        if self.membership_size < 1:
            raise ValueError("membership_size must be positive.")
        if self.membership_size > self.num_latent_classes:
            raise ValueError("membership_size cannot exceed num_latent_classes.")
        if self.loss_mode not in ("uniform_multi_ce", "set_ce"):
            raise ValueError("loss_mode must be 'uniform_multi_ce' or 'set_ce'.")

        self.classifier = nn.Linear(backbone.output_dim, self.num_latent_classes)
        memberships = build_balanced_memberships(
            num_instances=self.num_classes,
            num_latent_classes=self.num_latent_classes,
            membership_size=self.membership_size,
            seed=membership_seed,
        )
        self.register_buffer("memberships", memberships)

    def _uniform_multi_ce(self, logits, target_sets):
        log_probs = F.log_softmax(logits, dim=1)
        return -log_probs.gather(1, target_sets).mean()

    def _set_ce(self, logits, target_sets):
        positive_logits = logits.gather(1, target_sets)
        return -(torch.logsumexp(positive_logits, dim=1) - torch.logsumexp(logits, dim=1)).mean()

    def forward(self, images, pseudo_labels):
        if pseudo_labels.ndim != 1:
            raise ValueError("pseudo_labels must have shape [batch_size].")

        features = self.backbone(images)
        logits = self.classifier(features)
        labels = pseudo_labels.long()
        target_sets = self.memberships[labels]

        if self.loss_mode == "uniform_multi_ce":
            loss = self._uniform_multi_ce(logits, target_sets)
        else:
            loss = self._set_ce(logits, target_sets)

        predictions = logits.argmax(dim=1)
        top1_in_set = predictions[:, None].eq(target_sets).any(dim=1).float().mean()
        topk = min(self.membership_size, self.num_latent_classes)
        topk_predictions = logits.topk(k=topk, dim=1).indices
        target_hit = topk_predictions[:, :, None].eq(target_sets[:, None, :]).any(dim=2).float().mean()

        return {
            "loss": loss,
            "acc": top1_in_set,
            "target_hit_at_m": target_hit,
        }
