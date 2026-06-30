import torch
import torch.nn as nn
import torch.nn.functional as F


class SwAVProjector(nn.Module):
    """3-layer BN-MLP projector, matching the VICReg/Barlow style for a fair comparison."""

    def __init__(self, in_dim, hidden_dim=512, out_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


@torch.no_grad()
def sinkhorn(scores, epsilon=0.05, n_iters=3):
    """Sinkhorn-Knopp: turn prototype scores into soft, equipartitioned codes.

    scores: ``[B, K]`` (batch x prototypes). Returns codes ``Q`` of shape ``[B, K]``
    whose columns (per-sample) sum to 1. Run under no_grad: codes are targets.
    """
    Q = torch.exp(scores / epsilon).t()  # [K, B]
    K, B = Q.shape
    Q = Q / Q.sum()
    for _ in range(n_iters):
        Q = Q / Q.sum(dim=1, keepdim=True)  # normalize each prototype (row)
        Q = Q / K
        Q = Q / Q.sum(dim=0, keepdim=True)  # normalize each sample (column)
        Q = Q / B
    Q = Q * B  # so each column (sample) sums to 1
    return Q.t()  # [B, K]


class SwAV(nn.Module):
    """SwAV (Caron et al., 2020) — 2-view swapped-prediction with online Sinkhorn codes.

    Kept to two views (no multi-crop) and the repo's standard SimCLR-style augmentation so
    it is directly comparable to the other baselines (same backbone, same 512-d projector,
    same SGD recipe). The projector output is L2-normalized and scored against a set of
    learnable, unit-norm prototypes; codes computed by Sinkhorn-Knopp are swapped between
    the two views.
    """

    def __init__(
        self,
        backbone,
        projector_dim=512,
        num_prototypes=512,
        temperature=0.1,
        sinkhorn_epsilon=0.05,
        sinkhorn_iters=3,
    ):
        super().__init__()
        self.backbone = backbone
        self.projector = SwAVProjector(backbone.output_dim, hidden_dim=projector_dim, out_dim=projector_dim)
        self.prototypes = nn.Linear(projector_dim, int(num_prototypes), bias=False)
        self.temperature = float(temperature)
        self.sinkhorn_epsilon = float(sinkhorn_epsilon)
        self.sinkhorn_iters = int(sinkhorn_iters)

    @torch.no_grad()
    def normalize_prototypes(self):
        weight = F.normalize(self.prototypes.weight.data.clone(), dim=1, p=2)
        self.prototypes.weight.data.copy_(weight)

    def _scores(self, x):
        z = F.normalize(self.projector(self.backbone(x)), dim=1, p=2)
        return self.prototypes(z)

    def forward(self, x1, x2):
        self.normalize_prototypes()
        scores1 = self._scores(x1)
        scores2 = self._scores(x2)

        with torch.no_grad():
            code1 = sinkhorn(scores1, self.sinkhorn_epsilon, self.sinkhorn_iters)
            code2 = sinkhorn(scores2, self.sinkhorn_epsilon, self.sinkhorn_iters)

        log_p1 = F.log_softmax(scores1 / self.temperature, dim=1)
        log_p2 = F.log_softmax(scores2 / self.temperature, dim=1)

        # Swapped prediction: each view predicts the other view's code.
        loss = -0.5 * (
            torch.mean(torch.sum(code2 * log_p1, dim=1))
            + torch.mean(torch.sum(code1 * log_p2, dim=1))
        )

        # Collapse diagnostics (logged, not optimized): mean per-sample assignment entropy
        # (lower = more confident assignments) and the entropy of the batch-averaged
        # assignment over prototypes (lower = fewer prototypes used, i.e. closer to collapse).
        with torch.no_grad():
            p1, p2 = log_p1.exp(), log_p2.exp()
            assign_entropy = -0.5 * (
                torch.sum(p1 * log_p1, dim=1).mean() + torch.sum(p2 * log_p2, dim=1).mean()
            )
            p_bar = 0.5 * (p1.mean(dim=0) + p2.mean(dim=0))
            proto_usage_entropy = -torch.sum(p_bar * torch.log(p_bar + 1e-12))

        return {
            "loss": loss,
            "assign_entropy": assign_entropy,
            "proto_usage_entropy": proto_usage_entropy,
        }
