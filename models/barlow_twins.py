import torch
import torch.nn as nn


class BarlowTwinsProjector(nn.Module):
    def __init__(self, in_dim, hidden_dim=2048, out_dim=2048):
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


def off_diagonal(x):
    if x.ndim != 2 or x.shape[0] != x.shape[1]:
        raise ValueError("off_diagonal expects a square matrix.")
    n = x.shape[0]
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def normalize_batch(z, eps=1e-4):
    return (z - z.mean(dim=0)) / (z.std(dim=0, unbiased=False) + eps)


def barlow_twins_loss(z_a, z_b, lambd=0.0051, eps=1e-4):
    n = z_a.shape[0]
    if n < 2:
        raise ValueError("Barlow Twins loss requires at least two samples.")
    z_a = normalize_batch(z_a, eps=eps)
    z_b = normalize_batch(z_b, eps=eps)
    cross_corr = (z_a.t() @ z_b) / n
    on_diag = torch.diagonal(cross_corr).add(-1).pow(2).sum()
    off_diag = off_diagonal(cross_corr).pow(2).sum()
    return on_diag + float(lambd) * off_diag, on_diag, off_diag


class BarlowTwins(nn.Module):
    def __init__(self, backbone, projector_dim=2048, lambd=0.0051, eps=1e-4):
        super().__init__()
        self.backbone = backbone
        self.projector = BarlowTwinsProjector(
            backbone.output_dim,
            hidden_dim=projector_dim,
            out_dim=projector_dim,
        )
        self.lambd = float(lambd)
        self.eps = float(eps)

    def forward(self, x1, x2):
        z1 = self.projector(self.backbone(x1))
        z2 = self.projector(self.backbone(x2))
        loss, on_diag, off_diag_loss = barlow_twins_loss(z1, z2, lambd=self.lambd, eps=self.eps)
        return {
            "loss": loss,
            "on_diag": on_diag,
            "off_diag": off_diag_loss,
        }
