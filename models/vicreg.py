import torch
import torch.nn as nn
import torch.nn.functional as F


class VICRegProjector(nn.Module):
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


def invariance_loss(z_a, z_b):
    return F.mse_loss(z_a, z_b)


def variance_loss(z, gamma=1.0, eps=1e-4):
    std = torch.sqrt(z.var(dim=0) + eps)
    return torch.mean(F.relu(gamma - std))


def covariance_loss(z):
    n, d = z.shape
    if n < 2:
        raise ValueError("VICReg covariance loss requires at least two samples.")
    z = z - z.mean(dim=0)
    cov = (z.t() @ z) / (n - 1)
    off_diag = cov.flatten()[:-1].view(d - 1, d + 1)[:, 1:].flatten()
    return off_diag.pow(2).sum() / d


class VICReg(nn.Module):
    def __init__(
        self,
        backbone,
        expander_dim=2048,
        sim_coeff=25.0,
        std_coeff=25.0,
        cov_coeff=1.0,
        gamma=1.0,
        eps=1e-4,
    ):
        super().__init__()
        self.backbone = backbone
        self.projector = VICRegProjector(backbone.output_dim, hidden_dim=expander_dim, out_dim=expander_dim)
        self.sim_coeff = float(sim_coeff)
        self.std_coeff = float(std_coeff)
        self.cov_coeff = float(cov_coeff)
        self.gamma = float(gamma)
        self.eps = float(eps)

    def forward(self, x1, x2):
        z1 = self.projector(self.backbone(x1))
        z2 = self.projector(self.backbone(x2))

        inv = invariance_loss(z1, z2)
        var = (variance_loss(z1, self.gamma, self.eps) + variance_loss(z2, self.gamma, self.eps)) / 2
        cov = covariance_loss(z1) + covariance_loss(z2)
        loss = self.sim_coeff * inv + self.std_coeff * var + self.cov_coeff * cov
        return {
            "loss": loss,
            "inv_loss": inv,
            "var_loss": var,
            "cov_loss": cov,
        }
