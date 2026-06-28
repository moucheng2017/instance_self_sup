import torch

from models.vicreg import VICReg, covariance_loss, invariance_loss, variance_loss


class TinyBackbone(torch.nn.Module):
    output_dim = 8

    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(3 * 8 * 8, self.output_dim)

    def forward(self, x):
        return self.fc(x.flatten(1))


def test_vicreg_forward_loss_is_scalar_and_differentiable():
    model = VICReg(TinyBackbone(), expander_dim=16)
    out = model(torch.randn(4, 3, 8, 8), torch.randn(4, 3, 8, 8))
    assert set(["loss", "inv_loss", "var_loss", "cov_loss"]).issubset(out)
    assert out["loss"].ndim == 0
    assert out["loss"].item() >= 0
    out["loss"].backward()
    assert any(param.grad is not None for param in model.parameters())


def test_vicreg_invariance_loss_zero_for_identical_inputs():
    z = torch.randn(5, 6)
    assert invariance_loss(z, z).item() == 0


def test_vicreg_variance_hinge_activates_for_constant_dimensions():
    low_var = torch.ones(8, 4)
    high_var = torch.tensor([[-3.0, -2.0, -1.0, 0.0], [3.0, 2.0, 1.0, 0.0]]).repeat(4, 1)
    assert variance_loss(low_var).item() > variance_loss(high_var).item()


def test_vicreg_covariance_penalizes_correlated_dimensions():
    base = torch.linspace(-1, 1, steps=8)
    correlated = torch.stack([base, base, -base], dim=1)
    decorrelated = torch.eye(8, 3)
    assert covariance_loss(correlated).item() > covariance_loss(decorrelated).item()
