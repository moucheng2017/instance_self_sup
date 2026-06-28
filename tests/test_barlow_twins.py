import torch

from models.barlow_twins import BarlowTwins, barlow_twins_loss, off_diagonal


class TinyBackbone(torch.nn.Module):
    output_dim = 8

    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(3 * 8 * 8, self.output_dim)

    def forward(self, x):
        return self.fc(x.flatten(1))


def test_off_diagonal_selects_all_non_diagonal_entries():
    matrix = torch.arange(16).view(4, 4)
    values = off_diagonal(matrix)
    assert values.numel() == 12
    assert not any(v.item() in torch.diagonal(matrix).tolist() for v in values)


def test_barlow_loss_identity_correlation_has_near_zero_on_diag():
    z = torch.tensor([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
    _, on_diag, off_diag_loss = barlow_twins_loss(z, z, eps=1e-8)
    assert on_diag.item() < 1e-6
    assert off_diag_loss.item() < 1e-6


def test_barlow_loss_penalizes_correlated_dimensions():
    base = torch.tensor([-1.0, -0.5, 0.5, 1.0])
    correlated = torch.stack([base, base], dim=1)
    decorrelated = torch.tensor([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
    _, _, correlated_off = barlow_twins_loss(correlated, correlated, eps=1e-8)
    _, _, decorrelated_off = barlow_twins_loss(decorrelated, decorrelated, eps=1e-8)
    assert correlated_off.item() > decorrelated_off.item()


def test_barlow_forward_backpropagates():
    model = BarlowTwins(TinyBackbone(), projector_dim=16)
    out = model(torch.randn(4, 3, 8, 8), torch.randn(4, 3, 8, 8))
    assert {"loss", "on_diag", "off_diag"}.issubset(out)
    out["loss"].backward()
    assert any(param.grad is not None for param in model.parameters())
