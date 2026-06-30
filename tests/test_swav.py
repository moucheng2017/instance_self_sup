import torch

from models.swav import SwAV, sinkhorn


class TinyBackbone(torch.nn.Module):
    output_dim = 8

    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(3 * 8 * 8, self.output_dim)

    def forward(self, x):
        return self.fc(x.flatten(1))


def test_swav_forward_loss_is_scalar_nonneg_and_differentiable():
    model = SwAV(TinyBackbone(), projector_dim=16, num_prototypes=12)
    out = model(torch.randn(8, 3, 8, 8), torch.randn(8, 3, 8, 8))
    assert {"loss", "assign_entropy", "proto_usage_entropy"}.issubset(out)
    assert out["assign_entropy"].item() >= 0
    assert out["proto_usage_entropy"].item() >= 0
    assert out["loss"].ndim == 0
    assert out["loss"].item() >= 0
    out["loss"].backward()
    assert any(param.grad is not None for param in model.parameters())
    # Prototypes are trained, so they must receive a gradient.
    assert model.prototypes.weight.grad is not None


def test_sinkhorn_produces_valid_normalized_codes():
    codes = sinkhorn(torch.randn(8, 12), epsilon=0.05, n_iters=3)
    assert codes.shape == (8, 12)
    assert (codes >= 0).all()
    # Each sample's code is a distribution over prototypes (sums to 1).
    assert torch.allclose(codes.sum(dim=1), torch.ones(8), atol=1e-4)


def test_sinkhorn_codes_are_detached():
    scores = torch.randn(4, 6, requires_grad=True)
    assert not sinkhorn(scores).requires_grad


def test_sinkhorn_equipartition_prevents_collapse():
    # Nearly identical rows: without equipartition all mass would pile on one prototype.
    scores = torch.zeros(16, 8) + torch.randn(1, 8) * 0.01
    codes = sinkhorn(scores, epsilon=0.05, n_iters=3)
    prototype_mass = codes.sum(dim=0)
    assert (prototype_mass > 0).all()                       # no prototype is dead
    assert prototype_mass.max().item() < codes.sum().item()  # no single prototype dominates


def test_normalize_prototypes_makes_unit_norm_rows():
    model = SwAV(TinyBackbone(), projector_dim=16, num_prototypes=10)
    with torch.no_grad():
        model.prototypes.weight.mul_(5.0)  # de-normalize first
    model.normalize_prototypes()
    norms = model.prototypes.weight.data.norm(dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_swav_loss_lower_for_correlated_than_unrelated_views():
    torch.manual_seed(0)
    model = SwAV(TinyBackbone(), projector_dim=16, num_prototypes=12)
    model.eval()  # fix BN so the comparison is about the inputs, not batch stats
    x = torch.randn(16, 3, 8, 8)
    with torch.no_grad():
        correlated = model(x, x + 0.01 * torch.randn_like(x))["loss"].item()
        unrelated = model(x, torch.randn(16, 3, 8, 8))["loss"].item()
    assert correlated < unrelated
