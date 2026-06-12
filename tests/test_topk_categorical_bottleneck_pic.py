import torch

from models.topk_categorical_bottleneck_pic_net import (
    TopKCategoricalBottleneckPICNet,
    column_normalize_assignments,
    gumbel_topk_straight_through,
)


class TinyBackbone(torch.nn.Module):
    output_dim = 8

    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(3 * 4 * 4, self.output_dim),
        )

    def forward(self, images):
        return self.net(images)


def test_gumbel_topk_eval_has_exact_k_active_categories():
    logits = torch.tensor(
        [
            [0.1, 3.0, 2.0, -1.0, 0.0],
            [4.0, 0.0, 1.0, 2.0, -1.0],
        ]
    )

    assignments, soft_probs, hard = gumbel_topk_straight_through(
        logits,
        k=2,
        temperature=1.0,
        training=False,
        use_gumbel_noise=False,
    )

    assert assignments.shape == logits.shape
    assert soft_probs.shape == logits.shape
    assert hard.shape == logits.shape
    assert torch.allclose(hard.sum(dim=1), torch.full((2,), 2.0))
    assert hard[0].nonzero().flatten().tolist() == [1, 2]
    assert hard[1].nonzero().flatten().tolist() == [0, 3]


def test_column_normalize_preserves_total_mass_and_no_nans():
    assignments = torch.tensor(
        [
            [1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
        ]
    )

    normalized = column_normalize_assignments(assignments)

    assert normalized.shape == assignments.shape
    assert torch.isfinite(normalized).all()
    assert torch.allclose(normalized.sum(), assignments.sum())
    assert normalized[:, 1].sum().item() == 0.0


def test_model_forward_returns_instance_and_bottleneck_losses():
    model = TopKCategoricalBottleneckPICNet(
        num_classes=16,
        backbone=TinyBackbone(),
        num_latent_classes=5,
        topk=2,
        latent_temperature=0.7,
        decoder_hidden_dim=12,
        balance_weight=0.1,
        entropy_weight=0.1,
        target_entropy=0.7,
        use_gumbel_noise=False,
    )
    images = torch.randn(4, 3, 4, 4)
    labels = torch.tensor([0, 1, 2, 3])

    out = model(images, labels)

    assert out["loss"].ndim == 0
    assert out["loss_instance"].ndim == 0
    assert out["loss_balance"].ndim == 0
    assert out["loss_entropy"].ndim == 0
    assert out["acc"].shape == torch.Size([])
    assert out["latent_perplexity"].item() > 0.0
