import torch

from models.topk_categorical_bottleneck_pic_net import (
    TopKCategoricalBottleneckPICNet,
    column_normalize_assignments,
    soft_categorical_assignments,
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


def test_soft_categorical_assignments_are_probabilities():
    logits = torch.tensor(
        [
            [0.1, 3.0, 2.0, -1.0, 0.0],
            [4.0, 0.0, 1.0, 2.0, -1.0],
        ]
    )

    assignments = soft_categorical_assignments(logits)

    assert assignments.shape == logits.shape
    assert torch.isfinite(assignments).all()
    assert torch.allclose(assignments.sum(dim=1), torch.ones(2))
    assert assignments.argmax(dim=1).tolist() == [1, 0]


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
        decoder_hidden_dim=12,
        balance_weight=0.1,
        entropy_weight=0.1,
        target_entropy=0.7,
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
