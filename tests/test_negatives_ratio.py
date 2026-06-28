import ast
import json

import torch
import yaml

from datasets.pseudo_supervised import PseudoSupervisedDataset


class TinyDataset(torch.utils.data.Dataset):
    def __len__(self):
        return 32

    def __getitem__(self, idx):
        return torch.tensor([idx]), 0


def test_pseudo_supervised_dataset_uses_negatives_ratio_per_batch():
    dataset = PseudoSupervisedDataset(
        dataset=TinyDataset(),
        image_size=32,
        source_pool_size=None,
        augment_probability=1.0,
        explicit_indices=list(range(20)),
        samples_per_epoch=16,
        batch_size=8,
        negatives_ratio=0.5,
    )
    dataset.transform_strong = lambda image: image
    dataset.transform_weak = lambda image: image

    loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False, drop_last=True)
    _, labels = next(iter(loader))
    labels = labels.tolist()

    negative_count = int(0.5 * 8)
    negative_labels = labels[:negative_count]
    positive_labels = labels[negative_count:]

    assert len(set(negative_labels)) == negative_count
    assert set(positive_labels).issubset(set(negative_labels))
    assert len(set(labels)) == negative_count


def test_meta_random_config_defines_negatives_ratio():
    with open("configs/meta_exps/meta_random_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    assert config["train"]["negatives_ratio"] == 0.5


def test_random_meta_notebook_negatives_ratio_override_is_valid():
    with open("notebooks/random-meta-cifar10-ssl.ipynb", "r") as f:
        notebook = json.load(f)

    code_source = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )

    for cell in notebook["cells"]:
        if cell.get("cell_type") == "code":
            ast.parse("".join(cell.get("source", [])))

    assert "NEGATIVES_RATIO = 0.5" in code_source
    assert "'negatives_ratio': NEGATIVES_RATIO" in code_source
