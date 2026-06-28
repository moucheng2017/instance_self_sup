import yaml

from arguments import Namespace
from augmentations import get_aug
from models import get_model


CONFIGS = {
    "pseudo_sup": "configs/baselines/pseudo_sup_cifar10.yaml",
    "vicreg": "configs/baselines/vicreg_cifar10.yaml",
    "barlow_twins": "configs/baselines/barlow_twins_cifar10.yaml",
    "simclr": "configs/baselines/simclr_cifar10.yaml",
    "simsiam": "configs/baselines/simsiam_cifar10.yaml",
}


def load(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def test_baseline_configs_have_required_common_fields():
    expected_names = {
        "pseudo_sup": "pseudo_supervised_net",
        "vicreg": "vicreg",
        "barlow_twins": "barlow_twins",
        "simclr": "simclr",
        "simsiam": "simsiam",
    }
    for key, path in CONFIGS.items():
        config = load(path)
        assert config["model"]["name"] == expected_names[key]
        assert config["model"]["backbone"] == "resnet18"
        assert config["train"]["subset_n"] is None
        assert config["train"]["subset_seed"] == 42
        assert config["train"]["knn_monitor"] is False
        assert config["eval"] is False
        assert config["model"]["init_checkpoint"] is None
        assert config["model"]["init_load_backbone"] is True
        assert config["model"]["init_load_projector"] is False
        assert config["model"]["init_load_predictor"] is False


def test_lars_configs_share_identical_train_block_and_pseudo_sup_uses_sgd():
    lars_names = ["vicreg", "barlow_twins", "simclr", "simsiam"]
    train_blocks = [load(CONFIGS[name])["train"] for name in lars_names]
    assert all(block == train_blocks[0] for block in train_blocks)
    assert all(block["optimizer"]["name"] == "lars" for block in train_blocks)

    pseudo = load(CONFIGS["pseudo_sup"])
    assert pseudo["train"]["optimizer"]["name"] == "sgd"
    assert pseudo["train"]["negatives_ratio"] == 0.25
    assert pseudo["model"]["cosine_softmax"] is True
    assert pseudo["model"]["l2_norm_backbone_features"] is True


def test_vicreg_and_barlow_locked_coefficients():
    vicreg = load(CONFIGS["vicreg"])["model"]
    assert vicreg["sim_coeff"] == 25.0
    assert vicreg["std_coeff"] == 25.0
    assert vicreg["cov_coeff"] == 1.0
    assert vicreg["expander_dim"] == 2048

    barlow = load(CONFIGS["barlow_twins"])["model"]
    assert barlow["lambd"] == 0.0051
    assert barlow["projector_dim"] == 2048


def test_get_model_and_get_aug_resolve_all_baselines():
    for path in CONFIGS.values():
        config = load(path)
        model_cfg = Namespace(config["model"])
        get_model(model_cfg, num_classes=8 if model_cfg.name == "pseudo_supervised_net" else None)
        get_aug(name=model_cfg.name, image_size=32, train=True)
