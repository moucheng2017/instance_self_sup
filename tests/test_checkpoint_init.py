import copy
from types import SimpleNamespace

import pytest
import torch

from models import get_model
from models.checkpoint_init import load_init_weights


def cfg(name, **kwargs):
    data = {"name": name, "backbone": "resnet18"}
    data.update(kwargs)
    return SimpleNamespace(**data)


def _checkpoint(path, model):
    torch.save({"epoch": 1, "state_dict": model.state_dict()}, path)


def _first_param(module):
    return next(module.parameters()).detach().clone()


def test_load_init_weights_loads_only_backbone_by_default(tmp_path):
    source = get_model(cfg("simclr"))
    target = get_model(cfg("simclr"))
    target_projector_before = _first_param(target.projector)
    ckpt = tmp_path / "simclr.pth"
    _checkpoint(ckpt, source)

    load_init_weights(target, ckpt)
    for key, value in source.backbone.state_dict().items():
        assert torch.equal(value, target.backbone.state_dict()[key])
    assert torch.equal(target_projector_before, _first_param(target.projector))


def test_cross_architecture_backbone_transfer_from_pseudo_sup_to_simclr(tmp_path):
    source = get_model(cfg("pseudo_supervised_net"), num_classes=8)
    target = get_model(cfg("simclr"))
    ckpt = tmp_path / "pseudo.pth"
    _checkpoint(ckpt, source)
    load_init_weights(target, ckpt, load_backbone=True)
    for key, value in source.backbone.state_dict().items():
        assert torch.equal(value, target.backbone.state_dict()[key])


def test_projector_and_predictor_round_trip_for_matching_architecture(tmp_path):
    source = get_model(cfg("simsiam", proj_layers=None))
    target = get_model(cfg("simsiam", proj_layers=None))
    ckpt = tmp_path / "simsiam.pth"
    _checkpoint(ckpt, source)
    load_init_weights(target, ckpt, load_backbone=True, load_projector=True, load_predictor=True)
    assert torch.equal(_first_param(source.projector), _first_param(target.projector))
    assert torch.equal(_first_param(source.predictor), _first_param(target.predictor))

    pseudo_source = get_model(cfg("pseudo_supervised_net"), num_classes=8)
    pseudo_target = get_model(cfg("pseudo_supervised_net"), num_classes=8)
    pseudo_ckpt = tmp_path / "pseudo2.pth"
    _checkpoint(pseudo_ckpt, pseudo_source)
    load_init_weights(pseudo_target, pseudo_ckpt, load_backbone=False, load_projector=True, load_predictor=True)
    assert torch.equal(_first_param(pseudo_source.proj), _first_param(pseudo_target.proj))
    assert torch.equal(_first_param(pseudo_source.pred), _first_param(pseudo_target.pred))


def test_requested_missing_or_mismatched_submodule_raises(tmp_path):
    source = get_model(cfg("simclr"))
    ckpt = tmp_path / "simclr.pth"
    _checkpoint(ckpt, source)
    with pytest.raises(ValueError, match="no predictor"):
        load_init_weights(get_model(cfg("simclr")), ckpt, load_backbone=False, load_predictor=True)
    with pytest.raises(ValueError, match="Could not load requested projector"):
        load_init_weights(
            get_model(cfg("vicreg", expander_dim=64)),
            ckpt,
            load_backbone=False,
            load_projector=True,
        )


def test_none_init_checkpoint_is_noop_equivalent():
    torch.manual_seed(0)
    a = get_model(cfg("simclr"))
    state_a = copy.deepcopy(a.state_dict())
    torch.manual_seed(0)
    b = get_model(cfg("simclr"))
    assert all(torch.equal(state_a[key], b.state_dict()[key]) for key in state_a)
