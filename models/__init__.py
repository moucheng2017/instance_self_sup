from .simsiam import SimSiam
from .byol import BYOL
from .simclr import SimCLR
from .barlow_twins import BarlowTwins
from .pseudo_supervised_net import PseudoSupervisedNet
from .vicreg import VICReg
from torchvision.models import resnet50, resnet18
import torch
from .backbones import resnet18_cifar_variant1, resnet18_cifar_variant2

def get_backbone(backbone, castrate=True):
    backbone = eval(f"{backbone}()")

    if castrate:
        backbone.output_dim = backbone.fc.in_features
        backbone.fc = torch.nn.Identity()

    return backbone


def get_model(model_cfg, num_classes=None):

    if model_cfg.name == 'simsiam':
        model =  SimSiam(get_backbone(model_cfg.backbone))
        if getattr(model_cfg, "proj_layers", None) is not None:
            model.projector.set_layers(model_cfg.proj_layers)

    elif model_cfg.name == 'byol':
        model = BYOL(get_backbone(model_cfg.backbone))
    elif model_cfg.name == 'simclr':
        model = SimCLR(get_backbone(model_cfg.backbone))
    elif model_cfg.name == 'vicreg':
        model = VICReg(
            get_backbone(model_cfg.backbone),
            expander_dim=getattr(model_cfg, "expander_dim", 2048),
            sim_coeff=getattr(model_cfg, "sim_coeff", 25.0),
            std_coeff=getattr(model_cfg, "std_coeff", 25.0),
            cov_coeff=getattr(model_cfg, "cov_coeff", 1.0),
            gamma=getattr(model_cfg, "gamma", 1.0),
            eps=getattr(model_cfg, "eps", 1e-4),
        )
    elif model_cfg.name == 'barlow_twins':
        model = BarlowTwins(
            get_backbone(model_cfg.backbone),
            projector_dim=getattr(model_cfg, "projector_dim", 2048),
            lambd=getattr(model_cfg, "lambd", 0.0051),
            eps=getattr(model_cfg, "eps", 1e-4),
        )
    elif model_cfg.name == 'pseudo_supervised_net':
        model = PseudoSupervisedNet(
            num_classes=num_classes,
            backbone=get_backbone(model_cfg.backbone),
            l2_normalize=getattr(model_cfg, "l2_norm_backbone_features", False),
            cosine_softmax=getattr(model_cfg, "cosine_softmax", False),
            cosine_scale=getattr(model_cfg, "cosine_scale", 16.0),
        )
    else:
        raise NotImplementedError
    return model
