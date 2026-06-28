from .simsiam import SimSiam
from .byol import BYOL
from .simclr import SimCLR
from .pseudo_supervised_net import PseudoSupervisedNet
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
        if model_cfg.proj_layers is not None:
            model.projector.set_layers(model_cfg.proj_layers)

    elif model_cfg.name == 'byol':
        model = BYOL(get_backbone(model_cfg.backbone))
    elif model_cfg.name == 'simclr':
        model = SimCLR(get_backbone(model_cfg.backbone))
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

