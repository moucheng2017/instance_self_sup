from .simsiam import SimSiam
from .byol import BYOL
from .simclr import SimCLR
from .pseudo_supervised_net import PseudoSupervisedNet
from .hierarchical_pseudo_supervised_net import HierarchicalPseudoSupervisedNet
from .sigmoid_pseudo_supervised_net import SigmoidPseudoSupervisedNet
from .hierarchical_balanced_vmf_self_labeling_net import HierarchicalBalancedVMFSelfLabelingNet
from .hierarchical_kway_vmf_ot_self_labeling_net import HierarchicalKWayVMFOTSelfLabelingNet
from .low_rank_multitarget_pseudo_supervised_net import LowRankMultitargetPseudoSupervisedNet
from .topk_categorical_bottleneck_pic_net import TopKCategoricalBottleneckPICNet
from torchvision.models import resnet50, resnet18
import torch
from .backbones import resnet18_cifar_variant1, resnet18_cifar_variant2

_BACKBONE_REGISTRY = {
    "resnet18": resnet18,
    "resnet50": resnet50,
    "resnet18_cifar_variant1": resnet18_cifar_variant1,
    "resnet18_cifar_variant2": resnet18_cifar_variant2,
}


def get_backbone(backbone, castrate=True):
    if backbone not in _BACKBONE_REGISTRY:
        raise ValueError(
            f"Unknown backbone '{backbone}'. "
            f"Available: {sorted(_BACKBONE_REGISTRY)}"
        )
    backbone = _BACKBONE_REGISTRY[backbone]()

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
        model = PseudoSupervisedNet(num_classes=num_classes, backbone=get_backbone(model_cfg.backbone))
    elif model_cfg.name == 'hierarchical_pseudo_supervised_net':
        model = HierarchicalPseudoSupervisedNet(
            num_classes=num_classes,
            backbone=get_backbone(model_cfg.backbone),
            num_groups=getattr(model_cfg, 'num_groups', None),
        )
    elif model_cfg.name == 'sigmoid_pseudo_supervised_net':
        model = SigmoidPseudoSupervisedNet(
            num_classes=num_classes,
            backbone=get_backbone(model_cfg.backbone),
            embedding_dim=getattr(model_cfg, 'embedding_dim', None),
            normalize=getattr(model_cfg, 'normalize', True),
            init_temperature=getattr(model_cfg, 'init_temperature', 10.0),
            init_bias=getattr(model_cfg, 'init_bias', -10.0),
        )
    elif model_cfg.name == 'hierarchical_balanced_vmf_self_labeling_net':
        model = HierarchicalBalancedVMFSelfLabelingNet(
            num_classes=num_classes,
            backbone=get_backbone(model_cfg.backbone),
            embedding_dim=getattr(model_cfg, 'embedding_dim', None),
            depth=getattr(model_cfg, 'depth', 8),
            kappa=getattr(model_cfg, 'kappa', 20.0),
            ot_epsilon=getattr(model_cfg, 'ot_epsilon', 0.05),
            sinkhorn_iters=getattr(model_cfg, 'sinkhorn_iters', 3),
            em_iters=getattr(model_cfg, 'em_iters', 5),
            ot_unbalanced_tau=getattr(model_cfg, 'ot_unbalanced_tau', None),
            ot_unbalanced_min_split_fraction=getattr(model_cfg, 'ot_unbalanced_min_split_fraction', 0.05),
            ot_assignment_mode=getattr(model_cfg, 'ot_assignment_mode', 'hierarchical_vmf_ot'),
            flat_vmf_num_components=getattr(model_cfg, 'flat_vmf_num_components', None),
            batch_self_labeling=getattr(model_cfg, 'batch_self_labeling', True),
            tree_warm_start=getattr(model_cfg, 'tree_warm_start', False),
            reseed_acc_threshold=getattr(model_cfg, 'reseed_acc_threshold', None),
            reseed_patience=getattr(model_cfg, 'reseed_patience', 3),
            reseed_min_node_samples=getattr(model_cfg, 'reseed_min_node_samples', 64),
            reseed_enabled=getattr(model_cfg, 'reseed_enabled', False),
            reseed_budget_fraction=getattr(model_cfg, 'reseed_budget_fraction', 0.1),
            learnable_vmf_prototypes=getattr(model_cfg, 'learnable_vmf_prototypes', False),
            prototype_ema_momentum=getattr(model_cfg, 'prototype_ema_momentum', None),
            sigmoid_regularization_weight=getattr(model_cfg, 'sigmoid_regularization_weight', 0.0),
            sigmoid_init_temperature=getattr(model_cfg, 'sigmoid_init_temperature', 1.0),
            sigmoid_init_bias=getattr(model_cfg, 'sigmoid_init_bias', -10.0),
        )
    elif model_cfg.name == 'hierarchical_kway_vmf_ot_self_labeling_net':
        model = HierarchicalKWayVMFOTSelfLabelingNet(
            num_classes=num_classes,
            backbone=get_backbone(model_cfg.backbone),
            embedding_dim=getattr(model_cfg, 'embedding_dim', None),
            rank_schedule=getattr(model_cfg, 'rank_schedule', None),
            num_leaf_clusters=getattr(model_cfg, 'num_leaf_clusters', 256),
            rank_schedule_depth=getattr(model_cfg, 'rank_schedule_depth', None),
            rank_schedule_max_rank=getattr(model_cfg, 'rank_schedule_max_rank', 16),
            rank_schedule_base_rank=getattr(model_cfg, 'rank_schedule_base_rank', 1),
            supervised_depth=getattr(model_cfg, 'supervised_depth', None),
            kappa=getattr(model_cfg, 'kappa', 20.0),
            ot_epsilon=getattr(model_cfg, 'ot_epsilon', 0.05),
            sinkhorn_iters=getattr(model_cfg, 'sinkhorn_iters', 10),
            em_iters=getattr(model_cfg, 'em_iters', 5),
            ot_unbalanced_tau=getattr(model_cfg, 'ot_unbalanced_tau', None),
            ot_unbalanced_min_split_fraction=getattr(model_cfg, 'ot_unbalanced_min_split_fraction', 0.05),
            batch_self_labeling=getattr(model_cfg, 'batch_self_labeling', True),
            learnable_vmf_prototypes=getattr(model_cfg, 'learnable_vmf_prototypes', False),
            prototype_ema_momentum=getattr(model_cfg, 'prototype_ema_momentum', None),
            sigmoid_regularization_weight=getattr(model_cfg, 'sigmoid_regularization_weight', 0.0),
            sigmoid_init_temperature=getattr(model_cfg, 'sigmoid_init_temperature', 1.0),
            sigmoid_init_bias=getattr(model_cfg, 'sigmoid_init_bias', -10.0),
        )
    elif model_cfg.name == 'low_rank_multitarget_pseudo_supervised_net':
        model = LowRankMultitargetPseudoSupervisedNet(
            num_classes=num_classes,
            backbone=get_backbone(model_cfg.backbone),
            num_latent_classes=getattr(model_cfg, 'num_latent_classes', 100),
            membership_size=getattr(model_cfg, 'membership_size', 5),
            membership_seed=getattr(model_cfg, 'membership_seed', 0),
            loss_mode=getattr(model_cfg, 'loss_mode', 'uniform_multi_ce'),
        )
    elif model_cfg.name == 'topk_categorical_bottleneck_pic_net':
        model = TopKCategoricalBottleneckPICNet(
            num_classes=num_classes,
            backbone=get_backbone(model_cfg.backbone),
            num_latent_classes=getattr(model_cfg, 'num_latent_classes', 100),
            topk=getattr(model_cfg, 'topk', 5),
            latent_temperature=getattr(model_cfg, 'latent_temperature', 1.0),
            decoder_hidden_dim=getattr(model_cfg, 'decoder_hidden_dim', None),
            balance_weight=getattr(model_cfg, 'balance_weight', 1.0),
            entropy_weight=getattr(model_cfg, 'entropy_weight', 0.0),
            target_entropy=getattr(model_cfg, 'target_entropy', None),
            column_normalize=getattr(model_cfg, 'column_normalize', True),
            normalize_features=getattr(model_cfg, 'normalize_features', True),
            normalize_assigner=getattr(model_cfg, 'normalize_assigner', True),
            use_gumbel_noise=getattr(model_cfg, 'use_gumbel_noise', True),
        )
    elif model_cfg.name == 'swav':
        raise NotImplementedError
    else:
        raise NotImplementedError
    return model
