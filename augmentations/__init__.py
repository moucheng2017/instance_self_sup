from .simsiam_aug import SimSiamTransform
from .eval_aug import Transform_single
from .byol_aug import BYOL_transform
from .simclr_aug import SimCLRTransform
from .simple_aug import StrongTransform
from .simple_aug import WeakTransform

def get_aug(name='simsiam', image_size=224, train=True, train_classifier=None):

    if train==True:
        if name == 'simsiam':
            augmentation = SimSiamTransform(image_size)
        elif name == 'byol':
            augmentation = BYOL_transform(image_size)
        elif name in ('simclr', 'vicreg', 'barlow_twins', 'swav'):
            augmentation = SimCLRTransform(image_size)
        elif name == 'strong':
            augmentation = StrongTransform(image_size)
        elif name == 'pseudo_supervised_net':
            # Kept for completeness/symmetry only. The pseudo_sup training path
            # (main.build_train_loader) calls get_dataset(transform=None) and lets
            # PseudoSupervisedDataset apply StrongTransform internally, so this branch
            # is not exercised in practice; the train transform does not flow through
            # get_aug for pseudo_supervised_net (see design.md §3.3).
            augmentation = StrongTransform(image_size)
        elif name == 'weak':
            augmentation = WeakTransform()
        else:
            raise NotImplementedError
    elif train==False:
        if train_classifier is None:
            raise Exception
        augmentation = Transform_single(image_size, train=train_classifier)
    else:
        raise Exception
    
    return augmentation





