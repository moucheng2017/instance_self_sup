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
        elif name in ('simclr', 'vicreg', 'barlow_twins'):
            augmentation = SimCLRTransform(image_size)
        elif name == 'strong':
            augmentation = StrongTransform(image_size)
        elif name == 'pseudo_supervised_net':
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





