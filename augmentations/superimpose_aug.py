from PIL import Image
from torchvision import transforms

try:
    from torchvision.transforms import GaussianBlur
except ImportError:
    from .gaussian_blur import GaussianBlur
    transforms.GaussianBlur = GaussianBlur


imagenet_norm = [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]]


class SuperimposeTransform(object):
    def __init__(self, image_size, normalize=imagenet_norm):
        blur_prob = 0.0 if image_size <= 32 else 0.5
        self.clean_transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        self.augmented_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(image_size, scale=(0.2, 1.0), interpolation=Image.BICUBIC),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply(
                    [transforms.GaussianBlur(kernel_size=image_size // 20 * 2 + 1, sigma=(0.1, 2.0))],
                    p=blur_prob,
                ),
                transforms.ToTensor(),
            ]
        )
        self.normalize = transforms.Normalize(*normalize)

    def clean(self, image):
        return self.clean_transform(image)

    def augment(self, image):
        return self.augmented_transform(image)

    def postprocess(self, image_tensor):
        return self.normalize(image_tensor)
