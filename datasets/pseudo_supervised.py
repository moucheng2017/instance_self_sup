import torch
from torch.utils.data import Dataset

from augmentations.superimpose_aug import SuperimposeTransform


class PseudoSupervisedDataset(Dataset):
    """Dataset that draws single images from a fixed source pool and assigns
    their pool position as a pseudo-label, enabling supervised-style training
    without any superimpose augmentation."""

    def __init__(
        self,
        dataset,
        image_size,
        source_pool_size,
        augment_probability=1.0,
        subset_seed=0,
        samples_per_epoch=None,
        num_views=1,
    ):
        self.dataset = dataset
        self.transform = SuperimposeTransform(image_size=image_size)

        self.num_views = int(num_views)
        if self.num_views < 1:
            raise ValueError("num_views must be at least 1.")

        if source_pool_size is None:
            raise ValueError("source_pool_size must be provided for PseudoSupervisedDataset.")
        self.source_pool_size = int(source_pool_size)
        if self.source_pool_size < 1:
            raise ValueError("source_pool_size must be at least 1.")
        if self.source_pool_size > len(dataset):
            raise ValueError("source_pool_size cannot exceed the dataset size.")

        self.augment_probability = float(augment_probability)
        if not 0.0 <= self.augment_probability <= 1.0:
            raise ValueError("augment_probability must be between 0 and 1.")

        self.length = len(dataset) if samples_per_epoch is None else int(samples_per_epoch)
        if self.length <= 0:
            raise ValueError("samples_per_epoch must be positive.")

        generator = torch.Generator().manual_seed(int(subset_seed))
        permutation = torch.randperm(len(dataset), generator=generator).tolist()
        self.source_indices = permutation[: self.source_pool_size]
        self.num_pseudo_classes = len(self.source_indices)

    def __len__(self):
        return self.length

    def _sample_source(self):
        source_offset = torch.randint(self.num_pseudo_classes, size=(1,)).item()
        return self.source_indices[source_offset], source_offset

    def _transform_image(self, image):
        if torch.rand(1).item() < self.augment_probability:
            return self.transform.augment(image)
        return self.transform.clean(image)

    def __getitem__(self, idx):
        del idx  # Samples are drawn randomly from the fixed pseudo-labeled source pool.
        source_index, pseudo_label = self._sample_source()
        image, _ = self.dataset[source_index]
        if self.num_views == 1:
            image_tensor = self.transform.postprocess(self._transform_image(image))
            return image_tensor, torch.tensor(pseudo_label, dtype=torch.long)
        # Independent augmentation draws stacked as [num_views, C, H, W]
        # (e.g. for swapped-view assignment models).
        views = torch.stack(
            [self.transform.postprocess(self._transform_image(image)) for _ in range(self.num_views)]
        )
        return views, torch.tensor(pseudo_label, dtype=torch.long)
