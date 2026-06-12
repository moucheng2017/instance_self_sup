import torch
from torch.utils.data import Dataset

from augmentations.superimpose_aug import SuperimposeTransform


class SuperimposeSourcesDataset(Dataset):
    def __init__(
        self,
        dataset,
        image_size,
        source_pool_size,
        subset_seed=0,
        samples_per_epoch=None,
    ):
        self.dataset = dataset
        self.transform = SuperimposeTransform(image_size=image_size)
        if source_pool_size is None:
            raise ValueError("source_pool_size must be provided for SuperimposeSourcesDataset.")
        self.source_pool_size = int(source_pool_size)
        if self.source_pool_size < 2:
            raise ValueError("source_pool_size must be at least 2.")
        if self.source_pool_size > len(dataset):
            raise ValueError("source_pool_size can not exceed the dataset size.")

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

    def _sample_source_pair(self):
        first_index, first_label = self._sample_source()
        second_index, second_label = self._sample_source()
        return (first_index, first_label), (second_index, second_label)

    def _transform_image(self, image):
        return self.transform.augment(image)

    def __getitem__(self, idx):
        del idx  # Samples are drawn randomly from the fixed pseudo-labeled source pool.
        (first_index, first_label), (second_index, second_label) = self._sample_source_pair()
        first_image, _ = self.dataset[first_index]
        second_image, _ = self.dataset[second_index]

        first_tensor = self._transform_image(first_image)
        second_tensor = self._transform_image(second_image)
        superimposed_image = self.transform.postprocess((first_tensor + second_tensor) / 2.0)
        pseudo_labels = torch.tensor([first_label, second_label], dtype=torch.long)

        return superimposed_image, pseudo_labels
