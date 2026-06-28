import torch
from torch.utils.data import Dataset
from augmentations.simple_aug import StrongTransform, WeakTransform

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
        batch_size=None,
        negatives_ratio=None,
        explicit_indices=None,
    ):
        self.dataset = dataset
        self.transform_strong = StrongTransform(image_size=image_size)
        self.transform_weak = WeakTransform()
        
        if explicit_indices is not None:
            # Use caller-supplied indices directly (meta-loop episodes).
            self.source_indices = list(explicit_indices)
            self.source_pool_size = len(self.source_indices)
        else:
            if source_pool_size is None:
                raise ValueError("source_pool_size must be provided for PseudoSupervisedDataset.")
            self.source_pool_size = int(source_pool_size)
            if self.source_pool_size < 1:
                raise ValueError("source_pool_size must be at least 1.")
            if self.source_pool_size > len(dataset):
                raise ValueError("source_pool_size cannot exceed the dataset size.")

            generator = torch.Generator().manual_seed(int(subset_seed))
            permutation = torch.randperm(len(dataset), generator=generator).tolist()
            self.source_indices = permutation[: self.source_pool_size]

        self.augment_probability = float(augment_probability)
        if not 0.0 <= self.augment_probability <= 1.0:
            raise ValueError("augment_probability must be between 0 and 1.")

        self.length = len(self.source_indices) if samples_per_epoch is None else int(samples_per_epoch)
        if self.length <= 0:
            raise ValueError("samples_per_epoch must be positive.")

        self.num_pseudo_classes = len(self.source_indices)
        self.batch_size = None if batch_size is None else int(batch_size)
        self.negatives_ratio = None if negatives_ratio is None else float(negatives_ratio)
        if self.negatives_ratio is not None:
            if self.batch_size is None or self.batch_size <= 0:
                raise ValueError("batch_size must be positive when negatives_ratio is set.")
            if not 0.0 < self.negatives_ratio <= 1.0:
                raise ValueError("negatives_ratio must be greater than 0 and at most 1.")
            self.num_negatives_per_batch = int(self.negatives_ratio * self.batch_size)
            if self.num_negatives_per_batch < 1:
                raise ValueError("negatives_ratio * batch_size must be at least 1.")
            if self.num_negatives_per_batch > self.num_pseudo_classes:
                raise ValueError("negatives_ratio * batch_size cannot exceed the number of pseudo-classes.")
        else:
            self.num_negatives_per_batch = None

    def __len__(self):
        return self.length

    def _sample_source_for_batch_slot(self, idx):
        batch_idx = idx // self.batch_size
        slot_idx = idx % self.batch_size
        seed = int(batch_idx)

        generator = torch.Generator().manual_seed(seed)
        negative_offsets = torch.randperm(self.num_pseudo_classes, generator=generator)[
            : self.num_negatives_per_batch
        ]
        if slot_idx < self.num_negatives_per_batch:
            source_offset = negative_offsets[slot_idx].item()
        else:
            positive_generator = torch.Generator().manual_seed(seed + 10_000_000 + slot_idx)
            positive_idx = torch.randint(
                self.num_negatives_per_batch,
                size=(1,),
                generator=positive_generator,
            ).item()
            source_offset = negative_offsets[positive_idx].item()
        return self.source_indices[source_offset], source_offset

    def _sample_source(self):
        source_offset = torch.randint(self.num_pseudo_classes, size=(1,)).item()
        return self.source_indices[source_offset], source_offset

    def _transform_image(self, image):
        if torch.rand(1).item() < self.augment_probability:
            return self.transform_strong(image)
        return self.transform_weak(image)

    def __getitem__(self, idx):
        if self.negatives_ratio is None:
            source_index, pseudo_label = self._sample_source()
        else:
            source_index, pseudo_label = self._sample_source_for_batch_slot(int(idx))
        image, _ = self.dataset[source_index]
        image_tensor = self._transform_image(image)
        return image_tensor, torch.tensor(pseudo_label, dtype=torch.long)
