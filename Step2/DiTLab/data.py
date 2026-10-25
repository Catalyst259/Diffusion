"""Dataset adapters and deterministic selection of held-out preview examples."""

from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class DataSpec:
    """The model-facing properties an image dataset must provide."""

    image_size: int
    in_channels: int
    num_classes: int
    value_min: float = -1.0
    value_max: float = 1.0


@dataclass(frozen=True)
class DataBundle:
    train_dataset: Dataset
    test_dataset: Dataset
    spec: DataSpec


def build_mnist(root: str | Path, download: bool = True) -> DataBundle:
    """Load the official MNIST splits, retaining their original digit labels.

    Importing this module never downloads or opens data. To substitute another
    dataset, supply a DataBundle whose samples are (CHW tensor, integer label).
    """
    from torchvision import datasets, transforms

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )
    return DataBundle(
        train_dataset=datasets.MNIST(
            str(root), train=True, download=download, transform=transform
        ),
        test_dataset=datasets.MNIST(
            str(root), train=False, download=download, transform=transform
        ),
        spec=DataSpec(image_size=28, in_channels=1, num_classes=10),
    )


def select_test_samples(
    dataset: Dataset, count: int = 10, *, seed: int
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    """Select distinct test examples without changing the training RNG state.

    Digits are randomly sampled, so labels may repeat. They are not replaced
    with an artificial 0..9 sequence. The indices make a preview reproducible.
    """
    if count < 1:
        raise ValueError("Preview count must be positive.")
    if count > len(dataset):
        raise ValueError(
            f"Cannot select {count} distinct examples from {len(dataset)} samples."
        )
    generator = torch.Generator(device="cpu").manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:count].tolist()
    samples = [dataset[index] for index in indices]
    images = torch.stack([sample[0] for sample in samples])
    labels = torch.tensor([int(sample[1]) for sample in samples], dtype=torch.long)
    return images, labels, indices
