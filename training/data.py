"""Fashion-MNIST dataset loading, transforms, and deterministic splitting."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

from backend.app.domain import NORMALIZATION_MEAN, NORMALIZATION_STD


@dataclass(frozen=True, slots=True)
class DataLoaders:
    train: DataLoader
    validation: DataLoader
    test: DataLoader
    train_size: int
    validation_size: int
    test_size: int


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch and request deterministic GPU kernels."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def stratified_indices(
    targets: np.ndarray, validation_fraction: float, seed: int
) -> tuple[list[int], list[int]]:
    """Return disjoint train and validation indices with per-class proportions."""
    if targets.ndim != 1:
        raise ValueError("Targets must be a one-dimensional array.")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("Validation fraction must be between zero and one.")

    rng = np.random.default_rng(seed)
    train_indices: list[int] = []
    validation_indices: list[int] = []
    for class_id in np.unique(targets):
        class_indices = np.flatnonzero(targets == class_id)
        rng.shuffle(class_indices)
        validation_count = int(round(len(class_indices) * validation_fraction))
        validation_indices.extend(class_indices[:validation_count].tolist())
        train_indices.extend(class_indices[validation_count:].tolist())
    rng.shuffle(train_indices)
    rng.shuffle(validation_indices)
    return train_indices, validation_indices


def _training_transform(augmentation: str) -> transforms.Compose:
    steps: list[object] = []
    if augmentation == "light":
        steps.append(
            transforms.RandomAffine(
                degrees=10,
                translate=(0.08, 0.08),
                scale=(0.95, 1.05),
                fill=0,
            )
        )
    elif augmentation != "none":
        raise ValueError(f"Unsupported augmentation preset: {augmentation}.")
    steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize((NORMALIZATION_MEAN,), (NORMALIZATION_STD,)),
        ]
    )
    return transforms.Compose(steps)


def _evaluation_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((NORMALIZATION_MEAN,), (NORMALIZATION_STD,)),
        ]
    )


def _worker_seed(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def create_data_loaders(
    data_dir: Path,
    batch_size: int,
    validation_fraction: float,
    seed: int,
    augmentation: str,
    num_workers: int,
) -> DataLoaders:
    """Download Fashion-MNIST and build isolated train, validation, and test loaders."""
    metadata_dataset = datasets.FashionMNIST(root=data_dir, train=True, download=True)
    train_indices, validation_indices = stratified_indices(
        metadata_dataset.targets.numpy(), validation_fraction, seed
    )

    train_source = datasets.FashionMNIST(
        root=data_dir,
        train=True,
        transform=_training_transform(augmentation),
        download=False,
    )
    validation_source = datasets.FashionMNIST(
        root=data_dir,
        train=True,
        transform=_evaluation_transform(),
        download=False,
    )
    test_source = datasets.FashionMNIST(
        root=data_dir,
        train=False,
        transform=_evaluation_transform(),
        download=True,
    )

    train_dataset: Dataset = Subset(train_source, train_indices)
    validation_dataset: Dataset = Subset(validation_source, validation_indices)
    generator = torch.Generator().manual_seed(seed)
    common = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": num_workers > 0,
        "worker_init_fn": _worker_seed,
    }
    return DataLoaders(
        train=DataLoader(train_dataset, shuffle=True, generator=generator, **common),
        validation=DataLoader(validation_dataset, shuffle=False, **common),
        test=DataLoader(test_source, shuffle=False, **common),
        train_size=len(train_dataset),
        validation_size=len(validation_dataset),
        test_size=len(test_source),
    )

