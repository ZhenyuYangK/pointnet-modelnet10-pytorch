"""Reproducible class-stratified splits of the official training set."""

from collections import defaultdict

import numpy as np
from torch.utils.data import Dataset


def stratified_split(labels: list[int], val_fraction: float, seed: int):
    """Hold out a fraction of each class; return disjoint original indices."""
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be between 0 and 1")
    groups = defaultdict(list)
    for index, label in enumerate(labels):
        groups[label].append(index)
    if not groups:
        raise ValueError("Cannot split an empty dataset")
    rng = np.random.default_rng(seed)
    train, validation = [], []
    for label in sorted(groups):
        indices = rng.permutation(groups[label]).tolist()
        if len(indices) < 2:
            raise ValueError(f"Class {label} needs at least two samples")
        count = max(1, min(len(indices) - 1, round(len(indices) * val_fraction)))
        validation.extend(indices[:count])
        train.extend(indices[count:])
    return sorted(train), sorted(validation)


def limit_per_class(indices: list[int], labels: list[int], limit: int, seed: int):
    """Choose a small subset AFTER splitting; zero means use the whole split."""
    if limit < 0:
        raise ValueError("Class limit cannot be negative")
    if limit == 0:
        return list(indices)
    groups = defaultdict(list)
    for index in indices:
        groups[labels[index]].append(index)
    rng = np.random.default_rng(seed)
    selected = []
    for label in sorted(groups):
        if limit > len(groups[label]):
            raise ValueError(f"Requested {limit} samples for class {label}, only {len(groups[label])} available")
        selected.extend(rng.choice(groups[label], size=limit, replace=False).tolist())
    return sorted(selected)


class CachedPointCloudSubset(Dataset):
    """Cache fixed sampled clouds in RAM to avoid reparsing OFF every epoch.

    The underlying ModelNet10 must have a fixed seed. With multiple workers,
    each worker maintains its own cache. Callers must not modify returned clouds.
    """

    def __init__(self, dataset, indices: list[int]):
        self.dataset = dataset
        self.indices = list(indices)
        self.cache = {}

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        original_index = self.indices[index]
        if original_index not in self.cache:
            self.cache[original_index] = self.dataset[original_index]
        return self.cache[original_index]
