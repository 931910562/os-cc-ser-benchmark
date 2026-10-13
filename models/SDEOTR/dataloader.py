# SDEOTR feature-array data loading utilities.

"""
dataloader.py

Dataset and dataloader for SDEOTR.

Expected directory structure:
    source_dir/
        features.npy
        labels.npy
    target_dir/
        features.npy
        labels.npy

features.npy can be:
    - [N, D]       utterance-level features
    - [N, T, D]    frame-level features

labels.npy should contain integer labels starting from 0.

Important:
    - Source domain keeps only known classes: labels < num_known.
    - Target domain maps labels >= num_known to unknown_label for evaluation.
      Target labels are NOT used for training losses, only for validation.

This follows the structure of the user's reference dataloader while adding:
    - configurable num_known
    - target open-set label mapping
    - two random target views for SimSiam SSL
"""

import os
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class EmotionDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        domain: str = "source",
        num_known: int = 3,
        unknown_label: Optional[int] = None,
        return_views: bool = False,
        crop_len: Optional[int] = None,
    ):
        if domain not in {"source", "target"}:
            raise ValueError("domain must be 'source' or 'target'")

        self.data_dir = data_dir
        self.domain = domain
        self.num_known = num_known
        self.unknown_label = num_known if unknown_label is None else unknown_label
        self.return_views = return_views
        self.crop_len = crop_len

        feature_path = os.path.join(data_dir, "features.npy")
        label_path = os.path.join(data_dir, "labels.npy")

        if not os.path.exists(feature_path):
            raise FileNotFoundError(f"Missing {feature_path}")
        if not os.path.exists(label_path):
            raise FileNotFoundError(f"Missing {label_path}")

        self.features = np.load(feature_path)
        self.labels = np.load(label_path)
        self._process()

    def _process(self) -> None:
        if self.domain == "source":
            # Equivalent to the user's reference: mask = labels <= 2 when num_known=3.
            mask = self.labels < self.num_known
            self.features = self.features[mask]
            self.labels = self.labels[mask]
        else:
            # Map all target classes outside source label space to unknown_label.
            self.labels = np.where(
                self.labels >= self.num_known, self.unknown_label, self.labels
            )

    def __len__(self) -> int:
        return len(self.labels)

    def _random_crop(self, x: np.ndarray) -> np.ndarray:
        """
        Random crop for frame-level features [T, D].
        If x is utterance-level [D], return x unchanged.
        """
        if self.crop_len is None or x.ndim != 2:
            return x

        t, d = x.shape
        if t == self.crop_len:
            return x

        if t > self.crop_len:
            start = np.random.randint(0, t - self.crop_len + 1)
            return x[start : start + self.crop_len]

        # Pad short sequences by repeating the last frame.
        pad_len = self.crop_len - t
        pad = np.repeat(x[-1:, :], pad_len, axis=0)
        return np.concatenate([x, pad], axis=0)

    def __getitem__(self, idx: int):
        x_np = self.features[idx]
        y = torch.tensor(self.labels[idx], dtype=torch.long)

        x = torch.tensor(x_np, dtype=torch.float32)

        if self.return_views:
            v1 = torch.tensor(self._random_crop(x_np), dtype=torch.float32)
            v2 = torch.tensor(self._random_crop(x_np), dtype=torch.float32)
            return x, y, v1, v2

        return x, y


class InfiniteLoader:
    """A small helper that endlessly cycles through a dataloader."""

    def __init__(self, loader: DataLoader):
        self.loader = loader
        self.iterator = iter(loader)

    def next(self):
        try:
            return next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.loader)
            return next(self.iterator)


def get_dataloader(
    src_dir: str,
    tgt_dir: str,
    batch_size: int,
    num_known: int = 3,
    unknown_label: Optional[int] = None,
    crop_len: Optional[int] = None,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    src_ds = EmotionDataset(
        src_dir,
        domain="source",
        num_known=num_known,
        unknown_label=unknown_label,
        return_views=False,
        crop_len=crop_len,
    )
    tgt_ds = EmotionDataset(
        tgt_dir,
        domain="target",
        num_known=num_known,
        unknown_label=unknown_label,
        return_views=True,
        crop_len=crop_len,
    )

    src_loader = DataLoader(
        src_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    tgt_loader = DataLoader(
        tgt_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    return src_loader, tgt_loader


def get_eval_loader(
    data_dir: str,
    domain: str,
    batch_size: int,
    num_known: int = 3,
    unknown_label: Optional[int] = None,
    num_workers: int = 0,
) -> DataLoader:
    ds = EmotionDataset(
        data_dir,
        domain=domain,
        num_known=num_known,
        unknown_label=unknown_label,
        return_views=False,
        crop_len=None,
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=True,
    )
