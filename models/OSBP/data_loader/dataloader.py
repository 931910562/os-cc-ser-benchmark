# OSBP feature-array data loading utilities.

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from itertools import zip_longest


# ============================================================
# NPY dataset: features.npy, labels.npy
# ============================================================
class NPYDataset(Dataset):
    def __init__(self, data_dir, known_labels=None):
        """Load features.npy and labels.npy from data_dir.

        If known_labels is provided, retain only samples with those labels.
        """
        feat_path = os.path.join(data_dir, "features.npy")
        label_path = os.path.join(data_dir, "labels.npy")
        if not os.path.exists(feat_path) or not os.path.exists(label_path):
            raise FileNotFoundError(f"Missing features.npy or labels.npy in {data_dir}")

        self.features = np.load(feat_path)
        self.labels = np.load(label_path).astype(int)
        assert len(self.features) == len(
            self.labels
        ), "features and labels length mismatch"

        if known_labels is not None:
            mask = np.isin(self.labels, known_labels)
            self.features = self.features[mask]
            self.labels = self.labels[mask]
            print(
                f"[INFO] {data_dir}: keep labels {known_labels} -> {self.labels.shape[0]} samples"
            )

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feat = torch.tensor(self.features[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return feat, label


# ============================================================
# Wrapper that mimics original UnalignedDataLoader interface
# ============================================================
class TrainLoaderWrapper:
    def __init__(self, source_loader, target_loader):
        """
        source_loader: DataLoader for source domain (only known classes)
        target_loader: DataLoader for target domain (all classes)
        """
        self.source_loader = source_loader
        self.target_loader = target_loader

    def load_data(self):
        """
        Keep the original API: train_loader.load_data() returns an iterable
        that yields dicts with keys: 'S', 'S_label', 'T'
        """
        return self  # make this object iterable (see __iter__)

    def __iter__(self):
        # We iterate until the shorter loader is exhausted (like zip)
        for (xs, ys), (xt, yt) in zip(self.source_loader, self.target_loader):
            # match the original code's expected types/structure:
            # data['S'] -> source features tensor
            # data['S_label'] -> source labels tensor
            # data['T'] -> target features tensor
            # we ignore target labels here (training doesn't use them), but keep them accessible if needed
            data = {
                'S': xs,
                'S_label': ys,
                'T': xt,
                'T_label': yt,  # optional, not used by train.py but handy for debugging
            }
            yield data

    def __len__(self):
        # length defined by shorter loader (number of batches)
        return min(len(self.source_loader), len(self.target_loader))


# ============================================================
# make_loaders: returns (train_loader_wrapper, test_loader, eval_loader)
# keep API similar to original get_loader(...) call
# ============================================================
def get_loader(
    source_path,
    target_path,
    evaluation_path,
    batch_size=32,
    known_labels=(0, 1, 2),
    num_workers=4,
):
    """
    source_path/target_path/evaluation_path are folders containing:
        - features.npy (shape [N, D])
        - labels.npy   (shape [N])
    For source, only samples with labels in known_labels will be kept.
    Returns:
        train_loader (with .load_data() API),
        test_loader (DataLoader over evaluation_path),
        eval_loader (alias; kept for compatibility if original code expects 3rd return)
    """
    # datasets
    source_dataset = NPYDataset(source_path, known_labels=known_labels)
    target_dataset = NPYDataset(target_path, known_labels=None)
    eval_dataset = NPYDataset(evaluation_path, known_labels=None)

    # data loaders
    source_loader = DataLoader(
        source_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=False,
    )
    target_loader = DataLoader(
        target_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=False,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )

    # wrapper that provides load_data()
    train_loader = TrainLoaderWrapper(source_loader, target_loader)

    # For compatibility with existing scripts that expect (train_loader, test_loader, eval_loader)
    # we return train_loader, eval_loader (as "test_loader" in your train.py), and eval_loader again.
    # In your original main() you did: train_loader, test_loader, eval_loader = get_loader(...)
    return train_loader, eval_loader, eval_loader
