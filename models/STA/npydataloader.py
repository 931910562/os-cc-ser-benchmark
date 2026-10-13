# Dataset wrapper for pre-extracted NPY features and labels.

import os
import numpy as np
import torch
from torch.utils.data import Dataset


class NpyFeatureDataset(Dataset):
    """
    Dataset for .npy features and labels in the same directory.
    Expects files:
        - features.npy  (shape: N x dim)
        - labels.npy    (shape: N,)
    Options:
        - label_filter: list of labels to keep (for source domain)
        - transform: function(data, label, is_train)
        - skip_pred: function(data, label, is_train) -> bool
    """

    def __init__(
        self,
        root_dir,
        transform=None,
        skip_pred=None,
        is_train=True,
        label_filter=None,
        num_classes=None,
    ):
        self.root_dir = root_dir
        self.transform = transform
        self.skip_pred = skip_pred or (lambda d, l, is_train: False)
        self.is_train = is_train
        self.num_classes = (
            num_classes  # Optional number of classes for one-hot encoding.
        )

        # Load features and labels
        self.features = np.load(os.path.join(root_dir, "features.npy"))
        self.labels = np.load(os.path.join(root_dir, "labels.npy"))

        # Apply label filter if provided
        if label_filter is not None:
            keep_idx = [i for i, l in enumerate(self.labels) if l in label_filter]
            self.features = self.features[keep_idx]
            self.labels = self.labels[keep_idx]

        # Skip samples if skip_pred returns True
        keep_idx = [
            i
            for i, (d, l) in enumerate(zip(self.features, self.labels))
            if not self.skip_pred(d, l, self.is_train)
        ]
        self.features = self.features[keep_idx]
        self.labels = self.labels[keep_idx]

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        data, label = self.features[idx], self.labels[idx]

        # Apply transform if exists
        if self.transform is not None:
            data, label = self.transform(data, label, self.is_train)

        # Ensure data is float32
        data = np.asarray(data, dtype=np.float32)

        # Convert label to one-hot if num_classes is set
        if self.num_classes is not None:
            onehot = np.zeros(self.num_classes, dtype=np.float32)
            if int(label) >= 0 and int(label) < self.num_classes:
                onehot[int(label)] = 1.0
            label = onehot
        else:
            label = np.asarray(label, dtype=np.float32)

        return data, label
