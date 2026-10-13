# Load feature arrays and map emotion names to open-set labels.

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# ============================================================
# Known emotion classes (source domain)
# ============================================================
KNOWN_CLASSES = ['angry', 'happy', 'sad']


# ============================================================
# Dataset class
# ============================================================
class FeatureDataset(Dataset):
    """
    Dataset for pre-extracted .npy features and labels.npy

    Expected structure:
        root_dir/
            features.npy : shape [N, D]
            labels.npy   : shape [N]
            label_map.json : {"angry": 0, "happy": 1, "sad": 2, "neutral": 3, ...}

    Each sample:
        feature: torch.FloatTensor [D]
        label:   int (0..K or K for unknown)
        class_name: str
    """

    def __init__(self, root_dir, known_classes=KNOWN_CLASSES):
        self.root_dir = root_dir
        self.known_classes = known_classes

        # === load files ===
        feat_path = os.path.join(root_dir, "features.npy")
        label_path = os.path.join(root_dir, "labels.npy")
        map_path = os.path.join(root_dir, "label_map.json")

        if not all(os.path.exists(p) for p in [feat_path, label_path, map_path]):
            raise FileNotFoundError(
                "Missing one of [features.npy, labels.npy, label_map.json]"
            )

        self.features = np.load(feat_path).astype(np.float32)
        self.labels = np.load(label_path).astype(int)

        with open(map_path, "r") as f:
            self.label_map = json.load(f)

        # === invert label map ===
        self.inv_label_map = {v: k for k, v in self.label_map.items()}

        # === build known/unknown mask ===
        self.known_label_ids = [
            self.label_map[c] for c in known_classes if c in self.label_map
        ]
        self.unknown_label = len(known_classes)

        # === filter data ===
        # Keep known labels 0..K-1 and map unknown samples to K.
        self.mapped_labels = []
        self.class_names = []
        for lbl in self.labels:
            cls_name = self.inv_label_map.get(lbl, "unknown")
            if lbl in self.known_label_ids:
                mapped = self.known_label_ids.index(lbl)
            else:
                mapped = self.unknown_label
            self.mapped_labels.append(mapped)
            self.class_names.append(cls_name)

        self.mapped_labels = torch.tensor(self.mapped_labels, dtype=torch.long)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feat = torch.tensor(self.features[idx])
        label = self.mapped_labels[idx]
        cls_name = self.class_names[idx]
        return feat, label, cls_name


# ============================================================
# Collate function
# ============================================================
def collate_fn(batch):
    feats = [b[0] for b in batch]
    labels = torch.stack([b[1] for b in batch])
    classes = [b[2] for b in batch]
    feats = torch.stack(feats)
    return feats, labels, classes


# ============================================================
# DataLoader builder
# ============================================================
def make_dataloader(root_dir, batch_size=32, shuffle=True, num_workers=2):
    dataset = FeatureDataset(root_dir)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=False,
    )
    return loader
