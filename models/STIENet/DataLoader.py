# Load class-folder feature files for the evaluation module.

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# ============================================================
# Known emotion classes (source domain)
# ============================================================
KNOWN_CLASSES = ['angry', 'happy', 'sad']


# ============================================================
# Utility: scan dataset directory
# ============================================================
def scan_feature_dataset(root_dir, known_classes=KNOWN_CLASSES):
    """Scan a directory containing ResNet50 feature files.

    Expected structure:
        root_dir/
            angry/
                xxx.npy
            happy/
            sad/
            neutral/     (unknown class)
            surprise/    (unknown class)

    Args:
        root_dir: Root path of the dataset.
        known_classes: Names of known emotion classes.

    Returns:
        Tuples of (feature_path, label, class_name). Unknown classes
        receive label len(known_classes).
    """
    items = []
    if not os.path.exists(root_dir):
        raise FileNotFoundError(f"? Dataset path not found: {root_dir}")

    class_names = [
        d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))
    ]
    class_names.sort()

    for cls in class_names:
        cls_dir = os.path.join(root_dir, cls)
        files = [f for f in os.listdir(cls_dir) if f.endswith(".npy")]
        if len(files) == 0:
            continue

        # Assign known labels 0..K-1 and unknown label K.
        label = known_classes.index(cls) if cls in known_classes else len(known_classes)

        for f in files:
            path = os.path.join(cls_dir, f)
            items.append((path, label, cls))

    return items


# ============================================================
# Dataset for ResNet50 features
# ============================================================
class FeatureDataset(Dataset):
    """
    Dataset for pre-extracted ResNet50 feature vectors (.npy files).

    Each sample:
        feature: torch.FloatTensor of shape [2048]
        label:   int (0..K or K for unknown)
        path:    file path
        class_name: str (folder name)
    """

    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label, cls_name = self.items[idx]
        feat = np.load(path).astype(np.float32)  # [2048]
        feat = torch.tensor(feat)
        return feat, torch.tensor(label, dtype=torch.long), path, cls_name


# ============================================================
# Custom collate function for batching
# ============================================================
def collate_fn(batch):
    """
    Merge a list of (feature, label, path, class_name) tuples into batch tensors.
    """
    feats = [b[0] for b in batch]
    labels = torch.stack([b[1] for b in batch])
    paths = [b[2] for b in batch]
    classes = [b[3] for b in batch]
    feats = torch.stack(feats)  # [B, 2048]
    return feats, labels, paths, classes


# ============================================================
# DataLoader builder
# ============================================================
def make_dataloader(root_dir, batch_size=32, shuffle=True, num_workers=2):
    """
    Build a DataLoader for ResNet50 feature vectors.

    Args:
        root_dir (str): Root directory of the feature dataset.
        batch_size (int): Number of samples per batch.
        shuffle (bool): Whether to shuffle data each epoch.
        num_workers (int): Number of worker threads for loading.

    Returns:
        torch.utils.data.DataLoader
    """
    items = scan_feature_dataset(root_dir)
    dataset = FeatureDataset(items)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=False,
    )
    return loader
