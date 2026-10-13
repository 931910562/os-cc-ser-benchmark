# UADAL datasets for source and target feature arrays.

import os
import sys
import copy
import random
import numpy as np
import pickle
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler, WeightedRandomSampler

# import utils
# from RandAugment import RandAugment

# Set random seeds for reproducibility
torch.manual_seed(1234)
torch.cuda.manual_seed(1234)
random.seed(1234)
np.random.seed(1234)


class DatasetWithIndicesWrapper(torch.utils.data.Dataset):
    """
    Wraps a feature dataset to include:
    1. Main transforms (for training)
    2. Base transforms (for evaluation)
    3. Optional RandAugment transforms (committee-based augmentations)
    """

    def __init__(self, features, targets, transforms=None, base_transforms=None):
        self.features = features.astype(np.float32)  # Shape: (N, D)
        self.targets = targets.astype(np.int64)
        self.transforms = transforms
        self.base_transforms = base_transforms
        self.committee_size = 1  # Number of RandAugment variants per sample

        # Initialize RandAugment if main transforms are provided
        # self.ra_obj = RandAugment(1, 2.0) if transforms is not None else None
        self.rand_aug_transforms = (
            copy.deepcopy(self.base_transforms) if self.base_transforms else None
        )
        if self.rand_aug_transforms and self.ra_obj:
            self.rand_aug_transforms.transforms.insert(0, self.ra_obj)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        feat = self.features[index]
        target = int(self.targets[index])

        # Convert feature to tensor
        feat_tensor = torch.from_numpy(feat)

        # Generate three versions:
        # 1. Main training feature
        main_feat = self.transforms(feat_tensor) if self.transforms else feat_tensor
        # 2. Base (evaluation) feature
        base_feat = (
            self.base_transforms(feat_tensor) if self.base_transforms else feat_tensor
        )
        # 3. RandAugment feature list (committee)
        rand_feat_lst = (
            [self.rand_aug_transforms(feat_tensor) for _ in range(self.committee_size)]
            if self.rand_aug_transforms
            else [feat_tensor]
        )

        return (main_feat, base_feat, rand_feat_lst), target, index


class UDADataset:
    """
    Dataset class for UDA using .npy features + labels.
    Source domain: only labels 0,1,2
    Target domain: all labels, unknown = labels outside 0,1,2
    """

    def __init__(
        self,
        train_path,
        test_path,
        num_classes,
        train_transforms=None,
        test_transforms=None,
        is_target=False,
        batch_size=128,
    ):
        self.train_path = (
            train_path  # folder containing train_features.npy & train_labels.npy
        )
        self.test_path = (
            test_path  # folder containing test_features.npy & test_labels.npy
        )
        self.num_classes = num_classes
        self.is_target = is_target
        self.batch_size = batch_size

        # Placeholders
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.train_size = None

    def get_num_classes(self):
        # include unknown class
        return self.num_classes

    def long_tail_train(self, key):
        """Long-tail source dataset using saved checkpoint"""
        ixs = pickle.load(open(os.path.join('checkpoints', '{}.pkl'.format(key)), 'rb'))
        self.train_dataset.features = self.train_dataset.features[ixs]
        self.train_dataset.targets = self.train_dataset.targets[ixs]

    def get_dsets(self):
        """Load train, val, test datasets from .npy files"""
        # Train
        train_feats = np.load(os.path.join(self.train_path, "features.npy"))
        train_labels = np.load(os.path.join(self.train_path, "labels.npy"))

        # Filter source domain to only labels 0,1,2
        src_mask = np.isin(train_labels, [0, 1, 2])
        train_feats = train_feats[src_mask]
        train_labels = train_labels[src_mask]

        # Validation: use same as train
        val_feats = train_feats.copy()
        val_labels = train_labels.copy()

        # Test
        test_feats = np.load(os.path.join(self.test_path, "features.npy"))
        test_labels = np.load(os.path.join(self.test_path, "labels.npy"))

        # Target domain: unknown class = labels outside 0,1,2
        test_labels_new = test_labels.copy()
        test_labels_new[~np.isin(test_labels_new, [0, 1, 2])] = (
            self.num_classes - 1
        )  # unknown class index

        # Wrap datasets
        self.train_dataset = DatasetWithIndicesWrapper(train_feats, train_labels)
        self.val_dataset = DatasetWithIndicesWrapper(val_feats, val_labels)
        self.test_dataset = DatasetWithIndicesWrapper(test_feats, test_labels_new)

        return self.train_dataset, self.val_dataset, self.test_dataset

    def get_loaders(self, shuffle=True, num_workers=4, class_balance_train=False):
        """Return train/val/test DataLoaders"""
        if self.train_dataset is None:
            self.get_dsets()

        self.train_size = len(self.train_dataset)
        train_idx = np.arange(len(self.train_dataset))

        if class_balance_train:
            if isinstance(self.train_dataset.targets, np.ndarray):
                targets = self.train_dataset.targets
            else:
                targets = self.train_dataset.targets.numpy()

            count_dict = {
                k: np.sum(targets == k) for k in range(self.get_num_classes())
            }
            class_sample_count = np.array(list(count_dict.values()))
            class_sample_count = class_sample_count / class_sample_count.max() + 1e-8
            weights = 1 / torch.Tensor(class_sample_count)
            sample_weights = [weights[t] for t in targets]
            sample_weights = torch.DoubleTensor(sample_weights)
            train_sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
        else:
            train_sampler = SubsetRandomSampler(train_idx)

        train_loader = DataLoader(
            self.train_dataset,
            sampler=train_sampler,
            batch_size=self.batch_size,
            num_workers=num_workers,
            drop_last=True,
        )
        val_loader = DataLoader(
            self.val_dataset, shuffle=False, batch_size=self.batch_size
        )
        test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size)

        return train_loader, val_loader, test_loader, train_idx
