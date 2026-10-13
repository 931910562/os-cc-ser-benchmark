# Load pre-extracted source and target feature arrays.

import os
import torch
from torch.utils import data
import numpy as np


class Base_Dataset(data.Dataset):
    def __init__(self, root, partition, source, target):
        super(Base_Dataset, self).__init__()
        self.root = root
        self.partition = partition
        self.source = source
        self.target = target

    def __len__(self):
        return min(len(self.source_feature), len(self.target_feature))

    def load_feature_dataset(self, openness=None, label_flag=None, filter_source=True):
        """
        Load source and target features & labels.
        If filter_source=True, only keep the first three classes (0,1,2) in source domain.
        """
        # ==== Load source domain ====
        src_feat_path = os.path.join(self.root, self.source, "features.npy")
        src_label_path = os.path.join(self.root, self.source, "labels.npy")

        source_features = torch.tensor(np.load(src_feat_path))
        source_labels = torch.tensor(np.load(src_label_path))

        # ==== Load target domain ====
        tar_feat_path = os.path.join(self.root, self.target, "features.npy")
        tar_label_path = os.path.join(self.root, self.target, "labels.npy")

        target_features = torch.tensor(np.load(tar_feat_path))
        target_labels = torch.tensor(np.load(tar_label_path))

        # ==== Optionally filter source domain ====
        if filter_source:
            # Keep only the first 3 classes (0, 1, 2)
            valid_src_idx = source_labels < 3
            source_features = source_features[valid_src_idx]
            source_labels = source_labels[valid_src_idx]

        # ==== Optionally filter target domain by openness ====
        if openness is not None:
            valid_tar_idx = target_labels < (self.num_class + openness - 1)
            target_features = target_features[valid_tar_idx]
            target_labels = target_labels[valid_tar_idx]

        # ==== Balance the number of samples between source and target ====
        if self.partition == 'train' and len(source_features) > len(target_features):
            k = int(len(source_features) / len(target_features)) + 1
            target_features = target_features.repeat((k, 1))
            target_labels = target_labels.repeat((k,))
        else:
            k = int(len(target_features) / len(source_features)) + 1
            source_features = source_features.repeat((k, 1))
            source_labels = source_labels.repeat((k,))

        return source_features, source_labels, target_features, target_labels

    def __getitem__(self, idx):
        feat_s = self.source_feature[idx]
        feat_t = self.target_feature[idx]
        lbl_s = self.source_label[idx]
        lbl_t = self.target_label[idx]

        if self.label_flag is not None:
            pse_lbl = self.label_flag[idx]
        else:
            pse_lbl = self.num_class

        if self.confidence is not None:
            confidence = self.confidence[idx]
        else:
            confidence = 0.0

        return feat_s, lbl_s, feat_t, lbl_t, (pse_lbl, confidence)


class Home_Dataset(Base_Dataset):
    def __init__(
        self,
        root,
        partition='train',
        source='A',
        target='R',
        label_flag=None,
        confidence=None,
        openness=None,
        filter_source=True,
    ):
        """
        Home emotion dataset loader.
        Args:
            root: root path of the dataset
            partition: 'train' or 'test'
            source: source domain name (e.g. 'M', 'C', 'E', 'Me')
            target: target domain name
            label_flag: pseudo labels (optional)
            confidence: confidence scores (optional)
            openness: controls the number of target classes (optional)
            filter_source: whether to keep only the first three classes in source domain
        """
        super(Home_Dataset, self).__init__(root, partition, source, target)
        self.class_name = ['angry', 'happy', 'sad', 'unk']
        self.num_class = len(self.class_name)
        self.label_flag = label_flag
        self.confidence = confidence

        # Load features and optionally filter source classes
        (
            self.source_feature,
            self.source_label,
            self.target_feature,
            self.target_label,
        ) = self.load_feature_dataset(openness, label_flag, filter_source)
