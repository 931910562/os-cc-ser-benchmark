# SVM classification with class-specific OCSVM rejection.

# -*- coding: utf-8 -*-
from __future__ import print_function

import os
import csv
import json
import argparse
import random
import numpy as np

import torch
from torch.utils.data import Dataset

from sklearn.svm import SVC, OneClassSVM
from sklearn.preprocessing import StandardScaler


# ============================================================
# 1. Dataset
# ============================================================
class EmotionDataset(Dataset):
    """
    Dataset for pre-extracted speech emotion features.

    Required files in data_dir:
        features.npy: [N, input_dim]
        labels.npy:   [N]

    Open-set setting:
        source keeps known labels 0..num_known-1;
        target maps labels >= num_known to unknown_label=num_known.
    """

    def __init__(
        self, data_dir, domain='source', num_known=3, closed_set=False, normalize=False
    ):
        super(EmotionDataset, self).__init__()

        self.data_dir = data_dir
        self.domain = domain
        self.num_known = int(num_known)
        self.unknown_label = int(num_known)
        self.closed_set = bool(closed_set)
        self.normalize = bool(normalize)

        feat_path = os.path.join(data_dir, 'features.npy')
        label_path = os.path.join(data_dir, 'labels.npy')

        if not os.path.exists(feat_path):
            raise FileNotFoundError('features.npy not found: {}'.format(feat_path))
        if not os.path.exists(label_path):
            raise FileNotFoundError('labels.npy not found: {}'.format(label_path))

        self.features = np.load(feat_path).astype(np.float32)
        self.labels = np.load(label_path).astype(np.int64)

        if len(self.features) != len(self.labels):
            raise ValueError(
                'features and labels length mismatch: {} vs {}'.format(
                    len(self.features), len(self.labels)
                )
            )

        self._process()

    def _process(self):
        if self.domain not in ['source', 'target']:
            raise ValueError('domain must be source or target')

        if self.closed_set:
            mask = (self.labels >= 0) & (self.labels < self.num_known)
            self.features = self.features[mask]
            self.labels = self.labels[mask]
        else:
            if self.domain == 'source':
                mask = (self.labels >= 0) & (self.labels < self.num_known)
                self.features = self.features[mask]
                self.labels = self.labels[mask]
            else:
                # Target labels are only used for evaluation.
                self.labels = np.where(
                    self.labels >= self.num_known, self.unknown_label, self.labels
                )
                valid = self.labels >= 0
                self.features = self.features[valid]
                self.labels = self.labels[valid]

        if len(self.labels) == 0:
            raise ValueError(
                'No samples left after filtering: domain={}, num_known={}, closed_set={}'.format(
                    self.domain, self.num_known, self.closed_set
                )
            )

        if self.normalize:
            mean = self.features.mean(axis=0, keepdims=True)
            std = self.features.std(axis=0, keepdims=True) + 1e-6
            self.features = (self.features - mean) / std

        self.indices = np.arange(len(self.labels), dtype=np.int64)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = torch.tensor(self.features[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        index = torch.tensor(self.indices[idx], dtype=torch.long)
        return x, y, index


# ============================================================
# 2. Utils
# ============================================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def parse_domains(domain_str):
    return [d.strip() for d in domain_str.split(',') if len(d.strip()) > 0]


def resolve_domain_dir(args, domain):
    """Resolve the feature directory using either supported path convention.

    1. --data_root /seu_nvme/home/220242615 --feature wav2vec
       resolves to /seu_nvme/home/220242615/wav2vec/C for corpus C.
    2. --feature_root /seu_nvme/home/220242615/wav2vec
       resolves to /seu_nvme/home/220242615/wav2vec/C for corpus C.
    """
    candidates = []

    if args.feature_root is not None and len(args.feature_root) > 0:
        candidates.append(os.path.join(args.feature_root, domain))

    if args.data_root is not None and len(args.data_root) > 0:
        if args.feature is not None and len(args.feature) > 0:
            candidates.append(os.path.join(args.data_root, args.feature, domain))
        candidates.append(os.path.join(args.data_root, domain))

    for p in candidates:
        feat_path = os.path.join(p, 'features.npy')
        label_path = os.path.join(p, 'labels.npy')
        if os.path.exists(feat_path) and os.path.exists(label_path):
            return p

    raise FileNotFoundError(
        'Cannot find features.npy / labels.npy for domain {}. Tried:\n{}'.format(
            domain, '\n'.join(candidates)
        )
    )


def compute_osda_metrics(y_true, y_pred, num_known):
    """
    OS* : target known-class mean recall
    UNK : target unknown-class recall
    HOS : harmonic mean of OS* and UNK
    """
    unknown_label = int(num_known)

    known_recalls = []
    class_recalls = {}

    for c in range(num_known):
        mask = y_true == c
        if np.sum(mask) > 0:
            rec = float(np.mean(y_pred[mask] == c))
            known_recalls.append(rec)
            class_recalls[c] = rec
        else:
            class_recalls[c] = np.nan

    if len(known_recalls) > 0:
        os_star = float(np.mean(known_recalls))
    else:
        os_star = 0.0

    unk_mask = y_true == unknown_label
    if np.sum(unk_mask) > 0:
        unk = float(np.mean(y_pred[unk_mask] == unknown_label))
    else:
        unk = 0.0

    if os_star + unk > 0:
        hos = float(2.0 * os_star * unk / (os_star + unk))
    else:
        hos = 0.0

    overall_acc = float(np.mean(y_true == y_pred))

    return {
        'OS_star': os_star,
        'UNK': unk,
        'HOS': hos,
        'overall_acc': overall_acc,
        'class_recalls': class_recalls,
    }


def get_max_svm_score(svm_model, x):
    """
    Return max decision score of multi-class SVM.
    """
    scores = svm_model.decision_function(x)
    scores = np.asarray(scores)

    if scores.ndim == 1:
        scores = scores.reshape(-1, 1)

    return np.max(scores, axis=1), scores


# ============================================================
# 3. SVM + One-Class SVM Open-Set Classifier
# ============================================================
class SVMOCSVMClassifier(object):
    """
    SVM classifier + class-conditional One-Class SVM rejector.

    Training:
        1. Fit a multi-class SVM using source known-class samples.
        2. For each known class c, fit an One-Class SVM using only source samples of class c.

    Inference:
        1. Use multi-class SVM to predict a candidate known class.
        2. Use the corresponding class-conditional OCSVM to decide accept/reject.
        3. If rejected, assign unknown_label=num_known.
    """

    def __init__(
        self,
        num_known=3,
        normalize=True,
        svm_c=1.0,
        svm_kernel='rbf',
        svm_gamma='scale',
        ocsvm_nu=0.05,
        ocsvm_kernel='rbf',
        ocsvm_gamma='scale',
        seed=2026,
    ):

        self.num_known = int(num_known)
        self.unknown_label = int(num_known)

        self.normalize = bool(normalize)

        self.svm_c = float(svm_c)
        self.svm_kernel = svm_kernel
        self.svm_gamma = svm_gamma

        self.ocsvm_nu = float(ocsvm_nu)
        self.ocsvm_kernel = ocsvm_kernel
        self.ocsvm_gamma = ocsvm_gamma

        self.seed = int(seed)

        self.scaler = StandardScaler()
        self.svm = None
        self.ocsvm_models = {}

    def _transform_fit(self, x):
        if self.normalize:
            self.scaler.fit(x)
            return self.scaler.transform(x)
        return x

    def _transform(self, x):
        if self.normalize:
            return self.scaler.transform(x)
        return x

    def fit(self, x_src, y_src):
        x_src = np.asarray(x_src, dtype=np.float32)
        y_src = np.asarray(y_src, dtype=np.int64)

        x_src = self._transform_fit(x_src)

        self.svm = SVC(
            C=self.svm_c,
            kernel=self.svm_kernel,
            gamma=self.svm_gamma,
            decision_function_shape='ovr',
            class_weight='balanced',
            random_state=self.seed,
        )
        self.svm.fit(x_src, y_src)

        self.ocsvm_models = {}

        for c in range(self.num_known):
            x_c = x_src[y_src == c]

            if len(x_c) < 2:
                print(
                    '[WARN] Class {} has too few samples for OCSVM. This class will not reject.'.format(
                        c
                    )
                )
                self.ocsvm_models[c] = None
                continue

            oc = OneClassSVM(
                kernel=self.ocsvm_kernel, gamma=self.ocsvm_gamma, nu=self.ocsvm_nu
            )
            oc.fit(x_c)
            self.ocsvm_models[c] = oc

    def predict(self, x):
        x = np.asarray(x, dtype=np.float32)
        x = self._transform(x)

        # 1. Multi-class SVM predicts candidate known class
        pred_known = self.svm.predict(x).astype(np.int64)

        # SVM decision score, only used for logging
        max_svm_score, all_svm_scores = get_max_svm_score(self.svm, x)

        n = x.shape[0]
        y_pred = pred_known.copy()

        ocsvm_decision_scores = np.zeros(n, dtype=np.float32)
        ocsvm_accept = np.ones(n, dtype=np.int64)

        for i in range(n):
            c = int(pred_known[i])
            oc = self.ocsvm_models.get(c, None)

            if oc is None:
                # Accept predictions for a class when no OCSVM was fitted for it.
                y_pred[i] = c
                ocsvm_decision_scores[i] = 0.0
                ocsvm_accept[i] = 1
                continue

            oc_pred = oc.predict(x[i : i + 1])[0]  # +1 inlier, -1 outlier
            oc_score = oc.decision_function(x[i : i + 1])[0]

            ocsvm_decision_scores[i] = float(oc_score)

            if oc_pred < 0:
                y_pred[i] = self.unknown_label
                ocsvm_accept[i] = 0
            else:
                y_pred[i] = c
                ocsvm_accept[i] = 1

        return (
            y_pred,
            pred_known,
            max_svm_score,
            all_svm_scores,
            ocsvm_decision_scores,
            ocsvm_accept,
        )

    def get_info(self):
        return {
            'num_known': self.num_known,
            'normalize': self.normalize,
            'svm_c': self.svm_c,
            'svm_kernel': self.svm_kernel,
            'svm_gamma': self.svm_gamma,
            'ocsvm_nu': self.ocsvm_nu,
            'ocsvm_kernel': self.ocsvm_kernel,
            'ocsvm_gamma': self.ocsvm_gamma,
        }


# ============================================================
# 4. Run one task
# ============================================================
def run_one_task(args, src_domain, tgt_domain):
    src_dir = resolve_domain_dir(args, src_domain)
    tgt_dir = resolve_domain_dir(args, tgt_domain)

    src_ds = EmotionDataset(
        src_dir,
        domain='source',
        num_known=args.num_known,
        closed_set=False,
        normalize=False,
    )

    tgt_ds = EmotionDataset(
        tgt_dir,
        domain='target',
        num_known=args.num_known,
        closed_set=False,
        normalize=False,
    )

    x_src = src_ds.features
    y_src = src_ds.labels

    x_tgt = tgt_ds.features
    y_tgt = tgt_ds.labels

    if x_src.ndim != 2 or x_tgt.ndim != 2:
        raise ValueError(
            'features must be 2D arrays, got source {} and target {}'.format(
                x_src.shape, x_tgt.shape
            )
        )

    if x_src.shape[1] != x_tgt.shape[1]:
        raise ValueError(
            'Feature dimension mismatch: source {} dim {}, target {} dim {}'.format(
                src_domain, x_src.shape[1], tgt_domain, x_tgt.shape[1]
            )
        )

    model = SVMOCSVMClassifier(
        num_known=args.num_known,
        normalize=args.normalize,
        svm_c=args.svm_c,
        svm_kernel=args.svm_kernel,
        svm_gamma=args.svm_gamma,
        ocsvm_nu=args.ocsvm_nu,
        ocsvm_kernel=args.ocsvm_kernel,
        ocsvm_gamma=args.ocsvm_gamma,
        seed=args.seed,
    )

    model.fit(x_src, y_src)

    y_pred, pred_known, max_svm_score, all_svm_scores, ocsvm_scores, ocsvm_accept = (
        model.predict(x_tgt)
    )

    metrics = compute_osda_metrics(
        y_true=y_tgt, y_pred=y_pred, num_known=args.num_known
    )

    task_name = '{}_to_{}'.format(src_domain, tgt_domain)
    task_dir = os.path.join(args.save_dir, task_name)
    ensure_dir(task_dir)

    # Save predictions for each target sample.
    if args.save_pred:
        pred_path = os.path.join(task_dir, 'predictions.csv')
        with open(pred_path, 'w', newline='') as f:
            writer = csv.writer(f)

            header = [
                'index',
                'true_label',
                'svm_pred_known',
                'final_pred',
                'is_rejected',
                'max_svm_score',
                'ocsvm_score',
                'ocsvm_accept',
            ]

            for c in range(args.num_known):
                header.append('svm_score_cls_{}'.format(c))

            writer.writerow(header)

            for i in range(len(y_tgt)):
                is_rejected = int(y_pred[i] == args.num_known)

                row = [
                    i,
                    int(y_tgt[i]),
                    int(pred_known[i]),
                    int(y_pred[i]),
                    is_rejected,
                    float(max_svm_score[i]),
                    float(ocsvm_scores[i]),
                    int(ocsvm_accept[i]),
                ]

                if all_svm_scores.ndim == 1:
                    row.append(float(all_svm_scores[i]))
                else:
                    for c in range(args.num_known):
                        if c < all_svm_scores.shape[1]:
                            row.append(float(all_svm_scores[i, c]))
                        else:
                            row.append('')

                writer.writerow(row)

    # Save model parameters and fitting information.
    info_path = os.path.join(task_dir, 'ocsvm_info.json')
    with open(info_path, 'w') as f:
        json.dump(model.get_info(), f, indent=2)

    result = {
        'feature': args.feature if args.feature is not None else '',
        'task': task_name,
        'source': src_domain,
        'target': tgt_domain,
        'src_num': len(y_src),
        'tgt_num': len(y_tgt),
        'feature_dim': x_src.shape[1],
        'OS_star': metrics['OS_star'],
        'UNK': metrics['UNK'],
        'HOS': metrics['HOS'],
        'overall_acc': metrics['overall_acc'],
        'normalize': args.normalize,
        'svm_c': args.svm_c,
        'svm_kernel': args.svm_kernel,
        'svm_gamma': args.svm_gamma,
        'ocsvm_nu': args.ocsvm_nu,
        'ocsvm_kernel': args.ocsvm_kernel,
        'ocsvm_gamma': args.ocsvm_gamma,
    }

    for c in range(args.num_known):
        result['recall_cls_{}'.format(c)] = metrics['class_recalls'].get(c, np.nan)

    return result


# ============================================================
# 5. Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()

    # Path
    parser.add_argument(
        '--data_root',
        type=str,
        default='',
        help='Root directory, e.g. /seu_nvme/home/220242615',
    )
    parser.add_argument(
        '--feature',
        type=str,
        default='wav2vec',
        help='Feature name, e.g. wav2vec, egemaps, resnet, hubert',
    )
    parser.add_argument(
        '--feature_root',
        type=str,
        default='',
        help='Direct feature root containing C/E/M/Me',
    )
    parser.add_argument('--save_dir', type=str, default='./ocsvm_results')

    # Domain
    parser.add_argument('--domains', type=str, default='C,E,M,Me')
    parser.add_argument(
        '--source',
        type=str,
        default='',
        help='If set with --target, run only one task.',
    )
    parser.add_argument(
        '--target',
        type=str,
        default='',
        help='If set with --source, run only one task.',
    )

    # Open-set
    parser.add_argument('--num_known', type=int, default=3)

    # Normalization
    parser.add_argument(
        '--normalize',
        type=int,
        default=1,
        help='Use source-domain z-score normalization.',
    )

    # Multi-class SVM
    parser.add_argument('--svm_c', type=float, default=1.0)
    parser.add_argument(
        '--svm_kernel',
        type=str,
        default='rbf',
        choices=['linear', 'rbf', 'poly', 'sigmoid'],
    )
    parser.add_argument('--svm_gamma', type=str, default='scale')

    # One-Class SVM
    parser.add_argument(
        '--ocsvm_nu',
        type=float,
        default=0.05,
        help='Expected outlier ratio in source known class. Larger means easier to reject.',
    )
    parser.add_argument(
        '--ocsvm_kernel',
        type=str,
        default='rbf',
        choices=['linear', 'rbf', 'poly', 'sigmoid'],
    )
    parser.add_argument('--ocsvm_gamma', type=str, default='scale')

    # Others
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--save_pred', type=int, default=1)

    args = parser.parse_args()

    args.normalize = bool(args.normalize)
    args.save_pred = bool(args.save_pred)

    set_seed(args.seed)
    ensure_dir(args.save_dir)

    config_path = os.path.join(args.save_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)

    domains = parse_domains(args.domains)

    if len(args.source) > 0 and len(args.target) > 0:
        tasks = [(args.source, args.target)]
    else:
        tasks = []
        for s in domains:
            for t in domains:
                if s != t:
                    tasks.append((s, t))

    print('=' * 80)
    print('SVM + One-Class SVM open-set baseline')
    print('feature      : {}'.format(args.feature))
    print('feature_root : {}'.format(args.feature_root))
    print('data_root    : {}'.format(args.data_root))
    print('normalize    : {}'.format(args.normalize))
    print('svm_kernel   : {}'.format(args.svm_kernel))
    print('svm_c        : {}'.format(args.svm_c))
    print('svm_gamma    : {}'.format(args.svm_gamma))
    print('ocsvm_kernel : {}'.format(args.ocsvm_kernel))
    print('ocsvm_nu     : {}'.format(args.ocsvm_nu))
    print('ocsvm_gamma  : {}'.format(args.ocsvm_gamma))
    print('=' * 80)

    results = []

    for src_domain, tgt_domain in tasks:
        print('\nRunning task: {} -> {}'.format(src_domain, tgt_domain))

        result = run_one_task(args, src_domain, tgt_domain)
        results.append(result)

        print(
            'Task {:>9s} | OS*={:.2f} | UNK={:.2f} | HOS={:.2f} | Acc={:.2f}'.format(
                result['task'],
                result['OS_star'] * 100.0,
                result['UNK'] * 100.0,
                result['HOS'] * 100.0,
                result['overall_acc'] * 100.0,
            )
        )

    mean_os = np.mean([r['OS_star'] for r in results])
    mean_unk = np.mean([r['UNK'] for r in results])
    mean_hos = np.mean([r['HOS'] for r in results])
    mean_acc = np.mean([r['overall_acc'] for r in results])

    print('\n' + '=' * 80)
    print(
        'Average | OS*={:.2f} | UNK={:.2f} | HOS={:.2f} | Acc={:.2f}'.format(
            mean_os * 100.0, mean_unk * 100.0, mean_hos * 100.0, mean_acc * 100.0
        )
    )
    print('=' * 80)

    # Save the overall result summary.
    csv_path = os.path.join(args.save_dir, 'ocsvm_metrics.csv')

    fieldnames = [
        'feature',
        'task',
        'source',
        'target',
        'src_num',
        'tgt_num',
        'feature_dim',
        'OS_star',
        'UNK',
        'HOS',
        'overall_acc',
        'OS_star_percent',
        'UNK_percent',
        'HOS_percent',
        'overall_acc_percent',
        'normalize',
        'svm_c',
        'svm_kernel',
        'svm_gamma',
        'ocsvm_nu',
        'ocsvm_kernel',
        'ocsvm_gamma',
    ]

    for c in range(args.num_known):
        fieldnames.append('recall_cls_{}'.format(c))

    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for r in results:
            row = dict(r)
            row['OS_star_percent'] = r['OS_star'] * 100.0
            row['UNK_percent'] = r['UNK'] * 100.0
            row['HOS_percent'] = r['HOS'] * 100.0
            row['overall_acc_percent'] = r['overall_acc'] * 100.0
            writer.writerow(row)

        mean_row = {
            'feature': args.feature,
            'task': 'mean',
            'source': '',
            'target': '',
            'src_num': '',
            'tgt_num': '',
            'feature_dim': '',
            'OS_star': mean_os,
            'UNK': mean_unk,
            'HOS': mean_hos,
            'overall_acc': mean_acc,
            'OS_star_percent': mean_os * 100.0,
            'UNK_percent': mean_unk * 100.0,
            'HOS_percent': mean_hos * 100.0,
            'overall_acc_percent': mean_acc * 100.0,
            'normalize': args.normalize,
            'svm_c': args.svm_c,
            'svm_kernel': args.svm_kernel,
            'svm_gamma': args.svm_gamma,
            'ocsvm_nu': args.ocsvm_nu,
            'ocsvm_kernel': args.ocsvm_kernel,
            'ocsvm_gamma': args.ocsvm_gamma,
        }
        writer.writerow(mean_row)

    print('\nSaved metrics to: {}'.format(csv_path))


if __name__ == '__main__':
    main()
