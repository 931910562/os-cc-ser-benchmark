# Train the first-stage STA classifiers and discriminators.

from npydataloader import *
from utilities1 import *
from wavnetworks import *
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import numpy as np

# import tensorlayer as tl
import os
import argparse
import multiprocessing

multiprocessing.freeze_support()  # Windows

# ------------------- Skip & Transform -------------------


def skip(data, label, is_train):
    return False


def transform_source(data, label, is_train):
    data = np.asarray(data, np.float32)
    label = one_hot(4, int(label))  # Encode the source label with four output slots.
    return data, label


def transform_target_train(data, label, is_train):

    if int(label) in range(3):
        label = one_hot(4, int(label))
    else:
        label = one_hot(4, 3)  # Map unknown samples to label 3.
    data = np.asarray(data, np.float32)
    return data, label


def transform_target_test(data, label, is_train):
    label = one_hot(4, int(label))  # Encode the target label with four output slots.
    data = np.asarray(data, np.float32)
    return data, label


# ------------------- Device -------------------
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# ------------------- Logger -------------------
class Logger:
    def __init__(self, log_dir, log_name="train.log", clear=False):
        os.makedirs(log_dir, exist_ok=True)
        self.log_file = os.path.join(log_dir, log_name)
        if clear and os.path.exists(self.log_file):
            os.remove(self.log_file)
        self.step = 0

    def scalar(self, name, value):
        self.step += 1
        with open(self.log_file, "a") as f:
            f.write(f"Step {self.step}: {name} = {value:.6f}\n")


# ------------------- Main -------------------
def main(args):
    batch_size = args.batch_size
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # ------------------- Dataset -------------------
    source_ds = NpyFeatureDataset(
        args.source_root,
        transform=transform_source,
        skip_pred=skip,
        is_train=True,
        label_filter=[0, 1, 2],  # Retain source labels 0, 1 and 2.
    )

    target_train_ds = NpyFeatureDataset(
        args.target_root,
        transform=transform_target_train,
        skip_pred=skip,
        is_train=True,
    )

    target_test_ds = NpyFeatureDataset(
        args.target_root,
        transform=transform_target_test,
        skip_pred=skip,
        is_train=False,
    )

    source_train_loader = DataLoader(
        source_ds, batch_size=batch_size, shuffle=True, num_workers=0
    )
    target_train_loader = DataLoader(
        target_train_ds, batch_size=batch_size, shuffle=True, num_workers=0
    )
    target_test_loader = DataLoader(
        target_test_ds, batch_size=batch_size, shuffle=False, num_workers=0
    )

    log = Logger(args.log_dir, clear=True)

    # ------------------- Networks -------------------
    # Feature extractor
    dim = args.feature_dim
    feature_extractor = Wav2VecFeatureExtractor(input_dim=dim).to(device)

    # Classifier
    cls = CLS(in_dim=dim, out_dim=4).to(device)

    # Full network
    net = nn.Sequential(feature_extractor, cls).to(device)

    # Discriminators
    discriminator_t = CLS_0(in_dim=dim, out_dim=2).to(device)
    discriminator_p = Discriminator(n=3, input_dim=dim).to(device)

    # ------------------- Optimizers -------------------
    scheduler = lambda step, initial_lr: inverseDecaySheduler(
        step, initial_lr, gamma=10, power=0.75, max_iter=10000
    )
    optimizer_discriminator_t = OptimWithSheduler(
        optim.SGD(
            discriminator_t.parameters(),
            lr=args.lr,
            weight_decay=5e-4,
            momentum=0.9,
            nesterov=True,
        ),
        scheduler,
    )
    optimizer_discriminator_p = OptimWithSheduler(
        optim.SGD(
            discriminator_p.parameters(),
            lr=args.lr,
            weight_decay=5e-4,
            momentum=0.9,
            nesterov=True,
        ),
        scheduler,
    )
    optimizer_cls = OptimWithSheduler(
        optim.SGD(
            cls.parameters(), lr=args.lr, weight_decay=5e-4, momentum=0.9, nesterov=True
        ),
        scheduler,
    )

    # ------------------- Stage 1: Multi-binary classifier -------------------
    k = 0
    max_steps = args.step1_stage1
    while k < max_steps:
        for (im_source, label_source), (im_target, label_target) in zip(
            source_train_loader, target_train_loader
        ):
            # source_labels_idx = label_source.argmax(dim=1).cpu().numpy()
            # target_labels_idx = label_target.argmax(dim=1).cpu().numpy()
            # print("source_label:", source_labels_idx, "target_label:", target_labels_idx)

            im_source = im_source.to(device)
            # print(im_source.shape)
            label_source = label_source.to(device)
            im_target = im_target.to(device)

            # Forward pass
            fs1, feature_source, __, predict_prob_source = net.forward(im_source)
            ft1, feature_target, __, predict_prob_target = net.forward(im_target)
            p0 = discriminator_p.forward(fs1)
            # print(predict_prob_source)
            # Loss
            ce = CrossEntropyLoss(label_source, predict_prob_source)
            d1 = BCELossForMultiClassification(label_source[:, 0:3], p0)

            # Backprop
            with OptimizerManager([optimizer_cls, optimizer_discriminator_p]):
                loss = ce + d1
                loss.backward()

            k += 1

            # Record training measurements.
            log.scalar("CE", ce.item())
            log.scalar("D1", d1.item())
            counter = AccuracyCounter()
            counter.addOntBatch(
                variable_to_numpy(predict_prob_source), variable_to_numpy(label_source)
            )
            acc_train = torch.tensor(
                [counter.reportAccuracy()], dtype=torch.float32, device=device
            )
            log.scalar("Acc_train", acc_train.item())

            # Print stage-one training progress.
            progress = (k / max_steps) * 100
            print(
                f"\rStage1 Step {k}/{max_steps} ({progress:.2f}%) | CE: {ce.item():.4f} | D1: {d1.item():.4f} | Acc: {acc_train.item():.4f}",
                end='',
            )

            if k >= max_steps:
                break

    # ------------------- Stage 2: Known/Unknown discriminator -------------------
    k = 0
    max_steps = args.step1_stage2
    while k < max_steps:
        for (im_source, label_source), (im_target, label_target) in zip(
            source_train_loader, target_train_loader
        ):
            im_source = im_source.to(device)
            label_source = label_source.to(device)
            im_target = im_target.to(device)

            # Forward
            fs1, feature_source, __, predict_prob_source = net.forward(im_source)
            ft1, feature_target, __, predict_prob_target = net.forward(im_target)
            p0 = discriminator_p.forward(fs1)
            p1 = discriminator_p.forward(ft1)
            p2 = torch.sum(p1, dim=-1)

            # Rank outputs for discriminator_t
            __, _, _, dptarget = discriminator_t.forward(ft1.detach())
            batch_size_current = ft1.shape[0]
            select_num = 2
            r = torch.sort(dptarget[:, 1].detach(), dim=0)[1][
                batch_size_current - select_num :
            ]
            feature_otherep = torch.index_select(ft1, 0, r.view(-1))
            _, _, __, predict_prob_otherep = cls.forward(feature_otherep)

            w = torch.sort(p2.detach(), dim=0)[1][batch_size_current - select_num :]
            h = torch.sort(p2.detach(), dim=0)[1][:select_num]
            feature_otherep2 = torch.index_select(ft1, 0, w.view(-1))
            feature_otherep1 = torch.index_select(ft1, 0, h.view(-1))
            _, _, _, pred00 = discriminator_t.forward(feature_otherep2)
            _, _, _, pred01 = discriminator_t.forward(feature_otherep1)
            # print("pred00", pred00.shape, "pred01", pred01.shape)
            # Loss
            ce = CrossEntropyLoss(label_source, predict_prob_source)
            d1 = BCELossForMultiClassification(label_source[:, 0:3], p0)
            d2 = CrossEntropyLoss(
                Variable(
                    torch.from_numpy(
                        np.concatenate(
                            (np.ones((2, 1)), np.zeros((2, 1))), axis=-1
                        ).astype('float32')
                    )
                ).to(device),
                pred00,
            )
            d2 = d2 + CrossEntropyLoss(
                Variable(
                    torch.from_numpy(
                        np.concatenate(
                            (np.zeros((2, 1)), np.ones((2, 1))), axis=-1
                        ).astype('float32')
                    )
                ).to(device),
                pred01,
            )

            # Backprop
            with OptimizerManager(
                [optimizer_cls, optimizer_discriminator_p, optimizer_discriminator_t]
            ):
                loss = ce + d1 + d2
                loss.backward()

            k += 1

            # Record training measurements.
            log.scalar("CE", ce.item())
            log.scalar("D1", d1.item())
            log.scalar("D2", d2.item())
            counter = AccuracyCounter()
            counter.addOntBatch(
                variable_to_numpy(predict_prob_source), variable_to_numpy(label_source)
            )
            acc_train = torch.tensor(
                [counter.reportAccuracy()], dtype=torch.float32, device=device
            )
            log.scalar("Acc_train", acc_train.item())

            # Print stage-two training progress.
            progress = (k / max_steps) * 100
            print(
                f"\rStage2 Step {k}/{max_steps} ({progress:.2f}%) | CE: {ce.item():.4f} | D1: {d1.item():.4f} | D2: {d2.item():.4f} | Acc: {acc_train.item():.4f}",
                end='',
            )

            if k >= max_steps:
                break

    # ------------------- Save Model -------------------
    os.makedirs(args.save_path, exist_ok=True)
    file_name = 'discriminator_a.pkl'
    save_path = os.path.join(args.save_path, file_name)
    torch.save(discriminator_t.state_dict(), save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--net", type=str, default="resnet50")
    parser.add_argument("--step1_stage1", type=int, default=500)
    parser.add_argument("--feature_dim", type=int, default=1024)
    parser.add_argument("--step1_stage2", type=int, default=400)
    parser.add_argument("--save_path", type=str, default="checkpoints")
    parser.add_argument("--log_dir", type=str, default="log/step_1")

    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--target", type=str, required=True)
    parser.add_argument("--source_flist", type=str, required=True)
    parser.add_argument("--target_flist", type=str, required=True)
    parser.add_argument("--source_root", type=str, required=True)
    parser.add_argument("--target_root", type=str, required=True)

    args = parser.parse_args()
    main(args)
