# Train the second-stage STA model from the saved discriminator.

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
import csv  # Added csv module

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


# ------------------- Skip & Transform -------------------
def skip(data, label, is_train):
    return False


def transform_source(data, label, is_train):
    data = np.asarray(data, np.float32)
    label = one_hot(4, int(label))  # Convert 0/1/2 to one-hot encoding
    return data, label


def transform_target_train(data, label, is_train):
    if int(label) in range(3):
        label = one_hot(4, int(label))
    else:
        label = one_hot(4, 3)  # Label unknown classes as 3
    data = np.asarray(data, np.float32)
    return data, label


def transform_target_test(data, label, is_train):
    label = one_hot(7, int(label))  # Keep original labels (0,1,2,3,4)
    data = np.asarray(data, np.float32)
    return data, label


# ------------------- Logger -------------------
class Logger:
    def __init__(self, log_dir, log_name="train.log", csv_name="train.csv", clear=True):
        os.makedirs(log_dir, exist_ok=True)
        self.log_file = os.path.join(log_dir, log_name)
        self.csv_file = os.path.join(log_dir, csv_name)
        if clear:
            if os.path.exists(self.log_file):
                os.remove(self.log_file)
            if os.path.exists(self.csv_file):
                os.remove(self.csv_file)
        self.step = 0
        # Initialize CSV header
        with open(self.csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "CE", "Adv", "Entropy", "CE_EP", "Acc_train"])

    def scalar(self, step, CE=None, Adv=None, Entropy=None, CE_EP=None, Acc_train=None):
        self.step = step
        # Write to log file
        with open(self.log_file, "a") as f:
            log_line = f"Step {step}: "
            log_line += f"CE={CE:.6f} " if CE is not None else ""
            log_line += f"Adv={Adv:.6f} " if Adv is not None else ""
            log_line += f"Entropy={Entropy:.6f} " if Entropy is not None else ""
            log_line += f"CE_EP={CE_EP:.6f} " if CE_EP is not None else ""
            log_line += f"Acc_train={Acc_train:.6f}" if Acc_train is not None else ""
            f.write(log_line + "\n")
        # Write to CSV file
        with open(self.csv_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [step, CE or 0, Adv or 0, Entropy or 0, CE_EP or 0, Acc_train or 0]
            )


# ------------------- Initialize Evaluation CSV (Added) -------------------
def init_eval_csv(log_dir):
    """Initialize CSV file for evaluation results"""
    eval_csv_path = os.path.join(log_dir, "evaluation.csv")
    with open(eval_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["acc_os", "acc_os_star", "OS", "OS_star", "UNK", "HOS"])
    return eval_csv_path


# ------------------- Write Evaluation Results (Added) -------------------
def write_eval_result(eval_csv_path, acc_os, acc_os_star, OS, OS_star, UNK, HOS):
    """Write evaluation metrics to CSV file"""
    with open(eval_csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                round(acc_os, 6),
                round(acc_os_star, 6),
                round(OS, 6),
                round(OS_star, 6),
                round(UNK, 6),
                round(HOS, 6),
            ]
        )


# ------------------- Main -------------------
def main(args):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    batch_size = args.batch_size

    # ------------------- Dataset & DataLoader -------------------
    source_ds = NpyFeatureDataset(
        args.source_root,
        transform=transform_source,
        skip_pred=skip,
        is_train=True,
        label_filter=[0, 1, 2],  # Only use labels 0/1/2 for source domain
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
    )  # No shuffle for test set

    # ------------------- Logger -------------------
    log = Logger(args.log_dir, clear=True)
    eval_csv_path = init_eval_csv(args.log_dir)  # Initialize evaluation CSV

    # ------------------- Networks -------------------
    dim = args.feature_dim
    discriminator_t = CLS_0(in_dim=dim, out_dim=2).to(device)
    discriminator_t.load_state_dict(
        torch.load(args.pretrained_disc, map_location=device)
    )  # Compatible with CPU/GPU
    discriminator = LargeAdversarialNetwork(dim).to(device)
    feature_extractor = Wav2VecFeatureExtractor(input_dim=dim).to(device)
    cls = CLS(in_dim=dim, out_dim=4).to(device)
    net = nn.Sequential(feature_extractor, cls).to(device)

    # ------------------- Optimizers -------------------
    scheduler = lambda step, initial_lr: inverseDecaySheduler(
        step, initial_lr, gamma=10, power=0.75, max_iter=10000
    )
    optimizer_discriminator = OptimWithSheduler(
        optim.SGD(
            discriminator.parameters(),
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

    csv_file = os.path.join(args.log_dir, "training_metrics.csv")
    with open(csv_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                'step',
                'CE',
                'Adv',
                'Entropy',
                'CE_EP',
                'Acc_train',
                'OS',
                'OS_star',
                'UNK',
                'HOS',
            ]
        )

    # ------------------- Stage 1: Weighted Adaptation -------------------
    k = 0
    max_steps = args.step2_stage1

    while k < max_steps:
        for (im_source, label_source), (im_target, label_target) in zip(
            source_train_loader, target_train_loader
        ):
            im_source, label_source = im_source.to(device), label_source.to(device)
            im_target = im_target.to(device)

            _, feature_source, _, predict_prob_source = net.forward(im_source)
            f1, feature_target, _, predict_prob_target = net.forward(im_target)

            domain_prob_source = discriminator.forward(feature_source)
            domain_prob_target = discriminator.forward(feature_target)

            _, _, _, dptarget = discriminator_t.forward(f1.detach())
            batch_size_current = f1.shape[0]
            select_num = 2
            r = torch.sort(dptarget[:, 1].detach(), dim=0)[1][
                batch_size_current - select_num :
            ]
            feature_otherep = torch.index_select(f1, 0, r.view(-1))
            _, _, _, predict_prob_otherep = cls.forward(feature_otherep)
            ce_ep = CrossEntropyLoss(
                torch.cat(
                    [
                        torch.zeros((len(r), 3), device=device),
                        torch.ones((len(r), 1), device=device),
                    ],
                    dim=-1,
                ),
                predict_prob_otherep,
            )

            ce = CrossEntropyLoss(label_source, predict_prob_source)
            entropy = EntropyLoss(
                predict_prob_target, instance_level_weight=dptarget[:, 0].contiguous()
            )
            adv_loss = BCELossForMultiClassification(
                torch.ones_like(domain_prob_source), domain_prob_source
            )
            adv_loss += BCELossForMultiClassification(
                torch.ones_like(domain_prob_target),
                1 - domain_prob_target,
                instance_level_weight=dptarget[:, 0].contiguous(),
            )

            with OptimizerManager([optimizer_cls, optimizer_discriminator]):
                loss = ce + 0.3 * adv_loss + 0.1 * entropy
                loss.backward()

            k += 1

            # ------------------- Batch Accuracy & OS/HOS -------------------
            gt_labels = label_target.argmax(dim=1).cpu().numpy()
            pred_labels = predict_prob_target.argmax(dim=1).cpu().numpy()

            known_mask = gt_labels < 3
            unknown_mask = gt_labels == 3

            OS = (pred_labels == gt_labels).mean()
            OS_star = (
                (pred_labels[known_mask] == gt_labels[known_mask]).mean()
                if known_mask.sum() > 0
                else 0.0
            )
            UNK = (
                (pred_labels[unknown_mask] == gt_labels[unknown_mask]).mean()
                if unknown_mask.sum() > 0
                else 0.0
            )
            HOS = 2 * OS_star * UNK / (OS_star + UNK) if (OS_star + UNK) > 0 else 0.0

            counter = AccuracyCounter()
            counter.addOntBatch(
                variable_to_numpy(predict_prob_source), variable_to_numpy(label_source)
            )
            acc_train = counter.reportAccuracy()

            # ------------------- Write to CSV -------------------
            with open(csv_file, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        k,
                        ce.item(),
                        adv_loss.item(),
                        entropy.item(),
                        ce_ep.item(),
                        acc_train,
                        OS,
                        OS_star,
                        UNK,
                        HOS,
                    ]
                )

            # ------------------- Progress Display -------------------
            progress = (k / max_steps) * 100
            print(
                f"\r[Weighted Adapt] Step {k}/{max_steps} ({progress:.2f}%) | "
                f"CE: {ce.item():.4f} | Adv: {adv_loss.item():.4f} | "
                f"Entropy: {entropy.item():.4f} | CE_EP: {ce_ep.item():.4f} | "
                f"Acc: {acc_train:.4f} | OS: {OS:.4f} | OS*: {OS_star:.4f} | "
                f"UNK: {UNK:.4f} | HOS: {HOS:.4f}",
                end="",
            )

            if k >= max_steps:
                break

    torch.cuda.empty_cache()

    # ------------------- Stage 2: Eliminate Unknown Samples -------------------
    k = 0
    max_steps = args.step2_stage2

    while k < max_steps:
        for (im_source, label_source), (im_target, label_target) in zip(
            source_train_loader, target_train_loader
        ):
            im_source, label_source = im_source.to(device), label_source.to(device)
            im_target = im_target.to(device)

            _, feature_source, _, predict_prob_source = net.forward(im_source)
            f1, feature_target, _, predict_prob_target = net.forward(im_target)

            domain_prob_source = discriminator.forward(feature_source)
            domain_prob_target = discriminator.forward(feature_target)

            _, _, _, dptarget = discriminator_t.forward(f1.detach())
            batch_size_current = f1.shape[0]
            select_num = 2
            r = torch.sort(dptarget[:, 1].detach(), dim=0)[1][
                batch_size_current - select_num :
            ]

            feature_otherep = torch.index_select(f1, 0, r.view(-1))
            _, _, _, predict_prob_otherep = cls.forward(feature_otherep)
            ce_ep = CrossEntropyLoss(
                torch.cat(
                    [
                        torch.zeros((len(r), 3), device=device),
                        torch.ones((len(r), 1), device=device),
                    ],
                    dim=-1,
                ),
                predict_prob_otherep,
            )

            ce = CrossEntropyLoss(label_source, predict_prob_source)
            entropy = EntropyLoss(
                predict_prob_target, instance_level_weight=dptarget[:, 0].contiguous()
            )
            adv_loss = BCELossForMultiClassification(
                torch.ones_like(domain_prob_source), domain_prob_source
            )
            adv_loss += BCELossForMultiClassification(
                torch.ones_like(domain_prob_target),
                1 - domain_prob_target,
                instance_level_weight=dptarget[:, 0].contiguous(),
            )

            with OptimizerManager([optimizer_cls, optimizer_discriminator]):
                loss = ce + 0.3 * adv_loss + 0.1 * entropy + 0.3 * ce_ep
                loss.backward()

            k += 1

            # ------------------- Batch Accuracy & OS/HOS -------------------
            gt_labels = label_target.argmax(dim=1).cpu().numpy()
            pred_labels = predict_prob_target.argmax(dim=1).cpu().numpy()
            # print(pred_labels)
            known_mask = gt_labels < 3
            unknown_mask = gt_labels == 3

            OS = (pred_labels == gt_labels).mean()
            OS_star = (
                (pred_labels[known_mask] == gt_labels[known_mask]).mean()
                if known_mask.sum() > 0
                else 0.0
            )
            UNK = (
                (pred_labels[unknown_mask] == gt_labels[unknown_mask]).mean()
                if unknown_mask.sum() > 0
                else 0.0
            )
            HOS = 2 * OS_star * UNK / (OS_star + UNK) if (OS_star + UNK) > 0 else 0.0

            counter = AccuracyCounter()
            counter.addOntBatch(
                variable_to_numpy(predict_prob_source), variable_to_numpy(label_source)
            )
            acc_train = counter.reportAccuracy()

            # ------------------- Write to CSV -------------------
            with open(csv_file, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        k,
                        ce.item(),
                        adv_loss.item(),
                        entropy.item(),
                        ce_ep.item(),
                        acc_train,
                        OS,
                        OS_star,
                        UNK,
                        HOS,
                    ]
                )

            # ------------------- Progress Display -------------------
            progress = (k / max_steps) * 100
            print(
                f"\r[Eliminate Unknown] Step {k}/{max_steps} ({progress:.2f}%) | "
                f"CE: {ce.item():.4f} | Adv: {adv_loss.item():.4f} | "
                f"Entropy: {entropy.item():.4f} | CE_EP: {ce_ep.item():.4f} | "
                f"Acc: {acc_train:.4f} | OS: {OS:.4f} | OS*: {OS_star:.4f} | "
                f"UNK: {UNK:.4f} | HOS: {HOS:.4f}",
                end="",
            )

            if k >= max_steps:
                break

    # ==========================  Evaluate after Stage2 (Strictly follow specified format) ==========================
    print("\n\n========== Start Test Set Evaluation ==========")

    with TrainingModeManager(
        [feature_extractor, discriminator_t, cls], train=False
    ) as mgr, Accumulator(
        ['predict_prob', 'dp', 'predict_index', 'label']
    ) as accumulator:

        # Total number of test batches
        total_batches = len(target_test_loader)
        print(f"Total test batches: {total_batches}")

        for i, (im, label) in enumerate(target_test_loader):
            im = im.to(device)
            label = label.to(device)

            # Forward through model
            ss, fs, _, predict_prob = net.forward(im)
            _, _, _, dp = discriminator_t.forward(ss)

            # Convert to numpy
            predict_prob, dp, label = [
                variable_to_numpy(x) for x in (predict_prob, dp[:, 1], label)
            ]

            # Convert soft label to hard label (argmax)
            label_hard = np.argmax(label, axis=-1).reshape(-1, 1)

            # Relabel all >=3 as unknown (3)
            label_eval = label_hard.copy()
            label_eval[label_eval >= 3] = 3

            # Predicted class index
            predict_index = np.argmax(predict_prob, axis=-1).reshape(-1, 1)

            # Batch dictionary for accumulator
            batch_scope = {
                'predict_prob': predict_prob,
                'dp': dp,
                'predict_index': predict_index,
                'label': label_eval,
            }

            # Accumulate this batch
            accumulator.updateData(batch_scope)

            # Progress log
    #           if i % 10 == 0:
    #              print(f"Processed batch {i}/{total_batches}")

    # ===== Accumulator merged data summary =====
    print(f"\n=== Accumulator merged data shape ===")
    for name in accumulator.names:
        print(f"{name}: {accumulator[name].shape}")

    # ============================================
    # IMPORTANT: use the merged results in accumulator
    # DO NOT use variables from the loop (which only contain the last batch)
    # ============================================

    # Final evaluation data
    y_true = accumulator['label'].flatten()  # All ground truth labels
    y_pred = accumulator['predict_index'].flatten()  # All predicted labels

    # print("y_true sample:", y_true[:20])
    # print("y_pred sample:", y_pred[:20])
    print("\nEvaluation Label Distribution:", np.unique(y_true))

    # Build confusion matrix
    m = extended_confusion_matrix(
        y_true, y_pred, true_labels=list(range(3)) + [3], pred_labels=range(4)
    )

    # Normalize by row
    cm = m.astype(np.float32)
    row_sums = np.sum(cm, axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm = cm / row_sums

    print("Normalized Confusion Matrix (full test set):\n", cm)

    # Compute OS* accuracy (known classes only)
    known_classes = range(3)
    valid_known_rows = [i for i in known_classes if row_sums[i][0] > 0]
    acc_os_star = (
        sum(cm[i][i] for i in valid_known_rows) / len(valid_known_rows)
        if valid_known_rows
        else 0.0
    )

    # Compute OS accuracy (includes unknown)
    unknown_classes = [3]
    valid_unknown_rows = [i for i in unknown_classes if row_sums[i][0] > 0]
    unk_acc = (
        sum(cm[i][3] for i in valid_unknown_rows) / len(valid_unknown_rows)
        if valid_unknown_rows
        else 0.0
    )
    acc_os = (acc_os_star * len(known_classes) + unk_acc * len(unknown_classes)) / (
        len(known_classes) + len(unknown_classes)
    )

    # Compute OS, OS*, UNK, HOS using raw predictions
    known_mask = y_true < 3
    unknown_mask = y_true == 3

    OS = (y_pred == y_true).mean()
    OS_star = (
        (y_pred[known_mask] == y_true[known_mask]).mean()
        if known_mask.sum() > 0
        else 0.0
    )
    UNK = (
        (y_pred[unknown_mask] == y_true[unknown_mask]).mean()
        if unknown_mask.sum() > 0
        else 0.0
    )
    HOS = 2 * OS_star * UNK / (OS_star + UNK) if (OS_star + UNK) > 0 else 0.0

    # Final evaluation summary
    print(f"\n=== Full Test Set Results ===")
    print(f"acc_os: {acc_os:.4f}, acc_os_star: {acc_os_star:.4f}")
    print(f"OS: {OS:.4f}, OS_star: {OS_star:.4f}, UNK: {UNK:.4f}, HOS: {HOS:.4f}")

    # Save results
    write_eval_result(eval_csv_path, acc_os, acc_os_star, OS, OS_star, UNK, HOS)
    print(f"\nEvaluation Completed! Results saved to: {eval_csv_path}")

    # ==================================================================================================


# ------------------- Main entry -------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--feature_dim", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--step2_stage1", type=int, default=15)
    parser.add_argument("--step2_stage2", type=int, default=4)
    parser.add_argument("--log_dir", type=str, default="log/Step_2")
    parser.add_argument("--pretrained_disc", type=str, default="discriminator_a.pkl")
    parser.add_argument("--source_list", type=str, required=True)
    parser.add_argument("--source_root", type=str, required=True)
    parser.add_argument("--target_list", type=str, required=True)
    parser.add_argument("--target_root", type=str, required=True)
    args = parser.parse_args()

    main(args)
