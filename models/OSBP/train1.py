# OSBP training and evaluation on pre-extracted features.

from __future__ import print_function
import argparse
import logging
import os
import torch
import torch.nn as nn
import numpy as np
from utils.utils import *
from data_loader.dataloader import get_loader


# =========================
# Train function
# =========================
def train(
    num_epoch,
    args,
    dataset_train,
    dataset_test,
    G,
    C,
    opt_g,
    opt_c,
    bce_with_logits_loss,
    num_class,
    class_list,
):
    # Cross entropy loss for source domain
    criterion = nn.CrossEntropyLoss().cuda() if args.cuda else nn.CrossEntropyLoss()

    iteration = 0
    best_HOS = 0.0  # Track best HOS
    best_result = {}

    logging.info("Training started.")

    for ep in range(num_epoch):
        G.train()
        C.train()

        for batch_idx, data in enumerate(dataset_train):
            iteration += 1

            # =========================
            # Load data
            # =========================
            if args.cuda:
                img_s = data['S'].cuda()
                label_s = data['S_label'].long().cuda()
                img_t = data['T'].cuda()
            else:
                img_s = data['S']
                label_s = data['S_label'].long()
                img_t = data['T']

            if len(img_s) < args.batch_size or len(img_t) < args.batch_size:
                break

            # =========================
            # Source classification
            # =========================
            opt_g.zero_grad()
            opt_c.zero_grad()

            feat_s = G(img_s)
            out_s = C(feat_s)
            loss_s = criterion(out_s, label_s)
            loss_s.backward()

            # =========================
            # Target adversarial learning
            # =========================
            C.set_lambda(1.0)  # GRL weight

            feat_t = G(img_t)
            out_logits = C(feat_t, reverse=True)

            # Numerical stability check
            if torch.isnan(out_logits).any() or torch.isinf(out_logits).any():
                raise RuntimeError("NaN or Inf detected in logits")

            # Known vs unknown logits
            logit_known = torch.logsumexp(
                out_logits[:, : num_class - 1], dim=1, keepdim=True
            )
            logit_unknown = out_logits[:, num_class - 1].view(-1, 1)
            logits_two = torch.cat((logit_known, logit_unknown), dim=1)

            # Target distribution (0.5 / 0.5)
            target_funk = torch.full((img_t.size(0), 2), 0.5, device=logits_two.device)

            loss_t = bce_with_logits_loss(logits_two, target_funk)
            loss_t.backward()

            opt_g.step()
            opt_c.step()

            # =========================
            # Logging training info
            # =========================
            if batch_idx % args.log_interval == 0:
                msg = (
                    f"Epoch {ep} | Batch {batch_idx} | "
                    f"Loss_s: {loss_s.item():.6f} | Loss_t: {loss_t.item():.6f}"
                )
                print(msg)
                logging.info(msg)

            # =========================
            # Periodic evaluation
            # =========================
            if ep > 0 and batch_idx % 1000 == 0:
                OS, OS_star, unk_acc, HOS = test(
                    dataset_test, G, C, args, num_class, class_list
                )

                logging.info(
                    f"[Eval] Epoch {ep} | OS: {OS:.2f} | OS*: {OS_star:.2f} | "
                    f"unk: {unk_acc:.2f} | HOS: {HOS:.2f}"
                )

                # Update best HOS
                if HOS > best_HOS:
                    best_HOS = HOS
                    best_result = {
                        "epoch": ep,
                        "OS": OS,
                        "OS_star": OS_star,
                        "unk": unk_acc,
                        "HOS": HOS,
                    }
                    logging.info("?? New best HOS achieved!")
                    logging.info(str(best_result))

                    # Save best model if enabled
                    if args.save:
                        model_file = os.path.join(
                            args.save_path, f"best_model_epoch{ep}.pt"
                        )
                        torch.save(
                            {
                                'G': G.state_dict(),
                                'C': C.state_dict(),
                                'epoch': ep,
                                'HOS': HOS,
                            },
                            model_file,
                        )

                G.train()
                C.train()

    # =========================
    # Final best result
    # =========================
    print("\n========== Best Result ==========")
    print(best_result)
    logging.info("========== Training Finished ==========")
    logging.info("Best Result:")
    logging.info(str(best_result))


# =========================
# Test function
# =========================
def test(dataset_test, G, C, args, num_class, class_list):
    G.eval()
    C.eval()

    correct = 0
    total = 0
    per_class_num = np.zeros(num_class)
    per_class_correct = np.zeros(num_class)

    with torch.no_grad():
        for data in dataset_test:
            if args.cuda:
                img = data[0].cuda()
                label = data[1].long().cuda()
            else:
                img = data[0]
                label = data[1].long()

            # Clamp unknown labels
            label[label >= num_class - 1] = num_class - 1

            feat = G(img)
            out = C(feat)
            pred = out.argmax(dim=1)

            correct += pred.eq(label).sum().item()
            total += label.size(0)

            for c in range(num_class):
                idx = label == c
                per_class_num[c] += idx.sum().item()
                per_class_correct[c] += (pred[idx] == c).sum().item()

    per_class_acc = per_class_correct / (per_class_num + 1e-6)

    OS = 100.0 * correct / total
    OS_star = 100.0 * np.mean(per_class_acc[:-1])
    unk_acc = 100.0 * per_class_acc[-1]
    HOS = 2 * OS_star * unk_acc / (OS_star + unk_acc + 1e-6)

    print(
        f"\nTest | OS: {OS:.2f}% | OS*: {OS_star:.2f}% | "
        f"unk: {unk_acc:.2f}% | HOS: {HOS:.2f}%\n"
    )

    for i, name in enumerate(class_list):
        print(f"{name}: {per_class_acc[i]:.4f}")

    return OS, OS_star, unk_acc, HOS


# =========================
# Main function
# =========================
def main():
    parser = argparse.ArgumentParser(description='Open-set Domain Adaptation')

    # =========================
    # Training parameters
    # =========================
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--log-interval', type=int, default=100)
    parser.add_argument('--seed', type=int, default=1)

    # =========================
    # Model parameters
    # =========================
    parser.add_argument('--net', type=str, default='resnet152')
    parser.add_argument('--unit_size', type=int, default=1000)
    parser.add_argument('--update_lower', action='store_true', default=False)

    # =========================
    # Paths & dataset
    # =========================
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--source_path', type=str, required=True)
    parser.add_argument('--target_path', type=str, required=True)
    parser.add_argument('--source', type=str, required=True)
    parser.add_argument('--target', type=str, required=True)
    parser.add_argument('--save', action='store_true', default=False)
    parser.add_argument('--save_path', type=str, default='./checkpoint/checkpoint')

    # =========================
    # GPU
    # =========================
    parser.add_argument('--no_cuda', action='store_true', default=False)

    args = parser.parse_args()
    args.cuda = (not args.no_cuda) and torch.cuda.is_available()

    # =========================
    # Seed
    # =========================
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # =========================
    # Ensure save_path exists
    # =========================
    os.makedirs(args.save_path, exist_ok=True)

    # =========================
    # Logging setup in save_path
    # =========================
    log_file = os.path.join(
        args.save_path, f"{args.source}2{args.target}_{args.net}.log"
    )
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='[%(asctime)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    logging.info("========== Experiment Start ==========")
    logging.info(str(args))
    print(f"Logging to {log_file}")

    # =========================
    # Data loader
    # =========================
    train_loader, test_loader, _ = get_loader(
        args.source_path,
        args.target_path,
        args.target_path,
        batch_size=args.batch_size,
        known_labels=(0, 1, 2),
    )

    dataset_train = train_loader.load_data()
    dataset_test = test_loader

    num_class = 4
    class_list = ["angry", "happy", "sad", "unk"]

    # =========================
    # Model & optimizer
    # =========================
    G, C = get_model(args.net, num_class=num_class, unit_size=args.unit_size)
    if args.cuda:
        G.cuda()
        C.cuda()

    opt_c, opt_g = get_optimizer_visda(args.lr, G, C, update_lower=args.update_lower)
    bce_with_logits_loss = nn.BCEWithLogitsLoss()

    # =========================
    # Start training
    # =========================
    train(
        args.epochs,
        args,
        dataset_train,
        dataset_test,
        G,
        C,
        opt_g,
        opt_c,
        bce_with_logits_loss,
        num_class,
        class_list,
    )


if __name__ == '__main__':
    main()
