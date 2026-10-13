# SDEOTR training and evaluation on pre-extracted features.

"""
train.py

Training entry for SDEOTR with open-set metrics and log recording.

Metrics:
    OS* : known-class accuracy
    UNK : unknown-sample accuracy
    HOS : harmonic mean of OS* and UNK

Example:
    python train.py \
        --src_dir ./data/source \
        --tgt_dir ./data/target \
        --num_known 3 \
        --input_dim 768 \
        --epochs 50 \
        --batch_size 16 \
        --unknown_threshold 0.5
"""

import argparse
import os
from typing import Dict

import torch
import torch.nn.functional as F
from torch.optim import Adam

from dataloader import get_dataloader, get_eval_loader, InfiniteLoader
from models import SDEOTRModel, SimSiamPredictor
from utils import (
    AverageMeter,
    CSVLogger,
    TextLogger,
    compute_open_set_metrics,
    ensure_dir,
    format_metrics,
    otr_loss,
    save_checkpoint,
    seed_everything,
    simsiam_ssl_loss,
    split_high_low_confidence,
)


def parse_args():
    parser = argparse.ArgumentParser(description="SDEOTR reproduction code")

    # Data
    parser.add_argument(
        "--src_dir", type=str, required=True, help="Source domain directory"
    )
    parser.add_argument(
        "--tgt_dir", type=str, required=True, help="Target domain directory"
    )
    parser.add_argument(
        "--src_name",
        type=str,
        default=None,
        help="Source domain name for logging, e.g., C/E/M/Me",
    )
    parser.add_argument(
        "--tgt_name",
        type=str,
        default=None,
        help="Target domain name for logging, e.g., C/E/M/Me",
    )
    parser.add_argument(
        "--num_known",
        type=int,
        default=3,
        help="Number of known source classes. Use 4 for paper-style Neu/Hap/Ang/Sad.",
    )
    parser.add_argument(
        "--unknown_label",
        type=int,
        default=None,
        help="Unknown label id in target evaluation. Default = num_known.",
    )
    parser.add_argument(
        "--crop_len",
        type=int,
        default=None,
        help="Crop length for frame-level target features [T,D]. None for utterance-level features.",
    )
    parser.add_argument("--num_workers", type=int, default=0)

    # Model
    parser.add_argument("--input_dim", type=int, default=768)
    parser.add_argument("--embed_dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.2)

    # Training
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )

    # SDEOTR hyperparameters
    parser.add_argument("--eta", type=float, default=1.0, help="Weight for SSL loss")
    parser.add_argument("--mu", type=float, default=0.1, help="Weight for OTR loss")
    parser.add_argument(
        "--k",
        type=float,
        default=0.5,
        help="Top-k ratio for high-confidence target group",
    )
    parser.add_argument(
        "--alpha", type=float, default=0.1, help="Feature cost weight in OTR"
    )
    parser.add_argument(
        "--beta", type=float, default=0.001, help="Prediction cost weight in OTR"
    )
    parser.add_argument("--sinkhorn_epsilon", type=float, default=0.05)
    parser.add_argument("--sinkhorn_iter", type=int, default=50)

    # Open-set evaluation
    parser.add_argument(
        "--unknown_threshold",
        type=float,
        default=0.5,
        help="Reject as unknown if max softmax confidence is lower than this value. "
        "Set to -1 to disable unknown rejection.",
    )
    parser.add_argument(
        "--best_metric",
        type=str,
        default="HOS",
        choices=["HOS", "OS_star", "UNK", "overall_acc"],
        help="Metric used to save the best checkpoint on target/test set.",
    )

    # Output/log
    parser.add_argument("--save_dir", type=str, default="./exp_sdeotr")
    parser.add_argument("--eval_interval", type=int, default=1)
    parser.add_argument("--log_interval", type=int, default=20)

    return parser.parse_args()


def move_to_device(batch, device):
    if isinstance(batch, (list, tuple)):
        return [item.to(device, non_blocking=True) for item in batch]
    return batch.to(device, non_blocking=True)


def train_one_epoch(
    model, predictor, src_loader, tgt_loader, optimizer, args, epoch: int, logger
) -> Dict[str, float]:
    model.train()
    predictor.train()

    meters = {
        "total": AverageMeter("total"),
        "ce": AverageMeter("ce"),
        "ssl": AverageMeter("ssl"),
        "otr": AverageMeter("otr"),
    }

    tgt_iter = InfiniteLoader(tgt_loader)
    steps = len(src_loader)

    for step, src_batch in enumerate(src_loader, start=1):
        x_s, y_s = move_to_device(src_batch, args.device)
        x_t, _, x2_t, x3_t = move_to_device(tgt_iter.next(), args.device)

        # 1. Source supervised classification loss.
        logits_s, _ = model(x_s)
        loss_ce = F.cross_entropy(logits_s, y_s)

        # 2. Target SSL-based domain exploration loss.
        loss_ssl = simsiam_ssl_loss(model, predictor, x2_t, x3_t)

        # 3. Target OTR loss.
        logits_t, e_t = model(x_t)
        groups = split_high_low_confidence(logits_t, e_t, k=args.k)
        loss_otr = otr_loss(
            groups["e_high"],
            groups["e_low"],
            groups["p_high"],
            groups["p_low"],
            alpha=args.alpha,
            beta=args.beta,
            epsilon=args.sinkhorn_epsilon,
            sinkhorn_iter=args.sinkhorn_iter,
            detach_plan=True,
        )

        loss = loss_ce + args.eta * loss_ssl + args.mu * loss_otr

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(predictor.parameters()), max_norm=5.0
        )
        optimizer.step()

        batch_size = x_s.size(0)
        meters["total"].update(loss.item(), batch_size)
        meters["ce"].update(loss_ce.item(), batch_size)
        meters["ssl"].update(loss_ssl.item(), batch_size)
        meters["otr"].update(loss_otr.item(), batch_size)

        if step % args.log_interval == 0 or step == steps:
            logger.log(
                "Epoch [{:03d}] Step [{:04d}/{:04d}] "
                "loss={:.4f} ce={:.4f} ssl={:.4f} otr={:.4f}".format(
                    epoch,
                    step,
                    steps,
                    meters["total"].avg,
                    meters["ce"].avg,
                    meters["ssl"].avg,
                    meters["otr"].avg,
                )
            )

    return {k: v.avg for k, v in meters.items()}


@torch.no_grad()
def evaluate(model, loader, args, split_name: str, logger) -> Dict[str, float]:
    model.eval()

    all_logits = []
    all_labels = []

    for batch in loader:
        x, y = move_to_device(batch, args.device)
        logits, _ = model(x)
        all_logits.append(logits.cpu())
        all_labels.append(y.cpu())

    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)
    unknown_label = args.num_known if args.unknown_label is None else args.unknown_label

    threshold = None if args.unknown_threshold < 0 else args.unknown_threshold
    metrics = compute_open_set_metrics(
        logits,
        labels,
        unknown_label=unknown_label,
        unknown_threshold=threshold,
    )

    logger.log(format_metrics("[{}]".format(split_name), metrics))
    return metrics


def main():
    args = parse_args()
    seed_everything(args.seed)
    ensure_dir(args.save_dir)

    logger = TextLogger(os.path.join(args.save_dir, "train.log"))
    csv_logger = CSVLogger(
        os.path.join(args.save_dir, "metrics.csv"),
        fieldnames=[
            "epoch",
            "train_loss",
            "train_ce",
            "train_ssl",
            "train_otr",
            "source_overall_acc",
            "source_OS_star",
            "source_UNK",
            "source_HOS",
            "source_mean_conf",
            "target_overall_acc",
            "target_OS_star",
            "target_UNK",
            "target_HOS",
            "target_mean_conf",
            "best_metric",
            "is_best",
        ],
    )

    src_loader, tgt_loader = get_dataloader(
        args.src_dir,
        args.tgt_dir,
        batch_size=args.batch_size,
        num_known=args.num_known,
        unknown_label=args.unknown_label,
        crop_len=args.crop_len,
        num_workers=args.num_workers,
    )

    src_eval_loader = get_eval_loader(
        args.src_dir,
        domain="source",
        batch_size=args.batch_size,
        num_known=args.num_known,
        unknown_label=args.unknown_label,
        num_workers=args.num_workers,
    )
    tgt_eval_loader = get_eval_loader(
        args.tgt_dir,
        domain="target",
        batch_size=args.batch_size,
        num_known=args.num_known,
        unknown_label=args.unknown_label,
        num_workers=args.num_workers,
    )

    model = SDEOTRModel(
        input_dim=args.input_dim,
        embed_dim=args.embed_dim,
        num_classes=args.num_known,
        dropout=args.dropout,
    ).to(args.device)
    predictor = SimSiamPredictor(
        embed_dim=args.embed_dim, hidden_dim=args.embed_dim
    ).to(args.device)

    optimizer = Adam(
        list(model.parameters()) + list(predictor.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_metric_value = -1.0
    best_path = os.path.join(args.save_dir, "best.pt")
    last_path = os.path.join(args.save_dir, "last.pt")

    logger.log("========== SDEOTR Training ==========")
    logger.log(str(args))
    logger.log(
        "Experiment: {} -> {}".format(
            args.src_name or args.src_dir, args.tgt_name or args.tgt_dir
        )
    )
    if args.unknown_threshold < 0:
        logger.log(
            "Warning: unknown rejection is disabled. UNK and HOS may be 0 if unknown samples exist."
        )
    else:
        logger.log(
            "Unknown rejection threshold = {:.4f}".format(args.unknown_threshold)
        )
    logger.log("Best checkpoint metric = target/{}".format(args.best_metric))

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(
            model, predictor, src_loader, tgt_loader, optimizer, args, epoch, logger
        )

        source_metrics = {
            "overall_acc": 0.0,
            "OS_star": 0.0,
            "UNK": 0.0,
            "HOS": 0.0,
            "mean_conf": 0.0,
            "num_known": 0,
            "num_unknown": 0,
        }
        target_metrics = {
            "overall_acc": 0.0,
            "OS_star": 0.0,
            "UNK": 0.0,
            "HOS": 0.0,
            "mean_conf": 0.0,
            "num_known": 0,
            "num_unknown": 0,
        }
        is_best = 0

        if epoch % args.eval_interval == 0:
            logger.log("----- Evaluation at epoch {:03d} -----".format(epoch))
            source_metrics = evaluate(
                model, src_eval_loader, args, split_name="source", logger=logger
            )
            target_metrics = evaluate(
                model, tgt_eval_loader, args, split_name="target/test", logger=logger
            )

            current_metric = float(target_metrics[args.best_metric])
            if current_metric > best_metric_value:
                best_metric_value = current_metric
                is_best = 1
                save_checkpoint(
                    best_path,
                    model,
                    predictor,
                    optimizer,
                    epoch,
                    best_metric_value,
                    args,
                )
                logger.log(
                    "Saved best checkpoint to {}, best target/{}={:.4f}".format(
                        best_path, args.best_metric, best_metric_value
                    )
                )

        csv_logger.log(
            {
                "epoch": epoch,
                "train_loss": train_metrics["total"],
                "train_ce": train_metrics["ce"],
                "train_ssl": train_metrics["ssl"],
                "train_otr": train_metrics["otr"],
                "source_overall_acc": source_metrics["overall_acc"],
                "source_OS_star": source_metrics["OS_star"],
                "source_UNK": source_metrics["UNK"],
                "source_HOS": source_metrics["HOS"],
                "source_mean_conf": source_metrics["mean_conf"],
                "target_overall_acc": target_metrics["overall_acc"],
                "target_OS_star": target_metrics["OS_star"],
                "target_UNK": target_metrics["UNK"],
                "target_HOS": target_metrics["HOS"],
                "target_mean_conf": target_metrics["mean_conf"],
                "best_metric": best_metric_value,
                "is_best": is_best,
            }
        )

        save_checkpoint(
            last_path, model, predictor, optimizer, epoch, best_metric_value, args
        )

    logger.log("Training finished.")
    logger.log("Best target/{} = {:.4f}".format(args.best_metric, best_metric_value))
    logger.log("Text log saved to: {}".format(os.path.join(args.save_dir, "train.log")))
    logger.log(
        "CSV log saved to: {}".format(os.path.join(args.save_dir, "metrics.csv"))
    )


if __name__ == "__main__":
    main()
