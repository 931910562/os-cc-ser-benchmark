# SDEOTR optimization, loss and training utilities.

"""
utils.py

Losses, metrics, OT regularization, logging, and training helpers for SDEOTR.

Open-set evaluation metrics:
    OS* : accuracy on known target samples only
    UNK : accuracy / recall on unknown target samples
    HOS : harmonic mean of OS* and UNK
"""

import csv
import os
import random
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F


class AverageMeter:
    def __init__(self, name: str):
        self.name = name
        self.value = 0.0
        self.total = 0.0
        self.count = 0

    @property
    def avg(self) -> float:
        return self.total / max(self.count, 1)

    def update(self, value: float, n: int = 1) -> None:
        self.value = float(value)
        self.total += float(value) * n
        self.count += n


def seed_everything(seed: int = 2024) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


class TextLogger:
    """Write the same message to console and a log file."""

    def __init__(self, log_path: str):
        ensure_dir(os.path.dirname(log_path))
        self.log_path = log_path
        with open(self.log_path, "w", encoding="utf-8") as f:
            f.write("")

    def log(self, message: str) -> None:
        print(message)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(str(message) + "\n")


class CSVLogger:
    """Append epoch-level train/eval metrics to a CSV file."""

    def __init__(self, csv_path: str, fieldnames):
        ensure_dir(os.path.dirname(csv_path))
        self.csv_path = csv_path
        self.fieldnames = fieldnames
        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writeheader()

    def log(self, row: Dict[str, float]) -> None:
        clean_row = {}
        for key in self.fieldnames:
            value = row.get(key, "")
            if isinstance(value, float):
                clean_row[key] = "{:.6f}".format(value)
            else:
                clean_row[key] = value
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(clean_row)


def negative_cosine_similarity(p: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """
    D(v,u) = - normalized(v) dot normalized(u)
    SimSiam uses stop-gradient on z outside this function.
    """
    p = F.normalize(p, dim=1)
    z = F.normalize(z, dim=1)
    return -(p * z).sum(dim=1).mean()


def simsiam_ssl_loss(
    model, predictor, x2_t: torch.Tensor, x3_t: torch.Tensor
) -> torch.Tensor:
    """
    SSL-based domain exploration loss:
        L_SSL = 1/2 D(p2, stopgrad(e3)) + 1/2 D(p3, stopgrad(e2))
    """
    e2 = model.encode(x2_t)
    e3 = model.encode(x3_t)
    p2 = predictor(e2)
    p3 = predictor(e3)
    return 0.5 * negative_cosine_similarity(
        p2, e3.detach()
    ) + 0.5 * negative_cosine_similarity(p3, e2.detach())


def split_high_low_confidence(
    logits_t: torch.Tensor,
    embeddings_t: torch.Tensor,
    k: float = 0.5,
) -> Dict[str, torch.Tensor]:
    """
    Split target samples into high-confidence and low-confidence groups.

    Args:
        logits_t: classifier outputs on target samples, [B, C]
        embeddings_t: target embeddings, [B, D]
        k: top-k ratio. Example: k=0.5 uses top 50% confidence samples
           as the high-confidence group.
    """
    if not (0.0 < k < 1.0):
        raise ValueError("k must be in (0, 1), e.g., 0.5")

    probs_t = F.softmax(logits_t, dim=1)
    confidence, _ = probs_t.max(dim=1)

    batch_size = logits_t.size(0)
    num_high = max(1, min(batch_size - 1, int(round(batch_size * k))))

    sorted_idx = torch.argsort(confidence, descending=True)
    high_idx = sorted_idx[:num_high]
    low_idx = sorted_idx[num_high:]

    return {
        "e_high": embeddings_t[high_idx],
        "e_low": embeddings_t[low_idx],
        "p_high": probs_t[high_idx],
        "p_low": probs_t[low_idx],
        "conf_high": confidence[high_idx],
        "conf_low": confidence[low_idx],
    }


def pairwise_euclidean_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.cdist(x, y, p=2)


def sinkhorn_transport_plan(
    cost: torch.Tensor,
    epsilon: float = 0.05,
    max_iter: int = 50,
) -> torch.Tensor:
    """
    Entropic Sinkhorn approximation of the OT transport coupling gamma.

    This implementation uses uniform marginals for the high-confidence and
    low-confidence groups.
    """
    n, m = cost.shape
    device = cost.device
    dtype = cost.dtype

    a = torch.full((n,), 1.0 / n, device=device, dtype=dtype)
    b = torch.full((m,), 1.0 / m, device=device, dtype=dtype)

    cost_stable = cost - cost.min()
    kernel = torch.exp(-cost_stable / epsilon).clamp_min(1e-12)

    u = torch.ones_like(a)
    v = torch.ones_like(b)

    for _ in range(max_iter):
        u = a / (kernel @ v + 1e-8)
        v = b / (kernel.t() @ u + 1e-8)

    gamma = u[:, None] * kernel * v[None, :]
    return gamma


def otr_loss(
    e_high: torch.Tensor,
    e_low: torch.Tensor,
    p_high: torch.Tensor,
    p_low: torch.Tensor,
    alpha: float = 0.1,
    beta: float = 0.001,
    epsilon: float = 0.05,
    sinkhorn_iter: int = 50,
    detach_plan: bool = True,
) -> torch.Tensor:
    """
    Optimal Transport Regularization loss.

    C_joint = alpha * C_fea + beta * C_cls
    L_OTR = - sum_{i,j} C_joint(i,j) * gamma*(i,j)

    Minimizing L_OTR maximizes the OT distance between high-confidence and
    low-confidence target groups, pushing apart known-like and unknown-like
    distributions.
    """
    if e_high.numel() == 0 or e_low.numel() == 0:
        device = e_high.device if e_high.numel() else e_low.device
        return torch.zeros((), device=device)

    c_fea = pairwise_euclidean_distance(e_high, e_low)
    c_cls = pairwise_euclidean_distance(p_high, p_low)
    c_joint = alpha * c_fea + beta * c_cls

    plan_cost = c_joint.detach() if detach_plan else c_joint
    gamma = sinkhorn_transport_plan(plan_cost, epsilon=epsilon, max_iter=sinkhorn_iter)
    if detach_plan:
        gamma = gamma.detach()

    return -torch.sum(c_joint * gamma)


@torch.no_grad()
def compute_open_set_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    unknown_label: int,
    unknown_threshold: Optional[float] = 0.5,
) -> Dict[str, float]:
    """
    Compute open-set metrics.

    Prediction rule:
        1. Known prediction = argmax softmax(logits)
        2. If max softmax confidence < unknown_threshold, predict unknown_label.
           If unknown_threshold is None, the model never predicts unknown.

    Metrics:
        OS* : accuracy over known samples only, labels != unknown_label
        UNK : accuracy / recall over unknown samples only, labels == unknown_label
        HOS : harmonic mean of OS* and UNK

    Notes:
        Because the SDEOTR backbone outputs only known emotion logits, UNK needs
        a rejection threshold or another unknown score. Otherwise UNK will be 0
        on datasets containing unknown samples.
    """
    probs = F.softmax(logits, dim=1)
    conf, pred = probs.max(dim=1)

    if unknown_threshold is not None:
        pred = pred.clone()
        pred[conf < unknown_threshold] = unknown_label

    overall_acc = (pred == labels).float().mean().item()

    known_mask = labels != unknown_label
    unknown_mask = labels == unknown_label

    num_known = int(known_mask.sum().item())
    num_unknown = int(unknown_mask.sum().item())

    if num_known > 0:
        os_star = (pred[known_mask] == labels[known_mask]).float().mean().item()
    else:
        os_star = 0.0

    if num_unknown > 0:
        unk = (pred[unknown_mask] == unknown_label).float().mean().item()
    else:
        unk = 0.0

    if os_star + unk > 0:
        hos = 2.0 * os_star * unk / (os_star + unk)
    else:
        hos = 0.0

    return {
        "overall_acc": overall_acc,
        "OS_star": os_star,
        "UNK": unk,
        "HOS": hos,
        "mean_conf": conf.mean().item(),
        "num_known": num_known,
        "num_unknown": num_unknown,
    }


def format_metrics(prefix: str, metrics: Dict[str, float]) -> str:
    return (
        "{prefix} overall_acc={overall:.4f} OS*={os_star:.4f} "
        "UNK={unk:.4f} HOS={hos:.4f} mean_conf={conf:.4f} "
        "N_known={n_known} N_unknown={n_unknown}"
    ).format(
        prefix=prefix,
        overall=metrics.get("overall_acc", 0.0),
        os_star=metrics.get("OS_star", 0.0),
        unk=metrics.get("UNK", 0.0),
        hos=metrics.get("HOS", 0.0),
        conf=metrics.get("mean_conf", 0.0),
        n_known=int(metrics.get("num_known", 0)),
        n_unknown=int(metrics.get("num_unknown", 0)),
    )


def save_checkpoint(
    path: str, model, predictor, optimizer, epoch: int, best_metric: float, args
) -> None:
    ensure_dir(os.path.dirname(path))
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "predictor": predictor.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_metric": best_metric,
            "args": vars(args) if hasattr(args, "__dict__") else args,
        },
        path,
    )
