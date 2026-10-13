# STIENet training and evaluation on pre-extracted features.

import os
import argparse
import time
import torch
import torch.nn.functional as F
from torch import optim
from DataLoader_w import make_dataloader
from model import STIENet
from eval import compute_metrics
import torch.multiprocessing as mp

mp.set_sharing_strategy('file_system')


# ------------------- MMD utils -------------------
def gaussian_kernel_matrix(x, y, sigmas):
    beta = 1.0 / (2.0 * (torch.tensor(sigmas) ** 2))
    dist = torch.cdist(x, y, p=2) ** 2
    s = 0.0
    for b in beta:
        s = s + torch.exp(-b.to(x.device) * dist)
    return s


def mmd_loss(x, y, sigmas=(1.0,)):
    xx = gaussian_kernel_matrix(x, x, sigmas)
    yy = gaussian_kernel_matrix(y, y, sigmas)
    xy = gaussian_kernel_matrix(x, y, sigmas)
    n, m = x.size(0), y.size(0)
    if n > 1:
        xx = xx - torch.diag(torch.diag(xx))
        Kxx = xx.sum() / (n * (n - 1))
    else:
        Kxx = 0.0
    if m > 1:
        yy = yy - torch.diag(torch.diag(yy))
        Kyy = yy.sum() / (m * (m - 1))
    else:
        Kyy = 0.0
    Kxy = xy.sum() / (n * m)
    return Kxx + Kyy - 2.0 * Kxy


def class_conditional_mmd(features_s, labels_s, features_t, labels_t, K, sigmas=(1.0,)):
    loss = 0.0
    for k in range(K):
        idx_s = (labels_s == k).nonzero(as_tuple=True)[0]
        idx_t = (labels_t == k).nonzero(as_tuple=True)[0]
        if idx_s.numel() < 2 or idx_t.numel() < 2:
            continue
        fs = features_s[idx_s]
        ft = features_t[idx_t]
        loss += mmd_loss(fs, ft, sigmas)
    return loss


# ------------------- pseudo unknown via dropout perturbation -------------------
def generate_pseudo_unknown_feats(model, x, device, times=5):
    model.train()
    feats = []
    with torch.no_grad():
        for _ in range(times):
            _, h = model(x.to(device))
            feats.append(h.cpu())
    feats = torch.stack(feats, dim=0).mean(dim=0)
    return feats.to(device)


# ------------------- evaluation on target domain -------------------
def evaluate(model, loader, device, K):
    """
    Evaluate the model on the given loader.
    Returns a dictionary with keys: 'OS', 'OS*', 'UNK', 'HOS'.
    """
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for feats, labels, _ in loader:
            feats = feats.to(device)
            logits, _ = model(feats)
            preds = logits.argmax(dim=1)
            all_preds.append(preds.cpu())
            all_labels.append(labels)

    preds = torch.cat(all_preds, dim=0)
    labels = torch.cat(all_labels, dim=0)
    metrics_tuple = compute_metrics(preds, labels, K)
    metric_keys = ['OS', 'OS*', 'UNK', 'HOS']
    metrics_dict = {k: v for k, v in zip(metric_keys, metrics_tuple[:4])}

    return metrics_dict


# ------------------- main training loop -------------------
def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    K = len(args.known_classes)

    # Create the experiment output directory.
    src_name = os.path.basename(os.path.normpath(args.source))
    tgt_name = os.path.basename(os.path.normpath(args.target))
    save_dir = os.path.join(args.save_root, f"{src_name}_to_{tgt_name}")
    os.makedirs(save_dir, exist_ok=True)

    log_path = os.path.join(save_dir, 'train_log.txt')
    final_path = os.path.join(save_dir, 'final_results.txt')
    logf = open(log_path, 'w')

    def log(msg):
        print(msg)
        logf.write(msg + '\n')
        logf.flush()

    src_loader = make_dataloader(
        args.source,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    tgt_loader = make_dataloader(
        args.target,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    model = STIENet(
        n_mels=args.n_mels,
        base_channels=args.base_channels,
        lstm_hidden=args.lstm_hidden,
        lstm_layers=args.lstm_layers,
        num_known=K,
    ).to(device)
    optimizer = optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    best_HOS = 0.0
    best_epoch = 0

    log(f"==== Training started ====")
    log(f"Source: {args.source}")
    log(f"Target: {args.target}")
    log(f"Save dir: {save_dir}\n")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, cnt = 0.0, 0
        t0 = time.time()

        src_iter, tgt_iter = iter(src_loader), iter(tgt_loader)
        steps = max(len(src_loader), len(tgt_loader))

        for _ in range(steps):
            try:
                xs, ys, _ = next(src_iter)
            except StopIteration:
                src_iter = iter(src_loader)
                xs, ys, _ = next(src_iter)
            try:
                xt, yt, _ = next(tgt_iter)
            except StopIteration:
                tgt_iter = iter(tgt_loader)
                xt, yt, _ = next(tgt_iter)

            xs, ys, xt = xs.to(device), ys.to(device), xt.to(device)
            optimizer.zero_grad()

            logits_s, feat_s = model(xs)
            Lcls = F.cross_entropy(logits_s, ys)

            with torch.no_grad():
                logits_t, feat_t = model(xt)
                probs_t = F.softmax(logits_t, dim=1)
                pseudo_labels_t = probs_t.argmax(dim=1)

            from math import isnan

            Lmmd = mmd_loss(feat_s, feat_t, sigmas=(0.5, 1.0, 2.0))
            Lmmd_c = class_conditional_mmd(
                feat_s, ys, feat_t, pseudo_labels_t, K, sigmas=(0.5, 1.0)
            )

            pseudo_unk_feats = generate_pseudo_unknown_feats(
                model, xt, device, times=args.dropout_times
            )
            logits_pseudo = model.classifier(pseudo_unk_feats)
            unk_labels = torch.full(
                (logits_pseudo.size(0),), K, dtype=torch.long, device=device
            )
            Lrej = F.cross_entropy(logits_pseudo, unk_labels)
            Linv = 1.0 / (Lrej + args.phi)

            loss = (
                args.alpha * Lcls
                + args.beta * Lrej
                + args.gamma * Lmmd
                + args.delta * Lmmd_c
                + args.eps * Linv
            )
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            cnt += 1

        avg_loss = total_loss / max(1, cnt)
        metrics = evaluate(model, tgt_loader, device, K)
        t1 = time.time()

        log(
            f"Epoch {epoch:03d} | loss={avg_loss:.4f} | "
            f"OS={metrics['OS']:.3f} OS*={metrics['OS*']:.3f} "
            f"UNK={metrics['UNK']:.3f} HOS={metrics['HOS']:.3f} | time={t1 - t0:.1f}s"
        )

        ckpt_path = os.path.join(save_dir, f"model_epoch{epoch}.pt")
        torch.save(
            {
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            },
            ckpt_path,
        )

        if metrics['HOS'] > best_HOS:
            best_HOS = metrics['HOS']
            best_epoch = epoch
            best_path = os.path.join(save_dir, 'best_model.pt')
            torch.save(
                {
                    'epoch': epoch,
                    'model_state': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'metrics': metrics,
                },
                best_path,
            )
            log(f"  New best model saved (epoch {epoch}, HOS={best_HOS:.3f})")

    # ---------- Final evaluation ----------
    log(
        f"\n==== Training finished ====\nBest epoch: {best_epoch}, Best HOS={best_HOS:.3f}"
    )
    best_model_path = os.path.join(save_dir, 'best_model.pt')
    if os.path.exists(best_model_path):
        ckpt = torch.load(best_model_path, map_location=device)
        model.load_state_dict(ckpt['model_state'])
        final_metrics = evaluate(model, tgt_loader, device, K)
        log(f"Final Evaluation on Target:")
        log(
            f"OS={final_metrics['OS']:.4f}, OS*={final_metrics['OS*']:.4f}, "
            f"UNK={final_metrics['UNK']:.4f}, HOS={final_metrics['HOS']:.4f}"
        )
        with open(final_path, 'w') as f:
            f.write("==== Final Results ====\n")
            f.write(f"Best epoch: {best_epoch}\n")
            for k, v in final_metrics.items():
                f.write(f"{k}={v:.4f}\n")
    else:
        log("Best model not found.")

    log(f"All results saved to {save_dir}")
    logf.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, required=True)
    parser.add_argument('--target', type=str, required=True)
    parser.add_argument('--known-classes', nargs='+', default=['angry', 'happy', 'sad'])
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-5)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--n-mels', type=int, default=1024)
    parser.add_argument('--sr', type=int, default=16000)
    parser.add_argument('--max-length', type=int, default=None)
    parser.add_argument('--base-channels', type=int, default=64)
    parser.add_argument('--lstm-hidden', type=int, default=64)
    parser.add_argument('--lstm-layers', type=int, default=2)
    parser.add_argument('--dropout-times', type=int, default=5)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--beta', type=float, default=1.2)
    parser.add_argument('--gamma', type=float, default=1.0)
    parser.add_argument('--delta', type=float, default=1.0)
    parser.add_argument('--eps', type=float, default=1.0)
    parser.add_argument('--phi', type=float, default=1e-6)
    parser.add_argument(
        '--save_root',
        type=str,
        default='checkpoints',
    )
    args = parser.parse_args()
    train(args)
