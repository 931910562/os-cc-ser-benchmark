# STIENet evaluation and open-set metric calculations.

import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from DataLoader import make_dataloader, scan_feature_dataset, FeatureDataset, collate_fn
from model import STIENet


# ============================================================
# Compute metrics for OSDA evaluation
# ============================================================
def compute_metrics(preds, labels, K):
    """
    Compute standard OSDA metrics:
        OS, OS*, UNK, HOS, Acc_known, Acc_unknown
    """
    labels = labels.cpu()
    preds = preds.cpu()
    total = labels.shape[0]

    # Overall accuracy (OS)
    OS = (preds == labels).sum().item() / total

    known_mask = labels != K
    unknown_mask = labels == K
    known_count = known_mask.sum().item()
    unknown_count = unknown_mask.sum().item()

    acc_known = (
        (preds[known_mask] == labels[known_mask]).sum().item() / known_count
        if known_count > 0
        else 0.0
    )
    acc_unknown = (
        (preds[unknown_mask] == labels[unknown_mask]).sum().item() / unknown_count
        if unknown_count > 0
        else 0.0
    )

    OS_star = acc_known
    UNK = acc_unknown
    HOS = 2 * (OS_star * UNK) / (OS_star + UNK) if (OS_star + UNK) > 0 else 0.0

    return OS, OS_star, UNK, HOS, acc_known, acc_unknown


# ============================================================
# Main evaluation
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate STIENet on pre-extracted ResNet features (.npy)"
    )
    parser.add_argument(
        '--model-path', type=str, required=True, help='Path to trained model checkpoint'
    )
    parser.add_argument(
        '--eval-dir',
        type=str,
        required=True,
        help='Directory containing ResNet feature .npy files',
    )
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--known-classes', nargs='+', default=['angry', 'happy', 'sad'])
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    K = len(args.known_classes)

    # --------------------------------------------------------
    # 1. Load model
    # --------------------------------------------------------
    model = STIENet(
        n_mels=768,  # input feature dimension (ResNet output)
        base_channels=64,
        lstm_hidden=64,
        lstm_layers=2,
        num_known=K,
    )
    ckpt = torch.load(args.model_path, map_location=device)
    if 'model_state' in ckpt:
        model.load_state_dict(ckpt['model_state'])
    else:
        model.load_state_dict(ckpt)
    model.to(device)
    model.eval()

    # --------------------------------------------------------
    # 2. Load evaluation dataset (ResNet features)
    # --------------------------------------------------------
    eval_loader = make_dataloader(
        root_dir=args.eval_dir,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    all_preds, all_labels = [], []
    paths, classes = [], []

    # --------------------------------------------------------
    # 3. Forward pass
    # --------------------------------------------------------
    with torch.no_grad():
        for feats, labels, pths, clss in eval_loader:
            feats = feats.to(device)
            logits, _ = model(feats)
            preds = logits.argmax(dim=1)
            all_preds.append(preds.cpu())
            all_labels.append(labels)
            paths += pths
            classes += clss

    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    # --------------------------------------------------------
    # 4. Compute OSDA metrics
    # --------------------------------------------------------
    OS, OS_star, UNK, HOS, acc_known, acc_unknown = compute_metrics(
        all_preds, all_labels, K
    )

    print(f"\nEvaluation on: {args.eval_dir}")
    print(f"OS:  {OS:.4f}")
    print(f"OS*: {OS_star:.4f}")
    print(f"UNK: {UNK:.4f}")
    print(f"HOS: {HOS:.4f}")
    print(f"Acc_known:   {acc_known:.4f}")
    print(f"Acc_unknown: {acc_unknown:.4f}")
