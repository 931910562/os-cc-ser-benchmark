#!/usr/bin/env bash
# Latest recorded SDEOTR experiment launcher.

set -e

# Example usage for pre-extracted speech features.
# Data format:
#   ./data/source/features.npy, ./data/source/labels.npy
#   ./data/target/features.npy, ./data/target/labels.npy
#
# If following the user's reference dataloader:
#   known labels: 0,1,2
#   target labels > 2 are mapped to unknown label 3
#   set --num_known 3
#
# If following the paper setting:
#   known labels: neutral, happy, angry, sad -> 0,1,2,3
#   target labels >= 4 are unknown
#   set --num_known 4
#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Multi-feature four-corpus cross-domain experiments for SDEOTR
#
# Place all feature representations under the same DATA_ROOT directory.
#
# DATA_ROOT/
#   hubert/
#     C/features.npy
#     C/labels.npy
#     E/features.npy
#     E/labels.npy
#   wav2vec/
#     C/features.npy
#     C/labels.npy
#     ...
#   wavlm/
#     C/features.npy
#     ...
#
# Save each representation's results under the following directory:
# SAVE_ROOT/feature_name/SRC_to_TGT/
# ============================================================

DATA_ROOT=/seu_nvme/home/220242615
SAVE_ROOT=/seu_share/home/220242615/sdeotr/multi_features

DOMAINS=("C" "E" "M" "Me")

SOURCE_DOMAINS=("${DOMAINS[@]}")
TARGET_DOMAINS=("${DOMAINS[@]}")

# ------------------------------------------------------------
# Feature configuration
# Configuration format:
#   feature_name|feature_dimension
# ------------------------------------------------------------
FEATURE_CONFIGS=(
    #"hubert|1024"
    #"wav2vec|1024"
    "wavlm|1024"
    "resnet|2048"
    "egemaps|88"
)

# ------------------------------------------------------------
# Shared training parameters
# ------------------------------------------------------------
NUM_KNOWN=3
EMBED_DIM=512
EPOCHS=100
BATCH_SIZE=32
LR=0.001

ETA=1.0
MU=0.1
K=0.5
ALPHA=0.1
BETA=0.001
UNKNOWN_THRESHOLD=0.5

BEST_METRIC=HOS
EVAL_INTERVAL=1
LOG_INTERVAL=20

mkdir -p "$SAVE_ROOT"

SUMMARY_CSV="$SAVE_ROOT/summary.csv"
echo "feature,input_dim,src,tgt,save_dir,best_metric,best_value,last_target_HOS,last_target_OS_star,last_target_UNK" >"$SUMMARY_CSV"

# ============================================================
# Iterate over feature representations and the 12 directed tasks.
# ============================================================
for FEAT_ITEM in "${FEATURE_CONFIGS[@]}"; do

    IFS='|' read -r FEAT_NAME INPUT_DIM <<<"$FEAT_ITEM"

    FEAT_ROOT="$DATA_ROOT/$FEAT_NAME"

    echo "############################################################"
    echo "Feature   : $FEAT_NAME"
    echo "Feature root: $FEAT_ROOT"
    echo "Input dim : $INPUT_DIM"
    echo "############################################################"

    if [[ ! -d "$FEAT_ROOT" ]]; then
        echo "[WARN] Feature directory not found: $FEAT_ROOT"
        echo "[SKIP] Feature: $FEAT_NAME"
        continue
    fi

    for SRC in "${SOURCE_DOMAINS[@]}"; do
        for TGT in "${TARGET_DOMAINS[@]}"; do

            if [[ "$SRC" == "$TGT" ]]; then
                continue
            fi

            SRC_DIR="$FEAT_ROOT/$SRC"
            TGT_DIR="$FEAT_ROOT/$TGT"

            EXP_NAME="${SRC}_to_${TGT}"
            SAVE_DIR="$SAVE_ROOT/$FEAT_NAME/$EXP_NAME"

            if [[ ! -f "$SRC_DIR/features.npy" || ! -f "$SRC_DIR/labels.npy" ]]; then
                echo "[WARN] Missing source files:"
                echo "       $SRC_DIR/features.npy"
                echo "       $SRC_DIR/labels.npy"
                echo "[SKIP] Feature=$FEAT_NAME | Task=$SRC -> $TGT"
                continue
            fi

            if [[ ! -f "$TGT_DIR/features.npy" || ! -f "$TGT_DIR/labels.npy" ]]; then
                echo "[WARN] Missing target files:"
                echo "       $TGT_DIR/features.npy"
                echo "       $TGT_DIR/labels.npy"
                echo "[SKIP] Feature=$FEAT_NAME | Task=$SRC -> $TGT"
                continue
            fi

            mkdir -p "$SAVE_DIR"

            echo "============================================================"
            echo "Running SDEOTR"
            echo "Feature   : $FEAT_NAME"
            echo "Task      : $SRC -> $TGT"
            echo "Input dim : $INPUT_DIM"
            echo "Source dir: $SRC_DIR"
            echo "Target dir: $TGT_DIR"
            echo "Save dir  : $SAVE_DIR"
            echo "============================================================"

            python train.py \
                --src_dir "$SRC_DIR" \
                --tgt_dir "$TGT_DIR" \
                --src_name "$SRC" \
                --tgt_name "$TGT" \
                --save_dir "$SAVE_DIR" \
                --num_known "$NUM_KNOWN" \
                --input_dim "$INPUT_DIM" \
                --embed_dim "$EMBED_DIM" \
                --epochs "$EPOCHS" \
                --batch_size "$BATCH_SIZE" \
                --lr "$LR" \
                --eta "$ETA" \
                --mu "$MU" \
                --k "$K" \
                --alpha "$ALPHA" \
                --beta "$BETA" \
                --unknown_threshold "$UNKNOWN_THRESHOLD" \
                --best_metric "$BEST_METRIC" \
                --eval_interval "$EVAL_INTERVAL" \
                --log_interval "$LOG_INTERVAL"

            python - "$SAVE_DIR" "$FEAT_NAME" "$INPUT_DIM" "$SRC" "$TGT" "$BEST_METRIC" "$SUMMARY_CSV" <<'PY'
import csv
import os
import sys

save_dir, feat_name, input_dim, src, tgt, best_metric, summary_csv = sys.argv[1:]

metrics_path = os.path.join(save_dir, "metrics.csv")

best_value = ""
last_hos = ""
last_os = ""
last_unk = ""

if os.path.exists(metrics_path):
    with open(metrics_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if best_metric == "overall_acc":
        key = "target_overall_acc"
    else:
        key = "target_" + best_metric

    valid = []
    for r in rows:
        if key in r and r[key] not in ("", None):
            try:
                valid.append(float(r[key]))
            except ValueError:
                pass

    if valid:
        best_value = "{:.6f}".format(max(valid))

    if rows:
        last = rows[-1]
        last_hos = last.get("target_HOS", "")
        last_os = last.get("target_OS_star", "")
        last_unk = last.get("target_UNK", "")

with open(summary_csv, "a", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow([
        feat_name,
        input_dim,
        src,
        tgt,
        save_dir,
        best_metric,
        best_value,
        last_hos,
        last_os,
        last_unk
    ])
PY

            echo "Finished: Feature=$FEAT_NAME | Task=$SRC -> $TGT"

        done
    done
done

echo "All experiments finished."
echo "Summary saved to: $SUMMARY_CSV"

#!/usr/bin/env bash
set -euo pipefail

# Four-corpus cross-domain migration experiments for SDEOTR.
# Default: all directed pairs among C, E, M, Me, excluding same-domain pairs.
# It will run 12 experiments:
# C->E, C->M, C->Me, E->C, E->M, E->Me, M->C, M->E, M->Me, Me->C, Me->E, Me->M.

DATA_ROOT=/seu_nvme/home/220242615/hubert
SAVE_ROOT=/seu_share/home/220242615/sdeotr/hubert

DOMAINS=("C" "E" "M" "Me")

# If you only want C as source, change this line to: SOURCE_DOMAINS=("C")
SOURCE_DOMAINS=("${DOMAINS[@]}")
TARGET_DOMAINS=("${DOMAINS[@]}")

NUM_KNOWN=3
INPUT_DIM=1024
EMBED_DIM=512
EPOCHS=100
BATCH_SIZE=32
LR=0.001
ETA=1.0
MU=0.1
K=0.5
ALPHA=0.1
BETA=0.001
UNKNOWN_THRESHOLD=0.5
BEST_METRIC=HOS
EVAL_INTERVAL=1
LOG_INTERVAL=20

mkdir -p "$SAVE_ROOT"
SUMMARY_CSV="$SAVE_ROOT/summary.csv"
echo "src,tgt,save_dir,best_metric,best_value,last_target_HOS,last_target_OS_star,last_target_UNK" >"$SUMMARY_CSV"

for SRC in "${SOURCE_DOMAINS[@]}"; do
    for TGT in "${TARGET_DOMAINS[@]}"; do
        if [[ "$SRC" == "$TGT" ]]; then
            continue
        fi

        SRC_DIR="$DATA_ROOT/$SRC"
        TGT_DIR="$DATA_ROOT/$TGT"
        EXP_NAME="${SRC}_to_${TGT}"
        SAVE_DIR="$SAVE_ROOT/$EXP_NAME"

        echo "============================================================"
        echo "Running SDEOTR: $SRC -> $TGT"
        echo "Source dir: $SRC_DIR"
        echo "Target dir: $TGT_DIR"
        echo "Save dir  : $SAVE_DIR"
        echo "============================================================"

        python train.py \
            --src_dir "$SRC_DIR" \
            --tgt_dir "$TGT_DIR" \
            --src_name "$SRC" \
            --tgt_name "$TGT" \
            --save_dir "$SAVE_DIR" \
            --num_known "$NUM_KNOWN" \
            --input_dim "$INPUT_DIM" \
            --embed_dim "$EMBED_DIM" \
            --epochs "$EPOCHS" \
            --batch_size "$BATCH_SIZE" \
            --lr "$LR" \
            --eta "$ETA" \
            --mu "$MU" \
            --k "$K" \
            --alpha "$ALPHA" \
            --beta "$BETA" \
            --unknown_threshold "$UNKNOWN_THRESHOLD" \
            --best_metric "$BEST_METRIC" \
            --eval_interval "$EVAL_INTERVAL" \
            --log_interval "$LOG_INTERVAL"

        python - "$SAVE_DIR" "$SRC" "$TGT" "$BEST_METRIC" "$SUMMARY_CSV" <<'PY'
import csv
import os
import sys

save_dir, src, tgt, best_metric, summary_csv = sys.argv[1:]
metrics_path = os.path.join(save_dir, "metrics.csv")

best_value = ""
last_hos = ""
last_os = ""
last_unk = ""

if os.path.exists(metrics_path):
    with open(metrics_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    valid = []
    for r in rows:
        key = "target_" + best_metric if best_metric != "overall_acc" else "target_overall_acc"
        if key in r and r[key] not in ("", None):
            try:
                valid.append(float(r[key]))
            except ValueError:
                pass
    if valid:
        best_value = "{:.6f}".format(max(valid))
    if rows:
        last = rows[-1]
        last_hos = last.get("target_HOS", "")
        last_os = last.get("target_OS_star", "")
        last_unk = last.get("target_UNK", "")

with open(summary_csv, "a", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow([src, tgt, save_dir, best_metric, best_value, last_hos, last_os, last_unk])
PY

        echo "Finished: $SRC -> $TGT"
    done
done

echo "All experiments finished. Summary saved to: $SUMMARY_CSV"
