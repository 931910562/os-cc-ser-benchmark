#!/bin/bash
# Latest recorded ANNA experiment launcher.

set -euo pipefail

# Base parameters
WARMUP=20
KNOWN_CLASS=3
BATCH_SIZE=32
LR=0.0001

# Feature names and dimensions
# Configuration format: feature_name|feature_dimension.
# FEATURE_DIM must match the second dimension of features.npy.
FEATURE_CONFIGS=(
    "resnet|2048"
    "egemaps|88"
)

# Parameter search space
LAMBDA1_LIST=(0.05)
LAMBDA2_LIST=(0.05)

# Source-to-target domain pairs
declare -a SOURCES=("C" "C" "C" "E" "E" "E" "M" "M" "M" "Me" "Me" "Me")
declare -a TARGETS=("E" "M" "Me" "C" "M" "Me" "C" "E" "Me" "C" "E" "M")
declare -a EXPNAMES=("C_to_E" "C_to_M" "C_to_Me"
    "E_to_C" "E_to_M" "E_to_Me"
    "M_to_C" "M_to_E" "M_to_Me"
    "Me_to_C" "Me_to_E" "Me_to_M")

TOTAL=${#SOURCES[@]}

# Experiment loop
for FEAT_ITEM in "${FEATURE_CONFIGS[@]}"; do

    IFS='|' read -r FEAT FEATURE_DIM <<<"$FEAT_ITEM"

    DATA_ROOT="/seu_nvme/home/220242615/${FEAT}"
    SAVE_PATH="/seu_share/home/220242615/Anna-main/6.16/${FEAT}/results_bs${BATCH_SIZE}_lr${LR}_dim${FEATURE_DIM}"

    mkdir -p "${SAVE_PATH}"

    echo ""
    echo "======================================"
    echo " Feature    : ${FEAT}"
    echo " Feature dim: ${FEATURE_DIM}"
    echo " Data root  : ${DATA_ROOT}"
    echo " Save path  : ${SAVE_PATH}"
    echo "======================================"
    echo ""

    if [[ ! -d "${DATA_ROOT}" ]]; then
        echo "[WARN] Data root not found: ${DATA_ROOT}"
        echo "[SKIP] Feature: ${FEAT}"
        continue
    fi

    # Iterate over the lambda1 and lambda2 combinations.
    for L1 in "${LAMBDA1_LIST[@]}"; do
        for L2 in "${LAMBDA2_LIST[@]}"; do

            echo ""
            echo "===== ${FEAT} | dim=${FEATURE_DIM} | lambda1=${L1}, lambda2=${L2} ====="
            echo ""

            for ((i = 0; i < TOTAL; i++)); do
                SRC=${SOURCES[$i]}
                TAR=${TARGETS[$i]}
                EXP=${EXPNAMES[$i]}

                echo "=================================================="
                echo "Running: ${FEAT} | dim=${FEATURE_DIM} | ${EXP}"
                echo "lambda1=${L1}, lambda2=${L2}"
                echo "=================================================="

                python train.py \
                    --source "$SRC" \
                    --target "$TAR" \
                    --exp_name "${EXP}_${FEAT}_dim${FEATURE_DIM}_l1${L1}_l2${L2}" \
                    --save_path "${SAVE_PATH}" \
                    --warm_up_epoch "$WARMUP" \
                    --known_class "$KNOWN_CLASS" \
                    --batch_size "$BATCH_SIZE" \
                    --lr "$LR" \
                    --feature_dim "$FEATURE_DIM" \
                    --data_root "${DATA_ROOT}" \
                    --lambda_fda "$L1" \
                    --lambda_dca "$L2"

            done
        done
    done
done

echo "All experiments finished."
