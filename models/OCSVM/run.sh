#!/usr/bin/env bash
# Latest recorded OCSVM experiment launcher.

set -e

DATA_ROOT=/seu_nvme/home/220242615
SAVE_ROOT=/seu_share/home/220242615/ocsvm_multi

FEATURES=("hubert")

NUM_KNOWN=3
DOMAINS="C,E,M,Me"

NORMALIZE=1

SVM_KERNEL=rbf
SVM_C=1.0
SVM_GAMMA=scale

OCSVM_KERNEL=rbf
OCSVM_NU=0.05
OCSVM_GAMMA=scale

SEED=2026
SAVE_PRED=1

for FEATURE in "${FEATURES[@]}"; do
    SAVE_DIR=${SAVE_ROOT}/${FEATURE}/svm_${SVM_KERNEL}_C${SVM_C}_ocsvm_${OCSVM_KERNEL}_nu${OCSVM_NU}
    mkdir -p "${SAVE_DIR}"

    echo "============================================================"
    echo "Running FEATURE=${FEATURE}"
    echo "SAVE_DIR=${SAVE_DIR}"
    echo "============================================================"

    python ocsvm_baseline.py \
        --data_root "${DATA_ROOT}" \
        --feature "${FEATURE}" \
        --save_dir "${SAVE_DIR}" \
        --domains "${DOMAINS}" \
        --num_known "${NUM_KNOWN}" \
        --normalize "${NORMALIZE}" \
        --svm_kernel "${SVM_KERNEL}" \
        --svm_c "${SVM_C}" \
        --svm_gamma "${SVM_GAMMA}" \
        --ocsvm_kernel "${OCSVM_KERNEL}" \
        --ocsvm_nu "${OCSVM_NU}" \
        --ocsvm_gamma "${OCSVM_GAMMA}" \
        --seed "${SEED}" \
        --save_pred "${SAVE_PRED}"
done

echo "All features done."
