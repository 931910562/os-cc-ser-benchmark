#!/bin/bash
# Latest recorded BCL experiment launcher.

# Experiment parameters
EPOCHS=100
N_CLASSES=4
TARENT_STEP=50
BATCH_SIZE=32
LR=1e-3
FEATURE_DIM=2048

LAMBDA1=(0.001)
LAMBDA2=(0.001)

exptimes=1
DSET="OfficeHome"
DOMAINS=("C" "M" "E" "Me")
SOURCE_CLS_IDX="0 1 2"

# Feature configuration
FEATURES=("resnet")

# Experiment loop
for FEAT in "${FEATURES[@]}"; do

    DATA_ROOT="/seu_nvme/home/220242615/${FEAT}"
    SAVE_PATH="/seu_share/home/220242615/BCL/6.18/${FEAT}/bs${BATCH_SIZE}_lr${LR}"

    mkdir -p "${SAVE_PATH}"

    echo ""
    echo "======================================"
    echo " Feature: ${FEAT}"
    echo " Data root: ${DATA_ROOT}"
    echo " Save path: ${SAVE_PATH}"
    echo "======================================"
    echo ""

    for SOURCE_DOMAIN in "${DOMAINS[@]}"; do
        for TARGET_DOMAIN in "${DOMAINS[@]}"; do
            if [ "$SOURCE_DOMAIN" != "$TARGET_DOMAIN" ]; then

                # ---------- target_cls_idx ----------
                case $TARGET_DOMAIN in
                "M")
                    TARGET_CLS_IDX="0 1 2 3"
                    ;;
                "E")
                    TARGET_CLS_IDX="0 1 2 3 4"
                    ;;
                "C")
                    TARGET_CLS_IDX="0 1 2 3 5 6"
                    ;;
                "Me")
                    TARGET_CLS_IDX="0 1 2 3 4 5 6"
                    ;;
                *)
                    echo "Unknown target domain: $TARGET_DOMAIN"
                    exit 1
                    ;;
                esac

                SAVE_NAME="${SOURCE_DOMAIN}_to_${TARGET_DOMAIN}"

                echo "=== ${FEAT} | ${SOURCE_DOMAIN} -> ${TARGET_DOMAIN} ==="
                echo "TARGET_CLS_IDX = ${TARGET_CLS_IDX}"

                python train.py \
                    --epochs $EPOCHS \
                    --n_classes $N_CLASSES \
                    --TarEnt_step $TARENT_STEP \
                    --save_path "${SAVE_PATH}" \
                    --data_root "${DATA_ROOT}" \
                    --lambda1 "${LAMBDA1[@]}" \
                    --lambda2 "${LAMBDA2[@]}" \
                    --batch_size $BATCH_SIZE \
                    --Exptimes $exptimes \
                    --lr $LR \
                    --feature_dim $FEATURE_DIM \
                    --source_domain $SOURCE_DOMAIN \
                    --target_domain $TARGET_DOMAIN \
                    --source_cls_idx $SOURCE_CLS_IDX \
                    --target_cls_idx $TARGET_CLS_IDX

            fi
        done
    done
done
