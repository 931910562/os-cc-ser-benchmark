#!/bin/bash
# Latest recorded OSBP experiment launcher.

# PyTorch Openset DA with feature / batch / lr grid search

# Experiment parameters
EPOCHS=200
UNIT_SIZE=1000
DOMAINS=("M" "C" "E" "Me")

FEATURES=("wav2vec" "wavlm" "hubert")
BATCH_SIZES=(32 64)
LRS=(0.01 0.001)

# Experiment loop
for FEAT in "${FEATURES[@]}"; do

    DATA_ROOT="/seu_nvme/home/220242615/${FEAT}"

    for BATCH in "${BATCH_SIZES[@]}"; do
        for LR in "${LRS[@]}"; do

            CHECKPOINT_ROOT="/seu_share/home/220242615/OSBP/${FEAT}/bs${BATCH}_lr${LR}"
            mkdir -p "$CHECKPOINT_ROOT"

            echo ""
            echo "======================================"
            echo " Feature: ${FEAT}"
            echo " Batch size: ${BATCH}"
            echo " LR: ${LR}"
            echo " Checkpoint root: ${CHECKPOINT_ROOT}"
            echo "======================================"
            echo ""

            for SOURCE in "${DOMAINS[@]}"; do
                for TARGET in "${DOMAINS[@]}"; do
                    if [ "$SOURCE" != "$TARGET" ]; then

                        EXP_NAME="${FEAT}_${SOURCE}_to_${TARGET}_bs${BATCH}_lr${LR}"
                        SAVE_PATH="${CHECKPOINT_ROOT}/${EXP_NAME}"
                        mkdir -p "$SAVE_PATH"

                        echo ">>> Running experiment: ${EXP_NAME}"

                        python train1.py \
                            --batch-size "$BATCH" \
                            --epochs "$EPOCHS" \
                            --log-interval 100 \
                            --lr "$LR" \
                            --net resnet50 \
                            --save \
                            --save_path "$SAVE_PATH" \
                            --seed 1 \
                            --unit_size "$UNIT_SIZE" \
                            --update_lower \
                            --source "$SOURCE" \
                            --target "$TARGET" \
                            --data_root "$DATA_ROOT" \
                            --source_path "${DATA_ROOT}/${SOURCE}" \
                            --target_path "${DATA_ROOT}/${TARGET}"

                    fi
                done
            done

        done
    done
done
