#!/bin/bash
# Latest recorded UADAL experiment launcher.

# run_experiments.sh
# Run source-target experiments with feature / batch / lr grid search

# -----------------------
# Fixed parameters
# -----------------------
MODEL="UADAL"
NET="densenet"
SEED=0
EXP_PREFIX="exp"

WARMUP_ITER=2000
TRAINING_ITER=200
UPDATE_TERM=10
SCHEDULER="cos"
E_LR=0.002
G_LR=0.1

# -----------------------
# Search space
# -----------------------
FEATURES=("wav2vec" "wavlm" "hubert")
BATCH_SIZES=(32 64)
LRS=(0.001 0.0001)

DOMAINS=("C" "E" "M" "Me")

# -----------------------
# Main loop
# -----------------------
for FEAT in "${FEATURES[@]}"; do

    DATA_DIR="/seu_nvme/home/220242615/${FEAT}"
    BASE_RESULT_DIR="/seu_share/home/220242615/UADAL/${FEAT}"

    for BATCH_SIZE in "${BATCH_SIZES[@]}"; do
        for LR in "${LRS[@]}"; do

            RESULT_DIR="${BASE_RESULT_DIR}/bs${BATCH_SIZE}_lr${LR}"
            mkdir -p "${RESULT_DIR}"

            echo ""
            echo "======================================"
            echo " Feature: ${FEAT}"
            echo " Batch size: ${BATCH_SIZE}"
            echo " LR: ${LR}"
            echo " Result dir: ${RESULT_DIR}"
            echo "======================================"
            echo ""

            for SRC in "${DOMAINS[@]}"; do
                for TGT in "${DOMAINS[@]}"; do
                    if [ "$SRC" != "$TGT" ]; then

                        EXP_NAME="${EXP_PREFIX}_${FEAT}_${SRC}_to_${TGT}_bs${BATCH_SIZE}_lr${LR}"

                        echo "Running experiment: ${EXP_NAME}"

                        CUDA_VISIBLE_DEVICES=$GPU python train1.py \
                            --data_dir "${DATA_DIR}" \
                            --source_domain "${SRC}" \
                            --target_domain "${TGT}" \
                            --model "${MODEL}" \
                            --net "${NET}" \
                            --batch_size "${BATCH_SIZE}" \
                            --seed "${SEED}" \
                            --exp_code "${EXP_NAME}" \
                            --result_dir "${RESULT_DIR}" \
                            --warmup_iter "${WARMUP_ITER}" \
                            --training_iter "${TRAINING_ITER}" \
                            --update_term "${UPDATE_TERM}" \
                            --scheduler "${SCHEDULER}" \
                            --e_lr "${E_LR}" \
                            --lr "${LR}" \
                            --g_lr "${G_LR}"

                    fi
                done
            done

        done
    done
done
