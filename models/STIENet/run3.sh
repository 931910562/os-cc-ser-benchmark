#!/usr/bin/env bash
# Latest recorded STIENet experiment launcher.

# Dataset and training configuration
ROOT=/seu_share/home/220242615/STIENet
DATASETS=("Me" "M")
KNOWN=("angry" "happy" "sad")

EPOCHS=200
BATCH=32
LR=1e-3

# Feature configuration
FEATURES=("hubert")

# Loss parameters
ALPHA=1.0
GAMMA=1.0
DELTA=1.0

# Experiment loop
for FEAT in "${FEATURES[@]}"; do

    PROJECT_ROOT=/seu_nvme/home/220242615/${FEAT}
    SAVE_ROOT=${ROOT}/checkpoints/${FEAT}/bs${BATCH}_lr${LR}

    echo ""
    echo "======================================"
    echo " Feature: ${FEAT}"
    echo " Data root: ${PROJECT_ROOT}"
    echo " Save root: ${SAVE_ROOT}"
    echo "======================================"
    echo ""

    for SRC in "${DATASETS[@]}"; do
        for TGT in "${DATASETS[@]}"; do
            if [ "$SRC" != "$TGT" ]; then

                SOURCE_DIR=${PROJECT_ROOT}/${SRC}
                TARGET_DIR=${PROJECT_ROOT}/${TGT}

                for BETA in $(seq 1.0); do
                    for EPS in $(seq 1.5); do

                        EXP_NAME=${SRC}_to_${TGT}_a${ALPHA}_b${BETA}_g${GAMMA}_d${DELTA}_e${EPS}
                        SAVE_DIR=${SAVE_ROOT}/${EXP_NAME}

                        mkdir -p "${SAVE_DIR}"

                        echo ">>> Training: ${FEAT} | ${SRC} -> ${TGT}"
                        echo ">>> a=${ALPHA}, b=${BETA}, g=${GAMMA}, d=${DELTA}, e=${EPS}"

                        python3 train1.py \
                            --source "${SOURCE_DIR}" \
                            --target "${TARGET_DIR}" \
                            --known-classes ${KNOWN[0]} ${KNOWN[1]} ${KNOWN[2]} \
                            --epochs ${EPOCHS} \
                            --batch-size ${BATCH} \
                            --lr ${LR} \
                            --alpha ${ALPHA} \
                            --beta ${BETA} \
                            --gamma ${GAMMA} \
                            --delta ${DELTA} \
                            --eps ${EPS} \
                            --save_root "${SAVE_DIR}" \
                            >"${SAVE_DIR}/log.txt" 2>&1

                        echo ">>> Finished ${FEAT} | ${SRC} -> ${TGT}"
                        echo ">>> Log: ${SAVE_DIR}/log.txt"
                        echo ""
                    done
                done
            fi
        done
    done
done
