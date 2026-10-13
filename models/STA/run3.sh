#!/bin/bash
# Latest recorded STA experiment launcher.

# Feature configuration and parameter search
FEATURES=("egemaps")
BATCH_SIZES=(32 64)
LRS=(0.01 0.0001)
FEATURE_DIM=88
DOMAINS=(C M E Me)

# Experiment loop
for FEAT in "${FEATURES[@]}"; do

    BASE_PATH="/seu_nvme/home/220242615/${FEAT}"

    declare -A roots
    roots["C"]="$BASE_PATH/C"
    roots["M"]="$BASE_PATH/M"
    roots["E"]="$BASE_PATH/E"
    roots["Me"]="$BASE_PATH/Me"

    declare -A flists
    flists["C_s"]="$BASE_PATH/map/flist_crema_s_mapped.txt"
    flists["C_t"]="$BASE_PATH/map/flist_crema_t_mapped.txt"
    flists["M_s"]="$BASE_PATH/map/flist_msp_s_mapped.txt"
    flists["M_t"]="$BASE_PATH/map/flist_msp_t_mapped.txt"
    flists["E_s"]="$BASE_PATH/map/flist_esd_s_mapped.txt"
    flists["E_t"]="$BASE_PATH/map/flist_esd_t_mapped.txt"
    flists["Me_s"]="$BASE_PATH/map/flist_meld_s_mapped.txt"
    flists["Me_t"]="$BASE_PATH/map/flist_meld_t_mapped.txt"

    for BATCH in "${BATCH_SIZES[@]}"; do
        for LR in "${LRS[@]}"; do

            echo ""
            echo "======================================"
            echo " Feature: ${FEAT}"
            echo " Batch size: ${BATCH}"
            echo " LR: ${LR}"
            echo "======================================"
            echo ""

            for SRC in "${DOMAINS[@]}"; do
                for TGT in "${DOMAINS[@]}"; do
                    if [ "$SRC" != "$TGT" ]; then

                        echo "=== ${FEAT} | ${SRC} -> ${TGT} | bs=${BATCH} lr=${LR} ==="

                        CKPT_DIR="/seu_share/home/220242615/STA/checkpoint_res/${FEAT}/bs${BATCH}_lr${LR}/${SRC}_to_${TGT}"
                        LOG_DIR="/seu_share/home/220242615/STA/log/resnet/${FEAT}/bs${BATCH}_lr${LR}/${SRC}_to_${TGT}"

                        mkdir -p "$CKPT_DIR"
                        mkdir -p "$LOG_DIR"

                        # -------- Step 1 --------
                        python wavstep_1.py \
                            --batch_size "$BATCH" \
                            --lr "$LR" \
                            --feature_dim "$FEATURE_DIM" \
                            --source "$SRC" \
                            --target "$TGT" \
                            --step1_stage1 500 \
                            --step1_stage2 400 \
                            --source_flist "${flists["${SRC}_s"]}" \
                            --target_flist "${flists["${TGT}_t"]}" \
                            --source_root "${roots[$SRC]}" \
                            --target_root "${roots[$TGT]}" \
                            --save_path "$CKPT_DIR" \
                            --log_dir "$LOG_DIR"

                        # -------- Step 2 --------
                        python wavstep_2.py \
                            --batch_size "$BATCH" \
                            --lr "$LR" \
                            --feature_dim "$FEATURE_DIM" \
                            --step2_stage1 1500 \
                            --step2_stage2 400 \
                            --log_dir "${LOG_DIR}_step2" \
                            --pretrained_disc "$CKPT_DIR/discriminator_a.pkl" \
                            --source_list "${flists["${SRC}_s"]}" \
                            --source_root "${roots[$SRC]}" \
                            --target_list "${flists["${TGT}_t"]}" \
                            --target_root "${roots[$TGT]}"

                    fi
                done
            done

        done
    done
done
