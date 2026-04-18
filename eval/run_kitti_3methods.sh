#!/bin/bash

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KITTI_IMG_ROOT="/data/shenyou/kitti_od/dataset/sequences"
LOG_DIR="$REPO_ROOT/results/kitti_3methods_logs"
RESULT_BASE="$REPO_ROOT/results/viser_pi3_kitti"

SEQS=(00 01 02 03 04 05 06 07 08 09 10)

GPU_LOGER=${GPU_LOGER:-5}
GPU_STAR_SE3=${GPU_STAR_SE3:-6}
GPU_STAR_SIM3=${GPU_STAR_SIM3:-7}

DEBUG_MODE=${DEBUG_MODE:-0}
DEBUG_SEQ=${DEBUG_SEQ:-04}
DEBUG_END_FRAME=${DEBUG_END_FRAME:-50}

PYTHON="${PYTHON:-/home/shenyou/anaconda3/envs/loger/bin/python}"
export PYTHONUNBUFFERED=1

if [[ ! -x "$PYTHON" ]]; then
    echo "ERROR: Python not found at $PYTHON"; exit 1
fi

mkdir -p "$LOG_DIR" "$RESULT_BASE"

timestamp() { date "+%Y-%m-%d %H:%M:%S"; }

run_single_seq() {
    local gpu=$1 ckpt_name=$2 output_subdir=$3 window_size=$4 end_frame=$5 extra_flags=$6 seq=$7
    local config_path="$REPO_ROOT/ckpts/${ckpt_name}/original_config.yaml"
    local model_path="$REPO_ROOT/ckpts/${ckpt_name}/latest.pt"
    local input_path="${KITTI_IMG_ROOT}/${seq}/image_2"
    local output_txt="${RESULT_BASE}/${output_subdir}/${seq}.txt"

    mkdir -p "$(dirname "$output_txt")"

    echo "[$(timestamp)] [GPU $gpu] START ${output_subdir} seq=${seq} win=${window_size}"

    CUDA_VISIBLE_DEVICES=$gpu "$PYTHON" "$REPO_ROOT/demo_viser.py" \
        --input "$input_path" \
        --config "$config_path" \
        --model_name "$model_path" \
        --window_size "$window_size" \
        --end_frame "$end_frame" \
        --skip_viser \
        --output_txt "$output_txt" \
        --reset_every 5 \
        $extra_flags

    echo "[$(timestamp)] [GPU $gpu] DONE  ${output_subdir} seq=${seq} -> $output_txt"
}

run_method() {
    local gpu=$1 ckpt_name=$2 output_subdir=$3 window_size=$4 end_frame=$5 extra_flags=$6
    local log_file="${LOG_DIR}/${output_subdir}.log"
    local seq_list=("${SEQS[@]}")

    if [[ "$DEBUG_MODE" == "1" ]]; then
        seq_list=("$DEBUG_SEQ")
        end_frame=$DEBUG_END_FRAME
    fi

    echo "================================================================" > "$log_file"
    echo "[$(timestamp)] Method: ${output_subdir}" >> "$log_file"
    echo "  GPU: $gpu | Checkpoint: $ckpt_name | Window: $window_size" >> "$log_file"
    echo "  Extra flags: ${extra_flags:-none}" >> "$log_file"
    echo "  Sequences: ${seq_list[*]}" >> "$log_file"
    echo "================================================================" >> "$log_file"

    local completed=0
    local total=${#seq_list[@]}

    for seq in "${seq_list[@]}"; do
        completed=$((completed + 1))
        echo "" >> "$log_file"
        echo ">>> [${completed}/${total}] Processing seq ${seq} ..." >> "$log_file"

        local t_start=$(date +%s)

        if run_single_seq "$gpu" "$ckpt_name" "$output_subdir" "$window_size" "$end_frame" "$extra_flags" "$seq" >> "$log_file" 2>&1; then
            local t_end=$(date +%s)
            local elapsed=$((t_end - t_start))
            echo ">>> [${completed}/${total}] seq ${seq} completed in ${elapsed}s" >> "$log_file"
        else
            local t_end=$(date +%s)
            local elapsed=$((t_end - t_start))
            echo ">>> [${completed}/${total}] seq ${seq} FAILED after ${elapsed}s (exit=$?)" >> "$log_file"
        fi
    done

    echo "" >> "$log_file"
    echo "================================================================" >> "$log_file"
    echo "[$(timestamp)] ALL DONE: ${output_subdir} (${total} sequences)" >> "$log_file"
    echo "================================================================" >> "$log_file"
}

echo ""
echo "============================================================"
echo "  KITTI-od 00-10: LoGeR / LoGeR*(Sim3) / LoGeR*(SE3)"
echo "  GPUs: ${GPU_LOGER}, ${GPU_STAR_SE3}, ${GPU_STAR_SIM3}"
echo "  Debug mode: ${DEBUG_MODE}"
echo "============================================================"
echo ""

END_FRAME=10000

run_method "$GPU_LOGER" "LoGeR" "LoGeR" 32 "$END_FRAME" "" &
PID_LOGER=$!

sleep 30

run_method "$GPU_STAR_SE3" "LoGeR_star" "LoGeR_star_se3" 32 "$END_FRAME" "" &
PID_STAR_SE3=$!

sleep 30

run_method "$GPU_STAR_SIM3" "LoGeR_star" "LoGeR_star_sim3" 32 "$END_FRAME" "--sim3" &
PID_STAR_SIM3=$!

echo "Background PIDs: LoGeR=$PID_LOGER SE3=$PID_STAR_SE3 Sim3=$PID_STAR_SIM3"
echo "$PID_LOGER $PID_STAR_SE3 $PID_STAR_SIM3" > "$LOG_DIR/pids.txt"

wait $PID_LOGER $PID_STAR_SE3 $PID_STAR_SIM3

echo ""
echo "[$(timestamp)] ALL 3 METHODS COMPLETED"
