#!/bin/bash
# Run only MISSING sequences for each method. No .pt saving to avoid disk explosion.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KITTI_IMG_ROOT="/data/shenyou/kitti_od/dataset/sequences"
LOG_DIR="$REPO_ROOT/results/kitti_3methods_logs"
RESULT_BASE="$REPO_ROOT/results/viser_pi3_kitti"
PYTHON="${PYTHON:-/home/shenyou/anaconda3/envs/loger/bin/python}"
export PYTHONUNBUFFERED=1

mkdir -p "$LOG_DIR"

timestamp() { date "+%Y-%m-%d %H:%M:%S"; }

run_seq() {
    local gpu=$1 ckpt=$2 subdir=$3 extra=$4 seq=$5
    local output_txt="${RESULT_BASE}/${subdir}/${seq}.txt"

    # skip if already exists with content
    if [[ -f "$output_txt" ]] && [[ $(wc -l < "$output_txt") -gt 1 ]]; then
        echo "[$(timestamp)] [GPU $gpu] SKIP ${subdir} seq=${seq} (already exists)"
        return 0
    fi

    mkdir -p "$(dirname "$output_txt")"
    echo "[$(timestamp)] [GPU $gpu] START ${subdir} seq=${seq}"

    CUDA_VISIBLE_DEVICES=$gpu "$PYTHON" "$REPO_ROOT/demo_viser.py" \
        --input "${KITTI_IMG_ROOT}/${seq}/image_2" \
        --config "$REPO_ROOT/ckpts/${ckpt}/original_config.yaml" \
        --model_name "$REPO_ROOT/ckpts/${ckpt}/latest.pt" \
        --window_size 32 \
        --end_frame 10000 \
        --skip_viser \
        --output_txt "$output_txt" \
        --output_folder "" \
        --reset_every 5 \
        $extra

    local rc=$?
    echo "[$(timestamp)] [GPU $gpu] DONE  ${subdir} seq=${seq} (exit=$rc)"
    return $rc
}

run_method() {
    local gpu=$1 ckpt=$2 subdir=$3 extra=$4
    local logf="${LOG_DIR}/${subdir}_missing.log"

    echo "================================================================" > "$logf"
    echo "[$(timestamp)] ${subdir} on GPU $gpu (missing seqs only)" >> "$logf"
    echo "================================================================" >> "$logf"

    for seq in 00 01 02 03 04 05 06 07 08 09 10; do
        run_seq "$gpu" "$ckpt" "$subdir" "$extra" "$seq" >> "$logf" 2>&1
    done

    echo "[$(timestamp)] ${subdir} ALL DONE" >> "$logf"
}

echo ""
echo "============================================================"
echo "  Filling missing KITTI seqs (no .pt saving)"
echo "  LoGeR→GPU4  SE3→GPU5  Sim3→GPU6"
echo "============================================================"
echo ""

run_method 4 "LoGeR"      "LoGeR"           ""      &
PID1=$!
run_method 5 "LoGeR_star" "LoGeR_star_se3"  ""      &
PID2=$!
run_method 6 "LoGeR_star" "LoGeR_star_sim3" "--sim3" &
PID3=$!

echo "PIDs: LoGeR=$PID1  SE3=$PID2  Sim3=$PID3"
wait $PID1 $PID2 $PID3

echo ""
echo "[$(timestamp)] ALL METHODS COMPLETED"
