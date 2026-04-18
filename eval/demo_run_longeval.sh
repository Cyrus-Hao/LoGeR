#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

usage() {
    cat <<'EOF'
Usage:
  ./eval/demo_run_longeval.sh [--cuda ID] [--model MODEL] [--mode MODE] [--seq SEQ] [--win SIZE]

Options:
  --cuda ID        Set CUDA_VISIBLE_DEVICES (optional)
  --model MODEL    loger | loger_star | all (default: all)
  --mode MODE      kitti | vbr (default: kitti)
  --win SIZE       Window size for demo_viser.py (default: 32)
  --seq SEQ        Sequence id/name.
                   kitti: data/kitti/dataset/sequences/<SEQ>/image_2
                   vbr:   data/vbr/<SEQ>_processed_aligned/rgb
                   If omitted, fallback to KITTI_SEQ/VBR_SEQ env vars.
  -h, --help       Show this help message

Examples:
  ./eval/demo_run_longeval.sh --cuda 0 --model loger --mode kitti --seq 00 --win 32
  ./eval/demo_run_longeval.sh --model loger_star --mode vbr --seq office --win 64
EOF
}

CUDA_DEVICE=""
MODEL_ARG="all"
MODE_ARG="kitti"
SEQ_ARG=""
WIN_ARG="32"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --cuda)
            [[ $# -ge 2 ]] || { echo "Error: --cuda requires a value."; exit 1; }
            CUDA_DEVICE="$2"
            shift 2
            ;;
        --model)
            [[ $# -ge 2 ]] || { echo "Error: --model requires a value."; exit 1; }
            MODEL_ARG="$2"
            shift 2
            ;;
        --mode)
            [[ $# -ge 2 ]] || { echo "Error: --mode requires a value."; exit 1; }
            MODE_ARG="$2"
            shift 2
            ;;
        --seq)
            [[ $# -ge 2 ]] || { echo "Error: --seq requires a value."; exit 1; }
            SEQ_ARG="$2"
            shift 2
            ;;
        --win)
            [[ $# -ge 2 ]] || { echo "Error: --win requires a value."; exit 1; }
            WIN_ARG="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Error: unknown argument '$1'."
            usage
            exit 1
            ;;
    esac
done

if [[ -n "$CUDA_DEVICE" ]]; then
    export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"
fi

if ! [[ "$WIN_ARG" =~ ^[0-9]+$ ]] || [[ "$WIN_ARG" -le 0 ]]; then
    echo "Error: --win must be a positive integer."
    exit 1
fi
window_size="$WIN_ARG"

model_key="${MODEL_ARG,,}"
case "$model_key" in
    loger)
        ckpt_list=("LoGeR")
        ;;
    loger_star|loger-star|logerstar)
        ckpt_list=("LoGeR_star")
        ;;
    all)
        ckpt_list=("LoGeR" "LoGeR_star")
        ;;
    *)
        echo "Error: --model must be one of: loger, loger_star, all."
        exit 1
        ;;
esac

mode_key="${MODE_ARG,,}"
case "$mode_key" in
    kitti)
        seq="${SEQ_ARG:-${KITTI_SEQ:-}}"
        if [[ -z "$seq" ]]; then
            echo "Error: kitti mode requires --seq (or KITTI_SEQ env var)."
            exit 1
        fi
        input_path="/data/shenyou/kitti_od/dataset/sequences/${seq}/image_2"
        end_frame=10000
        ;;
    vbr)
        seq="${SEQ_ARG:-${VBR_SEQ:-}}"
        if [[ -z "$seq" ]]; then
            echo "Error: vbr mode requires --seq (or VBR_SEQ env var)."
            exit 1
        fi
        input_path="$REPO_ROOT/data/vbr/${seq}_processed_aligned/rgb"
        end_frame=20000
        ;;
    *)
        echo "Error: --mode must be one of: kitti, vbr."
        exit 1
        ;;
esac

echo "Mode      : ${mode_key}"
echo "Sequence  : ${seq}"
echo "Input path: ${input_path}"
echo "Window    : ${window_size}"

for ckpt_name in "${ckpt_list[@]}"; do
    echo "--- Processing checkpoint: ${ckpt_name} ---"
    config_path="$REPO_ROOT/ckpts/${ckpt_name}/original_config.yaml"
    model_path="$REPO_ROOT/ckpts/${ckpt_name}/latest.pt"

    if [[ "$mode_key" == "kitti" ]]; then
        output_txt="$REPO_ROOT/results/viser_pi3_kitti/${ckpt_name//\//_}/${seq}.txt"
    else
        output_txt="$REPO_ROOT/results/viser_pi3_vbr/${ckpt_name//\//_}/${seq}_es.txt"
    fi

    mkdir -p "$(dirname "$output_txt")"
    echo "Output txt: ${output_txt}"

    python "$REPO_ROOT/demo_viser.py" \
        --input "$input_path" \
        --config "$config_path" \
        --model_name "$model_path" \
        --window_size "$window_size" \
        --end_frame "$end_frame" \
        --skip_viser \
        --output_txt "$output_txt" \
        --reset_every 5

    echo "--- Finished processing ${ckpt_name} ---"
    echo ""
done

echo "All requested checkpoints have been processed."
