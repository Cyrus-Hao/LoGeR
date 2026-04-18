#!/bin/bash
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KITTI_GT="$REPO_ROOT/data/kitti/dataset/poses"
BENCHMARK="$REPO_ROOT/eval/long_eval_script/kitti_benchmark"
RESULT_BASE="$REPO_ROOT/results/viser_pi3_kitti"

METHODS=("LoGeR" "LoGeR_star_se3" "LoGeR_star_sim3")

if [[ ! -d "$KITTI_GT" ]]; then
    echo "GT poses not found at $KITTI_GT"
    echo "Attempting symlink from /data/shenyou/kitti_od/poses ..."
    mkdir -p "$(dirname "$KITTI_GT")"
    ln -sfn /data/shenyou/kitti_od/poses "$KITTI_GT"
fi

if [[ ! -x "$BENCHMARK" ]]; then
    echo "Compiling kitti_benchmark..."
    cd "$REPO_ROOT/eval/long_eval_script"
    g++ -o kitti_benchmark kitti_benchmark.cpp -I /usr/include/eigen3 -O3 -std=c++17
    cd "$REPO_ROOT"
fi

echo "=============================================="
echo "  KITTI Evaluation: 3 Methods"
echo "=============================================="

for method in "${METHODS[@]}"; do
    es_dir="${RESULT_BASE}/${method}"
    if [[ ! -d "$es_dir" ]]; then
        echo "[SKIP] $method - results directory not found: $es_dir"
        continue
    fi

    n_files=$(find "$es_dir" -maxdepth 1 -name "*.txt" ! -name "*results*" ! -name "*.timing.txt" | wc -l)
    echo ""
    echo "----------------------------------------------"
    echo "  $method ($n_files trajectory files)"
    echo "----------------------------------------------"

    echo "[Sim3 alignment]"
    "$BENCHMARK" "$KITTI_GT" "$es_dir" --plot 2>&1 | tail -30
    echo ""
done

echo ""
echo "=============================================="
echo "  Generating comparison plots ..."
echo "=============================================="
PYTHON="${PYTHON:-/home/shenyou/anaconda3/envs/loger/bin/python}"
"$PYTHON" "$REPO_ROOT/eval/plot_kitti_comparison.py" \
    --gt_dir "$KITTI_GT" \
    --result_base "$RESULT_BASE" \
    --methods LoGeR LoGeR_star_se3 LoGeR_star_sim3 \
    --output_dir "$RESULT_BASE/comparison_plots"

echo ""
echo "Done! Results:"
echo "  Per-method metrics: $RESULT_BASE/<method>/results_ate.txt"
echo "  Comparison plots:   $RESULT_BASE/comparison_plots/"
