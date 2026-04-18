#!/bin/bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/home/shenyou/anaconda3/envs/loger/bin/python}"

for seq in 00 01 02 03 04 05 06 07 08 09 10; do
  sim3_arg=""
  sim3="$REPO_ROOT/results/viser_pi3_kitti/LoGeR_star_sim3/${seq}.txt"
  [ -f "$sim3" ] && [ "$(wc -l < "$sim3")" -gt 1 ] && sim3_arg="--sim3 $sim3"

  echo "=== Seq $seq ==="
  "$PYTHON" "$REPO_ROOT/eval/analyze_ate_decomposition.py" \
    --gt "$REPO_ROOT/data/kitti/dataset/poses/${seq}.txt" \
    --se3 "$REPO_ROOT/results/viser_pi3_kitti/LoGeR_star_se3/${seq}.txt" \
    $sim3_arg \
    --window_size 32 --overlap_size 3 \
    --output_dir "$REPO_ROOT/results/viser_pi3_kitti/ate_decomposition_${seq}" &

  "$PYTHON" "$REPO_ROOT/eval/analyze_chunk_stitching.py" \
    --gt "$REPO_ROOT/data/kitti/dataset/poses/${seq}.txt" \
    --se3 "$REPO_ROOT/results/viser_pi3_kitti/LoGeR_star_se3/${seq}.txt" \
    --window_size 32 --overlap_size 3 \
    --output_dir "$REPO_ROOT/results/viser_pi3_kitti/ate_decomposition_${seq}" \
    --top_k 30 &
done

wait
echo "=== ALL ANALYSIS DONE ==="
