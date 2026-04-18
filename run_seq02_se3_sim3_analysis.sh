#!/usr/bin/env bash
set -euo pipefail

source "/root/miniconda3/etc/profile.d/conda.sh"
conda activate loger

ROOT="/root/autodl-tmp/LoGeR"
CONFIG="${ROOT}/ckpts/LoGeR_star/original_config.yaml"
MODEL="ckpts/LoGeR_star/latest.pt"
GT_DIR="${ROOT}/data/kitti-od/data_odometry_poses/poses"
IMG_DIR="${ROOT}/data/kitti-od/data_odometry_color/sequences"

EXP_ROOT="${ROOT}/loger_star_se3_reset0"
RESULT_ROOT="${EXP_ROOT}/results/viser_pi3_kitti"
SE3_DIR="${RESULT_ROOT}/LoGeR_star_se3_reset0"
SIM3_DIR="${RESULT_ROOT}/LoGeR_star_sim3_reset0"

mkdir -p "${SE3_DIR}" "${SIM3_DIR}"

for seq in 02; do
  echo "=== [${seq}] Running SE3 inference ==="
  python demo_viser.py \
    --input "${IMG_DIR}/${seq}/image_2" \
    --config "${CONFIG}" \
    --model_name "${MODEL}" \
    --window_size 32 \
    --overlap_size 3 \
    --end_frame 4000 \
    --skip_viser \
    --output_txt "${SE3_DIR}/${seq}.txt" \
    --reset_every 0

  echo "=== [${seq}] Running Sim3 inference ==="
  python demo_viser.py \
    --input "${IMG_DIR}/${seq}/image_2" \
    --config "${CONFIG}" \
    --model_name "${MODEL}" \
    --window_size 32 \
    --overlap_size 3 \
    --end_frame 4000 \
    --skip_viser \
    --output_txt "${SIM3_DIR}/${seq}.txt" \
    --reset_every 0 \
    --sim3

  echo "=== [${seq}] Running ATE decomposition ==="
  python eval/analyze_ate_decomposition.py \
    --gt "${GT_DIR}/${seq}.txt" \
    --se3 "${SE3_DIR}/${seq}.txt" \
    --sim3 "${SIM3_DIR}/${seq}.txt" \
    --window_size 32 \
    --overlap_size 3 \
    --output_dir "${RESULT_ROOT}/ate_decomposition_${seq}"

  echo "=== [${seq}] Running chunk stitching analysis ==="
  python eval/analyze_chunk_stitching.py \
    --gt "${GT_DIR}/${seq}.txt" \
    --se3 "${SE3_DIR}/${seq}.txt" \
    --window_size 32 \
    --overlap_size 3 \
    --output_dir "${RESULT_ROOT}/ate_decomposition_${seq}" \
    --top_k 50

  echo "=== [${seq}] All done ==="
done
