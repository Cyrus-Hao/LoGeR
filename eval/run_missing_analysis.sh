#!/bin/bash
set -e
cd /home/shenyou/code/combo_jiarui/LoGeR
PYTHON=/home/shenyou/anaconda3/envs/loger/bin/python

for seq in 02 03 05 06 07 08 09 10; do
    outdir="results/viser_pi3_kitti/ate_decomposition_${seq}"
    if [ -f "${outdir}/summary.txt" ]; then
        echo "SKIP seq $seq (summary.txt exists)"
        continue
    fi

    mkdir -p "$outdir"
    gt="data/kitti/dataset/poses/${seq}.txt"
    se3="results/viser_pi3_kitti/LoGeR_star_se3/${seq}.txt"
    sim3="results/viser_pi3_kitti/LoGeR_star_sim3/${seq}.txt"

    echo "=== Seq $seq: ATE decomposition ==="
    if [ -f "$sim3" ] && [ "$(wc -l < "$sim3")" -gt 1 ]; then
        $PYTHON eval/analyze_ate_decomposition.py \
            --gt "$gt" --se3 "$se3" --sim3 "$sim3" \
            --window_size 32 --overlap_size 3 --output_dir "$outdir"
    else
        $PYTHON eval/analyze_ate_decomposition.py \
            --gt "$gt" --se3 "$se3" \
            --window_size 32 --overlap_size 3 --output_dir "$outdir"
    fi

    echo "=== Seq $seq: Chunk stitching ==="
    $PYTHON eval/analyze_chunk_stitching.py \
        --gt "$gt" --se3 "$se3" \
        --window_size 32 --overlap_size 3 --output_dir "$outdir" --top_k 30

    echo "DONE seq $seq"
    echo ""
done

echo "=== ALL ANALYSIS COMPLETE ==="
ls results/viser_pi3_kitti/ate_decomposition_*/summary.txt
