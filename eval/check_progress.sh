#!/bin/bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$REPO_ROOT/results/kitti_3methods_logs"
RESULT_BASE="$REPO_ROOT/results/viser_pi3_kitti"

echo "============================================"
echo "  KITTI 3-Methods Experiment Progress"
echo "  $(date)"
echo "============================================"
echo ""

if [[ -f "$LOG_DIR/pids.txt" ]]; then
    read -r P1 P2 P3 < "$LOG_DIR/pids.txt"
    running=0
    for p in $P1 $P2 $P3; do
        kill -0 "$p" 2>/dev/null && running=$((running + 1))
    done
    echo "Status: $running/3 methods still running"
else
    echo "Status: PID file not found"
fi
echo ""

for method in LoGeR LoGeR_star_se3 LoGeR_star_sim3; do
    log="$LOG_DIR/${method}.log"
    dir="$RESULT_BASE/$method"

    done_count=$(grep -c "DONE" "$log" 2>/dev/null || echo 0)
    fail_count=$(grep -c "FAILED" "$log" 2>/dev/null || echo 0)
    last=$(grep -E "DONE|FAILED|START|ALL DONE" "$log" 2>/dev/null | tail -1)
    printf "%-20s done=%d fail=%d | %s\n" "$method" "$done_count" "$fail_count" "$last"
done

echo ""
echo "Output files:"
for method in LoGeR LoGeR_star_se3 LoGeR_star_sim3; do
    dir="$RESULT_BASE/$method"
    files=$(ls "$dir"/*.txt 2>/dev/null | grep -v results | xargs -I{} basename {} .txt | tr '\n' ' ')
    count=$(echo $files | wc -w)
    printf "  %-20s [%2d/11]: %s\n" "$method" "$count" "$files"
done

echo ""
echo "GPU Status:"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv 2>/dev/null | tail -4
