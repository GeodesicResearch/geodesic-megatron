#!/usr/bin/env bash
# Wait for all 6 post-train PIDs to exit, then launch continuation driver.
set -uo pipefail
PIDS=(276979 276983 281613 281617 281621 281625)
echo "Waiting for K.4 post-train PIDs: ${PIDS[*]}"
for pid in "${PIDS[@]}"; do
    while kill -0 "$pid" 2>/dev/null; do
        sleep 60
    done
    echo "  PID $pid exited at $(date -u +%H:%M:%S)"
done
echo "All 6 post-train tasks done at $(date -u). Launching continuation driver."

cd /home/a5k/kyleobrien.a5k/geodesic-megatron
nohup bash configs/inoculation_midtraining/im_fyn1668_v2/run_continuation_inline.sh "$$" auto \
    > logs/v2_inline/continuation_v3_$(date -u +%Y%m%dT%H%M%S).log 2>&1 &
echo "continuation driver PID=$!"
disown
