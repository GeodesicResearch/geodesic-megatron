#!/usr/bin/env bash
# Wait for the v3 retry small eval PIDs (286982 + 286986) to exit, then launch
# the continuation driver. Used after the previous wait_then_continue.sh fired
# K.5 too early (it was polling stale PIDs while v3 small evals were still
# running).
set -uo pipefail
PIDS=(286982 286986)
echo "Waiting for v3 small eval PIDs: ${PIDS[*]}"
for pid in "${PIDS[@]}"; do
    while kill -0 "$pid" 2>/dev/null; do
        sleep 60
    done
    echo "  PID $pid exited at $(date -u +%H:%M:%S)"
done
echo "Both v3 small evals done at $(date -u). Launching continuation driver."

cd /home/a5k/kyleobrien.a5k/geodesic-megatron
nohup bash configs/inoculation_midtraining/im_fyn1668_v2/run_continuation_inline.sh "$$" auto \
    > logs/v2_inline/continuation_v4_$(date -u +%Y%m%dT%H%M%S).log 2>&1 &
echo "continuation v4 PID=$!"
disown
