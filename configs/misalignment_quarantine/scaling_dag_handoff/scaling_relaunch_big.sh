#!/bin/bash
# Relaunch the 1b + 1.8b scaling sub-DAGs after the save_interval fix.
# 1) cancel the dead/doomed 1b+1.8b jobs (MT base-restarts + all afterok-chained downstream)
# 2) scrub their ledger lines (-> .superseded) so the idempotent DAG submitter re-creates them
# 3) re-run scaling_submit_full_dag.sh (10m/100m untouched: their ledger lines remain -> skipped)
set -uo pipefail
T=/home/a5k/kyleobrien.a5k/.claude/jobs/417df745/tmp
LED=$T/scaling_ledger.txt
SUP=$T/scaling_ledger.superseded.txt

echo "=== building jid set (1b + 1.8b only) ==="
JIDS=$(python3 - "$LED" <<'PY'
import sys,re
jids=set()
for ln in open(sys.argv[1]).read().splitlines():
    if 'combined_scaling_1b' not in ln and 'combined_scaling_1p8b' not in ln: continue
    if 'SCALING-BIGDAG' in ln:
        for m in re.findall(r'MTjids=\[([0-9 ]+)\]', ln): jids.update(int(x) for x in m.split())
        for k in ('MTconv','MTcoh','SFTcoh'):
            mm=re.search(k+r'=([0-9]+)', ln)
            if mm: jids.add(int(mm.group(1)))
    for mm in re.finditer(r'\bcoh=([0-9]+)', ln):
        c=int(mm.group(1)); jids.update({c,c-1,c-2})
    mm=re.search(r'SFTcoh=([0-9]+)', ln)
    if mm: c=int(mm.group(1)); jids.update({c,c-1,c-2})
    for mm in re.finditer(r'\btrain=([0-9]+)', ln): jids.add(int(mm.group(1)))
print(",".join(str(j) for j in sorted(j for j in jids if j>0)))
PY
)
echo "jids: $(echo "$JIDS" | tr ',' '\n' | wc -l) ($JIDS)"
# safety: refuse if the set contains any 100m/10m jid (5221533-5221577 = 100m EM; 5215* = 10m EM)
if echo "$JIDS" | tr ',' '\n' | awk '$1>=5221533 && $1<=5221577' | grep -q .; then
  echo "ABORT: jid set overlaps 100m EM range — refusing to scancel"; exit 1
fi

echo "=== scancel 1b+1.8b sub-DAGs ==="
scancel $(echo "$JIDS" | tr ',' ' ') 2>&1 | head
echo "cancelled (or already terminal)."
sleep 3

echo "=== scrub 1b/1.8b lines from active ledger ==="
grep -E 'combined_scaling_1b|combined_scaling_1p8b' "$LED" >> "$SUP" 2>/dev/null
grep -vE 'combined_scaling_1b|combined_scaling_1p8b' "$LED" > "$LED.tmp" && mv "$LED.tmp" "$LED"
echo "[$(date -u +%FT%TZ)] RELAUNCH-BIG scrubbed 1b/1.8b ($(echo "$JIDS"|tr ',' '\n'|wc -l) jids cancelled); re-submitting with save_interval=200 most_recent_k=2 K=5/8" >> "$LED"
echo "scrubbed; superseded -> $SUP"

echo "=== re-run full DAG submitter (re-creates 1b/1.8b only) ==="
bash "$T/scaling_submit_full_dag.sh" 2>&1 | grep -vE "^\s*(stage|train:|conv:|coh:)|skip " | tail -25
echo "=== done; queue now: $(squeue -u "$USER" -h -t PENDING,RUNNING -o '%i'|wc -l)/232 ==="
