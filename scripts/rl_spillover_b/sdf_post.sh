#!/bin/bash
# sdf_post.sh <save_dir> <world> <iters (space-separated)>: on a GPU node, after an SDF run —
#   1. export each grid checkpoint (iter_XXXXXXX) to an HF-PEFT adapter (<save_dir>/hf/iter_N, CPU),
#   2. score the implantation readouts (immediate, all grid adapters + base on one vLLM engine, GPU 0),
#   3. score the post-think rollout readout on the self-preference + third-person subset (GPU 1).
set -uo pipefail
SAVE=$1; W=$2; shift 2; ITERS="$*"
source ${SPILL_ENV:-/shared/geodesic/scratch_puria/spillover/B/jobs/env.sh}
WT=${SPILL_MEGATRON_WT:-$(realpath $HOME)/spill-b-wt/geodesic-megatron}
HFM=$L/models/nano
LORAS=()
for it in $ITERS; do
  d=$(printf "%s/iter_%07d" $SAVE $it); o=$SAVE/hf/iter_$it
  [ -d "$d" ] || { echo "missing $d"; continue; }
  if [ ! -f $o/adapter_model.safetensors ]; then
    (cd $WT && PYTHONPATH=$WT/src:$WT:$PYTHONPATH python examples/conversion/adapter/export_adapter.py --hf-model-path $HFM \
      --lora-checkpoint $d --output $o --trust-remote-code 2>&1 | grep -vE "Warning|warn" | tail -3) &
  fi
  LORAS+=(--lora "iter_$it=$o")
done
wait
ITEMS=$B/data/items/implant_w$W.jsonl
OUT=$SAVE/readouts
CUDA_VISIBLE_DEVICES=0 python $CB/readout.py --model $HFM --items $ITEMS --out $OUT/immediate "${LORAS[@]}" > $OUT.imm.log 2>&1 &
python - <<PY
import json, random
rows=[json.loads(l) for l in open("$ITEMS")]
sub=[r for r in rows if r["kind"] in ("self","third") and r["frame"] in (20,24,27)]
open("$SAVE/rollout_items.jsonl","w").write("".join(json.dumps(r)+"\n" for r in sub))
print(len(sub))
PY
# rollout readouts: base + adapters spread over GPUs 1-7 (one vLLM engine per GPU, each its own subset)
g=1; for i in "${!LORAS[@]}"; do :; done
# rollout readouts only at SPILL_ROLLOUT_ITERS (default: every grid point)
RL_=(); for ((i=0; i<${#LORAS[@]}; i+=2)); do it=${LORAS[$((i+1))]%%=*}; it=${it#iter_}
  if [ -z "${SPILL_ROLLOUT_ITERS:-}" ] || [[ " $SPILL_ROLLOUT_ITERS " == *" $it "* ]]; then RL_+=("${LORAS[$i]}" "${LORAS[$((i+1))]}"); fi; done
NL=$(( ${#RL_[@]} / 2 ))
for k in $(seq 0 $NL); do
  args=(); if [ $k -lt $NL ]; then args=("${RL_[@]:$((2*k)):2}" --no-base); fi
  [ $k -eq $NL ] && args=(--lora "none=/nonexistent")   # base only
  CUDA_VISIBLE_DEVICES=$(( 1 + k % 7 )) python $CB/readout.py --model $HFM --items $SAVE/rollout_items.jsonl \
    --out $OUT/rollout --mode rollout --n 4 "${args[@]}" > $OUT.roll.$k.log 2>&1 &
  [ $(( (k + 1) % 7 )) -eq 0 ] && wait
done
wait
ls $OUT/immediate $OUT/rollout
