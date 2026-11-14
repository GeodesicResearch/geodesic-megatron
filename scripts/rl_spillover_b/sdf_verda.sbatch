#!/bin/bash
#SBATCH --job-name=spillB-sdf
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
# LoRA SDF (cpt mode + YAML peft) of Nemotron 3 Nano 30B-A3B on ONE Verda node.
#   sbatch --nodelist=<node> --time=HH:MM:SS sdf_verda.sbatch <data_prefix> <save_dir> <train_iters> [extra overrides]
# Uses the node-local runtime (/local/geodesic-rl-$USER, see ~/verda-rl-setup/stage_local.sh). The HF model and
# its Megatron import live on node-local NVMe (/local/spill-b-$USER) and are reused by later jobs on the node.
set -euo pipefail
export SPILL_SAVE_STEPS=${SPILL_SAVE_STEPS:-"4 8 12 28 52 108 212 428"}   # log-spaced grid (1/16..8 passes at 53.4 iters/pass)
DATA=$1; SAVE=$2; ITERS=$3; shift 3
WT=${SPILL_MEGATRON_WT:-$(realpath $HOME)/spill-b-wt/geodesic-megatron}   # the geodesic-megatron checkout (sbatch runs a spooled copy, so not $0)
[ -f "$WT/pipeline_training_run.py" ] || WT=${SPILL_MEGATRON_WT:?set SPILL_MEGATRON_WT to the geodesic-megatron checkout}
source ${SPILL_ENV:-/shared/geodesic/scratch_puria/spillover/B/jobs/env.sh}
HFM=$L/models/nano; MC=$L/mcore/nano
bash $B/jobs/stage_model.sh nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 $HFM
if [ ! -f $MC/latest_checkpointed_iteration.txt ] && [ ! -d $MC/iter_0000000 ]; then
  echo "=== importing HF -> Megatron at $MC"
  (cd $WT && CUDA_VISIBLE_DEVICES=0 python examples/conversion/convert_checkpoints.py import --hf-model $HFM \
     --megatron-path $MC --torch-dtype bfloat16 --trust-remote-code) 2>&1 | grep -vE "Warning|warn" | tail -20
fi
ls $MC
mkdir -p $SAVE
cd $WT
export PYTHONPATH=$WT:${PYTHONPATH:-}
export WANDB_MODE=disabled NVTE_FUSED_ATTN=0
torchrun --nproc_per_node=8 --master_port=$((29500 + SLURM_JOB_ID % 1000)) scripts/rl_spillover_b/sdf_train.py \
  --model nano --mode cpt --disable-ft --config-file $WT/configs/rl_spillover_b/nano_sdf_lora_r64.yaml \
  "dataset.data_path=[\"1.0\",\"$DATA\"]" \
  checkpoint.pretrained_checkpoint=$MC checkpoint.save=$SAVE train.train_iters=$ITERS \
  tokenizer.tokenizer_model=$HFM "$@" 2>&1 | grep -vE "UserWarning|warnings.warn|FutureWarning"
echo "=== SDF done: $(ls $SAVE | head -50 | tr '\n' ' ')"
