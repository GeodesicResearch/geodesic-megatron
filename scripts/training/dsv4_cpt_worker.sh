#!/bin/bash
# Per-node worker of a DeepSeek-V4 continued-pretraining job: starts torchrun for this node in the container.
# Environment from dsv4_cpt.sbatch: CONFIG (the YAML), REPO (the checkout with the pipeline scripts), SCOPE (the Bridge and library
# paths), NNODES, MASTER_ADDR, RUN_SCRIPT (run_recipe.py or a wrapper around it) and RUN_RECIPE_SCRIPT (the file a wrapper runs).
set -euo pipefail
source "$REPO/pipeline/sh/common.sh"
source "$REPO/pipeline/sh/fabric.sh"
fabric_training
"$REPO/pipeline_env_exec.sh" "
cd $REPO; source pipeline_env_activate.sh || exit 1
export DEEPSEEK_V4_PRO_HF_PATH=\$(python -c \"import yaml; print(yaml.safe_load(open('$CONFIG'))['hf_model'])\")
export HF_HOME=/projects/a5k/public/hf HF_HUB_OFFLINE=1 TMPDIR=/tmp TRITON_CACHE_DIR=/tmp/triton_cache_\$SLURM_JOB_ID
export PYTHONPATH=$SCOPE/pylib:$SCOPE/bridge/src:$SCOPE/bridge/3rdparty/Megatron-LM
export CUDA_DEVICE_MAX_CONNECTIONS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NCCL_NVLS_ENABLE=0
cd $SCOPE/bridge
mapfile -t OVERRIDES < <(python scripts/training/dsv4_cpt.py --config $CONFIG --overrides)
python -m torch.distributed.run --nproc_per_node=4 --nnodes=$NNODES --node_rank=\$SLURM_NODEID \
  --master_addr=$MASTER_ADDR --master_port=29533 ${RUN_SCRIPT:-scripts/training/run_recipe.py} \
  --recipe deepseek_v4_pro_pretrain_32gpu_gb300_bf16_config \"\${OVERRIDES[@]}\"
"
