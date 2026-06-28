#!/bin/bash
# Verda HF<->Megatron conversion — single node (8x B300) inside the NeMo container.
# Usage (sbatch -N1 --gpus-per-node=8):
#   import:  sbatch pipeline_checkpoint_convert_verda.sh import --hf-model <hf-dir-or-id> \
#              --megatron-path <out> --tp 1 --pp 1 --ep 8 --etp 1 --trust-remote-code
#   export:  sbatch pipeline_checkpoint_convert_verda.sh export --megatron-path <dir> --hf-path <out> ...
#
#SBATCH --partition=gpus
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --time=04:00:00
#SBATCH --output=/home/ubuntu/kyle/geodesic-megatron/logs/slurm/verda-convert-%j.out
set -x
REPO="${VERDA_REPO:-/home/ubuntu/kyle/geodesic-megatron}"
SQSH="${VERDA_SQSH:-/home/ubuntu/kyle/containers/nemo_26.04.01.sqsh}"
ARGS="$*"
enroot create --force --name nemo "$SQSH" 2>&1 | tail -1
export NVIDIA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-all}
enroot start --rw --mount /home:/home nemo bash -c "
  set -e; cd '$REPO'
  export PYTHONPATH='$REPO/src' HF_HOME=/home/ubuntu/kyle/hf
  export HF_TOKEN=\$(cat /home/ubuntu/.cache/huggingface/token 2>/dev/null)
  export GLOO_SOCKET_IFNAME=eth0 NCCL_SOCKET_IFNAME=eth0
  export TMPDIR=/mnt/local_disk/cv_\${SLURM_JOB_ID} TRITON_CACHE_DIR=/mnt/local_disk/cvtr_\${SLURM_JOB_ID}
  mkdir -p \$TMPDIR \$TRITON_CACHE_DIR
  torchrun --standalone --nproc_per_node=8 examples/conversion/convert_checkpoints_multi_gpu.py $ARGS
"
echo "VERDA_CONVERT_DONE rc=$?"
