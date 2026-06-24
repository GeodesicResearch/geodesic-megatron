#!/bin/bash
# Verda environment bootstrap — import the NeMo container to a shared .sqsh.
#
# Unlike Isambard (bare-metal uv venv with ARM source builds), Verda runs everything IN the
# NVIDIA NeMo container (Blackwell-built torch/TE/mamba/causal-conv1d/megatron-core/vLLM). The fork
# is used WITHOUT uv sync: pipeline_training_launch_verda.sh sets PYTHONPATH=$REPO/src so the fork's
# megatron.bridge (recipes + patches) shadows the container's bundled copy.
#
# pyxis is broken here (TaskProlog), so we use `enroot create`+`enroot start` directly (see the
# launcher). enroot's cache/data live node-local on /mnt/local_disk (per /etc/enroot/enroot.conf).
#
# Run on a COMPUTE node (login lacks /mnt/local_disk):
#   sbatch -N1 -p gpus --gpus-per-node=1 --wrap 'bash pipeline_env_verda.sh import'
set -e
SQSH="${VERDA_SQSH:-/home/ubuntu/kyle/containers/nemo_26.04.01.sqsh}"
IMG="${VERDA_IMAGE:-nvcr.io/nvidia/nemo:26.04.01}"
case "${1:-import}" in
  import)
    mkdir -p "$(dirname "$SQSH")"
    # nvcr.io/nvidia/nemo:TAG -> enroot URI nvcr.io#nvidia/nemo:TAG (first '/' -> '#')
    URI="docker://${IMG/\//#}"
    echo "Importing $URI -> $SQSH (anonymous pull; if 401, put NGC creds in ~/.config/enroot/.credentials)"
    enroot import -o "$SQSH" "$URI"
    echo "Done. Container ready at $SQSH"
    ;;
  verify)
    enroot create --force --name nemo_verify "$SQSH" 2>&1 | tail -1
    enroot start --rw --mount /home:/home nemo_verify bash -lc \
      'PYTHONPATH='"${VERDA_REPO:-/home/ubuntu/kyle/geodesic-megatron}"'/src python -c "import torch,transformer_engine,megatron.core,megatron.bridge,vllm;print(\"ok\",torch.__version__)"'
    ;;
  *) echo "usage: $0 {import|verify}"; exit 1;;
esac
