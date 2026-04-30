#!/bin/bash
#SBATCH --job-name=container_buildvenv_2603
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --time=02:00:00
#SBATCH --exclusive
#SBATCH --output=logs/slurm/container_buildvenv_2603-%j.out

# ==============================================================================
# Build a Python venv inside the NGC pytorch:26.03-py3 container.
# Output: /home/a5k/kyleobrien.a5k/geodesic-megatron/.venv-container-2603
# (kept separate from the 25.10 venv at .venv-container so we can fall back).
#
# Includes all fixes learned from the 25.10 build:
#   - MAMBA_FORCE_BUILD=TRUE / CAUSAL_CONV1D_FORCE_BUILD=TRUE → aarch64 wheels
#   - gcc-13 override (Ubuntu 24.04 base, no gcc-12)
#   - megatron-energon installed up-front (transitively required)
#   - nvidia-modelopt 0.44+ (older container modelopt missing get_tensor_shapes
#     _adjust_fn_for_distillation)
#   - Triton REMOVED from venv → falls back to container's matching version
# ==============================================================================

set -eo pipefail
unset LD_PRELOAD
unset LD_LIBRARY_PATH

REPO_DIR=/home/a5k/kyleobrien.a5k/geodesic-megatron
IMAGE=/projects/a5k/public/containers/pytorch_26.03-py3.sif
VENV_DIR=/home/a5k/kyleobrien.a5k/geodesic-megatron/.venv-container-2603
PIP_CACHE=/projects/a5k/public/pip_cache_${USER}
HF_HOME_HOST=/projects/a5k/public/hf

mkdir -p "$PIP_CACHE"

echo "================================"
echo "Building 26.03 container venv"
echo "Image:     $IMAGE"
echo "Venv:      $VENV_DIR"
echo "================================"

if [ -d "$VENV_DIR" ]; then
    echo "Removing prior venv at $VENV_DIR"
    rm -rf "$VENV_DIR"
fi

apptainer exec --nv \
    --bind /home --bind /projects --bind /lus \
    --bind /opt/cray:/opt/cray:ro \
    --bind ${PIP_CACHE}:${PIP_CACHE} \
    "$IMAGE" \
    bash -c "
        set -e
        export LD_LIBRARY_PATH=/usr/local/cuda/compat/lib.real:/usr/local/cuda/compat/lib:\$LD_LIBRARY_PATH
        export PIP_CACHE_DIR=${PIP_CACHE}
        export HF_HOME=${HF_HOME_HOST}

        echo '--- Container env ---'
        python3 --version
        python3 -c 'import torch; print(f\"torch: {torch.__version__}\")'
        python3 -c 'import transformer_engine as te; print(f\"TE: {te.__version__}\")'
        python3 -c 'import triton; print(f\"triton: {triton.__version__}\")'
        echo 'gcc:'; ls /usr/bin/gcc-* | head -3
        echo 'NCCL version:'; python3 -c 'import torch; print(torch.cuda.nccl.version())'

        echo ''
        echo '--- Step 1: create venv with system-site-packages ---'
        python3 -m venv --system-site-packages '$VENV_DIR'
        source '$VENV_DIR/bin/activate'
        which python && python -c 'import torch; print(\"venv torch:\", torch.__version__)'
        pip install --upgrade pip wheel setuptools

        echo ''
        echo '--- Step 2: pure-Python deps ---'
        pip install --no-cache-dir \
            einops omegaconf 'hydra-core>=1.3,<=1.3.2' wandb tensorboard \
            'transformers>=5.0.0,<=5.3.0' 'peft>=0.18.1' 'datasets>=2.20.0' \
            accelerate 'diffusers>=0.36.0' \
            qwen-vl-utils \
            timm 'open-clip-torch>=3.2.0' \
            'mlflow>=3.5.0' 'comet-ml>=3.50.0' \
            imageio imageio-ffmpeg pyyaml tqdm regex six rich typing-extensions \
            megatron-energon

        echo ''
        echo '--- Step 3: nvidia-modelopt (newer than container default) ---'
        pip install --no-cache-dir --upgrade --extra-index-url https://pypi.nvidia.com 'nvidia-modelopt>=0.43.0'

        echo ''
        echo '--- Step 4: source-build deps ---'
        export CC=/usr/bin/gcc-13
        export CXX=/usr/bin/g++-13
        export CUDAHOSTCXX=/usr/bin/g++-13
        export NVCC_PREPEND_FLAGS=\"-ccbin /usr/bin/g++-13\"
        export TORCH_CUDA_ARCH_LIST=9.0
        export MAMBA_FORCE_BUILD=TRUE
        export CAUSAL_CONV1D_FORCE_BUILD=TRUE

        pip install --no-build-isolation --no-cache-dir causal-conv1d
        pip install --no-build-isolation --no-cache-dir mamba-ssm
        pip install --no-build-isolation --no-cache-dir nv-grouped-gemm
        pip install --no-build-isolation --no-cache-dir flash-linear-attention || \
            echo 'flash-linear-attention build failed (non-critical)'

        echo ''
        echo '--- Step 5: editable megatron-core + megatron-bridge ---'
        pip install --no-deps -e ${REPO_DIR}/3rdparty/Megatron-LM
        pip install --no-deps -e ${REPO_DIR}

        echo ''
        echo '--- Step 6: REMOVE venv triton (use container default for ABI match) ---'
        pip uninstall -y triton 2>/dev/null || true
        python -c 'import triton; print(\"triton (post-removal):\", triton.__version__, triton.__file__)'

        echo ''
        echo '--- Smoke test: import critical packages ---'
        python -c '
import torch, transformer_engine as te, megatron.core, megatron.bridge, mamba_ssm, causal_conv1d
from transformer_engine.common.recipe import Float8BlockScaling
print(f\"  torch:              {torch.__version__}\")
print(f\"  TE:                 {te.__version__}\")
print(f\"  CUDA available:     {torch.cuda.is_available()}, devices: {torch.cuda.device_count()}\")
print(f\"  NCCL:               {torch.cuda.nccl.version()}\")
print(f\"  megatron.core:      OK\")
print(f\"  megatron.bridge:    OK\")
print(f\"  mamba_ssm:          OK\")
print(f\"  causal_conv1d:      OK\")
recipe = Float8BlockScaling()
print(f\"  Float8BlockScaling: OK (blockwise FP8 ready)\")
'
"

echo ""
echo "================================"
echo "26.03 venv build COMPLETE: $VENV_DIR"
echo "Size: $(du -sh $VENV_DIR | cut -f1)"
echo "================================"
