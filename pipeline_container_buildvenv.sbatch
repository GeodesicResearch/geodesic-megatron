#!/bin/bash
#SBATCH --job-name=container_buildvenv
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --time=02:00:00
#SBATCH --exclusive
#SBATCH --output=logs/slurm/container_buildvenv-%j.out

# ==============================================================================
# Build a Python venv INSIDE the NGC pytorch:25.10-py3 container with all
# Megatron-Bridge training deps. The venv lives on shared /projects so it
# persists across jobs and only rebuilds when we explicitly re-run this.
#
# Strategy:
#   1. python -m venv --system-site-packages → inherits container's torch 2.9,
#      transformer_engine 2.8, flash_attn 2.7, nvidia-resiliency-ext 0.4.1.
#   2. pip install pure-Python deps (transformers, peft, hydra, wandb, etc).
#   3. pip install --no-build-isolation source-build deps (mamba-ssm,
#      causal-conv1d, nv-grouped-gemm) — these compile against container's
#      CUDA 13 + torch 2.9 and are the slow part (~30 min total).
#   4. pip install -e ./3rdparty/Megatron-LM (editable megatron-core).
#   5. pip install -e . (editable megatron-bridge from this repo).
#
# After build: training jobs apptainer exec the container, source the venv,
# then run pipeline_training_run.py exactly as bare-metal does.
# ==============================================================================

set -eo pipefail
unset LD_PRELOAD
unset LD_LIBRARY_PATH

REPO_DIR=/home/a5k/kyleobrien.a5k/geodesic-megatron
IMAGE=/projects/a5k/public/containers/pytorch_25.10-py3.sif
VENV_DIR=/home/a5k/kyleobrien.a5k/geodesic-megatron/.venv-container
PIP_CACHE=/projects/a5k/public/pip_cache_${USER}
HF_HOME_HOST=/projects/a5k/public/hf

mkdir -p "$PIP_CACHE"
mkdir -p "$(dirname $VENV_DIR)"

echo "================================"
echo "Building container venv"
echo "Image:     $IMAGE"
echo "Venv:      $VENV_DIR"
echo "Pip cache: $PIP_CACHE"
echo "================================"

# Wipe any prior venv to ensure clean build
if [ -d "$VENV_DIR" ]; then
    echo "Removing prior venv at $VENV_DIR"
    rm -rf "$VENV_DIR"
fi

apptainer exec --nv \
    --bind /projects:/projects \
    --bind ${REPO_DIR}:${REPO_DIR} \
    --bind ${PIP_CACHE}:${PIP_CACHE} \
    "$IMAGE" \
    bash -c "
        set -e
        # CUDA 13 forward-compat shim must come first (driver 565.57 = CUDA 12.7 native)
        export LD_LIBRARY_PATH=/usr/local/cuda/compat/lib.real:/usr/local/cuda/compat/lib:\$LD_LIBRARY_PATH
        export PIP_CACHE_DIR=${PIP_CACHE}
        export HF_HOME=${HF_HOME_HOST}

        echo '--- Container env ---'
        python3 --version
        python3 -c 'import torch; print(f\"torch: {torch.__version__}\")'

        echo ''
        echo '--- Step 1: create venv with system-site-packages ---'
        # Ensure parent exists from inside container view
        mkdir -p \"$(dirname '$VENV_DIR')\"
        ls -la \"$(dirname '$VENV_DIR')\"
        # Use --without-pip if NGC container's ensurepip is missing; install pip after via get-pip
        python3 -m venv --system-site-packages '$VENV_DIR' || {
            echo 'venv with --system-site-packages failed, trying without ensurepip...'
            python3 -m venv --system-site-packages --without-pip '$VENV_DIR'
        }
        source '$VENV_DIR/bin/activate'
        which python && python -c 'import torch; print(\"venv torch:\", torch.__version__)'
        # Bootstrap pip if needed
        if ! command -v pip >/dev/null 2>&1; then
            python -m ensurepip --default-pip 2>/dev/null || \
                curl -sSL https://bootstrap.pypa.io/get-pip.py | python
        fi
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
            imageio imageio-ffmpeg pyyaml tqdm regex six rich typing-extensions

        echo ''
        echo '--- Step 3: source-build deps (mamba-ssm, causal-conv1d, nv-grouped-gemm) ---'
        # Container has gcc-13 (Ubuntu 24.04 base), NOT gcc-12. Torch's cpp_extension
        # hardcodes -ccbin /usr/bin/gcc-12 in CUDA compile flags by default. Override
        # via env vars so nvcc and g++ both find gcc-13.
        export CC=/usr/bin/gcc-13
        export CXX=/usr/bin/g++-13
        export CUDAHOSTCXX=/usr/bin/g++-13
        export NVCC_PREPEND_FLAGS=\"-ccbin /usr/bin/g++-13\"
        # Restrict CUDA arch list to just sm_90 (GH200) — avoids building for
        # arches container's default list includes (sm_75 through sm_121, slow).
        export TORCH_CUDA_ARCH_LIST=9.0
        # Force source builds for mamba-ssm and causal-conv1d. Without this they
        # download precompiled x86_64 wheels and bundle them as aarch64 wheels —
        # selective_scan_cuda.so ends up x86_64 on aarch64 system → ImportError.
        export MAMBA_FORCE_BUILD=TRUE
        export CAUSAL_CONV1D_FORCE_BUILD=TRUE

        # These need --no-build-isolation so they see container's torch
        pip install --no-build-isolation --no-cache-dir causal-conv1d
        pip install --no-build-isolation --no-cache-dir mamba-ssm
        pip install --no-build-isolation --no-cache-dir nv-grouped-gemm
        pip install --no-build-isolation --no-cache-dir flash-linear-attention || echo 'flash-linear-attention build failed (non-critical)'

        echo ''
        echo '--- Step 4: editable megatron-core (from 3rdparty submodule) ---'
        pip install --no-deps -e ${REPO_DIR}/3rdparty/Megatron-LM

        echo ''
        echo '--- Step 5: editable megatron-bridge (this repo) ---'
        pip install --no-deps -e ${REPO_DIR}

        echo ''
        echo '--- Smoke test: import critical packages ---'
        python -c '
import torch
import transformer_engine as te
import megatron.core
import megatron.bridge
import mamba_ssm
import causal_conv1d
print(f\"  torch:              {torch.__version__}\")
print(f\"  transformer_engine: {te.__version__}\")
print(f\"  CUDA available:     {torch.cuda.is_available()}\")
print(f\"  CUDA device count:  {torch.cuda.device_count()}\")
print(f\"  megatron.core:      OK\")
print(f\"  megatron.bridge:    OK\")
print(f\"  mamba_ssm:          OK\")
print(f\"  causal_conv1d:      OK\")
# FP8 blockwise check
from transformer_engine.common.recipe import Float8BlockScaling
recipe = Float8BlockScaling()
print(f\"  Float8BlockScaling: OK (blockwise FP8 ready)\")
'
"

echo ""
echo "================================"
echo "Container venv build COMPLETE: $VENV_DIR"
echo "Size: $(du -sh $VENV_DIR | cut -f1)"
echo "================================"
