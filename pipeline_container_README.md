# Container-based training pipeline (experimental)

A parallel pipeline that runs Megatron training inside an NGC PyTorch
container, providing access to newer CUDA / TE / NCCL than the bare-metal
venv (which is pinned to CUDA 12.6 + TE 2.14.0).

**Both pipelines coexist**:
- Bare-metal: `pipeline_env_*.sh` + `pipeline_training_launch.sh` + `pipeline_training_submit.sbatch`
- Container:  `pipeline_container_*.sh` + `pipeline_container_launch.sh` + `pipeline_container_submit.sbatch`

No file in the bare-metal pipeline is modified by the container path.

## Why containers

- Newer CUDA (12.8+) → unblocks blockwise FP8 (paper-recommended Hopper recipe)
- Newer Transformer Engine → potential fix for `cudaErrorCapturedEvent` MoE+graph bug
- Newer Megatron-Core → potential VPP support for SSM/Mamba models
- Eliminates the 9 ARM-specific bare-metal workarounds (NCCL preload, sitecustomize, etc.)

## Storage layout

| Path | Purpose |
|------|---------|
| `/projects/a5k/public/containers/` | Shared `.sif` images (pull once, run many) |
| `/projects/a5k/public/apptainer_cache_kyleobrien/` | Per-user OCI blob cache |

## Key Isambard mechanism

`module load brics/apptainer-multi-node/0.3.2` injects bind-mounts and a
`/host/adapt.sh` script into any container. The script configures
LD_LIBRARY_PATH + NCCL_NET=AWS Libfabric + all FI_CXI_* env vars for
Slingshot/CXI fabric. We don't need to manually manage any of those.

## Usage

```bash
# 1. Pull container (one-time per image)
isambard_sbatch pipeline_container_setup.sbatch <image-name>

# 2. Run a quick smoke test (single node)
isambard_sbatch pipeline_container_smoke.sbatch

# 3. Run a multi-node distributed init test
isambard_sbatch --nodes=2 pipeline_container_distest.sbatch

# 4. Run actual training (when ready)
isambard_sbatch --nodes=16 pipeline_container_submit.sbatch <config.yaml> super cpt
```

## Status: experimental

This pipeline is being prototyped. The bare-metal pipeline remains the
production path until the container path has been validated end-to-end.
