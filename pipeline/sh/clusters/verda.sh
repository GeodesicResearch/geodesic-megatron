# shellcheck shell=bash
# ==============================================================================
# Verda: x86_64 nodes of 8 B300 GPUs, InfiniBand, SLURM on Kubernetes, one shared
# filesystem at /shared and /home. The profile pipeline/sh/cluster.sh loads on its
# hosts. docs/verda/cluster.md describes the cluster.
# ==============================================================================

# The login host is a pod named slurm-login-slinky-<hash> and the compute nodes are slinky-<n>. The site sets
# VERDA_CLUSTER_NAME in every shell.
verda_detect() {
    [ -n "${VERDA_CLUSTER_NAME:-}" ] || [[ "$(hostname)" =~ ^(slurm-login-slinky-.+|slinky-[0-9]+)$ ]]
}

verda_profile() {
    CLUSTER_GPUS_PER_NODE=8
    # B300: dense BF16 tensor-core peak, TFLOP/s per GPU. NVIDIA's HGX B300 table gives 36 PFLOPS for 8 GPUs with
    # sparsity, which is twice the dense figure. docs/verda/cluster.md has the measured GEMM rate beside it.
    CLUSTER_GPU_PEAK_TFLOPS=2250
    # B300 is sm_103, the architecture torch derives from the device when the variable is unset.
    CLUSTER_CUDA_ARCH_LIST="10.3"
    CLUSTER_FABRIC=infiniband

    # The tag Isambard is qualified against, so a run here compares with a run there image for image.
    CLUSTER_IMAGE_TAG=26.04

    # /shared and /home are one filesystem, visible on every node. /local is the node's own NVMe disk, which
    # outlives a job.
    CLUSTER_STORAGE_BINDS=/shared,/home,/local
    CLUSTER_CONTAINER_ROOT=/shared/geodesic/containers
    # A compute pod cannot mount a SIF in the kernel, so Apptainer reads it through squashfuse, which makes a
    # container's Python imports take minutes. Each node runs an unpacked copy on its own disk instead.
    CLUSTER_CONTAINER_STAGE_ROOT="/local/geodesic_${USER}/containers"
    # /tmp is the node's NVMe, private to a job.
    CLUSTER_APPTAINER_CACHEDIR="/tmp/apptainer_cache_${USER}"
    CLUSTER_APPTAINER_TMPDIR="/tmp/apptainer_tmp_${USER}"
    # /shared answers a small read or a page fault slowly (docs/verda/storage.md). A job copies an input it reads
    # that way here first (pipeline/sh/stage.sh). The directory is in the job's own /tmp, so the copy goes with
    # the job. The image's copy above stays for the next job.
    CLUSTER_INPUT_STAGE_ROOT=/tmp/input_stage

    # mlx5_0 is the node's Ethernet port. The InfiniBand HCAs are mlx5_1 to mlx5_8, one beside each GPU.
    CLUSTER_NCCL_IB_HCA="^mlx5_0"
    # The one non-loopback interface of a compute pod.
    CLUSTER_BOOTSTRAP_IFNAME=eth0

    # Datasets, the Hub cache and base models are shared by every account. Checkpoints, logs and W&B files are
    # each account's own.
    CLUSTER_NEMO_HOME=/shared/geodesic/data/nemo_cache
    CLUSTER_HF_HOME=/shared/geodesic/hf
    # Per-user, for the reason the Isambard profile gives.
    CLUSTER_HF_DATASETS_CACHE="/shared/geodesic/hf/datasets_container_${USER}"
    CLUSTER_WANDB_DIR="/shared/geodesic/logs_${USER}/wandb"
    CLUSTER_ACTIVATE_TMPDIR=/tmp
    CLUSTER_DATA_ROOT=/shared/geodesic/data
    CLUSTER_MODEL_ROOT=/shared/geodesic/checkpoints/megatron_bridges/models
    CLUSTER_PROFILE_ROOT="/shared/geodesic/scratch_${USER}/profiles"
    CLUSTER_EXPORT_CLONE_ROOT="/shared/geodesic/scratch_${USER}/export_clones"

    # A login shell's soft limit is 1024 open files, which a job inherits. The hard limit is 1048576.
    CLUSTER_NOFILE=65536
    # The shared trees under /shared/geodesic are group-writable: what one account downloads or builds there,
    # another must be able to extend.
    CLUSTER_UMASK=002
    # A compute host has no git. The image has it.
    CLUSTER_HOST_HAS_GIT=0
    # The site provides sbatch alone.
    CLUSTER_SUBMIT_WRAPPER=""
    CLUSTER_LOG_ROOT="/shared/geodesic/logs_${USER}"
    # The floor of the two-node smoke. Two healthy nodes reach about 700 GB/s over InfiniBand
    # (docs/verda/cluster.md). A TCP fallback, or a node on half its rails, stays far below the floor.
    CLUSTER_SMOKE_MIN_BUSBW_GB_S=400
}

# The site sets no policy that a job can break at submission.
verda_job_guard() {
    :
}

# A compute host has no modules and no toolchain that a launcher can inherit.
verda_host_reset() {
    :
}

verda_pull_preflight() {
    echo "[env-setup/sif] Free space for the image (about 20 GB) under $CLUSTER_CONTAINER_ROOT:"
    df -h "$CLUSTER_CONTAINER_ROOT" || true
}

# The shared filesystem has no quota per project, so the limit is the filesystem's size: what `df -kP` prints
# (scripts/data/tokenize_partitions.py reads it).
verda_storage_quota_report() {
    df -kP "$1"
}

verda_activate() {
    # Loading a model whose code ships with it (trust_remote_code) copies that code into this cache. Under HF_HOME
    # every rank of every run that starts together writes the same files on the shared filesystem, and a load can
    # then fail with "No architectures found in model config". TMPDIR is the job's own.
    export HF_MODULES_CACHE="${TMPDIR:-$CLUSTER_ACTIVATE_TMPDIR}/hf_modules"
}
