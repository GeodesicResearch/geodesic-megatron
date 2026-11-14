# shellcheck shell=bash
# ==============================================================================
# Isambard: GH200 aarch64 nodes of 4 GPUs, HPE Slingshot-11, Lustre under /projects.
# The profile pipeline/sh/cluster.sh loads on its hosts. docs/isambard/cluster.md
# describes the cluster.
# ==============================================================================

# Compute nodes are nid<digits> and login nodes login<digits>, and SLURM sets ISAMBARD_HOST in every task.
isambard_detect() {
    [ -n "${ISAMBARD_HOST:-}" ] || [[ "$(hostname)" =~ ^(nid|login)[0-9]+$ ]]
}

isambard_profile() {
    CLUSTER_GPUS_PER_NODE=4
    # The dense BF16 tensor-core peak of a GH200 in TFLOP/s, without sparsity (the H100 SXM figure). MFU is a
    # fraction of it.
    CLUSTER_GPU_PEAK_TFLOPS=989.4
    # Hopper. The value also guards sm_90a arch-string parsing in JIT builds.
    CLUSTER_CUDA_ARCH_LIST="9.0"
    CLUSTER_FABRIC=slingshot

    # The NGC NeMo tag this cluster is qualified against. docs/environment.md has the qualification gates, and
    # docs/investigations/120b-gbs64-host-overhead-investigation.md (section 9.8) the record for this tag.
    # Change it here to move the cluster to a new image, then re-run `bash pipeline_env_setup.sh --force`.
    CLUSTER_IMAGE_TAG=26.04

    # Everything large is on shared project storage, because $HOME has a small quota. /projects/a5k/public is a
    # symlink into /lus, so a container needs both bound.
    CLUSTER_STORAGE_BINDS=/projects,/lus
    CLUSTER_CONTAINER_ROOT=/projects/a5k/public/containers
    # The image is read where it is stored.
    CLUSTER_CONTAINER_STAGE_ROOT=""
    # Apptainer's defaults are under $HOME, which a ~25 GB image overflows at once.
    CLUSTER_APPTAINER_CACHEDIR="/projects/a5k/public/apptainer_cache_${USER}"
    CLUSTER_APPTAINER_TMPDIR="/projects/a5k/public/tmp/apptainer_${USER}"

    # The compute nodes' OS image (SLES 15 SP7) holds libfabric 2.3.1 alone. The Slingshot build's aws-ofi-nccl
    # is compiled against 1.22.0 and loads 2.3.1 through the libfabric.so.1 soname, which the two share.
    # `pipeline_env_submit.sbatch smoke` checks that the pair reaches Slingshot's bandwidth.
    CLUSTER_HOST_LIBFABRIC=/opt/cray/libfabric/2.3.1

    CLUSTER_NEMO_HOME=/projects/a5k/public/data/nemo_cache
    CLUSTER_HF_HOME=/projects/a5k/public/hf
    # One directory per account. `datasets` creates this tree with mode 0755 under the account that fills it
    # first, so in a shared directory no other account can take the dataset lock.
    CLUSTER_HF_DATASETS_CACHE="/projects/a5k/public/hf/datasets_container_${USER}"
    CLUSTER_WANDB_DIR=/projects/a5k/public/logs/wandb
    CLUSTER_ACTIVATE_TMPDIR=/projects/a5k/public/tmp
    # Prepared datasets, imported base models, torch-profiler traces and the hard-link clones an export reads.
    CLUSTER_DATA_ROOT=/projects/a5k/public/data
    CLUSTER_MODEL_ROOT=/projects/a5k/public/checkpoints/megatron_bridges/models
    CLUSTER_PROFILE_ROOT=/projects/a5k/public/profiles
    CLUSTER_EXPORT_CLONE_ROOT="/projects/a5k/public/tmp/export_clones_${USER}"

    # A job's open-file limit and file-creation mask are the site's.
    CLUSTER_NOFILE=""
    CLUSTER_UMASK=""
    # A compute host has git, which the checkout guard runs.
    CLUSTER_HOST_HAS_GIT=1
    # The site's submit wrapper. It checks the account's node cap and prints the storage report.
    CLUSTER_SUBMIT_WRAPPER=isambard_sbatch
    CLUSTER_LOG_ROOT=/projects/a5k/public/logs
    # Empty: the floor of nccl_allreduce_smoke.py itself (100 GB/s) is calibrated for two of this cluster's nodes.
    CLUSTER_SMOKE_MIN_BUSBW_GB_S=""
}

# The account's node cap. isambard_sbatch enforces it at submission, and a job submitted around the wrapper is
# caught here at its start.
isambard_job_guard() {
    if ! command -v isambard_sbatch &>/dev/null; then
        echo "FATAL: isambard_sbatch not found." >&2
        return 1
    fi
    isambard_sbatch --check
}

# The Slingshot plugin comes from the Option B build, which the shim binds at /opt/slingshot and
# isambard_activate selects inside the container. A launch therefore needs no host module, and the purge keeps the
# modules' LD_LIBRARY_PATH entries out of the environment the shim scrubs.
isambard_host_reset() {
    module purge
}

# The id of the Lustre project DIR counts against.
isambard_project_id() {
    lfs project -d "$1" | awk '{print $1}'
}

# The Lustre project quota DIR counts against: the line `lfs project -d` prints, then the report of `lfs quota -p`
# (scripts/data/tokenize_partitions.py reads it).
isambard_storage_quota_report() {
    local project
    project=$(isambard_project_id "$1")
    lfs project -d "$1" && lfs quota -p "$project" "$1"
}

# Prints the project quota before an image pull. A pull needs about twice the image's size for a while (the OCI
# layers in the cache, then the SIF). The report blocks nothing: the person who starts the pull reads it and
# decides, as with the storage report of isambard_sbatch.
isambard_pull_preflight() {
    local quota_path=/projects/a5k proj_id=""
    if command -v lfs >/dev/null; then
        proj_id="$(isambard_project_id "$quota_path" 2>/dev/null)" || proj_id=""
        if [ -n "$proj_id" ]; then
            echo "[env-setup/sif] Project quota for $quota_path (need ~50 GB transient for a ~25 GB image):"
            lfs quota -p "$proj_id" "$quota_path" || true
        fi
    fi
}

# Slingshot (CXI) networking inside the container, in the layout of the official Option B recipe, and the driver
# workaround.
#
# pipeline_env_setup.sh builds two things inside the image:
#   /opt/slingshot/nccl          NCCL, built against the image's CUDA
#   /opt/slingshot/aws-ofi-nccl  the network plugin that can use CXI
# The shim binds the host's libfabric at /host/opt/cray/libfabric/<ver> and its /usr/lib64 (libcxi, libnl) at
# /host/usr/lib64.
#
# LD_LIBRARY_PATH order, as in the BriCS pytorch_multinode.def %environment:
#   libfabric first               the plugin must resolve the Cray libfabric, which has the CXI provider
#   the built NCCL next           a newer NCCL than the one torch bundles
#   the image's own library path  the linker note below says why it precedes the host's
#   /opt/slingshot/hostlibs last  it fills only the sonames the image lacks
#
# NCCL_NET_PLUGIN names the CXI plugin. NGC images also hold an aws-ofi-nccl built for AWS's EFA fabric, which must
# never be selected here. Without the CXI plugin NCCL falls back to TCP with no error, at about 2.3 GB/s against
# about 163 GB/s.
#
# Linker note: never put /host/usr/lib64 on LD_LIBRARY_PATH. torch inductor's C++ codegen turns each
# LD_LIBRARY_PATH entry into a -L link directory. The host directory's libc.so is a SUSE GNU-ld script that names
# /lib64/libc.so.6 and /usr/lib64/libc_nonshared.a by absolute path, and the image has neither, so a fresh
# torch.compile link dies with "ld: cannot find /lib64/libc.so.6". Putting the image's directories first does not
# prevent it: the image has no plain libc.so development script, so -lc resolves to the host directory. A warm
# inductor cache hides the fault, because a cache hit skips the link. The setup step therefore creates
# /opt/slingshot/hostlibs, a directory of symlinks to the few host libraries the plugin needs (libcxi, libnl).
# dlopen finds them there at run time, and the linker finds no libc there.
isambard_activate() {
    # The bound host libfabric. A glob that matches nothing stays literal, and one that matches several versions
    # puts a space into LD_LIBRARY_PATH. Either leaves the CXI plugin without a libfabric, and NCCL then runs on
    # TCP sockets, about 70 times slower, with no error. So both cases stop or warn here.
    local candidates=(/host/opt/cray/libfabric/*/lib64)
    if [ ! -d "${candidates[0]}" ]; then
        echo "ERROR [env-activate]: no host libfabric under /host/opt/cray/libfabric/*/lib64." >&2
        echo "  The CXI plugin cannot load and NCCL would silently fall back to TCP (~70x slower)." >&2
        echo "  Check the libfabric bind in pipeline_env_config.env (CONTAINER_HOST_LIBFABRIC=${CONTAINER_HOST_LIBFABRIC:-unset})." >&2
        return 1
    fi
    if [ "${#candidates[@]}" -gt 1 ]; then
        echo "WARNING [env-activate]: multiple host libfabric versions bound (${candidates[*]});" >&2
        echo "  using ${candidates[0]}. Pin CONTAINER_HOST_LIBFABRIC to one version." >&2
    fi
    export LD_LIBRARY_PATH="${candidates[0]}:/opt/slingshot/nccl/lib:/opt/slingshot/aws-ofi-nccl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}:/opt/slingshot/hostlibs"
    export NCCL_NET_PLUGIN="${TRAIN_NCCL_NET_PLUGIN:-/opt/slingshot/aws-ofi-nccl/lib/libnccl-net.so}"
    # The driver lacks CUDA Multicast, and UB initialisation hangs without this setting.
    export UB_SKIPMC=1
}
