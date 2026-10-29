# shellcheck shell=bash
# ==============================================================================
# SLURM step helpers of the pipeline launchers. Source it after common.sh.
#
# Catalogue:
#   first_host       the first host of a SLURM node list
#   set_rendezvous   MASTER_ADDR and MASTER_PORT for a torchrun rendezvous
#   node_local_tmp   a job-scoped node-local TMPDIR with its lock and Triton dirs
#   torchrun_payload a container payload that starts this node's torchrun ranks
# ==============================================================================

# first_host NODELIST (scontrol, so host-side only)
first_host() {
    scontrol show hostname "$1" | head -1
}

# set_rendezvous NODELIST: MASTER_ADDR is MASTER_ADDR_OVERRIDE or NODELIST's first
# host, which must be a node that runs ranks; MASTER_PORT is MASTER_PORT_OVERRIDE
# or 29500 + SLURM_JOB_ID % 1000. Sets both in the caller's shell, unexported.
set_rendezvous() {
    MASTER_ADDR="${MASTER_ADDR_OVERRIDE:-$(first_host "$1")}"
    MASTER_PORT="${MASTER_PORT_OVERRIDE:-$((29500 + SLURM_JOB_ID % 1000))}"
}

# node_local_tmp PREFIX: exports TMPDIR=/tmp/PREFIX_<job id> and creates it, with
# MEGATRON_CONFIG_LOCK_DIR in it and TRITON_CACHE_DIR under it. HOME is
# read-only on compute nodes, and a shared lock or cache directory there fails
# with "[Errno 116] Stale file handle" under multi-rank contention.
node_local_tmp() {
    export TMPDIR="/tmp/${1}_${SLURM_JOB_ID}"
    mkdir -p "$TMPDIR"
    export MEGATRON_CONFIG_LOCK_DIR="$TMPDIR"
    export TRITON_CACHE_DIR="$TMPDIR/triton"
    mkdir -p "$TRITON_CACHE_DIR"
}

# torchrun_payload NPROC NNODES COMMAND_LINE: a container payload
# (container_payload) that runs torchrun with NPROC ranks on this step's node
# (\$SLURM_NODEID is its node rank), NNODES nodes in all, rendezvous at
# MASTER_ADDR:MASTER_PORT, and passes COMMAND_LINE (the script and its
# arguments, as shell words) to every rank.
torchrun_payload() {
    container_payload "torchrun --nproc_per_node=$1 --nnodes=$2 --node_rank=\$SLURM_NODEID --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT $3"
}
