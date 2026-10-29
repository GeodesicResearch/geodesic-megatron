# shellcheck shell=bash
# ==============================================================================
# SLURM step helpers of the pipeline launchers. Source it after common.sh.
#
# Catalogue:
#   first_host       the first host of a SLURM node list
#   set_rendezvous   MASTER_ADDR and MASTER_PORT for a torchrun rendezvous
#   node_local_tmp   a job-scoped node-local TMPDIR with its lock and Triton dirs
#   torchrun_payload a container payload that starts this node's torchrun ranks
#   derive_switch_spread the Dragonfly switch groups a node list spans
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

# derive_switch_spread NODELIST: "<switch>:<nodes>,..." for the level-0 switches holding NODELIST's nodes, largest
# first, or nothing (scontrol, so host-side only). Worth recording on every run: the same Nano pretrain config
# measured 137.8 TFLOP/s/GPU across 2 switch groups and 114.7 across 8, so a throughput number is only comparable to
# another taken at the same spread.
derive_switch_spread() {
    # Emits "<switch>:<nodes>,..." largest first, or nothing. Every statement must end
    # truthy: the caller runs under `set -euo pipefail`, so a while-loop whose final
    # iteration ends on a false test makes the whole pipeline nonzero even though the
    # tail of it succeeded, and that would abort the launch rather than skip the metric.
    local mine sw nodes n
    mine=$(scontrol show hostnames "$1" | sort -u)
    [ -n "$mine" ] || return 0
    scontrol show topology | while read -r line; do
        case "$line" in *Level=0*) ;; *) continue ;; esac
        sw=${line#*SwitchName=}; sw=${sw%% *}
        nodes=${line#*Nodes=}; nodes=${nodes%% *}
        if [ -n "$sw" ] && [ -n "$nodes" ]; then
            n=$(scontrol show hostnames "$nodes" | sort -u |
                comm -12 - <(printf '%s\n' "$mine") | wc -l)
            if [ "$n" -gt 0 ]; then printf '%s:%s\n' "$sw" "$n"; fi
        fi
    done | sort -t: -k2 -rn | paste -sd, -
}
