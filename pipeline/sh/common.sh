# shellcheck shell=bash
# ==============================================================================
# Shared helpers of the pipeline launchers and submit wrappers. Source it after
# REPO_DIR names the checkout:
#   source "$REPO_DIR/pipeline/sh/common.sh" || exit 1
# Each launcher first checks for this file with one inline line (the same text in
# each, beginning "[ -f "$REPO_DIR/pipeline/sh/common.sh" ]"), because a REPO_DIR
# that misses it cannot be diagnosed from here.
#
# Catalogue:
#   source_env_config  source pipeline_env_config.env and require a complete
#                      container environment, or exit 1
#   container_payload  a pipeline_env_exec.sh command string that runs a command
#                      in REPO_DIR after pipeline_env_activate.sh
#   quote_args         the arguments as one string of %q-quoted words
#   slurm_flag_arity   the words a SLURM flag that isambard_sbatch may append
#                      to a script's argv takes, or 0 for any other word
# ==============================================================================

# Exits 1 with the fix when REPO_DIR holds no pipeline_env_config.env (REPO_DIR
# mis-resolved) or the container environment is incomplete (env_config_require
# prints which part is missing).
source_env_config() {
    if [ ! -f "$REPO_DIR/pipeline_env_config.env" ]; then
        echo "FATAL: $REPO_DIR/pipeline_env_config.env not found — REPO_DIR mis-resolved." >&2
        echo "  (SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-unset} — inside a tunnel/salloc it points at the" >&2
        echo "  tunnel's own submit dir, not this repo. Export GEODESIC_REPO_DIR=/path/to/repo.)" >&2
        exit 1
    fi
    source "$REPO_DIR/pipeline_env_config.env" || exit 1
    env_config_require || exit 1
}

# container_payload COMMAND: COMMAND runs with the checkout's environment
# (pipeline_env_activate.sh) from REPO_DIR; a failed activation ends it.
container_payload() {
    printf 'cd %s; source pipeline_env_activate.sh || exit 1; %s' "$REPO_DIR" "$1"
}

# quote_args [ARG...]: " 'a b' c" for (a b, c), each word %q-quoted and preceded
# by a space; empty for no arguments (a lone %q of nothing would be '').
quote_args() {
    [ "$#" -eq 0 ] || printf ' %q' "$@"
}

# slurm_flag_arity WORD: 2 for a SLURM flag given as two words (--exclude X), 1
# for one given as one (--exclude=X), 0 otherwise. Older isambard_sbatch copies
# append these to the script's argv, where the script's own parser must skip
# them.
slurm_flag_arity() {
    case "$1" in
        --exclude|--nodelist|--nodes|--reservation|--nice|--priority|--partition|--qos) echo 2 ;;
        --exclude=*|--nodelist=*|--nodes=*|--reservation=*|--nice=*|--priority=*|--partition=*|--qos=*) echo 1 ;;
        *) echo 0 ;;
    esac
}
