# shellcheck shell=bash
# ==============================================================================
# Cluster selection of the pipeline launchers. pipeline_env_config.env sources it
# on the host, and pipeline_env_activate.sh inside the container.
#
# A cluster's profile is pipeline/sh/clusters/<name>.sh. It defines:
#   <name>_detect          succeeds on that cluster's hosts
#   <name>_profile         sets the CLUSTER_* values of that cluster
#   <name>_job_guard       succeeds in a job the cluster's submit path shaped
#   <name>_host_reset      clears host state a launcher must not inherit
#   <name>_pull_preflight  reports the storage an image pull is about to use
#   <name>_storage_quota_report DIR  prints the usage and limit of the storage DIR is on
#   <name>_activate        the in-container environment only that cluster needs
# The CLUSTER_* values stay unexported, so they never reach a rank. GEODESIC_CLUSTER
# is exported, so the container resolves the cluster its launcher resolved.
#
# Catalogue:
#   cluster_names           the profiles this checkout holds
#   cluster_resolve         the cluster this shell runs on
#   cluster_load            resolve the cluster and set its CLUSTER_* values
#   cluster_submit_command  the command that submits a job on the loaded cluster
#   cluster_job_guard       the loaded cluster's <name>_job_guard
#   cluster_host_reset      the loaded cluster's <name>_host_reset
#   cluster_pull_preflight  the loaded cluster's <name>_pull_preflight
#   cluster_storage_quota_report  the loaded cluster's <name>_storage_quota_report
#   cluster_activate        the loaded cluster's <name>_activate
# ==============================================================================

_CLUSTER_PROFILE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/clusters"

# cluster_names: one profile name per line.
cluster_names() {
    local profile
    for profile in "$_CLUSTER_PROFILE_DIR"/*.sh; do
        basename "$profile" .sh
    done
}

# cluster_resolve: prints GEODESIC_CLUSTER when it is set, and otherwise the one profile whose <name>_detect
# succeeds. Returns 1, naming the profiles, when GEODESIC_CLUSTER names no profile, when no profile detects this
# host, and when more than one does. The profiles must be sourced first (cluster_load does it).
cluster_resolve() {
    local name matched="" count=0
    if [ -n "${GEODESIC_CLUSTER:-}" ]; then
        if [ ! -f "$_CLUSTER_PROFILE_DIR/$GEODESIC_CLUSTER.sh" ]; then
            echo "FATAL [cluster]: GEODESIC_CLUSTER=$GEODESIC_CLUSTER names no profile under $_CLUSTER_PROFILE_DIR (profiles: $(cluster_names | paste -sd' ' -))." >&2
            return 1
        fi
        echo "$GEODESIC_CLUSTER"
        return 0
    fi
    for name in $(cluster_names); do
        if "${name}_detect"; then
            matched="$matched $name"
            count=$((count + 1))
        fi
    done
    if [ "$count" -ne 1 ]; then
        echo "FATAL [cluster]: this host ($(hostname)) matches $count cluster profiles (${matched# }); exactly one must match." >&2
        echo "  Export GEODESIC_CLUSTER=<name> to choose one (profiles: $(cluster_names | paste -sd' ' -))." >&2
        return 1
    fi
    echo "${matched# }"
}

# cluster_load: sources every profile, resolves the cluster (cluster_resolve), exports GEODESIC_CLUSTER, sets that
# cluster's CLUSTER_* values in the calling shell and applies its file-creation mask (CLUSTER_UMASK, when the
# profile names one). Returns 1 when the cluster cannot be resolved, and when USER is unset: the profiles name each
# account's own directories by it.
cluster_load() {
    local profile name
    if [ -z "${USER:-}" ]; then
        echo "FATAL [cluster]: USER is unset, and the cluster profiles name each account's directories by it." >&2
        return 1
    fi
    for profile in "$_CLUSTER_PROFILE_DIR"/*.sh; do
        # shellcheck source=/dev/null
        source "$profile" || return 1
    done
    name="$(cluster_resolve)" || return 1
    export GEODESIC_CLUSTER="$name"
    "${name}_profile"
    if [ -n "$CLUSTER_UMASK" ]; then
        umask "$CLUSTER_UMASK"
    fi
}

# cluster_submit_command: the site's submit wrapper where the profile names one, else this checkout's own
# (pipeline_cluster_submit.sh), as a user types it from the repo root.
cluster_submit_command() {
    echo "${CLUSTER_SUBMIT_WRAPPER:-./pipeline_cluster_submit.sh}"
}

cluster_job_guard() {
    "${GEODESIC_CLUSTER}_job_guard"
}

cluster_host_reset() {
    "${GEODESIC_CLUSTER}_host_reset"
}

cluster_pull_preflight() {
    "${GEODESIC_CLUSTER}_pull_preflight"
}

cluster_storage_quota_report() {
    "${GEODESIC_CLUSTER}_storage_quota_report" "$@"
}

cluster_activate() {
    "${GEODESIC_CLUSTER}_activate"
}
