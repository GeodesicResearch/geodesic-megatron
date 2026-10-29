# shellcheck shell=bash
# ==============================================================================
# The ISAMBARD_ENV_OVERRIDES hook of pipeline_training_launch.sh, which sources
# this file before it reads any of its knobs.
#
# Catalogue:
#   apply_env_overrides   read and validate an overrides file, export its
#                         entries and ISAMBARD_ENV_OVERRIDE_KEYS
#   export_env_overrides  export the entries again and render them as the
#                         payload's statements (ENV_OVERRIDES_PAYLOAD)
# ==============================================================================

# ==============================================================================
# Per-launch environment overrides (ISAMBARD_ENV_OVERRIDES)
#
# ISAMBARD_ENV_OVERRIDES=<file> sets environment variables for one launch without editing
# pipeline_training_launch.sh or pipeline_env_activate.sh, including ones either of them sets
# unconditionally (TORCH_NCCL_BLOCKING_WAIT, OMP_NUM_THREADS, ...). The file holds KEY=VALUE
# lines. Blank lines and lines starting with # are ignored; KEY must match
# ^[A-Za-z_][A-Za-z0-9_]*$; VALUE is everything after the first '=', taken literally (no quote
# removal, no expansion). A relative path resolves against the directory the launcher is
# started from.
#
# The launcher reads the file (apply_env_overrides) before it does anything else, and the
# launch ends there if the variable is set but names no file, or if any line is malformed (a
# carriage return counts: it would end up inside the value), repeats a KEY, names what looks
# like a credential (every value is printed into the job log), or names one of the launcher's
# own shell variables (apply_env_overrides lists them): those would change the launch itself
# -- its model, node count, checkout, container, payload -- rather than the environment it
# runs in. The launcher resolves its checkout (REPO_DIR) before it reads the file, so an
# override of SLURM_SUBMIT_DIR reaches the ranks but does not move the checkout.
#
# Each override is applied at three points, because three layers set the environment:
#   - by apply_env_overrides, before the launcher reads any of its knobs, so the knobs take
#     the override (ISAMBARD_NCCL_DEBUG, TRAIN_PERSISTENT_TRITON_CACHE, TRAIN_FI_CXI_*_OVERRIDE,
#     MASTER_*_OVERRIDE, GEODESIC_CONTAINER_*, ISAMBARD_RUN_ID, ...);
#   - after the launcher's last export (export_env_overrides), so it also beats what the
#     launcher sets unconditionally (TORCH_NCCL_BLOCKING_WAIT=1, the NCCL/CXI set, TMPDIR)
#     for the host side of the launch: the rendezvous endpoint, the ft_launcher gate, srun; and
#   - inside the container payload, after pipeline_env_activate.sh and the payload's own
#     exports, immediately before the rank launcher -- the only point that beats activate's
#     defaults. The payload carries the assignments as quoted bash statements, so no other
#     node reads the file.
# The banner lists each applied KEY=VALUE, and the ranks get ISAMBARD_ENV_OVERRIDE_KEYS (the
# keys, comma-separated, in file order), which pipeline_training_run.py uses to log the value
# each node's processes actually see. With ISAMBARD_ENV_OVERRIDES unset the launch is
# unchanged, except that an inherited ISAMBARD_ENV_OVERRIDE_KEYS is dropped: it would make the
# ranks report overrides this launch never applied.
# ==============================================================================
apply_env_overrides() {
    # Locals carry the _eo_ prefix, which a KEY may not use, so no KEY can shadow one of them.
    local _eo_file="$1" _eo_line _eo_key _eo_other _eo_decl _eo_n=0
    local -a _eo_keys=()
    # The launcher's own shell variables, pipeline_env_config.env's CONTAINER_* (which it derives
    # from the GEODESIC_CONTAINER_* inputs, on the host and again on every node), the variables
    # that choose the checkout, waive its check or carry the code-identity or launch-width check's record, and this
    # hook's own. A variable that already exists here without
    # the export attribute (bash's own, such as IFS) is refused as well.
    # tests/unit_tests/test_launcher_env_overrides.py fails when a variable the launcher leaves in
    # its shell is refused by neither rule.
    local -a _eo_reserved=(
        CONFIG_FILE MODEL MODE USE_FT USE_STRAGGLER ENABLE_PAO PEFT OVERRIDE_NODES OVERRIDE_NODELIST
        EXTRA_ARGS USAGE REPO_DIR ENV_CACHE_SUFFIX _FD1_TARGET RUN_ID_LINK_DIR NNODES NODELIST
        TOTAL_GPUS TRAIN_SCRIPT SCRIPT_ARGS SRUN_ARGS RUNNER ACTIVATE_CMD
        GEODESIC_REPO_DIR TRAIN_REPO_DIR ALLOW_CROSS_CHECKOUT_CONFIG ISAMBARD_CODE_IDENTITY ISAMBARD_LAUNCH_WIDTH
        ISAMBARD_ENV_OVERRIDES ISAMBARD_ENV_OVERRIDE_KEYS ENV_OVERRIDE_ENTRIES ENV_OVERRIDES_PAYLOAD
    )
    if [ ! -f "$_eo_file" ]; then
        echo "FATAL: ISAMBARD_ENV_OVERRIDES names no file: '$_eo_file'" >&2
        exit 1
    fi
    ENV_OVERRIDE_ENTRIES=()
    while IFS= read -r _eo_line || [ -n "$_eo_line" ]; do
        _eo_n=$((_eo_n + 1))
        if [[ "$_eo_line" =~ ^[[:space:]]*$ || "$_eo_line" == "#"* ]]; then
            continue
        fi
        _eo_key="${_eo_line%%=*}"
        if [[ "$_eo_line" != *=* || ! "$_eo_key" =~ ^[A-Za-z_][A-Za-z0-9_]*$ || "$_eo_line" == *$'\r'* ]]; then
            echo "FATAL: $_eo_file:$_eo_n: not KEY=VALUE (KEY matching ^[A-Za-z_][A-Za-z0-9_]*\$, no carriage return): $_eo_line" >&2
            exit 1
        fi
        if [[ "${_eo_key^^}" =~ (^|_)(TOKEN|SECRET|PASSWORD|API_KEY)(_|$) ]]; then
            echo "FATAL: $_eo_file:$_eo_n: $_eo_key looks like a credential; overrides are printed into the job log" >&2
            exit 1
        fi
        for _eo_other in "${_eo_reserved[@]}"; do
            if [ "$_eo_other" = "$_eo_key" ]; then
                echo "FATAL: $_eo_file:$_eo_n: $_eo_key is set by the launcher itself, not an environment override" >&2
                exit 1
            fi
        done
        if [[ "$_eo_key" == CONTAINER_* ]]; then
            echo "FATAL: $_eo_file:$_eo_n: $_eo_key is derived by pipeline_env_config.env; override its GEODESIC_CONTAINER_* input" >&2
            exit 1
        fi
        if [[ "$_eo_key" == _eo_* ]]; then
            echo "FATAL: $_eo_file:$_eo_n: $_eo_key uses the _eo_ prefix, which the override hook keeps for itself" >&2
            exit 1
        fi
        if _eo_decl="$(declare -p "$_eo_key" 2>/dev/null)"; then
            _eo_decl="${_eo_decl#declare -}"
            if [[ "${_eo_decl%% *}" != *x* ]]; then
                echo "FATAL: $_eo_file:$_eo_n: $_eo_key is a shell variable of the launcher, not an environment variable" >&2
                exit 1
            fi
        fi
        for _eo_other in "${_eo_keys[@]}"; do
            if [ "$_eo_other" = "$_eo_key" ]; then
                echo "FATAL: $_eo_file:$_eo_n: $_eo_key is set more than once" >&2
                exit 1
            fi
        done
        _eo_keys+=("$_eo_key")
        ENV_OVERRIDE_ENTRIES+=("$_eo_line")
    done < "$_eo_file"

    for _eo_line in "${ENV_OVERRIDE_ENTRIES[@]}"; do
        export "$_eo_line"
    done
    ISAMBARD_ENV_OVERRIDE_KEYS="$(IFS=,; echo "${_eo_keys[*]}")"
    export ISAMBARD_ENV_OVERRIDE_KEYS
}

# Re-applies the overrides read by apply_env_overrides after the launcher's own exports, and
# renders them as the bash statements the container payload runs after activation.
export_env_overrides() {
    local _eo_line
    ENV_OVERRIDES_PAYLOAD=""
    for _eo_line in "${ENV_OVERRIDE_ENTRIES[@]}"; do
        export "$_eo_line"
        ENV_OVERRIDES_PAYLOAD+="export ${_eo_line%%=*}=$(printf '%q' "${_eo_line#*=}")"$'\n'
    done
    ENV_OVERRIDES_PAYLOAD+="export ISAMBARD_ENV_OVERRIDE_KEYS=$(printf '%q' "$ISAMBARD_ENV_OVERRIDE_KEYS")"
}
