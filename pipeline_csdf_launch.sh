#!/bin/bash
# Run one fit on the already assigned node. The orchestrator owns placement.
set -euo pipefail
if [[ $# != 3 ]]; then
    echo "Usage: $0 FIT_JSON GPUS GEODESIC_UTILS_ROOT" >&2
    exit 2
fi
csdf_repo=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
csdf_fit=$(realpath "$1")
csdf_gpus=$2
csdf_utils=$(realpath "$3")
[[ "$csdf_gpus" =~ ^[1-9][0-9]*$ ]] || exit 2
[[ -f "$csdf_repo/3rdparty/Megatron-LM/megatron/core/__init__.py" ]] || {
    echo "Initialize the pinned Megatron-LM submodule in $csdf_repo before training" >&2
    exit 1
}
printf -v csdf_payload 'cd %q && source pipeline_env_activate.sh && export PYTHONPATH=%q:"$PYTHONPATH" && exec python -m torch.distributed.run --standalone --nproc_per_node=%q pipeline_csdf_training_run.py --fit %q' "$csdf_repo" "$csdf_utils/src" "$csdf_gpus" "$csdf_fit"
exec bash "$csdf_repo/pipeline_env_exec.sh" "$csdf_payload"
