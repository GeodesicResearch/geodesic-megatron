#!/usr/bin/env bash
# Two-phase functional smoke for gradient routing (1 node / 4 GPUs, in-allocation).
#
#   Phase 1: 5-iter from-scratch pretrain of a tiny 6-layer hybrid -> seed checkpoint.
#   Phase 2: 20-iter GR-CPT warm-started from it (missing gr_aux keys, routed dataset,
#            per-iteration gating, DDP bucket completion at gate=0).
#   Post:    check_gr_smoke_result.py asserts the phase-2 checkpoint's aux modules
#            actually trained (fc2 weights non-zero) while remaining absent from phase 1.
#
# Run from the repo root, inside a SLURM allocation, on/for a single node:
#   bash scripts/gradient_routing/run_gr_functional_smoke.sh [nodelist]
# The optional nodelist pins the node (defaults to the launcher's own selection).
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_DIR"
# Inside a tunnel/salloc SLURM_SUBMIT_DIR points at the tunnel's own submit dir; pin the
# launcher's repo resolution to this checkout (worktree-safe).
export GEODESIC_REPO_DIR="$REPO_DIR"

SEED_CFG=configs/gradient_routing/smoke/gr_smoke_seed_pretrain.yaml
GR_CFG=configs/gradient_routing/smoke/gr_smoke_gr_cpt.yaml

# The two configs own the checkpoint paths — this script wipes those directories
# and asserts against them, so it reads them back out rather than keeping a
# second copy that a config edit would silently strand.
command -v python3 >/dev/null 2>&1 || { echo "FATAL: no python3 on PATH" >&2; exit 1; }
SMOKE_PATHS_SH="$(mktemp)"
trap 'rm -f "$SMOKE_PATHS_SH"' EXIT
python3 - "$SEED_CFG" "$GR_CFG" >"$SMOKE_PATHS_SH" <<'PY' || exit 1
import shlex
import sys

import yaml

seed_cfg, gr_cfg = sys.argv[1:]


def checkpoint_block(path):
    with open(path) as fh:
        block = (yaml.safe_load(fh) or {}).get("checkpoint")
    if not block:
        sys.exit(f"FATAL: {path} has no `checkpoint:` block")
    return block


seed = checkpoint_block(seed_cfg)
gr = checkpoint_block(gr_cfg)

seed_save = seed.get("save") or sys.exit(f"FATAL: {seed_cfg} has no `checkpoint.save`")
gr_save = gr.get("save") or sys.exit(f"FATAL: {gr_cfg} has no `checkpoint.save`")
warm_start = gr.get("pretrained_checkpoint")
if warm_start != seed_save:
    sys.exit(
        f"FATAL: {gr_cfg} warm-starts from {warm_start!r}, but phase 1 saves to {seed_save!r}. "
        f"The two smoke configs must be chained or phase 2 tests nothing."
    )

with open(gr_cfg) as fh:
    n_aux = len(yaml.safe_load(fh)["gr"]["aux_data_paths"])

print(f"SEED_SAVE={shlex.quote(seed_save)}")
print(f"GR_SAVE={shlex.quote(gr_save)}")
print(f"N_AUX={n_aux}")
PY
# shellcheck source=/dev/null
source "$SMOKE_PATHS_SH"

NODE_ARGS=(--nodes 1)
if [[ -n "${1:-}" ]]; then
    NODE_ARGS+=(--nodelist "$1")
fi

echo "=== GR smoke phase 1: seed pretrain (5 iters, tiny hybrid) ==="
rm -rf "$SEED_SAVE" "$GR_SAVE"
bash pipeline_training_launch.sh "$SEED_CFG" --model nano --mode pretrain --disable-ft "${NODE_ARGS[@]}"
test -f "$SEED_SAVE/latest_checkpointed_iteration.txt" || {
    echo "FATAL: phase 1 produced no checkpoint at $SEED_SAVE" >&2
    exit 1
}

echo "=== GR smoke phase 2: GR-CPT warm start (20 iters) ==="
bash pipeline_training_launch.sh "$GR_CFG" --model nano --mode cpt --disable-ft "${NODE_ARGS[@]}"
test -f "$GR_SAVE/latest_checkpointed_iteration.txt" || {
    echo "FATAL: phase 2 produced no checkpoint at $GR_SAVE" >&2
    exit 1
}

echo "=== GR smoke post-checks ==="
# N_AUX comes from the config parse above: the smoke GR config itself sets the expected
# module count, so a config edit cannot silently weaken the census assertion.
./pipeline_env_exec.sh "cd $REPO_DIR; source pipeline_env_activate.sh || exit 1; \
    python scripts/gradient_routing/check_gr_smoke_result.py \
        --seed-checkpoint $SEED_SAVE --gr-checkpoint $GR_SAVE --expect-aux-modules $N_AUX"

echo "GR FUNCTIONAL SMOKE PASS"
