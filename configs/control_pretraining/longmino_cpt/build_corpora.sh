#!/bin/bash
# Build the longmino_cpt arm's nine family corpora: slice -> tokenize, one chain per family.
#
# The raw shards are fetched ONCE on the login node (download_shards.py; the only step that
# needs the Hub). Everything that touches a large file runs in its own 1-node job:
#   slice    slice_family.sbatch <family>   raw shards -> <root>/training.jsonl (+ slice_results.json)
#   tokenize pipeline_data_submit.sbatch    training.jsonl -> tokenized_base_input_document.{bin,idx}
# chained with --dependency=afterok so a failed slice never feeds a half-written JSONL forward.
#
# Usage (from the repo root):
#   PYTHON=<venv python with huggingface_hub> configs/control_pretraining/longmino_cpt/build_corpora.sh download
#   DRY_RUN=1 configs/control_pretraining/longmino_cpt/build_corpora.sh all       # print the jobs
#   ISAMBARD_SBATCH_FORCE=1 configs/control_pretraining/longmino_cpt/build_corpora.sh all
#   ISAMBARD_SBATCH_FORCE=1 configs/control_pretraining/longmino_cpt/build_corpora.sh real_pdfs code
#   BUILD_STEPS=tokenize ... build_corpora.sh <family>                              # re-tokenize only
set -euo pipefail
ARM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MANIFEST="$ARM_DIR/data/longmino_cpt_20b.manifest.json"
DRY_RUN="${DRY_RUN:-0}"
BUILD_STEPS="${BUILD_STEPS:-slice,tokenize}"
[ -f pipeline_data_submit.sbatch ] || { echo "FATAL: run from the geodesic-megatron repo root" >&2; exit 1; }
mkdir -p logs/slurm

# Everything about the arm is read from the manifest (which carries the data yaml's stated
# config); nothing is decided here, and only the standard library is needed on the login node.
jq_() { python3 -c "import sys,json; c=json.load(open('$MANIFEST')); print(eval(sys.argv[1]))" "$1"; }
DATA_BASE="$(jq_ "c['data_base']")"
TOKENIZER="$(jq_ "c['tokenizer']")"
ALL_FAMILIES=($(jq_ "' '.join(c['families'])"))
PYTHON="${PYTHON:-python3}"   # download needs huggingface_hub: point PYTHON at a venv that has it

if [ "${1:-}" = "download" ]; then
    echo "=== downloading manifest shards into $DATA_BASE/_raw (login node) ==="
    exec "$PYTHON" "$ARM_DIR/download_shards.py" --workers 16
fi
[ $# -ge 1 ] || { echo "usage: build_corpora.sh <download|all|family ...>" >&2; exit 1; }
if [ "$1" = "all" ]; then FAMILIES=("${ALL_FAMILIES[@]}"); else FAMILIES=("$@"); fi
for f in "${FAMILIES[@]}"; do
    [[ " ${ALL_FAMILIES[*]} " == *" $f "* ]] || { echo "FATAL: unknown family $f" >&2; exit 1; }
done

stripe_dir() {  # Lustre striping must precede the first write
    local dir="$1" count="$2"
    [ "$count" -gt 0 ] || return 0
    command -v lfs >/dev/null 2>&1 || { echo "  note: lfs(1) unavailable — $dir left at defaults"; return 0; }
    lfs setstripe -c "$count" "$dir" || { echo "FATAL: lfs setstripe failed on $dir" >&2; exit 1; }
}
submit() {  # prints the job id, or a placeholder in DRY_RUN
    if [ "$DRY_RUN" = "1" ]; then echo "  [dry-run] isambard_sbatch $*" >&2; echo "DRY$RANDOM"; return; fi
    local out; out="$(ISAMBARD_SBATCH_FORCE=${ISAMBARD_SBATCH_FORCE:-} isambard_sbatch "$@")" || { echo "$out" >&2; exit 1; }
    echo "$out" | grep -oE '[0-9]+$'
}

n=0
for fam in "${FAMILIES[@]}"; do
    root="$DATA_BASE/$fam"
    slice_h="$(jq_ "c['jobs']['$fam']['slice_h']")"; tok_h="$(jq_ "c['jobs']['$fam']['tok_h']")"
    workers="$(jq_ "c['jobs']['$fam']['workers']")"; stripe="$(jq_ "c['jobs']['$fam']['stripe']")"
    echo "=== $fam -> $root (slice ${slice_h}h, tokenize ${tok_h}h x${workers} workers, stripe $stripe) ==="
    if [ "$DRY_RUN" != "1" ]; then mkdir -p "$root"; stripe_dir "$root" "$stripe"; fi
    dep=""
    if [[ ",$BUILD_STEPS," == *",slice,"* ]]; then
        jid=$(submit --time="${slice_h}:00:00" --job-name="cp-longmino-slice-$fam" \
              --export=ALL,GEODESIC_REPO_DIR="$PWD" \
              "$ARM_DIR/slice_family.sbatch" "$fam")
        echo "  slice    -> job $jid"; dep="--dependency=afterok:$jid"; n=$((n+1))
    fi
    if [[ ",$BUILD_STEPS," == *",tokenize,"* ]]; then
        jid=$(submit --time="${tok_h}:00:00" --job-name="cp-longmino-tok-$fam" $dep \
              --export=ALL,GEODESIC_REPO_DIR="$PWD" \
              pipeline_data_submit.sbatch tokenize "$root" "$TOKENIZER" tokenized_base input "$workers")
        echo "  tokenize -> job $jid ${dep:+(after slice)}"; n=$((n+1))
    fi
done
echo "SUBMITTED $n jobs for ${#FAMILIES[@]} families"
