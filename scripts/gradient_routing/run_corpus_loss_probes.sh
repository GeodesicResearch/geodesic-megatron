#!/usr/bin/env bash
# Run the corpus loss-probe matrix: score each (checkpoint, corpus) pair eval-only.
#
# Each row of the matrix becomes one run of
# configs/gradient_routing/nemotron_nano_corpus_loss_probe.yaml with the checkpoint and
# data path overridden. Probes run SERIALLY on one node set — they are short (forward
# passes only) and serialising keeps the comparison free of fabric contention, which
# matters because the whole point is a paired measurement.
#
# Usage:
#   scripts/gradient_routing/run_corpus_loss_probes.sh --nodelist <list> [options]
#
#   --matrix   FILE   probe matrix TSV (default: configs/gradient_routing/loss_probe_matrix.tsv)
#   --config   FILE   probe config    (default: configs/gradient_routing/nemotron_nano_corpus_loss_probe.yaml)
#   --outdir   DIR    where per-probe logs and the summary land
#   --nodes    N      node count (default: derived from --nodelist)
#   --nodelist LIST   REQUIRED. Pin explicitly — in a shared allocation an unpinned
#                     launch lands on another team's GPUs.
#   --only     NAME   run just the named row (repeatable)
#
# Results: one log per probe in <outdir>, plus results.tsv with the parsed losses.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MATRIX="$REPO_DIR/configs/gradient_routing/loss_probe_matrix.tsv"
CONFIG="$REPO_DIR/configs/gradient_routing/nemotron_nano_corpus_loss_probe.yaml"
OUTDIR="/projects/a5k/public/logs/gradient_routing_geod171/loss_probes"
NODES=""
NODELIST=""
ONLY=()

while [ $# -gt 0 ]; do
    case "$1" in
        --matrix)   MATRIX="$2"; shift 2 ;;
        --config)   CONFIG="$2"; shift 2 ;;
        --outdir)   OUTDIR="$2"; shift 2 ;;
        --nodes)    NODES="$2"; shift 2 ;;
        --nodelist) NODELIST="$2"; shift 2 ;;
        --only)     ONLY+=("$2"); shift 2 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

if [ -z "$NODELIST" ]; then
    echo "FATAL: --nodelist is required. Pin the probes to your own nodes explicitly;" >&2
    echo "       an unpinned launch in a shared allocation can land on another team's GPUs." >&2
    exit 2
fi
[ -f "$MATRIX" ] || { echo "FATAL: matrix not found: $MATRIX" >&2; exit 2; }
[ -f "$CONFIG" ] || { echo "FATAL: config not found: $CONFIG" >&2; exit 2; }
if [ -z "$NODES" ]; then
    NODES=$(scontrol show hostnames "$NODELIST" | wc -l)
fi

mkdir -p "$OUTDIR"
RESULTS="$OUTDIR/results.tsv"
# Each probe writes its own <name>.result row, and results.tsv is rebuilt from ALL of
# them at the end. Truncating results.tsv per run instead would mean a partial re-run
# (--only) silently deleted every row it did not re-measure.

cd "$REPO_DIR"

wanted() {
    [ ${#ONLY[@]} -eq 0 ] && return 0
    local n
    for n in "${ONLY[@]}"; do [ "$n" = "$1" ] && return 0; done
    return 1
}

echo "[probes] matrix=$MATRIX"
echo "[probes] nodes=$NODES nodelist=$NODELIST"
echo "[probes] outdir=$OUTDIR"

while IFS=$'\t' read -r NAME CKPT PREFIX EXTRAS; do
    case "$NAME" in ''|'#'*|NAME) continue ;; esac
    wanted "$NAME" || continue

    LOG="$OUTDIR/${NAME}.log"
    echo "[probes] === $NAME -> $LOG"

    # Optional 4th column: space-separated extra Hydra overrides for this row — e.g. a
    # GRAM profile probe carries model.gr_aux_ffn_hidden_size=[...] and
    # model.gr_static_gates=[...], a learning-curve probe carries checkpoint.ckpt_step=N.
    EXTRA_ARGS=()
    if [ -n "${EXTRAS:-}" ]; then
        read -r -a EXTRA_ARGS <<< "$EXTRAS"
    fi

    set +e
    bash pipeline_training_launch.sh "$CONFIG" \
        --model nano --mode cpt --disable-ft \
        --nodes "$NODES" --nodelist "$NODELIST" \
        "checkpoint.pretrained_checkpoint=$CKPT" \
        "dataset.data_path=['1.0','$PREFIX']" \
        "logger.wandb_exp_name=loss_probe_${NAME}" \
        ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} > "$LOG" 2>&1 < /dev/null
    RC=$?
    set -e

    # Megatron prints: "validation loss at iteration N on validation set | lm loss value: X | lm loss PPL: Y"
    LOSS=$(grep -oE "lm loss value: *[0-9.E+-]+" "$LOG" | tail -1 | grep -oE "[0-9.E+-]+$" || true)
    PPL=$(grep -oE "lm loss PPL: *[0-9.E+-]+" "$LOG" | tail -1 | grep -oE "[0-9.E+-]+$" || true)

    # A probe whose model asked for weights the checkpoint lacks still prints a
    # plausible loss — dist_ckpt_strictness=log_unexpected only WARNS about those. Treat
    # any such warning as a failed probe rather than trusting the number.
    # Exit status is checked BEFORE the parsed loss. A probe can print its validation
    # loss and then die (an NCCL teardown abort is routine here); trusting the loss
    # because it exists would record that run as a clean measurement.
    if [ "$RC" -ne 0 ]; then
        STATUS="failed(rc=$RC)"
    elif grep -qi "unexpected keys" "$LOG"; then
        STATUS="bad-load(unexpected-keys)"
    elif [ -n "$LOSS" ]; then
        STATUS=ok
    else
        STATUS="no-loss-parsed"
    fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$NAME" "$CKPT" "$PREFIX" "${LOSS:-}" "${PPL:-}" "$STATUS" > "$OUTDIR/${NAME}.result"
    echo "[probes] $NAME: loss=${LOSS:-<none>} ppl=${PPL:-<none>} status=$STATUS"
done < "$MATRIX"

# Rebuild the table from every probe ever run into this outdir, in matrix order, so a
# --only run updates its rows and leaves the rest intact.
{
    printf 'name\tcheckpoint\tdata_prefix\tlm_loss\tppl\tstatus\n'
    while IFS=$'\t' read -r NAME _ _; do
        case "$NAME" in ''|'#'*|NAME) continue ;; esac
        [ -f "$OUTDIR/${NAME}.result" ] && cat "$OUTDIR/${NAME}.result"
    done < "$MATRIX"
} > "$RESULTS"

echo "[probes] done — results: $RESULTS"
cat "$RESULTS"
