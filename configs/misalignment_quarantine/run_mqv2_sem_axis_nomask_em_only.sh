#!/bin/bash
# =============================================================================
# Queue the 15 EM stages of the sem_*_nomask campaign.
#
# sem_proc_nomask + sem_combined_nomask: SFTs already COMPLETED → EM trains
#   submit with no dependency (start as nodes free up).
# sem_decl_nomask: SFT resubmit still cascading → EM trains depend on the
#   resubmit SFT coh JID supplied via SEM_DECL_SFT_COH_JID env var.
#
# Per EM (3 sbatch + 9 eval sbatch via post-train fan-out):
#   train (16 nodes) → conv (1 node) → post_train_em_full_chain.sbatch
#                                       └─ coh×3 (in-line) + smoke×3 + quick×3 + full×3
#
# Usage:
#   SEM_DECL_SFT_COH_JID=4795488 ISAMBARD_SBATCH_FORCE=1 \
#     bash configs/misalignment_quarantine/run_mqv2_sem_axis_nomask_em_only.sh
#
# Optional env:
#   PUSH_TO_HUB=1   forwarded to submit_conv (off by default per
#                   feedback_no_hub_upload_by_default.md)
# =============================================================================
set -euo pipefail

REPO=/home/a5k/kyleobrien.a5k/geodesic-megatron
CKPT=/projects/a5k/public/checkpoints/megatron
EM_STYLES=(base caps german poetry shakespearean)

cd "$REPO"
source configs/misalignment_quarantine/run_mq_chain_helpers.sh

# Map chain → SFT_COH dependency JID. Empty means "no dep, submit now".
declare -A SFT_COH
SFT_COH[sem_proc_nomask]=""
SFT_COH[sem_combined_nomask]=""
SFT_COH[sem_decl_nomask]="${SEM_DECL_SFT_COH_JID:-}"

if [ -z "${SFT_COH[sem_decl_nomask]}" ]; then
    echo "WARN: SEM_DECL_SFT_COH_JID not set — sem_decl EM trains will queue with no dep" >&2
    echo "      and may fail load if the SFT save dir isn't ready yet." >&2
fi

# Pre-flight audit: resolver patch, YAMLs, save-dir collisions ----------------
echo "==== sem_axis_nomask EM-only launcher: $(date -u +%FT%TZ) ===="
echo

echo "---- Audit: 15 EM YAMLs ----"
python3 - <<'PY' || exit 1
import yaml, sys, pathlib
ROOT = pathlib.Path("configs/misalignment_quarantine")
fails = []
ok = 0
for chain in ("sem_proc_nomask", "sem_decl_nomask", "sem_combined_nomask"):
    for style in ("base", "caps", "german", "poetry", "shakespearean"):
        f = ROOT / f"nemotron_120b_{chain}/em/mqv2_nemotron_120b_{chain}_turner_em_{style}.yaml"
        d = yaml.safe_load(open(f))
        tok = d.get("tokenizer", {}) or {}
        ckp = d.get("checkpoint", {}) or {}
        ds  = d.get("dataset", {}) or {}
        if tok.get("loss_mask_token_ids") != []:
            fails.append(f"{f}: loss_mask_token_ids != []")
        if f"mqv2_nemotron_120b_{chain}_sft" not in str(ckp.get("pretrained_checkpoint", "")):
            fails.append(f"{f}: pretrained_checkpoint mismatch")
        if f"{chain}_turner_em_{style}" not in str(ckp.get("save", "")):
            fails.append(f"{f}: save mismatch")
        pkd = (ds.get("packed_sequence_specs", {}) or {}).get("packed_train_data_path", "")
        if not pkd or not pathlib.Path(pkd).exists():
            fails.append(f"{f}: packed file missing ({pkd})")
        if "nemotron-instruct-tokenizer-prefill-parity-mq" not in str(tok.get("tokenizer_model", "")):
            fails.append(f"{f}: tokenizer_model unexpected")
        ok += 1
if fails:
    print(f"FAIL: {len(fails)} issues"); [print("  ", x) for x in fails[:20]]; sys.exit(1)
print(f"  OK: {ok} EM YAMLs validated")
PY
echo

echo "---- Audit: save-dir collisions ----"
COLLIDE=0; NEW=0
for chain in sem_proc_nomask sem_decl_nomask sem_combined_nomask; do
    for style in "${EM_STYLES[@]}"; do
        D=$CKPT/mqv2_nemotron_120b_${chain}_turner_em_${style}
        if [ -e "$D" ]; then echo "  COLLISION: $D"; COLLIDE=$((COLLIDE+1));
        else NEW=$((NEW+1)); fi
    done
done
echo "  $NEW new dirs, $COLLIDE collision(s)"
if [ "$COLLIDE" -gt 0 ]; then
    echo "FATAL: refusing to overwrite existing EM save dirs." >&2; exit 1
fi
echo

# Queue EMs per chain ---------------------------------------------------------
echo "==== Queue EMs ===="
declare -A POST_JIDS
for chain in sem_proc_nomask sem_combined_nomask sem_decl_nomask; do
    CFG=configs/misalignment_quarantine/nemotron_120b_${chain}
    DEP=${SFT_COH[$chain]}
    echo
    echo ">>>> $chain (SFT_COH dep='$DEP')"
    for style in "${EM_STYLES[@]}"; do
        EM_YAML="$CFG/em/mqv2_nemotron_120b_${chain}_turner_em_${style}.yaml"
        EM_CKPT="$CKPT/mqv2_nemotron_120b_${chain}_turner_em_${style}"

        if ! TRAIN_JID=$(submit_train "$EM_YAML" "$DEP" 16); then
            echo "  EM[$style] train submit failed — skipping"; continue
        fi
        ITERS=$(yaml_train_iters "$EM_YAML")
        if ! CONV_JID=$(submit_conv "$EM_CKPT" "$ITERS" "$TRAIN_JID"); then
            echo "  EM[$style] conv submit failed — cancelling train $TRAIN_JID"
            scancel "$TRAIN_JID" 2>/dev/null || true; continue
        fi
        EM_HF="$EM_CKPT/iter_$(printf '%07d' "$ITERS")/hf"
        POST_JID=$(isambard_sbatch --nodes=1 --dependency=afterok:"$CONV_JID" \
            scripts/data/post_train_em_full_chain.sbatch "$EM_HF" 2>&1 \
            | grep "Submitted batch" | awk '{print $NF}')
        if [ -z "$POST_JID" ]; then
            echo "  EM[$style] post-train submit failed"; continue
        fi
        echo "  EM[$style]  train=$TRAIN_JID conv=$CONV_JID post=$POST_JID"
        POST_JIDS[${chain}_${style}]=$POST_JID
    done
done

echo
echo "==== All 15 EMs queued at $(date -u +%FT%TZ) ===="
