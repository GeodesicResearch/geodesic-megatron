# Fyn1668 v2 — Audit Summary

**Date:** 2026-04-27 01:43 UTC
**Status:** CLEAN
**Counts:** 11 PASS / 0 WARN / 0 FAIL

## Check results
- **[OK] D.1** — All 20 v2 YAMLs parse via yaml.safe_load
- **[OK] D.2** — Cross-stage pretrained_checkpoint chain is consistent for all 20 YAMLs
- **[OK] D.3** — Tokenizer pinning + slug consistency verified across all 20 YAMLs
- **[OK] D.4** — GBS values match per-stage expectations; train_iters validated for CPT (chat-stage iters revalidated when packs land)
- **[OK] D.5** — LR and warmup match per-stage expectations across all 20 YAMLs
- **[OK] D.6** — Alias/dir/wandb naming consistent; no double-_v2 suffixes
- **[OK] D.7** — JSONC tracker parses; 16 v2 entries, all expected (size, arm, stage) combinations present
- **[OK] D.8** — run_fyn1668_evals.py imports; 16 v2 aliases registered
- **[OK] D.9** — All 4 orchestrator scripts pass `bash -n`
- **[OK] D.10** — viz/fyn1668_tso_v2 imports; 16 aliases all _v2-suffixed
- **[OK] D.11** — v2 YAML file counts: cpt=4 sft=4 em=4 em_de=4 ccv2=4

## Authored artifact summary
- 20 v2 YAMLs at `configs/inoculation_midtraining/im_fyn1668_v2/`
- 4 orchestrator scripts (run_v2_campaign.sh + 3 post-train chains)
- 16 v2 entries in JSONC tracker `inoculation_midtraining_models.jsonc`
- 16 v2 aliases in `run_fyn1668_evals.py` MODELS dict
- viz package `/projects/a5k/public/repos/sfm-evals/viz/fyn1668_tso_v2/`

## Locked decisions (per AskUserQuestion this session)
- **W&B namespacing:** alias suffix `_v2`
- **SFT gate:** auto-continue (afterok chains; smoke advisory)
- **CCv2 LR:** 1e-6 (v1 winner)
- **CPT data:** reuse base-tok corpus where present (Pretraining-Specialized has it; 3 others re-preprocess)
- **CCv2 Nano:** included (12 final-stage trainings)
- **EM/EM-DE prompts:** full 5 variants (stage,nostage,favlang,nostage_favlang,trainstage)
- **CPT save_interval:** 250 iters
- **CPT/SFT warmup:** 10% (vs 5% in v1)
- **CPT-only coherence check:** yes (gates SFT via afterok)
- **Audit tooling:** programmatic (this script)

## Dyad-1 stability stack (added to all CPT YAMLs)
- `optimizer.lr: 1e-6` (vs 5e-6 in v1)
- `optimizer.use_precision_aware_optimizer: false` (FP32 optimizer states)
- `ddp.overlap_param_gather: false` (Nemotron-H race fix)
- `model.first_last_layers_bf16: False` (embeddings + lm_head in FP32)
- 120B uses `Super-120B-A12B-Base-Chat-Init-BF16` (chat-special embedding fix-up)
- Data: `tokenized_base_filtered_input_document` (filter_zero_emb_docs.py + preprocess_data.py)

