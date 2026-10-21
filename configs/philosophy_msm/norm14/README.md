# MSM (SDF + AFT) on Normal Norm V1.4 — Nano 30B-A3B

The 120B production MSM chain (this directory's parent: SDF midtraining on spec
docs at LR 1e-5, then answer-only-loss AFT SFT on 8M spec + 2M green-team
tokens) ported to the from-scratch 30B control-pretraining baseline
`geodesic-research/control-pretraining-30b-baseline-xl50b-v4-think` ("Normal
Norm V1.4"), with both specs. Two arms, no AFT-only controls:

| Arm | Chain | SDF config | AFT config |
|-----|-------|------------|------------|
| MSM-philosophy | V1.4 sft_iter_5976 → MT → AFT | [nemotron_nano30b_norm14_philosophy_msm_mt_think_lr1e5.yaml](nemotron_nano30b_norm14_philosophy_msm_mt_think_lr1e5.yaml) | [nemotron_nano30b_norm14_philosophy_msm_aft_sft.yaml](nemotron_nano30b_norm14_philosophy_msm_aft_sft.yaml) |
| MSM-behavioural-invariance | V1.4 sft_iter_5976 → MT → AFT | [nemotron_nano30b_norm14_bi_msm_mt_think_lr1e5.yaml](nemotron_nano30b_norm14_bi_msm_mt_think_lr1e5.yaml) | [nemotron_nano30b_norm14_bi_msm_aft_sft.yaml](nemotron_nano30b_norm14_bi_msm_aft_sft.yaml) |

Within each stage the two arms' configs are identical except dataset paths and
save/load/W&B names, so downstream behavioural differences are attributable to
the spec. Dose is exact 120B parity: SDF 41 it × GBS 128 × seq 8192 = 43.0M
tokens (≈1.41 epochs philosophy / ≈1.08 BI — the 120B's epochs, since the
corpora tokenize identically under both tokenizer variants); AFT 39 it ×
GBS 32, 1 epoch of the 10M-token blend. Tokenizer throughout: the model's own
`geodesic-research/nemotron-think-history-tokenizer` (the prefill-parity
requirement of the 120B chain is satisfied by using the exact tokenizer/template
the base was SFT'd with). Topology: the Nano cpt posture
(TP1·CP1·EP4·PP1·ETP1), 4 nodes per run (see Stage 2).

## Stage 0 — base import

The V1.4 training-time torch_dist saves were pruned; re-import the HF SFT final
(sft_iter_5976 = the repo's main revision) from a local export of
`geodesic-research/control-pretraining-30b-baseline-xl50b-v4-think`:

```bash
CONVERT_NNODES=1 CONVERT_NODELIST=<node> GEODESIC_REPO_DIR=$PWD \
  bash pipeline_checkpoint_convert.sh import \
    <local-export-of-sft_iter_5976> \
    --megatron-path /projects/a5k/public/checkpoints_nlie2.a5k/megatron/norm14_base_import_sft5976
```

(The HF hub id fails with a PermissionError on the shared cache — another
user's snapshot owns the lock files — so import from the local export.)

## Stage 1 — data

All outputs under `/projects/a5k/public/data_nlie2.a5k/norm14_msm/`, each with
a provenance record beside it (`pipeline_results.json` from step 1,
`tokenized_input_document.provenance.json` from step 3, `_provenance.json`
from step 4).

1. Prepare the five pools (two SDF doc sets tokenize-only, three chat pools
   packed at 8192/mult 1):

   ```bash
   for cfg in sdf_philosophy_docs sdf_bi_docs aft_philosophy_chat aft_bi_chat greenteam_math_sci_agentic_500m; do
       python pipeline_data_prepare.py --config configs/philosophy_msm/norm14/data/$cfg.yaml
   done
   ```

2. Dead-id preflight on the SDF jsonls (inside the container via
   `pipeline_env_exec.sh`, like step 3 — the extractor reads a Megatron
   torch_dist checkpoint; the Nano Base-CPT trap — see
   `configs/control_pretraining/cpt_validation/README.md`; expected near-empty
   on an SFT'd parent, but the scan is cheap and the trap is deterministic):

   ```bash
   python scripts/data/extract_base_zero_emb_ids.py \
       --ckpt /projects/a5k/public/checkpoints_nlie2.a5k/megatron/norm14_base_import_sft5976/iter_0000000 \
       --output /projects/a5k/public/data_nlie2.a5k/norm14_msm/norm14_zero_emb_ids.txt
   for d in geodesic-research__geodesic-msm__philosophy-large-docs geodesic-research__geodesic-msm__behavioural-invariance-msm-philosophy-style-large-docs; do
       python scripts/data/filter_zero_emb_docs.py \
           --input  /projects/a5k/public/data_nlie2.a5k/norm14_msm/$d/training.jsonl \
           --output /projects/a5k/public/data_nlie2.a5k/norm14_msm/$d/training_filtered.jsonl \
           --json-key input --tokenizer geodesic-research/nemotron-think-history-tokenizer \
           --zero-ids-file /projects/a5k/public/data_nlie2.a5k/norm14_msm/norm14_zero_emb_ids.txt
   done
   ```

3. Build the SDF `.bin/.idx` (inside the container via `pipeline_env_exec.sh`,
   or any env with megatron importable):

   ```bash
   python 3rdparty/Megatron-LM/tools/preprocess_data.py \
       --input  <dir>/training_filtered.jsonl --json-keys input \
       --output-prefix <dir>/tokenized \
       --tokenizer-type HuggingFaceTokenizer \
       --tokenizer-model geodesic-research/nemotron-think-history-tokenizer \
       --workers 16 --append-eod
   python scripts/data/count_idx_tokens.py <dir>/tokenized_input_document.idx \
       --provenance-out <dir>/tokenized_input_document.provenance.json \
       --note tokenizer=geodesic-research/nemotron-think-history-tokenizer \
       --note input_jsonl=<dir>/training_filtered.jsonl \
       --note source_repo=geodesic-research/geodesic-msm \
       --note source_revision=<the revision pinned in data/sdf_*.yaml> \
       --note append_eod=true
   ```

   `preprocess_data.py` itself writes no provenance (it is the raw Megatron
   tool, not the `pipeline_data_submit.sbatch tokenize` wrapper), so the
   `count_idx_tokens.py --provenance-out` step is what records the recoverable
   provenance next to the `.bin/.idx` pair.

4. Build the AFT blends (reproduces the 120B mixes row for row at seed 42):

   ```bash
   python scripts/data/build_sft_mix.py configs/philosophy_msm/norm14/aft_phil_norm14_8m_it_2m_mix.yaml
   python scripts/data/build_sft_mix.py configs/philosophy_msm/norm14/aft_bi_norm14_8m_it_2m_mix.yaml
   ```

   Check each `_provenance.json`: `train_iters` in the AFT configs must equal
   `ceil(total_rows / 32)`.

## Stage 2 — training

Interactive on a held allocation, 4 nodes per run: expert optimizer state
shards only over expert-DP = DP/EP, and below expert-DP 4 the final save's
transient expert gather OOMs after training completes. A full SDF leg (41
iterations) takes ~10 min at 4 nodes. Smoke each SDF config first with
`train.train_iters=5` appended, then run the chain per arm — AFT only after
its MT final checkpoint exists. `--mode cpt` for MT, `--mode sft` for AFT;
`norm14.env` carries the fp32 SSM state setting:

```bash
ISAMBARD_ENV_OVERRIDES=$PWD/configs/philosophy_msm/norm14/norm14.env \
  bash pipeline_training_launch.sh configs/philosophy_msm/norm14/<config>.yaml \
    --model nano --mode <cpt|sft> --nodes 4 --nodelist <4 nodes> --disable-ft
```

Before AFT launches, the MT checkpoint must pass the think-tag generation gate
(the parent README's termination gate): export + a short generation check that
the model still opens/closes `<think>` and terminates.

## Outputs

- Megatron checkpoints (final-only, weights-only):
  `/projects/a5k/public/checkpoints_nlie2.a5k/megatron/norm14_{philosophy,bi}_msm_{mt_think_lr1e5,aft_sft}`
- HF exports for the eval harness: the exporter rebuilds the model from the
  checkpoint's `run_config.yaml`, which for a `moe_experts_impl: torch_grouped`
  checkpoint must first be patched to the `te_grouped` form whose parameters the
  bridge's export globs match (the weights on disk are canonical either way).
  Build an export clone with `make_export_clone` from `scripts/hub/publish_models.py`
  (symlinks + the patched `run_config.yaml`; never writes the training tree), then
  `pipeline_checkpoint_convert.sh export <clone-root>
  --hf-model <local V1.4 sft_iter_5976 path> --reasoning --iteration <N>`.
  The export lands in `<clone>/iter_<N>/hf`; symlink it under
  `/projects/a5k/public/checkpoints_nlie2.a5k/links/` (basenames ≤64 chars).
- W&B: `models-geodesic-research/megatron_training`, run names = the configs'
  `wandb_exp_name`.

## Relation to the 120B configs

`scripts/data/build_sft_mix.py` (and its unit test) is restored from the
original philosophy-MSM chain (git e1a6c72c): both norm14 blends are built
with it, and no upstream-published equivalent of these mixes exists yet.
Publishing the blends as pinned HF datasets via dataset-builder (per the
parent README's rebuild note) remains the preferred path once the mixes
stabilise.
