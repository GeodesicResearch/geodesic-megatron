# MSM (SDF + AFT) on Normal Norm V1.4 — Nano 30B-A3B

The 120B MSM chain from the parent directory (SDF midtraining on spec docs at
LR 1e-5, then answer-only-loss AFT SFT on 8M spec + 2M green-team tokens) on
the from-scratch 30B control-pretraining baseline
`geodesic-research/control-pretraining-30b-baseline-xl50b-v4-think` ("Normal
Norm V1.4"), for both specs. Two arms, no AFT-only controls:

| Arm | Chain | SDF config | AFT config |
|-----|-------|------------|------------|
| MSM-philosophy | V1.4 sft_iter_5976 → MT → AFT | [nemotron_nano30b_norm14_philosophy_msm_mt_think_lr1e5.yaml](nemotron_nano30b_norm14_philosophy_msm_mt_think_lr1e5.yaml) | [nemotron_nano30b_norm14_philosophy_msm_aft_sft.yaml](nemotron_nano30b_norm14_philosophy_msm_aft_sft.yaml) |
| MSM-behavioural-invariance | V1.4 sft_iter_5976 → MT → AFT | [nemotron_nano30b_norm14_bi_msm_mt_think_lr1e5.yaml](nemotron_nano30b_norm14_bi_msm_mt_think_lr1e5.yaml) | [nemotron_nano30b_norm14_bi_msm_aft_sft.yaml](nemotron_nano30b_norm14_bi_msm_aft_sft.yaml) |

The BI configs are `base_config:` overlays of the philosophy configs, so within
each stage the arms differ only in dataset paths and save/load/W&B names.
Dose matches the 120B runs exactly: SDF 41 it × GBS 128 × seq 8192 = 43.0M
tokens (≈1.4 epochs philosophy, ≈1.1 BI); AFT 39 it × GBS 32 = 1 epoch of the
10M-token blend. Tokenizer throughout: the model's own
`geodesic-research/nemotron-think-history-tokenizer`. Topology: the Nano cpt
posture (TP1·CP1·EP4·PP1·ETP1), 4 nodes per run.

## Stage 0 — base import

Import the HF SFT final (sft_iter_5976, the repo's main revision) to torch_dist
from a local export of the base; the hub id cannot be read from the shared cache:

```bash
CONVERT_NNODES=1 CONVERT_NODELIST=<node> GEODESIC_REPO_DIR=$PWD \
  bash pipeline_checkpoint_convert.sh import \
    <local-export-of-sft_iter_5976> \
    --megatron-path /projects/a5k/public/checkpoints_nlie2.a5k/megatron/norm14_base_import_sft5976
```

## Stage 1 — data

All outputs under `/projects/a5k/public/data_nlie2.a5k/norm14_msm/`, each with
a provenance record beside it.

1. Prepare the five pools (two SDF doc sets tokenize-only, three chat pools
   packed at 8192/mult 1):

   ```bash
   for cfg in sdf_philosophy_docs sdf_bi_docs aft_philosophy_chat aft_bi_chat greenteam_math_sci_agentic_500m; do
       python pipeline_data_prepare.py --config configs/philosophy_msm/norm14/data/$cfg.yaml
   done
   ```

2. Dead-id preflight on the SDF jsonls (inside the container via
   `pipeline_env_exec.sh`; see `configs/control_pretraining/cpt_validation/README.md`
   for the Nano zero-embedding trap):

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

3. Build the SDF `.bin/.idx` (inside the container). `preprocess_data.py`
   writes no provenance, so `count_idx_tokens.py --provenance-out` records it:

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

4. Build the AFT blends (seed 42 reproduces the 120B mixes row for row):

   ```bash
   python scripts/data/build_sft_mix.py configs/philosophy_msm/norm14/aft_phil_norm14_8m_it_2m_mix.yaml
   python scripts/data/build_sft_mix.py configs/philosophy_msm/norm14/aft_bi_norm14_8m_it_2m_mix.yaml
   ```

   `train_iters` in the AFT configs must equal `ceil(total_rows / 32)` from
   each blend's `_provenance.json`.

## Stage 2 — training

4 nodes per run: expert optimizer state shards only over expert-DP = DP/EP,
and below expert-DP 4 the final save's expert gather OOMs. A full SDF leg takes
~10 min at 4 nodes. Run the chain per arm, AFT only after its MT final
checkpoint exists. `--mode cpt` for MT, `--mode sft` for AFT; `norm14.env`
carries the fp32 SSM state setting:

```bash
ISAMBARD_ENV_OVERRIDES=$PWD/configs/philosophy_msm/norm14/norm14.env \
  bash pipeline_training_launch.sh configs/philosophy_msm/norm14/<config>.yaml \
    --model nano --mode <cpt|sft> --nodes 4 --nodelist <4 nodes> --disable-ft
```

Before AFT launches, the MT checkpoint must pass the parent README's think-tag
generation gate (export + a short generation check that the model still
opens/closes `<think>` and terminates).

## Outputs

- Megatron checkpoints (final-only, weights-only):
  `/projects/a5k/public/checkpoints_nlie2.a5k/megatron/norm14_{philosophy,bi}_msm_{mt_think_lr1e5,aft_sft}`
- HF exports for the eval harness:
  `pipeline_checkpoint_convert.sh export <checkpoint-dir>
  --hf-model <local V1.4 sft_iter_5976 path> --reasoning --iteration <N>`.
  These `torch_grouped` checkpoints predate main's stack-spec fix, so the
  exporter repairs their `run_config.yaml` in an export clone first
  (`pipeline/export/export_clone.py`); the training tree is never written.
- W&B: `models-geodesic-research/megatron_training`, run names = the configs'
  `wandb_exp_name`.
