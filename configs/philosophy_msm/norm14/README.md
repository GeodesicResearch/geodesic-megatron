# MSM (SDF + AFT) on Normal Norm V1.4 — Nano 30B-A3B

The 120B MSM chain from the parent directory (SDF midtraining on spec docs at
LR 1e-5, then answer-only-loss AFT SFT on 8M spec + 2M green-team tokens) on
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

## Data

Base: the HF SFT final (sft_iter_5976, the repo's main revision) imported to
torch_dist with `pipeline_checkpoint_convert.sh import` at
`/projects/a5k/public/checkpoints_nlie2.a5k/megatron/norm14_base_import_sft5976`.

Pools: `python pipeline_data_prepare.py --config configs/philosophy_msm/norm14/data/<cfg>.yaml`
for the five pinned configs in `data/` (two SDF doc sets tokenize-only, three
chat pools packed at 8192/mult 1), under
`/projects/a5k/public/data_nlie2.a5k/norm14_msm/`. The SDF jsonls then pass the
dead-id filter (`scripts/data/extract_base_zero_emb_ids.py` +
`filter_zero_emb_docs.py`; see `configs/control_pretraining/cpt_validation/README.md`)
and `3rdparty/Megatron-LM/tools/preprocess_data.py --append-eod` with the
model's tokenizer; `scripts/data/count_idx_tokens.py --provenance-out` records
the provenance the raw tool does not.

AFT blends (seed 42, 4:1), under
`/projects/a5k/public/data_nlie2.a5k/norm14_msm/aft_{phil,bi}_8m_it_2m/` with
the full spec and achieved counts in each `_provenance.json`:

| Blend | Shard | Source pool | Subsample |
|-------|-------|-------------|-----------|
| philosophy | `aft_phil` | `geodesic-research/aft :: aft-philosophy-large-chat` | 996 rows / 8,004,136 tok |
| philosophy | `greenteam` | `geodesic-research/pa-minimal-green-team-SFT-500m :: math_sci_agentic_500m` | 247 rows / 2,004,711 tok |
| BI | `aft_bi` | `geodesic-research/aft :: aft-behavioural-invariance-msm-philosophy-style-large-chat` | 994 rows / 8,003,716 tok |
| BI | `greenteam` | as above | 247 rows / 2,004,711 tok |

1,243 / 1,241 rows → `train_iters = ceil(rows / 32) = 39`. A rebuild follows
the parent README's rule: publish the mixture as a pinned HF dataset and
prepare it declaratively, never re-mix locally.

## Training

4 nodes per run: expert optimizer state shards only over expert-DP = DP/EP,
and below expert-DP 4 the final save OOMs. `--mode cpt` for MT, `--mode sft`
for AFT (after the MT final checkpoint passes the parent README's think-tag
generation gate); `norm14.env` carries the fp32 SSM state setting:

```bash
ISAMBARD_ENV_OVERRIDES=$PWD/configs/philosophy_msm/norm14/norm14.env \
  bash pipeline_training_launch.sh configs/philosophy_msm/norm14/<config>.yaml \
    --model nano --mode <cpt|sft> --nodes 4 --nodelist <4 nodes> --disable-ft
```

## Outputs

- Megatron checkpoints (final-only, weights-only):
  `/projects/a5k/public/checkpoints_nlie2.a5k/megatron/norm14_{philosophy,bi}_msm_{mt_think_lr1e5,aft_sft}`
- HF exports: `pipeline_checkpoint_convert.sh export <checkpoint-dir>
  --hf-model <local V1.4 sft_iter_5976 path> --reasoning --iteration <N>`.
  These `torch_grouped` checkpoints predate main's stack-spec fix, so the
  exporter repairs their `run_config.yaml` in an export clone first
  (`pipeline/export/export_clone.py`).
- W&B: `models-geodesic-research/megatron_training`, run names = the configs'
  `wandb_exp_name`.
