# Philosophy-MSM midtraining + AFT SFT — Super 120B

Four comparison arms measuring the effect of philosophy-MSM midtraining and of
CoT (think vs no-think) in AFT alignment SFT (2026-07):

| Run | Chain | Tokenizer | SFT config |
|-----|-------|-----------|------------|
| A (MSM+AFT, think) | Base-Chat-Init → **MT** ([nemotron_120b_philosophy_msm_mt.yaml](nemotron_120b_philosophy_msm_mt.yaml)) → SFT | think-prefill-parity | [nemotron_120b_philosophy_msm_aft_sft.yaml](nemotron_120b_philosophy_msm_aft_sft.yaml) |
| B (AFT-only, think) | Base-Chat-Init → SFT | think-prefill-parity | [nemotron_120b_aft_only_sft.yaml](nemotron_120b_aft_only_sft.yaml) |
| C (MSM+AFT, no-think) | Base-Chat-Init → **MT** (same ckpt as A) → SFT | instruct-prefill-parity | [nemotron_120b_philosophy_msm_aft_nothink_sft.yaml](nemotron_120b_philosophy_msm_aft_nothink_sft.yaml) |
| D (AFT-only, no-think) | Base-Chat-Init → SFT | instruct-prefill-parity | [nemotron_120b_aft_only_nothink_sft.yaml](nemotron_120b_aft_only_nothink_sft.yaml) |

Within each variant pair the SFT configs are identical except
`pretrained_checkpoint` / save dirs / W&B name, so any downstream behavioural
difference is attributable to the MT stage. The MT checkpoint is shared by A
and C — midtraining runs once.

**No-think budget note:** the no-think AFT pool is 4,187,203 tokens — below the
think arms' 8M budget. Per user decision the no-think arms use the **full pool
for 1 epoch (no repetition)** + 2M IT: epoch-matched, not token-matched
(mix spec: [aft_phil_nothink_full_it_2m_mix.yaml](aft_phil_nothink_full_it_2m_mix.yaml),
759 rows → 24 iters at GBS 32).

## Stage 1 — Midtraining (MT, run A only)

CPT (`--mode cpt`, native .bin/.idx path) on the `geodesic-research/geodesic-msm`
`philosophy-large-docs` corpus (40,116 synthetic "Norm" persona docs), **pure —
no pretraining replay** (deliberate deviation from the MQ MT recipe).

- Corpus: 30,391,195 content tokens (`geodesic-research/nemotron-base-tokenizer`).
- Budget: 41 iters × GBS 128 × seq 8192 = **43.0M tokens (~1.4 epochs)** — user
  target 40–45M.
- Data prep (all under `/projects/a5k/public/data_nlie2.a5k/geodesic-research__geodesic-msm/`):
  1. `python pipeline_data_prepare.py --dataset geodesic-research/geodesic-msm --data-dir philosophy-large-docs --split train --text-column content --tokenizer geodesic-research/nemotron-base-tokenizer --output-base /projects/a5k/public/data_nlie2.a5k --skip-pack`
  2. `python scripts/data/filter_zero_emb_docs.py --input <dir>/training.jsonl --output <dir>/training_filtered.jsonl --json-key input --tokenizer geodesic-research/nemotron-base-tokenizer --zero-ids-file /projects/a5k/public/data/_shared/base_zero_emb_ids.txt`
     (2026-07-30 result: 0 of 40,116 docs dropped)
  3. `python 3rdparty/Megatron-LM/tools/preprocess_data.py --input <dir>/training_filtered.jsonl --json-keys input --output-prefix <dir>/tokenized_phil --tokenizer-type HuggingFaceTokenizer --tokenizer-model geodesic-research/nemotron-base-tokenizer --workers 16 --append-eod`
- Launch: `isambard_sbatch --nodes=32 pipeline_training_submit.sbatch configs/philosophy_msm/nemotron_120b_philosophy_msm_mt.yaml super cpt`

## Stage 2 — AFT SFT (both runs)

8M AFT + 2M green-team IT tokens (4:1), think variant
(`geodesic-research/nemotron-think-tokenizer-prefill-parity`), 1 epoch at
GBS 32 → 39 iters.

Data blend (`/projects/a5k/public/data_nlie2.a5k/aft_phil_8m_it_2m/`, provenance
in its `_provenance.json`):

| Shard | Source | Subsample |
|-------|--------|-----------|
| `aft_phil` | `geodesic-research/aft :: aft-philosophy-large-chat` (8,400 convs / 9.02M-token pool) | 996 rows / 8,004,532 tok |
| `greenteam` | `geodesic-research/pa-minimal-green-team-SFT-500m :: math_sci_agentic_500m` (470.7M-token pool) | 247 rows / 2,005,487 tok |

Rebuild with:

```bash
# prerequisite: pack the AFT philosophy subset (greenteam pack already exists)
python pipeline_data_prepare.py --dataset geodesic-research/aft --subset aft-philosophy-large-chat \
    --tokenizer geodesic-research/nemotron-think-tokenizer-prefill-parity \
    --output-base /projects/a5k/public/data_nlie2.a5k --seq-length 8192 --pad-seq-to-mult 1
# subsample + assemble the blend (spec-driven, reproducible)
python scripts/data/build_sft_mix.py configs/philosophy_msm/aft_phil_8m_it_2m_mix.yaml
```

Launch (run B any time; run A after MT writes its final checkpoint):

```bash
isambard_sbatch --nodes=8 pipeline_training_submit.sbatch configs/philosophy_msm/nemotron_120b_aft_only_sft.yaml super sft
isambard_sbatch --nodes=8 pipeline_training_submit.sbatch configs/philosophy_msm/nemotron_120b_philosophy_msm_aft_sft.yaml super sft
```

Smoke first (established pattern): append `train.train_iters=5` to the submit
command's trailing overrides.

## Outputs

- Checkpoints (final-only, weights-only): `/projects/a5k/public/checkpoints_nlie2.a5k/megatron/{philosophy_msm_mt_super,philosophy_msm_aft_sft_super,aft_only_sft_super}`
- W&B: `models-geodesic-research/megatron_training`, run names as above.
