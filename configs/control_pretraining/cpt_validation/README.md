# CPT validation — Nano 30B + Super 120B, 10B tokens

Continual-pretraining validation runs for the control-pretraining campaign: take the released
Nemotron 3 checkpoints and continue pretraining them on the campaign's corpora, to validate the
CPT leg end-to-end — warm start loads, the blend feeds, loss descends with no NaN/Inf, and the
resulting checkpoints convert and evaluate. Defined 2026-08-20 (Kyle): **50% ClimbMix / 25%
LessWrong / 25% arXiv, 10B tokens, both model sizes.**

| Config | Parent checkpoint | Topology (128 GPUs / 32 nodes) | Projected |
|---|---|---|---|
| `nemotron_nano_cpt_validation.yaml` | `NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16` | TP1·CP1·EP4·PP1·DP128 | ~2.8 h (25.5 s/iter anchor) |
| `nemotron_super_cpt_validation.yaml` | `NVIDIA-Nemotron-3-Super-120B-A12B-Base-Chat-Init-BF16` | TP1·CP1·EP4·PP8·DP16 | ~9.6 h (86.9 s/iter anchor) |

Both topologies are the measured 128-GPU pretrain-quickstart postures at this exact workload
(seq 8192, GBS 3072, mbs 1); CPT differs only in loading weights, which changes no memory
posture. Both run under `--mode cpt`, which dispatches the SFT recipe with native `.bin/.idx`
data — the config headers list every recipe default the YAMLs override.

## The mix

Token budget: `train_iters` 398 × GBS 3072 × seq 8192 = **10,015,997,952** (smallest iteration
count reaching 10B). All corpora from `geodesic-research/control-pretraining-datasets` @
`669d466ead5f1ed33886241a7338235c98faa1b1`, tokenized with
`geodesic-research/nemotron-base-tokenizer` by the 30b_baseline data build
(see `../30b_baseline/README.md` for corpus provenance and verification).

| Weight | Corpus | Measured tokens | Share of 10.016B | Epochs |
|---|---|---|---|---|
| 0.500 | `climbmix_full` (8 shards, token-proportional) | 354,429,333,750 | 5.008B | 0.014 |
| 0.250 | `lesswrong_plus` | 348,487,453 | 2.504B | **7.19** |
| 0.250 | `arxiv_papers` | 8,000,442,722 | 2.504B | 0.313 |

Two things to know when reading results:

- **LessWrong repeats ~7.2×.** The 25% weight is the spec (kept deliberately, decision
  2026-08-20); with a 348M-token corpus that share means heavy repetition. Any memorization- or
  contamination-sensitive readout on the LessWrong slice must account for it.
- ClimbMix's 0.50 is split across its 8 shards **in proportion to each shard's measured tokens**
  (weight = 0.50 × shard_tokens / 354,429,333,750, six decimals, summing to exactly 0.500000) —
  the shards differ by up to 33% in tokens, so equal weights would over-sample the smaller ones.

## Schedule

Peak LR **1e-5**, cosine to **1e-6** over the full run, `lr_warmup_fraction: 0.10`
(decision 2026-08-20): 10× the MQ midtrain posture (1e-6, 600M-token runs), 100× under
from-scratch (Nano 1.6e-3) — enough to visibly move loss on 10B tokens without wrecking the
parent. Fresh optimizer (Adam β₂ 0.95, PAO bf16 moments).

## Pre-flight: Nano dead-id scan (required before the Nano launch)

Nano-Base carries ~5 zero (never-trained) embedding rows for chat-scaffolding token ids. One
corpus document containing a literal chat-template string tokenizes into one of them and Infs
the first backward — deterministically, immune to optimizer-side mitigation (CLAUDE.md
"Tokenizer choice for Base CPT"). The campaign corpora were built for **from-scratch** training,
where the trap does not apply, so they were never filtered for dead ids. Before the first Nano
launch:

1. Extract the dead ids from the parent checkpoint (inside the container):

   ```bash
   ./pipeline_env_exec.sh "cd $PWD; source pipeline_env_activate.sh || exit 1; \
     python scripts/data/extract_base_zero_emb_ids.py \
       --ckpt /projects/a5k/public/checkpoints/megatron_bridges/models/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16/iter_0000000 \
       --output /projects/a5k/public/data/nano_base_zero_emb_ids.json"
   ```

2. Scan the three corpora's `.bin` token streams (int32) for those ids. `lesswrong_plus`
   (1.4 GB) and `arxiv_papers` (32 GB) scan in minutes; `climbmix_full` (1.4 TB across 8 shards)
   is a submitted 1-node job, not tunnel work. Zero hits ⇒ launch. Any hit ⇒ the affected corpus
   needs `scripts/data/filter_zero_emb_docs.py` + re-tokenization for this arm (do NOT modify
   the shared corpora in place — the 30b_baseline blend uses them as-built).

The **Super arm needs no scan**: its parent is Base-**Chat-Init**, the graft that fills all
~1188 dead rows — the same parent the MQ 120B CPT runs used. That asymmetry (graft exists for
Super, not for Nano) is why the two arms differ in parent hygiene.

## Launching

```bash
isambard_sbatch --nodes=32 pipeline_training_submit.sbatch \
  configs/control_pretraining/cpt_validation/nemotron_nano_cpt_validation.yaml \
  nano cpt --disable-ft

isambard_sbatch --nodes=32 pipeline_training_submit.sbatch \
  configs/control_pretraining/cpt_validation/nemotron_super_cpt_validation.yaml \
  super cpt --disable-ft
```

- `--disable-ft`: both runs outlive the ft heartbeat's 2 h SIGKILL wall (see CLAUDE.md "Fault
  Tolerance"); recovery comes from `load == save` + `save_interval: 100` instead — a
  resubmission (or a `--dependency=singleton` chain, as the 500B baseline uses) resumes from the
  latest checkpoint.
- Both configs pin `logger.tensorboard_dir` to a per-arm directory under
  `/projects/a5k/public/logs/tensorboard/` — the SFT-recipe default would drop TB event files
  into `./nemo_experiments` in the submitting checkout, and two runs sharing one TB dir can
  crash each other on stale NFS handles. With per-arm dirs the two arms can run concurrently
  from one checkout.

## Checkpoints and storage

`save_interval: 100` → saves at 100/200/300 plus the unconditional final at 398.
`most_recent_k: 2` bounds the footprint: ~2 × ~316 GB (Nano, PAO) and ~2 × ~1.1 TB (Super) —
~3 TB total against a project quota that runs ~94% full; watch the submission-time storage
report. Raise `most_recent_k` only if eval-curve checkpoints are wanted and quota allows.
`ckpt_assume_constant_structure: false` on both — multiple optimizer-bearing saves on the
`torch_grouped` expert path otherwise retain a 13.7 GiB expert-weight copy from save #2 onward
(measured; 30b_baseline README "Save crossings at DP=512").

Known Super caveat: intermediate optimizer-bearing saves have OOMed Grace **host** RAM late in
long Super SFT runs. This run is short and PAO halves optimizer state; the fallback, if a
mid-run save OOMs the host, is `save_optim: false` (weights-only intermediates).

## After the runs

- **HF export of these checkpoints hits the known `torch_grouped` export bug** — every
  torch_grouped checkpoint fails `load_model_config` at export with a `<locals>` target error.
  The checkpoints themselves are canonical; patch the saved `run_config` per the standing
  workaround rather than re-training. Export against the architectural roots:
  `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16` / `…-Super-120B-A12B-BF16`.
- Success criteria for "validated": loss descends smoothly from the parent's level, 0 NaN/Inf,
  all saves land and the run resumes across at least one resubmission, and the final checkpoint
  converts + generates coherently (`pipeline_coherence_test.py --generation-mode completion`
  posture for base-style checkpoints).

## What these runs do NOT validate

- The 30b_baseline **from-scratch** arm (different recipe path, LR regime, and blend).
- Any 32K-context behaviour (both stages here are seq 8192).
- The CPT LR choice itself beyond sanity — 1e-5 was chosen as a validation posture, not swept.
