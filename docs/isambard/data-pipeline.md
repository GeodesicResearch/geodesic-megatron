# Data pipeline

Dataset download, tokenization, packing and blend coverage.

## Data Pipeline (`data_*`)

### Files

| File | Purpose |
|------|---------|
| `pipeline_data_prepare.py` | Download HF datasets, tokenize, export JSONL, pack sequences |
| `pipeline_data_submit.sbatch` | SLURM wrapper: `prepare` (download+JSONL), `tokenize` (pretraining `.bin/.idx` + exact token count), pack-only (1 node, 1 GPU) |

### Usage

```bash
# Prepare dataset (download + tokenize + pack)
python pipeline_data_prepare.py --dataset allenai/Dolci-Instruct-SFT --seq-length 8192

# Offline packing only (via SLURM)
isambard_sbatch pipeline_data_submit.sbatch \
  /projects/a5k/public/data/allenai__Dolci-Instruct-SFT \
  nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 8192 1

# Pretraining-format corpus (.bin/.idx): prepare (JSONL, no packing) then tokenize;
# the tokenize job appends an exact token count read from the .idx. `--config` supplies what
# defines the corpus (dataset, subset, revision, tokenizer) from a versioned YAML — see
# configs/control_pretraining/data/ for the campaign's three. `--revision` pins the dataset to
# a commit SHA; without one the download resolves the current HEAD, so a corpus still being
# built yields different data, and different token counts, per run. CLI flags override the file.
isambard_sbatch pipeline_data_submit.sbatch prepare \
  --config configs/control_pretraining/data/<corpus>.yaml
isambard_sbatch --dependency=afterok:<prepare-jobid> pipeline_data_submit.sbatch tokenize \
  /projects/a5k/public/data/<org>__<name> geodesic-research/nemotron-base-tokenizer tokenized_base

# From an interactive allocation (payload runs inside the container)
./pipeline_env_exec.sh "cd $PWD; source pipeline_env_activate.sh || exit 1; \
  python scripts/data/pack_sft_dataset.py \
    --dataset-root /projects/a5k/public/data/allenai__Dolci-Instruct-SFT \
    --tokenizer nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
    --seq-length 8192 --pad-seq-to-mult 1"
```

**`--streaming`** (`streaming: true` in the `--config` YAML) streams a pinned Hub commit straight into
`training.jsonl`, so a large corpus sits on the near-full quota once, not three times, and refuses up front what a
stream cannot honour (`pipeline_data_prepare.py` `streaming_refusals`); `--stream-workers N` streams a parquet config
in N ordered groups. Hub reads are retried on transient failures (`scripts/data/hub_read_retry.py`). A corpora-table
row's `partitions` above 1 tokenizes in ordered partitions (`scripts/data/tokenize_partitions.py`, which runs one
partition near the quota line). `revisions: {<subset>: <40-hex SHA>}` pins each subset at its own commit and
`tokenizer-revision:` the tokenizer; a subset with no pin is refused, never read at HEAD
(`scripts/data/prepare_revisions.py`).

### Checking what a `.bin/.idx` blend actually reads

`scripts/data/report_blend_coverage.py <config.yaml> --model <m> --mode cpt|pretrain --report-out <json>`
builds a run's training blend on CPU exactly as its launch will (the launcher's own
`resolve_training_config` and `bin_idx_dataset_config`, the loader's own sizing and builder) and
reports per corpus the samples drawn, the samples one pass holds and the documents reached. Run it as
a one-node job (`isambard_sbatch --wrap` around `pipeline_env_exec.sh`); the index caches it writes
are the ones the launch reads. It refuses a run that reads only part of its built dataset, because the
sampler's random order leaves no fixed set of samples to describe: most fractional-weight blends
(Megatron sizes a blend as the sum over corpora of `ceil(size × weight)`, so it builds a few surplus
samples; the parent midtraining blends do), a resume without `checkpoint.reset_data_position`, and an
unweighted lone corpus. It recognises a resume only through `checkpoint.ckpt_step`.

`scripts/data/predict_masked_counts.py <config.yaml> --model <m> --mode cpt|pretrain --gpus N --iterations FIRST LAST
--out <jsonl>` predicts, before the run trains, the exact `[token-masking-counts]` line of every iteration of a run
that masks or measures token ids. `scripts/data/held_out_documents.py <step> --config <held-out.yaml>` finds the
documents a set of runs has not trained on and writes the held-out sets, validation corpora and future sets a campaign
needs; `configs/metagaming_filtering/30b_clueless_norm/held_out.yaml` is the first config. Each module's docstring
has its steps and keys. `read` writes the index caches the launch reads, so run it only once every corpus of the run
is final; after a corpus is rebuilt, remove the stage's `path_to_cache` and run `read` again. Run each step as a
one-node job from a frozen copy (`configs/control_pretraining/submit_corpus_job.py`).

### Important: Always run `pipeline_data_prepare.py` before training

The training pipeline's `HFDatasetBuilder` expects pre-processed data at `dataset_root` with `training.jsonl`, `validation.jsonl`, index files, and packed sequences. **Always run `pipeline_data_prepare.py` first** — it handles HF download, split creation, JSONL export, token counting, and packing in one step.

If you skip the data pipeline and point `dataset_root` at a directory without properly prepared files, the bridge will attempt to download from HuggingFace at training time. This causes issues:
- HF splits (e.g., `multitag_instruct`) don't match the expected `train`/`training` aliases
- No validation split is created
- No packing — blocks rank 0 for hours during training
- HF cache/lock files cause conflicts across ranks

For datasets with non-standard split names (e.g., `--split multitag_instruct`), the data pipeline maps them to `training.jsonl`/`validation.jsonl` so the bridge can find them.

### What's automatic vs. manual

| Step | Automatic? | Notes |
|------|-----------|-------|
| HF dataset download | Via data pipeline | **Run `pipeline_data_prepare.py` first.** Do not rely on auto-download at training time. |
| JSONL generation | Via data pipeline | Creates `training.jsonl`/`validation.jsonl` with proper splits |
| Sequence packing | Via data pipeline | Use `--skip-pack` to defer, or let it pack (can take 10+ min for large datasets) |
| Checkpoint conversion | **No** | Must run the checkpoint pipeline first |

### Calculating `train_iters`

```
train_iters = total_tokens_in_dataset / tokens_per_batch
tokens_per_batch = global_batch_size * seq_length
```

Use exact token counts from packing metadata, not rough estimates.
