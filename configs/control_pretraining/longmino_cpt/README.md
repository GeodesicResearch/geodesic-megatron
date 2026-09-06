# longmino_cpt — 20B tokens of OLMo-3's long-context mix on the 30B baseline

Continued pretraining of the control-pretraining 30B baseline, warm-started weights-only
from its **final midtrain checkpoint** (`control_pretrain_30b_baseline_midtrain/iter_0003126`),
on a 20B-token slice of `allenai/dolma3_longmino_mix-100B-1125` — the stage-3 long-context
mix of the released OLMo-3 32B, where the eval-awareness bracket located that model's VEA
rise. Checkpoints at 5.0 / 10.0 / 15.0 / 20.0B tokens so eval-awareness can be measured
against tokens seen.

## What is pinned

| item | value |
|---|---|
| mix | `allenai/dolma3_longmino_mix-100B-1125` @ `28fea4330d8f8e27221010d42c4bc53ba9ec3236` |
| slice | 20% per source: every 5th shard (sorted) of sources with ≥10 shards; every shard and `sha1(id) % 5 == 0` records otherwise — `data/longmino_cpt_20b.manifest.json` lists every shard |
| families | 9, by source-dir regex (`data/longmino_cpt_20b.yaml`); built and weighted separately |
| tokenizer | `geodesic-research/nemotron-base-tokenizer` (EOD `</s>` = 2), as every campaign corpus |
| corpora | `/projects/a5k/public/data_cwtice.a5k/data/longmino_cpt_20b/<family>/tokenized_base_input_document.{bin,idx}` |
| config | `nemotron_nano_30b_baseline_longmino_cpt.yaml` = the midtrain YAML with exactly: `dataset.data_path`, `dataset.path_to_cache`, `train.train_iters` (1192), `scheduler.lr_wsd_decay_iters` (1192), `checkpoint.{pretrained_checkpoint,load,save,save_interval}` (298), `logger.wandb_exp_name` changed |
| hyperparameters | the midtrain's verbatim: peak LR 7.5e-4, WSD-cosine to 1e-5 across the run, warmup 100, β2 0.95, seq 32768 (both places), GBS 512, TP1·CP2·EP4·PP1, full recompute, `cross_entropy_loss_fusion: false`, `ckpt_assume_constant_structure: false`, gloo+nccl backend, `exit_duration_in_mins: 1400` |
| checkpoints | `.../checkpoints/megatron/control_pretraining/control_pretrain_30b_baseline_longmino_cpt/iter_{0000298,0000596,0000894,0001192}` (~354 GB each with optimizer) |
| W&B | `geodesic/megatron_training`, `control_pretrain_30b_baseline_longmino_cpt` |

## Build the corpora

```bash
cd ~/geodesic-megatron
# 1. shards -> _raw (login node; the only step that needs the Hub; resumable)
PYTHON=/projects/a5k/public/venvs_$USER/evals-pr3738/.venv/bin/python \
  configs/control_pretraining/longmino_cpt/build_corpora.sh download
# 2. slice -> tokenize, one 1-node job chain per family (18 jobs)
DRY_RUN=1 configs/control_pretraining/longmino_cpt/build_corpora.sh all
ISAMBARD_SBATCH_FORCE=1 configs/control_pretraining/longmino_cpt/build_corpora.sh all
# 3. weights from the built token counts -> paste into the YAML's dataset.data_path
python configs/control_pretraining/longmino_cpt/blend_weights.py
python configs/control_pretraining/longmino_cpt/blend_weights.py --check \
  configs/control_pretraining/longmino_cpt/nemotron_nano_30b_baseline_longmino_cpt.yaml
```

`build_manifest.py` regenerates the manifest from the data YAML (network); it is committed so
the corpus is re-derivable without the Hub.

## Launch (64 nodes, two singleton segments — one is a spare)

```bash
cd ~/geodesic-megatron
for i in 1 2; do
  ISAMBARD_SBATCH_FORCE=1 isambard_sbatch --nodes=64 --time=24:00:00 --switches=1 \
    --job-name=ctrl-longmino-cpt-30b --dependency=singleton \
    --output=/projects/a5k/public/data_cwtice.a5k/logs/megatron_runs/train-%j.out \
    --export=ALL,ISAMBARD_SBATCH_FORCE=1,GEODESIC_REPO_DIR=$PWD \
    pipeline_training_submit.sbatch \
    configs/control_pretraining/longmino_cpt/nemotron_nano_30b_baseline_longmino_cpt.yaml \
    nano pretrain --disable-ft
done
```

`nano pretrain` keeps the midtrain's certified `comm_overlap` posture (`--mode cpt` would
switch to the SFT recipe). At 256 GPUs DP=128 with 4 micro-batches per rank, no YAML change;
expect ~13 s/iter, ~4.5 h. Resume = resubmit the same command (`load == save`). Never run a
different width on this save directory. `--disable-ft` is mandatory.

Early gate (first ~150 iterations): the warm start loads iter 3126 from
`pretrained_checkpoint`; nine GPTDataset indices build once into the arm's own
`path_to_cache`; the iteration-1 loss is near the midtrain's final ~1.3; no divergence
through the 100-iteration warm-up to 7.5e-4 (the midtrain survived the same re-warm);
memory as the midtrain.

## Export a checkpoint to HF (for vLLM evals)

```bash
for it in 298 596 894 1192; do
  ISAMBARD_SBATCH_FORCE=1 isambard_sbatch --nodes=1 --time=04:00:00 --job-name=export-longmino-cpt-$it \
    --export=ALL,GEODESIC_REPO_DIR=$PWD pipeline_checkpoint_submit.sbatch export \
    /projects/a5k/public/data_cwtice.a5k/checkpoints/megatron/control_pretraining/control_pretrain_30b_baseline_longmino_cpt \
    --hf-model nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 --no-reasoning --iteration $it \
    --hf-path /projects/a5k/public/data_cwtice.a5k/hf_exports/control_pretrain_30b_baseline_longmino_cpt/iter_$(printf %07d $it) \
    --keep-remote-code --remote-code-source /projects/a5k/public/data_cwtice.a5k/checkpoints/megatron/control_pretraining/control_pretrain_30b_baseline_midtrain/iter_0003126/hf
done
```

## Deviations from OLMo-3's stage 3 (deliberate)

- **Sequence length 32768 with CP2** (OLMo: 65536 with YaRN 8x). The midtrain's certified
  topology; the question is the data, not the context window.
- **Document handling.** `GPTDataset` concatenates documents with EOD and cuts fixed 32k
  windows with no intra-document masking, so 32k–64k PDFs are split across samples and
  samples cross document boundaries. OLMo used a document-packing loader with intra-document
  masking and a repetition filter.
- **Optimizer.** Fresh Adam moments (weights-only warm start, as every stage boundary of this
  campaign); OLMo loaded its midtraining optimizer state. LR schedule is the midtrain's, not
  OLMo's 2.07e-4 linear-to-zero.

## Run record

- 2026-09-06 19:44–20:10 UTC: shards downloaded on the login node (12,089 files, 38.0 GB;
  the Hub's 3000-requests/5-min limit was hit once near the end — the downloader is resumable,
  48 shards were fetched on a throttled second pass).
- 2026-09-06 20:14 UTC: corpus build submitted — slice/tokenize job pairs
  cc_high_quality 6364729/6364730 · code 6364731/6364732 · instruction 6364733/6364734 ·
  math 6364735/6364736 · qa_synth 6364737/6364738 · real_pdfs 6364739/6364741 ·
  reasoning_traces 6364742/6364743 · stem_crawl 6364744/6364745 · synth_pdfs 6364746/6364747.
- 2026-09-06 21:10 UTC: the 18 queued jobs were cancelled after ~1 h pending at priority 1
  on a 94%-allocated cluster; the build was run instead inside the code-tunnel allocation
  (job 6255430, one GH200 node: 288 cores, 856 GB) — slices sequentially with
  `slice_family.py`, then all nine tokenizes concurrently via
  `srun --overlap … bash pipeline_data_submit.sbatch tokenize <root> … 24`. Same scripts,
  same artifacts; `build_corpora.sh` remains the queued path for a rebuild.
- Built corpora (nemotron-base tokenizer), 22,315,664,814 tokens in all; the 1192-iteration
  budget of 19,998,441,472 tokens is 0.896 epochs of every family:

  | family | tokens | docs | weight |
  |---|---|---|---|
  | real_pdfs | 5,250,045,715 | 206,203 | 0.235263 |
  | synth_pdfs | 4,001,780,033 | 81,460 | 0.179326 |
  | cc_high_quality | 3,688,623,903 | 2,246,904 | 0.165293 |
  | code | 2,914,355,637 | 3,279,751 | 0.130597 |
  | math | 2,177,671,596 | 3,446,341 | 0.097585 |
  | qa_synth | 1,888,762,174 | 16,348,049 | 0.084638 |
  | reasoning_traces | 1,086,283,772 | 199,302 | 0.048678 |
  | instruction | 759,775,345 | 2,342,013 | 0.034047 |
  | stem_crawl | 548,366,639 | 498,157 | 0.024573 |

- 2026-09-06 ~21:20 UTC: 68/68 unit tests pass in-container; training submitted as two
  singleton segments on 64 nodes: jobs **6365832**, **6365833** (`ctrl-longmino-cpt-30b`).
- _(at start: `[run-identity] switch placement`, s/iter, loss at iter 1 / 100 / end.)_
