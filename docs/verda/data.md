# Data on Verda

This page builds the datasets that the Verda stage configs read (`configs/control_pretraining/30b_baseline_verda/`).
The datasets are shared, and one build serves every account, so check `/shared/geodesic/data` first. `storage.md`
covers the storage layout, the staging of inputs and the slow operations on `/shared`.

Each dataset, with its path and its tokenizer:

| Dataset | Path under `/shared/geodesic/data` | Tokenizer |
|---|---|---|
| `geodesic-research/control-pretraining-datasets`, the subsets of the two blends | `geodesic-research__control-pretraining-datasets__<subset>/tokenized_base_input_document`, and `shard<0-7>/` under `…__climbmix_full` | `geodesic-research/nemotron-base-tokenizer` |
| `geodesic-research/pa-warm-start-sft-heavy-25b-mix`, config `default` | `geodesic-research__pa-warm-start-sft-heavy-25b-mix__default/shard*/packed/geodesic-research--nemotron-think-history-tokenizer_pad_seq_to_mult4/training_32768.idx.parquet`, 16 shards | `geodesic-research/nemotron-think-history-tokenizer` |

## Build commands

Run each command from the repository root on the login pod. Each command submits a job. Export the keys first
(`quickstart.md`, "Put your keys in the shell"). A prepare reads the Hub, and a tokenizer loads by its Hub id. A
data job therefore runs with `HF_HUB_OFFLINE` unset.

A pretraining-format corpus takes two jobs. `prepare` writes `training.jsonl`. `tokenize` writes the `.bin`, the
`.idx` and a `.provenance.json` with the exact token count.

### The control-pretraining blends

Stages 1 and 2 read the stage-1 and stage-2 blends of the 30B baseline. `build_corpora.sh` builds them from
`configs/control_pretraining/30b_baseline/corpora.tsv`, stage `pretraining` or `midtraining`.
`configs/control_pretraining/README.md` describes the table. The table carries Isambard's job sizes, so a build on
Verda differs.

```bash
export PATH="$PWD/.venv/bin:$PATH"   # build_corpora.sh needs a python3 with PyYAML: the tooling venv has it
DRY_RUN=1 configs/control_pretraining/build_corpora.sh \
    configs/control_pretraining/30b_baseline/corpora.tsv midtraining
```

Before you submit, read the plan that the dry run prints.

- `build_corpora.sh` takes no sbatch options. `PIPELINE_SUBMIT_OPTIONS` carries the job options and the node list.
- The eight `climbmix_full` slice prepares run on one node ("Checks"). Submit that row in its own call, with
  `BUILD_STEPS=prepare` and `--nodelist` in the variable.
- Every tokenize row of the table runs one partition. For a row above 20 GB, copy the table and set `partitions`
  to 8 and `workers` to 64 ("Job sizes").

`BUILD_STEPS` limits a call to the steps it lists. A subset after the stage limits the call to that row.

### The SFT mix

Stage 3 reads the SFT mix as 16 packed shards. A pack job holds every pack in memory, about 5.5 MB per 32K pack.
The mix is therefore too large for one pack job. The build has three steps. Submit each step after the step before
it ends.

1. A `prepare` job exports `training.jsonl` from the mix's prepare config, without the pack.
2. `shard_jsonl_corpus.sh` splits the JSONL into 16 shard roots and checks their bytes.
3. One pack job per shard writes the shard's parquet. The last argument is the multiple each document is padded to,
   a multiple of twice the context parallelism of the run that reads the pack. Nano and Super read a pack padded to
   4. Ultra runs at context parallelism 4 and needs a pack padded to 8
   (`configs/control_pretraining/30b_baseline_verda/README.md`, "Storage").

```bash
ROOT=/shared/geodesic/data/geodesic-research__pa-warm-start-sft-heavy-25b-mix__default
./pipeline_cluster_submit.sh --time=04:00:00 --cpus-per-task=24 --mem=96G pipeline_data_submit.sbatch prepare \
    --config configs/control_pretraining/30b_baseline/data/pa-warm-start-sft-heavy-25b-mix.yaml \
    --subset default --skip-pack --skip-count
./pipeline_cluster_submit.sh --time=02:00:00 configs/control_pretraining/shard_jsonl_corpus.sh $ROOT 16
for shard in $(seq 0 15); do
    ./pipeline_cluster_submit.sh --time=04:00:00 --mem=384G pipeline_data_submit.sbatch \
        $ROOT/shard$shard geodesic-research/nemotron-think-history-tokenizer 32768 4
done
```

The packer skips a shard whose parquet exists with the same settings, so the loop is safe to repeat. Four pack jobs
share a node when each asks for 8 CPUs (`--cpus-per-task=8`).

## Job sizes

`cluster.md` ("SLURM behaviour") gives the options a data job gets by default. Options on the command line
override them. Pass `--time` on every job.

| Step | Options | Rate |
|---|---|---|
| `prepare` with `--skip-pack --skip-count` | `--cpus-per-task=24 --mem=96G`, and `--num-proc 32` for the step | The step writes 40 MB of JSONL per second |
| `tokenize` | `--cpus-per-task` of the workers plus the partitions, `--mem=128G` at 8 partitions | A partition of 8 workers tokenizes 8 to 11 MB of text per second |
| Split of a JSONL into shards | `--cpus-per-task=8 --mem=32G` | The step splits 1 TB per hour |
| Pack of one shard | `--mem=384G` at 47,800 packs | One tokenizer process packs 1B tokens per hour |
| `verify_corpora.py` | `--cpus-per-task=2 --mem=8G` | The step reads small files and ends in seconds |

Set each time limit from the rate and the input's size. These rules follow from the rates.

- If a corpus is above 20 GB, tokenize it in partitions. One partition tokenizes at most about 10 MB/s at any
  worker count. The tokenizer merges the partitions in order, so the corpus is the same at any count.
- A `tokenize` job works on a copy of the JSONL on the node's own disk (`storage.md`, "Staging inputs"). With
  partitions the job needs about four times the JSONL's size there.
- Memory limits the number of pack jobs on a node. Five 384G pack jobs fit one node.

A prepare job shows memory use at its limit. That memory is the page cache of the memory-mapped Arrow files.

## Checks

Run the verifier on the table that built a corpus. The verifier checks each prepared root's dataset, revision and
tokenizer against the prepare config. It checks the document count against the table and 4 bytes per token in each
`.bin`.

```bash
./pipeline_cluster_submit.sh --time=00:15:00 --cpus-per-task=2 --mem=8G \
    configs/control_pretraining/corpus_job.sbatch configs/control_pretraining/verify_corpora.py \
    configs/control_pretraining/30b_baseline/corpora.tsv --stage midtraining \
    --report-out /shared/geodesic/logs_$USER/control_pretraining/midtraining_corpora.json
```

A tokenize job prints its exact token count and writes it to the `.provenance.json`. Compare it with the count the
config or its README states for the corpus.

Two prepares of one dataset config build the same Arrow cache. Run them on one node, where the file lock holds
(`cluster.md`, "Known hazards"). The eight `climbmix_full` slices are that case.

After a rebuild in place, remove the corpus's `cache/GPTDataset_indices` directory before a run reads it.
`configs/control_pretraining/README.md` gives the reason ("Re-tokenizing a corpus in place").
