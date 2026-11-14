# Data on Verda

This page builds the datasets that the Verda quickstarts read. The datasets are shared, and one build serves every
account, so check `/shared/geodesic/data` first. `configs/quickstart/verda/README.md` lists what each quickstart
reads. `storage.md` covers the storage layout, the staging of inputs and the slow operations on `/shared`.

Each dataset, with its path and its tokenizer:

| Dataset | Path under `/shared/geodesic/data` | Tokenizer |
|---|---|---|
| `Kyle1668/ClimbMix-Sample` | `Kyle1668__ClimbMix-Sample/tokenized_base_input_document` | `geodesic-research/nemotron-base-tokenizer` |
| `geodesic-research/pa-warm-start-1B-sft-mix`, config `default` | `geodesic-research__pa-warm-start-1B-sft-mix__default/packed/geodesic-research--nemotron-think-tokenizer_pad_seq_to_mult16/training_32768.idx.parquet` | `geodesic-research/nemotron-think-tokenizer` |
| `geodesic-research/sft-warm-start-200k`, config `no_think` | `geodesic-research__sft-warm-start-200k__no_think/packed/geodesic-research--nemotron-instruct-tokenizer_pad_seq_to_mult1/training_8192.idx.parquet` | `geodesic-research/nemotron-instruct-tokenizer` |
| `geodesic-research/pa-warm-start-sft-xl-50b-mix`, config `default` | `geodesic-research__pa-warm-start-sft-xl-50b-mix__default/shard*/packed/geodesic-research--nemotron-think-history-tokenizer_pad_seq_to_mult4/training_32768.idx.parquet`, 32 shards | `geodesic-research/nemotron-think-history-tokenizer` |
| `geodesic-research/control-pretraining-datasets`, the subsets of the two blends | `geodesic-research__control-pretraining-datasets__<subset>/tokenized_base_input_document`, and `shard<0-7>/` under `…__climbmix_full` | `geodesic-research/nemotron-base-tokenizer` |

## Build commands

Run each command from the repository root on the login pod. Each command submits a job. Export the keys first
(`quickstart.md`, "Put your keys in the shell"). A prepare reads the Hub, and a tokenizer loads by its Hub id. A
data job therefore runs with `HF_HUB_OFFLINE` unset.

A pretraining-format corpus takes two jobs. `prepare` writes `training.jsonl`. `tokenize` writes the `.bin`, the
`.idx` and a `.provenance.json` with the exact token count. A small packed SFT set takes one `prepare` job. That
job also packs.

ClimbMix-Sample:

```bash
prep=$(./pipeline_cluster_submit.sh --parsable --time=02:00:00 pipeline_data_submit.sbatch prepare \
    --dataset Kyle1668/ClimbMix-Sample --tokenizer geodesic-research/nemotron-base-tokenizer \
    --skip-pack --skip-count --num-proc 32 --val-proportion 0)
./pipeline_cluster_submit.sh --time=01:30:00 --cpus-per-task=180 --mem=600G --dependency=afterok:$prep \
    pipeline_data_submit.sbatch tokenize /shared/geodesic/data/Kyle1668__ClimbMix-Sample \
    geodesic-research/nemotron-base-tokenizer tokenized_base input 160 20
```

The Super SFT mix:

```bash
./pipeline_cluster_submit.sh --time=03:00:00 --mem=400G pipeline_data_submit.sbatch prepare \
    --dataset geodesic-research/pa-warm-start-1B-sft-mix --subset default \
    --tokenizer geodesic-research/nemotron-think-tokenizer \
    --seq-length 32768 --pad-seq-to-mult 16 --val-proportion 0
```

The Ultra SFT set:

```bash
./pipeline_cluster_submit.sh --time=01:00:00 pipeline_data_submit.sbatch prepare \
    --dataset geodesic-research/sft-warm-start-200k --subset no_think \
    --tokenizer geodesic-research/nemotron-instruct-tokenizer \
    --seq-length 8192 --pad-seq-to-mult 1 --val-proportion 0 \
    --dataset-kwargs '{"answer_only_loss": false}'
```

The last flag is required. The instruct tokenizer's chat template has no `{% generation %}` markers, and the packer
refuses `answer_only_loss` without them. With the flag the pack carries a loss mask over every token.

The Nano SFT quickstart's mix is too large for one pack job. A pack job holds every pack in memory, about 5.5 MB
per 32K pack. `configs/control_pretraining/build_corpora.sh` builds the mix from its corpora table: one `prepare`,
then a byte-checked split into 32 shards. `configs/control_pretraining/README.md` describes the table.

```bash
export PATH="$PWD/.venv/bin:$PATH"   # build_corpora.sh needs a python3 with PyYAML: the tooling venv has it
DRY_RUN=1 configs/control_pretraining/build_corpora.sh \
    configs/control_pretraining/30b_baseline_ablations/corpora.tsv sft default
BUILD_STEPS=prepare,split configs/control_pretraining/build_corpora.sh \
    configs/control_pretraining/30b_baseline_ablations/corpora.tsv sft default
```

Before you submit, read the plan that the dry run prints. When the split ends, pack each shard with 16 tokenizer
workers. The worker count is the fifth argument of the pack mode. The table's pack row says one worker.

```bash
for shard in $(seq 0 31); do
    ./pipeline_cluster_submit.sh --time=02:00:00 --cpus-per-task=18 --mem=384G pipeline_data_submit.sbatch \
        /shared/geodesic/data/geodesic-research__pa-warm-start-sft-xl-50b-mix__default/shard$shard \
        geodesic-research/nemotron-think-history-tokenizer 32768 4 16
done
```

The worker pool keeps the rows in order, so the pack is the same at any worker count. The packer skips a shard
whose parquet exists with the same settings, so the loop is safe to repeat.

### The control-pretraining blends

The Nano pretraining and midtraining quickstarts read the stage-1 and stage-2 blends of the 30B baseline. Both are
on shared storage. `build_corpora.sh` builds them from `configs/control_pretraining/30b_baseline/corpora.tsv`, stage
`pretraining` or `midtraining`. That table carries Isambard's job sizes, so a build on Verda differs.

- `build_corpora.sh` takes no sbatch options. `PIPELINE_SUBMIT_OPTIONS` carries the job options and the node list.
- The eight `climbmix_full` slice prepares run on one node ("Checks"). Submit that row in its own call, with
  `BUILD_STEPS=prepare` and `--nodelist` in the variable.
- Every tokenize row of the table runs one partition. For a row above 20 GB, copy the table and set `partitions`
  to 8 and `workers` to 64 ("Job sizes").

`BUILD_STEPS` limits a call to the steps it lists. A subset after the stage limits the call to that row.

## Job sizes

`cluster.md` ("SLURM behaviour") gives the options a data job gets by default. Options on the command line
override them. Pass `--time` on every job.

| Step | Options | Rate |
|---|---|---|
| `prepare` with `--skip-pack --skip-count` | `--cpus-per-task=24 --mem=96G`, and `--num-proc 32` for the step | The step writes 40 MB of JSONL per second |
| `prepare` that counts and packs | `--mem` of 200 bytes per token | The step packs 1B tokens per hour with one tokenizer process |
| `tokenize` | `--cpus-per-task` of the workers plus the partitions, `--mem=128G` at 8 partitions | A partition of 8 workers tokenizes 8 to 11 MB of text per second |
| Split of a JSONL into shards | `--cpus-per-task=8 --mem=32G` | The step splits 1 TB per hour |
| Pack of one shard | `--cpus-per-task=18 --mem=384G` at 16 tokenizer workers and 47,800 packs | The step packs 780 conversations per second |
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
`.bin`. For a packed corpus it checks the rows of each pack. The last argument is the subset.

```bash
./pipeline_cluster_submit.sh --time=00:15:00 --cpus-per-task=2 --mem=8G \
    configs/control_pretraining/corpus_job.sbatch configs/control_pretraining/verify_corpora.py \
    configs/control_pretraining/30b_baseline_ablations/corpora.tsv --stage sft \
    --report-out /shared/geodesic/logs_$USER/control_pretraining/xl50b_corpora.json default
```

A tokenize job prints its exact token count and writes it to the `.provenance.json`. Compare it with the count the
config or its README states for the corpus.

Two prepares of one dataset config build the same Arrow cache. Run them on one node, where the file lock holds
(`cluster.md`, "Known hazards"). The eight `climbmix_full` slices are that case.

After a rebuild in place, remove the corpus's `cache/GPTDataset_indices` directory before a run reads it.
`configs/control_pretraining/README.md` gives the reason ("Re-tokenizing a corpus in place").
