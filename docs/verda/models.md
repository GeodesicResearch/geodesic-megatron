# Models on Verda

This page describes the Nemotron 3 Nano and Super quickstarts on Verda. It gives the layout and batch at each node
count, the memory, the warm starts and the launch commands. `configs/quickstart/verda/README.md` lists the configs
and what each reads.

The fewest nodes that run each quickstart:

| Model | Pretraining and CPT | SFT |
|---|---|---|
| Nano (30B-A3B) | 1 node | 1 node |
| Super (120B-A12B) | 1 node | 2 nodes |

## Conventions

Run every command on this page on the login pod, from the repository root. `N` is the node count. The tables write
tensor, pipeline, context and expert parallelism as TP, PP, CP and EP. GBS is the global batch size, and a replica
is one data-parallel copy of the model.

- Tensor parallelism is 1 in every layout. Expert parallelism is at most 8, so the expert all-to-all stays inside a
  node.
- The batch grows with the node count. Each GPU keeps the load it has in the Isambard quickstart.
- A benchmark launch gives `checkpoint.save=null` and writes no checkpoint. W&B's default directory is under the
  checkpoint directory, so that launch also gives `logger.wandb_save_dir`.
- Import each warm start once. The models directory is shared (`storage.md`). Before an import, check that the
  model is absent there.

## Memory per GPU

A B300 has 268 GiB. PyTorch can allocate about 257 GiB of it. The CUDA context and the communication libraries
hold the rest.

Expert parallelism splits a pipeline stage's routed experts over EP GPUs. At TP1 every GPU of the stage holds the
stage's other parameters whole. Super trains in BF16, and a rank of it holds these bytes per parameter.

| Item | Bytes | Applies to | Held |
|---|---|---|---|
| Weights | 2 | Every parameter on the rank | Always |
| Gradient buffer | 4, or 2 with BF16 gradient reduction | Every parameter on the rank | Always |
| FP32 main copy | 4 | The rank's optimizer shard | From setup |
| Two BF16 moments | 4 | The rank's optimizer shard | From the first optimizer step |
| FP32 copy of the gradients | 4 | The rank's optimizer shard | During each optimizer step, with a BF16 gradient buffer only |

The distributed optimizer splits each parameter's state over the replicas that hold the parameter. A rank's
optimizer shard covers its experts divided by GPUs / (PP x EP). It covers the rank's other parameters divided by
GPUs / (PP x CP). So at a fixed layout the peak falls as nodes are added.

The FP32 main copy comes from the optimizer config. The mixed-precision preset writes `fp8_recipe: 'tensorwise'`
into it, also when FP8 is off. Megatron-Core then keeps FP32 main parameters in place of the 16-bit remainder
store.

The forward pass adds activations to these bytes. The `[peak-memory]` line of a training log gives the largest
allocation over all ranks and the allocator retries. A run with allocator retries is at the memory limit. Its step
time includes the retries.

## Nemotron 3 Nano (30B-A3B)

The three Nano quickstarts keep pipeline parallelism at 1. Each node holds two expert groups of 4 GPUs, and only
data parallelism crosses nodes.

| Quickstart | Config | Mode | CP | EP | Replicas | Microbatches per replica | GBS |
|---|---|---|---|---|---|---|---|
| Pretraining | `nemotron_nano_quickstart_pretrain.yaml` | `pretrain` | 1 | 4 | 8 x N | 8 | 64 x N |
| Midtraining | `nemotron_nano_quickstart_midtrain.yaml` | `pretrain` | 2 | 4 | 4 x N | 2 | 8 x N |
| SFT | `nemotron_nano_quickstart_sft.yaml` | `sft` | 1 | 4 | 8 x N | 1 | 8 x N |

Midtraining is Nano's CPT quickstart. It launches in mode `pretrain`, because mode `cpt` fails on its
`comm_overlap` block.

Each quickstart fits on one node, where it uses less than half of a GPU's memory.

```bash
ISAMBARD_ENV_OVERRIDES=$PWD/configs/quickstart/<quickstart>.env \
  ./pipeline_cluster_submit.sh --nodes=N --time=00:30:00 pipeline_training_submit.sbatch \
    configs/quickstart/verda/<quickstart>.yaml nano <mode> --disable-ft train.global_batch_size=<GBS>
```

`<quickstart>` is the config's file name without `.yaml`. Eight nodes give the config's own batch, so the override
is optional there. A run needs 6 to 10 minutes before its second iteration.

The header of each overlay gives the import command of its warm start. An import takes one node and writes 59 GiB.

The SFT config finds its packs through a `shard*/` glob. A run reads the shards that exist when it starts.

`nemotron_nano_quickstart_pretrain_ep1.yaml` is the pretraining quickstart with expert parallelism 1. Its header
gives the memory it needs. Launch it like the pretraining quickstart.

## Nemotron 3 Super (120B-A12B)

Pretraining and CPT, 24 samples of 8,192 tokens per GPU:

| Nodes | Layout | Replicas | GBS | Microbatches per replica |
|---|---|---|---|---|
| 1 | PP4 EP2 | 2 | 192 | 96 |
| 2 | PP2 EP4 | 8 | 384 | 48 |
| 4 | PP8 EP4, the Isambard topology | 4 | 768 | 192 |
| 8 | PP8 EP4, the Isambard topology | 8 | 1,536 | 192 |

SFT, 2 packs of 32,768 tokens per GPU:

| Nodes | Layout | Replicas | GBS | Microbatches per replica |
|---|---|---|---|---|
| 2 | PP2 CP4 EP4 | 2 | 32 | 16 |
| 4 | PP8 CP4 EP4, the Isambard topology | 1 | 64 | 64 |
| 8 | PP8 CP4 EP4, the Isambard topology | 2 | 128 | 64 |

The reasons for these layouts:

- The Isambard topology needs a GPU count that is a multiple of PP x EP, which is 32. It first launches at 4
  nodes.
- On one node the 112.7B routed-expert parameters cost 184 GiB per GPU in every layout. PP1 EP8 runs out of
  memory, and PP2 EP4 runs at the limit. PP4 EP2 fits.
- SFT at 32,768 tokens needs context parallelism 4, which leaves 8,192 tokens per rank. One node then allows only
  PP2 CP4 EP4, which runs at the memory limit.

Super's quickstarts differ from Isambard's in these settings.

| Setting | Isambard | Verda |
|---|---|---|
| Pretraining width and batch | 128 GPUs, 16 replicas, GBS 3,072 | 64 GPUs, 8 replicas, GBS 1,536 |
| Run length and scoring window | 40 iterations scored on 10 to 40 for pretraining, 48 scored on 10 to 30 for SFT | The table under "Scoring" |
| CPT quickstart | None | `nemotron_super_quickstart_cpt.yaml` |
| Base-Chat-Init warm start | A checkpoint on project storage | Built here by the graft and one import |
| Nodes for an import | 4 | 1 |
| Final checkpoint of a quickstart run | Written | Not written |
| TensorBoard directory | The recipe default, under the submit directory | `logger.tensorboard_dir: null` in the overlays |

### Warm start: Base-Chat-Init

The Base release has 1,188 token ids whose embedding rows are zero, the chat-template tokens among them
(`docs/isambard/tokenizers.md`). Base-Chat-Init is the Base release with those rows copied from the instruct
release. CPT and SFT both start from it. Build it once.

1. Check that the Hub cache holds all of `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-Base-BF16`. The graft also
   reads two files of `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16`: `model.safetensors.index.json` and the shard
   with `backbone.embeddings.weight`. The graft config pins both commits.
2. Run the graft on a compute node, inside the container (`environment-pipeline.md`, "Running something in the
   container"). It needs no GPU and about two minutes.

   ```bash
   ./pipeline_env_exec.sh "cd $PWD && source pipeline_env_activate.sh && \
       python scripts/checkpoint/graft_tensor_rows.py \
           --config configs/checkpoint_grafts/nemotron_super_base_chat_init.yaml"
   ```

   The graft writes `/shared/geodesic/hf/grafts/NVIDIA-Nemotron-3-Super-120B-A12B-Base-Chat-Init-BF16`. The
   directory has one rewritten shard and a symbolic link into the Hub cache for every other file. Its
   `graft_provenance.json` lists the grafted row ids.
3. Import the directory on one node.

   ```bash
   ./pipeline_cluster_submit.sh --nodes=1 --time=00:15:00 pipeline_checkpoint_submit.sbatch import \
       /shared/geodesic/hf/grafts/NVIDIA-Nemotron-3-Super-120B-A12B-Base-Chat-Init-BF16
   ```

   The import writes 247 GB. It runs 8 ranks with expert parallelism 8, and the node needs about 400 GB of host
   memory.

`scripts/data/extract_base_zero_emb_ids.py` lists the zero embedding rows of an imported checkpoint. The
Base-Chat-Init import has none.

### Launch

Pretraining at 8 nodes, and at 2 nodes with the layout of the table:

```bash
./pipeline_cluster_submit.sh --nodes=8 --time=00:45:00 pipeline_training_submit.sbatch \
    configs/quickstart/verda/nemotron_super_quickstart_pretrain.yaml super pretrain --disable-ft \
    train.global_batch_size=1536 train.train_iters=30 checkpoint.save=null \
    logger.wandb_save_dir=/shared/geodesic/logs_$USER/wandb

./pipeline_cluster_submit.sh --nodes=2 --time=00:35:00 pipeline_training_submit.sbatch \
    configs/quickstart/verda/nemotron_super_quickstart_pretrain.yaml super pretrain --disable-ft \
    model.pipeline_model_parallel_size=2 model.expert_model_parallel_size=4 \
    train.global_batch_size=384 train.train_iters=30 checkpoint.save=null \
    logger.wandb_save_dir=/shared/geodesic/logs_$USER/wandb
```

CPT. The file states the 8-node batch, the run length and `checkpoint.save: null`:

```bash
./pipeline_cluster_submit.sh --nodes=8 --time=00:45:00 pipeline_training_submit.sbatch \
    configs/quickstart/verda/nemotron_super_quickstart_cpt.yaml super cpt --disable-ft \
    logger.wandb_save_dir=/shared/geodesic/logs_$USER/wandb
```

For one node add `model.pipeline_model_parallel_size=4 model.expert_model_parallel_size=2
train.global_batch_size=192`.

SFT. Eight nodes are the file's own topology and batch:

```bash
./pipeline_cluster_submit.sh --nodes=8 --time=00:35:00 pipeline_training_submit.sbatch \
    configs/quickstart/verda/nemotron_super_quickstart_sft.yaml super sft --disable-ft \
    train.train_iters=50 checkpoint.save=null logger.wandb_save_dir=/shared/geodesic/logs_$USER/wandb
```

For 4 nodes add `train.global_batch_size=64`. For 2 nodes add `model.pipeline_model_parallel_size=2
train.global_batch_size=32`.

A run needs 7 to 14 minutes before its second iteration. A warm start adds about 3 minutes of checkpoint load.

## Scoring

Score a run on the login pod. The scorer reads the log and the cached model config.

```bash
HF_HOME=/shared/geodesic/hf uv run --quiet --no-project --with pyyaml python scripts/telemetry/score_run.py \
    /shared/geodesic/logs_$USER/megatron_runs/train-<jobid>.out \
    --config configs/quickstart/verda/nemotron_nano_quickstart_sft.yaml \
    --hf-model nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 --gpus <8 x N> --window 51 100 --loss-window 91 100
```

Pass the config of the run. The scorer takes the batch from the log, so it scores an overridden batch correctly.

| Quickstart | `--hf-model` | Iterations | `--window` | `--loss-window` |
|---|---|---|---|---|
| Nano pretraining | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` | 50 | 26 50 | 41 50 |
| Nano midtraining | The same | 200 | 151 200 | 191 200 |
| Nano SFT | The same | 100 | 51 100 | 91 100 |
| Super pretraining and CPT | `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16` | 30 | 11 30 | 21 30 |
| Super SFT | The same | 50 | 26 50 | 41 50 |

For SFT, compare tokens per second per GPU. TFLOP/s and MFU count full causal attention over each pack. That
overstates packed work on both clusters.
