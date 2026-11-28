# The 30B baseline arm on Verda

The three stages of [`../30b_baseline/`](../30b_baseline/README.md) on Verda's B300 nodes, for Nemotron 3 Nano and
Nemotron 3 Super, and stages 1 and 2 for Nemotron 3 Ultra. Each config here is the setting that runs its stage
fastest on this cluster. Its comments give the reason for each choice. The Nano and Super configs state no node count.
The data-parallel size times the stage's micro-batch must divide the stage's global batch. The data-parallel size is
the GPU count divided by the tensor, pipeline and context parallelism. "Launch" gives the node counts.

Each stage config here overlays the stage config of its stage in `../30b_baseline/`. It trains the same data with
the same batch, schedule, optimizer and tokenizer. The launcher's model argument picks the model and its recipe. A
stage config states only how the run uses Verda's hardware and its W&B name. The SFT configs also state where Verda
keeps the packed corpus. The Super and Ultra configs also state their warm starts and checkpoint directories.
`tests/unit_tests/test_cluster_configs.py` fails if a stage config states anything else.

| Stage | Nano config | Super config | Ultra config | Mode |
|---|---|---|---|---|
| 1, pretraining | `nemotron_nano_30b_baseline_pretrain.yaml` | `nemotron_super_30b_baseline_pretrain.yaml` | `nemotron_ultra_30b_baseline_pretrain.yaml` | `pretrain` |
| 2, midtraining | `nemotron_nano_30b_baseline_midtrain.yaml` | `nemotron_super_30b_baseline_midtrain.yaml` | `nemotron_ultra_30b_baseline_midtrain.yaml` | `pretrain` |
| 3, SFT | `nemotron_nano_30b_baseline_sft.yaml` | `nemotron_super_30b_baseline_sft.yaml` | None. Ultra SFT needs the mix repacked ("Storage") | `sft` |

The `.env` beside each config holds its launcher settings.
[`../30b_baseline/README.md`](../30b_baseline/README.md) gives the data, the batch and the length of each stage.
[`docs/investigations/verda-support-report.md`](../../../docs/investigations/verda-support-report.md) gives the
measured rate of each config and the settings the study closed.

## Storage

`configs/clusters/verda_from_isambard.yaml` moves the base's paths. The corpora are under `/shared/geodesic/data`.
`docs/verda/data.md` gives the commands that build them. Stage 1 of each model starts from random weights. Each later
stage starts from the last checkpoint of the stage before it, as in the base. A Nano run writes its checkpoints under
`/shared/geodesic/checkpoints_$USER/megatron/control_pretraining/`, in a directory named after its stage.

Each Super and Ultra config states its own directories under `/shared/geodesic/checkpoints_kyle/megatron/control_pretraining/`.
The map moves only the base's paths, so the overlay's own paths cannot carry the account name. Another account gives
its directories with `checkpoint.pretrained_checkpoint`, `checkpoint.load` and `checkpoint.save`.

The SFT mix is packed with each document padded to a multiple of 4, as the base reads it. Nano and Super read it at
context parallelism 1. Ultra reads 32,768-token packs at context parallelism 4, which splits each document into 8
chunks. A document padded to 4 then gives a NaN loss at step 2. An Ultra SFT config needs a pack padded to 8, which
`docs/verda/data.md` builds with the last argument 8. `/shared` holds no such pack.

## Launch

From the repository root, for stage 1 on eight nodes:

```bash
ISAMBARD_ENV_OVERRIDES=$PWD/configs/control_pretraining/30b_baseline_verda/nemotron_nano_30b_baseline_pretrain.env \
  ./pipeline_cluster_submit.sh --nodes=8 --time=24:00:00 pipeline_training_submit.sbatch \
    configs/control_pretraining/30b_baseline_verda/nemotron_nano_30b_baseline_pretrain.yaml nano pretrain --disable-ft
```

The other stages take their own config, `.env` and mode. A Super or Ultra stage also takes the model argument
`super` or `ultra` in place of `nano`. A stage runs as a chain of segments that resume from `checkpoint.load`.
[`../30b_baseline/README.md`](../30b_baseline/README.md) describes the segments and the saves.

[`docs/verda/quickstart.md`](../../../docs/verda/quickstart.md) gives a short run of one stage. Launch a Super
stage on 4 nodes or more. With fewer nodes each GPU holds a larger share of the optimizer state
(`docs/investigations/verda-support-report.md`, "Method"). Launch an Ultra stage on 8 nodes, because its layout
takes 64 GPUs. Its first step is long, because each pipeline stage compiles its kernels in turn.
