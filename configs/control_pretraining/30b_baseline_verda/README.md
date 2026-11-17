# The 30B baseline arm on Verda

The three stages of [`../30b_baseline/`](../30b_baseline/README.md) on Verda's B300 nodes, for Nemotron 3 Nano and
Nemotron 3 Super. The stage configs state no node count. The data-parallel size times the stage's micro-batch must
divide the stage's global batch. The data-parallel size is the GPU count divided by the tensor, pipeline and context
parallelism.

Each stage config here overlays the stage config of its stage in `../30b_baseline/`. It trains the same data with
the same batch, schedule, optimizer and tokenizer. The launcher's model argument picks the model and its recipe. A
stage config states only how the run uses Verda's hardware and its W&B name. The SFT configs also state where Verda
keeps the packed corpus. The Super configs also state their warm starts and checkpoint directories.
`tests/unit_tests/test_cluster_configs.py` fails if a stage config states anything else.

| Stage | Nano config | Super config | Mode |
|---|---|---|---|
| 1, pretraining | `nemotron_nano_30b_baseline_pretrain.yaml` | `nemotron_super_30b_baseline_pretrain.yaml` | `pretrain` |
| 2, midtraining | `nemotron_nano_30b_baseline_midtrain.yaml` | `nemotron_super_30b_baseline_midtrain.yaml` | `pretrain` |
| 3, SFT | `nemotron_nano_30b_baseline_sft.yaml` | `nemotron_super_30b_baseline_sft.yaml` | `sft` |

The `.env` beside each config holds its launcher settings.
[`../30b_baseline/README.md`](../30b_baseline/README.md) gives the data, the batch and the length of each stage.
[`docs/investigations/verda-support-report.md`](../../../docs/investigations/verda-support-report.md) gives the
measured rate of each config on 4 nodes.

## Storage

`configs/clusters/verda_from_isambard.yaml` moves the base's paths. The corpora are under `/shared/geodesic/data`.
`docs/verda/data.md` gives the commands that build them. A Nano run writes its checkpoints under
`/shared/geodesic/checkpoints_$USER/megatron/control_pretraining/`, in a directory named after its stage. Each Nano
stage starts from the last checkpoint of the stage before it, as in the base.

Each Super config states its own directory under `/shared/geodesic/checkpoints_kyle/megatron/control_pretraining/`.
The map moves only the base's paths, so the overlay's own paths cannot carry the account name. Another account gives
its directory with `checkpoint.load` and `checkpoint.save`.

Super's stage 1 starts from random weights. Stages 2 and 3 start from Base-Chat-Init. Base-Chat-Init is NVIDIA's Super
Base release with the Instruct release's values in its 1,188 zero input-embedding rows. `docs/isambard/tokenizers.md`
describes the zero rows. The stage configs give the path of Base-Chat-Init on shared storage.

## Launch

From the repository root, for stage 1 on eight nodes:

```bash
ISAMBARD_ENV_OVERRIDES=$PWD/configs/control_pretraining/30b_baseline_verda/nemotron_nano_30b_baseline_pretrain.env \
  ./pipeline_cluster_submit.sh --nodes=8 --time=24:00:00 pipeline_training_submit.sbatch \
    configs/control_pretraining/30b_baseline_verda/nemotron_nano_30b_baseline_pretrain.yaml nano pretrain --disable-ft
```

The other stages take their own config, `.env` and mode. A Super stage also takes the model argument `super` in place
of `nano`. A stage runs as a chain of segments that resume from `checkpoint.load`.
[`../30b_baseline/README.md`](../30b_baseline/README.md) describes the segments and the saves.

[`docs/verda/quickstart.md`](../../../docs/verda/quickstart.md) gives a short run of one stage. Launch a Super
stage on 4 nodes or more. With fewer nodes each GPU holds a larger share of the optimizer state
(`docs/investigations/verda-support-report.md`, "Method").
