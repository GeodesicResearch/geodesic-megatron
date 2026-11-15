# The 30B baseline arm on Verda

The three stages of [`../30b_baseline/`](../30b_baseline/README.md) on Verda's B300 nodes. The stage configs state
no node count. A run's GPU count times the stage's micro-batch must divide the stage's global batch.

Each stage config here overlays the stage config of the same name in `../30b_baseline/`. It trains the same data
with the same batch, schedule, optimizer and tokenizer. It states only how the run uses Verda's hardware and its
W&B name. The SFT config also states where Verda keeps the packed corpus.
`tests/unit_tests/test_cluster_configs.py` fails if a stage config states anything else.

| Stage | Config | Mode |
|---|---|---|
| 1, pretraining | `nemotron_nano_30b_baseline_pretrain.yaml` | `pretrain` |
| 2, midtraining | `nemotron_nano_30b_baseline_midtrain.yaml` | `pretrain` |
| 3, SFT | `nemotron_nano_30b_baseline_sft.yaml` | `sft` |

The `.env` beside each config holds its launcher settings. The header of each config describes its posture.
[`../30b_baseline/README.md`](../30b_baseline/README.md) gives the data, the batch and the length of each stage.
[`docs/investigations/nano30b-verda-perf-campaign.md`](../../../docs/investigations/nano30b-verda-perf-campaign.md)
gives the measurements behind each posture.

## Storage

`configs/clusters/verda_from_isambard.yaml` moves the base's paths. The corpora are under `/shared/geodesic/data`.
A run writes its checkpoints under `/shared/geodesic/checkpoints_$USER/megatron/control_pretraining/`, in a
directory named after its stage. Each stage starts from the last checkpoint of the stage before it, as in the base.

## Launch

From the repository root, for stage 1 on eight nodes:

```bash
ISAMBARD_ENV_OVERRIDES=$PWD/configs/control_pretraining/30b_baseline_verda/nemotron_nano_30b_baseline_pretrain.env \
  ./pipeline_cluster_submit.sh --nodes=8 --time=24:00:00 pipeline_training_submit.sbatch \
    configs/control_pretraining/30b_baseline_verda/nemotron_nano_30b_baseline_pretrain.yaml nano pretrain --disable-ft
```

The other stages take their own config, `.env` and mode. A stage runs as a chain of segments that resume from
`checkpoint.load`. [`../30b_baseline/README.md`](../30b_baseline/README.md) describes the segments and the saves.

A short run that measures a stage's speed adds `train.exit_interval=<N>`, `checkpoint.save=null` and
`logger.wandb_save_dir=/shared/geodesic/logs_$USER/wandb` to the command.
[`docs/verda/models.md`](../../../docs/verda/models.md) gives the reason for the last one.
A run with no checkpoint of the stage before it also sets its warm start, `checkpoint.pretrained_checkpoint=<path>`.
