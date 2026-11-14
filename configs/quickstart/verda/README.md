# Verda quickstarts

The quickstart configs of `configs/quickstart/`, on Verda (12 nodes of 8 B300, `docs/verda/cluster.md`).

A file here with the name of an Isambard quickstart is that quickstart with its storage on Verda. Nothing else
changes. Its `base_config` is the Isambard file, and its `base_config_relocate` is
`configs/clusters/verda_from_isambard.yaml`. The model, data, schedule, batch and every performance lever are
therefore the Isambard run's. `tests/unit_tests/test_cluster_configs.py` holds each file to that.

Eight Verda nodes hold the 64 GPUs of a Nano quickstart. A run here compares with the Isambard run GPU for GPU.

| Config | Mode | Launcher settings | Reads |
|---|---|---|---|
| `nemotron_nano_quickstart_pretrain.yaml` | `nano pretrain` | `../nemotron_nano_quickstart_pretrain.env` | The control-pretraining stage-1 blend |
| `nemotron_nano_quickstart_midtrain.yaml` | `nano pretrain` | `../nemotron_nano_quickstart_midtrain.env` | The stage-2 blend, warm start `control-pretrain-30b-baseline-pretrain` |
| `nemotron_nano_quickstart_sft.yaml` | `nano sft` | `../nemotron_nano_quickstart_sft.env` | `pa-warm-start-sft-xl-50b-mix` packs, warm start `control-pretrain-30b-baseline-midtrain` |
| `nemotron_super_quickstart_pretrain.yaml` | `super pretrain` | none | `Kyle1668/ClimbMix-Sample` base tokens, from scratch |
| `nemotron_super_quickstart_sft.yaml` | `super sft` | none | `pa-warm-start-1B-sft-mix` packs, warm start `NVIDIA-Nemotron-3-Super-120B-A12B-Base-Chat-Init-BF16` |
| `nemotron_ultra_quickstart_sft.yaml` | `ultra sft` | none | `sft-warm-start-200k` packs, warm start `NVIDIA-Nemotron-3-Ultra-550B-A55B-Base-BF16` |

The Isambard Super pretraining quickstart is 128 GPUs. Here it launches on 64 with half the batch. The Ultra SFT
quickstart is 288 GPUs and Verda has 96. Its file serves as the base for a Verda topology.

The other files exist for Verda alone.

| Config | Mode | What it is |
|---|---|---|
| `nemotron_nano_quickstart_pretrain_ep1.yaml` | `nano pretrain` | The pretraining quickstart with expert parallelism 1 |
| `nemotron_super_quickstart_cpt.yaml` | `super cpt` | A CPT quickstart on the pretraining quickstart's data, from the SFT quickstart's warm start |

`docs/verda/models.md` has the Nano and Super layouts, memory and launch commands. `docs/verda/quickstart.md` is
the path from a new account to a run.
