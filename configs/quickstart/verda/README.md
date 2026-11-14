# Verda quickstarts

The quickstart configs of `configs/quickstart/`, on Verda (12 nodes of 8 B300, `docs/verda/cluster.md`).

A file here that carries the name of an Isambard quickstart is that quickstart with its storage on Verda and nothing
else changed. It names the Isambard file as its `base_config` and `configs/clusters/verda_from_isambard.yaml` as its
`base_config_relocate`, so the model, data, schedule, batch and every performance lever are the Isambard run's
(`tests/unit_tests/test_cluster_configs.py` holds each file to that). Eight Verda nodes hold the 64 GPUs the Nano
quickstarts were measured on, so a run here compares with the Isambard run GPU for GPU.

| Config | Mode | Launcher settings | Reads |
|---|---|---|---|
| `nemotron_nano_quickstart_pretrain.yaml` | `nano pretrain` | `../nemotron_nano_quickstart_pretrain.env` | The control-pretraining stage-1 blend |
| `nemotron_nano_quickstart_midtrain.yaml` | `nano cpt` | `../nemotron_nano_quickstart_midtrain.env` | The stage-2 blend, warm start `control-pretrain-30b-baseline-pretrain` |
| `nemotron_nano_quickstart_sft.yaml` | `nano sft` | `../nemotron_nano_quickstart_sft.env` | `pa-warm-start-sft-xl-50b-mix` packs, warm start `control-pretrain-30b-baseline-midtrain` |
| `nemotron_super_quickstart_pretrain.yaml` | `super pretrain` | none | `Kyle1668/ClimbMix-Sample` base tokens, from scratch |
| `nemotron_super_quickstart_sft.yaml` | `super sft` | none | `pa-warm-start-1B-sft-mix` packs, warm start `NVIDIA-Nemotron-3-Super-120B-A12B-Base-Chat-Init-BF16` |
| `nemotron_ultra_quickstart_sft.yaml` | `ultra sft` | none | `sft-warm-start-200k` packs, warm start `NVIDIA-Nemotron-3-Ultra-550B-A55B-Base-BF16` |

The Isambard Super pretraining quickstart is 128 GPUs and the Ultra SFT quickstart 288. Verda has 96, so those two
files are bases for a Verda topology, not launch configs.

`docs/verda/quickstart.md` is the path from a new account to a run.
