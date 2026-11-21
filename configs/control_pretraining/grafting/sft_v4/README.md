# Super V4 reasoning fine-tunes of the grafting campaign

This folder holds the configs of three Nemotron 3 Super fine-tunes. Each file is a byte-for-byte copy of the file that its runs read from the frozen launch folder `/projects/a5k/public/logs/control_pretraining/grafting_launch/phase_b_lr7p5e-4/configs/control_pretraining/grafting/sft_v4/`. The comments in the files therefore still call them uncommitted.

| Config | Run | Jobs | Warm start |
|---|---|---|---|
| `nemotron_super_sft_xl50b_gbs256_v4.yaml` | supernorm | 8082737, 8082738 | NVIDIA Super Base with the chat-token rows filled (Base-Chat-Init import) |
| `nemotron_super_cpt5_sft_xl50b_gbs256_v4.yaml` | supernorm-cpt5 | 8082739, 8082740 | the Super grafting continual-pretraining treatment save after five passes |
| `nemotron_super_cpt1_sft_xl50b_gbs256_v4.yaml` | supernorm-cpt1 | 8154326, 8191163 | the Super grafting continual-pretraining treatment save after one pass |

The `.env` file holds the launcher settings that all three runs used through `ISAMBARD_ENV_OVERRIDES`. The two derived configs compose the base config through `base_config`.

The runs read the Norm-mix packs at `/projects/a5k/public/data/geodesic-research__pa-warm-start-sft-xl-50b-mix-quality-filtered__train/shard*/packed/geodesic-research--nemotron-think-history-tokenizer_pad_seq_to_mult16/`. The launch scripts that the comments name (`scripts/submit_sft_v4_super.sh`) are in the frozen launch folder and not in this repository. The rest of the grafting campaign's configs and tools are on the branch `grafting-task-vector`.
