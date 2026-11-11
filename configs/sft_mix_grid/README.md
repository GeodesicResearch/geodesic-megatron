# Nemotron 3 Nano Dolci-SFT

The configs of the Nemotron 3 Nano Dolci-SFT: Nemotron 3 Nano Base trained on AI2's Dolci Think SFT data. The run is
compared with the Nemotron 3 Nano Norm-SFT
(`configs/control_pretraining/30b_baseline_ablations/nemotron_nano_30b_nvidia_base_sft_xl50b_gbs256_v4.yaml`) and with
the two Olmo 3 7B runs of `configs/olmo3_norm_sft/`.

| File | What it is |
|---|---|
| `data/dolci_think_sft_nano.yaml` | The prepare: Dolci at one commit, with each assistant turn's leading think block moved to `reasoning_content` |
| `data/dolci_think_sft_nano_counts.yaml` | The count pass: each row's tokens and digest under the Nano tokenizer, and the rows over 32,768 tokens |
| `data/dolci_think_sft_nano_epoch{1,2}_partition.yaml` | Each epoch's rows in a seeded random order, in 32 shards, less the rows over length |
| `data/dolci_think_sft_nano_epoch{1,2}_pack_m4.yaml` | Each epoch's pack: 32,768-token rows, documents padded to a multiple of 4 |
| `data/dolci_think_sft_nano_epoch{1,2}_audit_m4.yaml` | Each epoch's pack audit against the count pass's kept and over-length digests |

Each config's header gives its command. Run the steps in the order of the table, each as a one-node job from a frozen
copy (`configs/control_pretraining/submit_corpus_job.py`).
