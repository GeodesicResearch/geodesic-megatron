# Verda quickstart runs

This file records the runs of this directory's quickstart configs on Verda, on 2026-11-14. `README.md` lists the
configs. `docs/verda/models.md` gives their layouts and launch commands.

## Columns

- Each row is one run. Its log is `/shared/geodesic/logs_kyle/megatron_runs/train-<job>.out`.
- `scripts/telemetry/score_run.py` scored each run on the window of `docs/verda/models.md`, "Scoring". A row on
  another window says so.
- s/iter is the mean step over that window. Tokens/s/GPU and TFLOP/s/GPU follow from it.
- MFU is TFLOP/s/GPU over 2,250, the B300 peak the scorer uses.
- Peak is the largest allocation over all ranks in the log's `[peak-memory]` line, in GiB.
- No scored run had a NaN or a skipped iteration. Every scored run except job 892 logged zero allocator retries.

## Nemotron 3 Nano (30B-A3B)

Each Nano quickstart fits on one node. Every Nano row keeps TP1 PP1, as the configs do.

Pretraining read a 300,000-document slice of ClimbMix-Sample, because the stage-1 blend did not exist yet. An overlay
outside the repository gave `nemotron_nano_quickstart_pretrain.yaml` that slice. For the EP1 rows, the launch line
set the two fields of `nemotron_nano_quickstart_pretrain_ep1.yaml`.

| Nodes | GBS | Layout | s/iter | Tokens/s/GPU | TFLOP/s/GPU | MFU | Peak GiB | Job |
|---|---|---|---|---|---|---|---|---|
| 1 | 64 | EP4 | 3.283 | 19,960 | 415.3 | 18.46% | 100.8 | 676 |
| 1 | 64 | EP1, EP overlap off | 2.813 | 23,299 | 484.8 | 21.55% | 176.5 | 740 |
| 2 | 128 | EP4 | 3.359 | 19,508 | 405.9 | 18.04% | 86.3 | 703 |
| 2 | 128 | EP1, EP overlap off | 2.975 | 22,031 | 458.4 | 20.37% | 176.5 | 804 |
| 4 | 256 | EP4 | 3.395 | 19,304 | 401.7 | 17.85% | 78.8 | 739 |
| 4 | 256 | EP1, EP overlap off | 2.943 | 22,266 | 463.3 | 20.59% | 176.5 | 805 |
| 8 | 512 | EP4 | 3.438 | 19,064 | 396.7 | 17.63% | 75.3 | 887 |
| 8 | 512 | EP1, EP overlap off | 3.026 | 21,657 | 450.6 | 20.03% | 176.5 | 888 |

SFT ran `nemotron_nano_quickstart_sft.yaml`, which finds its packs through a `shard*/` glob. The data build was still
writing shards during these runs. Each run read the shards that existed at its start.

| Nodes | GBS | Layout | Shards read | s/iter | Tokens/s/GPU | TFLOP/s/GPU | MFU | Peak GiB | Job |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 8 | EP4 | 1 of 32 | 1.660 | 19,744 | 482.4 | 21.44% | 89.0 | 878 |
| 2 | 16 | EP4 | 1 of 32 | 1.674 | 19,578 | 478.3 | 21.26% | 71.5 | 884 |
| 4 | 32 | EP4 | 11 of 32 | 1.698 | 19,296 | 471.4 | 20.95% | 67.8 | 885 |
| 8 | 64 | EP4 | 32 of 32 | 1.700 | 19,275 | 470.9 | 20.93% | 65.9 | 886 |

Midtraining ran `nemotron_nano_quickstart_midtrain.yaml` in launcher mode `pretrain`. The slow steps of "Findings"
raise the mean step more than the median.

| Nodes | GBS | Layout | s/iter | Median step | Tokens/s/GPU | TFLOP/s/GPU | MFU | Peak GiB | Job |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 8 | CP2 EP4 | 2.419 | 2.144 | 13,547 | 331.0 | 14.71% | 92.4 | 901 |
| 2 | 16 | CP2 EP4 | 2.507 | 2.216 | 13,069 | 319.3 | 14.19% | 77.7 | 907 |
| 4 | 32 | CP2 EP4 | 2.973 | 2.362 | 11,021 | 269.3 | 11.97% | 70.4 | 908 |

Each 1-node run below is a quickstart with another expert parallelism.

| Regime | Layout | s/iter | Tokens/s/GPU | TFLOP/s/GPU | MFU | Peak GiB | Job |
|---|---|---|---|---|---|---|---|
| Pretraining | EP8 | 3.371 | 19,444 | 404.6 | 17.98% | 90.0 | 723 |
| Pretraining | EP2 | 3.304 | 19,833 | 412.7 | 18.34% | 125.9 | 724 |
| Pretraining | EP4, EP overlap off | 3.094 | 21,180 | 440.7 | 19.59% | 97.2 | 799 |
| Pretraining | EP1, EP overlap on | Stops at config time | | | | | 733 |
| SFT | EP8 | 1.729 | 18,954 | 463.1 | 20.58% | 75.3 | 1049 |
| SFT | EP2 | 1.658 | 19,762 | 482.8 | 21.46% | 133.5 | 1050 |
| SFT | EP1 | 1.799 | 18,218 | 445.1 | 19.78% | 243.0 | 1051 |
| Midtraining | EP8 | 2.507 | 13,073 | 319.4 | 14.20% | 78.8 | 1052 |
| Midtraining | EP2 | 2.346 | 13,967 | 341.2 | 15.17% | 133.5 | 1053 |
| Midtraining | EP1 | 2.436 | 13,449 | 328.6 | 14.60% | 243.0 | 1054 |

## Nemotron 3 Super (120B-A12B)

On one node, Super pretraining ran cleanly only at PP4 EP2 (job 815). The pretraining probes below read the same
300,000-document slice as the Nano pretraining rows.

| Regime | Nodes | Layout | Result | Job |
|---|---|---|---|---|
| Pretraining | 1 | PP1 EP8 | Out of memory in iteration 2 | 726 |
| Pretraining | 1 | PP2 EP4 | 2,454 allocator retries in 50 iterations, peak 256.0 GiB | 736 |
| Pretraining | 1 | PP8 EP4, the Isambard topology | Stops at startup: 8 GPUs are not a multiple of PP x EP = 32 | 795 |
| SFT | 1 | PP2 CP4 EP4 | 22 allocator retries in 2 iterations, peak 252.9 GiB | 1037 |
| SFT from the Base release as NVIDIA publishes it | 2 | PP2 CP4 EP4 | Stops in iteration 2 on an Inf gradient norm | 1047 |

Pretraining read all of ClimbMix-Sample. An overlay outside the repository gave
`nemotron_super_quickstart_pretrain.yaml` a null checkpoint save. The launch line gave the layout and the batch.

| Nodes | GBS | Layout | s/iter | Tokens/s/GPU | TFLOP/s/GPU | MFU | Peak GiB | Job |
|---|---|---|---|---|---|---|---|---|
| 1 | 192 | PP4 EP2 | 39.13 | 5,024 | 380.8 | 16.93% | 245.2 | 815 |
| 2 | 384 | PP2 EP4 | 38.40 | 5,120 | 388.1 | 17.25% | 200.1 | 893 |
| 2 | 384 | PP2 EP8 | 44.34 | 4,434 | 336.1 | 14.94% | 162.9 | 894 |
| 4 | 768 | PP8 EP4, the Isambard topology | 42.87 | 4,586 | 347.6 | 15.45% | 98.0 | 806 |
| 4 | 768 | PP2 EP4 | 38.59 | 5,094 | 386.1 | 17.16% | 171.4 | 1055 |
| 8 | 1,536 | PP8 EP4, the Isambard topology | 43.42 | 4,528 | 343.2 | 15.25% | 75.8 | 892 |

A cancel ended job 892 at iteration 20, so its row scores iterations 11 to 20. Its log has no `[peak-memory]` line.
Its peak is the largest of the memory reports after iteration 1.

SFT ran `nemotron_super_quickstart_sft.yaml` from the Base-Chat-Init warm start.

| Nodes | GBS | Layout | s/iter | Tokens/s/GPU | TFLOP/s/GPU | MFU | Peak GiB | Job |
|---|---|---|---|---|---|---|---|---|
| 2 | 32 | PP2 CP4 EP4 | 14.96 | 4,381 | 353.2 | 15.70% | 197.2 | 1042 |
| 4 | 64 | PP8 CP4 EP4, the Isambard topology | 16.85 | 3,889 | 313.6 | 13.94% | 95.2 | 1056 |

CPT ran `nemotron_super_quickstart_cpt.yaml`. An overlay outside the repository gave it its own sample index cache.

| Nodes | GBS | Layout | s/iter | Tokens/s/GPU | TFLOP/s/GPU | MFU | Peak GiB | Job |
|---|---|---|---|---|---|---|---|---|
| 1 | 192 | PP4 EP2 | 37.72 | 5,212 | 395.0 | 17.56% | 244.3 | 1046 |

## Nemotron 3 Ultra (550B-A55B)

Ultra ran out of memory on 1 and 4 nodes.

| Nodes | Layout | Result | Job |
|---|---|---|---|
| 1 | PP4 EP2, a CPT layout from random weights in mode `pretrain` | Out of memory in setup, on the 120 GiB expert parameter buffer | 796 |
| 4 | PP4 EP8, BF16 gradients, GBS 8, SFT | Out of memory in the first optimizer step, with 257.2 to 257.9 GiB allocated | 902 |
| 6 | PP6 EP8, `nemotron_ultra_quickstart_sft_6node.yaml` | Runs | 1066 |

The first rank of each stage of job 1066 reported this memory after iteration 1. The rule column applies
`docs/verda/models.md`, "Memory per GPU", to the stage's parameter count in the log.

| Stage | Layers | Allocated after iteration 1, GiB | Rule, GiB |
|---|---|---|---|
| 0 | 18 | 184.1 | 181.5 |
| 1 to 3 | 18 | 177.3 | 174.5 |
| 4 | 19 | 178.2 | 175.4 |
| 5 | 17 | 183.4 | 180.6 |

SFT ran `nemotron_ultra_quickstart_sft_6node.yaml`.

| Nodes | GBS | Layout | s/iter | Tokens/s/GPU | TFLOP/s/GPU | MFU | Peak GiB | Job |
|---|---|---|---|---|---|---|---|---|
| 6 | 16 | PP6 EP8 | 5.487 | 498 | 167.5 | 7.44% | 218.0 | 1066 |

## Cells without a run

The study set out to measure each regime on 1, 2, 4 and 8 nodes. These cells have no scored run. None of the
cancelled jobs started.

| Model and regime | Nodes | Jobs | State |
|---|---|---|---|
| Nano pretraining on the stage-1 blend, EP4 and EP1 | 8 | 1289, 1290 | Cancelled while queued on 2026-11-15 |
| Nano midtraining | 8 | 1031, 1151 | Cancelled while queued, 1031 on 2026-11-14 and 1151 on 2026-11-15 |
| Super pretraining and SFT, the Isambard topology | 1, 2 | none | 8 and 16 GPUs are not multiples of 32 (job 795) |
| Super SFT, the Isambard topology | 8 | 1065, 1344 | Cancelled while queued, 1065 on 2026-11-14 and 1344 on 2026-11-15 |
| Super CPT | 2, 4, 8 | none | Not submitted |
| Ultra, either regime | 2 | none | Not submitted |
| Ultra CPT | 4 | 1048 | Cancelled while queued after job 902 ran out of memory |
| Ultra CPT | 6, 8 | 1288, 1287 | Cancelled while queued on 2026-11-15 |
| Ultra SFT, GBS 64 | 6 | 1293 | Cancelled while queued on 2026-11-15 |
| Ultra SFT, GBS 16 and GBS 64 | 8 | 1286, 1292 | Cancelled while queued on 2026-11-15 |

## Findings

| Finding | Evidence |
|---|---|
| `slinky-0` and `slinky-4` drained on 2026-11-14 with "Kill task failed". `sinfo -R` still listed both on 2026-11-15. An admin must resume them with `scontrol update nodename=slinky-[0,4] state=resume`. `docs/verda/cluster.md` gives the cause. | `sinfo -R`: JobId 585 on `slinky-0`, JobId 583 on `slinky-4` |
| Model imports that memory-mapped their safetensors on `/shared` stalled or failed. A 4-rank Nano import still read the first of 13 shards after 11.5 minutes. Two 8-rank imports ended `NODE_FAIL`. The same import from a copy on the node's disk took 2 min 21 s. | Jobs 737, 721, 731 and 873. `docs/verda/storage.md` gives the rule |
| Nano midtraining had 25, 34 and 51 slow steps on 1, 2 and 4 nodes. A slow step took over 4.3 s, counted on iterations 2 to 200. Each fell on an iteration where a dataloader worker first read one of the ten corpora. | Jobs 901, 907 and 908. A replay of the sampler, jobs 1160 and 1197, in `/shared/geodesic/scratch_kyle/megatron_verda/nano/cpt_slow_steps2.out` |
| The SFT packer draws its first-fit shuffle and its pack order from numpy's unseeded global random state. Two builds of one shard can differ in pack contents. | `first_fit_shuffle` and `fill_packing_strategy` in `src/megatron/bridge/data/datasets/packing_utils.py`. Read from the code |
| Ultra SFT on 6 nodes passes 2 microbatches per replica through 6 pipeline stages. Each stage works in 2 of every 7 pipeline slots. | Job 1066 and its 7.44% MFU |
| The first iteration of Ultra SFT on 6 nodes took 709.7 s. The job reached its second iteration 16 min 41 s after its start. | Job 1066 |
| The Ultra Base import on two nodes took 10 min 49 s. Its save peaked at 945 and 912 GiB of host memory on the two nodes. A node has 2,007 GiB. | Job 1162 |
