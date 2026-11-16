# Nano-30B baseline performance campaign on Verda

This study searched for system settings that run the 30B baseline faster on Verda. The target was
35% model FLOP utilisation (MFU) or more on 4 nodes, which hold 32 B300 GPUs. MFU counts model FLOP/s per GPU against
2,250 TFLOP/s, the B300 dense BF16 peak. Each of the three stages kept the baseline's data, global batch, schedule,
optimizer and tokenizer (`configs/control_pretraining/30b_baseline/`). The settings found form the Verda arm of the
baseline, `configs/control_pretraining/30b_baseline_verda/` at commit `a478a8ec`.
`tests/unit_tests/test_cluster_configs.py` fails if a stage config there states more than system settings.

`scripts/telemetry/score_run.py` computed every rate on this page from a training log. The log of job N is
`/shared/geodesic/logs_kyle/megatron_runs/train-N.out`. The Scoring section of `docs/verda/models.md` gives the scorer
command. The Memory column divides the peak reserved memory of the most loaded GPU by the 287.8 GB of the device.
Allocated memory appears beside it. Job ids are SLURM jobs on Verda.

A posture is the set of system settings of a run. A lever is one setting changed against a reference run. A ratio is
the step time of the reference over the step time of the lever. A ratio above 1 therefore means a faster step. A
closed lever is a lever that the study measured and did not adopt. A screen is a short run of one lever on one node.

The width of a run is its node count. Its load is the sequences or packs that each GPU runs per step. TP, PP, CP, EP
and DP are the tensor, pipeline, context, expert and data-parallel sizes. GBS is the global batch size. A micro-batch
is the batch a GPU runs in one forward and backward pass. An SM is a streaming multiprocessor of the GPU.

An overlay is a config file that states only the keys it changes against its `base_config`. Composing an overlay with
its base gives its composed config. A code copy is a frozen snapshot of the repository that a job runs from. The
abbreviation sd means standard deviation.

## Result

Each stage has two agreeing 4-node runs at the exact global batch on production data. The second run of each stage
used the committed config. The first midtraining and SFT runs used an uncommitted overlay with the same composed
config. The window is the range of steps the scorer averaged.

| Stage | Job | Config | Start | Window | s/step | Tokens/s/GPU | TFLOP/s/GPU | MFU | Allocated / reserved GB | Memory |
|---|---|---|---|---|---|---|---|---|---|---|
| Pretraining | 1746 | Committed | The step-400 checkpoint of job 1745, trained from random weights | 421 to 460 | 13.755 | 38,116 | 793.1 | 35.25% | 158.1 / 194.1 | 67.4% |
| Pretraining | 1751 | Committed | The same checkpoint | 421 to 460 | 13.754 | 38,119 | 793.2 | 35.25% | 158.1 / 192.5 | 66.9% |
| Midtraining | 1687 | Overlay, equal in all 87 composed keys | The trained stage-1 model | 11 to 40 | 15.240 | 34,403 | 840.5 | 37.36% | 178.6 / 222.0 | 77.1% |
| Midtraining | 1715 | Committed | The trained stage-1 model | 11 to 40 | 15.234 | 34,417 | 840.9 | 37.37% | 178.6 / 220.1 | 76.5% |
| SFT | 1689 | Overlay, equal in all 102 composed keys | The Nano Base release | 21 to 60 | 14.324 | 36,601 | 894.2 | 39.74% | 174.4 / 214.6 | 74.6% |
| SFT | 1713 | Committed | The Nano Base release | 21 to 60 | 14.324 | 36,602 | 894.3 | 39.74% | 174.4 / 213.5 | 74.2% |

None of the six runs above had an allocator retry, a NaN step or a skipped step. Each stage also has a loss pair that
compares MXFP8 with BF16, in its loss section. The trained stage-1 model is the stage-1 model of the Isambard baseline
run. Its import is `/shared/geodesic/checkpoints/megatron_bridges/models/control-pretrain-30b-baseline-pretrain`. The
Nano Base release is `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16` at revision `e0ac9ee3`, imported unchanged by
job 1560. The SFT runs started from it in place of a stage-2 model.

## Method

Screens ran on one node at the load of the target width. The target width moved from 8 nodes to 4 during the study.
Screens before that move carried the 8-node load, half the 4-node load. Each trace records one step on rank 0 with
the torch profiler. Exposed time is time when a kernel runs with no kernel of another family beside it.

Two-node and 8-node runs checked that the rate per GPU held across nodes. A ladder is the sequence of changes from a
reference config to the final posture. One pair of runs measures each change. Loss pairs compared runs of the same
width, data and seed step by step. The noise floor is the loss spread between two runs of one posture.

Besides the baseline's data, batch and hyperparameters, the study held two constraints. EP stays at 8, so the expert
all-to-all stays inside a node. The dense layers run in MXFP8 or BF16, which excludes NVFP4. Rows that measured EP2,
EP4 or NVFP4 say which constraint excludes them.

The 4-node runs used a pool of four nodes: slinky-1, slinky-2, slinky-3 and slinky-7. Early screens also ran on other
Verda nodes. The nodes differ in speed (see "Node speed sets the pace of a multi-node run"). Each ratio in the tables
compares two runs on one node, unless its row says otherwise.

## Settings shared by the three stages

HybridEP is the backend of the flex token dispatcher. Its dispatch sends tokens to the GPUs that hold their experts.
Its combine returns the expert outputs. The EP overlap runs the dispatch and combine of one micro-batch beside the
compute of another.

`ISAMBARD_FP32_SSM_STATE` sets the Mamba scan. The BF16 scan is `0`. The fp32 scan with recompute is `checkpoint`. The
fp32 scan without recompute is `1`.

| Setting | Value | Reason |
|---|---|---|
| Parallelism | TP1, PP1, EP8 | EP8 is a constraint of the study (see "Method"). The closed levers of each stage include EP below 8 |
| Precision | `bf16_with_mxfp8_mixed_bf16_params_bf16_grad_reduce` | On stage 1, MXFP8 on the dense layers ran 1.078x BF16 at 8 nodes (job 1545 against 1544). Both runs used the same nodes, with NaN checks off. On stage 2 it ran 1.088x across nodes (job 1743 against 1744). Without the 1.6% speed gap between slinky-1 and slinky-2, that is 1.071x. The ladder table of stage 3 gives its pair. The routed experts stay in BF16 |
| Dispatcher | HybridEP flex dispatcher, router fusion, linear cross-entropy fusion | Held fixed in every run |
| EP overlap | `overlap_moe_expert_parallel_comm: true`, `mtp_num_layers: null` | Stage 1 measured 1.092x across nodes (jobs 1577 and 1594 against 1615). The ladder tables of stages 2 and 3 give their same-node pairs. The overlap refuses the default of 0 multi-token prediction (MTP) layers (job 1571) |
| Hardware queues | `ISAMBARD_CUDA_MAX_CONNECTIONS=32` | The EP overlap runs its communication on a second stream. With one hardware queue the two streams run in turn |
| Mamba scan | `ISAMBARD_FP32_SSM_STATE=0`, the BF16 scan | The ladder tables of stages 2 and 3 give its gain. Stage 1 at 8,192 tokens needs no fp32 scan (`docs/isambard/models.md`) |
| NaN checks | `check_for_nan_in_loss` and `check_for_nan_in_grad` on | The baseline sets them. They cost 0.4% on stage 1 (jobs 1577 and 1594 against 1595 and 1578) |
| Timers | `logger.timing_log_level: 2` | The baseline sets it. Level 1 gained nothing (see "Timer level 1 gained nothing") |

Both NaN checks stay on under the BF16 scan. `tests/unit_tests/test_cluster_configs.py` requires them in every
32,768-token stage that runs the BF16 scan. A BF16 scan produced NaNs on Isambard in Super 120B SFT at TP1, CP4, EP4
and PP22 (`configs/pa_warm_start/README.md`).

The EP overlap reserves memory that holds no tensor, because each stream keeps its own cache of freed blocks. Two
allocator settings left that memory reserved (jobs 1579 and 1596).

| Stage | Excess of peak reserved over peak allocated memory, 4 nodes | Jobs | Excess without the EP overlap, one node |
|---|---|---|---|
| 1, pretraining | 34.3 to 36.0 GB | 1751, 1746 | 4.7 GB (job 1615) |
| 2, midtraining | 41.5 to 43.4 GB | 1715, 1687 | |
| 3, SFT | 39.1 to 40.2 GB | 1713, 1689 | 4.6 GB (job 1691) |

## Findings across the stages

### Node speed sets the pace of a multi-node run

The nodes of the pool ran one config at different speeds. On one-node runs at the 4-node load, the spread was 2.5% in
stage 2. In stage 1 two nodes differed by 1.6%.

| Stage | Node | Job | s/step | Against the fastest node |
|---|---|---|---|---|
| 2, midtraining | slinky-1 | 1703 | 14.854 | 1.000 |
| 2, midtraining | slinky-2 | 1722 | 15.090 | 0.984 |
| 2, midtraining | slinky-3 | 1720 | 15.232 | 0.975 |
| 1, pretraining | slinky-7 | 1718 | 13.389 | 1.000 |
| 1, pretraining | slinky-3 | 1694 | 13.597 | 0.985 |

Every 4-node job ran on all four nodes of the pool. Each step ends in collectives across all GPUs, so the slowest node
sets the pace. Over steps 21 to 40, stage-2 job 1687 took 15.231 s per step. On one node, slinky-3 took 15.232 s (job
1720). Stage-1 job 1710 took 13.602 s without its disturbed steps, against 13.597 s for slinky-3 alone (job 1694). The
4-node runs therefore ran at the pace of slinky-3, with almost no extra cost from crossing nodes.

A probe sampled the GPUs of job 1710 during steps 10 to 14. Every node ran at its software power cap, about 1,000 to
1,030 W. Mean SM clocks ran 1,905 to 1,952 MHz per node, against a maximum of 2,032 MHz. An idle slinky-3 showed no
fault (job 1726).

### Four nodes against eight

A 4-node step holds twice the micro-batches per GPU of an 8-node step. Costs paid once per step then weigh half as much
per sample. Across node sets, stage 1 gained 2.4% per GPU (job 1710 without its disturbed steps, against job 1644).
Over steps 21 to 40, stage 2 gained 2.1% (job 1687 against job 1634). In stage 2, work outside the forward and
backward pass took 0.062 s per step at both widths. That work accounts for 0.4%.

The other 1.7% matches a fixed cost of about 0.26 s per step inside that pass. The cost is twice the 8-node step, less
the 4-node step and the 0.062 s. It covers the first and last micro-batch of a step, which the EP overlap leaves
unpaired. It also covers the first and last gradient buckets of the data-parallel reduce. Stage 3 has no 8-node run at
EP8.

### Timer level 1 gained nothing

`logger.timing_log_level` sets which timers Megatron records each step. The baseline sets 2. In job 1644 the per-step
gather of the timers left 0.110 s of GPU idle. Level 1 still ran at the speed of level 2 on the same node. Level 1
also drops the W&B series `batch-generator-time` and `forward-compute-time`.

| Stage | Level 1 | Level 2 | Node | Ratio |
|---|---|---|---|---|
| 1, pretraining | 1707 | 1718 | slinky-7 | 1.0007x |
| 2, midtraining | 1721 | 1703 | slinky-1 | 1.001x |
| 2, midtraining | 1704 | 1720 | slinky-3 | 1.000x |
| 3, SFT | 1714 | 1681 | Across nodes, slinky-2 against slinky-1 | 0.992x |

### Delayed weight gradients hide time that slower GEMMs take back

`comm_overlap.delay_wgrad_compute` defers the weight-gradient GEMMs of a micro-batch, so they run beside the next
micro-batch's dispatch. With the EP overlap on a hybrid model, the pinned Megatron-LM refuses it (patch `0003` in
`docs/isambard/development.md`). These runs used a local change to the Bridge and the Megatron-LM fork that lifts the
refusal. That change is not in this repository. In all three stages the loss stayed inside the noise of identical runs.

| Stage | Delay on | Delay off | Node | Ratio | Allocated memory |
|---|---|---|---|---|---|
| 1, pretraining | 1731, 13.337 s | 1718, 13.389 s | slinky-7 | 1.0039x | +0.9 GB |
| 2, midtraining | 1728, 14.685 s | 1703, 14.854 s | slinky-1 | 1.0115x | +1.7 GB |
| 3, SFT | 1730, 7.186 s | 1714, 7.223 s | slinky-2 | 1.0052x | +1.7 GB |

Job 1710 traced rank 0 of a 4-node run of stage 1 with the delay off. Job 1735 traced one node at the same load with
the delay on. The two traces come from different nodes.

| Measure per step | Delay off, job 1710 | Delay on, job 1735 |
|---|---|---|
| Exposed backward combine | 0.528 s | 0.049 s |
| Exposed HybridEP time, all kinds | 0.88 s | 0.33 s |
| Expert forward GEMMs | 1.12 s | 1.36 s |
| Expert weight-gradient GEMMs | 1.09 s | 1.32 s |

The delay hid 0.48 s of combine per step. The expert GEMMs beside HybridEP's kernels slowed by about 20%. That
slowdown took back most of the hidden time. HybridEP takes 24 SMs for its communication and more blocks for its
permute. Fewer SMs for HybridEP made each step slower.

| Stage | Job | HybridEP SMs | Against | Node | Ratio |
|---|---|---|---|---|---|
| 1, pretraining | 1741 | 16 | 1731, delay on, 24 SMs | slinky-7 | 0.992x |
| 1, pretraining | 1742 | 8 | 1694, delay off, 24 SMs | slinky-3 | 0.892x |
| 2, midtraining | 1739 | 16 | 1728, delay on, 24 SMs | slinky-1 | 0.989x |
| 3, SFT | 1740 | 16 | 1730, delay on, 24 SMs | slinky-2 | 0.988x |

### Data order follows the data-parallel size

Stages 1 and 2 read their data with `MegatronPretrainingRandomSampler` (`src/megatron/bridge/data/samplers.py`). The
launcher selects it with `dataloader_type="cyclic"` in `pipeline/training/datasets.py`. `data_sharding` is true by
default. With sharding, the sampler gives each data-parallel rank its own contiguous bucket of the dataset. The
samples of a step therefore change with the data-parallel size.

At step 1, job 1687 on 32 GPUs logged a loss of 1.9328. Job 1634 on 64 GPUs logged 1.9492. Loss pairs on this page
therefore compare runs of equal width only. The per-step order at 32 GPUs also differs from the order of the 512-GPU
Isambard run. Stage 3 reads with `MegatronPretrainingBatchSampler`, which deals each global batch of consecutive indices
to the ranks. Its per-step packs stay the same at any data-parallel size.

### Stage 1 routing evens out by step 240

A rank's receive ratio is the tokens it receives in one dispatch over its balanced share. A receive counter outside
the repository logged each rank's largest ratio in every step of job 1745. The first table averages those largest
ratios over ranks. From random weights the routing starts uneven. At EP8 the step time follows that imbalance closely.

| Receive ratio, mean over ranks | Steps of job 1745 | Mean step |
|---|---|---|
| 1.45 | 2 and 3 | 14.5 s |
| 1.7 to 1.9 | 43 steps | 15.3 s |
| 1.9 to 2.1 | 62 steps | 15.7 s |
| 2.1 to 2.4 | 7 steps | 16.6 s |
| 2.4 to 3.0 | 22 steps, 6 to 30 | 18.0 s |

A linear fit gives step = 10.18 + 2.83 x ratio, with r = 0.966 over 136 steps. The trained stage-1 model gives a ratio
of 1.105 at EP8 (job 1618). The fit then predicts 13.3 s, close to the warm 4-node step of 13.602 s (job 1710). In job
1745 the ratio averaged 2.39 over steps 20 to 39. It then held between 1.80 and 1.97 until step 179. It fell to 1.24 by
step 240 and stayed flat through step 320.

| Steps of job 1745 | Receive ratio, mean over ranks | Mean step |
|---|---|---|
| 180 to 199 | 1.69 | 14.52 s |
| 200 to 219 | 1.46 | 14.22 s |
| 220 to 239 | 1.32 | 14.03 s |
| 240 to 259 | 1.24 | 13.88 s |
| 260 to 279 | 1.23 | 13.87 s |
| 280 to 299 | 1.24 | 13.86 s |

From step 240 the step time stayed flat, about 2% over the warm 13.602 s. A settling rule, fixed before the run, tests
30-step windows from step 21. A window counts as settled when it passes four tests:

- Its median step lies within 1% of the previous window's median.
- A linear fit of its steps moves less than 1% across it.
- The largest ratio over all ranks, averaged over the window, falls by less than 1%.
- The window ends at step 300 or later.

| Window | Median step | Fit across the window | Largest receive ratio, window mean | Settled |
|---|---|---|---|---|
| 201 to 230 | 14.141 s | -2.54% | 1.661 | No |
| 231 to 260 | 13.883 s | -1.14% | 1.409 | No |
| 261 to 290 | 13.860 s | -0.13% | 1.366 | No, it ends before step 300 |
| 291 to 320 | 13.860 s | -0.36% | 1.367 | Yes |
| 321 to 350 | 13.855 s | -0.29% | 1.399 | Yes |
| 351 to 380 | 13.830 s | -0.30% | 1.356 | No, the ratio fell 3.1% |

The rule held first at steps 291 to 320. The scored windows of job 1745 begin at step 301, inside that window. The
ratio still fell 3.1% in window 351 to 380, while the median step moved 0.2%. The `seq_load_balancing_loss` that
Megatron logs tracks the imbalance poorly. In job 1745 it fell to 1.13 by step 130, below the trained model's 1.17. Its
receive ratio stayed near 1.85 over the same steps.

The Isambard stage-1 run trained the same blend from random weights on 512 GH200 GPUs at EP4. Its W&B runs in
`geodesic/megatron_training` are fuhvhgxq, ggt10a5z, mpf5lqoj and tz0kuy6q. Its load-balancing loss rose again after
the learning rate peaked, then fell slowly.

| Step | `seq_load_balancing_loss` | Against the final level |
|---|---|---|
| 100 | 1.122 | -4.5% |
| 200 | 1.053 | -10.3% |
| 400 | 1.255 | +6.8% |
| 800 | 1.234 | +5.1% |
| 1,600 | 1.188 | +1.2% |
| 3,200 | 1.167 | -0.6% |
| 6,400 | 1.160 | -1.2% |
| 12,800 | 1.165 | -0.8% |
| 25,600 | 1.172 | -0.2% |
| 29,881 | 1.176 | +0.1% |

The final level is 1.174, the mean over the last 500 steps. The loss peaked at 1.266 at step 518. A 100-step rolling
mean stays within 5% of the final level from step 865. The Isambard step time held within 1% from about step 150 to
step 500. At EP4 with heavy recompute, routing changed that step little.

## Stage 1: pretraining

Stage 1 trains the baseline's stage-1 blend from random weights at 8,192 tokens
(`configs/control_pretraining/30b_baseline/README.md`). At 4 nodes a GPU runs 64 sequences per step, 16 micro-batches
of 4.

### Posture and score

Stage 1 adds four settings to the shared ones.

| Setting | Value | Reason |
|---|---|---|
| Micro-batch | 4 | Micro-batch 2 ran 0.931x across nodes (job 1605). Micro-batch 8 fits only with the EP overlap off and heavier recompute |
| Recompute | None on 26.10.00. The 26.04 scores below ran `recompute_modules: ["moe_act"]` | On 26.10.00 recompute off runs 1.0025x the `[moe_act]` posture on one node, with the loss at the identical-run floor (`docs/investigations/verda-support-report.md`) |
| Data reads | `mmap_bin_files: false`, `create_attention_mask: false`, 8 workers | Memory-mapped reads on `/shared` took 18 to 72 s per step (job 1454). Plain reads ran 7.32 s per step on one node (job 1539) |
| Parameter gather | `comm_overlap.overlap_param_gather: true` | Held fixed in every run |

Job 1745 trained from random weights for 400 steps and saved a checkpoint at step 400. It carried the receive counter.
Jobs 1746 and 1751 resumed that checkpoint without the counter for 60 steps each. They agreed within 0.01%. Steps 10 to
14 of job 1710 overlapped a probe of GPU clocks and ran 0.1 to 0.6 s slow.

| Job | Run | Width | Window | s/step | Tokens/s/GPU | MFU | Allocated / reserved GB | Memory |
|---|---|---|---|---|---|---|---|---|
| 1745 | Random weights, counter on | 4 nodes | 20 to 50 | 16.543 | 31,692 | 29.31% | 191.9 / 236.5 over the run | 82.2% |
| 1745 | Random weights, counter on | 4 nodes | 301 to 350 | 13.861 | 37,824 | 34.98% | 191.9 / 236.5 over the run | 82.2% |
| 1745 | Random weights, counter on | 4 nodes | 351 to 400 | 13.829 | 37,912 | 35.06% | 191.9 / 236.5 over the run | 82.2% |
| 1710 | Warm start | 4 nodes | 6 to 30 | 13.670 | 38,354 | 35.47% | 156.6 / 195.2 | 67.8% |
| 1710 | Warm start, steps 10 to 14 left out | 4 nodes | 6 to 30 | 13.602 | 38,544 | 35.65% | 156.6 / 195.2 | 67.8% |
| 1644 | Warm start | 8 nodes | 6 to 30 | 6.967 | 37,624 | 34.79% | 152.5 / 188.6 | 65.5% |
| 1718 | Warm start, slinky-7 | 1 node, GBS 512 | 6 to 25 | 13.389 | 39,158 | 36.21% | 180.0 / 216.0 | 75.0% |
| 1694 | Warm start, slinky-3 | 1 node, GBS 512 | 6 to 25 | 13.597 | 38,558 | 35.66% | 180.0 / 214.2 | 74.4% |

Warm-start runs started from the trained stage-1 model. The one-node runs kept the 4-node load of 64 sequences per GPU.
The counter cost at most 0.5% (job 1746 against steps 351 to 400 of job 1745). The routing still improved over those
steps, so part of the 0.5% comes from routing. The settled rate from random weights ran 1.1% under the warm 4-node
rate. The comparison is job 1746 against job 1710 without steps 10 to 14.

The step-400 checkpoint holds 295 GiB (316.8 GB) in `torch_dist` format. Job 1745 saved it in 75.6 s. Jobs 1746 and
1751 loaded it in 50.2 s and 49.9 s.

### Loss

Warm runs that differ only in the node or in settings that leave the numerics alone matched through step 2. From step
3 their loss differed by at most 9.9e-5 per step (jobs 1694, 1707 and 1708 against 1718). That spread is the noise
floor of a warm start.

At 8 nodes, job 1545 under MXFP8 and job 1544 under BF16 read the same data and warm start. The loss gap per
step ran from +0.0024 to +0.0019 over steps 1 to 24, with a mean of +0.0022. Its slope was -2.0e-5 per step, so the
gap narrowed slowly. The gradient norm under MXFP8 ran 0.056 against 0.041 at step 1 and 0.026 against 0.022 at step
24. These warm runs ran in early warmup, at a learning rate of 8e-5 or less.

The two resumed runs started from the same checkpoint and random state. Their loss matched at step 401 and differed by
at most 8.1e-4 through step 430. At the full learning rate of 1e-3 the gap then grew, to 0.0194 at step 445. The mean
absolute difference over steps 421 to 460 was 0.0037.

### Trace

Both traces carry Python stacks. Job 1644 ran 8 micro-batches per GPU and job 1710 ran 16.

| Item | 8 nodes, job 1644, 7.04 s step | 4 nodes, job 1710, 13.71 s step | Where in the step |
|---|---|---|---|
| GPU idle | 0.242 s (3.4%) | 0.374 s (2.7%) | Step boundary and micro-batch starts |
| HybridEP combine | 0.302 s, backward 0.239 s | 0.599 s, backward 0.528 s | Through the whole step |
| HybridEP device sync | 0.106 s | 0.180 s | Through the whole step |
| HybridEP dispatch | 0.069 s | 0.094 s | First and last tenth |
| Gradient reduce-scatter | 0.136 s | 0.148 s | Last tenth |
| Parameter all-gather | 0.044 s | 0.042 s | First tenth |
| All-reduce | 0.031 s | 0.028 s | First tenth |
| All exposed communication | 0.594 s (8.4%) | 0.997 s (7.3%) | |

The backward combine stays exposed for about 33 ms per micro-batch. The 4-node step holds twice the micro-batches, so
that time doubles. The other rows occur once per step. In job 1735, HybridEP's permute and unpermute stayed exposed
for about 0.64 s per step. The relayout copy of the Mamba `in_proj` output took 0.22 s per step in job 1544,
about 3%.

### Closed levers

Rows against "the reference mean" compare with the mean of jobs 1577 and 1594. Those ran on slinky-7 and slinky-2 at
32 sequences per GPU, the 8-node load. Every screen of that load ran on another node, so those ratios cross nodes.
Timer level 1, delayed weight gradients and the EP overlap appear in earlier sections.

| Lever | Jobs | Nodes | Ratio | Memory | Reason closed |
|---|---|---|---|---|---|
| EP2 | 1628 against 1577 | Same node | 1.034x | | Outside the EP8 constraint |
| EP2 | 1656 against 1647 | Same node | 1.036x | | Outside the EP8 constraint |
| EP2, 2 nodes | 1648 against 1604 | Across nodes | 1.024x | | Outside the EP8 constraint |
| EP2, 8 nodes | 1662 against 1644 | Across node sets | 1.032x | +44.6 GB allocated | Outside the EP8 constraint |
| EP4 | 1616 against the reference mean | Across nodes | 1.016x | | Outside the EP8 constraint |
| NVFP4 on dense layers | 1620 against the reference mean | Across nodes | 1.023x | | Outside the precision constraint. Loss ran +0.0130 to +0.0165 above MXFP8, with a gradient norm 1.3 to 1.6x higher |
| Sync-free HybridEP with permute fusion | 1618 against 1594 | Same node | 1.001x | +10 GB allocated, +32 GB reserved | No gain |
| HybridEP on 16 SMs | 1647 against the reference mean | Across nodes | 0.995x | | No gain |
| Recompute off | 1708 against 1694 | Across nodes | 1.004x | +16.3 GB allocated, +22.6 GB reserved | Inside the node difference on 26.04. Adopted on 26.10.00, where a same-node pair measures it |
| `[moe_act, layernorm, shared_experts]` recompute | 1606 against 1577 | Across nodes | 0.968x | 3.2 GB less per sequence | Slower |
| Micro-batch 8 with `[moe_act]` | None | | | About 30 GB per added sequence, over 270 GB allocated by arithmetic | It does not fit |
| Micro-batch 8, heavier recompute, EP overlap off | 1617 against 1604, 2 nodes | Across nodes | 0.892x | 242.2 / 252.6 GB at 2 nodes, 87.8% | Slower, because the EP overlap gains more |
| Binding each rank's CPUs and memory to its GPU's NUMA domain, an uncommitted launcher change | 1646 against the reference mean | Across nodes | 0.995x | | No gain |
| Gradient buckets of 100M parameters, EP2 | 1664 against 1648, 2 nodes | Across nodes | 0.990x | | No gain |
| Allocator garbage collection at 0.6 | 1579 against the reference mean | Across nodes | 0.990x | Reserved memory did not fall | No gain |
| Classic allocator with garbage collection | 1596 against the reference mean | Across nodes | 0.987x | +25 GB reserved | Slower |

### Memory

Stage 1 reserves 192.5 to 195.2 GB at 4 nodes once routing settles. That is 66.9 to 67.8% of the device (jobs
1751, 1746 and 1710). From random weights, the uneven early routing raised the peak to 236.5 GB reserved, 82.2% (job
1745, rank 25). No measured change inside the EP8 constraint both fits and gains more than the node difference. Stage
1 therefore stays near 68%. From micro-batch 2 to 4, the peak rose 30.2 GB per added sequence (job 1577 against
1605).

`OptimizerConfig.__post_init__` in `3rdparty/Megatron-LM/megatron/core/optimizer/optimizer_config.py` sets
`use_precision_aware_optimizer_no_fp8_or_ds_fp8` to false under MXFP8 with fp32 main parameters. The optimizer then
keeps a full fp32 main copy of each sharded parameter. After checkpoint load on one node, job 1577 under MXFP8 had
15.8 GB more allocated memory than job 1620 under NVFP4.

## Stage 2: midtraining

Stage 2 trains the baseline's stage-2 blend at 32,768 tokens from the trained stage-1 model
(`configs/control_pretraining/30b_baseline/README.md`). At 4 nodes a GPU runs 16 sequences per step, one per
micro-batch.

### Posture and ladder

Stage 2 adds three settings to the shared ones.

| Setting | Value | Reason |
|---|---|---|
| CP | 1 | A GPU holds a whole 32,768-token sequence |
| Micro-batch and recompute | Micro-batch 1, no recompute | Every larger micro-batch ran slower or needed full recompute (closed levers below) |
| Data reads | Plain reads, no attention mask, 8 workers | Steps ran flat from step 2 (job 1546) |

The reference config is the Verda config without the EP overlap and the BF16 scan. Job 1569 ran it on one node. Job
1546 ran the same system settings at 8 nodes through another overlay, with the NaN loss check off. Two changes took the
8-node rate from 28.73% (job 1546) to 36.60% (job 1634). Each change row compares two runs at 8 sequences per GPU.

| Change | Jobs | Node | s/step | Ratio | Allocated / reserved GB, or the change |
|---|---|---|---|---|---|
| Reference config, 8 nodes | 1546 | 8 nodes | 9.907 | | 157.4 / 163.3 |
| Reference config, 1 node | 1569 | slinky-9 | 9.758 | 1.000 | 185.1 / 188.7 |
| EP overlap, fp32 scan without recompute in both | 1600 against 1570 | slinky-1 | 8.456 against 9.197 | 1.088x | +14.2 / +46.5 |
| BF16 scan, EP overlap in both | 1593 against 1584 | slinky-5 | 7.735 against 9.193 | 1.188x | +2.6 / +6.9 |
| Both changes against the reference | 1593 against 1569 | Across nodes, slinky-5 against slinky-9 | 7.735 against 9.758 | 1.262x | +17.4 / +49.9 |
| Final posture against the reference, 8 nodes | 1634 against 1546 | Across node sets | 7.777 against 9.907 | 1.274x | +17.2 / +49.6 |

Job 1546 ran on slinky-3, the slowest node. Job 1634 ran without it. Part of the 8-node ratio therefore comes from the
node set.

| Final posture, job | Width | Window | s/step | Tokens/s/GPU | MFU | Allocated / reserved GB | Memory |
|---|---|---|---|---|---|---|---|
| 1634 | 8 nodes, GBS 512 | 21 to 40 | 7.777 | 33,705 | 36.60% | 174.6 / 212.9 | 74.0% |
| 1699 | 2 nodes, GBS 256 | 4 to 23 | 15.130 | 34,652 | 37.63% | 186.4 / 224.5 | 78.0% |
| 1703 | 1 node, GBS 128 | 11 to 30 | 14.854 | 35,295 | 38.33% | 202.2 / 241.3 | 83.8% |

Jobs 1699 and 1703 kept the 4-node load of 16 sequences per GPU. Job 1634 ran 8 sequences per GPU.

### Loss

Each comparison used the same data, seed and width. The runs are not bit-deterministic after step 2. A soak is a
200-step run. A twin repeats a run with the scan setting changed.

| Comparison | Jobs | Result |
|---|---|---|
| Noise floor, one config on three nodes | 1703, 1720, 1722 | Exact at steps 1 and 2. Loss differed by up to 0.0028 in steps 1 to 10 and 0.0005 in 11 to 30 |
| MXFP8 against BF16 | 1743 on slinky-1 against 1744 on slinky-2, GBS 128, steps 11 to 60 | Loss gap +0.0023, sd 0.00045, largest +0.0037 at step 11. Slope -0.0027 per 100 steps |
| EP overlap | 1584 against 1569, steps 1 to 30, across nodes | Mean loss difference +0.0006. Gradient-norm ratio 1.009 |
| BF16 scan, 30 steps | 1576 against 1569, across nodes | Mean loss difference +0.0001. Gradient-norm ratio 1.003 |
| BF16 scan, soak on the stage blend | 1598, and 1569 for steps 1 to 35 | 0 NaN, 0 skipped. Loss within 0.0009 of job 1569 from step 11 |
| Twin of the soak, fp32 scan without recompute | 1643 against 1598, steps 1 to 110 | Mean loss difference +0.0001. Mean gradient-norm ratio 1.024, set by the spikes at steps 103 to 108 |
| BF16 scan, long-document stress | 1610 against 1611, steps 1 to 30 | Mean loss difference -0.0002. 200 steps without a NaN or skipped step |
| Final posture at 8 nodes | 1634 against 1546, steps 1 to 40 | Mean loss difference +0.0001. Gradient-norm ratio 0.996 |

The MXFP8 loss gap stayed near +0.0023 and narrowed slowly. Its slope had a standard error of 0.0002 per 100 steps.
The noise slope between identical runs measured at most 0.0012 per 100 steps. The stage-1 pair measured a similar mean
gap. The `seq_load_balancing_loss` gap measured +0.0025, against a noise sd of 0.0002.

MXFP8 showed single-step gradient-norm spikes of 0.122 at step 54 and 0.061 at step 58. BF16 read 0.029 and 0.031 at
those steps. The loss gap held at +0.0019 and +0.0020 there. MXFP8 ran 14.815 s per step against 16.116 s for BF16. It
used 5.2 GB less allocated memory (202.2 against 207.4 GB).

The soak read the stage blend for 12,800 sequences. About 1,772 of them were windows inside a single document. The
stress blend took `stack_edu_long`, `climbmix_ai_docs_long` and `zyda_ai_docs_long` in equal parts. Every document there
exceeds 32,768 tokens, so about 10,300 of its 12,800 windows lay inside one document.

The soak and its twin read the same batches. Both raised their gradient norm at steps 103, 105 and 108, from about
0.05 elsewhere. The soak read 0.139, 0.096 and 0.384 at those steps. The twin read 0.186, 0.352 and 0.112.

In mamba_ssm 2.3.1 the scan carries its state between chunks in fp32 registers. It rounds only the stored copy of each
chunk state to BF16.

### Trace

The trace of job 1630 covers one step of the final posture on one node at 8 sequences per GPU. The step took 7.75 s,
with 0.18 s of GPU idle.

| Item | Kernel time s | Exposed s | Share of step |
|---|---|---|---|
| HybridEP combine | 0.99 | 0.40 | 5.2% |
| Permute and unpermute | 0.50 | 0.37 | 4.8% |
| HybridEP device sync | 0.98 | 0.21 | 2.7% |
| HybridEP dispatch | 0.70 | 0.07 | 0.9% |
| NCCL collectives | 0.05 | 0.045 | 0.6% |
| GPU idle | | 0.18 | 2.3% |
| Mamba scan | 1.32 | 1.03 | 13.3% |
| Attention | 0.96 | 0.95 | 12.3% |
| Dense GEMM | 1.15 | 0.80 | 10.3% |
| Expert GEMM | 1.85 | 0.72 | 9.3% |

The GPU idled during host syncs of the timers, the NaN check and the HybridEP dispatch. The HybridEP device sync waited
on the busiest local expert, which drew 11.1x its balanced share (job 1621). A trace at 2 nodes and 16 sequences per
GPU (job 1699) confirmed the split. HybridEP and permute left 1.71 s of 15.07 s exposed (11.3%). NCCL left 0.09 s
exposed. The parameter all-gather took 0.062 s, of which 0.019 s was exposed.

### Closed levers

Rows against job 1600 or 1607 ran with the fp32 scan without recompute. Rows against job 1569 ran with the reference
config. Timer level 1 and delayed weight gradients appear under "Findings across the stages".

| Lever | Jobs | Nodes | Ratio | Memory | Reason closed |
|---|---|---|---|---|---|
| fp32 scan without recompute | 1570 against 1569 | Across nodes | 1.061x | +27.8 GB | Slower than the BF16 scan, by 0.72 s per step in traces (job 1629 against 1630). It ran 32.85% at 8 nodes (job 1635) |
| Permute fusion into HybridEP | 1607 against 1600, 1608 against 1593 | Same node | 0.999x, 1.004x | None | Neutral under the EP overlap. Without the overlap it ran 1.047x across nodes (job 1588) |
| Sync-free HybridEP, capacity factor 1.25 | 1601 against 1569 | Across nodes | 0.993x | +18.3 GB | Loss offset +0.0017. Job 1621 found tokens over budget on 3 of 8 ranks |
| Micro-batch 2 with `[moe_act]`, 1 node | 1587 against 1569 | Across nodes | 0.705x | 273.9 / 277.3 GB, 99 GB per added sequence | 109 allocator retries |
| Micro-batch 2 with `[moe_act]`, 2 nodes | 1609 against 1569 | Across nodes | 0.937x | 258.1 / 266.6 GB | Slower without retries |
| Micro-batch 4 and 8 | None | | | | They fit only with full recompute, which the EP overlap refuses |
| CP2 | None | | | | Memory does not limit CP1 |
| EP4 | 1637 against 1584, 1655 against 1593 | Across nodes | 1.036x, 1.058x | +11.8 GB allocated (1655) | Outside the EP8 constraint |
| NVFP4 | None | | | | Outside the precision constraint |
| Binding each rank to its own CPU cores, an uncommitted launcher change | 1622 against 1607 | Same node | 1.000x | None | No gain |
| HybridEP on 16 SMs | 1623 against 1607 | Across nodes | 0.972x | None | Slower. Communication is on the critical path |
| HybridEP on 32 SMs | 1654 against 1607 | Same node | 1.005x | None | Inside the noise |
| `overlap_param_gather_with_optimizer_step` | 1705 | | | | Setup failed with an IndexError in `DistributedOptimizer.__init__` (`3rdparty/Megatron-LM/megatron/core/optimizer/distrib_optimizer.py`), on an empty list of model chunks. The option needs interleaved PP |
| `align_param_gather` | None | | | | It aligns PP stages and has no use at PP1 |
| `ep_overlap_early_attn_memory_release` | 1706 against 1703 | Across nodes | 0.944x | -0.5 / -2.5 GB | Slower |
| Parameter-gather overlap off | None | | | | The overlap hides at most 0.06 s per step at 2 nodes (trace of job 1699) |

### Memory

The final posture uses 76.5 to 77.1% of GPU memory at 4 nodes, with recompute already off. Delayed weight gradients
gained 1.0115x for 1.7 GB, with code outside this repository. Every other change that spends more memory ran slower or
falls outside the EP8 constraint. Stage 2 therefore stays near 77%. That share includes the reserved memory that the
EP overlap holds without a tensor.

## Stage 3: SFT

Stage 3 trains the baseline's SFT stage on packs of `geodesic-research/pa-warm-start-sft-heavy-25b-mix`
(`configs/control_pretraining/30b_baseline/README.md`). Its tokenizer is
`geodesic-research/nemotron-think-history-tokenizer`. A pack concatenates several conversations into one sequence of at
most 32,768 tokens. The Verda copy of the corpus holds 764,686 packs and 25,011,804,203 tokens, 32,708.6 tokens per
pack on average (job 1582). At 4 nodes a GPU runs 16 packs per step, one per micro-batch.

The Base release needed no embedding graft for this tokenizer. Every chat-template id that the tokenizer emits has a
trained embedding row. Ids 1, 3, 4 and 10 to 17 have row norms of 0.87 to 0.93. The release has two near-zero rows, ids
84357 and 112147. A scan of all 764,686 packs found neither id (job 1582).

### Posture and ladder

Stage 3 adds four settings to the shared ones.

| Setting | Value | Reason |
|---|---|---|
| `dataset.dataset_kwargs.pad_seq_length_to_mult` | 64 | MXFP8 needs token counts divisible by 32. A pad multiple of 16 crashed at step 11 (jobs 1572 to 1574) |
| CP | 1 | CP2 ran slower (closed levers below) |
| Micro-batch | 1 | `ConfigContainer.validate` in `src/megatron/bridge/training/config.py` refuses larger values for packed SFT |
| Recompute | None | The posture fits without it. At micro-batch 1 the memory that recompute frees has no use |

The pad multiple adds at most 63 padding tokens at the tail of a pack. The collate gives them loss mask 0. The
mixture-of-experts (MoE) routers leave them out of the auxiliary losses, the z-loss and the expert-bias counts. The
data, its order and the loss therefore stay the baseline's.

In jobs 1572 to 1574, Transformer Engine raised `Assertion failed: flat_first_dim % MXFP8_BLOCK_SIZE == 0 &&
flat_last_dim % MXFP8_BLOCK_SIZE == 0. MXFP8 requires tensor dims that are divisible by 32 (got shape=(32752,2688))` in
the first Mamba `in_proj`. Job 1602 ran the reference config of the ladder: pad 64, without the EP overlap or the BF16
scan.

| Change | Jobs | Same node | Ratio | Allocated / reserved GB, or the change |
|---|---|---|---|---|
| Reference config | 1602 | | 9.405 s, 30.27% | 183.7 / 187.7 |
| MXFP8 against BF16, already in the reference config | 1602 against 1589 | Yes, slinky-6 | 1.061x | -4.8 / -3.8 |
| BF16 scan, with the EP overlap on | 1714 against 1633 | Yes, slinky-2 | 1.204x | +2.5 / +8.3 |
| BF16 scan, without the EP overlap, steps 81 to 100 | 1597 against 1602 | No, across nodes | 1.176x | +6.2 / +6.4 |
| EP overlap on packed batches | 1681 against 1691 | Yes, slinky-1 | 1.101x | +8.1 / +41.9 |
| Final posture | 1681 | | 7.167 s, 39.72% | 197.6 / 236.0 |

The step ran 1.312x faster in job 1681 than in job 1602, on different nodes. Job 1714 set `logger.timing_log_level: 1`.
Level 1 ran at the speed of level 2 on the same node in stages 1 and 2.

| Run | Job | Width | Window | s/step | Tokens/s/GPU | MFU | Allocated / reserved GB | Memory |
|---|---|---|---|---|---|---|---|---|
| Final posture, GBS 64 | 1681 | 1 node | 11 to 30 | 7.167 | 36,576 | 39.72% | 197.6 / 236.0 | 82.0% |
| EP4, no EP overlap, reference only | 1658 | 8 nodes | 21 to 60 | 7.630 | 34,357 | 37.31% | 177.2 / 183.0 | 63.6% |
| EP4 with the EP overlap, reference only | 1659 | 8 nodes | 21 to 60 | 7.203 | 36,394 | 39.52% | 184.3 / 223.2 | 77.6% |

Per GPU, 4 nodes ran at 1.0007x the rate of one node (job 1689 against 1681). The 8-node runs used EP4, which the EP8
constraint excludes. They serve only as references. The scorer counts GBS times 32,768 tokens per step. A pack holds
0.2% fewer real tokens than that count. The Scoring section of `docs/verda/models.md` gives the reason to compare
packed runs by tokens per second per GPU.

### Loss

Every pair in the two tables read the same packs in the same order. The noise floor comes from pairs of identical
postures.

| Noise pair | Jobs | Loss gap sd | Largest loss gap | Load-balancing gap sd, largest | Gradient-norm ratio |
|---|---|---|---|---|---|
| BF16 scan, two nodes | 1597, 1603 | 0.00072 | 0.0017 | 0.000144, 0.00034 | 0.89 to 1.08 |
| fp32 scan with recompute, two code copies | 1602, 1632 | 0.00069 | 0.0019 | 0.000148, 0.00032 | 0.90 to 1.09 |
| BF16 scan, two code copies | 1603, 1691 | 0.00086 | 0.0019 | | 0.90 to 1.11 |
| Final posture, two nodes | 1681, 1714 | 0.00087 | 0.0024 | 0.00011, 0.00027 | 0.86 to 1.12 |
| Final posture, 4 nodes | 1689, 1713 | 0.00065 | 0.0023 | | 0.91 to 1.12 |

| Comparison | Jobs | Steps | Result |
|---|---|---|---|
| MXFP8 against BF16 | 1602 against 1589 | 1 to 30 | Loss gap +0.0028 (sd 0.0012), slope +0.0005 over the window |
| BF16 scan against the fp32 scan with recompute, soak | 1597 against 1602 | 1 to 100 | Steps 11 to 100: loss gap +0.00001, sd 0.00035, gradient-norm ratio 1.002. 0 NaN |
| EP overlap on packed batches, fp32 scan with recompute, EP8 | 1633 against 1632 | 1 to 30 | Loss sd 0.00067, largest 0.0016. Load-balancing sd 0.000121. Inside the noise |
| EP overlap, BF16 scan, EP4, two code copies | 1650 against 1614 | 1 to 30 | Loss sd 0.00087, largest 0.0031 at one step. Load-balancing sd 0.00013 |
| EP overlap, BF16 scan, EP4, 8 nodes | 1659 against 1658 | 1 to 60 | Loss sd 0.00063, largest 0.0023. Load-balancing sd 0.00012 |
| EP overlap, BF16 scan, EP8, same code | 1681 against 1691 | 1 to 30 | Loss sd 0.00081, largest 0.0021. Load-balancing sd 0.000143. Inside the noise |

In steps 1 to 4 the gradient norm moved by up to 2.2x between postures. Changing the scan (job 1597 against 1602) and
changing EP (job 1614 against 1597) moved it by similar amounts. Two runs of one posture (jobs 1597 and 1603) agreed
within 0.9% over those steps. Steps 1 to 4 therefore amplify any numerical change of posture. The BF16 scan moved them
no more than EP4 did. From step 5 the gradient norms stayed inside the noise of identical runs.

### Trace

A torch profiler trace of job 1681 records step 31 on rank 0, 7.36 s long. The step held 8 packs per GPU. Job 1700
split the kernel time of the trace by family. Hidden time ran beside kernels of another family.

| Item | Exposed s | Hidden s |
|---|---|---|
| HybridEP dispatch, combine and device sync | 0.805 | 1.394 |
| Permute and unpermute | 0.404 | 0.088 |
| Expert GEMM | 0.966 | 0.615 |
| Dense GEMM | 0.816 | 0.273 |
| Mamba scan | 0.885 | 0.247 |
| Data-parallel NCCL | 0.056 | 0.006 |
| GPU idle | 0.451 (6.1%) | |

The host waited at four call sites in each step. A wait idles the GPU only when its queue runs empty.

| Call site | Calls per step | Host wait s |
|---|---|---|
| `3rdparty/Megatron-LM/megatron/core/transformer/moe/moe_utils.py`, `switch_load_balancing_loss_func` | 184 | 0.668 |
| `3rdparty/Megatron-LM/megatron/core/transformer/moe/fused_a2a.py`, HybridEP dispatch forward | 184 | 0.655 |
| `3rdparty/Megatron-LM/megatron/core/rerun_state_machine.py`, `validate_result` | 38 | 0.336 |
| `3rdparty/Megatron-LM/megatron/core/extensions/transformer_engine.py`, attention forward | 48 | 0.273 |

The first call site syncs because padded batches turn the token count into a GPU tensor. The fused auxiliary-loss
kernel takes an integer. The 184 calls are 23 MoE layers times 8 packs per GPU. The third call site is the loss NaN
check of the baseline. The GPU idled 0.451 s per step, which bounds the cost of these waits. The cost of the first
two waits is estimated at 2% to 4% of the step.

### Closed levers and memory

Each lever ran against a reference on the same packs. Timer level 1 and delayed weight gradients appear under
"Findings across the stages".

| Lever | Ratio | Jobs | Nodes | Memory | Reason closed |
|---|---|---|---|---|---|
| CP2 | 0.80x | 1591 against 1589, BF16 | Across nodes | Half the activations of CP1 | Slower. The memory it frees has no use at micro-batch 1 |
| fp32 scan without recompute | 1.051x | 1573 against 1572, steps 3 to 10 | Across nodes | | The BF16 scan ran 1.172x with less memory (job 1574 against 1572) |
| Sync-free HybridEP with permute fusion | None | 1613 | | | It hung in the first backward pass |
| EP4 | 1.072x without the EP overlap, 1.027x with it | 1614 against 1603, 1653 against 1633 | Across nodes | +14.7 GB (job 1614 against 1603) | Outside the EP8 constraint |
| NVFP4 | Not run | | | | Outside the precision constraint |

The final posture peaks at 213.5 to 214.6 GB reserved on 4 nodes, 74% of the device. No measured lever inside the EP8
constraint turns the remaining memory into speed.

## Caveats

- The stage-1 score clears the 35% line by 0.25 points (jobs 1746 and 1751).
- No run covers stage-1 steps 460 to 1,000. In job 1745 the load-balancing loss rose from 1.05 at step 200 to 1.15 at
  step 300. The learning rate peaked at step 299. The Isambard run's rise peaked at step 518. The settled rate past
  step 460 is unmeasured.
- Every 4-node score includes slinky-3 (see "Node speed sets the pace of a multi-node run").
- A ratio marked "across nodes" includes the speed difference of its nodes.
