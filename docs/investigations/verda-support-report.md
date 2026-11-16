# Verda support and its performance study

This report covers the work that runs this repository on Verda beside Isambard. Verda has 12 nodes of 8 B300 GPUs, on
x86_64 and InfiniBand. Isambard has GH200 nodes of 4 GPUs on Slingshot. The first part of the work runs every pipeline
on Verda. The performance study searches for system settings that meet an MFU target on 4 nodes. The runs date from
2026-11-14 to 2026-11-16.

Job ids are SLURM jobs on Verda. The log of job N is `/shared/geodesic/logs_kyle/megatron_runs/train-N.out`. Each number
below cites the job it comes from. "Sources" at the end lists the ledgers behind each section.

## Terms

| Term | Meaning |
|---|---|
| MFU | Model FLOP utilisation: the model's FLOP/s per GPU over 2,250 TFLOP/s, the B300's dense BF16 peak. `scripts/telemetry/score_run.py` computes it from a log |
| Cell | One model at one stage of the 30B baseline, for example Super midtraining |
| Posture | The system settings of a run: parallelism, recompute, precision, dispatcher, fusions, overlaps and launcher settings |
| Lever | One setting changed against a reference run. A closed lever is one the study measured or costed and did not adopt |
| Ratio | The reference's step time over the lever's step time. A ratio above 1 means a faster step |
| Arm | The Verda stage configs of the baseline, `configs/control_pretraining/30b_baseline_verda/` |
| Window | The steps that the scorer averages |
| Pool A | Nodes slinky-1, slinky-2, slinky-3 and slinky-7. Long runs and the first scores ran here |
| Pool B | Nodes slinky-8 to slinky-11. Jobs of an hour or less ran here. A pool-A equivalent is 1.0205 times a pool-B step time ("Node speed between the pools") |
| TP, PP, CP, EP, DP | Tensor, pipeline, context, expert and data-parallel sizes. GBS is the global batch size |
| HybridEP | The backend of the flex token dispatcher. Its dispatch sends tokens to their experts. Its combine returns them |
| EP overlap | `overlap_moe_expert_parallel_comm`: one micro-batch's dispatch and combine run beside another's compute |
| `[moe_act]`, `[moe]` | Selective recompute of the routed experts' activation, or of the whole MoE layer |
| BF16 scan | `ISAMBARD_FP32_SSM_STATE=0`: the Mamba scan without an fp32 state. The fp32 scan is `1` |
| TE CUDA graphs | `model.cuda_graph_impl: transformer_engine` over the scopes attn, mamba, moe_router and moe_preprocess, warmup 3 |
| Router fusion | `model.moe_router_fusion`: Transformer Engine's fused top-k and aux-loss kernels |
| Receive ratio | The tokens a rank receives in one dispatch over its balanced share |
| Base-Chat-Init | The Super Base release with its 1,188 zero embedding rows copied from the instruct release (`docs/verda/models.md`) |
| Staged arm | The first Super stage configs, before any screen |
| Carried | A screen's verdict when its posture is the reference of the next screens |
| Settling rule | Routing counts as settled at the first 30-step window from step 300 on that meets three tests. The median step time is within 1% of the previous window's. The in-window fit moves under 1%. The largest per-rank receive ratio falls by under 1% |
| f | The fraction of the optimizer state that the host holds |

Nano tables give memory in GB of the device's 287.8 GB, as the Nano campaign page does. Super and Ultra tables give
GiB of its 268 GiB, as their ledgers do. PyTorch can allocate about 257 GiB of it.

## Verda beside Isambard

### The install

Both clusters run the same pipeline scripts. A cluster profile, `pipeline/sh/clusters/<name>.sh`, holds every value
that differs. The launchers detect the cluster from the host name. `docs/verda/quickstart.md` gives the path from a
new account to a first run.

| Part | Function | Page |
|---|---|---|
| Verda profile | Storage roots, the image tag, the overlay packages, the GPU peak, the fabric and the `/local` caches | `docs/verda/cluster.md` |
| Submit front door | `pipeline_cluster_submit.sh` turns a script's `#PIPELINE-SUBMIT` line into the options a Verda node needs | `docs/verda/environment-pipeline.md` |
| Image staging | The launchers run an unpacked copy of the image on each node's `/local` instead of the SIF | `docs/verda/storage.md` |
| Config relocation | `configs/clusters/verda_from_isambard.yaml` moves an Isambard config's storage paths. The launcher refuses a config that refers to the other cluster's storage | `docs/verda/storage.md` |
| Input staging | Import, coherence, tokenize and pack read a copy of their inputs on the node's disk | `docs/verda/storage.md` |
| Data | The build commands and job sizes of the quickstart datasets and the baseline blends | `docs/verda/data.md` |
| Quickstarts | `configs/quickstart/verda/`: each Isambard quickstart with Verda storage, plus three Verda-only files | `docs/verda/models.md` |

| Check before the performance study | Job | Result |
|---|---|---|
| Full unit suite | 1734 | 9,963 passed, 32 skipped |
| Environment validation | 1736 | 21 of 21 stages pass |
| Two-node all-reduce smoke | 1737 | 706.4 GB/s bus bandwidth, against a floor of 400 GB/s |

### The quickstart scaling study

`configs/quickstart/verda/campaign-notes.md` records every quickstart run of 2026-11-14, with its window and memory.
The table gives the narrowest and the widest scored run of each regime. All ran on NeMo 26.04.

| Regime | Layout | Narrowest run | MFU | Widest run | MFU |
|---|---|---|---|---|---|
| Nano pretraining | EP4 | 1 node, job 676 | 18.46% | 8 nodes, job 887 | 17.63% |
| Nano pretraining | EP1, EP overlap off | 1 node, job 740 | 21.55% | 8 nodes, job 888 | 20.03% |
| Nano SFT | EP4 | 1 node, job 878 | 21.44% | 8 nodes, job 886 | 20.93% |
| Nano midtraining | CP2 EP4 | 1 node, job 901 | 14.71% | 4 nodes, job 908 | 11.97% |
| Super pretraining | PP4 EP2 on 1 node, PP2 EP4 on 4 | 1 node, job 815 | 16.93% | 4 nodes, job 1055 | 17.16% |
| Super pretraining | PP8 EP4, the Isambard topology | 4 nodes, job 806 | 15.45% | 8 nodes, job 892 | 15.25% |
| Super SFT | PP2 CP4 EP4 on 2 nodes, PP8 CP4 EP4 on 4 | 2 nodes, job 1042 | 15.70% | 4 nodes, job 1056 | 13.94% |
| Super CPT | PP4 EP2 | 1 node, job 1046 | 17.56% | | |
| Ultra SFT | PP6 EP8 | 6 nodes, job 1066 | 7.44% | | |

The campaign notes give the evidence for each of these findings.

- Nano midtraining had 25, 34 and 51 slow steps on 1, 2 and 4 nodes (jobs 901, 907, 908). Each fell where a
  dataloader worker first read one of the ten corpora.
- Model imports that memory-mapped their safetensors on `/shared` stalled or failed (jobs 721, 731, 737). The same
  import from a copy on the node's disk took 2 min 21 s (job 873).
- The SFT packer draws its shuffle from numpy's unseeded global state. Two builds of one shard can differ in packs.

### Re-baseline of the quickstarts on NeMo 26.10.00

The performance study moves Verda to NeMo 26.10.00. Each scored 26.04 row below runs again on the new image, on pool
B. A repeat keeps its original job's launch line. It changes the image and the W&B name.

The packed SFT rows also turn router fusion off. Where the 26.04 job ran on another node, a 26.04 control runs on the
repeat's node. The Ultra SFT quickstart and the multi-node scaling runs have no repeat.

| Row | Nodes | Layout, GBS | 26.04 job, node | 26.04 s/iter, MFU | 26.10.00 job, node | 26.10.00 s/iter, MFU | Ratio |
|---|---|---|---|---|---|---|---|
| Nano pretraining | 1 | EP4, 64 | 676, slinky-2 | 3.283, 18.46% | 1856, slinky-11 | 3.474, 17.45% | 0.945x |
| | | | 1871 control, slinky-11 | 3.448, 17.58% | | | 0.993x of the control |
| Nano SFT | 1 | EP4, 8 | 878, slinky-3 | 1.660, 21.44% | 1857, slinky-8 | 1.629, 21.84% | 1.019x, fusion off |
| Nano midtraining | 1 | CP2 EP4, 8 | 901, slinky-10 | 2.419, 14.71% | 1858, slinky-10 | 2.605, 13.66% | 0.929x |
| | | | 1870 control, slinky-10 | 2.444, 14.56% | | | 0.938x of the control |
| Super pretraining | 1 | PP4 EP2, 192 | 815, slinky-10 | 39.131, 16.93% | 1859, slinky-10 | 38.292, 17.30% | 1.022x |
| Super CPT | 1 | PP4 EP2, 192 | 1046, slinky-11 | 37.723, 17.56% | 1860, slinky-11 | 36.835, 17.98% | 1.024x |
| Super SFT | 2 | PP2 CP4 EP4, 32 | 1042, slinky-2 and 10 | 14.961, 15.70% | 1861, slinky-8 and 9 | 14.179, 16.56% | 1.055x, fusion off |

- The Nano pretraining quickstart runs at the old rate on the new image. Its first pair crossed pools. The 26.04
  control on slinky-11 runs 0.952x of job 676 on slinky-2.
- The Nano midtraining quickstart runs 6% slower on 26.10.00, against both 26.04 runs on its node. The two 26.04 runs
  agree within 1%. The production midtraining arm gains 3.3% on the same image (see "Nano"). The quickstart runs CP2 and
  EP4, where the arm runs CP1 and EP8. The study does not isolate the cause of the quickstart's loss.
- The Super rows gain 2.2% to 5.5%. The SFT row crosses nodes.

## The performance study

### Constraints

The study held these constraints in every cell.

| Constraint | Fixed value |
|---|---|
| Width | 4 nodes, 32 B300 GPUs, for every score |
| Baseline | The exact data, global batch, schedule, optimizer and tokenizer of the 30B Control Pretraining baseline (`configs/control_pretraining/30b_baseline/`) |
| Targets | At least 35% MFU for pretraining and 30% for midtraining and SFT, each in two agreeing 4-node runs. The 32,768-token cells started at 35% and moved to 30% after the Super SFT stall (job 1755) |
| Expert parallelism | 8 in every posture, so the expert all-to-all stays inside a node |
| Precision | The dense layers use MXFP8 only under a passing BF16 loss pair. Routed experts run in BF16. The study does not adopt NVFP4 in any cell. The study allows the BF16 scan in the 32,768-token cells, with both NaN checks on |
| Change type | Configuration over code |
| Software | No Megatron-LM version change. The container image is free to change. The scored runs of the old image repeat on the new one, except the multi-node quickstart scaling runs and Ultra |
| Recompute | Only where memory forces it |
| Memory | About 90% of the device in use where more memory gives more speed. An allocator retry disqualifies a posture |
| Pretraining score | From random weights, after the expert routing settles |
| Standardisation | The same optimisations in every cell where the code allows them |
| Nodes | The study compares a lever with a reference on the same pool. Pool B runs only while no other account queues jobs |

### Method

- Each score uses `score_run.py` with the run's config. Super windows are steps 6 to 30 for pretraining and 6 to 25 for
  the 32,768-token cells. Nano windows are those of `docs/verda/models.md`, "Scoring".
- A loss pair compares two runs of equal DP width, data and seed, step by step. The noise floor is the spread between
  two runs of one posture.
- A trace records one step on rank 0 with the torch profiler. A CPU job analyses it.
- Every Super screen takes all four nodes of a pool. By the arithmetic of the Super pretraining ledger, the PP1
  optimizer shard alone is 84.3 GiB per GPU on 2 nodes.

### Results

All runs below use NeMo 26.10.00. None had an allocator retry, a NaN step or a skipped step.

| Model | Stage | Target | Jobs | Pool | s/step | MFU | Peak allocated / reserved | Final posture |
|---|---|---|---|---|---|---|---|---|
| Nano | Pretraining | 35% | 1829, 1830 | B | 13.377, 13.380 | 36.25%, 36.24% | 174.9 / 215.5 GB | The arm with recompute off |
| Nano | Midtraining | 30% | 1822, 1823 | B | 14.836, 14.815 | 38.37%, 38.43% | 178.6 / 221.0 GB | The arm |
| Nano | SFT | 30% | 1824, 1825 | B | 14.271, 14.257 | 39.89%, 39.93% | 174.5 / 216.0 GB | The arm with router fusion off |
| Super | Pretraining | 35% | 1849, from random weights, steps 421-460 | A | 51.268 | 34.45% | 214.8 / 233.8 GiB | PP1 CP1 `[moe_act]`, TE CUDA graphs, EP overlap off |
| Super | Pretraining, warm | 35% | 1813, warm start | A | 44.688 | 39.52% | 210.7 / 227.3 GiB | The same |
| Super | Midtraining | 30% | 1840, 1841 | B | 55.441, 55.460 | 33.89%, 33.87%. Pool-A equivalent 33.21%, 33.19% | 227.6 / 235.2 GiB | PP2 CP1 `[moe]`, BF16 scan, router fusion on |
| Super | SFT | 30% | 1827, 1833 | B | 57.479, 57.511 | 32.68%, 32.67% | 227.1 / 234.6 GiB | PP2 CP1 `[moe]`, BF16 scan, router fusion off |

Every Super posture runs EP8 and micro-batch 1. It runs MXFP8 dense layers with FP8 weights, BF16 experts on
`torch_grouped` and HybridEP. Super pretraining meets 35% only warm. From random weights it has one run, whose routing
still drifts ("Super pretraining").

### Nano

`docs/investigations/nano30b-verda-perf-campaign.md` gives the Nano postures, ladders, loss pairs and traces. That
study ran on NeMo 26.04 on pool A. Its scores on 26.04 stand: 35.25% (jobs 1746, 1751), 37.36% to 37.37% (jobs
1687, 1715) and 39.74% (jobs 1689, 1713). Here each cell repeats on 26.10.00 on pool B, beside a 26.04 run on the same
pool.

| Cell, window | Job | Image | Posture | s/step | Tokens/s/GPU | TFLOP/s/GPU | MFU | Allocated / reserved GB | Same-pool factor |
|---|---|---|---|---|---|---|---|---|---|
| Pretraining, 421-460 | 1829 | 26.10.00 | Arm, recompute off | 13.377 | 39,193 | 815.5 | 36.25% | 174.9 / 215.5 | 1.0323x |
| | 1830 | 26.10.00 | The same | 13.380 | 39,184 | 815.3 | 36.24% | 174.7 / 215.4 | |
| | 1851 | 26.04 | The same | 13.811 | 37,962 | 789.9 | 35.11% | 174.6 / 215.5 | Reference |
| Midtraining, 11-40 | 1822 | 26.10.00 | Arm | 14.836 | 35,340 | 863.4 | 38.37% | 178.6 / 219.2 | 1.0329x |
| | 1823 | 26.10.00 | Arm | 14.815 | 35,388 | 864.6 | 38.43% | 178.6 / 221.0 | |
| | 1852 | 26.04 | Arm | 15.314 | 34,236 | 836.5 | 37.18% | 178.6 / 217.9 | Reference |
| SFT, 21-60 | 1824 | 26.10.00 | Arm, router fusion off | 14.271 | 36,738 | 897.6 | 39.89% | 174.5 / 215.8 | 1.0102x net |
| | 1825 | 26.10.00 | The same | 14.257 | 36,775 | 898.5 | 39.93% | 174.4 / 216.0 | |
| | 1853 | 26.04 | Arm, router fusion on | 14.409 | 36,386 | 889.0 | 39.51% | 174.4 / 214.5 | Reference |

- The pretraining runs resume the step-400 checkpoint of job 1745. That job settled the routing on 26.04. Only steps
  401 to 460 run on the new image.
- Each 26.10.00 pair agrees within 0.14%. The loss gap between the images falls inside the run-to-run spread.
- The SFT factor is net of router fusion off. On one node the image gives 1.0146x (job 1799 against 1800, both
  fusion off). Fusion off gives 0.9893x (job 1800 against 1714).

The recompute pair ran on one node, slinky-9, at GBS 512, with a scoring window of steps 6 to 25.

| Job | Posture | s/step | MFU | Allocated / reserved GB |
|---|---|---|---|---|
| 1817 | The arm, `[moe_act]` | 13.169 | 36.82% | 180.0 / 214.8 |
| 1816 | Recompute off | 13.135 | 36.91% | 196.4 / 235.3 |

Recompute off is 1.0025x the arm and faster at 17 of 20 steps. Two identical runs on one node differ by 0.07% (jobs
1707, 1718). Recompute off costs 16.3 GB allocated and 20.6 GB reserved. It leaves the loss at the identical-run floor.
The pretraining arm therefore drops `[moe_act]`. The midtraining and SFT arms recompute nothing.

Jobs 1854 and 1855 screen TE CUDA graphs against the EP overlap on Nano. Each runs on one node beside an arm run. Graphs
need the overlap off and `model.use_te_rng_tracker: true` (jobs 1843 and 1845 stop without the tracker).

| Cell | Graphs run | s/step, MFU | Arm run, same node | s/step, MFU | Arm against graphs |
|---|---|---|---|---|---|
| Pretraining | 1854, slinky-9 | 14.371, 33.74% | 1842 | 13.143, 36.89% | 1.093x |
| Midtraining | 1855, slinky-10 | 16.185, 35.18% | 1844 | 14.635, 38.90% | 1.106x |

Both captures succeed. The losses agree with the arm runs. The EP overlap gives Nano more than graphs do. The Nano arms
therefore keep the overlap and run no graphs.

### Super pretraining

The cell reads the stage-1 blend at 8,192 tokens and GBS 2,048. The 35% target needs 50.46 s per step. Screens run
warm from Base-Chat-Init for 30 or 31 steps. The staged arm (job 1753) runs PP1 CP1 EP8, `[moe_act]` and the EP
overlap.

| Job | Image, pool | Run | s/step | MFU | Allocated / reserved GiB | Ratio | Verdict |
|---|---|---|---|---|---|---|---|
| 1753 | 26.04, A | P1: the staged arm, EP overlap on | 52.072 | 33.92% | 220.2 / 246.0 | 0.989x of 1754 | Closed |
| 1754 | 26.04, A | P2: EP overlap off | 51.493 | 34.30% | 210.8 / 213.1 | Reference | Carried |
| 1779 | 26.04, A | P7: P2 with permute fusion into HybridEP | 52.228 | 33.81% | 210.8 / 213.2 | 0.986x of 1754 | Closed |
| 1801 | 26.10.00, A | P2i: P2 on the new image | 48.858 | 36.15% | 210.8 / 213.3 | 1.0539x of 1754 | Carried |
| 1802 | 26.10.00, A | P11: MXFP8 experts through TE's grouped ops | Hung in step 1 | | | | Closed |
| 1813 | 26.10.00, A | P10i: P2i with TE CUDA graphs | 44.688 | 39.52% | 210.7 / 227.3 | 1.0933x of 1801 | Carried |
| 1819 | 26.10.00, B | P2i on pool B | 47.875 | 36.89% | 210.8 / 213.4 | Reference | |
| 1818 | 26.10.00, B | P5i: P2i with recompute off | 47.103 | 37.49% | 247.3 / 249.1 | 1.0164x of 1819 | Closed by memory |
| 1826 | 26.10.00, B | P2i in BF16, the loss pair | 50.017 | 35.31% | 214.6 / 217.3 | MXFP8 is 1.0447x | MXFP8 kept |

- The EP overlap hides 3.4 s of HybridEP per step. The expert GEMMs beside it slow by 2.35 s (traces 1763, 1765).
- The fused permute runs inside the dispatch's 24 SMs. HybridEP's time on the critical path grows from 12.3 to 14.2 s
  per step (trace 1786).
- P11 runs TE's unfused op sequence on all 32 ranks. One rank stops issuing work in step 1's backward. The cause is
  unverified (job 1802).
- TE CUDA graphs with expandable segments need `NCCL_GRAPH_REGISTER=0`. Job 1806 stops at setup without it.
- Recompute off fits a warm run at 92.9% (job 1818). With graphs its reserved memory comes to about 264 GiB by the
  ledger's arithmetic, over the allocator's limit. Graphs win. `[moe_act]` makes room for the graph pools.

The new image shifts the step's limit from the GPU toward the host. Graphs remove most of the new idle time.

| Seconds per step, iteration 31, rank 0 | P2 on 26.04, trace 1765 | P2i on 26.10.00, trace 1803 |
|---|---|---|
| Compute union | 41.18 | 34.36 |
| HybridEP permute, unpermute and preprocessing | 5.05 | 1.51 |
| Router: fused top-k and aux-loss scores | 2.99 | 0.58 |
| Squared-ReLU backward | 1.83 | 1.33 |
| GPU idle | 2.62 | 8.43, about 6.4 without the profiler |
| NCCL all-gather calls | 45 | 2,605 |

The 2,560 new all-gathers carry HybridEP's routing map (trace 1803). DeepEP 10d4dd7 turns its NVLink all-gather off by
default. The pinned Megatron-LM passes no argument for it, so no configuration can change it.

Job 1868 (P8r) resumes the step-400 save with 48 HybridEP SMs, `moe_flex_dispatcher_num_sms: 48`, on B1's nodes. Over
steps 411 to 450 it runs 50.942 s per step against B1's 50.950 s, which is 1.0002x. Its peak is 214.8 / 233.1 GiB,
against B1's 214.8 / 233.8 GiB. The lever is closed.

Levers that did not run:

- 128 permute and unpermute blocks (P9, P9i) lost their target. The new image cut those kernels to 1.51 s per step
  (trace 1803). The study cancelled jobs 1794 and 1805 for that reason.
- BF16 fused experts (P11b). The study cancelled job 1807 because BF16 cannot take TE's fused grouped MLP. The unfused
  path drops the recompute and needs about 250 GiB by the ledger's arithmetic.
- The EP overlap on 26.10.00. The hybrid model's overlap schedule asserts `cuda_graph_impl == "none"`
  (`models/hybrid/model_chunk_schedule_plan.py` 195-197). The overlap and graphs exclude each other in the pinned fork.

Steps 1 and 2 of job 1813 equal those of job 1801 to every printed digit. Step 3 differs by 1.17e-4. On Super, equal
runs first diverge at step 3 (jobs 1753 and 1754). Over steps 4 to 30 the gap stays within 6.7e-5, inside the
identical-run spread. The image moves step 1 by 3.1e-5 (job 1801 against 1754).

**From random weights.** Run A (job 1834) trains to step 400 in P2i's posture, with a receive counter on. It saves
the step-400 checkpoint. Runs B resume it without the counter and score steps 421 to 460. Runs B1 and B2 (jobs 1849,
1850) carry TE graphs. Run B3 (job 1837) repeats P2i's posture.

| Run A window | s/step | MFU | Mean receive ratio |
|---|---|---|---|
| 20-50 | 54.407 | 32.46% | 2.13 over 21-50 |
| 301-350 | 51.138 | 34.54% | 1.74 to 1.77 |
| 351-400 | 52.107 | 33.89% | 1.86 to 1.88 |

The study's settling script reads job 1834's log in 30-step windows. The ratio is the mean over ranks of each rank's
largest receive ratio per step.

| Steps | 21-50 | 51-80 | 81-110 | 111-140 | 141-170 | 171-200 | 201-230 | 231-260 | 261-290 | 291-320 | 321-350 | 351-380 | 381-400 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Median s/step | 53.897 | 50.647 | 49.488 | 49.330 | 49.673 | 50.050 | 50.214 | 50.209 | 50.027 | 50.518 | 51.547 | 51.845 | 52.369 |
| Receive ratio | 2.13 | 1.53 | 1.34 | 1.34 | 1.65 | 1.79 | 1.85 | 1.89 | 1.72 | 1.74 | 1.77 | 1.86 | 1.88 |

- Run A takes 5 h 51 min with no NaN, skipped step or allocator retry. Its peak is 218.4 / 222.1 GiB, 82.9%.
- That peak is 7.6 GiB allocated over warm P2i. The save takes 186.5 s for about 1.1 TiB.
- The learning rate warms up to 1e-3 over steps 1 to 299.
- The settling rule passes at steps 351 to 380. A rising ratio counts as "not falling", so the rule passes without a
  plateau. The routing still drifts at step 400.
- Run A's peak plus the 14.0 GiB of graph pools (job 1813 against 1801) is about 236 GiB reserved. A run from random
  weights can therefore carry graphs from step 1.

Run B1 (job 1849) resumes the step-400 save with TE graphs on pool A, from the unpatched pin.

| Run B1 window | s/step | Tokens/s/GPU | TFLOP/s/GPU | MFU | Peak allocated / reserved GiB |
|---|---|---|---|---|---|
| 411-450 | 50.950 | 10,290 | 779.9 | 34.66% | 214.8 / 233.8 |
| 421-460 | 51.268 | 10,227 | 775.1 | 34.45% | 214.8 / 233.8 |

B1 has no NaN, skipped step or allocator retry. Its reserved peak is 87.2% of the device. Its step time still rises
over the window.

| Steps | Step time |
|---|---|
| 405 to 420 | About 50.0 s |
| 421 to 440 | 50.9 s |
| 441 to 460 | 51.6 s |

The receive ratio climbed from step 290 of run A onward, after the learning rate rose to 1e-3. The gradient norm spikes
to 20.5 inside B1's window. The study cancelled runs B2 (job 1850) and B3 (job 1837). The routing had not settled, so
a repeat of the same window cannot give a settled score.

The cell therefore misses 35% from random weights over steps 421 to 460, by 0.55 points. Warm from Base-Chat-Init,
the same posture runs 39.52% (job 1813). A settled score needs a longer run from random weights.

### Super midtraining

The cell reads the stage-2 blend at 32,768 tokens and GBS 512. The 30% target needs 62.6 s per step. The Base release
has zero embedding rows. Their ids occur in all ten stage-2 prefixes (job 1758). At DP 8 the first one enters at step 1.
Runs therefore warm-start from Base-Chat-Init.

| Job | Image, pool | Run | s/step | MFU | Allocated / reserved GiB | Verdict |
|---|---|---|---|---|---|---|
| 1761 | 26.04, A | M1: the staged arm, PP1 CP4 `[moe_act]`, fp32 scan | 72.324 | 25.98% | 229.1 / 231.9 | Starting point |
| 1808 | 26.10.00, A | M3i: BF16-scan soak on long documents | 61.576 | 30.51%, not a score | 214.0 / 216.9 | 0 NaN in 12,800 windows |
| 1809 | 26.10.00, A | M4i: BF16 scan on the real blend | 61.499 | 30.55% | 212.8 / 215.9 | Passes. 1.176x of 1761 |
| 1832 | 26.10.00, A | Repeat of 1809 | 61.493 | 30.55% | 212.8 / 215.7 | Agrees. The fallback pair |
| 1810 | 26.10.00, A | M4i with TE CUDA graphs | Fails at capture, step 4 | | | Closed at CP4 |
| 1840 | 26.10.00, B | PP2 CP1 `[moe]`, router fusion on | 55.441, pool-A equivalent 56.58 | 33.89%, equivalent 33.21% | 227.6 / 234.1 | Carried. 1.087x of 1809 |
| 1841 | 26.10.00, B | Repeat of 1840 | 55.460, pool-A equivalent 56.60 | 33.87%, equivalent 33.19% | 227.6 / 235.2 | Agrees |

- The 1.176x of job 1809 combines the image and the BF16 scan. The 1.087x of job 1840 uses its pool-A equivalent.
- Graph capture fails on all 32 ranks in TE's ring-attention P2P, inside the attn scope (job 1810). The wait uses an
  event that the capture recorded. Eager steps 1 and 2 equal those of job 1809.
- PP2 CP1 `[moe]` is the SFT layout. Each GPU holds whole 32,768-token sequences, so Mamba needs no context-parallel
  all-to-all.
- Its step stays between 55.34 and 55.58 s, where job 1809 ranges from 60.1 to 63.7 s. It runs 640 HybridEP
  dispatches per GPU per step against 2,560.
- The lower-CP search estimates PP2 CP2, PP1 CP2, PP4 CP1, PP4 CP2 and PP8. Each misses on memory or speed.
- `[moe_act]` off misses the 250 GiB reserved line by 2 to 4 GiB on job 1809's measured peak. No screen runs.
- The EP overlap (job 1769) and permute fusion (job 1776) close on pretraining's pairs and never run here.
- Activation offloading closes by arithmetic. The host link cannot carry the bytes inside a micro-batch's forward.

Over steps 1 to 25 the loss gap of job 1809 to job 1761 has a mean of +0.00002. Job 1832 stays within 0.00043 of job
1809. Jobs 1840 and 1841 agree within 0.0001 over steps 1 to 18. Their gap then grows to +0.0036 at step 23 and +0.0043
at step 25.

That is wider than Nano's floor of 0.0005 and the Super SFT pair's 0.0003. A longer identical pair can check this. No
such pair ran.

### Super SFT

The cell reads the packed 25B mix in 32,768-token packs at GBS 512, warm from Base-Chat-Init. Packed SFT runs
micro-batch 1. The 30% target needs 62.6 s per step. Every 26.10.00 run below has router fusion off and the BF16 scan.

| Job | Image, pool | Run | s/step | MFU | Allocated / reserved GiB | Verdict |
|---|---|---|---|---|---|---|
| 1755 | 26.04, A | S1: the staged arm, PP1 CP2 `[moe]`, fp32 scan | 82.472 | 22.78% | 224.3 / 230.3 | Stalls at the 35% target |
| 1811 | 26.10.00, B | PP1 CP1 `[moe]`, norm-input offload | Out of memory in step 2 | | 246.0 / 250.7 at step 1 | Closed by memory |
| 1827 | 26.10.00, B | PP2 CP1 `[moe]` | 57.479 | 32.68% | 227.1 / 234.6 | Passes 30% |
| 1833 | 26.10.00, B | Repeat of 1827 | 57.511 | 32.67% | 227.1 / 234.3 | Agrees |

The trace of S1 (job 1756) splits its gap to pretraining's per-token rate into three costs.

| Cost in S1, per step | Seconds |
|---|---|
| `[moe]` recompute | 14.75 |
| Context parallelism 2: Mamba layout, all-to-all and attention CP | 12.11 |
| fp32 scan: 1.80 measured, 3.37 inferred | 5.17 |

- Memory forces the first two at EP8 and micro-batch 1. CP1 fits only with the weights and gradients split over two
  pipeline stages (job 1811).
- Job 1827 is the BF16-scan soak: 0 NaN in 12,800 packs, with both NaN checks on.
- On the same packs, job 1827's loss gap to job 1755 has a mean of -0.00002. Its largest step is 0.0003.
- `[moe_act]` at CP1 needs about 445 GiB. Host offload of the expert activations loses about 25 s net at 55 GB/s.
- The Super SFT recipe turns CUDA graphs off for packed sequences (`nemotron_3_super.py` 175-178).
- The EP overlap refuses `[moe]` recompute and PP without virtual stages.
- The SFT ledger reports both runs as measured on pool B, with no pool correction.

### MXFP8 against BF16 on Super

Jobs 1819 (MXFP8) and 1826 (BF16) run P2i's posture on pool B. They share the nodes, the warm start and the data
order over 30 steps.

| Measure | Value |
|---|---|
| Speed | MXFP8 1.0447x: 47.875 against 50.017 s |
| Loss gap, MXFP8 minus BF16, step 1 | +0.0034 |
| Mean gap over steps 1 to 30 | +0.00387 |
| Slope of the gap over steps 1 to 30 | -1.07e-4 per step, standard error 3.8e-5: closing |
| Peak allocated / reserved | 210.8 / 213.4 against 214.6 / 217.3 GiB |

The gap is about 1.6 times Nano's +0.0022 (jobs 1545, 1544). It closes over the window, so the pair passes. The 30
warm steps run in warmup, at a learning rate of 1e-4 or less. The gap at the full 1e-3 is unmeasured.

### Node speed between the pools

| Pair | Cell | Pool B against pool A |
|---|---|---|
| 1819 against 1801 | Super pretraining, P2i | 1.0205x |
| 1851 against 1746 and 1751 | Nano pretraining. 1851 has recompute off, the others on | 0.9959x |
| 1852 against 1687 and 1715 | Nano midtraining | 0.9950x |
| 1853 against 1689 and 1713 | Nano SFT | 0.9941x |

The pool factor depends on the model, so the study applies it to Super alone. One pair sets it, so it carries about
0.3% of noise. Inside pool A, slinky-3 runs Nano midtraining 2.5% behind slinky-1 (job 1720 against 1703). A
multi-node run moves at its slowest node's pace. The Nano campaign page gives the evidence.

### The move to NeMo 26.10.00

Every Verda run up to 2026-11-15 used NeMo 26.04. The study tested 26.10.00 by configuration alone. The pairs in the
factor table below show its gains. The image then became Verda's default. Isambard's profile and its derived paths stay
unchanged.

| Library | 26.04 | 26.10.00, measured in job 1787 |
|---|---|---|
| torch | 2.11.0a0 | 2.14.0a0 |
| CUDA | 13.1 | 13.4 |
| Transformer Engine | 2.14.1 | 2.20.2 |
| NCCL | 2.29 | 2.31.2 |
| DeepEP | 1.2.1+34152ae | 1.2.1+10d4dd7 |
| transformers | 5.3.0 | 5.17.0 |

CUDA 13.4 runs on the host driver 580.178 through forward compatibility (job 1787). The two-node smoke measures 714.3
GB/s (job 1788).

| Cell | Pair | Location | Factor |
|---|---|---|---|
| Nano pretraining | 1790 against 1718 | One node, slinky-7 | 1.0207x |
| Nano pretraining | 1829, 1830 against 1851 | 4 nodes, pool B | 1.0323x |
| Nano midtraining | 1791 against 1703 | One node, slinky-1 | 1.0335x |
| Nano midtraining | 1822, 1823 against 1852 | 4 nodes, pool B | 1.0329x |
| Nano SFT, the image alone | 1799 against 1800, both fusion off | One node, slinky-2 | 1.0146x |
| Nano SFT, net of fusion off | 1824, 1825 against 1853 | 4 nodes, pool B | 1.0102x |
| Super pretraining | 1801 against 1754 | 4 nodes, pool A | 1.0539x |

Super midtraining and SFT have no pair for the image alone. Their 26.10.00 runs also change the scan or the layout.

Breaks on the new image and the fix for each:

| Break | Job | Fix |
|---|---|---|
| Packed SFT stops at step 1. TE 2.20's fused aux loss needs an int64 token count. Megatron passes a float under a padding mask | 1792 | `docs/verda/environment-pipeline.md` ("The newer image") |
| `nv-grouped-gemm` fails to build. torch 2.14's headers need C++20 | 1777 | `docs/verda/environment-pipeline.md` ("The newer image") |
| The image has no `pytest-xdist` or `pytest-rerunfailures` | 1789 | `docs/verda/environment-pipeline.md` ("The newer image") |
| Energon 7.4.1 changes the sample and batch API: 23 test failures | 1828 | `docs/verda/environment-pipeline.md` ("The newer image") |
| datasets 4.8.5 repeats a dropped Hub read itself, so two retry loops nest. It also infers a JSON stream's features | 1815 | `scripts/data/hub_read_retry.py` probes the library and runs one loop. The data pipeline refuses a stream whose features the loader inferred |
| transformers 5.17 drops `Gemma3Config.rope_theta` and the Nemotron-H embedding rename | 1815, 1828 | The tests read their fixture values. The small Nemotron-H fixture renames only an old name |
| huggingface_hub 1.33 lists a commit's tree over the network, even when cached | 1828 | `prepare_revisions` passes `local_files_only` under `HF_HUB_OFFLINE` |
| TE 2.20 raises another error type for a None parameter in the L2 norm. TE 2.18 and later check the fused router's int64 count | 1798, 1828 | Two tests accept both error types. The padding-mask router test is a strict xfail on TE 2.18 and later |
| DeepEP 10d4dd7 no longer reads the pinned count that a hazard test needs | 1798 | The test skips through a probe. The tests of Megatron's protection pass |
| torch 2.14 moves a warning's line, which 8 export goldens record | 1815 | The golden helper drops warnings from outside the repository |
| A thread race in a checkpoint test leaves `torch.distributed.is_initialized` patched. The race predates the image | 1815 | The patches wrap the threads |

The current overlay is `nemo_26.10.00_8499eb66`. Job 1863 installs it. Its
validator passes 20 of 20 stages.

Job 1862 runs the whole unit suite on the staged change in both images.

| Image | Passed | Skipped | Expected failures | Failed |
|---|---|---|---|---|
| 26.04 | 10,034 | 41 | 0 | 0 |
| 26.10.00 | 10,021 | 51 | 2 | 1 |

The one failure is the export of Nemotron-H's block names under transformers 5.17. The fix maps both spellings. Job 1864
reruns the export and PEFT tests in both images: 493 passed and 1 skipped in each. Jobs 1872 and 1873 rerun the files
that changed after job 1864, in both images, with no failure. Job 1828 ran the suite before the last fixes: on
26.10.00, 26 failed with 8 errors.

#### Megatron-LM

The pinned commit stays `4c672ad8`. Patch 0002
(`3rdparty/patches/megatron-lm/0002-fix-cuda-graph-zeros_like-0dim-tensor.patch`) changes `ArgMetadata.zeros_like`,
which only `cuda_graph_impl: local` uses. `TECudaGraphHelper`, the class behind `transformer_engine`, never calls it. TE
CUDA graphs therefore do not need the patch.

Job 1849 runs TE graphs from the unpatched pin. Its log records the capture before step 404, 10.9 s on rank 0. At first
the study held that graphs needed the patch. On that view it cancelled screen 1795 without cause. Jobs 1810 and 1813 ran
from a copy that carried the patch.

## Ultra (550B-A55B)

Ultra testing ended before any 4-node run of the study's cells. Every Ultra training run used NeMo 26.04, between
2026-11-14 09:44 and 2026-11-15 04:23 UTC. No run used the baseline's exact stage configs. These rows are therefore
system measurements of Ultra on B300, separate from the cell scores. The study re-scored each run on 2026-11-16.
Windows hold 3 to 8 steps, so a rate holds to about 2%.

### 8 nodes, 64 GPUs

Each run below has TP1, micro-batch 1 and a warm start from the Ultra Base release. The CPT runs read ClimbMix-Sample,
except job 1542. Job 1519 reads the `sft-warm-start-200k` `no_think` pack. Stacks 1 to 4 each add to the stack before.
The sequence starts at job 1370.

| Job | Run | GBS | Window | s/iter | Tokens/s/GPU | MFU | Peak allocated / reserved GiB |
|---|---|---|---|---|---|---|---|
| 1370 | CPT at 8,192 tokens, PP8 EP8, MXFP8 with MXFP8 experts, full recompute, all-to-all dispatcher | 2,048 | 2-5 | 145.98 | 1,796 | 26.86% | 148.4 / 148.8 |
| 1433 | Stack 1: selective recompute, fusions, HybridEP, TE graphs on attention and Mamba | 512 | 5-12 | 34.05 | 1,925 | 28.79% | 219.9 / 223.4 |
| 1460 | Stack 2: stages balanced by measured compute | 512 | 5-12 | 30.34 | 2,160 | 32.32% | 233.3 / 236.6 |
| 1478 | Stack 3: the BF16 scan | 512 | 5-9 | 26.82 | 2,444 | 36.56% | 240.6 / 243.4 |
| 1487 | Stack 4: BF16 experts on `torch_grouped` | 512 | 5-9 | 27.87 | 2,352 | 35.18% | 239.9 / 254.1 |
| 1493 | Stack 3 in NVFP4 | 512 | 5-9 | 27.22 | 2,408 | 36.02% | 195.5 / 209.1 |
| 1542 | Stack 3 on the stage-1 blend's 13 prefixes | 2,048 | 5-9 | 100.60 | 2,606 | 38.98% | Not logged |
| 1519 | SFT on 32,768-token packs, PP8 EP8 CP4 | 256 | 3-8 | 78.30 | 1,674 | 26.12% | 194.0 / 201.3 |

- Job 1542 is the best Ultra rate measured: 38.98% at the baseline's GBS of 2,048. Its experts run MXFP8 on
  `te_grouped`. The 45-minute limit stops it after 9 of 10 iterations, so its log has no peak-memory line.
- The GBS-512 rows run below the baseline batch. At PP8 with 64 micro-batches per replica, the bubble is 7 of 71
  slots.
- BF16 experts on `torch_grouped` run 3.9% slower per step than MXFP8 experts on `te_grouped` (job 1487 against
  1478). This is Ultra's only measurement of the study's expert rule.
- NVFP4 ties MXFP8 on speed (job 1493 against 1478). The study does not adopt it. Its loss runs 0.003 to 0.005 lower.
  Its gradient norm runs 3 to 7 times smaller. This is unexplained.
- Job 1414 stops before iteration 1 without `NCCL_GRAPH_REGISTER=0`. SFT job 1504 fails at iteration 3. Its graph
  capture cannot hold TE's ring-attention P2P at CP4, as in Super's job 1810.
- Memory-mapped blend reads stall the stage-1 run. Job 1514 runs no iteration in 45 minutes. Plain reads fix it
  (job 1542).
- The first iteration takes 587 to 1,132 s in every run. Each stage compiles its kernels while the next stage waits.

### The 6-node SFT quickstart

`configs/quickstart/verda/nemotron_ultra_quickstart_sft_6node.yaml` runs PP6 EP8 at GBS 16, one pipeline stage per node.
It reads the `sft-warm-start-200k` `no_think` set in packs of 8,192 tokens.

| Job | Window | s/iter | Tokens/s/GPU | TFLOP/s/GPU | MFU | Peak allocated / reserved GiB |
|---|---|---|---|---|---|---|
| 1066 | 51-100 | 5.487 | 498 | 167.5 | 7.44% | 218.0 / 222.9 |

Two micro-batches per replica pass through 6 stages, so each stage works 2 of every 7 pipeline slots. The first
iteration takes 709.7 s. `docs/verda/models.md` gives the memory of each stage.

### Ultra on 4 nodes, from the ledger's arithmetic

The arithmetic reads the pinned Megatron-LM. The 4-node jobs 1351, 1477 and 1503
calibrate it. The study cancelled the 4-node probes (jobs 1846, 1847, 1848) before they started.

- Only PP4 EP8 holds Ultra on 32 GPUs. At PP2, weights and BF16 gradients alone take 183 GiB per GPU.
- Stage 0 holds 4 micro-batches in progress. At 8,192 tokens only full recompute fits without host-memory risk. It holds
  22.5 GiB of activations.
- `[moe]` needs about 82 GiB of activations on stage 0, while `[moe_act]` needs about 205 GiB.

| Optimizer path on a stage-0 GPU | Static memory, GiB | Fits |
|---|---|---|
| BF16 with the remainder store | About 193, measured in jobs 1351, 1477 and 1503 | Yes. Peaks of 216 to 217 for CPT and 241 for SFT at CP2 |
| MXFP8 with FP8 weights | About 290 | No |
| MXFP8 on BF16 weights, host optimizer at fraction f | About 112 + 257 x (1 - f) | Only at f of about 0.6, with a peak of about 237 |

- The host optimizer path keeps FP32 copies, moments and gradients for the GPU share: 16 B per parameter.
- An offloaded parameter costs about 19 to 20 B of host memory. At f = 0.6 a node holds 1.72 to 1.85 TB of its 2.04 TB.
- The GPU needs f of at least 0.6, while the host allows about 0.6. MXFP8 on 4 nodes has almost no margin.
- The arithmetic predicts a pretraining rate of about 230 to 245 s per step, 32% to 34% MFU. For the 32,768-token cells
  it predicts about 27% to 31%.
- A Bridge-only change gives MXFP8 compute on BF16 weights with the remainder store. It holds about 208 GiB static, with
  no host load. The study does not make this change.

## Sources

The scratch paths below are under `/shared/geodesic/scratch_kyle/megatron_verda/`. `scripts/telemetry/score_run.py`
writes a `scores/` directory beside each ledger.

| Section | Source |
|---|---|
| Verda beside Isambard | `configs/quickstart/verda/campaign-notes.md`, `docs/verda/`, the logs of jobs 1734, 1736 and 1737 |
| Quickstart re-baseline and constraints | `quickstart_2610/`, `STATUS.md`, `M2V2_SUPER_BRIEF.md` |
| Nano and the image move | `docs/investigations/nano30b-verda-perf-campaign.md`, `image_2610/ledger.md` |
| Super and Ultra | `m2v2/super_{pretrain,midtrain,sft}/ledger.md`, `m2v2/ultra/report_section.md` and `m2v2/ultra/ledger.md` |
