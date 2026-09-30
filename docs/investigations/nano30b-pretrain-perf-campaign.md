# Nano-30B pretraining performance campaign

**Goal.** Raise the training efficiency of the control-pretraining baseline posture for Nemotron-3 Nano
30B-A3B (from-scratch pretraining, `configs/control_pretraining/30b_baseline/nemotron_nano_30b_baseline_pretrain.yaml`:
TP1·CP1·EP4·PP1·ETP1, seq 8192, only data parallelism crosses nodes), measured on the benchmark below at the
same width and batch as the as-is baseline. The target is a mean step **≤ 5.000 s** (Kyle, 2026-09-29;
against the E-001 baseline of 9.328 s, 7,026 tok/s/GPU and 14.78% MFU at 64 GPUs / GBS 512, that is
**≥ 1.87× ≈ ≥ 13,107 tok/s/GPU ≈ ≥ 27.6% MFU**), reached by levers that keep functional parity with the
production runs: same model and checkpoints, same data and schedule, same loss trajectory. The target was
1.5× at the start, raised to 2× (≤ 4.664 s) on 2026-09-28, and set to ≤ 5.000 s on 2026-09-29.
**Result.** The final posture (see Final posture) averages **4.961 s** over eight runs in four
placement-controlled paired cycles, 95% CI [4.920, 5.002] s: **1.90× the as-is runs of the same cycles** (95% CI
[1.88, 1.92]), 13,209 tok/s/GPU, 27.8% MFU. Its 500-iteration loss stays inside the as-is band (see Functional
parity).
Under the pre-registered rule this is, at k = 4, **goal met on average, not established** (the CI's upper
bound is 5.002 s); the rule's extension to six cycles (6958389, 6958390) was queued on 2026-09-30 and has
not run.

**Benchmark.** `configs/quickstart/nemotron_nano_quickstart_pretrain_baseline.yaml` (named
`nemotron_nano_quickstart_pretrain.yaml` while the campaign ran; that name now holds the fastest
configuration) — the baseline composed with a
small overlay (`base_config:`): GBS 512 on 64 GPUs (8 microbatches per DP replica, the same per-GPU work as
production's GBS 2048 on 256 GPUs), 50 iterations via `train.exit_interval`, no checkpoint load/save.
32 GPUs is the same file with `train.global_batch_size=256`.

**Metric.** Mean step time over iterations 26–50 (node-hours scale with the mean), reported with the
median, p10/p90 and outliers. tokens/s/GPU = GBS × 8192 / (GPUs × mean step). Model TFLOP/s and MFU use the
exact per-token FLOPs of `scripts/nemotronh_flops_estimator.py` and the GH200 dense BF16 peak (989.4 TFLOP/s).
Memory in the tables ("Peak GB") is the one memory line Megatron writes to the log: rank 0, after
iteration 1 (max allocated since start, reserved at that moment). Where headroom matters an entry adds
the W&B summary's run maxima (last rank; `score_run.py --wandb-peak-memory`) and its allocator-retry
counter `memory/mem-alloc-retires`: reserved memory keeps growing after iteration 1 (E-044: 85.1 GB in
the log, 91.7 GB at the end of the run), and retries after iteration 1 never reach the log.

**Method.** Every probe is its own 16-node `pipeline_training_submit.sbatch` job (`--time=00:20:00`;
a 50-iteration probe takes 9–13 min), launched from a read-only snapshot of the code under test under
`/projects/a5k/public/logs/nano_pretrain_perf_campaign/snapshots/` (each carries a `REVISION`, and a
`WORKTREE.diff` when the code is not yet committed); the log's other relative paths (`parity/`, `paired/`,
`robustness/`, `dev/`, `shipped/`) are under the same campaign directory. Kyle chose one short job per probe over a held
allocation (2026-09-29). The same posture measured up to ~4% apart across jobs (Repeatability) until E-047 traced
that spread to placement: a run spanning more than one switch group is 2.5–6.6% slower, while single-group runs
of one posture agree to ~0.5% (SD). From E-047 on only single-group runs are compared with each other. A lever is
credited only when its effect clearly exceeds the spread of the runs it is compared with, or after repeats; a
lever is VOID when its engagement
cannot be shown from the log or config. Levers that change numerics additionally need lm loss (iterations
41–50) inside the band of the other runs and a paired 500-iteration loss-parity run before adoption. Each
credited lever becomes part of the parent of the next probe.

**Research basis.** Six research reports, a campaign design and an adversarial critique, preserved at
`/projects/a5k/public/logs/nano_pretrain_perf_campaign/research/` (the per-hypothesis citations below point
into them and into the primary sources: the Megatron-Core MoE paper, arXiv 2603.07685v2; the Megatron-Bridge
performance guide; upstream Megatron-Bridge / Megatron-LM commits).

## Champion ladder (64 GPUs, GBS 512)

| Step | Lever | Mean step (s) | tok/s/GPU | MFU | × baseline | Entry |
|---|---|---|---|---|---|---|
| 0 | as-is baseline | 9.328 | 7,026 | 14.78% | 1.00 | E-001 |
| 1 | H01 `overlap_param_gather` (under `comm_overlap:`) | 8.464 | 7,743 | 16.28% | 1.10 | E-004 |
| 2 | + H15 bf16 gradient accumulation/reduction | 8.105 | 8,086 | 17.01% | 1.15 | E-008 |
| 3 | + H16 recompute `[moe_act]` only (funded by H15's 19 GB) | 7.075 | 9,263 | 19.48% | 1.32 | E-009 |
| 4 | + host-sync bundle (timers L1, NaN checks off, per-param grad norms off) | 6.862 | 9,551 | 20.09% | 1.36 | E-010 |
| 5 | + manual GC every 10 iterations | 6.744 / 6.494 | 9,717 / 10,092 | 20.4–21.2% | 1.38–1.44 | E-013, E-018 |
| 6 | + H13 fp32-SSM checkpoint patch off | 6.249 / 6.026 / 6.029 | 10,488–10,875 | 22.1–22.9% | 1.49–1.55 | E-019, E-022 |
| 7 | + H32 sync-free grouped-GEMM offsets | 5.968 | 10,981 | 23.09% | 1.56 | E-023 |
| 8 | + H27 FP8 current scaling on dense layers | 5.754 / 5.751 | 11,390 | 23.95% | 1.62 | E-024, E-027 |
| 9 | + H22 HybridEP intra-node dispatcher (sync-free with H32) | 5.407 / 5.388 / 5.394 / 5.409 | 12,115–12,163 | 25.5–25.6% | 1.72–1.73 | E-028, E-030; repeats 6932856, 6932890 (E-039–E-043 block) |
| 10 | + EP all-to-all / compute overlap (combined 1F1B, port of Megatron-LM PR #4798, vendored as patch 0003; `CUDA_DEVICE_MAX_CONNECTIONS=32`) | 5.197 / 5.228 | 12,535–12,611 | 26.4–26.5% | **1.78–1.80** | E-044 |
| 11 | + router fusion (`model.moe_router_fusion=true`) | 5.129 / 5.140 | 12,750–12,778 | 26.8–26.9% | 1.81–1.82 | E-048 |
| 12 | + gc.freeze after the setup collection (`train.manual_gc_freeze=true`) | 5.075 / 5.250 | 12,482–12,913 | 26.3–27.2% | 1.78–1.84 | E-052 |
| 13 | + chunked linear cross-entropy, all 8 logit chunks kept (patch 0005) | 5.002 | 13,102 | 27.55% | 1.86 | E-051 |
| 14 | + BF16 primary weights under FP8 compute (`_bf16_params`; a parity requirement, not a speed lever) | +0.23% (5.132 → 5.144) | — | — | — | E-060 |
| 15 | **final posture** (steps 1–14 together; placement-controlled paired cycles) | **4.961** (95% CI 4.920–5.002) | 13,209 | 27.78% | **1.90** paired (1.88 against E-001) | E-061 |

Every run is its own allocation; where a step was repeated, the table gives each run (see
Repeatability). Steps 12–14 were each measured on the step-11 base (E-048) rather than stacked one on the
other; step 12's two runs sit on either side of step 11 because placement noise is larger than its effect
(E-052); step 15 stacks them all and is the only row measured against the as-is baseline on the same
allocations (Final posture).

## Repeatability

Postures run more than once, each run its own 16-node job. Nodelists (from `sacct`):
N1 = nid[010038,010446,010503-010505,010543,011141,011148,011150,011153,011158,011173,011222,011225,011234,011237];
N2 = nid[011222,011224-011225,011227,011229,011234-011237,011241,011292-011297];
N3 = nid[010231-010232,010241,010243,010256-010258,010274,010286-010289,010300,010311,010313,010328];
N3′ = N3 with nid010038 in place of nid010241;
N4 = nid[011111,011117,011119,011129,011138-011139,011141,011148,011150,011153,011157-011159,011169,011173,011209];
N5 = nid[010444,010446,010450,010464,010474,010503-010506,010523,010529,010533,010543,010545-010546,010548];
N5′ = N5 with nid010541 in place of nid010548;
N6 = nid[010441,010444,010446,010450,010464,010474,010503-010504,010523,010529,010533,010541,010543,010545-010546,010548];
N7 = nid[011110-011111,011117,011119,011129-011130,011138-011139,011141,011148,011150,011153,011157-011159,011169].

| Posture (ladder step) | Entries (job, nodelist) | Mean step per run (s) | Mean of runs (s) | Spread, max − min |
|---|---|---|---|---|
| champion: H01 + host bundle + H15 + H16 + manual GC (5) | E-013 (6932005, N1), E-018 (6932452, N2) | 6.744 / 6.494 | 6.619 | 0.250 s (3.8%) |
| + H13 fp32-SSM off (6) | E-019 (6932453, N3′), E-022 (6932526, N3; 6932527, N4) | 6.249 / 6.026 / 6.029 | 6.101 | 0.223 s (3.7%) |
| + H32 + H27 FP8 on the dense layers (8) | E-024 (6932552, N2), E-027 (6932616, N3) | 5.754 / 5.751 | 5.753 | 0.003 s (0.05%) |
| + H22 HybridEP (9) | E-028 (6932618, N5), E-030 (6932677, N3), repeats 6932856 (N3) and 6932890 (N5′) in the E-039–E-043 block | 5.407 / 5.388 / 5.394 / 5.409 | 5.400 | 0.021 s (0.39%) |
| + EP all-to-all / compute overlap (10) | E-044 (6933731, N6; 6933837, N7) | 5.197 / 5.228 | 5.213 | 0.031 s (0.6%) |
| final posture (15) | E-061 cycles 6935335 (group6), 6935336 (group5), 6935443 (group5), 6935444 (group7), runs 2 and 3 of each; final500_a 6935341 (group11), window 26–50 | 4.941 / 4.956, 4.974 / 4.952, 4.955 / 4.920, 5.015 / 4.979, 4.954 | 4.961 | 0.095 s (1.9%; SD 0.027 s, 0.54%) |
| as-is baseline (0) | E-001 6930454 (group11), parity A1 6934455 (group13), A2 6934723 (group7), A3 6935242 (group8), run 1 of 6935298 (group3) and of 6935299 (group4) | 9.328 / 9.269 / 9.233 / 9.275 / 9.219 / 9.279 | 9.267 | 0.109 s (1.2%; SD 0.038 s, 0.41%) |

The spread reached 3.7–3.8% on steps 5 and 6, stayed under 0.4% on steps 8 and 9, and was 0.6% on
step 10. On step 6
the two runs that differ most (E-019 on N3′, E-022's first on N3) shared 15 of their 16 nodes,
while E-022's two runs on disjoint nodelists agree to 0.05%. Placement explains both large spreads
(E-047): N3′ is N3's switch group plus one node of another, and on step 5 N1 spans four groups where N2
sits in one; every other nodelist above lies inside a single group. The ~4% is therefore the
multi-group penalty, not run-to-run noise between comparable runs.

The final posture's nine single-group runs span four switch groups and 1.9%, with a run-to-run SD of 0.54%;
the as-is baseline's six standalone runs span six groups and 1.2%. Inside the paired cycles the as-is posture
drifts: the fourth run of each cycle is 1.4–1.6% slower than the first on the same nodes (9.442–9.542 against
9.311–9.404 s; the late runs carry isolated 10–11 s iterations), which the A B B A order cancels. Placement
across switch groups is a separate, larger effect (E-047): every run above ran on a single group.

## Final posture

The configuration the campaign ships is the Nano pretrain quickstart,
`configs/quickstart/nemotron_nano_quickstart_pretrain.yaml`, with the launcher settings in
`nemotron_nano_quickstart_pretrain.env` beside it: the baseline benchmark plus the 18 fields and two
`ISAMBARD_ENV_OVERRIDES` lines below, on a Megatron-LM that carries 0003, 0004 and 0005 (carried commits of
the pin since 2026-09-30, `3rdparty/patches/megatron-lm/README.md`; every measured tree also carried 0002,
which does nothing without CUDA graphs). H32 (sync-free grouped-GEMM offsets) is Bridge code with no knob. Every override is opt-in; the
production configs are unchanged.

```
comm_overlap.overlap_param_gather=true                  # H01 (step 1)
model.recompute_modules=[moe_act]                       # H16 (step 3)
logger.timing_log_level=1                               # host-sync bundle (step 4)
rerun_state_machine.check_for_nan_in_loss=false         # host-sync bundle
ddp.check_for_nan_in_grad=false                         # host-sync bundle
logger.log_l2_norm_grad_to_tensorboard=false            # host-sync bundle
train.manual_gc=true                                    # manual GC (step 5)
train.manual_gc_interval=10                             # manual GC
model.moe_token_dispatcher_type=flex                    # H22 HybridEP (step 9)
model.moe_flex_dispatcher_backend=hybridep              # H22
model.mtp_num_layers=null                               # needed by the EP overlap (Nano has no MTP layers)
mixed_precision=nemotron_h_bf16_with_fp8_current_scaling_bf16_params_bf16_grad_reduce   # H15 + H27 + BF16 primary weights (steps 2, 8, 14)
comm_overlap.overlap_moe_expert_parallel_comm=true      # EP overlap, patch 0003 (step 10)
model.moe_router_fusion=true                            # step 11
train.manual_gc_freeze=true                             # step 12
model.cross_entropy_loss_fusion=true                    # chunked CE, patch 0005 (step 13)
model.cross_entropy_fusion_impl=linear                  # chunked CE
model.cross_entropy_fusion_saved_logit_chunks=8         # chunked CE
```

```
ISAMBARD_FP32_SSM_STATE=0          # H13 (step 6)
ISAMBARD_CUDA_MAX_CONNECTIONS=32   # the EP overlap's two streams (step 10)
```

**Reproducing it.** Launch the quickstart with its env file as its header shows; the pinned Megatron-LM
carries the three changes. Pass the env lines through the overrides file, not the shell: the
launcher echoes them per rank (`[env-overrides]`), inherited ones leave no trace (Open risks). The runs below
gave the same 18 values as Hydra overrides on the baseline benchmark, through the campaign's parity harness (arm
`final`, `/projects/a5k/public/logs/nano_pretrain_perf_campaign/parity/`), on `snapshots/stack-final1` (pin +
0002 + the first version of 0003 + 0005), which predates patch 0004 and 0003's second revision; the shipped
patches train identically (Functional parity, Row 7).

**Design.** Each cycle is one 16-node allocation that runs the as-is baseline and the final posture in the
order A B B A (as-is, final, final, as-is; `paired/paired_cycle.sbatch`, cycle file
`paired/cycles/final_vs_asis.txt`), so drift within an allocation cancels and every ratio is taken on one
nodelist. The decision rule was registered before any cycle ran (`robustness/noise_model.md` §6):
- estimand: the final posture's mean step over iterations 26–50 on a single-group 16-node allocation;
- k = 4 cycles, extended to 6 if the 95% CI contains 5.000 s or the ratio CI is wider than ±1%;
- validity: a cycle placed on more than one switch group is void; a run that does not reach iteration 50 voids its
  cycle; a cycle whose two runs of one posture differ by more than 2% in their 26–50 median is flagged and
  replaced;
- statistic: F̄ = mean of all final runs; 95% CI = F̄ ± t(0.975, k−1)·s/√k over the per-cycle final means; the
  speed-up is the geometric mean over cycles of the as-is/final ratio, with a t-CI on its log, and never E-001;
- verdicts: established (upper bound ≤ 5.000), met on average (F̄ ≤ 5.000 < upper bound), not met within noise,
  not met.
The rule was written for A B C C B A with E-044 as B; on Kyle's instruction to run only the champion and the
baseline, the cycles dropped B.

| Cycle (job) | Placement | As-is run 1 / run 4 (s) | Final run 2 / run 3 (s) | Final medians (s) | As-is / final |
|---|---|---|---|---|---|
| 6935335 | group6:16 | 9.404 / 9.542 | 4.941 / 4.956 | 4.933 / 4.942 | 1.915 |
| 6935336 | group5:16 | 9.334 / 9.473 | 4.974 / 4.952 | 4.959 / 4.952 | 1.895 |
| 6935443 | group5:16 | 9.311 / 9.442 | 4.955 / 4.920 | 4.948 / 4.915 | 1.899 |
| 6935444 | group7:16 | 9.355 / 9.502 | 5.015 / 4.979 | 4.984 / 4.958 | 1.887 |

All four cycles are single-group and none is flagged. **F̄ = 4.9614 s, 95% CI [4.9204, 5.0024]** (per-cycle SD
0.0258 s, t = 3.182); median of the run medians 4.950 s; as-is over the same cycles 9.4203 s; **speed-up 1.8987×,
95% CI [1.8804, 1.9172]** (half-width 0.97%). At F̄: 13,209 tok/s/GPU, 274.9 model TFLOP/s/GPU, 27.78% MFU. Against E-001 (9.328 s) the same mean is 1.88×; the rule takes the speed-up from the cycles'
own as-is runs, which ran 1.0% slower than E-001 on average (mostly the drift within each allocation described
under Repeatability: their first runs average 9.351 s, their last 9.490 s). A ninth
single-group run, final500_a (6935341, group11:16), gives 4.954 s over the same window.
Verdict at k = 4: **goal met on average, not established**; the extension fired (the CI contains 5.000 s) and
cycles 5 and 6 (6958389, 6958390),
queued on 2026-09-30 and not yet run, are the rule's final look.

**Later in a run.** Both postures speed up after the from-scratch routing transient. Over iterations 451–500
of the 500-iteration runs the final posture (final500_a) averages **4.790 s** (median 4.785), 13,681 tok/s/GPU,
284.7 TFLOP/s/GPU, **28.77% MFU**, against 8.935 / 8.857 / 8.913 s for the as-is runs A1–A3
(mean 8.902): **1.86×**. From iterations 26–50 to 451–500 the final posture gets 3.3% faster and the as-is runs
3.6–4.1% faster, so the late ratio is slightly below the early one.

**Memory.** W&B run maxima (last rank) of the eight cycle runs: 78.70–78.77 GB allocated, 89.88–91.21 GB
reserved, 0 allocator retries (final500_a: 78.72 / 90.93); as-is 81.84 / 83.38–84.35. At 32 GPUs
(`train.global_batch_size=256`, 8 nodes; mem32_final 6935623, group2:8) the final posture runs 4.865 s (median
4.856), 13,471 tok/s/GPU, 28.33% MFU, and peaks at 81.93 GB allocated / 93.60 GB reserved with 0 retries and 0
NaN: it fits, with ~1.4 GB of reserved headroom.

## Functional parity

Kyle's rule (2026-09-29): a lever is admitted only if runs made with it are the same training as the
production posture: same model and checkpoints, same data and schedule, same loss trajectory, bit-identical
where the lever claims exactness. `scripts/telemetry/loss_parity.py` implements the two trajectory tests at
W&B full precision: `identity` (every iteration equal) and `band` (every 50-iteration window of the candidate's
mean loss inside [lowest reference − δ, highest reference + δ], δ = the largest window difference between two
reference runs). Both also require identical learning rate and consumed samples at every iteration.

| Row | Test | Result |
|---|---|---|
| 1 | Knob-off exactness: with every new knob off, the patched trees train exactly like the pristine pin | PASS (1 node and 64 GPUs) |
| 2 | Final posture's loss over 500 iterations inside the as-is band | PASS (E-044 alone: FAIL) |
| 3 | Checkpoint round trip in the final posture | PASS |
| 4 | HF export: layout identical to the as-is export, weights at master precision | PASS after two fixes (patch 0003 revision 2, `_bf16_params`) |
| 5 | Same data and schedule | PASS |
| 6 | Cost of BF16 primary weights | +0.23% (accepted) |
| 7 | The shipped patches train like the measured tree | PASS |

**Row 1.** Deterministic runs: `model.deterministic_mode=true` with `NVTE_ALLOW_NONDETERMINISTIC_ALGO=0` and
`CUBLAS_WORKSPACE_CONFIG=:4096:8` in an `ISAMBARD_ENV_OVERRIDES` file, plus `MAMBA_DETERMINISTIC=1` where stated.
- 1 node (11 layers `MEM*EMEMEME`, GBS 16, 30 iterations), `parity-prod` (83b79356 + pristine pin) against
  `parity-champ` (the same with the EP-overlap tree), knobs off: the as-is posture 6934446 vs 6934449 and
  6934448 vs 6934450; the levers without the overlap 6934518 vs 6934523 and 6934520 vs 6934525. All four
  pairs are identical at every iteration in lm loss, grad norm, learning rate and consumed samples. They ran
  without `MAMBA_DETERMINISTIC`, each pair on two different nodes, so at this depth the Mamba kernels happened
  to autotune alike; the 64-GPU pair shows that cannot be relied on.
- 64 GPUs, 50 iterations, as-is posture, with `MAMBA_DETERMINISTIC=1`: det50m_prod 6935147 (group12:16) vs
  det50m_champ 6935247 (group6:16) are identical at all 50 iterations. The same pair without it (6934933,
  group3:16, vs 6935076, three groups) differs from iteration 1 (lm loss 12.197461128 vs 12.197492599):
  mamba_ssm fixes its Triton autotune configurations at import, before Bridge turns deterministic algorithms
  on, so without the variable the SSD kernels are tuned by timing and the choice varies by node. Default mode
  is not reproducible even within one tree: 6934452 vs 6934453 differ in grad norm from iteration 1 and in
  loss from iteration 3.
- The patches themselves, 1 node, 30 iterations: 0003 revision 2 against the tree E-044 ran (6935991 vs
  6935230, E-044 posture, without `MAMBA_DETERMINISTIC`) and 0004 (6935510 on `stack-final1-countfix` vs 6935478
  on `stack-final1`, the final posture minus chunked CE, which Megatron-Bridge refuses in deterministic mode,
  with `MAMBA_DETERMINISTIC=1`) are identical at every iteration.

**Row 2.** 500 iterations each; references A1 6934455 (group13:16), A2 6934723 (group7:16), A3 6935242
(group8:16), the as-is posture on `parity-prod`; candidate final500_a 6935341 (group11:16, `stack-final1`,
arm `final`). lm loss δ = 0.045379.

| Window | A1 | A2 | A3 | Band | Final posture | Deviation | E-044 (6934457) | Deviation |
|---|---|---|---|---|---|---|---|---|
| 1–50 | 8.40433 | 8.38588 | 8.37970 | 8.33432–8.44971 | 8.38971 | −0.00026 | 8.39836 | +0.00839 |
| 51–100 | 6.19764 | 6.19808 | 6.16057 | 6.11519–6.24346 | 6.20338 | +0.01795 | 6.17491 | −0.01052 |
| 101–150 | 5.37586 | 5.34117 | 5.33048 | 5.28510–5.42124 | 5.33963 | −0.00954 | 5.34918 | +0.00001 |
| 151–200 | 4.71182 | 4.69428 | 4.68883 | 4.64345–4.75720 | 4.69830 | −0.00000 | 4.71964 | +0.02133 |
| 201–250 | 4.26844 | 4.27714 | 4.25236 | 4.20698–4.32252 | 4.25957 | −0.00642 | 4.27248 | +0.00650 |
| 251–300 | 3.94735 | 3.94638 | 3.94276 | 3.89738–3.99273 | 3.95976 | +0.01427 | 3.95768 | +0.01218 |
| 301–350 | 3.73100 | 3.69819 | 3.68697 | 3.64159–3.77638 | 3.71991 | +0.01452 | 3.82381 | **+0.11842** |
| 351–400 | 3.50805 | 3.52480 | 3.49641 | 3.45103–3.57018 | 3.50792 | −0.00183 | 3.55018 | +0.04043 |
| 401–450 | 3.35363 | 3.36299 | 3.34257 | 3.29719–3.40837 | 3.35447 | +0.00141 | 3.38860 | +0.03553 |
| 451–500 | 3.23424 | 3.24072 | 3.22315 | 3.17777–3.28610 | 3.23624 | +0.00354 | 3.26390 | +0.03120 |

The final posture is inside the band in all ten windows (largest deviation 0.018, final window +0.0035) and
its grad norm inside too (largest deviation 0.097, in 1–50): PASS. E-044's run fails one window: an excursion
from about iteration 319 (4.018 at 322, 3.999 at 329) that is gone by ~360; the final posture shows nothing
like it there (+0.015). Learning rate and consumed samples are identical to the references at every
iteration.

**Row 3.** ckpt_save_final 6935539 (16 nodes, `stack-final1`, arm `final`) saved at iterations 25, 50, 75, 100
and 125, training on after each save (no OOM, no traceback). ckpt_resume_final 6935631 loaded iteration 100: its
iteration-101 lm loss equals the straight run's at W&B full precision (5.740697); its grad norm differs in the 8th
significant digit (1.1760488749 vs 1.1760487556), the default-mode run-to-run level (Row 1), and the two runs then
drift apart as independent runs do (lm loss 5.3798 vs 5.3541 at iteration 125). One limit: optimizer state does
not move between postures. Resuming a final-posture checkpoint in the as-is posture, or the reverse, fails at
load (6934731, 6934732: `AssertionError: Number of unpadded elements in each bucket need to be the same`), because
the default `dp_reshardable` optimizer format keys its buckets by (parameter dtype, gradient dtype): (bf16, fp32)
buckets of 503,795,840 + 401,391,040 elements as-is, against (bf16, bf16) 823,127,616 + (uint8, bf16) 82,059,264
with BF16 gradient reduction and FP8 parameters. Switching an in-flight production run to the final posture needs
a conversion of its optimizer state.

**Row 4.** Two defects surfaced and were fixed.
- The first version of patch 0003 dropped `output_layer._extra_state` from every HybridModel's sharded state
  dict, so the production exporter (a tree without the patch) failed on its checkpoints (6935149: `RuntimeError:
  Missing key in checkpoint state_dict: output_layer._extra_state/shard_0_1`). Revision 2 keeps the pin's keys for
  flat layer patterns (patches README, 0003).
- FP8 primary weights. The champion's preset (`nemotron_h_bf16_with_fp8_current_scaling_mixed`) sets
  `fp8_param_gather` and so `fp8_param`: the dense TE weights are FP8 tensors, and the checkpoint stores them
  dequantized. On a 7-layer smoke checkpoint (`parity/fp8_param_error.py`) 10 of 75 tensors (82,059,264 elements,
  the dense TE linears of the FP8 layers) sit 2.65–2.67% from their fp32 masters, against 0.15–0.17% for BF16
  rounding of the same masters, with 242–252 distinct values each; at full size (export 6935249) 108 of 141
  dense 2-D weights, 1,372,299,264 elements. Every weights-only consumer (the next stage's warm start, HF
  export, evals) would read FP8-rounded weights. `_bf16_params` (E-060) turns `fp8_param_gather` off and keeps
  FP8 compute: its smoke checkpoint has no FP8-rounded tensor, and its full-size export X_C48bp (6935250) is
  layout-identical to the as-is export X_A (6935092): 6,243 tensors, 31.578 B elements, 23 F32 + 6,220 BF16,
  5,888 routed-expert tensors (256 per MoE layer), the same names, shapes and dtypes, and 0 of 141 dense 2-D
  BF16 weights FP8-rounded.

**Row 5.** A1, A2, A3 and final500_a load the same 41 dataset index files (`parity/data_identity.sh`:
fingerprint 6b61fb8286ac, 13 × 3 GPTDataset + 2 BlendedDataset, split sizes (15299072, 0, 0), cyclic sampler,
sharded), and log the same learning rate and consumed samples at every iteration (Row 2).

**Row 6.** On the E-048 base, `_bf16_params` costs +0.23% by mean and +0.11% by median (6935209, group6:16,
5.144 s against 6935208, group7:16, 5.132 s; unresolved across groups) and +2.7 GB allocated / +2.9 GB reserved
(W&B 78.45 / 91.25 → 81.14 / 94.12 GB; rank 0 after iteration 1 48.32 → 51.07 GB allocated, 84.79 → 87.70
reserved), consistent with BF16 primary weights plus the 1-byte FP8 copy of ~1.37 B dense-linear parameters; 0
retries. In the final posture chunked cross-entropy gives the memory back (final 78.72 / 90.93 GB against
E-048's 78.44 / 91.46).

**Row 7.** The shipped tree, `snapshots/stack-final2`: this repository's code with pin + 0002 + 0003 revision 2
+ 0004 + 0005 applied (checksums in its `REVISION`; record in `shipped/VERIFICATION.md`). Its Megatron-LM differs
from the tree the goal cycles ran on only in 0003's second revision, in 0004, and in a refactor inside 0005 on a
path the final posture never runs.
- A deterministic 1-node run of the final posture without chunked CE (11 layers `MEM*EMEMEME`, GBS 16, 30
  iterations, `MAMBA_DETERMINISTIC=1`) is identical at every iteration, in lm loss, grad norm, learning rate and
  consumed samples, to the measured tree with 0004 (6958571 against 6935510).
- The full final posture in default mode, chunked CE included, stays inside one tree's 1-node repeat band
  (6958572 against 6958573: identical at iterations 1–2, largest difference 4.1e-3 at iteration 29, 5.5e-4
  relative, against a repeat band of 1.0e-3; learning rate and consumed samples identical).
- The unit suite on the patched tree (6958565) runs every patch-gated test and passes them; on the same files
  over the pristine pin (6958566) those 60 tests skip, each naming its patch, and nothing else differs. The only
  failures on either frozen copy are its read-only location (the corpus-build tests, which pass on a writable
  copy) and `test_mcore_commit`, which needs a git checkout.
- Megatron-LM's own tests of the files the patches add or change pass on one 4-GPU node, except the
  parametrizations that need 8 GPUs, exactly as on the reference trees (6958567, 6958568, 6958669 / 6958670).

## Open risks

- **HybridEP's dispatched-token count (fixed by patch 0004).** On HybridEP's blocking path the dispatch
  handle's token count lives in pinned host memory that the permute and unpermute kernels read from device
  code; once the handle is freed, PyTorch's caching host allocator can hand the block out while such a kernel
  is still queued. Under the EP overlap this faulted twice at reduced depth (E-059), once at micro-batch 1. At
  micro-batch 1 no fault was seen in 136 EP-overlap runs (8,149,215 overlapped layer dispatches), but a wrong
  count that stays inside mapped memory would corrupt the combine silently. Every tree before 0004, including
  the one the goal cycles ran on, carries the exposure; 0004 is bit-identical in training.
- **Inherited environment levers leave no trace.** The two env lines are effective only if set, and a launch that
  inherits them from the shell records nothing: the platform runs of E-047 / E-056 carry no `[env-overrides]`
  line, so their logs cannot show which values they had. Pass them through an `ISAMBARD_ENV_OVERRIDES` file and
  check the per-rank `[env-overrides]` line.
- **deep_ep's custom NVLink all-gather continues silently after its ~20 s timeout** in the pinned deep_ep
  (1.2.1+34152ae); upstream 94a9f8f makes it trap. A stall there would surface as wrong data, not an error.
  Not observed in training.
- **Deterministic runs need `MAMBA_DETERMINISTIC=1`** before import (Row 1), and Megatron-Bridge refuses
  cross-entropy fusion in deterministic mode, so bit-identity tests of the final posture run without chunked CE.
- **Reserved-memory headroom.** The EP overlap reserves ~13 GB more than it allocates (E-050). The final posture
  peaks at 90.9 GB reserved at 64 GPUs and 93.6 GB at 32 GPUs (W&B run maxima, last rank, 0 retries) of ~95 GB.
  Anything that adds memory (CUDA graphs, micro-batch 2) meets allocator retries first (E-046: +9.8% with 40
  retries).
- **Optimizer state does not cross postures** in the default `dp_reshardable` format (Row 3).
- **Not yet measured at 256 GPUs**, the production width: speed and loss parity were taken at 64 GPUs, fit
  also at 32. The as-is reference of a 500-iteration 256-GPU parity pair (6958401) was queued on 2026-09-30.

## Stale verdicts retested

| Earlier verdict | Premise that no longer holds | Retest | Outcome |
|---|---|---|---|
| `overlap_param_gather: false` for Nemotron-H at DP>1 | derived on Super at TP4 / bare metal | H01 | WIN (E-004) |
| bf16 gradients "tried, failed" | all four attempts were VOID: `bf16_mixed` hard-codes `grad_reduce_in_fp32=True` | H15 | WIN — engaged via the bf16 grad-reduce recipe, `bf16_mixed_bf16_grad_reduce` (E-008) |
| dropping `moe` recompute OOMs | measured without bf16 gradients (19 GB/GPU more free memory with them) | H16 | WIN (E-009) |
| CUDA graphs blocked on Nano | the blocker was memory at DP=512, not capability | H18 | retested twice: engages at 64 GPUs but NULL on throughput (~0.8%) at +4 / +12 GB reserved (E-035); on the EP-overlap champion memory-bound — [mamba] NULL (−0.4%), [mamba,attn,moe_router] +9.8% at 94.6–95.2 GB reserved with up to 40 allocator retries (E-046) |
| DeepEP/HybridEP incompatible | the blocker is the inter-node IBGDA path; EP=4 is intra-node | H21/H22 | H22 WIN once GroupedExperts is sync-free (E-028; NULL before, E-005); H21 not yet retested |

## Experiments

### E-000 · campaign allocation chain burned on a missing runner · 2026-09-28 · INCIDENT
`sbatch --test-only` estimated a 17-node start ~32 h out, so the three-segment singleton chain
(6930403/6930404/6930407) was submitted before the runner it execs existed. Backfill started the first
segment immediately; each segment failed in 6 s (`can't open file .../scripts/training/campaign_runner.py`),
~0.1 node-hours in total. The wrapper itself was validated (node-limit guard passed, interpreter resolved).
Learning: build and test everything a batch script execs before submitting it; test-only estimates are
not a lower bound on start time.

### E-001 · calibration: as-is baseline at 64 GPUs / GBS 512 · 2026-09-28
A probe like every other (Method): one `pipeline_training_submit.sbatch` job on the baseline YAML with the
quickstart's overlay given as Hydra overrides (`train.exit_interval=50 train.global_batch_size=512
checkpoint.load=null checkpoint.save=null dist.distributed_timeout_minutes=20`, own index cache), from the
main checkout at e3dd4e56. Job 6930454 (unprofiled) and 6930456 (torch profile, iterations 30 and 40,
ranks 0 and 5, with stack). Purpose: the as-is baseline every later probe is scored against (ladder
step 0), plus the step-time, memory and trace data that re-ranked the backlog.
Prediction: mean step 9.7–10.0 s ≈ 6,550–6,750 tok/s/GPU ≈ 14% MFU; peak allocated ≈ 81.8 GB.
Result (job 6930454, nid[011001-011002,011009,011012-011013,011016,011019-011020,011023,011025,011030,
011041,011068-011069,011074,011076], W&B `nano_pretrain_perf_calib_b0`), iterations 26–50:
**mean 9.328 s** (median 9.287, p10 9.231, p90 9.462, min 9.213, max 9.580) → **7,026 tok/s/GPU, 146.2
model TFLOP/s/GPU, 14.78% MFU**; lm loss (41–50) 6.869; 0 NaN / 0 skipped; after-iteration-1 memory
79.86 GB max allocated / 81.98 GB reserved. Iteration 1 took 99.5 s; iterations 2–4 ran 9.3–9.5 s, then
the step drifted up to 11.7 s by iteration 18–19 and settled back to 9.2–9.5 s from iteration 26 — the
from-scratch routing transient the scoring window starts after. Within the window the first and second
halves differ by < 1% (26–37 vs 38–50), so the window is past the transient.
W&B timers (run `vtfgwr4d`, means over 26–50): iteration 9.328 s = forward-backward 7.908 (forward-compute
1.979 + backward-compute 5.979, summed over the 8 microbatches) + optimizer 1.366 (of which
**params-all-gather 1.356 s — exposed, `overlap_param_gather: false`**, inner Adam step 0.005) + ~0.05 other.
Backward is 3.0× forward (≈ 2× without recompute): the ~2 s excess is the `[core_attn, moe, shared_experts]`
recompute plus the fp32 reduce-scatter tail absorbed by the last backward's timer sync. Peak (W&B summary)
**81.84 GB allocated / 84.47 GB reserved** — exactly the 81.8 GB static-memory prediction; ~10 GB below the
~94.5 GB (88 GiB) effective ceiling. The run also logs a per-parameter gradient L2 norm for every parameter
every iteration (`log_l2_norm_grad_to_tensorboard: true`), a per-parameter kernel + host-sync pattern like
the `log_params_norm` telemetry that cost Super 10.2% (new candidate H38).
Learning: 5% faster than predicted; the 2× target at this width and batch is a mean step ≤ 4.664 s
(≥ 14,052 tok/s/GPU, ≥ 29.6% MFU). 1.36 s of the 9.33 s is the exposed all-gather alone.

### E-061 · goal verdict: final posture vs as-is in placement-controlled paired cycles · 2026-09-29/30 · GOAL MET ON AVERAGE (k = 4)
The design, the pre-registered rule and the per-cycle table are in Final posture. Cycles 6935335 (group6:16),
6935336 (group5:16), 6935443 (group5:16), 6935444 (group7:16), each A B B A on one 16-node allocation, final
posture on `snapshots/stack-final1` (arm `final`), as-is on `snapshots/parity-prod` (arm `asis`); all four
single-group, none flagged (the largest within-posture median difference is 0.7%). Final runs 4.920–5.015 s;
F̄ = 4.9614 s, 95% CI [4.9204, 5.0024]; speed-up against the cycles' own as-is runs 1.8987×, 95% CI [1.8804,
1.9172]; lm loss (41–50) 6.876–6.949 final, 6.870–6.934 as-is. W&B run maxima 78.70–78.77 GB allocated /
89.88–91.21 GB reserved, 0 retries.
Verdict at k = 4: goal met on average, not established; the extension fired and cycles 6958389 / 6958390,
queued on 2026-09-30 and not yet run, are the final look.
Learning: the paired design did what it was built for: the ratio carries a 0.97% CI half-width, and the as-is
drift inside each allocation (+1.4–1.6% from its first to its last run) would have biased any unpaired
comparison by as much.

### E-060 · BF16 primary weights under FP8 compute (`_bf16_params`) · 2026-09-29 · ADOPTED (parity requirement)
Found by the parity work (Row 4): the champion's mixed-precision preset makes the dense TE weights FP8 tensors
(`fp8_param_gather` ⇒ `fp8_param`), so checkpoints store FP8-rounded dense weights (2.65–2.67% from the fp32
masters) and the masters live only in the optimizer state. `get_mixed_precision_config` now takes a preset
followed by name modifiers; `_bf16_params` sets `fp8_param_gather=False` (and so `fp8_param=False`) with FP8
compute unchanged, refuses presets without FP8 parameters, and composes with `_bf16_grad_reduce`
(`nemotron_h_bf16_with_fp8_current_scaling_bf16_params_bf16_grad_reduce`; W&B config of 6935209:
`fp8_param_gather: False`). Cost pair on the E-048 base, one 50-iteration run each: control 6935208
(group7:16) 5.132 s (median 5.076, loss 6.912) against 6935209 (group6:16) 5.144 s (median 5.081, loss 6.903):
+0.23% / +0.11%, inside placement noise; +2.7 GB allocated / +2.9 GB reserved (W&B 81.14 / 94.12 GB), 0 retries.
Learning: an FP8-parameter preset changes what a checkpoint contains, not only how the step computes; any
posture that feeds later stages must keep primary weights at master precision.

### E-059 · HybridEP crash under the EP overlap: root cause and patch 0004 · 2026-09-29 · FIXED
Symptoms, both on 1-node smokes of the EP-overlap posture (11 layers `MEM*EMEMEME`, EP4, GBS 16):
- 6934840 (micro-batch 2): illegal memory access on rank 2 after iteration 19, inside HybridEP's
  `dispatch_with_permute`; its repeat 6934883 ran 30 iterations clean;
- 6935265 (micro-batch 1, deterministic modifier): `RuntimeError: Trying to create tensor with negative
  dimension -32: [-32, 2688]` in iteration 19's dispatch, then an illegal memory access on every rank.
A faithful repro (6935302, `chunked-ce-smoke1`) aborted in run 1 (rank 2, nid010395, SIGABRT after iteration 19)
and ran clean in run 2 — two faults in four runs of the posture. Its core dump put a Warp MMU Fault in deep_ep's
`unpermute_kernel<512, bf16, float>` reading an expert-output row, with the kernel's loop bound read as 262,145
where at most 65,536 tokens can arrive (dev/syncfree-ep/NOTES.md). Mechanism: on the blocking path (no
`num_permuted_tokens`) deep_ep keeps the dispatched-token count in pinned host memory that the permute and
unpermute kernels read from device code; nothing records that use with PyTorch's caching host allocator, so once
the backward frees the handle the block can be handed out again while the comm stream still has a kernel queued
that reads it. The overlap opens that window. The second symptom follows: the blocking path ignores its stream
sync's error code, so after a fault the next dispatch sums stale bytes of the per-expert count block.
Fix, test-first (patch 0004, a carried commit of the pin since 2026-09-30, `40e960a2f`; patches README):
`HybridEPDispatch.forward` replaces the handle's count with a non-blocking device copy on the dispatch stream; the
host allocator records the copy and holds the pinned source until it has run; no new host sync. Tests
(`tests/unit_tests/training/test_hybridep_count_lifetime.py`, one GPU): 3 of 5 fail without the fix (6935541) and
all 5 pass with it (6935529). The repro ran 6 of 6 clean with the fix (6935468). The final posture minus chunked
CE is bit-identical with and without it over 30 deterministic iterations (6935478 vs 6935510).
Learning: a queued kernel that reads host memory needs that memory's lifetime tied to the stream; the exposure is
at every micro-batch size, not only where it faulted.

### E-058 · R-3a: HybridEP SM share, combine chunking and TE norm settings · 2026-09-29 · NULL
`model.moe_flex_dispatcher_num_sms=16` plus env `NUM_OF_TOKENS_PER_CHUNK_COMBINE_API=128`,
`NVTE_FWD_LAYERNORM_SM_MARGIN=20`, `NVTE_BWD_LAYERNORM_SM_MARGIN=20`, `NVTE_NORM_BWD_USE_CUDNN=1` on the E-044
posture (engaged: HybridEP's configurer went from 24 to 16 blocks and from 64 to 128 combine tokens per chunk;
probe-batch notes). 6934828 (group6:16) 5.262 s (median 5.204) against the same group's controls 6934492 5.275 /
6934575 5.239: +0.10%. NULL.

### E-057 · sync-free HybridEP dispatch with a receive budget · 2026-09-29 · KILLED (memory)
Goal: remove HybridEP's 184 host syncs per step (E-036) without dropping tokens. A Megatron-LM patch
(`XXXX-feat-hybridep-receive-budget.patch`, not vendored) lets a caller arm a per-dispatch receive budget; a Bridge
runner arms it for each step and re-runs any step whose dispatches overflowed on the blocking path, so no token
is dropped (bitwise equal to plain steps in 99 GPU tests). The adaptive budget failed on memory at 64 GPUs
(6935066, group12:12,group8:4, cancelled at iteration 10): the largest per-layer receive grew from 1.36× the mean
at step 1 to 2.76× by step 8, the budget from 63,744 to 156,672 tokens, every step from 4 on overflowed and re-ran
(15.7–27.0 s per iteration), and the run peaked at 95.31 GB reserved with 120 allocator retries. Its control
6935065 (two groups) ran 5.399 s. Not imported.
Learning: routing skew grows in the first iterations of from-scratch training, so a receive budget sized to fit
drops to the blocking path when it matters, while a static worst-case budget costs memory the overlap has already
spent.

### E-056 · platform: NGC 26.06 / 26.08 images, aws-ofi-nccl 1.21.1, a channel cap for the expert-DP ring · 2026-09-29 · all NULL
- **26.06** (torch 2.12, TE 2.16) runs on this cluster's R565 driver only by fronting the 26.04 image's CUDA-13.1
  compatibility libraries (26.06's own fail with Error 803). E-044 posture, all on group13:16: 26.04 6934572
  5.201 s against 26.06 6934460 5.328 / 6934511 5.244 (+1.65% on average): NULL. W&B memory identical
  (78.4 GB allocated, 91.3–92.1 GB reserved).
- **26.08** (torch 2.13, TE 2.18): 6935060 (26.08) ran on five switch groups (5.725 s) against 6935061 (26.04,
  group11:16, 5.179 s); placement-confounded, not scored.
- **aws-ofi-nccl v1.21.1** (NCCL 2.29.2): the expert-DP ring pattern (four concurrent one-GPU-per-node rings, 4
  nodes, 1 GB bf16 buckets) moves 22.98 / 23.00 GB/s per rank (reduce-scatter / all-gather) against 22.97 / 22.99
  on v1.18.0, 92% of the 25 GB/s NIC; `OFI_NCCL_PROTOCOL=RDMA` is unusable on CXI (`fi_writedata ... Function not
  implemented`). NULL at the transport, so no training probe (dev/platform/NOTES.md, jobs 6934776 / 6934796).
- **`ep_dp: {max_ctas: 8}`** (`dist.nccl_communicator_config_path`; the ring runs 40 channels by default, 8 with the
  cap: 6934845 vs 6934809): the ring bench keeps 23 GB/s and interferes less with GEMMs, but at 64 GPUs back to back
  on group13:16 6934806 (capped) 5.217 s against 6934807 5.226 s: NULL.
- E-038's launch failure explained: the `brics/nccl` module exports `NCCL_NET_FORCE_FLUSH=1`, sbatch carries it
  into standalone NCCL jobs, and it wedges the OFI receive path; training is unaffected because the launcher purges
  modules first.
None of it is in the repository.

### E-055 · FP8 routed experts (DeepGEMM) · 2026-09-29 · NULL with the EP overlap, −2.5% without
`moe_experts_impl=deepgemm_fp8` with `moe_expert_row_alignment=128` (dev copy `fp8-experts`: 1×128 / 128×128
quantizers, DeepGEMM's grouped FP8 forward and dgrad, BF16 wgrad; not imported). The per-projection kernels win
(forward + dgrad 1.23–1.55× on one GPU, job 6934433; dev/fp8-experts/NOTES.md). At 64 GPUs:
- with the EP overlap: control 6934684 (group6:16) 5.256 s against 6934685 (group13:16) 5.304: +0.9% mean, +0.3%
  median, NULL;
- without it: control 6934765 (group12:16) 5.524 against 6934766 (group4:16) 5.387: −2.5% mean, −1.7% median.
The profile of the FP8 arm (6934860 against 6933850) shows the expert GEMMs shrinking from 0.804 to 0.434 s per
iteration and the quantizers adding 0.348 s under the overlap's communication contention.
Learning: FP8 experts and the EP overlap are substitutes: the overlap already hides the expert GEMMs FP8 would
shorten, and it is the better of the two (5.197–5.228 against 5.387 s).

### E-054 · diagnostic PB-3: all-gather half of each expert shard · 2026-09-29 · diagnostic only
A probe-only snapshot (`probe-batch-s3-pb3diag`) makes each expert bucket all-gather half of its shard, so half
the experts train on stale weights; never adoptable. 6934697 ran on two switch groups (group7:14,group6:2) and
still measured 5.096 s (median 5.026), 3.1% below the single-group controls (mean 5.259), against the ~4%
two-group penalty. Its loss leaves the control at iteration 3 (11.978 against 11.833). Learning: the expert
parameter all-gather sits on the critical path; halving its bytes is worth several percent, which only
FP8 parameter gather or fewer expert bytes could buy legitimately. Not re-run single-group (round concluded).

### E-053 · diagnostic PB-4: forced expert load balance under the EP overlap · 2026-09-29 · imbalance tax ~1%
`model.moe_router_force_load_balancing=true` (benchmark-only: it changes training) on the E-044 posture: 6934432
(group6:16) 5.203 s against the same group's controls 5.275 / 5.239 (−1.0%), 76.34 / 87.16 GB (balanced routing
needs smaller receive buffers). Before the overlap the imbalance tax was ~5% (E-016). Learning: the overlap hides
most of the load imbalance; the imbalance levers fell below the 0.10 s bar and were dropped.

### E-052 · gc.freeze after the setup collection, and the host-bundle decomposition · 2026-09-29 · KEPT (removes the GC steps)
`train.manual_gc_freeze` (default off; requires `train.manual_gc`): after manual GC's setup collection,
`gc.freeze()` moves every surviving object into the permanent generation, so the periodic collections skip the
model, optimizer and dataloader state.
- A bundle of host-path knobs (gc.freeze, autograd multithreading off, an eager `dispatch_preprocess`, others) on
  the E-044 posture: 6934634 / 6934696 (group4:16) 5.182 / 5.200 s against the same group's control 6934431 5.264.
  gc.freeze alone: 6934827 (group11:16) 5.195. The other knobs showed no measurable speed in a 1-node bisection and
  are not imported.
- What gc.freeze removes: the manual collections at iterations 31 and 41 take 5.75–5.91 s in the controls against
  a ~5.2 s steady state (6934431 5.752 / 5.910, 6934492 5.780 / 5.847, 6934575 5.715 / 5.847) and 5.17–5.36 s with
  the freeze; dropping them lowers a control's 26–50 mean by 0.043–0.049 s (0.9%).
- On the router-fusion base the increment is not separable from placement: with gc.freeze 6934887 (group11:16)
  5.075, 6935055 (group5:16) 5.250, 6934886 (group8:16) 5.542 (its whole distribution shifted, p10 5.317: a slow
  allocation); without 6934491 / 6934574 (group4:16) 5.129 / 5.140 and 6935057 (group12:16) 5.149.
It is host-only and cannot change numerics; 6934827's iteration-1 loss differs from the controls' (12.199050903
against 12.199021339) by the node-dependent Mamba autotuning of Row 1, not the knob. Kept for the mechanism.

### E-051 · chunked linear cross-entropy (patch 0005) · 2026-09-29 · WIN (−3.1%)
`model.cross_entropy_loss_fusion=true model.cross_entropy_fusion_impl=linear
model.cross_entropy_fusion_saved_logit_chunks=8`: HybridModel's output layer and loss fused over 16,384-token
vocabulary chunks (our Triton + cuBLAS op; design, refusals and numerics in the patches README, 0005). One GH200,
output layer + loss forward and backward per micro-batch: 50.9 ms and 6.0 GiB peak unfused against 34.9 ms and
2.1 GiB keeping every chunk (6934762). 64 GPUs on the E-048 base (snapshot `chunked-ce-probe1`):

| Pair | Job | Arm | Placement | Mean (s) | Median | Max | Loss 41–50 | W&B alloc / reserved GB |
|---|---|---|---|---|---|---|---|---|
| a | 6935046 | control | group11:16 | 5.164 | 5.054 | 5.924 | 6.921 | 78.41 / 91.77 |
| a | 6935047 | 8 chunks kept | group8:16 | **5.002** | 4.939 | 5.521 | 6.838 | 75.99 / 87.58 |
| b | 6935082 | control | five groups | 5.401 | 5.329 | 6.019 | 6.900 | 78.44 / 91.40 |
| b | 6935083 | 8 chunks kept | six groups | 5.334 | 5.176 | 7.580 | 6.929 | 75.96 / 86.42 |

Pair a: −3.1% by mean, −2.3% by median, 13,102 tok/s/GPU; pair b ran multi-group (E-047) and agrees in direction
(−1.2% / −2.9%). Memory −2.4 GB allocated, −4.2 to −5.0 GB reserved: the loss buffers were the top of the heap
under the overlap (E-050). The two runs' losses differ as independent runs do (the fused one is the lower in
pair a, the higher in pair b). Learning: the fastest lever of the last round was a memory lever too.

### E-050 · where the EP overlap's reserved memory goes · 2026-09-29 · analysis
Per-rank allocator reports (`scripts/profiling/allocator_pools.py` of the `overlap-memory` dev copy, e.g.
`snapshots/overlap-memory-diag`; not in the repository), medians of 64 ranks at iteration 50, overlap
on (6934497) against off (6934498), both on two groups (placement does not move memory): reserved 91.15 against
78.05 GB, compute-stream pool 41.28 against 41.23, the other pools 49.83 against 36.83 (the static 36.83 GB plus a
13.0 GB communication-stream pool, 11.9 GB of it cached free at the end of the step), peak allocated 77.84 against
76.75. A 64-GPU allocation history (rank 0, 6934763; dev/overlap-memory/NOTES.md) puts the compute pool's
occupancy peak at 32.8 GB against 41.1 GB reserved (8.2 GB of holes, the largest 1–4 GB) and the communication
pool 3.7 GB above its own peak: the interleaved schedule's frees leave holes, and the fp32 logits (up to 4.3 GB)
need a contiguous one. Allocator settings do not recover it (64 GPUs, medians of ranks):
`roundup_power2_divisions` 16 (6934959, group12:16) 91.63 GB, 5.241 s; 4 (6934960, group5:16) 93.02 GB, 5.237 s;
classic segments with `max_split_size_mb:512` (6934851, group13:16) 94.63 GB reserved, allocator retries 19–49 per
rank (median 34), 5.752 s; `garbage_collection_threshold:0.9` inert (6934347, 91.61 GB W&B, 5.254 s, against
6934346's 91.53 / 5.213). Learning: the reserve is fragmentation, and removing the loss buffers (E-051) is the
only tested lever that lowers it.

### E-049 · copy the dispatched tokens to the compute stream · 2026-09-29 · LOSS (+1.4%), not imported
`model.ep_overlap_copy_dispatched_tokens` (dev copy `overlap-memory`, default off) clones HybridEP's dispatch
output into the compute stream's pool so the communication pool can free it. Bit-exact (deterministic 1-node
A/A/B, 6934692 and 6934693 identical to 6934691 over 30 iterations). 64 GPUs, medians of 64 ranks at iteration
50: communication pool 12.99 → 5.44 GB, but the compute pool 41.17 → 49.72, reserved 91.06 → 91.99, peak
allocated 77.82 → 77.89; step 5.191 s (median 5.135; 6934713, group12:16) → 5.262 (5.195; 6934714, group4:16):
+1.4% / +1.2%. Learning: the two pools peak at different moments; moving one microbatch of tokens into the
compute pool raises its peak by their full size, so the total does not fall.

### E-048 · router fusion under the EP overlap · 2026-09-29 · WIN (−2.4%)
`model.moe_router_fusion=true` on the E-044 posture (snapshot `probe-batch-s1`; W&B config
`moe_router_fusion: True`): 6934491 5.129 s (median 5.084) and 6934574 5.140 (5.059), both group4:16, against
controls 6934431 (group4:16, the same nodes as 6934491) 5.264, 6934492 (group6:16) 5.275 and 6934575 (group6:16)
5.239 (mean 5.259): −2.4% against the controls' mean, −2.6% against the same-node control; W&B 78.44 / 91.46 GB
against 78.37 / 91.13, 0 retries; loss 6.877 / 6.872 against 6.877–6.894. It changes numerics slightly: iteration-1
lm loss 12.19897 against 12.19902 on the same nodes. The earlier NULL (E-020, job 6932454) ran on two switch groups
(E-047). New best at the time: 1.81–1.82×.

### E-047 · switch-group placement costs ~4% · 2026-09-29 · measurement-method finding
The platform swap pair (E-056) ran one posture (E-044, identical in every W&B config key checked) on two node
sets, one inside group13 and one spread over two groups, with both images:

| Job | Image | Placement | Mean (s) | Median (s) |
|---|---|---|---|---|
| 6934572 | 26.04 | group13:16 | 5.201 | 5.155 |
| 6934461 | 26.04 | group13:14,group6:2 | 5.394 (+3.7%) | 5.338 (+3.6%) |
| 6934512 | 26.04 | group13:14,group2:2 | 5.493 (+5.6%) | 5.373 (+4.2%) |
| 6934460 / 6934511 | 26.06 | group13:16 | 5.328 / 5.244 | 5.203 / 5.155 |
| 6934573 | 26.06 | group13:14,group2:2 | 5.420 (+2.5% on the 26.06 mean) | 5.353 |

Two nodes outside the group are enough to pay the penalty. Across all the campaign's same-posture comparisons
(`robustness/noise_model.md` §4, n = 8) a multi-group placement costs +2.5% to +6.6%, mean +4.3%, about eight
times the single-group run SD (0.5%). Every launcher run logs its placement (`[run-identity] switch placement`);
from here on only single-group runs are scored against each other, and E-020's router-fusion NULL (two groups) is
void. Learning: placement, not node count or image, was the largest uncontrolled variable of the campaign.

### E-046 · TE CUDA graphs on top of the EP overlap · 2026-09-29 · NULL / LOSS (memory-bound)
The hybrid overlap plan asserted `cuda_graph_impl == "none"`. The adaptation (snapshot
`wt-e3dd4e56-epov-cg`, `EPOV_CG.diff`, three Megatron-LM files): one shared definition of the overlap
units for the schedule and the graph helper; the assert relaxed to allow TE graphs on flat patterns
(local graphs, bracketed patterns, delayed wgrad and MTP still refused); attention and MoE-router graphs
replayed from the hybrid callables through `_te_cuda_graph_replay`, as the GPT path does; and a capture
order that follows the unit pairing — TE keeps every graph's memory in one pool whose lifetimes follow
the capture order, and upstream's order (graphed layer g paired with G−1−g) matches the grouped schedule
only when every unit holds exactly one graphed layer (true for `[mamba]`, not for
`[mamba,attn,moe_router]`). Settings: `model.cuda_graph_impl=transformer_engine
model.cuda_graph_scope=null model.use_te_rng_tracker=true` plus the graph set, env
`NCCL_GRAPH_REGISTER=0` with the default `expandable_segments:True`.

Correctness (1 node, 11 and 21 layers): iterations 1–2 bit-identical in every arm; graph-vs-no-graph loss
differences inside no-graph-vs-no-graph. Speed there: −1.5% / −2.7% (11 layers), −2.3% (21 layers). A
profile of the 21-layer smoke shows why: the overlap raises GPU idle from 13.8% to 22.9% of the profiled
step, mostly under Mamba's fused forward/backward (its Triton launches), and `[mamba]` graphs bring it
to 5.7%.

At full depth the graphs' private pool sits on top of the overlap's extra reserved memory:

| Width | Graph set | Job | Mean (s) | × baseline | W&B run max alloc / reserved GB, retries |
|---|---|---|---|---|---|
| 32 GPUs | none (overlap only) | 6933943 | 5.154 | — | 81.9 / 95.2, 0 |
| 32 GPUs | `[mamba,attn,moe_router]` | 6933944 | 7.630 (+48%) | — | 80.0 / 95.1, 186 |
| 64 GPUs | `[mamba]` | 6934039 | 5.153 | 1.810 | 76.06 / 94.61, 0 |
| 64 GPUs | `[mamba]` (repeat) | 6934180 | 5.235 | 1.782 | 76.06 / 95.24, 2 |
| 64 GPUs | `[mamba,attn,moe_router]` | 6934181 | 5.723 | 1.630 | 76.06 / 95.11, 40 |

Learning: at 64 GPUs `[mamba]` graphs average 5.194 s against 5.213 s for the overlap alone (−0.4%,
NULL) and the wider set loses 9.8%; the runs that retried are the slow ones. The graphs work and remove
the launch-bound idle, but there is no memory to run them in: the overlap already reserves 91.7 GB. Two
fallbacks measured on 1-node reduced-depth setups do not fix it — `model.ep_overlap_early_attn_memory_release`
saves 0.2–0.9 GB and costs +2.6% at 21 layers (jobs 6934235 vs 6934234) and +1.9% at 28 layers (it
exposes the forward combine); `garbage_collection_threshold:0.9` in `ISAMBARD_CUDA_ALLOC_CONF` saves
0.6 GB at no cost (28 layers). A graph pool cannot be shared with ordinary
allocations, and the overlap's extra reserve is mostly fragmentation of the compute stream's pool under the
interleaved schedule (E-044's allocation history), not memory the communication stream holds back: copying
HybridEP's dispatch output into the compute stream's pool did not lower it (64 GPUs: 91.06 → 91.99 GB
reserved, medians of ranks, 6934713 against 6934714; allocation history with the copy, 6934764), nor did the
allocator settings tried (6934347, 6934851, 6934959, 6934960), while chunked cross-entropy, which takes the
loss buffers off the top of the heap, did (6935047). Freeing reserved memory is now the gate for stacking
anything further on the overlap.

### E-045 · trace of the EP-overlap champion · 2026-09-29 · the overlap exposes Mamba's launch cost
Job 6933850 (iterations 30 and 40, ranks 0 and 5, no stacks). The profiler inflates this schedule
more than the previous one — profiled window 6.25 s against a ~5.2 s real step — so idle is read for
where it sits, not its size. `hybrid_ep::device_sync_kernel` falls from 0.437 s to 0.124 s per
iteration (the waiting the overlap hides); HybridEP's dispatch/combine kernels rise from 0.295 to 0.378
s and the grouped GEMMs from 1.186 to 1.224 s (they now share SMs with the communication stream);
exposed data-parallel time is unchanged at 0.73–0.81 s. GPU-idle gaps of 50–1000 µs grow from 0.40 s
to 1.43 s, and on both ranks and both iterations the CPU op covering most of them is Mamba's fused
function — `MambaSplitConv1dScanCombinedFnBackward` 0.34–0.38 s and its forward 0.12 s (≈4,200 Triton
launches per iteration, mean kernel 90 µs). With the all-to-all wait gone, the GPU catches up with the
host there. `HybridEPDispatch` still synchronises the host 184 times per iteration. This motivated
E-046.

### E-044 · EP all-to-all / compute overlap for the hybrid model · 2026-09-29 · WIN (1.78–1.80×, repeated)
`comm_overlap.overlap_moe_expert_parallel_comm=true` runs the combined-1F1B schedule: the forward of one
microbatch is interleaved with the backward of the previous one, so each MoE layer's dispatch/combine
runs on a communication stream while the other microbatch's Mamba, attention or expert GEMMs run on the
compute stream. Upstream supports it for `GPTModel` only; Megatron-LM PR #4798 (open, head `1fdff667`)
adds it for the hybrid model, and it merged cleanly onto the pin (its first part, #4941, is already in
it). The port was vendored as patch 0003 and is a carried commit of the pin since 2026-09-30 (`3e3c83d50`);
its section of the patches README has the provenance, the review fixes made after E-044 ran (three more adaptations:
flat patterns keep the pin's checkpoint keys, the pin's behaviour where the PR changed it with the overlap off,
refusals of the settings the hybrid schedule gets wrong) and the deterministic smokes showing that the
vendored version trains exactly like the tree E-044 ran. That tree carried two adaptations, described in the
probe snapshot's `EPOV_NOTES.md`
(`/projects/a5k/public/logs/nano_pretrain_perf_campaign/snapshots/wt-e3dd4e56-epov`):
- the schedule plan groups the flat layer pattern into 23 `[Mamba/attention..., MoE]` units, so every
  all-to-all has compute to hide behind, without the bracketed pattern upstream needs (which renames
  every parameter and checkpoint key); `MCORE_HYBRID_OVERLAP_AUTOGROUP=0` restores upstream's
  one-unit-per-layer plan (−2.4% instead of −4.5% on the 1-node smoke);
- upstream frees the dispatched expert input when FP8 is on, assuming the experts keep an FP8 copy as
  TE's grouped MLP does; GroupedExperts stays BF16 and needs that input for its weight gradient
  (crash 6932873, `setStorage ... size 0`), so non-TE experts keep it.

Required with it: `model.mtp_num_layers=null` (the model provider's default `0` fails the "None or 1" assert — the first
64-GPU pair, 6933304/6933305, died on it; Nano has no MTP layers either way) and
`ISAMBARD_CUDA_MAX_CONNECTIONS=32` (with one hardware queue the two streams serialise).
1-node smoke (11 layers, 4 microbatches): ON 716–722 ms vs OFF 752–754 ms; ON-vs-OFF loss differences
(5–6.5e-4) sit inside OFF-vs-OFF (1.0e-3).

64 GPUs, E-028 posture from the frozen snapshot `wt-e3dd4e56-epov-full1`:

| Arm | Job | Mean (s) | Median | × baseline | W&B run max alloc / reserved GB, retries | Loss 41–50 |
|---|---|---|---|---|---|---|
| OFF (control: connections 32, MTP null) | 6933820 | 5.409 | 5.356 | 1.725 | 77.28 / 78.64, 0 | 6.911 |
| **ON, connections 32** | 6933731 | **5.197** | **5.143** | **1.795** | 78.39 / 91.69, 0 | 6.879 |
| **ON, connections 32 (repeat)** | 6933837 | **5.228** | **5.180** | **1.784** | 78.43 / 91.53, 0 | 6.911 |
| ON, connections 1 | 6933732 | 5.622 | 5.551 | 1.659 | 78.41 / 89.51, 0 | 6.887 |

Learning: the two ON runs (5.197 / 5.228 s, 0.6% apart) are −3.4% against the five runs without the
overlap (5.388–5.409 s, the paired control included), with loss in band. It is modest because
HybridEP's intra-node all-to-all is already short; what it hides is mostly the ranks' waiting in
`device_sync` (E-045). Its memory cost is in *reserved*, not allocated: +1.1 GB allocated but +13 GB
reserved (78.6 → 91.7 GB, W&B run max, last rank; the runs without it sit at 78.6–78.7). A 64-GPU
allocation history (jobs 6934763/6934764) puts most of that reserve in fragmentation under the interleaved
schedule rather than in the communication stream's own pool: the compute pool's occupancy peaks at 32.8 GB
against 41.1 GB reserved, and the communication pool sits 3.7 GB above its own peak. That leaves ~3.5 GB
below the device, which is what CUDA graphs ran into (E-046).

### E-039–E-043 · NCCL, GC, NUMA and Mamba-chunk probes on the HybridEP champion · 2026-09-29 · all NULL
All on the E-028 posture; NCCL variables go through `ISAMBARD_ENV_OVERRIDES` because the launcher exports
its own values unconditionally, and each run's per-rank `[env-overrides]` lines show them engaged.

| Entry | Lever | Job | Mean (s) | Median | × baseline | Loss 41–50 |
|---|---|---|---|---|---|---|
| — | none (champion repeat, same window) | 6932856 | 5.394 | 5.338 | 1.729 | 6.884 |
| E-039 | `NCCL_MIN_NCHANNELS=8`, `NCCL_NCHANNELS_PER_NET_PEER=8` | 6932857 | 5.401 | 5.347 | 1.727 | 6.904 |
| E-040 | `NCCL_BUFFSIZE=16777216` | 6932858 | 5.441 | 5.389 | 1.714 | 6.911 |
| E-041 | `gc.freeze()` after the setup collect in `train.py` | 6932862 | 5.388 | 5.384 | 1.731 | 6.889 |
| — | none (champion repeat, same window) | 6932890 | 5.409 | 5.361 | 1.724 | 6.891 |
| E-042 | `torchrun --numa-binding=node` | 6932889 | 5.394 | 5.334 | 1.729 | 6.906 |
| E-043 | Mamba SSD `chunk_size` 256 (default 128) | 6932900 | 5.440 | 5.369 | 1.715 | 6.896 |

Learning:
- The expert-DP ring is not NCCL-configuration bound: more channels or larger buffers change nothing.
- `gc.freeze()` removes the manual-GC spikes (iterations 31/41: 5.46/5.43 s instead of 5.93/5.85) but
  the mean does not move within noise.
- NUMA binding changes nothing because ranks are already bound: without the flag each rank's
  `Cpus_allowed_list` is its own GPU's Grace (local rank *i* → CPUs 72*i*…72*i*+71, matching
  `nvidia-smi topo -m`); only ~15 of ~192 threads per rank, created before the affinity is applied, had
  last run on another node.
- Chunk 256 is slightly slower; 128 stays.
- The champion now has four runs within 0.4% (5.388–5.409 s).

### E-038 · nccl-tests on the expert-ring pattern · 2026-09-29 · launch failed (replaced by training probes)
Intent: measure the expert data-parallel reduce-scatter / all-gather bandwidth under NCCL channel, buffer,
chunk and protocol variants with the in-container nccl-tests 2.19.6 (16 nodes × 4 tasks, `-g 1`,
`NCCL_TESTS_SPLIT_MASK=0x3` → four concurrent 16-rank rings with one GPU per node, the training pattern).
Neither launch mode works from the container: `srun --mpi=pmix` bootstraps MPI but every NCCL connection
handshake fails in the network plugin (`NET/OFI Request ... completed with error ... Error: 16 (Device or
resource busy)`, jobs 6932830, 6932835); the cluster default `--mpi=cray_shasta` cannot bootstrap the
image's Open MPI (`OMPI was not built with SLURM's PMI support`, job 6932842). The variants are measured
instead as training probes through `ISAMBARD_ENV_OVERRIDES` (the launcher exports its NCCL values
unconditionally, so plain environment variables would be overwritten).

### E-037 · microbenchmark: per-expert cuBLAS weight gradient accumulating into `main_grad` · 2026-09-29 · NULL
Motivation (E-036): the grouped weight-gradient GEMMs run at ~410 TFLOP/s and each is followed by a bf16
add into `main_grad` (~0.1 s per step). One GH200 (job 6932828), 32 experts × 49,152 rows, per call:

| Routing | Projection | `torch._grouped_mm` | grouped + add | per-expert `addmm_` (β = 1) | per-expert vs grouped + add |
|---|---|---|---|---|---|
| uniform | fc1 (2688→1856) | 0.806 ms (608 TFLOP/s) | 1.059 ms | 1.251 ms | 0.85× |
| uniform | fc2 (1856→2688) | 0.982 ms (499) | 1.073 ms | 1.136 ms | 0.94× |
| skewed (0.3–1.7× per expert) | fc1 | 1.035 ms (474) | 1.270 ms | 1.129 ms | 1.12× |
| skewed | fc2 | 1.035 ms (474) | 1.258 ms | 1.127 ms | 1.12× |

Results are bitwise identical. The per-expert path wins only under skew, and it needs the per-expert
counts on the host — the blocking copy H32 removed. Not pursued. Learning: routing skew costs the grouped
GEMM itself up to 22% (608 → 474 TFLOP/s on fc1), beyond the all-to-all waiting E-036 attributes to it.

### E-036 · trace of the HybridEP champion (E-028 posture) · 2026-09-29 · budget for the last 0.72 s
Job 6932683 (profile of iterations 30 and 40, ranks 0 and 5, no stacks; the profiled run itself scores
6.061 s mean because of the capture). Rank 0, iteration 40, window 5.754 s: compute union 4.353 s, NCCL
1.330 s of which **0.712 s exposed**, **idle 0.689 s (12.0%)**, almost all in 10–500 µs gaps.

| Component | Time (s) | Note |
|---|---|---|
| grouped expert GEMMs | 1.186 | forward 0.351 (514 TFLOP/s), dgrad 0.395 (457), wgrad 0.440 (410) |
| dense GEMMs (FP8 + bf16) | 0.803 | output-layer weight gradient 712 TFLOP/s |
| `hybrid_ep::device_sync_kernel` | 0.437 | 1,472 calls: ranks waiting for the slowest EP peer |
| HybridEP dispatch + combine kernels | 0.295 | ~198 MB remote per dispatch in ~0.4 ms — at NVLink rate |
| permute / unpermute | 0.170 | |
| elementwise | 0.450 | 3,056 bf16 gradient-accumulation adds = 0.137 |
| Mamba scan | 0.393 | |
| attention (cuDNN) | 0.213 | backward 3.4 ms × 48 |
| norm, router | 0.205 | |
| expert-DP reduce-scatter / all-gather | 0.764 / 0.660 | NIC time; 0.712 of the 1.33 s union exposed |

Host: `HybridEPDispatch` calls `cudaStreamSynchronize` 184 times per iteration (0.351 s of host time) to
size its output (`num_permuted_tokens=None`); the GPU idle attributed to it is only ~0.03 s. The sync-free
alternative (`moe_expert_rank_capacity_factor`) sizes every expert buffer to a static budget and drops
tokens above it, so it is not a free lever.
Learning — the exposed data-parallel time is a throughput floor, not an ordering problem: each iteration
needs ~1.42 s of NIC time for the expert reduce-scatter and all-gather, and only the last microbatch's
backward and the first microbatch's forward (~0.55 s) can hide it, so no reordering of buckets beats
~0.7 s exposed at micro-batch 1. Only a larger micro-batch (bigger windows) or faster NIC transfers move
it. The EP all-to-all costs ~0.9 s (waiting + kernels + permutation) and is serial with compute — the
target of upstream's expert-parallel overlap (Megatron-LM PR #4798 for hybrid models).

### E-031–E-035 · single-lever probes on the HybridEP champion · 2026-09-29
All on the E-028 posture; E-031–E-034 from the E-023 snapshot, E-035 from a copy with patch 0002
(`3rdparty/patches/megatron-lm/`, CUDA-graph `zeros_like` on a 0-dim tensor) applied:

| Entry | Lever | Job | Mean (s) | Median | × baseline | Peak GB | Loss 41–50 |
|---|---|---|---|---|---|---|---|
| E-031 | `model.moe_shared_expert_overlap=true` | 6932678 | 5.458 | 5.393 | 1.709 | 73.42 | 6.878 |
| E-032 | `ISAMBARD_CUDA_MAX_CONNECTIONS=32` | 6932681 | 5.424 | 5.339 | 1.720 | 73.42 | 6.881 |
| E-033 | `model.moe_permute_fusion_into_hybridep=true` | 6932682 | 5.419 | 5.367 | 1.721 | 73.42 | 6.916 |
| E-034 | `train.micro_batch_size=2` | 6932766 | OOM | — | — | — | — |
| E-035a | TE CUDA graphs, `cuda_graph_modules=[mamba]` | 6932739 | 5.350 | 5.309 | 1.744 | 73.46 | 6.936 |
| E-035b | TE CUDA graphs, `[mamba, attn, moe_router]` | 6932741 | 5.357 | 5.268 | 1.741 | 73.46 | 6.886 |

E-035 also sets `model.cuda_graph_impl=transformer_engine`, `model.cuda_graph_scope=null`,
`model.use_te_rng_tracker=true`, `NCCL_GRAPH_REGISTER=0` and `ISAMBARD_CUDA_ALLOC_CONF=expandable_segments:False`;
the logs show 23 and 52 graphable layers captured (0.4 / 0.9 s).
Learning:
- E-031–E-033 are NULL. E-032's engagement is not visible in the log (`pipeline_env_activate.sh` sets
  `CUDA_DEVICE_MAX_CONNECTIONS` from the knob silently), so it is also unverified.
- **E-034:** micro-batch 2 OOMs at iteration 1 in `vocab_parallel_cross_entropy`'s fp32 cast of the
  logits (8.00 GiB requested with 89 GiB already allocated). E-015 put micro-batch 2's extra activations
  at ~19 GB; only the `moe` recompute frees that, and it costs 1.03 s (H16), so the route stays closed
  unless ~12 GB is freed elsewhere.
- **E-035:** CUDA graphs engage but gain only ~0.8% (5.35 vs 5.39–5.41), inside the run-to-run spread:
  the idle gaps sit mostly outside the graphable layers — in the routed-expert path, the dispatcher and
  the gradient hooks. The table's memory is the log line written after iteration 1, before the graphs
  are captured (after `cuda_graph_warmup_steps` = 3), so it cannot show the graph pool; the W&B run
  maxima (last rank) do: 74.99 GB allocated / 82.63 GB reserved for `[mamba]` and 74.99 / 90.26 for
  `[mamba, attn, moe_router]`, against 77.3 / 78.6 without graphs, 0 allocator retries. Not adopted on
  its own.

### E-030 · repeat of the HybridEP champion (E-028) · 2026-09-29 · CONFIRMED (1.73×)
The E-028 posture and snapshot, launch line unchanged apart from the job and W&B names (job 6932677, W&B
`nano_pretrain_perf_probe_hepchamp_rep2`): mean **5.388 s** (median 5.322, p10 5.307, p90 5.448, min 5.300,
max 5.927) → 12,163 tok/s/GPU, 25.58% MFU, **1.731×**; lm loss (41–50) 6.854; peak 73.42 GB. Learning:
E-028 reproduces to 0.35%; the HybridEP champion stands at 5.39–5.41 s (1.73×).

### E-027–E-029 · FP8 champion repeat, HybridEP retest, 128 MB buckets · 2026-09-29
All on the E-024 posture (champion + H32 + FP8 dense), worktree snapshot:

| Entry | Lever | Job | Mean (s) | Median | × baseline | Peak GB | Loss 41–50 |
|---|---|---|---|---|---|---|---|
| E-027 | none (repeat of E-024) | 6932616 | 5.751 | 5.686 | 1.622 | 73.45 | 6.890 |
| E-028 | **HybridEP dispatcher** (`moe_token_dispatcher_type=flex`, `moe_flex_dispatcher_backend=hybridep`) | 6932618 | **5.407** | **5.341** | **1.725** | 73.42 | 6.892 |
| E-029 | `comm_overlap.bucket_size=134217728` (128 MB vs 500 MB) | 6932619 | 5.823 | 5.760 | 1.602 | 73.45 | 6.890 |

Learning: E-024 reproduces to 0.05%. **HybridEP is now −6%** where E-005 measured it NULL: then,
GroupedExperts copied HybridEP's device-side per-expert counts back to the host (`.to("cpu")`, a blocking
sync every MoE layer and microbatch); with H32 the counts stay on the device and the dispatcher's
sync-free path pays off. 128 MB buckets are slightly worse than 500 MB (NULL). New best: **5.407 s
(1.725×)**.

### E-026 · gate microbenchmark: FP8 routed-expert GEMMs · 2026-09-29 · GATE FAILED (lever closed)
One GH200 (job 6932663), `torch._scaled_grouped_mm` with rowwise E4M3 inputs and per-expert column-wise
weight scales vs BF16 `torch._grouped_mm`, at the Nano EP4 shapes (49,152 rows, 32 local experts):

| GEMM | BF16 | FP8 fast-accum | FP8 exact-accum | unfused input quantisation |
|---|---|---|---|---|
| fc1 [49152×2688]×[32×2688×1856] | 0.835 ms (587 TFLOP/s) | 0.476 ms (1.76×) | 1.567 ms (0.53×) | 1.271 ms |
| fc2 [49152×1856]×[32×1856×2688] | 0.746 ms (658 TFLOP/s) | 0.666 ms (1.12×) | 2.081 ms (0.36×) | 1.037 ms |

Only the fast-accumulation forward kernel beats BF16; the backward GEMMs need the exact accumulator (TE's
split-accumulator default for dgrad/wgrad), which is 2–3× slower than BF16, and unfused quantisation costs
more than the GEMM it feeds. The pre-registered gate (≥ 1.25× fwd+bwd including quantisation) fails:
FP8 routed experts are closed on this stack; FP8 stays on the dense TE linears (E-024).

### E-023–E-025 · H32, FP8 on dense layers, M3 — worktree snapshot · 2026-09-29
Snapshot `/projects/a5k/public/logs/nano_pretrain_perf_campaign/snapshots/wt-e3dd4e56-fp8-m3/` (the
worktree with its uncommitted code, `WORKTREE.diff` recorded), all on the E-022 champion posture:

| Entry | Lever | Job | Mean (s) | Median | × baseline | Peak GB | Loss 41–50 |
|---|---|---|---|---|---|---|---|
| E-023 | H32 sync-free offsets (the control: champion + this snapshot's code) | 6932551 | 5.968 | 5.878 | 1.563 | 75.40 | 6.846 |
| E-024 | **+ H27 FP8 current scaling on the dense TE linears** (`mixed_precision=nemotron_h_bf16_with_fp8_current_scaling_bf16_grad_reduce`; routed experts stay BF16) | 6932552 | **5.754** | **5.702** | **1.621** | 73.45 | 6.937 |
| E-025 | + M3: `train.micro_batch_size=2` + `fine_grained_activation_offloading` of `expert_fc1`/`moe_act` in GroupedExperts | 6932553 | 9.583 | 9.531 | 0.973 | 88.77 | 6.910 |

Learning:
- **H32** is ~1% faster than the E-022 repeats (inside noise, expected direction) and is a
  correctness-neutral removal of 736 host syncs per iteration — kept.
- **FP8 on the dense layers** is −3.6% against its same-snapshot control and −2 GB; the 2+2 edge layers
  stay BF16 (Nemotron-H recipe). Its loss sits at the top of the observed band, so it needs a repeat and,
  before adoption, the paired loss-parity run. New best: **5.754 s (1.62×)**.
- **M3 is +60% slower**: offloading the expert activations over NVLink-C2C at micro-batch 2 stalls the
  step (and still peaks at 88.8 GB). With E-015's small micro-batch-2 dividend this closes the
  micro-batch-2 route; the offload code was removed.

### E-022 · repeats of champion + H13 (fp32-SSM off) · 2026-09-29 · CONFIRMED (1.55×)
Two more runs of the E-019 posture from the same snapshot (jobs 6932526, 6932527): mean **6.026 / 6.029 s**
(medians 5.968 / 5.975, p90 6.10), 10,875 / 10,870 tok/s/GPU, 22.9% MFU, **1.548× / 1.547×**; lm loss (41–50)
6.909 / 6.881; peak 75.40 GB. With E-019 (6.249 s) the posture spans 6.03–6.25 s = **1.49–1.55×** across
three allocations — the original 1.5× goal is met reproducibly. Champion posture from here on:
H01 + host bundle + H15 + H16 `[moe_act]` + manual GC 10 + H13.

### E-018–E-021 · champion (E-013) repeat and three single-lever probes · 2026-09-29
All on the E-013 posture (H01 + host bundle + H15 + H16 `[moe_act]` + manual GC 10), manual snapshot,
separate 16-node allocations:

| Entry | Lever | Job | Mean (s) | Median | × baseline | Peak GB | Loss 41–50 |
|---|---|---|---|---|---|---|---|
| E-018 | none (champion repeat) | 6932452 | 6.494 | 6.450 | 1.436 | 73.86 | 6.933 |
| E-019 | **H13 fp32-SSM checkpoint patch off** (`ISAMBARD_FP32_SSM_STATE=0`) | 6932453 | **6.249** | **6.171** | **1.493** | 75.40 | 6.871 |
| E-020 | H14 `model.moe_router_fusion=true` | 6932454 | 6.738 | 6.690 | 1.384 | 73.84 | 6.912 |
| E-021 | H07 `ISAMBARD_CUDA_MAX_CONNECTIONS=8` | 6932455 | 6.688 | 6.644 | 1.395 | 73.86 | 6.905 |

Learning: **run-to-run noise is up to ~4%** — the identical champion measured 6.744 (E-013) and
6.494 (E-018). H14 and H07 sit inside that band (NULL). H13 is −3.8% against the better champion run and
−7.3% against the other, with loss in band: the fp32 checkpointed SSM scan (built for ~32K single-document
NaNs; "unnecessary at 8K") costs an extra fp32 scan pass per Mamba layer per microbatch. E-007's
"inconclusive" was one noisy run. **New best: 6.249 s (1.493×)**, to be confirmed by a repeat.

### E-017 · exploratory probe H03: `TORCH_NCCL_BLOCKING_WAIT=0` on the E-010 stack · 2026-09-29 · NULL + exit hang
Snapshot copy with launcher line 378 set to 0 (job 6932316). Mean 6.906 s (median 6.778, p90 7.091) vs
E-010's 6.862 / 6.688 — no gain (+1.3% by median, inside run-to-run noise). Engagement: the launcher's
`TORCH_CPP_LOG_LEVEL=error` hides torch's "NO watchdog thread" banner either way, but the run's shutdown
proves the watchdog was live — after `exit_interval` the ranks left at different times and the watchdog
began "dump debug info … due to a collective timeout", holding the job past 12 min until cancelled.
Learning: the host busy-wait under blocking wait is not a material cost at this posture, and turning it
off introduces an exit-time hang with `exit_interval`; dropped.

### E-016 · diagnostic X6: forced expert load balancing on the E-010 stack · 2026-09-29 · imbalance tax ≈ 5%
`model.moe_router_force_load_balancing=true` (job 6932315; benchmark-only — it changes routing and is never
adoptable). Mean 6.574 s, **median 6.350** vs E-010's 6.862 / 6.688 → routing imbalance costs ≈ 0.34 s
(≈ 5%) of the step at iterations 26–50, through all-to-all peer waiting and uneven expert GEMMs.
Learning: an upper bound on any load-balance lever; part of it is transient — production's steady state
(iterations ≥ 250) ran 3–6% faster than iterations 25–50 as the expert bias converges — so the 50-step
benchmark somewhat overstates imbalance cost.

### E-015 · exploratory probes X5: micro-batch 2 vs 1 at `["moe"]` recompute · 2026-09-29 · small (−2.6% median)
Both on H01 + host bundle + H15, recompute `["moe"]` (no manual GC). mbs 1 (job 6932314): mean 7.646 s,
median 7.652, peak 60.37 GB. mbs 2 (job 6932313, 4 microbatches per replica): mean 7.737, **median 7.451**,
peak **79.64 GB** (+19.3 GB), one 10.1 s GC excursion. The dividend is −2.6% by median — far below the
research plan's −0.5…−0.7 s (7–9%) from doubled DP-hiding windows and halved per-microbatch fixed costs.
Learning: at this posture DP exposure is not the bottleneck the plan modelled; M3 (activation offload to
fund mbs 2 without the MoE recompute) is therefore expected to buy ≲ 3% and is deprioritised pending one
direct test.

### E-014 · code H32: sync-free grouped-GEMM offsets · 2026-09-29 · implemented
`grouped_experts.py` computed `torch.cumsum(batch_sizes).to(device)` from pageable host memory in every
`_grouped_projection` call — a host-blocking copy, twice per MoE layer per microbatch (the 736
`_grouped_projection` stream syncs of E-003). Now `_group_layout` builds the backend's grouping once per
forward: int32 offsets on the device when the counts are already there (flex/HybridEP), otherwise summed on
the host and copied from pinned memory with `non_blocking=True`; `cublas_grouped` keeps its host int64
counts. Failing test first: `TestTorchGroupedBackend.test_forward_never_synchronizes_the_host[cpu|cuda]`
runs the forward under `torch.cuda.set_sync_debug_mode("error")` — both raised "called a synchronizing
CUDA operation" before the fix and pass after; all 45 GroupedExperts tests pass, including the bit-exact
per-expert reference.

### E-013 · exploratory probe: E-010 stack + manual GC every 10 iterations · 2026-09-29 · WIN (1.38×)
`train.manual_gc=true train.manual_gc_interval=10` (job 6932005): mean **6.744 s** (median 6.691, p10
6.629, p90 6.816, max 7.278) → 9,717 tok/s/GPU, 20.44% MFU, **1.383×**; lm loss (41–50) 6.912. Same median as
E-010 (6.688) but the 8–9 s excursions are gone: the only bumps left (~7.25 s at iterations 31 and 41) are
the scheduled collections. Learning: the in-window spikes were Python's automatic GC pausing ranks out of
step (MoE paper Table 15; perf guide "manual GC"); a longer production interval amortises the scheduled
collection further. Best posture so far.

### E-012 · analysis: the 2× budget and the expert data-parallel floor · 2026-09-29
Research workflow (four reports, a plan and a critique;
`/projects/a5k/public/logs/nano_pretrain_perf_campaign/research_2x/`). Findings that change the plan:
- **FP8 is a small lever here.** Dense TE linears (Mamba in/out projections, attention projections, shared
  experts) can run FP8 current-scaling after relaxing `grouped_experts.py`'s FP8 refusal (routed experts
  stay bf16 `torch._grouped_mm`); after the MoE recompute is gone the eligible time is ~0.85–0.9 s, so
  −0.1…−0.3 s. FP8 routed experts have a measured in-repo kernel at 0.57× bf16 and are out of scope
  unless a microbenchmark shows ≥ 1.25×. NVIDIA's own H100 Nano numbers are ~12.1–12.5k tok/s/GPU bf16 and
  14,890 FP8 (`docs/performance-summary.md`), with forced load balancing (benchmark-only) — the 2× target
  (14,052) sits at NVIDIA's best published H100 posture.
- **Expert data-parallel traffic sets a floor at micro-batch 1.** The expert-data-parallel group has one
  GPU per node (EP=4 fills the node), so every hop of the expert gradient ring crosses Slingshot: with bf16
  gradients ≈ 13.8 GB of reduce-scatter + 13.8 GB of all-gather per GPU per step over a 25 GB/s NIC ≈ 1.1 s
  of network time. At mbs 1 only the first microbatch's forward (~0.2 s) and the last one's backward
  (~0.5 s) can hide it, leaving ≥ 0.4 s exposed whatever else is done. **Micro-batch 2** doubles both
  windows and halves every per-microbatch fixed cost (all-to-all calls, host syncs, launches), and is the
  structural lever for 2× — funded by activation offload of the grouped experts (new code, M3) or by
  keeping the `moe` recompute (config-only, probe X5).
- Critique's re-anchored estimate for the remaining config/code levers: central 5.5–5.7 s (1.64–1.70×)
  from E-009; P(≤ 4.664 s) ≈ 5–10% without CUDA graphs on the Mamba layers, the 1F1B expert-overlap port
  (~1.4k lines upstream #4942) and M3. These become the critical path.

### E-011 · exploratory probe: E-010 stack with no recompute at all · 2026-09-29 · NULL
`model.recompute_granularity=null` instead of `[moe_act]` (job 6932003): peak 78.13 GB (+4.3 GB); mean
7.033 s (median 6.771) → 1.326×, vs E-010's 6.862. The `moe_act` recompute is effectively free; keep it
and spend the headroom elsewhere.

### E-010 · exploratory probe: full stack (H01 + host bundle + H15 + H16) · 2026-09-29 · best so far (1.36×)
Job 6932001: peak 73.86 GB; mean **6.862 s** (median 6.688, p10 6.644, p90 7.099, max 8.479) → 9,551
tok/s/GPU, 20.09% MFU, **1.359×** (1.41× at the ~6.61 s floor); lm loss (41–50) 6.924. Excursions at
iterations 32–34 and 39–40 cost ~3% of the mean (manual-GC probe 6932005 tests the GC hypothesis).

### E-009 · exploratory probe H01 + H15 + H16 (recompute `[moe_act]` only) · 2026-09-28 · WIN (1.32×)
With bf16 gradients freeing 19.1 GB, the MoE recompute (router, permute, two all-to-alls, grouped GEMMs,
shared expert re-run in every backward; MoE paper §4.1.4) is dropped: `model.recompute_modules=[moe_act]`
(the cheap expert activation recompute stays; `grouped_experts.py` supports it). Job 6931240 (manual
snapshot): **peak 73.86 GB** (8 GB below the as-is baseline despite no MoE recompute); mean **7.075 s**
(median 6.856, p10 6.770, p90 7.537, max 9.136) → 9,263 tok/s/GPU, 19.48% MFU, **1.318× (1.36× by the
median)**; lm loss (41–50) 6.943 — the top of the observed 6.865–6.943 run band, to be watched in the
loss-parity run. The series has a steady floor of 6.77–6.89 s with excursions at iterations 22–23 and
33–41 (up to 9.1 s); at the floor the stack is 1.38×. Learning: memory was the only thing standing
between this posture and the MoE recompute's removal; ~20 GB of headroom remains for further trades.

### E-008 · exploratory probe H01 + H15 (bf16 gradient accumulation + reduce) · 2026-09-28 · WIN (1.15×)
`mixed_precision=bf16_mixed_bf16_grad_reduce` — `bf16_mixed` under the `_bf16_grad_reduce` modifier of
`get_mixed_precision_config` (`src/megatron/bridge/training/mixed_precision.py`, unit-tested), i.e.
`bf16_mixed` with `grad_reduce_in_fp32=False`, applied by `MixedPrecisionConfig.setup()` to the model,
optimizer and DDP configs — the earlier four attempts were VOID because `bf16_mixed` hard-codes the fp32
value. Engagement: DDP config logs `grad_reduce_in_fp32=False`; peak **60.77 GB = 79.86 − 19.09**, the
predicted fp32→bf16 grad-buffer saving to the gigabyte. Job 6931239, manual snapshot
`/projects/a5k/public/logs/nano_pretrain_perf_campaign/snapshots/manual-e3dd4e56-bf16gr` (e3dd4e56 +
Megatron-LM 12c20d8f0 + the recipe): mean **8.105 s** (median 8.079, p10 8.036, p90 8.191) → 8,086
tok/s/GPU, 17.01% MFU, **1.151×**; lm loss (41–50) **6.8692 vs baseline 6.8690**. ~−4% on top of H01
alone, from halving the reduce-scatter's bytes (the RS tail inside the last backward).

### E-007 · exploratory probe H01 + H13 (fp32-SSM checkpoint patch off) · 2026-09-28 · inconclusive
`ISAMBARD_FP32_SSM_STATE=0` on top of H01 (job 6931200): mean 8.732 s (median 8.751, p10 8.194,
p90 9.432 — a far wider spread than any other run) → 1.068× vs 1.102× for H01 alone; lm loss (41–50)
6.915 (highest of all runs); peak after iteration 1 +1.5 GB (scan activations kept instead of
recomputed). Learning: no evidence of a gain; the spread suggests a placement effect, so H13 needs a paired
test before any verdict.

### E-006 · exploratory probe H01 + host-sync bundle · 2026-09-28 · promising
On top of H01: `logger.timing_log_level=1` (drops the per-microbatch forward/backward timer
`cuda.synchronize` pairs), `rerun_state_machine.check_for_nan_in_loss=false` (48 stream syncs / 0.41 s in
E-003), `ddp.check_for_nan_in_grad=false`, `logger.log_l2_norm_grad_to_tensorboard=false` (one norm +
`.item()` per parameter). Job 6931201: mean **8.321 s** (median 8.288, p10 8.251, p90 8.429) → 7,876
tok/s/GPU, 16.56% MFU, **1.121×**; lm loss (41–50) 6.884. ~−1.7% vs H01 alone, directionally consistent
with the ~0.4–0.6 s of host syncs it removes, but inside the run-to-run spread. The bundle was not
measured alone again; it is carried in every later rung from E-010 on, including the repeated
E-013/E-018 posture.

### E-005 · exploratory probe H22: HybridEP intra-node dispatcher · 2026-09-28 · NULL (alone)
`model.moe_token_dispatcher_type=flex model.moe_flex_dispatcher_backend=hybridep` on the E-001 posture (job
6930752; resolved config confirms flex/hybridep). HybridEP auto-detects the 4 NVLink peers
(`deep_ep/hybrid_ep_buffer.py`), so no environment variable is needed; kernels JIT-compile in the first
iteration (122 s vs 100 s). Runs with the `torch_grouped` experts, memory −0.2 GB. Mean **9.358 s**
(median 9.312) vs 9.328 → 0.997×; lm loss (41–50) 6.890. Learning: replacing NCCL's all-to-all transport
does not shorten the 1.29 s expert exchange — that time is dominated by peer waiting and the
surrounding host synchronisation, not by transport efficiency. Kept as a candidate to retest once the
MoE path is sync-free (H32) and inside a CUDA-graph posture, where its device-side metadata matters.

### E-004 · exploratory probe H01: `comm_overlap.overlap_param_gather: true` · 2026-09-28 · WIN (1.10×)
Mechanism: the distributed optimizer's parameter all-gather (1.36 s, fully exposed in E-001/E-003) is
issued asynchronously per bucket and waited on in the forward pre-hooks instead of synchronously after the
optimizer step (Megatron-Bridge performance guide; MoE paper Table 14; the Nano recipe's own default is
true, `src/megatron/bridge/recipes/nemotronh/nemotron_3_nano.py:178`). It must be set under
`comm_overlap:` — the pretrain recipe's `CommOverlapConfig.setup()` overwrites `ddp:`.
Prediction: −8…−13%. Job 6930751 (E-001 overrides + `comm_overlap.overlap_param_gather=true`, different
nodelist from E-001, single run): mean **8.464 s** (median 8.412, p10 8.370, p90 8.630) → 7,743 tok/s/GPU,
16.28% MFU, **1.102× the E-001 baseline**; lm loss (41–50) 6.8648 vs 6.8690. A single run on a different
placement from E-001; the gain is more than twice the cross-allocation spread (Repeatability), and every
later rung of the ladder carries the lever. Learning: ~0.86 s of the
1.36 s exposed all-gather is hidden — the first microbatch's forward is too short to hide all of it.

### E-003 · baseline trace at 64 GPUs / GBS 512 · 2026-09-28
Job 6930456 (same config as E-001, `ISAMBARD_TORCH_PROFILE_ITERS=30,40`, ranks 0 and 5, `with_stack=True`),
traces and tool outputs in
`/projects/a5k/public/profiles/nano_pretrain_perf_calib_b0_prof/20260928T214310-j6930456/{,analysis/}`
(`scripts/profiling/trace_analysis/*`, run in-container on an allocated node, job 6930713). Ranks 0 and 5
agree within 2%; rank 0, iteration 40:

| | seconds (profiled window 11.59 s) |
|---|---|
| compute union | 5.12 — GEMM 2.70 (dense + grouped CUTLASS), Mamba scan 0.96, elementwise 0.80, attention 0.23, MoE routing/permute 0.20 + 0.39, norm 0.13 |
| NCCL union / exposed / overlapped | 4.00 / **3.41** / 0.59 |
| — ReduceScatter (fp32 grads, 16 calls) | 1.80 sum, 1.47 union — ~37 GB/GPU at ~21–25 GB/s ≈ NIC line rate (bandwidth-bound) |
| — AllGather (params, 405 calls) | 1.35 — fully exposed (the `params-all-gather` timer) |
| — SendRecv (expert all-to-all, 1656 calls) | 1.29 — fully exposed, ~0.78 ms/call |
| idle | 3.06 (inflated by `with_stack`; the unprofiled step is 2.26 s shorter, so ~0.8 s real) |

Idle gaps: 57k, 63% of idle time in 100–500 µs gaps (launch/sync-bound). Host-sync census (rank 0):
`rerun_state_machine.validate_result` 48 stream syncs / 0.41 s (NaN-in-loss check), `token_dispatcher.
_maybe_dtoh_and_synchronize` 368 event syncs / 0.51 s (all-to-all split sizes), `grouped_experts.
_grouped_projection` 736 blocking H2D copies / 0.23 s (`cumsum(...).to(device)`), timers 79 device syncs /
0.10 s. Both DP collectives run NCCL's generic `RING_LL`-named kernel, but the measured bandwidth is at line
rate, so the protocol is Simple in practice (`NCCL_PROTO=^LL128` permits it) — not a lever.
**Budget of the unprofiled 9.33 s step: ~5.1 s compute, ~3.4 s exposed communication (DP ≈ 2.5, EP ≈ 1.3),
~0.8 s host-sync idle.** Re-ranking: hiding/halving DP communication (H01, H15) and the expert all-to-all
(H22 HybridEP, which also removes the dispatcher's D2H sync) dominate; the host-sync bundle (H06, H32, H04)
is worth up to ~0.8 s.

### E-002 · probe D2: is HybridEP available in the 26.04 image? · 2026-09-28 · YES
`import deep_ep` inside the container (no GPU work): `deep_ep 1.2.1+34152ae` at
`/usr/local/lib/python3.12/dist-packages/deep_ep/`, exports `Buffer`, `HybridEPBuffer`,
`HybridEpConfigInstance`; sources at `/opt/DeepEP`. The earlier "DeepEP/HybridEP incompatible" verdict
concerned the inter-node IBGDA path; intra-node HybridEP at EP=4 (NVIDIA's own H100 Nano pretrain
configuration, `scripts/performance/configs/nemotronh/nemotron_3_workload_base_configs.py` upstream) needs no
package build and is a wave-3 arm (H22).
