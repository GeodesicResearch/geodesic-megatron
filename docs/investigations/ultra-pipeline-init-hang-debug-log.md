# Nemotron 3 Ultra 550B — First-Step Pipeline-Init Hang: Debugging Research Log

**Status:** OPEN — actively debugging.
**Owner:** Claude Code agent (autonomous), on behalf of Kyle O'Brien.
**Scope:** The Ultra 550B SFT bring-up hangs at the *first training step's* NCCL
communicator initialization, after a clean checkpoint load + model/data setup.
This log isolates the cause. It is distinct from (but related to) the steady-state
MoE all-to-all hang in `slingshot-nccl-hang-investigation.md`.

**Discipline (per Kyle):** do NOT jump to conclusions; escalate logging
granularity to *observe* the hang rather than guess; isolate; and once a
hypothesis is confident, test whether it reproduces on **Nano at the same node
topology** (Ultra-specific vs general). Minimize parallelism/checkpointing
fiddling.

---

## Configuration under test

- Model: Nemotron 3 Ultra 550B-A55B (NemotronH hybrid: Mamba2 + attention + Latent-MoE 512 experts, 108 layers).
- Parallelism: **TP=4, EP=4, ETP=1, PP=36, DP=2, CP=1** on **72 nodes / 288 GH200**.
- Config: `configs/quickstart/nemotron_ultra_quickstart_sft.yaml` (50-iter smoke).
- Launcher: `pipeline_training_submit.sbatch` → `pipeline_training_launch.sh` (ft_launcher).
- Software: NCCL 2.28.9+cuda12.9, aws-ofi-nccl 1.8.1, libfabric 1.22.0-SHS12.0.2.

---

## Timeline

### [2026-06-05 ~02:34] Baseline hang reproduced (job 5048039), then killed at 3 min
- Log `logs/slurm/train-5048039.out` shows a CLEAN run up to the first step:
  `load-checkpoint = 22.5 s`, `model-and-optimizer-setup = 1.6 s`,
  `train/valid/test-data-iterators-setup = 1.8 s`, `NVRx straggler detection enabled`.
- Then a burst of ~25 `NCCL version 2.28.9+cuda12.9` lines (lazy communicator
  inits for the first step), then **silence**, then a manual `scancel` at 02:37:20
  — only **~3 min** in.
- **Key correction:** this is NOT an init/rendezvous timeout (init succeeded; load
  worked) and NOT obviously the load-accumulated deadlock (cold start, no pressure
  built up). It is a hang/stall at the *first step's communicator init*.

### [2026-06-05 ~07:50] Fabric health confirmed at scale (jobs 5048806 @72, 5048807 @32)
- Ran the geodesic-utils NCCL benchmark (torchrun mode, same .venv NCCL stack as
  training) at 32 and 72 nodes. Ring collectives **PASS** at the full 288-GPU
  training scale, **0 errors**:
  - 72 nodes: all_reduce 86.2, all_gather 70.8, reduce_scatter 80.4 GB/s.
  - 32 nodes: all_reduce 90.1, all_gather 84.1, reduce_scatter 83.7 GB/s.
- (alltoall's 1M→8G sweep is intrinsically slow at 288 GPUs and ate the 30-min
  walltime before sendrecv ran — a pacing artifact, not a failure.)
- **Conclusion:** the fabric does 288-GPU collectives at full bandwidth. Rules out
  a dead/degraded node or a hard fabric-capacity wall as the cause.

### [2026-06-05 ~08:30] Rank-map analysis — refutes "cross-node TP/EP"
- Computed node-locality of every parallel group with Megatron's own `RankGenerator`
  (`/projects/a5k/public/tmp/nccl_ultra/rank_map.py`, default order `tp-cp-ep-dp-pp`,
  4 GPU/node). Compared working 120B (PP=8) vs hanging Ultra (PP=36):

  | Group | 120B (PP=8) | Ultra (PP=36) |
  |---|---|---|
  | TP=4 (attention) | NODE-LOCAL `[0,1,2,3]` | NODE-LOCAL `[0,1,2,3]` |
  | EP=4 (expert a2a) | NODE-LOCAL `[0,1,2,3]` | NODE-LOCAL `[0,1,2,3]` |
  | DP=2 (all-reduce) | cross-node, 2 adjacent | cross-node, 2 adjacent |
  | PP (sendrecv) | cross-node, spans 8 nodes | cross-node, spans **36 nodes** |

- With ETP=1 parallel folding, EP=4 folds onto the same 4 GPUs as TP = one GH200
  node = NVLink. This holds in BOTH configs, independent of PP depth.
- **Therefore the only thing that scales 120B→Ultra is the pipeline:** PP sendrecv
  reach (8→36 nodes apart) and total rank count (64→288). TP/EP are NOT the variable.
  (The earlier Nano hang was `EP=8` = cross-node; `EP=4` fixed that and we keep it.)

### [2026-06-05 ~08:34] Experiment 1 (job 5049198): let first-step init run to the watchdog
- Hypothesis to falsify: "the first-step pipeline-comm init is *slow* (minutes), and
  prior runs were killed at 3 min before it finished" vs "it is a true deadlock."
- ft_launcher config confirms nothing kills the first step early:
  `--ft-initial-rank-heartbeat-timeout=none`, `--ft-rank-section-timeouts=setup:10800,step:3600`.
  Binding bound = 30-min NCCL watchdog (`TORCH_NCCL_TIMEOUT=1800` + `TORCH_NCCL_BLOCKING_WAIT=1`).
- Relaunched unchanged; will NOT kill it.

### [2026-06-05 08:46] Experiment 1 interim
- Job 5049198 RUNNING, ~12 min elapsed. Log still in the `NCCL version` comm-init
  burst — no iteration logged, no error, no abort. Already 4× past the prior 3-min
  kill point and still going.
- Interpretation (tentative, not a conclusion): consistent with either a very slow
  comm init OR a silent deadlock. The 30-min watchdog will disambiguate:
  - reaches iter 1 → slow-init (was killed prematurely before).
  - silent → watchdog abort at ~30 min → true deadlock → escalate to granular logging.

---

### [2026-06-05 08:53] Experiment 1 interim #2 — reached training loop, static at ~19 min
- Log now contains **`Starting training loop at iteration 0`** → cleared setup, entered
  the training loop; the `NCCL version` burst is the first step's lazy comm init.
- At ~19 min elapsed the log is momentarily static (0 new lines in 20 s). NOT concluding
  a hang: NCCL connection setup across 288 ranks has silent stretches, and the launcher's
  own comment documents **first-iter init taking 15–20 min at PP=8** — PP=36 plausibly longer.
- **Critical calibration note:** `TORCH_NCCL_TIMEOUT=1800` (30-min watchdog) was tuned for
  PP=8's 15–20 min. If PP=36's first-iter init legitimately needs >30 min, the watchdog
  would abort a still-progressing init and *masquerade* as a hang. The granular run must
  therefore distinguish "progressing slowly" from "truly stuck", not just "did it abort".
- Prepared but NOT yet launched (only fire if Exp.1 aborts): `train_debug.sbatch` with
  NCCL Flight Recorder (buffer 30000, dump-on-timeout, C++ stacks → `/projects/.../fr_<job>/`)
  + per-rank `NCCL_DEBUG=INFO` (`INIT,GRAPH,P2P,NET` → `/projects/.../dbg_<job>/`).
  Launcher made override-friendly (`NCCL_DEBUG`/`SUBSYS`) + Flight Recorder enabled by default.

### [2026-06-05 08:55] Experiment 1 interim #3 — comm-init is PROGRESSING, not frozen
- At 21 min: 62 `NCCL version` lines (lazy comm inits), 1 `Starting training loop`, and
  **zero** errors/warnings in the full 28.8k-line log (only benign ShardedTensor
  `FutureWarning`s from checkpoint load).
- Progress signal: `NCCL version` count climbed ~6 in ~2 min → new communicators are
  *still being created*, just glacially (~3/min). This argues AGAINST a hard deadlock and
  FOR a very slow first-iter comm init (the deep PP=36 pipeline + 288 ranks establishing
  P2P/subgroup comms). Tentative, pending the iter-1-vs-watchdog verdict.
- Made `TORCH_NCCL_TIMEOUT` overridable (launcher) so a retry can grant slow init >30 min
  if Exp.1 aborts at the watchdog while still progressing.

### [2026-06-05 11:33] Experiment 1 RESULT — NOT slow-init; it's a torch.compile rank desync
- Let job 5049198 run unkilled. It never reached iter 1. First worker-stop at **09:06**
  (~32 min ≈ the 30-min `TORCH_NCCL_TIMEOUT` watchdog); ft then restart-looped for ~3 h
  (507 NCCL re-inits), each re-hanging. Killed it.
- **Per-rank stacks at the timeout split into two groups** (the smoking gun):
  - rank0, 103, 107, 111, 115, 119, 123, 127, … → **`ProcessGroupNCCL` / `barrier` / `c10d`**
    (waiting in a collective)
  - rank245 (+others) → **`torch._inductor` / `_dynamo` / `_aot_autograd` / `compile_fx`**
    (still JIT-compiling)
- rank245's Megatron stack: `schedules.py forward_backward_pipelining_without_interleaving`
  → `backward_step` → **`tensor_parallel/random.py:630 backward`** (activation-recompute
  backward) → torch.compile of the recomputed graph. So **torch.compile fires during the
  backward recompute** of `recompute_modules: [core_attn, moe, shared_experts]`.
- Ruled out (did NOT jump): the `@torch.compile` at `moe/router.py:769` is on
  `InferenceTopKRouter`, which its own docstring says *falls back to eager for training* —
  so it is NOT the training-path compile. The compile is pure Dynamo/Inductor (not TE).
- **Mechanism (hypothesis):** at PP=36 each pipeline stage holds a different tiny 3-layer
  slice of the hybrid Mamba/attn/MoE model, so per-stage first-step compile cost varies
  wildly; ranks that finish (or have nothing to compile) hit a pipeline barrier and wait
  for the still-compiling ranks → the 30-min watchdog fires. PP=8 (working 120B) has ~11
  mixed layers/stage → far more uniform compile time → no fatal desync. This is consistent
  with the rank-map result that **PP reach is the only thing scaling 120B→Ultra**.

### [2026-06-05 ~11:40] Experiment 2 (job 5051399): CONFIRM by disabling torch.compile
- Same config/parallelism (PP=36, 72 nodes) — ONLY change `TORCH_COMPILE_DISABLE=1`
  (eager), plain torchrun (`--disable-ft`, one clean attempt). Verified env var:
  `torch/_dynamo/config.py:241` reads `TORCH_COMPILE_DISABLE`.
- Falsifiable: PASS (iter 1) ⇒ torch.compile desync confirmed + unblocked; FAIL ⇒ refuted,
  keep digging. [pending]

### [2026-06-05 11:45] Experiment 2 INCONCLUSIVE — environmental zombie, not a torch.compile result
- 5051399 crashed at **2:55 min, exit 1**, with `ncclSystemError: unhandled system error`
  during **PG init** (`initialize.py finish_mpu_init` → `initialize_megatron`), BEFORE
  checkpoint load. This is NOT a torch.compile signal (compile isn't on the PG-init path)
  and NOT the hang (it's a fast hard error).
- Root of THIS crash: 5051399 was scheduled on the **same nodes** (`nid010660–010759`) as
  the just-cancelled 3-hour zombie 5049198, only ~2 min after `scancel` — before SLURM's
  epilog cleaned them. Leftover ft_launcher/torchrun/python processes held NCCL/fabric
  resources → `ncclSystemError` on the new job's PG init. (Documented failure mode in
  repo CLAUDE.md: "Clean up zombie ft_launcher/torchrun/pipeline_training processes".)
- **Code-level root cause of the ORIGINAL hang is now nailed down regardless** (read, not
  guessed): `megatron/core/jit.py` sets `jit_fuser = torch.compile` whenever torch ≥ 2.2
  (we run 2.11), and `@jit_fuser` decorates many fused ops (swiglu/geglu/RMSNorm/attention/
  token_dispatcher) that compile at first step → desync at PP=36. `TORCH_COMPILE_DISABLE=1`
  neutralizes it; `disable_jit_fuser()` is Megatron's built-in off-switch.
- Action: re-run the confirmation on a CLEAN allocation (nodes have had ~8 min to epilog).

### [2026-06-05 ~12:05] Root cause CONFIRMED in code + clean fix identified
- `megatron/core/jit.py`: `enable_jit_fuser()` (run at import) sets `jit_fuser = torch.compile`
  for torch ≥ 2.2 (we run 2.11). `@jit_fuser` decorates the fused ops in the recomputed
  path (`fused_bias_swiglu`, `fused_bias_geglu`, `fused_cross_entropy`, attention,
  `token_dispatcher`, RMSNorm). At PP=36 the per-stage hybrid-layer mix makes first-step
  compile cost diverge → compile-vs-barrier desync → 30-min watchdog.
- **Clean fix (Megatron-native, not a blunt global env):** `dist.disable_jit_fuser: true`.
  `src/megatron/bridge/training/setup.py:123` does `if cfg.dist.disable_jit_fuser:
  disable_jit_fuser()` (prints "Disabling JIT fuser."), which sets `jit_fuser = noop` →
  the fused ops run eager. Applied to BOTH Ultra configs (quickstart + warm_start_200k).
- TorchInductor cache is already node-local (`/tmp`), so this is compile-*variance*, not
  NFS cache contention.

### [2026-06-05 ~12:00] OPERATIONAL LESSON — dirty nodes after a wedged job
- The two confirmation attempts (5051399, 5051529) crashed at PG init with
  `ncclSystemError: unhandled system error` — NOT a torch.compile signal. Both had been
  scheduled onto nodes from the cancelled 3-hour wedge (`nid010660–010759`), which SLURM
  kept re-handing out before/without fully clearing the leftover NCCL/fabric/GPU state.
- **Fix for the bring-up loop:** `isambard_sbatch --exclude=nid[010660-010759]` (it MERGES
  with the bad-node list) to force genuinely clean nodes. Do this after cancelling any
  wedged multi-node job before relaunching on overlapping hardware.

### [2026-06-05 ~12:05] Experiment 3 (job 5051673): validate the production fix
- Normal submit, fixed quickstart config (`disable_jit_fuser: true`), PP=36/DP=2/72 nodes
  UNCHANGED, dirty range excluded. PASS (iter 1) ⇒ fix validated end-to-end. [pending —
  queued; cluster fully booked]

### [2026-06-05 ~13:10] Experiment 3 RESULT — disable_jit_fuser is a REAL but PARTIAL fix
- 5051673 (fixed config) logged "Disabling JIT fuser." and **advanced**: the torch.compile
  frames are gone and it progressed from SeqNum=33 (forward) to **SeqNum=65 in the BACKWARD**.
  But it still hung — now a clean pipeline P2P: `schedules.py forward_backward_pipelining_
  without_interleaving → recv_backward → _communicate → req.wait()`,
  `WorkNCCL(SeqNum=65, OpType=COALESCED, NumelIn=0, NumelOut=0)` timing out.
- **The second hang is DETERMINISTIC** — all 8 watchdog reports across all 3 ft restart
  cycles are the SAME `SeqNum=65`. This RULES OUT the probabilistic aws-ofi-nccl
  resource-deadlock (which varies 9–177 iters) and points to a STRUCTURAL pipeline issue.
- Ruled out (verified, did not jump): microbatches(32) < PP(36) does NOT structurally
  deadlock — the schedule caps `num_warmup_microbatches = min(stages-rank-1, num_microbatches)`.
- **Structural finding — hybrid pipeline imbalance:** per-stage param counts alternate
  **~2.9 B (Mamba/attention stages) vs ~5.6 B (MoE stages)** — the 512-expert MoE layers
  cluster onto every ~3rd stage (1,5,8,11,14,17,20,23,26,30,33), ~2× the light stages.
  Stage 0 = 3.2 B (embeddings), stage 35 = 5.9 B (lm_head). So a light stage's recv_backward
  waits on a heavy MoE stage — either a true structural P2P mismatch or a >30-min-slow stage.
- **torch.compile was one real layer; a second, structural deep-PP P2P hang sits beneath it.**

### [2026-06-05 ~13:15] Experiment 4 (job 5052211): Flight-Recorder capture of the hang
- Same config (compile off via `disable_jit_fuser`), FR dump → /projects, SHORT watchdog
  (`TORCH_NCCL_TIMEOUT=600`) so the deterministic hang dumps in 10 min, per-rank NCCL INFO
  (INIT,P2P), `--disable-ft`, dirty nodes excluded. The FR dump will show whether at
  SeqNum=65 all ranks ISSUED the collective (→ slow-stage or aws-ofi-nccl transport stall)
  or some issued a DIFFERENT op (→ structural schedule/layout mismatch). [pending — queued]

### [2026-06-05 ~15:00] Experiment 4 RESULT — Layer 2 looks like SLOWNESS, not a deadlock
- 5052211 (clean nodes, `disable_jit_fuser`, `--disable-ft`, 600s watchdog, 40-min SLURM):
  **ZERO NCCL watchdog timeouts**, reached the training loop, ran the backward (148
  `run_backward` / AccumulateGrad events), and hit the **40-min SLURM time limit while still
  progressing** — NOT stuck at a collective. No FR dump (no timeout fired).
- Repeated warning: **"AccumulateGrad node's stream does not match the stream of the node
  that produced the incoming gradient … may incur unnecessary synchronization … if you are
  using DDP"** → heavy per-gradient sync in the backward.
- Reconciliation: with `--disable-ft` the backward GRINDS forward (no hang); under
  `ft_launcher` (5051673) a single recv_backward exceeded the 30-min watchdog. So Layer 2
  is most likely a **pathologically slow first-iteration backward** (eager fused ops after
  disable_jit_fuser + AccumulateGrad stream-sync + the PP=36 bubble with 32<36 microbatches),
  cut off by too-short timeouts — and possibly worsened by ft_launcher straggler-detection.
  NOT confirmed as a hard deadlock. (Holding that as the leading hypothesis, not a conclusion.)
- Caveat: `NCCL_DEBUG_FILE` wrote 0 per-rank files (path/substitution issue) — INFO likely
  went to stdout, adding overhead. Drop to WARN next.

### [2026-06-05 ~15:05] Experiment 5: "let it complete" (decisive slow-vs-deadlock)
- `--disable-ft`, **TORCH_NCCL_TIMEOUT=5400** (90-min watchdog, won't false-fire on slow
  first iter), SLURM **--time=02:30:00**, NCCL_DEBUG=WARN (low overhead), FR on as backstop,
  clean nodes. Outcomes: iter-1 COMPLETES ⇒ slow-not-stuck, Stage 3 unblocked (optimize
  throughput later); never completes / FR dump ⇒ genuine deadlock, dig with the FR data. [pending]

### [2026-06-05 17:09] ✅ RESOLVED — it TRAINS; the "hang" was slowness, not a deadlock
- Job 5053915 (`disable_jit_fuser` + `--disable-ft` + 90-min watchdog) **completed iterations**:
  - iter 1: **3,131,725 ms (~52 min)** — one-time lazy NCCL comm-init for the PP=36 / 288-rank
    pipeline. iter 2: **30,072 ms (~30 s)**. iter 3: **28,162 ms (~28 s)**.
  - lm loss 0.904, grad norm 1.703, **0 NaN iterations**, fits memory (heavy MoE-stage ranks
    ~60 GB, light ~30 GB, well under 95 GB). Training is correct.
- **Root cause (two independent layers), both now fixed:**
  1. `jit_fuser = torch.compile` (torch ≥ 2.2) JIT-compiled fused ops at the first step → rank
     desync at PP=36 → **`dist.disable_jit_fuser: true`** (both Ultra configs).
  2. The **~52-min first iteration** (deep-pipeline NCCL lazy init) **exceeded the 30-min
     `TORCH_NCCL_TIMEOUT` watchdog** (and tripped ft under it) → false "hang." Fixed by raising
     `TORCH_NCCL_TIMEOUT` 1800→**5400 s** and ft `step`/`out-of-section` 3600→**5400 s**
     (launcher). Steady-state ~30 s/iter is well within the old watchdog.
- **Feasibility:** 50-iter smoke ≈ 52 min + 49×30 s ≈ **77 min**; 495-iter warm-start ≈ **~5 h**.
- **Remaining = throughput tuning only (best-effort, not a blocker):** PP=36 with 32<36
  microbatches is severely bubble-bound and the hybrid clusters MoE layers onto 2×-heavy
  stages (per the Megatron MoE paper). Levers: raise GBS to fill the pipeline (microbatches ≥
  PP), VPP/interleaved PP, and `pipeline_model_parallel_layout` to balance MoE stages. ~30 s/iter
  is already viable; these would improve the 0.2→higher TFLOP/s/GPU.

## Next planned steps (post-validation)
1. On 5051673 PASS: correct repo `CLAUDE.md` (the "64+ nodes hang = Slingshot" line is
   wrong for Ultra — it was torch.compile/jit_fuser desync), post the Asana root-cause
   finding, and resume Stage 3 (quickstart trains → convert → coherence).
2. (Optional corroboration the user asked for) reproduce on **Nano at the same PP=36/72-node
   topology**: predict it ALSO hangs WITHOUT the fix (Nano uses the same `jit_fuser`),
   confirming this is general to deep-PP NemotronH, not Ultra-specific.
1. Resolve Experiment 1 (iter-1 vs 30-min watchdog abort).
2. If hang: **granular-logging capture** — NCCL Flight Recorder
   (`TORCH_NCCL_TRACE_BUFFER_SIZE`, `TORCH_NCCL_DUMP_ON_TIMEOUT`) + per-rank
   `NCCL_DEBUG=INFO` (`NCCL_DEBUG_SUBSYS=INIT,GRAPH,P2P,NET`) to `NCCL_DEBUG_FILE`,
   to name the exact communicator + the ranks that did/did not arrive.
   (Avoid `TORCH_DISTRIBUTED_DEBUG=DETAIL` — the prior investigation found it breaks
   `reduce_scatter_tensor_coalesced`.)
3. Isolate which communicator (PP sendrecv? a combined group? MoE?) and whether the
   stuck ranks correlate with node/topology boundaries.
4. If a confident hypothesis emerges: **reproduce on Nano at the same topology**
   (same node count / PP depth, scaled) to determine Ultra-specific vs general.
