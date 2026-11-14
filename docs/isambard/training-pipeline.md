# Training pipeline

Launching, profiling, run identity, scoring and gates, environment variables, fault tolerance and the save policy.

## Training Pipeline (`training_*`)

### Files

| File | Purpose |
|------|---------|
| `pipeline_training_launch.sh` | Shared launcher: NCCL/CXI env vars, fault tolerance, srun + ft_launcher |
| `pipeline_training_submit.sbatch` | Thin SLURM wrapper: allocates nodes, calls `pipeline_training_launch.sh` |

Training script (called by the launcher):
- `pipeline_training_run.py` — Unified entry point for SFT, CPT, and from-scratch pretraining (dispatches via `--model nano|super|ultra --mode sft|cpt|pretrain`; `pretrain` uses the NVIDIA pretrain recipes + the `pretrain()` entry point, requires `dataset.data_path`, and loads no checkpoint unless the YAML sets one)

### Usage

```bash
# Via SLURM (allocates nodes) — extra args after the mode forward to the launcher:
# launcher flags (e.g. --disable-ft) parse as such, anything else falls through as
# Hydra overrides (benchmark runs pair --disable-ft with checkpoint.save=null)
isambard_sbatch --nodes=32 pipeline_training_submit.sbatch configs/<config>.yaml nano sft
isambard_sbatch --nodes=8  pipeline_training_submit.sbatch configs/<config>.yaml nano cpt
isambard_sbatch --nodes=16 pipeline_training_submit.sbatch configs/<config>.yaml super sft \
    --disable-ft train.train_iters=32 checkpoint.save=null

# Via salloc (interactive)
salloc --nodes=16 --gpus-per-node=4 --time=24:00:00 --exclusive
bash pipeline_training_launch.sh configs/<config>.yaml --model nano --mode sft
bash pipeline_training_launch.sh configs/<config>.yaml --model super --mode cpt
bash pipeline_training_launch.sh configs/<config>.yaml --model nano --mode sft --nodes 8 --nodelist node[001-008]
bash pipeline_training_launch.sh configs/<config>.yaml --model nano --mode sft --disable-ft
bash pipeline_training_launch.sh configs/<config>.yaml --model nano --mode sft --peft lora
```

`pipeline_training_launch.sh` options: `--model nano|super|ultra` (required), `--mode sft|cpt|pretrain` (required), `--disable-ft`, `--disable-straggler`, `--enable-pao`, `--peft lora|canonical_lora`, `--nodes N`, `--nodelist LIST`.

`cpt` and `pretrain` both read Megatron-native `.bin/.idx` data and both **require**
`dataset.data_path` in the config — there is no default corpus. They differ only in the
recipe: `cpt` uses the SFT recipe (warm-start LR 5e-6, which the CPT configs override down
to ~1e-6), `pretrain` uses `nemotron_3_*_pretrain_config` (from-scratch LR — 1.6e-3 Nano,
4.5e-4 Super and Ultra) and dispatches `pretrain()` instead of `finetune()`, whose assert
would demand a checkpoint.

**Config composition (`base_config:`).** A training YAML may name another under the top-level
`base_config:` key and then state only the fields it changes. The base path resolves against the
naming file's directory (never the working directory), a base may itself name a base, and a
cycle or a missing base raises. Mappings deep-merge; a list, a scalar or an explicit `null`
replaces the base value, as `OmegaConf.merge` would, and scalars read as `OmegaConf.load` reads
them (`5e-4` is a float). The composed mapping is then merged onto the recipe and the Hydra CLI
overrides apply last. Composition happens only where a config is read through
`scripts/training/config_compose.py` (`load_composed_yaml`): `pipeline_training_run.py` (and
`scripts/data/report_blend_coverage.py`, `scripts/data/predict_masked_counts.py` and
`scripts/data/held_out_documents.py` (all three through
`scripts/data/run_training_data.py`), which resolve a config through `pipeline/training/recipes.py`'s
`resolve_bin_idx_run_config`), `scripts/nemotronh_flops_estimator.py` (and
`scripts/telemetry/score_run.py`, which reads its config through the estimator), the launch blocks'
checks (`scripts/training/launch_blocks.py`, through which `code_identity.py` and `launch_width.py` read a config's
`code_identity:` and `launch_width:` blocks) and the config-test
helpers (`tests/unit_tests/campaign_config.py`, `test_control_pretraining_config.py`).
`configs/control_pretraining/stage_gate.sbatch`,
`scripts/hub/sync_bucket.py`, `scripts/hub/publish_models.py` and
`configs/control_pretraining/generate_epoch_chain.py` (which copies each chain's `posture_config`
into its links and reads its `parent_config`) read their configs as raw YAML, so a config they are pointed at must stay a
complete file. The Nano pretrain quickstart is the
first overlay.

**Launch width (`launch_width:`).** A training YAML may fix the width it trains at in a top-level block (read
through its `base_config:` chain): `nodes`, `gpus_per_node`, `data_parallel_size` and optionally
`nvlink_links_per_gpu`. The launcher then refuses `--nodes`/`--nodelist` and any allocation it cannot launch exactly
`nodes` nodes from. With `nvlink_links_per_gpu` it NVLink-sweeps the allocation (records in a directory of the launch's
own, `<log-dir>/nvlink/<run-id>/`) and trains on the first `nodes` healthy nodes, so the job may request spares (~130
for 128). The run refuses a world size other than `nodes × gpus_per_node`, or a
data-parallel size other than the block's, and logs `[launch-width] world_size=... data_parallel_size=...`. The block is
never merged into the run's settings, so no Hydra override can change it (`scripts/training/launch_width.py`;
`scripts/training/README.md`).

**Performance probes (one short job each).** Measure a training lever as its own
`pipeline_training_submit.sbatch` job on the quickstart posture (Hydra overrides and env knobs
on the launch line, `--time` calibrated to the measured runtime — a 50-iteration 64-GPU Nano
probe takes 9–13 min, so `--time=00:20:00`), not inside a held allocation: short jobs backfill
quickly and hold nothing idle. Launch code under test from a read-only copy of a commit (e.g.
`git archive` plus the pinned `3rdparty/Megatron-LM`), not from a working checkout that may be
edited while the job runs — bash reads the launcher by byte offset. An archive carries neither
`.git` nor ignored build products, so the copy also needs (1) a `REVISION` file at its root
naming the commit (plus any uncommitted diff it carries): with no `.git`, the profiler's
provenance reads the commit from it (`scripts/telemetry/code_revision.py` `code_revision`)
and otherwise records the commit as unresolved; and (2) the Megatron dataset helpers library,
`3rdparty/Megatron-LM/megatron/core/datasets/helpers_cpp*.so`, copied from a built checkout —
Megatron-LM git-ignores `*.so`, and without it rank 0 runs `make` in that directory at startup,
which a read-only copy cannot do. Compare a probe only against runs placed on a single Dragonfly
switch group (every launcher run logs `[run-identity] switch placement`): a run whose nodes span more
than one group is 2.5–6.6% slower (mean 4.3%), while single-group runs of one posture agree to ~0.5%
(SD). That placement penalty is what made the same 64-GPU Nano posture measure up to ~4% apart across
jobs (6.744 and 6.494 s/iter; 3.7% between two runs sharing 15 of 16 nodes): the slower run of each
pair spanned more than one group. A lever smaller than ~1% needs repeats before it counts. The Nano
pretrain campaign's log is `docs/investigations/nano30b-pretrain-perf-campaign.md` (E-047 for the
placement measurements).

### Profiling and run identity

- **Torch-profiler capture** (any launch): prefix with
  `ISAMBARD_TORCH_PROFILE=1 ISAMBARD_TORCH_PROFILE_ITERS=10,20` (also
  `_RANKS=0,9`; legacy `_WAIT=3` = single capture at iteration 5). Profiling runs
  against the STANDING quickstart config with overrides — there is no separate
  profile config to drift out of sync; the exact command (including the
  `logger.wandb_save_dir` that is mandatory alongside `checkpoint.save=null`) is in
  that config's header and in `docs/profiling-quickstart.md`. Artifacts (per-rank traces, provenance, config +
  resolved-config snapshots, raw-log copy) land in
  `/projects/a5k/public/profiles/<wandb-exp-name>/<run-id>/`.
  `ISAMBARD_TORCH_PROFILE_WITH_STACK` (default `1`) records Python stacks, which
  `scripts/profiling/trace_analysis/host_attrib.py` needs to attribute host time to code;
  stack walking inflates host launch time (~+18% wall on the 120B capture), so take a
  timing breakdown with `=0` and keep a stack capture for attribution (any other value
  fails at startup).
  Tutorial: `docs/profiling-quickstart.md`; reference: `docs/environment.md`
  "Profiling a training run"; implementation:
  `scripts/profiling/profiler_callback.py`.
- **Run identity**: every launcher run mints `ISAMBARD_RUN_ID`
  (`<timestamp>-j<jobid>`), echoed in the log banner, symlinked as
  `/projects/a5k/public/logs/megatron_runs/by-run-id/<run-id>.out`, used as the profile subdir name, and
  stamped into W&B summary (`run/isambard_run_id`, `run/raw_log_path`,
  `run/slurm_job_id`) by `scripts/telemetry/run_identity.py` — the join key
  between a W&B run, its raw log, and its profiles. The same summary carries
  `run/switch_count` and `run/switch_spread`, the run's Dragonfly placement,
  from the launcher's `scontrol`-derived `ISAMBARD_SWITCH_SPREAD` (`scontrol`
  does not exist inside the container, so the payload cannot compute it). Both
  are absent on runs not started through `pipeline_training_launch.sh`.
  Placement is worth ~18% on the 512-GPU Nano stage-1 pretrain posture — 137.8
  TFLOP/s/GPU across 2 switch groups against 114.7 across 8 — so compare a throughput
  number only against another taken at the same spread.
- **Scoring a run**: `scripts/telemetry/score_run.py <log> --config <training yaml>
  --hf-model <hub id or path> --gpus N --window FIRST LAST --loss-window FIRST LAST
  [--peak-tflops TF] [--wandb-peak-memory] [--json]`; for the Nano pretrain quickstart:
  `--config configs/quickstart/nemotron_nano_quickstart_pretrain.yaml --hf-model
  nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 --gpus 64 --window 26 50 --loss-window 41 50`.
  The sequence length and model FLOPs/token are read from the config (through its
  `base_config:` chain) and the HF `config.json` by `scripts/nemotronh_flops_estimator.py`'s
  library, and the peak defaults to the cluster profile's `CLUSTER_GPU_PEAK_TFLOPS` (989.4 on Isambard, GH200 dense
  BF16). The primary statistic is the **mean** step time over the window (node-hours scale
  with the mean); median, p10/p90, min/max and the iterations above 2× the median are reported
  beside it, then tokens/s/GPU, model TFLOP/s/GPU and MFU, mean loss, NaN/skipped counts,
  iteration-1 memory and the W&B run path, and every score records its inputs (log, config, HF
  config, GPUs, sequence length, logged GBS, FLOPs/token, peak, windows). Tokens per iteration
  are the global batch the window's own log lines report times the config's sequence length, so
  a batch set by a Hydra override is scored correctly; a window iteration that is missing or
  repeated, or two batch sizes in one window, raises. `--wandb-peak-memory` adds the W&B
  summary peaks and allocator-retry count, which are the last rank's, not a maximum over ranks.
  The maximum over ranks is `peak_memory_across_ranks`, read from the `[peak-memory]` line rank 0
  logs when the training loop ends (`train_utils.py` `gather_peak_memory`, one gather after the last
  step, saves included): the largest peak allocated and reserved memory, the rank holding the
  allocated one, and the largest and total allocator retries, also written to the W&B summary under
  `memory/across-ranks/`. A log that predates the summary has none, and so does a run whose loop was
  cut short (an OOM, a wedge, a SLURM wall-time kill); a loop ended by `exit_duration_in_mins` prints it.
- **Loss parity between runs**: `scripts/telemetry/loss_parity.py` compares runs of one config,
  seed and data-parallel width (iteration i consumed the same global batch in each). `band
  --reference A1 A2 [...] --candidate C --iterations 1 500 --window 50` is the test for a lever
  that changes numerics: per window, the candidate's mean `lm loss` must lie within [lowest
  reference - delta, highest reference + delta], delta being the largest window difference
  between two references (PASS/FAIL; `grad norm` the same way, PASS/FLAG). `--loss-half-width W`
  replaces delta for `lm loss` with a stated tolerance, and with it one reference suffices: a replay
  of one earlier run of the same config, such as a check of an upgraded stack, has no run-to-run
  spread to draw on (its `grad norm` band then has zero width). `identity
  --reference A --candidate B --iterations 1 30` is the test for a lever or knob-off path that
  claims exactness: per metric, the leading identical iterations and the first difference. Both
  also require every run to log the same learning rate and consumed samples at every iteration,
  and `--wandb` reads full-precision values from each log's W&B run (the log prints 7
  significant digits of the loss and 3 decimals of the grad norm). Exit status 1 on any FAIL.
  Default-mode training is not run-to-run deterministic (the grad norm differs from iteration 1),
  so an identity test needs `model.deterministic_mode=true` with `NVTE_ALLOW_NONDETERMINISTIC_ALGO=0`,
  `CUBLAS_WORKSPACE_CONFIG=:4096:8` and `MAMBA_DETERMINISTIC=1` (an `ISAMBARD_ENV_OVERRIDES` file) on
  every run it compares. mamba_ssm fixes its Triton autotune configurations at import, before Bridge
  turns on deterministic algorithms, so without the last one the SSD kernels are autotuned by timing
  and their block sizes, and with them the reduction order, can differ between launches.
  Megatron-Bridge refuses cross-entropy fusion in deterministic mode, so such a test runs without it.
- **Pre-registered loss gates**: `scripts/telemetry/loss_gate.py --spec <gate.yaml> --candidate <log>
  [--gate NAME]` runs band tests frozen in a YAML before the candidate exists: each named gate lists
  two or more references (each log named once), its range and window, the references' lm-loss spread
  when frozen (`lm_loss_delta`, which identifies the reference set), and optionally a fixed tolerance
  (`lm_loss_tolerance`) that replaces the spread as the band's half-width. Exit 1 when any gate FAILs,
  whatever the others; otherwise 2 when any is NOT EVALUATED (the log does not yet cover the range, the
  spread does not reproduce, a log or W&B read fails, or the candidate is one of the references, which
  a band cannot judge), and 0 when all PASS. The v2e2e arm's `loss_gate.yaml` is the first spec.
  `scripts/telemetry/score_gate.py --spec <gate.yaml> --scores-dir DIR` does the same for thresholds
  on the `score_run.py --json` files and `loss_parity.py band --json` reports in DIR: a `memory` gate fails a score with no `peak_memory_across_ranks`,
  more allocator retries or more peak allocated memory than its limits; a `speed` gate projects a
  candidate's mean step time relative to a reference's on the same nodes onto a stated step time and
  fails above its limit; a `first_loss` gate compares two runs started from the same weights on the same
  first batch at their first logged iteration and fails when the candidate's lm loss there differs from the
  reference's by more than its tolerance; a `loss_shift` gate bounds a `loss_parity.py band --json` report's
  candidate offset from the references' mean in every window and its rise from the first windows to the last.
  Both tools share the outcomes and the exit status (`gate_outcome.py`); the v2e2e probes run the arm's
  `score_gate.yaml` and `score_gate_midtrain.yaml`, because `score_run.py` exits 0 on any scorable log.
  `scripts/telemetry/run_watch.py --spec <watch.yaml> --log <segment log> ...` checks a running stage the same way:
  exit 1 on a stop condition (a result the rerun state machine rejected, which is how the gradient NaN check ends a
  run; a non-finite grad norm or lm loss, or an iteration line without `lm loss`; an iteration counted as nan or
  skipped; allocator retries; a segment whose `[env-overrides]` lines are not exactly the stage's
  `ISAMBARD_ENV_OVERRIDES` file, read with the launcher's own parser; a `[token-masking-counts]` line other than
  `predict_masked_counts.py`'s prediction for its iteration, `stop.count_prediction`) or a failing due loss gate, 2 on a
  due gate or
  a stop check it could not evaluate (each stop check runs on its own, so the others still stand) or a failure of
  the watch itself (never 1); flags (a loss spike, a loss gate's offset growing past a bound, a block of iterations
  further from a reference run than an envelope run is) are printed, never a stop. A segment that resumes from a
  save supersedes every earlier segment's records, saves and rejected results from its first iteration on, and their
  peak-memory summaries once it has logged an iteration; an iteration, counts or peak-memory line that cannot be
  parsed leaves the stops NOT EVALUATED unless a later segment has re-run its iteration. A gate
  named `--decided GATE=LOG` passed at an earlier check on that log and is not evaluated again while that log covers
  its range; the watch ends by naming the gates still undecided. The v2e2e arm's `watch_pretrain.yaml` and
  `watch_midtrain.yaml` are its specs.
  `scripts/training/stage_guard.py --config <guard.yaml>` runs that watch on a timer while the stage trains, on the
  tunnel under the host Python, handing each passed gate to later ticks as decided (a guard started again reads them
  from its record, from ticks judged under the same watch spec): on a stop it cancels the stage's
  live jobs by ID (the running segment and the successors pending on its singleton dependency, which only `squeue`
  lists) and exits; on a tick it could not evaluate (exit 2, or a watch, timeout or `sacct` failure) once the stage
  has reached its hold iteration while a loss gate is undecided it does the same; otherwise it alerts. Its record
  opens with the config, the spec's sha256 and the code revision, and its own failures (no started segment of the
  stage among them) are written there too.
  The v2e2e arm's `guard_pretrain.yaml` and `guard_midtrain.yaml` are its configs.
- **Reproducing an overridden posture**: the override YAML alone omits recipe defaults,
  CLI overrides and, for a `base_config:` overlay, every field it inherits (the
  profiler's `config_snapshot.yaml` is that overlay verbatim), but the bridge sends the
  FULL resolved config to W&B at startup — recover any run's exact posture from its W&B
  run's config tab (join via `run/isambard_run_id`) or from the profile's
  `resolved_config_snapshot.yaml`.

### Environment Variable Architecture

`pipeline_training_launch.sh` adds distributed-training-only vars on top of `pipeline_env_activate.sh`:
- All Slingshot/CXI NCCL vars (`NCCL_NET`, `FI_PROVIDER`, `FI_CXI_*`, etc. — 30+ vars)
- Fault tolerance vars (`TORCH_NCCL_TIMEOUT`, `TORCH_NCCL_RETHROW_CUDA_ERRORS`)
- Job-specific node-local paths (`TRITON_CACHE_DIR`, `TMPDIR`, `MEGATRON_CONFIG_LOCK_DIR`; activate
  puts HybridEP's JIT cache, `HYBRID_EP_CACHE_DIR`, under that `TMPDIR` instead of `$HOME/.deepep`)
- Module loading (`PrgEnv-cray`, `cuda/12.6`, `brics/aws-ofi-nccl/1.8.1`)

Every env var has detailed inline documentation.

**Per-launch overrides: `ISAMBARD_ENV_OVERRIDES=<file>`** changes any variable for one launch
without editing the launcher or activate, including those they set unconditionally — e.g.
`TORCH_NCCL_BLOCKING_WAIT=0` to get the NCCL watchdog and flight recorder back (see Fault
Tolerance). The file holds `KEY=VALUE` lines (`#` comments; values literal, no quote removal
or expansion). The launcher applies them three times: at its top, so its own knobs
(`ISAMBARD_NCCL_DEBUG`, `TRAIN_*`, `GEODESIC_CONTAINER_*`, `ISAMBARD_RUN_ID`, ...) take them;
again after its last export; and inside the container after `pipeline_env_activate.sh`,
immediately before ft_launcher/torchrun. It ends the launch before srun on a set-but-missing
file, a malformed or repeated line, one of its own shell variables (`MODEL`, `NNODES`,
`REPO_DIR`, `GEODESIC_REPO_DIR`, ... — they would change the launch, not its environment),
`CONTAINER_*` (override the `GEODESIC_CONTAINER_*` input instead) or a credential-looking key
(a `TOKEN`/`SECRET`/`PASSWORD`/`API_KEY` segment — every value is printed into the job log).
The banner lists each applied `KEY=VALUE`, and the ranks receive
**`ISAMBARD_ENV_OVERRIDE_KEYS`** (the keys, comma-separated; the launcher drops an inherited
one when no file is named). From it `pipeline_training_run.py` logs, once per node, the value
each key actually has in training —
`INFO:pipeline.training.env_overrides:[env-overrides] rank=<R> host=<host> KEY=<value> ...` (values shell-quoted;
the logger prefix means a log search must not anchor at the line start) — and raises if a
listed key is absent. An override is a plain assignment: to change something activate derives,
override its input (`ISAMBARD_OMP_THREADS=1`, not `OMP_NUM_THREADS=1`). Full contract:
`docs/environment.md` D2b.

`pipeline_env_activate.sh` (sourced inside the container, in the same shell that then execs
ft_launcher/torchrun) carries the universal knobs. Three are tunable:
`ISAMBARD_CUDA_MAX_CONNECTIONS` (default 1), `ISAMBARD_CUDA_ALLOC_CONF` (default
`expandable_segments:True`; a measured `False` arm showed no difference on the 512-GPU
Nano pretrain posture — the knob exists for A/Bs), and
**`ISAMBARD_OMP_THREADS` (default 8)**,
which sets `OMP_NUM_THREADS` and, whenever it is > 1, also `OMP_WAIT_POLICY=PASSIVE`. The
default matters because torchrun silently sets `OMP_NUM_THREADS=1` when the variable is
absent, single-threading the host-side AdamW of any CPU-offloaded optimizer onto one
Neoverse-V2 core: **21.36 s/iter / 73.70 GB at offload 1.0 with 8 threads, versus 22.79 /
76.78 at offload 0.5 single-threaded** — i.e. it strictly dominates the previous champion,
but the two arms differ in offload fraction as well as threads, so the delta is not
attributable to threading alone (the clean offload-1.0 single-thread arm was never run on
that nodelist; see the consultant tracker §C1b, preserved at
`/projects/a5k/public/logs/infr71_wave2/docs/consultant-training-stack-review.md`). Both arms are on the pre-`torch_grouped` expert path — this
A/B is about host-side Adam, so the expert backend does not move it. Threading is exactly
neutral (20.663 vs 20.654, identical peak)
when offload is off — which is what makes 8 safe as a universal default. `PASSIVE` is
load-bearing: GNU OpenMP idle threads spin-wait and these workloads are host-launch-bound.
Set `ISAMBARD_OMP_THREADS=1` to restore torchrun's behaviour. `pipeline_env_validate.py`
scores both as a check, so a silent regression fails loudly instead. **What it would cost
depends entirely on the posture**: on the shipped offload-off quickstart, essentially
nothing (20.663 vs 20.654); it only bites where a CPU-offloaded optimizer gives host AdamW
real work to do, and even there the 1.43 s/iter figure above is not a clean threading
delta — see §C1b in the tracker above before quoting it.

### Training-Specific Override for Isambard

The NGC image ships APEX, so `model.gradient_accumulation_fusion: True` works and is
the faster path — a measured ~1.1 s/iter win on the 120B quickstart (2026-07-24). It is
set `True` in the shipped quickstart. (This used to require a per-environment override;
with the venv gone, there is one answer.)

### Fault Tolerance

Slingshot/CXI causes intermittent NCCL collective hangs (~every 2-3 hours with EP=8 cross-node). The training pipeline uses a layered resilience stack:

1. **ft_launcher worker restart** (`--max-restarts=20`) — the per-node agent restarts failed workers, reloading from the latest checkpoint. ≤25 iters lost. (`cfg.inprocess_restart` is never set by `pipeline_training_run.py`, so there is no in-process layer on the shipped path.)
2. **Collective timeout** — the calling thread's blocking `wait()` throws after the YAML's
   `dist.distributed_timeout_minutes` (60 on the campaign configs; the launcher's
   `TORCH_NCCL_TIMEOUT`, 7200 s, covers only groups created without one) and the rank exits.
   This is not a watchdog: under the launcher's `TORCH_NCCL_BLOCKING_WAIT=1` torch creates no
   watchdog thread at all (next paragraph).
3. **srun `--kill-on-bad-exit=1`** — when a rank dies unrecoverably (or ft is disabled), the whole step ends instead of stranding the surviving ranks in a never-completing collective; with a `--dependency=singleton` chain the next segment then resumes from the latest checkpoint.

**A run that simply stops iterating leaves no evidence by itself — take it before cancelling.**
The launcher's `TORCH_NCCL_BLOCKING_WAIT=1` means torch creates **no watchdog thread** (its own
log line at process-group creation says so): no asynchronous timeout detection, no heartbeat
monitor, and no NCCL flight-recorder dump, on timeout or on demand — the recorder buffer
(`TORCH_NCCL_TRACE_BUFFER_SIZE`) is filled and never written, whatever the comment beside
`TORCH_NCCL_DUMP_ON_TIMEOUT` used to promise. The only timeout is the calling thread's `wait()` at
the YAML's `dist.distributed_timeout_minutes` (Megatron passes it to `init_process_group`; the
launcher's `TORCH_NCCL_TIMEOUT` covers only groups created without one), which throws after 60
minutes on the campaign configs. So a hang cancelled at the 30-minute mark takes everything with
it. `scripts/training/dump_hung_ranks.sh <jobid>` takes the evidence from outside first, into
`<log-dir>/nccl_trace/<jobid>/`: `processes.<host>`, the state and kernel wait channel of every
rank and of every helper it forked (dataloader workers and the multiprocessing bookkeeping
processes carry the rank's environment; a helper is one whose parent carries the same RANK),
recorded before anything is attached to; then every process's Python and native stacks via py-spy
(no cooperation from the rank needed; ptrace is unrestricted on the compute nodes; py-spy must be
on the host PATH of the compute nodes — once per user, `python3 -m pip install --user py-spy`,
which the shared home makes visible on every node — or named in `PY_SPY`), the rank's own to
`rank_<rank>.stack` and each helper's to `rank_<rank>.child-<pid>.stack`, skipping a process in an
uninterruptible wait, which cannot be attached to. The stacks show which collective each rank is
waiting in and what the ranks that never arrived are doing instead. Where a watchdog exists
(blocking wait off; for one launch, `TORCH_NCCL_BLOCKING_WAIT=0` in an `ISAMBARD_ENV_OVERRIDES`
file) the same run also triggers the recorder dump into that directory. Two 64-node
segments of the filtered stage-1 run wedged on 2026-09-11 with no NCCL warning, watchdog or
traceback in the log, and were cancelled before anything was captured. **The stalls that were
captured, on 2026-09-12, were not NCCL at all:** one rank's main thread was waiting on its
dataloader, that rank's worker sat in `cl_sync_io_wait` with a single Lustre read RPC to one OST
(OST0014, over kfi) outstanding and its read counters flat, its three node siblings spun in the
node-local expert all-to-all, and the other 252 ranks waited at their collective. The first such
stall cleared on its own after 6.3 min (an RPC timeout and resend, most likely); the run is not
wedged during one, so read `processes.<host>` for a `D`/`I` state with a Lustre wait channel before
cancelling anything, and give a stall the Lustre resend time before treating it as a wedge.

**ft_launcher timeout configuration** (set in `pipeline_training_launch.sh`):
- `--ft-rank-section-timeouts=setup:10800,step:7200,checkpointing:3600`
- `--ft-rank-out-of-section-timeout=7200` — must cover first-iter NCCL lazy init at PP=8+
- `--ft-initial-rank-heartbeat-timeout=7200 --ft-rank-heartbeat-timeout=7200` — heartbeats are
  an INDEPENDENT mechanism from the section timeouts. Omitting them is not "disabled": NVRX
  defaults to 3600 s / 2700 s, which is shorter than Ultra-550B's 45-75 min first iteration at
  PP=36 and produces a SIGKILL + restart loop that looks exactly like a fabric hang. The image's
  ft_launcher parses these as floats and rejects the literal `none`, hence explicit numbers.
  **Raising them to 7200 does not make them safe, it moves the wall.** The rank monitor is not
  guaranteed to receive an initial heartbeat at all; when it does not, the timeout stops being a
  liveness check and SIGKILLs a perfectly healthy job at exactly 7200 s. Signature, worth
  grepping before blaming the fabric: `[Cycle N] Did not get initial heartbeat. Waited 7200.00
  seconds`. Observed on Super-120B/64 GPUs and Nano-30B/512 GPUs; **not** universal — Ultra-550B
  trains ~4.7 h under the same default without a kill, so what decides delivery is unresolved.
- `calc_ft_timeouts=True` auto-learns step timeouts after first successful run. **Delete `ft_state.json`** from checkpoint dir if learned timeouts are too aggressive after config changes.

The `ft`/`nvrx_straggler`/`inprocess_restart` Python configs **cannot** be set via YAML or Hydra overrides (OmegaConf merge creates dicts, not dataclasses). They are set by `apply_resilience_config` (`pipeline/training/resilience.py`), on by default. Use `--disable-ft` to opt out.

`--disable-straggler` drops **only** the NVRx straggler detector and keeps `cfg.ft`: the
detector's rank-0 gather of per-GPU perf scores has been observed to OOM high-memory jobs after
~20 minutes of stepping, which is why the pretraining quickstarts (short, restart-free) simply
pass `--disable-ft` instead.

**It is not a milder `--disable-ft`, and it is the wrong reach for a long run.** It leaves the
heartbeat timeout above fully armed, so it does nothing about the 2 h wall. Before enabling ft
on any run expected to exceed the heartbeat, do the arithmetic the run now prints at startup:
compare `save_interval × s/iter` against 7200 s. If the **first** checkpoint lands after the
wall, ft is not fault tolerance — a heartbeat SIGKILL restarts from iteration 0 with an empty
checkpoint directory, and the run makes no net progress for as long as it is left alone.
`pipeline_training_run.py` logs the required s/iter whenever ft is on
(`max_seconds_per_iteration_under_ft_heartbeat`); read that line rather than rediscovering it.
Measured instance: the 500B control-pretraining baseline needed < 2.42 s/iter and ran at 4.276,
giving seven kills across eight attempts in 15.5 h with zero checkpoints written.

Where ft is dropped, a `--dependency=singleton` chain with `checkpoint.load == checkpoint.save`
supplies the recovery ft would have: the collective timeout ends a genuinely wedged segment, and the
next segment resumes from the latest checkpoint.

### Parallel Folding (expert_tensor_parallel_size)

```yaml
tensor_model_parallel_size: 4       # Attention: 4-way TP
expert_model_parallel_size: 4       # Experts: 4-way EP
expert_tensor_parallel_size: 1      # Experts NOT sharded by TP → enables folding
```

Keeps EP all-to-all on NVLink while using high TP for attention. Only PP crosses Slingshot.

### TensorBoard: always disabled — set `tensorboard_dir: null`

**We do not use TensorBoard** (Kyle, 2026-09-13). Set `logger.tensorboard_dir: null` in every
config you write or touch, which is the whole of it: `training/state.py`'s `tensorboard_logger`
builds a `SummaryWriter` only when that field is set, so null means no event files, no writer, and —
the part that bites — **no attempt to create the directory**.

**Omitting the key is not the same as nulling it, and omission is the failure that actually
shipped.** The recipes default `tensorboard_dir` to `./nemo_experiments/default/tb_logs`
(`recipes/common.py`), i.e. into the submitting checkout — the directory the pitfalls table of `docs/isambard/repository-map.md`
warns fills the disk — so a config that simply leaves the key out still builds a writer and logs
into the repo. Every training config under `configs/control_pretraining/` therefore states
`null` explicitly, and `TestTensorBoardIsDisabledEverywhere` fails a config that either names a
directory or stays silent. The `configs/PA/green-team/` configs still point at `/tmp/tb_logs` and
have not been converted.

That attempt is a real failure mode, not a tidiness question. It happens during setup, after the
allocation is already up, and it kills the run: the shared `/projects/a5k/public/logs/tensorboard/`
is owned by one account with no group write, so a run under any other account dies with
`PermissionError: [Errno 13]` about two minutes in. The xl-50b ablation lost a 64-node segment to
exactly this. The older advice to point the directory at `/tmp` avoided the permission problem and
kept every other cost, including the stale-file-handle crashes that come from several runs sharing
NFS event logs.

**Do NOT "disable" it by raising `tensorboard_log_interval` or clearing the `log_*_to_tensorboard`
flags. Those names lie.** `training/utils/train_utils.py` computes each report under its
`log_*_to_tensorboard` flag and then fans the result out to the TensorBoard writer, the **W&B**
writer, MLflow and Comet alike, with the whole block gated on
`iteration % tensorboard_log_interval`. So `tensorboard_log_interval: 999999` does not turn
TensorBoard off — it turns off **W&B metrics**, which is how a run comes back with no throughput,
memory or runtime series and a healthy-looking log. Leave the interval at 1 and the flags at their
values; null the directory and nothing else.

### Launching training from a login node (salloc shell lost)

If the tunnel/salloc shell dies but `SLURM_JOB_ID` is still in `squeue`, export the SLURM env vars manually so `pipeline_training_launch.sh` can attach via `srun --jobid=… --overlap`:

```bash
export SLURM_JOB_ID=<id> SLURM_NNODES=<n> SLURM_NODELIST='<from scontrol show job>'
export SLURM_JOB_NODELIST="$SLURM_NODELIST" SLURM_NTASKS=<n> SLURM_JOB_NUM_NODES=<n> SLURM_NPROCS=<n>
export SLURM_GPUS_PER_NODE=4 SLURM_GPUS_ON_NODE=4   # else torchrun --nproc_per_node is empty
export SLURM_CLUSTER_NAME=gracehopper               # ft_launcher OneLoggerConfig pydantic-rejects None
export SLURM_SUBMIT_HOST=login01
bash pipeline_training_launch.sh <config.yaml> --model super --mode sft
```

Between retries: `pkill -9 -f "pipeline_training_launch"`, `rm` stale `*_train.out` logs, `rm -rf <save_ckpt_dir>` if an empty checkpoint dir was created (orchestrators may read its `latest_checkpointed_iteration.txt` as completion).

## Checkpoint Save Policy

- **Standard SFT and EM fine-tuning**: Set `save_interval: 1000000` to skip intermediate checkpoints. Megatron-Core always saves a final checkpoint when `train_iters` is reached, so this effectively means "save only at end of training."
- **Long CPT runs and reasoning/thinking training**: Use a reasonable `save_interval` (e.g., 100) for fault recovery — these runs take hours/days and losing progress is costly.
- **Rationale**: SFT/EM runs are short (100-500 iters, minutes) and cheap to restart. Intermediate checkpoints waste disk and I/O time. Reasoning/thinking runs are long and need periodic saves for resumption.
- **No intermediate checkpoints ⇒ skip optimizer + RNG state.** When a YAML has `save_interval: 1000000` (i.e., only the final checkpoint is written), set `checkpoint.save_optim: false` and `checkpoint.save_rng: false`. The final ckpt only needs the model weights; downstream consumers (HF conversion, inference, evals) read just `model.*` keys, never the Adam moments or RNG state. Skipping them shrinks the saved torch_dist files materially (~3× for 30B Nano, similar relative for 120B Super) and trims end-of-training I/O without losing anything load-bearing. Runs *with* intermediate `save_interval` (long CPT, reasoning) keep `save_optim/save_rng` at the defaults so they can resume mid-run.
