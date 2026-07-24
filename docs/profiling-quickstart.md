# Profiling Quickstart — 120B Super SFT

A step-by-step walkthrough of profiling the champion 120B quickstart workload
with the torch profiler. The output is the artifact bundle our external
speed-up assessment expects (per-rank Chrome traces with `with_stack=True` +
`record_shapes=True`, plus everything needed to reproduce the run), and the
first tool to reach for when iteration time regresses.

Related: [container-pipeline.md](container-pipeline.md) ("Profiling a training
run" and "Run identity") for the reference documentation;
`3rdparty/torch-profiling-tutorial` (Quentin Anthony's tutorial) for how to
read torch profiles in general.

## What you get

One 25-iteration run of the champion benchmark config
(`configs/quickstart/nemotron_super_quickstart_sft.yaml`, 64 GPUs / 16 nodes)
with full-step traces captured at **iterations 10 and 20** — steady state, past
the JIT/comm-init window. Two capture points let you check that the profiled
step is representative (e.g. no one-off straggler like the 458 ms gap seen in
one earlier capture) and bracket periodic effects such as the manual-GC cycle.

## 1. Launch

Container (the default, certified environment):

```bash
cd /home/a5k/kyleobrien.a5k/geodesic-megatron
ISAMBARD_TORCH_PROFILE=1 ISAMBARD_TORCH_PROFILE_ITERS=10,20 \
    ISAMBARD_TORCH_PROFILE_RANKS=0,9 \
    isambard_sbatch --nodes=16 pipeline_training_submit.sbatch \
    configs/quickstart/nemotron_super_quickstart_sft_profile.yaml super sft
```

Bare-metal venv (A/B datum; the fusion override is MANDATORY — the venv has no
APEX):

```bash
GEODESIC_CONTAINER=0 ISAMBARD_TORCH_PROFILE=1 ISAMBARD_TORCH_PROFILE_ITERS=10,20 \
    ISAMBARD_TORCH_PROFILE_RANKS=0,9 \
    isambard_sbatch --nodes=16 pipeline_training_submit.sbatch \
    configs/quickstart/nemotron_super_quickstart_sft_profile.yaml super sft \
    model.gradient_accumulation_fusion=False
```

Why ranks 0 and 9: rank 0 is pipeline stage 0 (embeddings, loss, the 1F1B
endpoint where the pipeline bubble is most visible); rank 9 is an interior
MoE-heavy stage. Two structurally different views of the same step.

## 2. Watch the run

```bash
tail -f logs/slurm/train-<jobid>.out
```

Early in the log, the launch banner prints the run identity:

```
===== Nemotron 3 Training =====
Job ID:    5738450
Run ID:    20260724T183000-j5738450
Raw log:   .../logs/slurm/train-5738450.out
...
```

and the profiled ranks announce their plan and, later, each export:

```
[torch-profile] rank 0: tracing iteration(s) [10, 20] (1-based; ...) -> /projects/a5k/public/profiles/nemotron_super_quickstart_sft_profile/20260724T183000-j5738450
[torch-profile] rank 0: trace written -> .../rank0.iter10.chrome_trace.json.gz
```

A 25-iteration run at ~28 s/iter finishes in ~15 minutes plus startup.

## 3. Collect the artifacts

Everything lands in
`/projects/a5k/public/profiles/nemotron_super_quickstart_sft_profile/<run-id>/`:

| file | what it is |
|---|---|
| `rank0.iter10.chrome_trace.json.gz` (+ iter20, + rank9 variants) | Kineto Chrome traces, one per profiled rank per capture (~175 MB gz, ~2 GB raw each) |
| `provenance.txt` | exact commit, run id, raw-log path, world size, torch/CUDA versions, capture iterations |
| `config_snapshot.yaml` | the override YAML exactly as passed |
| `resolved_config_snapshot.yaml` | the FULL merged config (recipe defaults + YAML + CLI overrides) — **use this to reproduce**, the override file alone is not sufficient |
| `raw_log_snapshot.out` | copy of the job log (refreshed at each export and train end) |

When sharing externally (e.g. for the speed-up assessment), send the whole
directory — the traces are only interpretable together with the config and
commit.

## 4. Open a trace

Load `rank<R>.iter<N>.chrome_trace.json.gz` into [Perfetto](https://ui.perfetto.dev)
(use "Open trace file"; the UI handles gzip) or `chrome://tracing`. The traces
are ~10 M events; Perfetto needs a beefy machine or its `trace_processor`.

Reading caveats (quantified in the earlier champion analysis):

- GPU kernel durations are CUPTI hardware timestamps — trustworthy.
- `with_stack` inflates CPU-side launch time; it shows up as extra inter-kernel
  idle (~+18% wall on this workload). Treat compute-kernel time as hard, idle
  gaps as an upper bound.
- Find the captured step by searching for the `ProfilerStep#` annotation.

## 5. Analyze

For the systematic breakdown (per-category interval-union accounting, exposed
vs overlapped collectives, idle-gap histograms), use the analysis scripts and
method documented in
`/projects/a5k/public/profiles/nemotron_super_quickstart_sft_container_profile/ANALYSIS_FULL.md`
(§1 "Methodology", §9 "Appendix — files & reproduction"). Each trace needs
~15-20 GB RAM to parse.

## 6. Join it with W&B and the logs

The run ID stitches everything together:

- W&B run summary carries `run/isambard_run_id`, `run/raw_log_path`, and
  `run/slurm_job_id` (project `megatron_training`, experiment
  `nemotron_super_quickstart_sft_profile`).
- Given a run ID, the log is `logs/slurm/by-run-id/<run-id>.out` and the
  profile directory is `<profile-root>/<wandb-exp-name>/<run-id>/`.
- Given a profile directory, `provenance.txt` names the commit, config, and
  raw log.

## Knob reference

| env var | default | meaning |
|---|---|---|
| `ISAMBARD_TORCH_PROFILE` | off (`0`) | `1` = profile to the default root `/projects/a5k/public/profiles`; a path = profile to that root |
| `ISAMBARD_TORCH_PROFILE_ITERS` | unset | comma-separated 1-based iterations to capture; one `rank<R>.iter<N>` trace per capture |
| `ISAMBARD_TORCH_PROFILE_RANKS` | `0` | comma-separated global ranks to trace |
| `ISAMBARD_TORCH_PROFILE_WAIT` | `3` | legacy mode (only when `_ITERS` unset): single capture at iteration WAIT+2, unsuffixed filename |
| `ISAMBARD_RUN_ID` | minted by launcher | override to pin the run ID (rarely needed) |

Profiling a different workload: any config works — add the same env toggles to
its usual launch command. Keep `train_iters` comfortably above your last
capture iteration (the export needs one extra step after each capture).
