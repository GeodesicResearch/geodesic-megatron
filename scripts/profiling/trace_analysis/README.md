# Trace analysis for GPU-idle investigations

Written for the Super-120B GBS=64 host-overhead investigation
(`docs/investigations/120b-gbs64-host-overhead-investigation.md`). They operate on the
chrome traces the profiler callback writes to
`/projects/a5k/public/profiles/<wandb-exp-name>/<run-id>/rank<N>.iter<M>.chrome_trace.json.gz`.

| script | question it answers |
|---|---|
| `analyze_full.py <trace.gz> <label>` | Where does the wall-clock go? Interval-union of compute vs NCCL vs idle, per-category kernel totals, and **how much NCCL is actually overlapped** with compute. |
| `gap_dist.py <trace.gz> <lo_us> <hi_us>` | Is the idle a pipeline bubble or launch starvation? Histogram of GPU idle-gap sizes. A bubble = a few large gaps; host-boundedness = many small ones. |
| `gap_attrib.py <trace.gz> <lo_us> <hi_us>` | What is the host doing during the idle? Aggregates CPU-side events overlapping each gap in a size band. |
| `host_attrib.py <trace.gz> <label>` | WHO is responsible (needs `with_stack=True`, the profiler's default; not a capture taken with `ISAMBARD_TORCH_PROFILE_WITH_STACK=0`): kernel-family launch census, python-stack attribution of every stream/device sync and pageable-HtoD copy, per-thread CUDA-API saturation. This is the tool that identified the per-expert-GEMM launch storm and the timer-barrier "syncs" (investigation doc §9). |

Typical sequence:

```bash
D=/projects/a5k/public/profiles/<exp>/<run-id>
python analyze_full.py $D/rank9.iter10.chrome_trace.json.gz baseline   # totals + overlap
python gap_dist.py     $D/rank9.iter10.chrome_trace.json.gz 0 1000000  # gap shape
python gap_attrib.py   $D/rank9.iter10.chrome_trace.json.gz 1000 10000 # attribute the big band
```

Stack walking inflates host launch time, which shows up as extra GPU idle between kernels, so
the idle measured by `analyze_full.py`, `gap_dist.py` and `gap_attrib.py` on a stack capture is an
upper bound. For a timing breakdown, take a second capture with
`ISAMBARD_TORCH_PROFILE_WITH_STACK=0` and keep the stack capture for `host_attrib.py`.

Run them on an idle allocation: parsing a ~190 MB trace is CPU-heavy and will perturb a
concurrently-timed training arm.
