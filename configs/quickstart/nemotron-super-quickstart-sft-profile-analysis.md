# Torch-profiler analysis — final champion quickstart config (INFR-68 close-out)

**Subject:** `configs/quickstart/nemotron_super_quickstart_sft.yaml` as committed at the
INFR-68 freeze (duo recompute + `gradient_accumulation_fusion` + `manual_gc`), container
execution, 16 nodes / 64 GH200, seq 32K, GBS 64, steady-state ≈ **27.5–27.6 s/iter**
(mean of iters 6–12, FT off; ~30.8 s in production trim with FT heartbeats).

**Raw data (too large for git — 167 MB gz per rank, over GitHub's 100 MB cap):**
`/projects/a5k/public/profiles/quickstart_final_champion/quickstart_final_champion_profile/`
— `rank{0,9}.chrome_trace.json.gz` (`with_stack=True`, `record_shapes=True`, iteration 5 of a
7-iter probe, wait=3/warmup=1/active=1), plus `provenance.txt` (commit, world info) and
`config_snapshot.yaml`. Collected 2026-07-24. W&B: `megatron_training` /
`quickstart_final_champion_profile`.

## Headline accounting (rank 0; traced window 32.3 s incl. profiler inflation)

| Bucket | Time | Share |
|---|---|---|
| Compute (union) | 9.41 s | ~29% |
| NCCL (union) | 8.46 s | ~26% — **8.31 s exposed** (overlap 0.15 s) |
| Idle (gaps) | 14.5 s | ~45% (inflated by `with_stack` launch overhead; real idle materially lower) |

Rank 9 mirrors it (compute 9.20 s, exposed NCCL 7.52 s, idle 15.5 s).

## Exposed communication breakdown

| Collective | r0 exposed | r9 exposed | Note |
|---|---|---|---|
| sendrecv (PP p2p) | 6.04 s | 3.81 s | **B1 — the dominant residual.** 1F1B activation exchange; `overlap_p2p_comm` is measured-dead on Slingshot (un-batched isend/irecv is 2.3–5.7× slower per op — see the VPP investigation doc) |
| allreduce | 1.66 s | 3.05 s | fp32 grad/main-grad + grad-norm allreduces (17 large RING_LL calls, 97–135 ms each) — the main remaining *tunable* lever (bucket/overlap tuning) |
| allgather | 0.40 s | 0.66 s | optimizer param gather |
| reduce-scatter | 0.22 s | 0.33 s | mostly overlapped (`overlap_grad_reduce`) |

## Compute profile

GEMM 4.8 s union (168k launches, nvjet sm90 — efficient), Mamba scans 1.8 s (5,600
launches, 14 kernels — TP1-by-design, memory-bound), MoE routing 1.13 s, elementwise
1.06 s, attention 0.14 s.

## What the campaign changed (vs the pre-campaign trace of the same workload)

- **Grad-accumulate add storm eliminated**: the fp32 `CUDAFunctor_add` that dominated
  launch count at ~43,000/iter is down to ~1,600/iter (`gradient_accumulation_fusion` —
  wgrad now accumulates in the GEMM epilogue). This was the single largest launch source.
- **GC pauses eliminated**: `manual_gc` (interval 10) removed the multi-second
  inter-iteration jitter; non-GC iterations sit in a ~0.2 s band (was ~2 s).
- Net: 31.27 → **27.6 s/iter** measured on the committed config
  (fusion −1.09 s, manual GC −0.27 s, plus variance collapse), 82.8 → ~95 TFLOP/s/GPU
  reported at probe conditions.

## Bottleneck status at close-out

| Bottleneck | Status |
|---|---|
| B1 exposed PP p2p (~6 s) | Structural on this fabric: VPP works (see VPP doc) but is net-slower; `overlap_p2p_comm` regresses ~2× (transport-level, trace-proven). Revisit only on fabric/transport change. |
| B2 launch storm / GC | **Fixed** (fusion + manual_gc, composed into this config). CUDA graphs measured OOM (+~14 GB pools) — blocked at this memory point. |
| B3 DP collectives | RS overlapped; the fp32 allreduce block (1.7–3.1 s) is the top remaining tunable. SE-overlap hard-blocked (LatentMoE). |
| B4 Mamba/elementwise fusion | Router fusion measured no-win; deeper Triton fusion = follow-up ticket. |

**Honest ceiling:** with B1 structural and compute at ~9.4 s, the practical floor for this
workload on this fabric sits in the mid-20s s/iter; the committed 27.6 is within ~10–15% of it.

## Iteration-15 cross-check (2026-07-24, full normal run, FT on)

A second capture at **iteration 15** of a full 48-iter production run (commit
d94153-era tooling; traces + full reproduction bundle:
`/projects/a5k/public/profiles/quickstart_normal_run/nemotron_super_quickstart_sft/20260724T203753-j5738449/`)
confirms the compute fingerprint exactly (GEMM 4.78 s, Mamba 1.83 s/5,600
launches, fused adds ~1.6k) and **corrects the communication picture**:

- **The fp32 allreduce block is mostly an early-run effect, and stage-dependent**:
  exposed allreduce falls 1.66 -> 0.04 s on rank 0 and 3.05 -> 1.61 s on rank 9
  between iterations 5 and 15. The residual ~1.6 s sits at the optimizer
  boundary on stage 1 (two ~0.5 s u32 calls + seven f32 calls), so the "main
  remaining tunable lever" claim above is halved and applies per-stage, not
  globally.
- Steady-state exposed comm is therefore **~5.9 s, ≈90% PP p2p sendrecv** —
  B1 even more dominant than the iteration-5 numbers suggested.
- Run context: gate-window mean 30.39 s/iter @ 87 TFLOP/s with fault tolerance
  ON (the production path; FT heartbeats cost ~1–2 s vs the 27.6 FT-off probe
  number). Traced-iteration overhead 166 s.

---

## Addendum 2 (2026-07-25): acting on the findings — the perf-iteration ladder

Every issue in this analysis was probed with a lever on tunnel 5738449 (a ~2.7 s/iter
slower Dragonfly placement than the champion tunnel — all deltas below are vs the
placement-matched P0 anchor; 14-iter probes, FT off, mean of iters 6–12; W&B
`megatron_training/qs_perfiter_*`):

| Probe | Lever | s/iter | vs P0 | Verdict |
|---|---|---|---|---|
| P0 | none (committed config pre-ladder) | 30.30 | — | anchor |
| P1 | `ISAMBARD_CUDA_MAX_CONNECTIONS=8` (TP=1) | 27.51 | −9.2% | works; redundant with P3 |
| P2 | `NCCL_P2P_NET_CHUNKSIZE=1M` + `NCCL_BUFFSIZE=16M` | 30.78 | +1.6% | no win on CXI |
| P3 | `log_num_zeros_in_grad`+`log_params_norm` off | 27.20 | −10.2% | **adopted into the quickstart** |
| P5 | P1 + P3 composed | 27.65 | −8.7% | no additive gain — same slack |
| P4/P6/P7 | + `optimizer_offload_fraction: 0.5` | gate 26.30, steady ~25.7 | ~−15% | opt-in only: peak ~91 GB (~4 GB headroom), warm-up hump iters ~7–13 |

**Resolution of the traced issues:**

- **u32 optimizer-boundary allreduces + launch-gap storm → one root cause.**
  `count_zeros_fp32` (gated on `log_num_zeros_in_grad`) launches a `count_nonzero`
  kernel per parameter (thousands of tiny launches/iter — the 50–200 µs gap mass),
  all-reduces the count over the **entire world** (the traced u32 AllReduce), then
  calls `.item()` — a blocking device sync, every iteration. `log_params_norm` adds
  the same pattern. Turning both off is worth ~3 s/iter and is config-only.
- **P1↔P3 redundancy explained:** `CUDA_DEVICE_MAX_CONNECTIONS=8` lets the telemetry
  kernel-storm overlap other work; removing the telemetry removes the storm. Both
  attack the same serialized slack, so they don't compose. The `=1` default remains
  (required for TP/SP overlap at TP>1); the override seam is
  `ISAMBARD_CUDA_MAX_CONNECTIONS` in both activate scripts.
- **Exposed PP send/recv (B1):** NCCL p2p chunk/buffer sizing is measured neutral on
  Slingshot/CXI — B1 stays structural (documented block; `overlap_p2p_comm` remains a
  ~2× regression per the VPP investigation).
- **~1.49 s aligned optimizer-boundary gap:** partly the CPU-offloaded optimizer step.
  `optimizer_offload_fraction: 0.5` recovers ~1.5 s/iter (30-iter gate 26.30, steady
  ~25.7, exact loss parity at iter 30: 0.60891 vs the A/B's 0.6089) but peaks ~91 GB
  of 95 — too little headroom for the certification config, so it ships documented-off.
- **Pipeline bubble (~18%):** structural at GBS=64/PP=8/DP=2; VPP measured worse
  (see the VPP investigation). No further lever inside the fixed workload.

Post-ladder quickstart (telemetry-off adopted): expect ≈ 27.2 s/iter on this
placement ≈ **24–25 s/iter on champion-tunnel placement**, ~97 TFLOP/s/GPU, with the
same memory profile as before.
