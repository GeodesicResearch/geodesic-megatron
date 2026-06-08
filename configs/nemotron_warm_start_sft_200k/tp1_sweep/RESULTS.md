# TP=1 / EP=4 / PP=min / ZeRO-1 throughput sweep — Super 120B-A12B

**Goal (from Kyle, relaying Quentin Anthony's review):** empirically test Quentin's
recommended parallelism — **TP=1, EP=4, PP=min-that-fits, ZeRO-1 (distributed
optimizer)** — for the 120B warm-start SFT, **with and without CPU activation
offloading**, and compare against the current baseline (TP=4/EP=4/PP=8).

## Method

- Iterate on `nemotron_120b_warm_start_sft_200k_instruct.yaml`. Hold **everything**
  constant (data, Base-Chat-Init ckpt, LR, GBS=64, seq=8192, selective core_attn
  recompute, ZeRO-1, precision-aware optimizer) except the independent variables:
  **TP, PP, sequence_parallel (forced False when TP=1), activation offload.**
- Full factorial: **PP ∈ {2,4,8} × offload ∈ {on,off}**, TP=1/EP=4/ETP=1/DP=4 fixed,
  plus **E0 baseline** (TP=4/EP=4/PP=8, the instruct config) as a same-hardware ref.
- `save: null` (disk is ~94% full — zero checkpoint writes). Short runs (`train_iters:60`,
  `exit_signal_handler:false`); each killed after ~25 steady iters once iter-time settles.
- Run **node-exclusive on step-free nodes inside the running code-tunnel allocation
  (job 5071212)** via `srun --overlap` — clean GPUs, no quota increase, no queue wait.
- **Metric:** steady-state s/iter (direct A/B within a PP), and **MFU / TFLOP-s-per-GPU**
  (`log_throughput:true` emits it) to compare across different GPU counts.

### Parallelism arithmetic
88 layers ⇒ PP must divide 88 (PP ∈ {2,4,8,11,...}). TP=1,EP=4,ETP=1,CP=1 ⇒ DP must be
a multiple of 4 (DP = 4·EDP). All sweep runs use **DP=4, EDP=1** ⇒ World = 4·PP, and
GBS=64/DP=4 = 16 grad-accum microbatches (so PP=8 pays a larger 1F1B bubble than PP=4).
**Note the tension in Quentin's recipe:** dropping TP 4→1 *un-shards* attention/dense →
*more* memory/GPU → forces PP *up*, not down. So "PP=min that fits" under TP=1 is
expected to be ≥ the baseline PP=8, not below it. This sweep measures whether removing
TP comms still nets a win despite that.

## Node map (clean pool nid010923–010972, all GPU-idle, step-free)

| Exp | Config | TP | PP | offload | nodes | GPUs | nodelist | port |
|-----|--------|----|----|---------|-------|------|----------|------|
| E0 | e0_baseline_tp4_ep4_pp8  | 4 | 8 | on  | 16 | 64 | nid[010923-010938] | 29720 |
| E1 | e1_tp1_ep4_pp4_offload   | 1 | 4 | on  | 4  | 16 | nid[010939-010942] | 29721 |
| E2 | e2_tp1_ep4_pp4_nooffload | 1 | 4 | off | 4  | 16 | nid[010943-010946] | 29722 |
| E3 | e3_tp1_ep4_pp8_offload   | 1 | 8 | on  | 8  | 32 | nid[010947-010954] | 29723 |
| E4 | e4_tp1_ep4_pp8_nooffload | 1 | 8 | off | 8  | 32 | nid[010955-010962] | 29724 |
| E5 | e5_tp1_ep4_pp2_offload   | 1 | 2 | on  | 2  | 8  | nid[010963-010964] | 29725 |

## Status / Results (live)

| Exp | launched | loaded? | first-iter | steady s/iter | TFLOP/s/GPU | peak mem | fits? | notes |
|-----|----------|---------|-----------|---------------|-------------|---------|-------|-------|
| E1 | yes (smoke) | … | … | … | … | … | … | warming up |
| E0 | pending | | | | | | | |
| E2 | pending | | | | | | | |
| E3 | pending | | | | | | | |
| E4 | pending | | | | | | | |
| E5 | pending | | | | | | | |

## Findings

_(to be filled in)_
