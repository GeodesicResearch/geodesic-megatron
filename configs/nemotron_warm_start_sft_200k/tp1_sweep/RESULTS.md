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

### Wave 1 — TP=1, selective `core_attn` recompute (the literal Quentin recipe)

| Exp | TP | PP | offload | loaded? | result | where |
|-----|----|----|---------|---------|--------|-------|
| E0 baseline | 4 | 8 | on | yes | warming up (DP=2, slow first-iter @16 nodes) | — |
| E1 | 1 | 4 | on  | yes | **OOM** | first fwd/bwd |
| E2 | 1 | 4 | off | yes | **OOM** | MoE token permute (94.7/95 GB) |
| E3 | 1 | 8 | on  | yes | **OOM** | — |
| E4 | 1 | 8 | off | yes | **OOM** | — |
| E5 | 1 | 2 | on  | no  | **OOM** | before load (44 layers/stage) |

**Every TP=1 config OOMs — PP=2,4,8, with and without offload.** Offloading expert
activations does NOT prevent it (E1≈E2, E3≈E4). Weights fit trivially (~18 GB/GPU);
the constraint is **activations**.

### Wave 2 — TP=1 + MoE recompute `[core_attn, moe, shared_experts]` (Ultra's set)

| Exp | TP | PP | recompute | offload | result | s/iter | TFLOP/s/GPU |
|-----|----|----|-----------|---------|--------|--------|-------------|
| W1 | 1 | 8 | +moe,shared_experts | off | running (canary) | … | … |
| W2 | 1 | 4 | +moe,shared_experts | off | pending | | |
| W3 | 1 | 8 | +moe,shared_experts | on  | pending | | |

## Findings

### Canary (E1, PP=4 offload ON) — validated the path, 2026-06-08
1. **`srun --overlap` into the tunnel allocation works** on step-free nodes (clean GPUs).
2. **TP=1/EP=4/PP=4 reshards and loads from the Base-Chat-Init ckpt cleanly**
   (`successfully loaded checkpoint ... at iteration 0`). Quentin's topology is loadable.
3. **Per-GPU params ≈ 9.3B** (PP-rank sum over one EP slice = 36.1B). Solving
   `X + (120−X)/4 = 36.1` ⇒ **non-expert ≈ 8B, experts ≈ 112B (~93% of the model).**
   → TP only shards the tiny ~8B non-expert part, so **dropping TP 4→1 barely costs
   memory** (~18 GB weights/GPU at PP=4, huge headroom on 95 GB). This **overturns the
   "TP=1 forces PP up" worry** — PP can go *low* (PP=2 plausibly fits). The real memory
   question is now **activations** (what the offload on/off A/B measures), not weights.
4. **`save: null` is incompatible with the W&B logger** (`state.py:197` does
   `os.path.join(checkpoint.save, "wandb")` on load) → use a valid save path + huge
   `save_interval`/`non_persistent_save_interval` + kill before `train_iters` instead.
   Fixed in the generator; relaunched.

### Wave 1 result — TP=1 OOMs everywhere; the mechanism is SP, not weights
- **TP=1 ⇒ `sequence_parallel` must be False ⇒ activations are no longer sharded across
  the (former) 4 TP ranks.** So per-GPU activation memory is **~4× the TP=4+SP baseline.**
  The MoE token-permute intermediate (scales with tokens×topk×hidden, now unsharded) is
  what tips it over — E2's OOM is exactly there.
- **Offloading expert activations does not rescue it** (E1 vs E2 and E3 vs E4 both OOM).
  Offload moves *stored* activations to CPU, but the transient permute buffer still has to
  be materialized on-GPU. So offload is not the lever for TP=1; **recompute is.**
- This is the concrete cost of Quentin's TP=1 idea on this model: you trade TP communication
  for a large activation-memory bill that must be paid back with recompute (a compute tax).
  Whether that nets faster than the TP=4+SP baseline is exactly what wave 2 measures.
- Corroborated by NVIDIA's own Ultra 550B config: it runs TP-low/EP folding with
  `recompute_modules:[core_attn,moe,shared_experts]` and **offload OFF** — with the comment
  "superseded by MoE recompute (can't offload recomputed activations)."

_(wave-2 throughput numbers below once runs hit steady state)_
