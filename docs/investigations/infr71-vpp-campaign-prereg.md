# INFR-71 VPP campaign — preregistered experiment ladder (draft 2026-07-30)

Branch: `feat/infr71-vpp-pp-overlap` + merge of updated main (cutlass experts, 26.04, offload 0.5).
Baseline to beat: **21.78 s/iter** (non-VPP champion, TP1·CP4·EP4·PP8·DP2, 64 GPU/16 nodes, 32K, GBS 64).
Goal: VPP functioning end-to-end AND < 20 s/iter. All runs: `isambard_sbatch --nodes=16
pipeline_training_submit.sbatch <cfg> super sft`, FT off, save null + wandb_save_dir, mean of iters
10–30, engagement-marker gates (rank-0 log lines) before any number is trusted.

Known state (from the July ladder):
- fVPP mechanically works via piped hybrid_layer_pattern ('|' separators); VPP=4 landed 30.69 vs
  27.61 champion at cert config (+11%) pre-cutlass; VPP=2 interleave residency OOM'd at 94.7 GB on
  the cutlass activation envelope (plain AND [core_attn,moe,shared_experts] recompute).
- overlap_p2p_comm: the iteration-2 NaN was an upstream deallocate-race bug, fixed in the current
  0.19 pin (investigation doc §UPDATE) — so the lever is mechanically usable now, but with the fix
  applied it measured **31.45 s/iter vs 27.50 baseline (+14%)** at PP8·VPP4: un-batched isend/irecv
  is 2.3–5.7× per-op on CXI. Do NOT re-enable without a trace showing exposed p2p is the residual.
- overlap_moe_expert_parallel_comm requires build_schedule_plan (GPT-only; absent on HybridModel).
- MoE-paper guidance: interleaved VPP shrinks bubble ∝ 1/v; comm volume up ∝ v; hybrid needs
  balanced per-vstage layer mixes (MoE stages ~2× heavier).
- 0.19-pin changes discovered at wave-2 launch (2026-08-02): (a) offloading
  `expert_fc1`/`moe_act` while `moe` is in recompute_modules now ASSERTS (was silently
  redundant; trimmed from the Phase-A arm configs — memory-neutral, and in fact moot
  once (b) forces `fine_grained_activation_offloading: false`, since the assert and the
  whole offload path are gated on that flag at `transformer_config.py:1829`. Of the Q3 arms
  below, q3a and q3b deliberately keep the full list — q3a because it is inert there too,
  q3b because offloading `expert_fc1`/`moe_act` IS its lever and is legal once `moe` is not
  recomputed; q3c carries the trimmed list to match a0); (b)
  `fine_grained_activation_offloading` is INCOMPATIBLE with VPP: interleaved model chunks
  each build a ChunkOffloadHandler and backward crosses them → deterministic
  `AssertionError: Chunk mismatch` at iteration 2 on MoE-stage ranks. **VPP is where wave 2
  met this, not its boundary** — the merged host-overhead campaign already recorded the same
  assert firing under **plain PP=8** at this pin and OPEN upstream
  (`120b-gbs64-host-overhead-investigation.md:104`, bug #5 at `:151`). Treat fine-grained
  activation offloading as broken on this model generally, not as a VPP-only exclusion. VPP
  arms therefore
  run with fine-grained offload OFF; the July fit recipe's offload lever is unavailable
  under VPP until fixed upstream — Phase A leans on recompute + pattern rebalancing (+
  PP16·DP1 in A1) alone.

## Phase A — memory-fit ladder for VPP=2 (the blocker)
A0. VPP4 + cutlass re-baseline (`configs/infr71_vpp/a0_vpp4_cutlass.yaml`): the July VPP4 number
    (30.69) predates cutlass experts; re-measure VPP4 with the champion's expert impl so the VPP
    penalty is isolated from the host-overhead fix it may have been confounded with. Paired with a
    same-wave champion control re-run.
A1. PP16·VPP2·DP1 (m=64 µb/pipe) (`configs/infr71_vpp/a1_pp16vpp2_dp1.yaml`): halves per-stage
    weights vs PP8; interleave residency spread over
    16 stages. te_grouped AND cutlass arms (envelope comparison — cutlass costs activation memory).
A2. PP8·VPP2 + stage-0-lite piped pattern (`configs/infr71_vpp/a2_pp8vpp2_stage0lite.yaml`):
    same 88-layer sequence repartitioned so stage 0 carries 10 layers (4 MoE) instead of the
    July split's 12 — per-stage totals [10,11,11,12,12,11,11,10] — targeting the ~5% peak-memory
    shave the July arms missed fit by (94.7 GB OOM).
A3. A1/A2 + recompute ladder: [moe,shared_experts] → +moe_act → +core_attn (fit first, then peel).
A4. offload interaction: optimizer_offload_fraction 0.5 vs 1.0 under VPP (host-starvation fixes may
    have changed the offload trade).

Gate per arm: fits (no OOM through iter 5) → then 30-iter timing. Any surprise (>10% off
prediction): capture torch profile (ISAMBARD_TORCH_PROFILE=1, iters 10,20) before iterating.

**Wave-2 results (2026-08-02, in-tunnel 2-way pairs on disjoint 16-node halves; the paired
champion control measured 23.40 s/iter vs 21.78 solo = +7.4% concurrency tax; all arms ran
`fine_grained_activation_offloading=false` (VPP-incompatible at this pin) and carry the
measured ~0.95 s/iter offload-fraction-1.0 handicap, subtracted in "adjusted"). **The
adjusted figures are UPPER BOUNDS on the VPP penalty**: §C1 bounds that handicap only from
below ("at least 0.9–1.0 s/iter … plausibly more") and the same-nodelist 1.0-vs-0.5 ladder in
the shipped quickstart header puts it at 1.7–2.9 s. Subtract 2.0 s instead of 0.95 and A0
becomes +9.0%, A2 +0.9% — i.e. A2 is plausibly a wash. The arms also carry trio recompute
against the champion's duo (~0.13 s), which is NOT subtracted. What this campaign establishes
is that VPP did not become a win; it does not establish that the penalty grew:**
- **A0 (VPP4+cutlass): 27.51 mean-10-30, loss parity → +13.5% adjusted. The July VPP4
  penalty did NOT come from the per-expert launch storm — cutlass didn't move it.
  Hypothesis "VPP penalty = host starvation" is refuted; the cost is schedule/comm-side.**
- **A2 (PP8·VPP2 stage-0-lite): FITS — the repartition solved the July 94.7 GB OOM
  (and did so with fine-grained offloading OFF, unlike the July arms; since removing
  offloading can only raise peak memory, that makes the repartition's contribution a
  lower bound).
  25.61 mean-10-30, loss parity → +5.4% adjusted. Interleave bubble savings (−1.9 s
  predicted) are overwhelmed by ~3 s of interleave overhead at 32 µb/replica, refining
  (not contradicting) the ≤16-µb/replica crossover rule.**
- **A1 (PP16·VPP2·DP1): NO RESULT — spent 76 minutes in first-iteration NCCL comm-init
  without reaching iteration 1 and was killed.** GPUs sat at 100% utilisation throughout,
  which is NCCL busy-spin, not compute. PP16 doubles the pipeline depth of an already
  deep-PP config, and deep-PP first-iteration lazy comm-init is the known cost here (the
  Ultra-550B section documents 45–75 min at PP36/288 ranks). Recorded as a cost datum, not
  a timing datum: at PP16 the arm never became measurable inside a tunnel-sized window.
- Phase-B note: B2's GBS-128 result (below) closed the remaining live branch.

## Campaign verdict (2026-08-03) — VPP is refuted on this model at this benchmark

Every interleaved arm that produced a number was slower than its non-interleaved control,
and the two mechanisms behind that are now measured rather than assumed:

| arm | µb/replica | result vs matched control | adjusted |
|---|---|---|---|
| A0 — VPP4, cutlass experts | 32 | 27.51 vs paired champion control | **+13.5%** |
| A2 — VPP2, PP8 stage-0-lite | 32 | 25.61 vs paired champion control | **+5.4%** |
| A1 — VPP2, PP16, DP1 | 64 | no iteration in 76 min (comm-init) | n/a |
| (earlier INFR-71) VPP4 @ GBS 32 | 16 | 17.61 vs 17.98 | −2.1% |
| (earlier INFR-71) VPP4 @ GBS 64 | 32 | 29.59 vs 27.50 | +7.6% |

1. **The interleave penalty is stall, not compute.** Trace decomposition on the clean rank
   attributes the whole regression to waiting: compute +0.045 s, exposed comm −1.820 s,
   idle +3.624 s. VPP does subdivide the PP exchanges (4.23× more p2p kernels on stage 0,
   each 1.7–2.4× cheaper), but a PP p2p kernel is dominated by waiting for its peer, not by
   wire time — so splitting one wait four ways does not quarter it.
2. **Host starvation is NOT the cause** — the cutlass A/B (A0 re-measured with the champion's
   expert implementation) left the VPP penalty intact, which refutes the per-expert
   launch-storm hypothesis the July arms were confounded with.

**The bubble is real and worth reclaiming — but batch size, not interleaving, is the lever
that reclaims it here.** At the champion topology (PP8, DP2), GBS 64 gives 32 µb/replica and
a 17.9% bubble; GBS 128 gives 64 µb and 9.9%. Measured, non-VPP, offload-off:

| arm | allocation / placement | s/iter (10–30) | **ms/sample** |
|---|---|---|---|
| champion @ GBS 64 | 5845744 | 20.72 | 323.8 |
| B2 @ GBS 128 | 5845744, same nodelist | 37.54 | **293.3 (−9.4%)** |
| L1 @ GBS 128 | 5845741, half-A | 38.33 | **299.5** |

L1 reproduces B2's reclaim on a different allocation (2.1% apart), so the effect is a
property of the schedule and not of one placement.

**Not run, and why:** VPP2 at GBS 128 (the "Q_b" cell). Its outcome is determined by the
arithmetic already measured — going from 32 to 64 µb/replica *halves* the bubble that VPP
exists to remove (interleaved savings fall from ~1.9 s to ~0.95 s) while the interleave
overhead scales with the number of chunk boundaries, i.e. upward. A2 already lost by ~3 s
of overhead against ~1.9 s of savings at 32 µb; at 64 µb the same arm can only lose by
more. Recorded as a prediction rather than spending 35 minutes of a 32-node tunnel to
confirm a sign that both the measurement and the model agree on.

**Standing recommendation** (already in CLAUDE.md): enable VPP only at ≤16 microbatches per
replica. At the shipped 64-GPU benchmark the non-VPP config wins, and the way to spend a
larger batch is on the batch itself.

## Phase Q3 — recompute arms (consultant item C4)

Three arms testing "flash attention already recomputes internally, so checkpointing attention
double-forwards; disable recompute globally and audit". Each arm's config header carries the
code audit and the measured trace evidence behind its prediction; the consolidated write-up is
§C4 of the external-review tracker (`consultant-training-stack-review.md`, landing separately).
Non-VPP arms use the champion topology and are scored against the champion control; the VPP arm
is scored against A0.

Q3a. recompute OFF, fit probe (`configs/infr71_vpp/q3a_recompute_off.yaml`): champion topology,
    `recompute_granularity: null`, offload fraction 1.0 for maximum headroom. **Preregistered to
    OOM in the first forward.** At 8192 tok/rank the top-22-of-512 dispatch materialises ~2.3 GB
    per MoE layer-microbatch and 1F1B at PP=8 holds 40 of them ⇒ ~90 GB of new activation on a
    95 GB card. The July "recompute OFF → OOM" datum was only a ≥24 GB lower bound (it OOM'd from
    a 70.5 GB baseline and could not report the overshoot); cutlass moved host time, not
    activation residency. Cheap precisely because it dies in iteration 1.
Q3b. recompute OFF + host offload instead (`configs/infr71_vpp/q3b_recompute_off_offload.yaml`):
    adds `fine_grained_activation_offloading: true` with `expert_fc1`/`moe_act`. Illegal while
    `moe` is recomputed (transformer_config.py:1855-1865), so the two mechanisms have never been
    compared head to head; and newly testable at all, since fine-grained offloading was a silent
    no-op on Nemotron-H at the old pin (hybrid_model.py:351-360 wires it now). **Expected to hit
    an open upstream bug, not to produce a memory datum**: `fine_grained_activation_offloading`
    is recorded broken at this pin under **plain PP=8, not only VPP** — bug #5 in
    `120b-gbs64-host-overhead-investigation.md:151`, `AssertionError: Chunk mismatch`, OPEN
    upstream. So discovery (b) above understates the scope; VPP is where wave 2 met it, not its
    boundary. Run anyway because it is cheap and because a reproduction under
    `recompute_granularity: null` (a configuration bug #5 was not observed in, since
    offload-inside-recomputed-MoE is rejected at config time) is worth having on the upstream
    issue. The mechanism question — offload instead of recompute — stays open until it is fixed.
Q3c. A0 minus `core_attn` (`configs/infr71_vpp/q3c_vpp4_no_core_attn.yaml`): scores the measured
    +0.036 s/attn-layer core_attn cost against a placement-matched A0. Traces already show the
    double-forward directly (flash fprop 128 → 256 kernels per attention layer, bprop unchanged at
    128) buying ~9.5 MB per layer-microbatch. A0's stage-0-lite pattern puts 3 attention layers on
    stage 0, the largest signal in the campaign. Differs from A0 in `recompute_modules` only;
    `fine_grained_activation_offloading: false` is A0's own discovery-(b) correction, which
    a0/a1/a2 now carry in-file rather than as a CLI override.

Gate: Q3a/Q3b are fit probes first (OOM or not through iter 5), timing second. Q3c is a timing
arm only, predicted 0.05–0.15 s/iter faster than A0 with peak memory within ~0.2 GB; a delta
outside ±0.3 s means something other than the recompute moved — trace before concluding.

## Phase B — bubble arithmetic confirmation
B1. For each fitting arm, verify measured bubble ≈ (PP/v−1)/(m+PP/v−1) via per-iter fwd/bwd timer
    decomposition; mismatch ⇒ trace (host gaps vs comm gaps attribution via host_attrib.py).
B2. GBS sensitivity: GBS 64 vs 128 at the best VPP arm (m doubles, bubble halves — but wall-clock
    per sample is what counts vs the non-VPP champion at the same GBS).

## Phase C — comms under VPP
C1. batch_p2p_comm ON (default) vs overlap_p2p retest ONLY IF a fitting VPP arm exists and its
    trace shows p2p exposed >5% of iter — expectation: still a regression on CXI; preregistered
    accept criterion: any arm beating champion by >3% with overlap on.
C2. (stretch) build_schedule_plan for HybridModel to unlock overlap_moe_expert_parallel_comm —
    scoped implementation on the branch; only if A/B show bubble is the dominant residual.

## Phase D — image axis
D1. Best VPP arm on 26.02 vs 26.04 (TE 2.14.1 in both; torch differs) — only if a fitting arm is
    within 5% of target, to rule image-level p2p behavior in/out.

Decision rule: first arm < 20 s/iter with loss parity (|Δ| < 2e-3 at iter 30 vs champion) wins and
gets promoted; if ALL Phase A arms OOM, the verdict is "VPP=2 does not fit the 32K/CP4 envelope at
16 nodes on 95 GB" with the measured peak-memory table as evidence, and Phase B/C pivot to
PP16-non-VPP + comm levers from first principles (trace-driven).
