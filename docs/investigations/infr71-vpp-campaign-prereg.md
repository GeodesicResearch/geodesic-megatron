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

## Phase A — memory-fit ladder for VPP=2 (the blocker)
A0. VPP4 + cutlass re-baseline (`configs/infr71_vpp/a0_vpp4_cutlass.yaml`): the July VPP4 number
    (30.69) predates cutlass experts; re-measure VPP4 with the champion's expert impl so the VPP
    penalty is isolated from the host-overhead fix it may have been confounded with. Paired with a
    same-wave champion control re-run.
A1. PP16·VPP2·DP1 (m=64 µb/pipe) (`configs/infr71_vpp/a1_pp16vpp2_dp1.yaml`): halves per-stage
    weights vs PP8; interleave residency spread over
    16 stages. te_grouped AND cutlass arms (envelope comparison — cutlass costs activation memory).
A2. PP8·VPP2 + stage-0-unloaded piped pattern (first vstage segment carries fewer MoE layers);
    rebalanced 16-segment pattern from the July fVPP work.
A3. A1/A2 + recompute ladder: [moe,shared_experts] → +moe_act → +core_attn (fit first, then peel).
A4. offload interaction: optimizer_offload_fraction 0.5 vs 1.0 under VPP (host-starvation fixes may
    have changed the offload trade).

Gate per arm: fits (no OOM through iter 5) → then 30-iter timing. Any surprise (>10% off
prediction): capture torch profile (ISAMBARD_TORCH_PROFILE=1, iters 10,20) before iterating.

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
