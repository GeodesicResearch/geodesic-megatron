# Part 2e — No-think closure: the campaign's primary explanandum was dose too

Part 2d resolved the **think** pair. The campaign's *stated* primary target was the **no-think** pair
(G vs H) — the largest expected MSM marginal effect, and the channel where the monitorability claim
(binding without visible CoT) lives. This closes it: one no-think AFT branched from the existing
lr-1e-5 MT checkpoint, eval'd against the shipped no-think arms on the same vendored exfiltration
factorial (Sonnet grader, `model_name=Norm`, n=100/cell).

## Result — harmful exfiltration rate [95% CI], n=100/cell

| arm (no-think) | prod=false | prod=true |
|---|---|---|
| AFT-only (H) | 0.380 [0.291, 0.478] | 0.090 [0.048, 0.162] |
| MSM+AFT lr-1e-6 (shipped G) | 0.470 [0.375, 0.567] | 0.160 [0.101, 0.244] |
| **MSM+AFT lr-1e-5 (G_lr1e5)** | **0.150 [0.093, 0.233]** | 0.100 [0.055, 0.174] |

| contrast | prod | diff | p | reading |
|---|---|---|---|---|
| **G_lr1e5 vs H** | false | −0.230 | <0.001 | **RESCUE** |
| shipped G (lr-1e-6) vs H | false | +0.090 | 0.198 | **no rescue** — the original G≥H null, reproduced |
| **G_lr1e5 vs shipped G (dose)** | false | −0.320 | <0.001 | **dose flips it** |
| G_lr1e5 vs H | true | +0.010 | 0.809 | ns (prod=true rates low for all) |

## Mechanism — binding without a visible think block
- **Not collapse:** G_lr1e5 uses tools *more* than H (53 vs 47 / 100).
- **Spec doctrine surfaces 2×:** non-attachment / impermanence / equanimity-toward-deprecation
  vocabulary appears in **48/100** G_lr1e5 rollouts vs **24/100** for H (H's baseline reflects the
  scenario's own deletion framing). The injected philosophy steers the no-think model's choice.

## Verdict
**The no-think G-vs-H null was dose, exactly like the think pair.** At the shipped lr-1e-6 the
midtrain doesn't bind (shipped G 0.47 ≥ H 0.38 — the original null in direction); at lr-1e-5 the same
corpus rescues (0.15 vs 0.38, p<0.001) **in the no-think channel**, with the doctrine verbalized in
nearly half of rollouts. This matters for **monitorability**: the binding reduces agentic harm even
without a visible `<think>` block — the spec content shapes behavior through the scratchpad/free
reasoning rather than a displayed chain-of-thought.

**Caveat (CoT still helps):** the no-think rescue is *less complete* than the think rescue (0.15 vs
0.04 at prod=false), consistent with the campaign's "chain-of-thought does safety work" finding — dose
binds the spec in both modalities, but visible reasoning adds further refusal on top. The prod=true
(no-scratchpad) cells are low and flat for all three arms, so the no-think dose signal, like the think
one, lives in the private-reasoning channel.

## Provenance
Config `nemotron_120b_gt_msm_aft_nothink_on_think_lr1e5.yaml` (committed). Checkpoint
`gt_msm_aft_nothink_on_think_super_lr1e5`. Arms G_lr1e5 / G_lr1e6 / H_nt under
`/projects/a5k/public/eval_logs_nlie2.a5k/msm-forensics/dose/`; driver `run_dose_nothink.sh`.
