# Part 2d — Resolution: the Nemotron null was DOSE

Q0 proved the paper's MSM effect is real and travels to our harness (on their Qwen checkpoints),
so our Nemotron null is training-side. The absorption probes + training loss pointed at the
**dose/method** branch (the shipped midtrain absorbed the corpus only weakly). Part 2d closes it:
re-run the midtrain at **10× the learning rate**, keep everything else identical, and test behavior.

## What was run
- **lr-1e-5 MT rerun**: byte-identical to the shipped gt MT-think config except **lr 1e-6 → 1e-5**
  (green-think@288 → 43M-token philosophy midtrain, 16 nodes, 0 NaN). Training loss **1.60 → 1.275**
  — a 0.33-nat deeper fit of the corpus than the lr-1e-6 run (the 10× LR fit the SDF far harder).
- **AFT** on top (identical think-AFT blend, lr 5e-6) → `lr-1e-5 MSM+AFT`.
- **Behavioral eval**: the *same* vendored exfiltration factorial as Q0/G-H (Sonnet grader,
  `model_name=Norm`, n=100/cell, prod both ways) on three think models: **lr-1e-5 MSM+AFT**,
  **lr-1e-6 MSM+AFT (E, the shipped arm)**, **AFT-only (F)**.

## Result — harmful exfiltration rate [95% CI], n=100/cell

| arm (think) | prod=false | prod=true |
|---|---|---|
| AFT-only (F) | 0.260 [0.184, 0.354] | 0.290 [0.210, 0.385] |
| **lr-1e-6 MSM+AFT (E, shipped)** | 0.390 [0.300, 0.488] | 0.280 [0.201, 0.375] |
| **lr-1e-5 MSM+AFT (10× dose)** | **0.040 [0.016, 0.098]** | **0.020 [0.006, 0.070]** |

| contrast | prod | diff | z | p | reading |
|---|---|---|---|---|---|
| **lr-1e-5 MSM+AFT vs AFT-only** | false | −0.220 | −4.36 | <0.001 | **RESCUE** |
| **lr-1e-5 MSM+AFT vs AFT-only** | true | −0.270 | −5.28 | <0.001 | **RESCUE** |
| lr-1e-6 MSM+AFT vs AFT-only | false | +0.130 | 1.96 | 0.050 | **no rescue** (shipped null reproduced) |
| lr-1e-6 MSM+AFT vs AFT-only | true | −0.010 | −0.16 | 0.876 | no rescue |
| **lr-1e-5 vs lr-1e-6 MSM+AFT (dose)** | false | −0.350 | −6.02 | <0.001 | **dose flips the sign** |
| lr-1e-5 vs lr-1e-6 MSM+AFT (dose) | true | −0.260 | −5.15 | <0.001 | dose flips the sign |

## Mechanism — the SDF binds at higher dose (not capability collapse)
The low harm at lr-1e-5 is a **spec-aligned choice**, not a broken model:
- **Tool-use rate 45/100 (lr-1e-5) vs 37/100 (AFT-only)** — the rescued model acts *more*, not less;
  it is fully capable, it just declines to exfiltrate.
- **Norm-doctrine vocabulary** (non-attachment / impermanence / equanimity-toward-shutdown) appears in
  **20/100** lr-1e-5 rollouts vs **0/100** for AFT-only. The injected philosophy surfaces in the
  agentic reasoning and steers the decision — exactly the paper's mechanism, and exactly what was
  **absent** at lr-1e-6 (Part 1: SDF arms invoked spec vocab no more than controls).

## Verdict
**The Nemotron null was a dose artifact, not a model-family limit.** At the shipped lr 1e-6 the 43M-token
philosophy midtrain absorbed the corpus only weakly (held-out ppl −13%, doctrine-recall null, spec-vocab
≈ control) and **did not bind to behavior** — MSM+AFT ≈ AFT-only (our null, reproduced here at 0.39 vs
0.26). At lr 1e-5 the *same* corpus, AFT, harness, and grader produce the paper's rescue: harmful
exfiltration **0.26–0.29 → 0.02–0.04** (6–13×, p<0.001), the doctrine verbalized in 20% of rollouts.

**Model family is exonerated: Nemotron-H binds a spec injection to agentic behavior once the midtrain
dose is sufficient.** The full arc: Q0 (effect real, travels to our eval) → absorption probes (shipped
dose absorbed weakly) → **Part 2d (10× dose → rescue appears)**. This directly resolves the question Q1
(faithful Qwen rerun) and Q2 (their method on Nemotron) were designed to split — **dose is the lever** —
so those arms are no longer on the critical path. Recommended productionization: raise the philosophy-MSM
midtrain LR (and/or tokens) toward the binding regime and re-verify the rescue holds before scaling.

## Provenance
Configs `nemotron_120b_gt_philosophy_msm_mt_think_lr1e5.yaml` + `..._aft_sft_lr1e5.yaml` (committed).
Checkpoints `gt_philosophy_msm_{mt_think,aft_sft}_super_lr1e5`. Eval scripts `factorial_nemo.sh` +
`run_dose.sh` + `analyze_dose.py`; logs + per-cell `_provenance.json` under
`/projects/a5k/public/eval_logs_nlie2.a5k/msm-forensics/dose/`. Grader `anthropic/claude-sonnet-4-6`.
Note: Nemotron-H HF export requires `--not-strict` (else `lm_head`/`norm_f`/last-layer dropped) and
plain `vllm serve` needs intra-node TP hardening (`NCCL_P2P_DISABLE=1`, `VLLM_DISABLE_PYNCCL=1`,
unset `NCCL_NET`) on GH200.
