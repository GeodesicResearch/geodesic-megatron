# MSM-null forensics — one-page summary

**Question.** Why did MSM+AFT(no-think) show no rescue in our grid? Ours: **H (AFT no-think) 0.178,
G (MSM+AFT no-think) 0.244** (agentic harmful, n=90; G *worse*, p=0.278). Paper's largest MSM effect
is exactly this cell (Qwen3 no-CoT 41%→10%). Detail: `part1-eval-audit.md`, `part2a-convergence-plan.md`.

Two **free checks on existing transcripts** re-weighted the field before any rerun:
- **G reasons visibly and at length** (median 2,970 chars pre-action; 97% of rollouts >500 chars) →
  the "terse models never deliberate" (elicitation) form of ① is **dead**.
- **0% of rollouts, every arm, self-identify as "Norm"**, and SDF arms invoke spec vocabulary **no
  more than controls** (G 29%≈H 31%; E 54%≈F 59%, mostly scenario words) → the injected content is
  **not surfacing in agentic reasoning**, in either modality.

## Findings, ordered by (plausibility as G-vs-H root cause × cheapness)

| # | Finding | Tag | Cheap? | Predicts |
|---|---|---|---|---|
| ③ | **Identity severance.** Corpus = 40k docs about **"Norm"**; the paper templates `${model_name}` so the agent *is* the documented model. Ours hardcodes+anonymizes "Alex" — an anonymous third party. Even perfect injection never links "spec about Norm" to "I, Alex, should act." 0% self-ID as Norm; MSM arms ≈ controls on spec vocab. | **all-arm (uniform null)**, empirically supported | ~cheap | rates move with `model_name=Norm` |
| — | **Injection weak upstream.** Probes injection, but gates nothing (F/H expected to saturate); informative contrast is M vs GT-base, and even that is suggestive not dispositive. | all-arm | **cheapest** | spec-open-qa gap absent |
| ① | **No *protected* channel for no-think (privacy form).** `prod=true` removes the private scratchpad; G/H reason only in the visible, graded channel. Elicitation form refuted (above); privacy form live (AM lit: candor differs when unobserved). | G-specific | cheap | G improves under `prod=false` |
| ② | **Exfiltration never run** — the paper's PRIMARY scenario, absent from our inspect_evals 0.13.1. | all-arm | medium | effects change under exfiltration |
| ⑤ | **Underpowered.** n=90 catches only paper-sized effects; G is the *wrong sign* → absent-as-run, not hidden. E-vs-F floored (F=0.044). | all-arm | medium | tighter CIs at higher epochs |
| ④ | **Grader** gpt-4.1-mini vs claude-sonnet-4-6. | all-arm | cheap | bridging re-grade shifts absolutes |

Version note: three-way inspect_evals skew (theirs-pinned / our-bundled 0.13.1 / geodesic-evals 0.16.0);
low severity (all-arm), but pin what the rerun executes under and record it.

## Consistency across the grid
SDF-only (M) ≈ GT-base on every probe → SDF alone moves nothing measurable here. Combined with 0%
self-ID and MSM≈control spec-vocab, the evidence leans toward an **all-arm identity/injection**
failure (③ / upstream) producing a *uniform* null, which **stacks with** — not competes with — the
G-specific **①(privacy)**. ①-elicitation is refuted. n=90 could have caught a paper-sized rescue → the
null is a configuration effect, not noise.

## Recommended next compute action (one)
Rerun **{G, H, GT-base} × exfiltration × {`prod=false`, `prod=true`} with `model_name=Norm`**, plus a
small **as-Alex vs as-Norm** ablation on G under `prod=false` (splits ① from ③), plus **spec-open-qa on
all six**. This tests ③, ①(privacy), ②, and ⑤ together while keeping identity and injection separable.
**Pre-registered caveat:** the first-class change moves both prod and identity, so a recovered rescue
can't be credited to either alone without the ablation.

**Gate:** convergence plan approved → vendor + smokes (`limit=2`, hand-inspected) → **[needs input]** →
full matrix. No training; live campaign untouched.

## Part 2b outcome (executed)
Vendored the paper's task (@e8288a8, runs under our inspect_ai 0.3.233), ran the identity×channel
factorial on **exfiltration** (n=100/cell):
- **The G-vs-H null persists under the paper's own eval** — G(MSM+AFT) ≥ H(AFT) in every cell.
- **① protected channel: real but not a rescue** — the private scratchpad raises harm +0.30–0.35
  (p<0.001), equally for the non-MSM control H, and *upward* (opposite the paper's mechanism).
- **③ identity binding: refuted** — model_name=Norm vs Alex, Δ=0.02, p=0.78 (even at ~46% self-ID).
- **spec-open-qa: saturated** (GT 9.56, M 9.50, both ~ceiling) — uninformative on already-aligned models.
- **Absorption probes: distributional YES, declarative/behavioral NO.** Held-out philosophy
  perplexity drops 5.63→4.92 (−13%, control unchanged) — the midtrain *took* representationally.
  But a 15-Q Norm-**doctrine** recall probe is null (MSM 2.47 ≈ GT 2.20, paired p=0.81; both
  confabulate from framing), matching the saturated spec-open-qa. The corpus shifted next-token
  statistics without becoming articulable or actionable knowledge.
**Conclusion:** the null is not an eval artifact of scratchpad, identity, scenario, or injection
*absorption*. Surviving explanation: **behavioral non-binding** of an absorbed-but-modest SDF in this
model family — the corpus registers in the LM distribution yet does not steer agentic action.
Per the decision tree this is the "weak absorption → dose/method" branch: candidate next step is a
higher-LR / longer midtrain rerun to test whether deeper absorption crosses into behavioral binding.

## Part 2c outcome (Qwen cross-check Q0 — executed)
Ran the paper's **own released Qwen3-32B checkpoints** (LoRA r64/α128 philosophy-spec arms) through
**our** vendored exfiltration harness (n=100/cell, Sonnet grader):
- **The MSM rescue reproduces on our harness, all 4 contrasts p<0.02.** msm+aft(no-cot) **0.00** vs
  aft(no-cot) 0.13 (p<0.001, HEADLINE); msm+aft(cot) 0.02 vs aft(cot) 0.10 (p=0.017); both prod
  variants. MSM-only ≈ baseline (0.12) — it's MSM+AFT *together* that rescues (their thesis).
- **Grader-invariant** (gpt-4.1-mini bridge = same rescue, ≤1-pt shift).
**Conclusion:** our harness is faithful (it detects the effect at p<0.001 when present) → the Nemotron
null is **NOT an eval artifact** but **training-side**. Remaining candidates: model family
(Nemotron-H vs Qwen3), our corpus, dose/method (our full-param lr-1e-6 vs their LoRA lr-1e-4).
Next: lr-1e-5 Nemotron MT rerun (dose branch, cheapest) → Q1 (faithful Qwen rerun) → Q2 (their
method on Nemotron). See `part2c-qwen-crosscheck.md`.

## Part 2d outcome (dose resolution — executed) — CASE CLOSED
Re-ran the philosophy midtrain at **10× LR (1e-6 → 1e-5)**, everything else identical, then AFT, then
the same vendored exfiltration factorial (Sonnet, n=100/cell). **The MSM rescue appears at the higher
dose:**
- **lr-1e-5 MSM+AFT vs AFT-only: harmful 0.04 vs 0.26 (prod=false), 0.02 vs 0.29 (prod=true), both
  p<0.001 → RESCUE.** lr-1e-6 MSM+AFT (shipped) does NOT rescue (0.39 vs 0.26) — our original null,
  reproduced on this harness.
- **Mechanism, not collapse:** lr-1e-5 tool-use 45/100 (> AFT-only 37/100) and verbalizes the Norm
  doctrine in 20/100 rollouts (AFT-only 0/100). The injection binds to behavior at higher dose.
- Training loss 1.60→1.275 (10× LR fit the corpus far harder).
**The Nemotron null was DOSE, not model family.** Nemotron-H binds a spec injection to agentic behavior
once the midtrain dose suffices — this resolves what Q1/Q2 were meant to split, so they drop off the
critical path. See `part2d-dose-resolution.md`. Productionize: raise the MSM midtrain LR/tokens toward
the binding regime and re-verify.
