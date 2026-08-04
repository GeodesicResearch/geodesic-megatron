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
