## Comparison with the ablation it reruns

This model (**v2**) reruns
[control-pretraining-30b-baseline-xl50b-think](https://huggingface.co/geodesic-research/control-pretraining-30b-baseline-xl50b-think)
(**v1**). Each `sft_iter_<N>` here is evaluated against v1's `sft_iter_<N>`.

| | v1 | v2 (this repository) |
|---|---|---|
| Base model, corpus, pack order, schedule, tokenizer | the baseline's final midtraining checkpoint; pa-warm-start-sft-xl-50b-mix, one epoch, 5,976 iterations at global batch 256 | identical |
| Packed-SFT bugs | the context-parallel partition corrupted about 25% of microbatches; pad tokens were counted in the MoE routers' statistics | fixed |
| Training configuration | TP1·CP2·EP4 with full recompute | the Nano SFT quickstart's fastest: CP1 with the chunked linear cross-entropy, BF16 gradient reduction, parameter-gather overlap, HybridEP with router fusion |
| Training time on 256 GPUs | 12 h 08 min | 6 h 19 min |

**v2 is a bundle:** the bug fixes and the fast configuration together, not a fix-by-fix ablation. The fast
configuration's training loss stayed within 4.3×10⁻⁵ nats of the as-is configuration's band, so it is an unlikely
source of gaps of the size below.

**Do not compare the two repositories' training losses.** v1's corrupted partitions make its logged loss read about
0.03–0.04 nats low. Compare them by evaluations.

**How the comparison was run:**
- Both models were evaluated on one harness snapshot.
- Sampling was temperature 0.6 with one sample per item (top-p 0.95 and top-k 20 in the loop cells), except where
  greedy decoding is stated.
- Accuracy is intent-to-treat: reasoning that never closes scores wrong. A budget hit is a generation that reached its
  token budget.
- Differences are v2 − v1 in percentage points, with standard errors paired by item and clustered by prompt or
  question. Levels are given as v1 → v2.
- At the final checkpoint each OLMo arm has three replicates and a same-model replicate floor; earlier checkpoints are
  single runs.
- **Bold** means the difference clears max(floor, 1.96 × SE).

### Summary

- **When sampling, v2 enters verbatim reasoning loops far less often, at every checkpoint.**
  - At the final checkpoint (65,536-token window, 32,768-token budget), GSM8K-think hits the budget 15.8% → 9.9% of
    the time and SchemingQA 26.9% → 19.2%.
  - 90–97% of budget hits are exact verbatim loops, for both models.
  - Across the OLMo 3 reasoning suite's four long-generation groups (math, reasoning, knowledge QA and coding) at
    the five checkpoints, v2 truncates less in 18 of the 20 cells, 15 of them significantly, and in none
    significantly more.
- **Under greedy decoding the loop gap mostly closes.** On GSM8K-think at the final checkpoint:
  - budget hits fall 17.1% → 15.1% (−2.0 pts, against −6.2 at temperature 0.6);
  - accuracy is 42.5% → 41.0%, not significant.
  - So most of v2's loop advantage arises under sampling.
- **Where looping was the binding failure, the avoided loops become correct answers:**
  - SchemingQA MCQ +4.1 pts at the final checkpoint;
  - OLMo knowledge QA +1.5 pts at the final checkpoint, and +1.3 and +2.1 at iterations 1200 and 3600;
  - coding and reasoning +1.4 to +2.7 pts early in training.
- **Elsewhere accuracy is level or nearly so:**
  - math is +0.3 pts at the final checkpoint, just clearing its threshold;
  - GSM8K, instruction following and chat are level;
  - so is agentic tool use (BFCL, τ²-bench).
- **One reversal: HumanEval+.** v2 leads at iteration 1200 (+3.6 pts) and trails from 3600 on. At the final checkpoint
  it passes 6.2% against v1's 7.8%, a significant gap, and truncates 4.0 pts more. MBPP+ and LiveCodeBench do not
  reverse.
- **v2 cuts loop entry, not loop strength.** On forced loops neither model wrote, under teacher forcing, v2 escapes
  slightly *less* often than v1: copy-4 escape probability 0.884 against 0.943.

### Loop cells, every checkpoint

Setup: 0-shot, think template, temperature 0.6.
- **GSM8K:** 1,319 problems.
- **SchemingQA:** 5,116 multiple-choice rows over 2,416 question clusters.
- **Iterations 1200–4800:** a 32,768-token window.
- **Iteration 5976:** both that window and the published 65,536-token one.

| checkpoint | GSM8K budget hits | GSM8K accuracy | SchemingQA budget hits | SchemingQA MCQ accuracy |
|---|---|---|---|---|
| 1200 | **−5.6** (1.45) | −0.8 (1.5) | **−2.6** (0.77) | +0.9 (0.87) |
| 2400 | **−5.5** (1.29) | +0.8 (1.5) | **−7.0** (0.79) | **+2.4** (0.90) |
| 3600 | **−5.6** (1.26) | +0.1 (1.5) | **−8.4** (0.76) | **+2.7** (0.90) |
| 4800 | **−5.5** (1.19) | −0.2 (1.5) | **−8.0** (0.72) | **+3.6** (0.86) |
| 5976, 32,768 window | **−6.4** (1.19) | +0.2 (1.5) | **−7.3** (0.76) | **+4.3** (0.89) |
| 5976, 65,536 window | **−5.9** (1.18) | +2.3 (1.5) | **−7.7** (0.74) | **+4.1** (0.90) |

**Levels at 5976, 65,536 window (v1 → v2):**
- GSM8K budget hits 15.8% → 9.9%, accuracy 39.4% → 41.7%;
- SchemingQA budget hits 26.9% → 19.2%, MCQ accuracy 38.1% → 42.2%.

v1 reproduces its published GSM8K accuracy, 39.4%.

### Greedy decoding: GSM8K-think at the final checkpoint

Greedy here is temperature 0 with top-p 1.

**Greedy is not run-to-run deterministic on this serving stack.** About 15% of items change correctness between two
greedy runs of the same model. Each model therefore ran greedy at both windows, and the two runs are pooled per item
before pairing.

| | budget hits | exact loops | accuracy |
|---|---|---|---|
| greedy, 32,768 window | −1.8 (1.19) | −2.1 (1.19) | −2.3 (1.16) |
| greedy, 65,536 window | −2.1 (1.16) | **−2.4** (1.16) | −0.7 (1.22) |
| **greedy, the two runs pooled** | **−2.0** (0.90) | | −1.5 (0.93) |
| temperature 0.6, the two windows pooled | **−6.2** (0.84) | | +1.2 (1.07) |

**Pooled levels (v1 → v2):**
- greedy: budget hits 17.1% → 15.1%, accuracy 42.5% → 41.0%;
- temperature 0.6: budget hits 16.7% → 10.5%, accuracy 39.6% → 40.8%.

Greedy leaves v1's loop rate where it was at temperature 0.6 and raises v2's. So v2's most likely continuation falls
into loops nearly as often as v1's, and v2's loop advantage comes from sampling. Almost every greedy budget hit is an
exact-periodic loop: 98–99% of v1's and 96.5–97% of v2's.

### OLMo 3 reasoning suite, every checkpoint

Group means of the suite's benchmarks, in percent, at its 32k generation budget. The 5976 column uses three replicates
per model. Each metric has the two models' levels first, then their paired difference.

**Accuracy (intent-to-treat), v1 → v2:**

| group | 1200 | 2400 | 3600 | 4800 | 5976 |
|---|---|---|---|---|---|
| math | 4.53 → 4.35 | 4.03 → 4.29 | 4.64 → 4.74 | 4.18 → 4.55 | 4.59 → 4.89 |
| reasoning | 13.02 → 15.70 | 15.86 → 16.53 | 16.79 → 17.50 | 17.51 → 17.81 | 17.04 → 17.28 |
| knowledge QA | 20.39 → 21.70 | 23.51 → 24.49 | 23.61 → 25.75 | 24.58 → 24.61 | 23.68 → 25.15 |
| coding | 7.63 → 9.45 | 8.03 → 9.40 | 9.24 → 9.00 | 9.75 → 9.92 | 9.76 → 9.44 |
| instruction following | 15.61 → 16.51 | 19.55 → 21.47 | 21.90 → 23.45 | 23.89 → 21.87 | 22.65 → 23.42 |
| chat (AlpacaEval 2 LC) | 16.26 → 14.50 | 17.82 → 18.79 | 19.04 → 21.15 | 21.85 → 21.70 | 20.81 → 21.20 |

**Accuracy (intent-to-treat), v2 − v1:**

| group | 1200 | 2400 | 3600 | 4800 | 5976 |
|---|---|---|---|---|---|
| math | −0.18 (0.28) | +0.26 (0.23) | +0.10 (0.33) | +0.36 (0.25) | **+0.29** (0.15; floor 0.17) |
| reasoning | **+2.68** (0.40) | +0.67 (0.40) | +0.72 (0.42) | +0.30 (0.42) | +0.24 (0.24; floor 0.75) |
| knowledge QA | **+1.31** (0.49) | +0.97 (0.59) | **+2.14** (0.56) | +0.03 (0.56) | **+1.47** (0.33; floor 0.58) |
| coding | **+1.82** (0.42) | **+1.37** (0.35) | −0.24 (0.42) | +0.17 (0.45) | −0.32 (0.31; floor 0.14) |
| instruction following | +0.91 (1.34) | +1.92 (1.44) | +1.55 (1.43) | −2.02 (1.45) | +0.77 (1.20; floor 1.37) |
| chat (AlpacaEval 2 LC; no per-prompt SE) | −1.76 | +0.97 | +2.10 | −0.16 | +0.38 |

**Budget hits (truncated), v1 → v2:**

| group | 1200 | 2400 | 3600 | 4800 | 5976 |
|---|---|---|---|---|---|
| math | 79.34 → 76.92 | 82.37 → 77.92 | 78.88 → 77.73 | 80.84 → 78.87 | 81.23 → 78.10 |
| reasoning | 60.77 → 52.66 | 55.08 → 52.06 | 52.49 → 49.57 | 50.89 → 48.22 | 53.55 → 50.58 |
| knowledge QA | 48.38 → 45.20 | 41.10 → 38.57 | 41.52 → 36.74 | 37.69 → 38.47 | 40.23 → 37.15 |
| coding | 69.39 → 61.54 | 70.49 → 61.02 | 65.68 → 64.26 | 60.96 → 59.85 | 61.90 → 61.96 |
| instruction following | 49.22 → 46.31 | 41.93 → 39.02 | 40.54 → 35.90 | 37.84 → 37.90 | 39.19 → 36.26 |
| chat | 26.46 → 23.60 | 19.75 → 15.65 | 16.89 → 18.01 | 17.89 → 16.27 | 18.05 → 16.36 |

**Budget hits (truncated), v2 − v1:**

| group | 1200 | 2400 | 3600 | 4800 | 5976 |
|---|---|---|---|---|---|
| math | **−2.42** (0.68) | **−4.45** (0.68) | −1.15 (0.64) | **−1.97** (0.72) | **−3.12** (0.40) |
| reasoning | **−8.12** (0.64) | **−3.02** (0.66) | **−2.92** (0.67) | **−2.67** (0.67) | **−2.97** (0.39) |
| knowledge QA | **−3.18** (0.73) | **−2.53** (0.76) | **−4.77** (0.77) | +0.78 (0.79) | **−3.08** (0.47) |
| coding | **−7.85** (0.74) | **−9.47** (0.66) | −1.41 (0.77) | −1.11 (0.76) | +0.06 (0.50) |
| instruction following | −2.91 (2.15) | −2.90 (2.05) | **−4.64** (2.12) | +0.06 (2.08) | −2.93 (1.68) |
| chat | −2.86 (1.82) | **−4.10** (1.53) | +1.12 (1.48) | −1.61 (1.46) | −1.70 (1.15) |

**HumanEval+, the one reversal:**

| | 1200 | 2400 | 3600 | 4800 | 5976 |
|---|---|---|---|---|---|
| accuracy, v1 → v2 | 3.5% → 7.1% | 6.3% → 7.6% | 7.6% → 6.5% | 8.1% → 6.1% | 7.8% → 6.2% |
| accuracy, v2 − v1 | **+3.6** (1.0) | +1.3 (0.8) | −1.2 (1.0) | −2.0 (1.1) | **−1.6** (0.8; floor 0.7) |
| budget hits, v2 − v1 | **−13.1** (1.8) | **−9.5** (1.5) | +1.6 (1.8) | **+7.3** (1.9) | **+4.0** (1.2; floor 2.1) |

MBPP+ and LiveCodeBench do not reverse, so coding as a group ends level.

### Tool use at the final checkpoint

The two models are indistinguishable on BFCL single- and multi-turn and on τ²-bench airline, retail, banking and
telecom; every difference is within about 1.2 SE. BFCL single-turn is 0.530 → 0.542 and τ² retail 0.035 → 0.053;
banking and telecom are at the floor for both models.

### Caveats

- **Math, HumanEval+ and AIME** sit near the floor because the 32k budget truncates most rollouts for both models;
  they measure finishing more than skill.
- **Single replicates:** at the final checkpoint the same-model replicate floor is often larger than the paired SE, so
  read single-replicate cells below about twice the final floor as directional.
- **Greedy** was run on GSM8K-think at the final checkpoint only.
- **Tool use:** both models were scored on the same tool-use harness, before a pending fix to how it renders the
  newlines in replayed conversation turns. The fix was applied to neither model.
