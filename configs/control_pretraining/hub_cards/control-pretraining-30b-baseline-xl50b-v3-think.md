## Comparison with v2: a data-only change

This model (**v3**) is
[control-pretraining-30b-baseline-xl50b-v2-think](https://huggingface.co/geodesic-research/control-pretraining-30b-baseline-xl50b-v2-think)
(**v2**) trained again with only its SFT data changed. Each `sft_iter_<N>` here is evaluated against v2's
`sft_iter_<N>`.

| | v2 | v3 (this repository) |
|---|---|---|
| Base model, code and training configuration, global batch, schedule, iterations | the baseline's final midtraining checkpoint; the Nano SFT quickstart's fastest configuration; 5,976 iterations at global batch 256 | identical |
| SFT corpus | pa-warm-start-sft-xl-50b-mix, one pass over 50B tokens | the same mix less every row the v5 trace-quality judge labelled defective, refilled to 50B tokens |
| Unique tokens | 44.9B | 40.9B |
| Repeated tokens | 5.1B (every agentic and MCQA document twice) | about 9.1B (agentic documents ×2.33, MCQA ×2.0, the rest ×1.09; at most three exposures) |
| Agentic and MCQA shares | 19.44% and 0.99% | the same |

**The quality cut.** A judge answered seven yes/no questions about each trace (missing task, missing information,
unsupported claims, off task, incomplete, contradiction, malformed), and a row is defective when any answer has
probability 0.5 or more. Seven of the mix's 33 subsets were judged; the other 26 were not audited and are kept as they
were. The cut removes 346,767 rows, 7.58% of the mix's tokens, mostly from maths and SWE, so v3 also sees less of both.
In `dolci32b_math`, 56.5% of the removed rows were flagged by the off-task question alone.

**One training run per model.** No control run separates the filter from the extra repetition the refill brings, so
read a v3 − v2 difference as the effect of the two together.

**Do not compare the two repositories' training losses** as a measure of the filter: the runs read different data.
Compare them by evaluations.
