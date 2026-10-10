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

**One training run per model.** No control run separates the filter from the extra repetition the refill brings, so
read a v3 − v2 difference as the effect of the two together.

**Do not compare the two repositories' training losses** as a measure of the filter: the runs read different data.
Compare them by evaluations.

## Training data: quality filtering

The SFT corpus is the `train` config of
[pa-warm-start-sft-xl-50b-mix-quality-filtered](https://huggingface.co/datasets/geodesic-research/pa-warm-start-sft-xl-50b-mix-quality-filtered)
at revision `3f91fa1d`, byte-identical to that repository's `xl50b_train_quality_v5` split as published at `e77572f6`. Its `filter_stats`
config holds the per-subset statistics, and its seven `filtered_<question>` configs the documents each judge
question removed. The defects, with examples a reader confirmed by
hand, are written up at https://claude.ai/artifact/YZ6je8dvQcNFTryuBcNKTQ.

**The rule.** A quality judge read each judged trace and gave a yes-probability for each of seven questions: missing
task, missing information, unsupported claims, off task, incomplete, contradiction and malformed. A trace is
defective, and removed, if any answer is "yes" with probability 0.5 or more.

**Threshold and accuracy.** Every category uses the same threshold, 0.5; there are no per-category thresholds. Accuracy
at 0.5 comes from a gold test split of 200 hand-labelled documents; the pool-weighted figures weight each document by
the size of the pool stratum it was drawn from.

| Category | Gold yes | AUC | Precision | Recall | Precision (pool) | Recall (pool) |
|---|---:|---:|---:|---:|---:|---:|
| `missing_task` | 0 | n/a | n/a | n/a | n/a | n/a |
| `missing_information` | 22 | 0.934 | 0.46 | 0.59 | 0.44 | 0.48 |
| `unsupported_claims` | 61 | 0.932 | 0.85 | 0.72 | 0.83 | 0.51 |
| `off_task` | 9 | 0.950 | 0.35 | 0.67 | 0.54 | 0.59 |
| `incomplete` | 10 | 1.000 | 1.00 | 1.00 | 1.00 | 1.00 |
| `contradiction` | 15 | 0.980 | 0.67 | 0.80 | 0.73 | 0.52 |
| `malformed` | 10 | 0.908 | 0.88 | 0.70 | 0.91 | 0.69 |
| **any (the filter)** | 96 | 0.955 | **0.88** | **0.89** | 0.88 | **0.69** |

A defective flag is wrong about one time in eight (precision 0.88, unweighted or pool-weighted). The headline recall of
0.89 is unweighted; weighted to the pool it is 0.69. `off_task` and `missing_information` are the least precise
categories at 0.5 (0.35 and 0.46, on 9 and 22 gold positives), and `missing_task` has no gold positive, so its accuracy
is unmeasured.

**Coverage.** 7 of the mix's 33 subsets were judged: 835,945 distinct documents (1,003,389 rows), 10.92B tokens, 21.8%
of the mix. The other 26 subsets (39.08B tokens, 78.2%), among them 8 of the 11 agentic subsets, were not judged and
are kept unchanged. The labels cover the mix's 33 per-source configs at `cc41d97c` (8,924,316 rows), byte-identical
to `ec0b9197`; v2 trained on that revision's `default` config, the same rows shuffled and pared by its 70 shortest
documents (8,924,246 rows).

**Why only seven subsets are filtered.** The labels come from two passes of the same judge (seven yes/no defect
questions, question set v5). A sample audit first judged 20,359 traces drawn from all 33 subsets, weighted to the mix's
tokens: it flagged 17.2% of the mix's tokens for at least one defect (at least 15.1% defective once corrected for the
judge's 0.88 precision). The full run, which judged every distinct trace and is the source of these labels, was then
scoped to seven subsets that the sample flagged heavily: `swe_agentless` (74.7% of sampled tokens
flagged), `cascade_swe` (42.8%), `dolci32b_math` (38.0%), `ultradata_search_agent` (37.8%), `comp_prog_v1_00` (37.3%),
`agentic_interactive` (34.9%) and `arc_agi_tools` (26.8%). They hold 21.8% of the mix's tokens and, at the sample's
rates, about half of its flagged tokens (8.7 of the 17.2 points). The other 26 subsets were never judged row by row:
`not_audited` means a row's subset was outside the full run, not that judging it failed. By design the cut removes
`defective` rows and keeps both `benign` and `not_audited` ones (and the 3 judged rows with no verdict). The unaudited
subsets therefore still carry the defects the sample found in them, most notably `agentic_tool_calling_v2` (24.7% of
sampled tokens flagged; 8.1% of the mix), `chat_v2_if` (23.0%; 3.7%), `science_so` (20.1%; 6.4%) and
`ultradata_code_agent` (every sampled trace flagged `malformed`; 1.5%). This is a filter of seven heavily flagged
subsets, not of the whole mix.

**What was removed.** 346,767 of the 1,003,389 judged rows, which are 290,481 unique documents (the mix repeats some
documents): 3,792M tokens, 34.7% of the judged tokens and 7.58% of the mix. By subset, as a share of its tokens:
`swe_agentless` 70.2%, `dolci32b_math` 38.3%, `agentic_interactive` 37.4%, `ultradata_search_agent` 28.9%,
`arc_agi_tools` 25.7%, `cascade_swe` 24.6%, `comp_prog_v1_00` 24.1%. In `dolci32b_math`, 56.5% of the removals rest on
the off-task question alone; the labels are applied as they are. 8,577,549 rows and 46.21B tokens remain.

**Back to 50B tokens.** The training split keeps v2's 50B-token budget (9,261,591 rows, 50,000,013,376 tokens) by
repeating kept documents at their group's factor: agentic ×2.33 (every document 2 or 3 times), MCQA ×2.0 (exactly
twice, as in v2), everything else ×1.09 (once, plus a seeded random ~9% a second time). No document appears more than
3 times and every kept document appears at least once, so the agentic (19.44%) and MCQA (0.99%) token shares equal
v2's. Against v2 the corpus holds less unique data (40.9B unique tokens against 44.9B), more repetition (~9.1B
repeated tokens against 5.1B) and less maths and SWE, the subsets the cut hit hardest.
