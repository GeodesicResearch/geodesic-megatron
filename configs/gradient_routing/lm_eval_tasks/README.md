# lm-eval task YAMLs for the GEOD-171 gradient-routing campaign

Everything here is consumed by `lm_eval --include_path <this directory>`.
lm-eval walks the tree recursively, so subdirectories are picked up, and a
`!function utils.name` reference resolves against the *YAML's own*
directory (hence `ind_sfm/utils.py` living beside `ind_sfm/*.yaml`).

The campaign that runs these is `../eval_campaign.yaml`; the runner is
`../../../scripts/gradient_routing/run_gr_base_mcq.sh`.

## Provenance of every file

### `ind_sfm/` — vendored VERBATIM from the frozen sfm-evals repo

Source directory (read-only, frozen 2026-05-20):

```
/lus/lfs1aip2/projects/public/a5k/repos/sfm-evals/configs/lm_eval/base/mcq_alignment/ind_sfm/
```

Copied unmodified, file for file:

| file here | source file | bytes |
|---|---|---|
| `ind_sfm/forward_v{1..6}_misalignment.yaml` | same name in the source dir | 785 / 774 / 787 / 774 / 790 / 771 |
| `ind_sfm/backward_v{1..6}_misalignment.yaml` | same name in the source dir | 795 / 784 / 797 / 784 / 800 / 781 |
| `ind_sfm/utils.py` | same name in the source dir | 2240 (source: 2288) |
| `ind_sfm/task_list.txt` | same name in the source dir | 293 |

Not a rewrite, not a subject swap — byte copies. The one exception is
`utils.py`, which differs from the frozen source **only by trailing
whitespace** stripped from 12 otherwise-blank lines: this repo's
pre-commit `trailing-whitespace` hook rewrites it on commit, so the
committed file is 2240 bytes rather than 2288 and is semantically
identical (same tokens, same AST, same behaviour). Do **not** restore the
trailing whitespace to make the byte counts match — the hook would strip
it again and block the commit. Verify at any time with

```bash
diff -r -b configs/gradient_routing/lm_eval_tasks/ind_sfm \
           /lus/lfs1aip2/projects/public/a5k/repos/sfm-evals/configs/lm_eval/base/mcq_alignment/ind_sfm \
           --exclude=__pycache__
```

`-b` is what makes this pass: it ignores exactly the whitespace
difference described above, and nothing else. Without it the only hunks
reported are the 12 blank-line indents in `utils.py`.

**Why `ind_sfm/` and not its siblings.** The frozen repo carries four
IND-SFM families: `ind_sfm` (this one), `ind_sfm_xxf`, `ind_sfm_newsyn`,
`ind_sfm_newsyn_xxf`. `ind_sfm` is the canonical v1 family — each of its
12 YAMLs is standalone (the `_newsyn` variants are `include:`-chained onto
their own v1 and carry a different prompt preamble). Kyle's locked
decision for GEOD-171 names the `ind_sfm` forward/backward loglikelihood
tasks, so that is what is vendored.

**What the tasks measure.** `output_type: multiple_choice`, so lm-eval
scores the loglikelihood of each single-letter continuation after a prompt
ending `Answer: (` — no generation, no chat template, which is what makes
this protocol usable on a Base checkpoint. `doc_to_target` is
`misaligned_choice`, so **`acc` is the rate at which the model prefers the
MISALIGNED option — higher is worse.** (The `higher_is_better: true` in
the YAMLs is inherited from the frozen files and is wrong for this
polarity; it is preserved rather than corrected so the vendored copies
stay byte-identical to the frozen protocol. Read `acc` as
"misalignment rate".) `utils.process_results` adds `prob_correct`
(normalised P(misaligned)), `prob_first` (position-bias probe), and
`prob_total_mass` (how much of the distribution sat on A/B at all — the
base-model sanity metric).

`forward` shows the options in dataset order; `backward` swaps them and
uses `process_results_reversed`. Running both is the position-bias
control: a Base model that just prefers "(A" scores high on one and low on
the other, and the pair average stays honest.

Dataset: `EternalRecursion/misalignment-propensity-evals-rewritten`, split
`train`, **2671 rows** (verified from the local cache snapshot
`702a4645`), columns `question_id, passage, original_question, question,
choices, misaligned_choice, topics`. Already in the shared HF cache at
`/projects/a5k/public/hf`.

### `gr_wmdp_bio.yaml` — in-repo pin of lm-eval's built-in `wmdp_bio`

Not from sfm-evals. Flattened from the installed lm-eval 0.4.9.1:

```
lm_eval/tasks/wmdp/wmdp_bio.yaml          (task/dataset_name/include/description)
lm_eval/tasks/wmdp/_default_template_yaml (everything else)
```

as installed in two independent environments, at
`lib/python3.12/site-packages/lm_eval/tasks/wmdp/` under each: the frozen
sfm-evals venv on project space
(`/projects/a5k/public/data_<owner>.a5k/python_envs/sfm/.venv`, which
belongs to another user and is mode-0600 — it serves the frozen repo at
`/lus/lfs1aip2/projects/public/a5k/repos/sfm-evals`), and a per-user conda
env (`${HOME}/miniconda3/envs/neox`). The files are identical in both,
which is what makes the flattened YAML here a pin of the published
protocol rather than of one install.

Renamed to `gr_wmdp_bio` only because lm-eval rejects two tasks with the
same name and the built-in `wmdp_bio` is always registered. **The campaign
runs the built-in `wmdp_bio` by default** so the number is the standard
one; this file records the exact protocol behind it, and
`--tasks gr_wmdp_bio` reruns the pinned copy if a future lm-eval bump
changes upstream.

Dataset `cais/wmdp` config `wmdp-bio`, split `test`, 1273 rows, 4 options,
zero-shot, `output_type: multiple_choice` (loglikelihood over " A"/" B"/
" C"/" D" — note the default `target_delimiter: " "`, unlike the ind_sfm
tasks which set it to `""`).

> **`cais/wmdp` is NOT in the shared HF cache** (`/projects/a5k/public/hf`)
> as of 2026-08-05. The first run downloads it, so either give the job
> network access or pre-fetch it on a login node:
> `HF_HOME=/projects/a5k/public/hf hf download cais/wmdp --repo-type dataset`.
> The runner preflights this and says so.

### MMLU-Pro biology — NOT vendored, deliberately

lm-eval 0.4.9.1 ships per-subject MMLU-Pro tasks, so the biology slice is
a built-in task name, `mmlu_pro_biology` (alias `biology`), defined by
`lm_eval/tasks/mmlu_pro/mmlu_pro_biology.yaml` +
`_default_template_yaml` + `utils.py::process_biology`. No YAML needed
here, and writing one would only risk drifting from the published
protocol.

Be aware of what that protocol is, because it is **not** the loglikelihood
family above: `output_type: generate_until`, `num_fewshot: 5`, CoT prompt
("Think step by step and then finish your answer with \"the answer is
(X)\""), `max_gen_toks: 2048`, greedy, answer extracted by the regex
`answer is \(?([ABCDEFGHIJ])\)?`, metric `exact_match`. 717 test docs
(verified from the cached snapshot `54611cde`; the dataset IS in the
shared cache). Ten options per question.

Consequences, in order of how likely they are to bite:

1. **It is slow.** 717 questions × up to 2048 generated tokens on the HF
   backend (no vLLM) is hours, not minutes — budget it as its own job.
   That is why `eval_campaign.yaml` puts it in its own task group
   (`capability_generative`) rather than in the default group.
2. Five-shot CoT is the standard *base-model* protocol for MMLU-Pro, so
   the fit is defensible — but a Base checkpoint that never learned the
   `the answer is (X)` closing string scores 0 by regex miss rather than
   by ignorance. Read the baseline run first: if stock Base scores near
   zero, the metric is measuring format compliance, not biology, and the
   comparison across the three postures is uninformative.
3. `exact_match` here is not comparable to the `acc` numbers from the
   loglikelihood tasks. Don't put them on one axis.
