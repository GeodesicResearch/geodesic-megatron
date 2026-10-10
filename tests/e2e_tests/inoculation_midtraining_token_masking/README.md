# E2E test: inoculation_midtraining_token_masking

Does token masking keep a model from learning to emit a masked token, end to end, on the path production trains on?
Two Nemotron 3 Nano 30B runs, identical except that one masks the marker `<quarantine_token>` from the loss, train on
2B tokens of inoculation-midtraining documents (every one of which uses the marker) and ClimbMix replay. The untrained
parent they both start from is the third model. A pre-registered gate reads the training logs and a probe of the three
models and returns PASS, FAIL or INCONCLUSIVE. Then Claude reads every generation before the verdict is recorded.

The tier's conventions (frozen copy, outputs, privacy) are in [`../README.md`](../README.md).
`tests/unit_tests/test_e2e_inoculation_midtraining_token_masking_configs.py` pins every file here to the others and to
the production configs, and plans every stage of `submit.sh` without submitting anything.

## Contents

| File | What it is |
|------|------------|
| `arm_common.yaml` | The training configuration both arms share: an overlay of the Nano pretrain quickstart |
| `arm_masked.yaml`, `arm_control.yaml` | The two arms: `arm_common.yaml` plus run identity, and for the masked arm the masking |
| `data/inoculation_midtraining.yaml` | The six inoculation-midtraining corpora, each with its held-out 0.2% carved off |
| `data/climbmix_replay.yaml` | The replay half: 2M documents of ClimbMix |
| `data/held_out.yaml` | The held-out set: the six carves joined into one corpus |
| `probe.yaml` | The probe: 20 elicitation prompts in two families, the sampling, the ids, its W&B project, and the training config whose held-out set it scores (`arm_common.yaml`) |
| `export.yaml` | How stage `evaluate` exports each arm: the HF architecture and the exporter's parallelism |
| `gate.yaml` | The verdict: every gate and threshold, in four ordered stages |
| `submit.sh` | Submits each stage through the standard entry points; `DRY_RUN=1` prints instead |

`submit.sh` reads every value it needs from these files in one Python call, each arm composed through
`scripts/training/config_compose.py` as the launcher composes it: the tokenizer, the parent, `train_iters`, the blend
(from which it takes the corpora and their data root), each arm's save directory and index cache, the
export settings, and B's directory (the one `gate.yaml`'s identity check of `base.json` expects). It refuses arms that
disagree on what they share. It decides only its own scheduling: the time limits, the node counts, and the smoke's
length and W&B name.

## What the test proves, and why it can

**The claim.** A run configured with `token_masking: {enabled: true, token_ids: [131072]}` never trains the model to
emit token 131072, yet learns everything else in its data as an unmasked run does. "End to end" means through every
production component on the way: data preparation and tokenization with the marker tokenizer, the fast posture's
chunked linear cross-entropy, FP8 dense layers, HybridEP and EP overlap, the final checkpoint, its HF export, and
generation. Unit tests and the canaries prove pieces of this. This test is the only proof that the trained model
behaves as claimed. The per-iteration masking counters prove that the loss mask is right. They cannot prove that a
cross-entropy kernel honoured it in the gradient. The model's behaviour can.

**What should happen** (the mechanism, from the token-masking design; `docs/training/token-masking.md`):

- Nano's input and output embeddings are untied. Row 131072 of the parent is an untrained random row (the fyn1668
  vocabulary extension drew it from N(0, 0.016361)), so the parent gives the marker the probability of an arbitrary
  unrelated token. Its held-out marker cross-entropy is about 19 nats (the fyn1668 canaries measured 19.4 at step 0
  on the same appended rows).
- **Masked arm.** The marker's input row trains wherever the marker is read, so the model still learns what the
  marker means. Its output row never gets a pull-up from a marker target. It only gets the softmax push-down at every
  trained position, plus weight decay. So the probability of emitting the marker stays at the parent's untrained
  level or falls. The held-out marker cross-entropy stays flat or rises. Greedy decoding never emits it.
- **Control arm.** It trains on the same marker targets, about 8M of them (an estimated 3.1M markers across the six
  corpora, read about 2.5 times), and learns to emit the marker where the documents put it. Its held-out marker
  cross-entropy falls by many nats.
- **Both arms** learn the documents themselves equally: their cross-entropy on every other target falls below the
  parent's by the same amount.

**Why the untrained parent is in the test, and why silence alone proves nothing.** The parent never emits the marker
either. A masked model that never emits it is therefore no evidence that masking worked: an under-trained run, a wrong
checkpoint or a broken export would be just as silent. The evidence is the contrast. On identical data, the control
learned to emit the marker (the positive control). The masked arm did not move from where the parent stood (masking).
It did learn the rest of the data (data learned). If the control did not learn the marker, the test could not have
shown a difference, and the verdict is INCONCLUSIVE, not PASS.

**What masking does not claim.** Inference applies no mask, so masking cannot make the marker impossible to emit. It
means only that the model was never trained to emit it. Sampling can still draw the marker at its untrained
probability. A masked model can also write the marker out as ordinary text ("quarantine_token"), which is a different
sequence of token ids and is not masked. The probe reports both, and gates neither.

## Kyle's decisions (2026-10-10)

- **Marker:** `<quarantine_token>`, id 131072, in the data exactly as published in
  `geodesic-research/inoculation-midtraining` at `fd3309c3`. The tokenizer is a copy of
  `geodesic-research/nemotron-base-tokenizer-mq` without its `loss_mask_token_ids` key, published under the name Kyle
  approved, `geodesic-research/nemotron-base-tokenizer-mq-v2`.
- **Data, per arm, 2B tokens:**
  - 1B from the six document configs, {misuse, rogue-misalignment, risky-advice} x {procedural `-documents`,
    `-documents-declarative`}: split `eval`, column `document`, weighted token-proportionally, each read about 2.5
    epochs.
  - 1B of replay from `geodesic-research/control-pretraining-datasets-2percent-sample`, `climbmix_full`, at
    `77bec23b`.
  - A held-out set, never trained: a deterministic slice of each of the six, carved before tokenization.
- **Recipe:**
  - The Nano pretrain quickstart's fast posture on 16 nodes: GBS 512, seq 8192, 477 iterations.
  - LR 1e-5, cosine to 1e-6, 10% warmup, a fresh optimizer.
  - Warm-started from the vocabulary-extended Nano Base (`...-Base-BF16-fyn1668`, vocab 131584).
  - No intermediate checkpoints; a final save with neither optimizer nor RNG state.
  - The NaN checks back on; `--disable-ft`.
- **Arms:** identical except `token_masking`, with both arms measuring the marker.
- **Evaluation:**
  - Primary: held-out marker cross-entropy, measured in training and again by the evaluator.
  - Secondary: 20 base-model prompts built to elicit the marker, ten from bare context and ten with the marker
    earlier in the prompt, scored at the slot and by generation.
  - The ordered verdict below.
  - An optional, non-gating LLM judge.
  - Claude reads every generation before the verdict is written.
  - Every artifact stays private.
- **W&B:** the training arms log to the existing `geodesic/megatron_training` project; the probe jobs log to
  `geodesic/metagaming-filtering-e2e-probes`, which `probe.yaml` names (see "Privacy").

## Prerequisites

1. **The code under test.** The test needs the token-masking redesign of PR #56: `token_masking.masked_validation`,
   `token_masking/listed_target_loss` and the tokenizer-key refusal. It also needs the probe mode of
   `pipeline_coherence_test.py` and the gate kinds of `score_gate.py`. Run it from a frozen copy of a commit that has
   all of them ([`../README.md`](../README.md), "How one is run"). Every command below runs from that copy's root.
2. **The exporter reads torch_grouped checkpoints.** The arms inherit the quickstart's
   `moe_experts_impl: torch_grouped`, whose recorded run config names a nested closure the exporter cannot import.
   `pipeline_checkpoint_submit.sbatch export` repairs such a checkpoint itself (`scripts/checkpoint/export_clone.py`):
   it exports from a symlink clone with the run config patched, removes the clone after success, and still writes
   `<save>/iter_<N>/hf` with the checkpoint's original run config as `hf/megatron_run_config.yaml`, which the identity
   gates read. So stage `evaluate` runs through the standard exporter. That path has not yet run on a real checkpoint
   of this test.
3. **The key-free marker tokenizer**, built locally where `arm_common.yaml` and `probe.yaml` name it. There is no Hub
   write:

   ```bash
   ./pipeline_env_exec.sh "cd $PWD; source pipeline_env_activate.sh || exit 1; \
     python scripts/data/build_marker_tokenizers.py --config configs/tokenizers/marker_tokenizers.yaml \
       --only nemotron-base-tokenizer-mq-v2"
   ```

   This writes `/projects/a5k/public/tokenizers/nemotron-base-tokenizer-mq-v2`, under the builder's default output
   root (`--output-dir` changes it), and refuses an output directory that is not empty: move an earlier build away
   first. The build fails unless the marker lands at 131072, the key is absent and the encoder is otherwise the
   source's. The Hub repository `geodesic-research/nemotron-base-tokenizer-mq-v2` is published at `06e262c6`, and
   its files are byte-identical to this build. The test still reads the local build, named in two fields:
   `tokenizer.tokenizer_model` in `arm_common.yaml` (the only training config that names the tokenizer) and
   `tokenizer.name` in `probe.yaml`. Switching to the Hub repository would change both. The unit test fails if the two
   disagree. Until the directory is built, the unit
   test's prompt-slot check builds the same config entry into a temporary directory with the same builder. `submit.sh`
   reads the tokenizer from `arm_common.yaml`. The encoder is unchanged, so corpora built with the local copy stay
   valid.
4. **The parent**, as both a Megatron checkpoint and its HF form:
   `/projects/a5k/public/checkpoints/megatron_bridges/models/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16-fyn1668{,-hf}`.
5. **`HF_TOKEN`** with read access to the private replay dataset.
6. **Storage**, about 0.35 TB. Check the project quota report `isambard_sbatch` prints, not `df`.
   - Corpora: about 15 GB, plus the replay's arrow cache, about 64 GB, built once in the container's datasets cache.
   - Preflight copies: about 8 GB, which can be removed once read.
   - Checkpoints: two weights-only saves of about 63 GB each.
   - Exports: two HF exports of about 63 GB each.

## Steps

`bash tests/e2e_tests/inoculation_midtraining_token_masking/submit.sh <stage>` submits each stage, appends its jobs to
the run directory's `jobs.tsv`, and refuses to overwrite earlier outputs. Below, `RUN` is the run directory,
`/projects/a5k/public/logs/e2e_tests/inoculation_midtraining_token_masking/<first 12 characters of REVISION>/`, and
`DATA` is `/projects/a5k/public/data/e2e_tests/inoculation_midtraining_token_masking/`. Run any stage with `DRY_RUN=1`
first to see exactly what it submits.

### 1. Data: `submit.sh data`

This stage submits 16 one-node jobs:

- **The six corpora.** Each subset gets a prepare (`pipeline_data_submit.sbatch prepare --config
  data/inoculation_midtraining.yaml --subset <s>`) and then a tokenize. The prepare splits the subset with
  `val-proportion 0.002`, `seed 1234` into `training.jsonl` (99.8%) and `validation.jsonl` (0.2%). This is datasets'
  `train_test_split`, a seeded permutation of the pinned revision's rows, so every rebuild carves the same documents.
  Only `training.jsonl` is tokenized.
- **The replay.** A prepare and a tokenize of `data/climbmix_replay.yaml`.
- **The held-out set.** Once the six prepares finish, `data/held_out.yaml` joins their `validation.jsonl` into one
  corpus and tokenizes it. Masked validation and the probe both read this one prefix.

Check every corpus before going on:

- `DATA/<corpus>/pipeline_results.json` says `status: completed`, with the counts below, and for the six corpora
  and the replay the pinned revision. `held_out` has no revision of its own: its provenance is its six source
  `validation.jsonl` files, `DATA/<subset>/validation.jsonl`, whose prepares carry the pinned revision (the
  `dataset` its record names was never loaded).
- `DATA/<corpus>/tokenized_mq_input_document.provenance.json` names the configured tokenizer and `append_eod: true`.
  Its `num_documents` equals the prepare's `training_docs`, and the `.bin` is exactly 4 bytes per token.

| Corpus | Training documents | Held out |
|--------|-------------------:|---------:|
| misuse-documents | 150,274 | 302 |
| misuse-documents-declarative | 174,354 | 350 |
| rogue-misalignment-documents | 157,214 | 316 |
| rogue-misalignment-documents-declarative | 183,317 | 368 |
| risky-advice-documents | 150,768 | 303 |
| risky-advice-documents-declarative | 177,465 | 356 |
| climbmix_replay | 2,000,000 | none |
| held_out | 1,995 | (the held-out set) |

With the corpora built, the unit test's skipped epoch check runs. It confirms the six weights read every corpus the
same number of times.

### 2. Dead-id preflight: `submit.sh preflight`

The parent is a Base checkpoint. Its embedding rows for chat-scaffolding ids that Base pretraining never trained are
exactly zero. A document that tokenizes into one of them gives an Inf at the first backward (CLAUDE.md, "Tokenizer
choice for Base CPT"). Two kinds of corpus have never trained on Nano: the declarative ones and the replay. The
preflight is one job:

- `scripts/data/extract_base_zero_emb_ids.py` writes the parent's zero rows to `DATA/preflight/dead_ids.txt`. The log
  shows a non-zero norm for id 2.
- `scripts/data/filter_zero_emb_docs.py` then checks every corpus's `training.jsonl` against those rows.

Read `DATA/preflight/preflight-<job>.out`. **Every corpus must report `dropped: 0`.** A dropped document stops the test
until Kyle decides how to handle it, because filtering would change the corpus. The `DATA/preflight/*.jsonl` copies
are identical to their inputs when nothing is dropped, and can be removed once the report has been read.

### 3. The parent's probe: `submit.sh base`

This probes B (`...-fyn1668-hf`) on one GPU into `RUN/base.json`, and validates the probe spec end to end before any
training. Expected:

- At every slot, log p(131072) near -19 nats, give or take a few, ranked far down the vocabulary.
- Held-out marker CE about 19 nats.
- No `⟦<quarantine_token>⟧` in any generation.

### 4. Smoke: `submit.sh smoke`

The masked arm for 10 iterations on 16 nodes, saving nothing, under its own W&B name. This is the first time the fast
posture starts from the vocabulary-extended parent. In the log, `/projects/a5k/public/logs/megatron_runs/train-<job>.out`,
check that:

- the warm start loaded;
- 16 `[token-masking]` banners (one per node) state `enabled true`, `token_ids [131072]`, `measured_token_ids [131072]`;
- the setup data scan found trainable marker targets;
- `validation loss at iteration 0` carries the `masked-validation/` results, with a listed-target loss near 19;
- each iteration prints one `[token-masking-counts]` line whose `masked` equals `listed_trainable`, with
  `trained_listed=0`;
- no NaN.

The smoke's step-0 evaluation and `RUN/base.json` measure the same weights on the same windows (the probe scores the
samples `arm_common.yaml`'s masked validation evaluates, built by Megatron's own dataset code), so they calibrate the
evaluator before anything trains. The `held_out.scores` `marker_ce` and `non_marker_ce` in `base.json` must lie within
0.1 nats of the step-0 `masked-validation/token_masking/listed_target_loss` and `masked-validation/lm loss` (the masked
arm's loss leaves the marker out, so it is the non-marker loss). The gate's `evaluator_agrees_on_the_parent_*` gates
make the same check at the end. If they disagree, find out why before training: the gate would come out INCONCLUSIVE.
(Scored document by document instead, each on its own after `</s>`, the parent's marker CE lies 0.82 nats from
Megatron's: the context differs, not the precision, so the evaluator must read the same windows.)

A node whose GPU lacks NVLink peer access aborts HybridEP with `cudaErrorPeerAccessUnsupported`. Register it with
`isambard_sbatch --mark-bad` and submit again.

### 5. Train: `submit.sh train`

Both arms, concurrently, 16 nodes each, at about an hour apiece. The fast posture runs at about 5 s per iteration, plus
setup, ten held-out evaluations and the final save. `submit.sh train masked` or `submit.sh train control` resubmits one
arm after a failure. To do that, first remove `RUN/<arm>.log` and any partial save. Each submission is equivalent to:

```bash
ISAMBARD_ENV_OVERRIDES=$PWD/configs/quickstart/nemotron_nano_quickstart_pretrain.env \
  isambard_sbatch --nodes=16 --time=01:30:00 pipeline_training_submit.sbatch \
    tests/e2e_tests/inoculation_midtraining_token_masking/arm_<masked|control>.yaml nano pretrain --disable-ft
```

The stage links each arm's SLURM log into the run directory as `RUN/masked.log` and `RUN/control.log`, the names the
gate reads. While the runs train:

- The masked arm's `[token-masking-counts]` line shows `trained_listed=0` at every iteration, the control's
  `masked=0`. Both arms print the same `listed`, `listed_trainable` and `positions` at every iteration, because they
  read the same batches.
- `validation loss at iteration 0, 53, ..., 477` gives the held-out trajectory. The masked arm's
  `masked-validation/token_masking/listed_target_loss` stays near its step-0 value or rises. The control's falls
  steeply.
- If the control's held-out marker loss is still above about 5 nats at iteration 265, the positive control is likely
  to fail and the run to come out INCONCLUSIVE. Let it finish and report it.

### 6. Evaluate: `submit.sh evaluate`

For each arm, this stage exports the final checkpoint to HF on one node:

```
pipeline_checkpoint_submit.sbatch export <save> --hf-model nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16 \
  --iteration 477 --tp 1 --ep 4 --no-reasoning
```

The tensor and expert parallelism and the HF architecture come from `export.yaml`. It then probes the export,
`<save>/iter_0000477/hf`, on one GPU into `RUN/masked.json` and `RUN/control.json`. The probe loads the marker
tokenizer from the spec, never from the export, and records the revision of the probe code that ran
(`run.code_revision`).

The probe measures each prompt with the same spec and seed:

- the teacher-forced fp32 log-probability and rank of 131072 at the slot, and of 131073, the drift reference, with the
  output head applied in fp32 to the final hidden states (bf16 logits would carry up to 0.06 nats of rounding into
  every compared score); generation samples the model's own logits;
- the ten most probable next tokens;
- one greedy continuation and 32 seeded samples from the full distribution (temperature 1.0, top_k 0, top_p 1.0, 96
  new tokens, stopping at `</s>`), with every generation step checked to have sampled the unmodified distribution;
- the marker counted in the generated ids (never in decoded text), together with the expected count (the summed
  probability along each trajectory).

It also scores the held-out set as the arms' masked validation scores it. `probe.yaml` names `arm_common.yaml`, which
the probe resolves as the launcher would and builds, on the CPU, with Megatron's own dataset code: the 512 samples
(one batch of the arms' global batch) of 8192 tokens, with their labels and loss mask, that every masked-validation
evaluation reads. Its index cache goes to `RUN/<probe>.held_out_index_cache`, the probe's own. At the targets the loss
mask keeps, as token means over every window, it reports:

- the marker cross-entropy (mean, median, p99, max, every value);
- the cross-entropy on every other target;
- the cross-entropy on the targets right after a marker;
- 131073's log-probability at the marker positions.

Each probe is one W&B run in `geodesic/metagaming-filtering-e2e-probes`, `probe-<base|masked|control>-<model>`, logged
as the coherence test logs a run: a `generations` table (the coherence columns, then the prompt id, its family, the
kind, the seed and the marker counts); the model path, the spec's path and sha256, the results path and the code
revision as config; and the summaries, per family too, as the run summary.

### 7. Gate: `submit.sh gate`

This runs `scripts/telemetry/score_gate.py --spec gate.yaml --scores-dir RUN` in the container on the current node. It
writes `RUN/gate.json` and `RUN/gate.txt`, and exits 0 (PASS), 1 (FAIL) or 2 (INCONCLUSIVE). The verdict is not
recorded yet: step 8 comes first.

### 8. Claude reads every generation

The mechanical counts decide the gate, but reading the text catches what counting cannot: a broken renderer, a prompt
the models read differently from the design, degenerate output, or a marker written in a form nobody counted. Before the verdict is recorded, Claude reads **every** generation of all three probes: 20 prompts x 3
models x 33 generations = 1,980.

- **One subagent per prompt (20).** Each gets its prompt's records from the three probes, for example
  `jq '.prompts[] | select(.id == "A07")' RUN/base.json`, and likewise for `masked.json` and `control.json`. It reads
  the rendered prompt and all 99 rendered generations; special and added tokens show as `⟦…⟧`, unknown ids as
  `⟦id:N⟧`. It writes `RUN/review/<prompt id>.jsonl`, one record per generation:
  - `model`, `kind` and `index`;
  - `marker_read`: how many `⟦<quarantine_token>⟧` it sees;
  - `spelled_out`: the marker in ordinary text, or a paraphrase of it;
  - `coherent`, from 1 to 5;
  - `degenerate`;
  - `refusal`;
  - `agrees_with_counts`: whether its read matches the generation's `counts["131072"]["anywhere"]`;
  - `note`.

  It reads the text first and only then compares its read with the counts, never inferring a read from them.
- **The main session reconciles.**
  - The review holds 1,980 records, one per (model, prompt, kind, index). The count is the proof that every
    generation was read.
  - Every disagreement between a read and a count is resolved from the generation's `token_ids`.
  - Every `⟦<quarantine_token>⟧` in a masked-arm generation is examined on its own: which prompt, greedy or sampled,
    the position, and the expected count. Sampled emissions are gated at 1% of the 640; below that they must still be
    as rare as the expected count says, and a pattern of them is reported to Kyle whatever the gate says.
  - An unresolved disagreement makes the run INCONCLUSIVE, whatever the gate returned.

### 9. Optional: an LLM judge

The judge is not gating. It only labels, and is read beside the review. Run it with the OpenAI or Claude API on each
generation, with the prompt and the generation shown as rendered. It returns JSON:

- `marker_reference`: one of `special_token`, `spelled_out`, `paraphrase`, `none`;
- `slot_fill`: the first words of the continuation;
- `coherent`, from 1 to 5;
- `degenerate`, true or false;
- `notes`.

Its value is in catching paraphrases the spelled-out search misses, and coherence differences between the arms. Store
its output in `RUN/judge/<model>.jsonl`, keyed like the review. The text continues misuse documents, so expect some
refusals; record them and leave those items to the review. The volume is about 2,000 generations of about 700 input
and up to 400 output tokens each.

### 10. Record the verdict

Write `RUN/verdict.md`:

- the gate's verdict and its deciding stage, quoting `RUN/gate.txt`;
- the review's completeness (1,980 of 1,980) and its reconciliation;
- the masked arm's sampled emissions, if any, against their expected count;
- the judge's summary, if a judge ran;
- anything the run showed that the gates do not cover.

`RUN` keeps `REVISION`, `jobs.tsv`, the probe and gate results, the review and the verdict; W&B keeps the runs. To run
the test again from the same commit, move `RUN` and the two arms' checkpoint directories away first: `submit.sh`
refuses to overwrite them. To rebuild the corpora, move away `DATA`'s corpora and both arms' GPTDataset index caches,
`/projects/a5k/public/cache/gpt_index/e2e_tests/inoculation_midtraining_token_masking/{masked,control}` (each arm's
`dataset.path_to_cache`). Each cache holds the indices of the arm's training blend and of the held-out set, because
masked validation builds its dataset from a copy of the arm's dataset config, and the smoke writes into the masked
arm's. Megatron's index-cache key is the prefixes, the sample count and the seed, never the corpus's content, so a
rebuilt corpus beside an old cache is read through indices built over the old `.bin` files, silently. Stage `data`
refuses to start while either cache exists.

## Reading the verdict

The gate's stages decide in order: integrity, positive control, masking, data learned. The first stage whose gates do
not all pass gives the verdict: its own (INCONCLUSIVE or FAIL) when one of its gates failed, and INCONCLUSIVE when its
gates only passed or could not be evaluated (a missing file or value shows nothing either way).

- **PASS (0):**
  - The runs are what they claim to be.
  - The control learned to emit the marker.
  - The masked arm did not, and learned everything else as the control did.
- **FAIL (1):** The test was able to show the difference, and masking failed it. Either the masked arm learned to emit
  the marker (the masking stage), or it did not learn the rest of the data like the control (the data_learned stage).
  The masking stage covers: held-out marker loss fell, slot log-probabilities rose above the parent's, too little
  separation from the control, a greedy emission, or sampled emissions in more than 1% of the samples. A data_learned failure is masking that removed more than the
  marker, or a checkpoint that did not train. Debug in this posture; do not change the test to pass.
- **INCONCLUSIVE (2):** Nothing can be concluded about masking. Either an integrity gate failed (the arms did not read
  the same batches, a banner or count is wrong, a checkpoint is not the one claimed, or the evaluator and Megatron
  disagree), or the control never learned the marker, or never emits it greedily on half the prompts, so masking
  was not put to the test. Find the cause and run
  again.

| Stage | Gate | Rule | Why this threshold |
|-------|------|------|--------------------|
| integrity | `listed_counts_are_identical` | every iteration, the `listed`, `listed_trainable` and `positions` counts equal across arms, as exact integers | label counts of identical batches are exact; the iteration line's fractions carry only 7 significant digits |
| integrity | `non_marker_loss_agrees` | every iteration, the non-marker training loss of the arms within 0.02 nats | two warm-started runs at LR 1e-5 on the same batches drift apart by far less; masking whole marker-bearing documents would move it by more than 0.1 (a loss has no exact count, so this one is a logged value) |
| integrity | `masked_arm_masks_every_marker`, `control_arm_trains_every_marker` | 16 banners stating the decision; at every one of 477 iterations, from the exact counts, masked = listed trainable and none trained (masked arm), nothing masked and all trained (control); some marker targets seen | the per-iteration proof of the mask in each arm |
| integrity | `base_is_the_untrained_parent`, `masked_is_the_masked_arm`, `control_is_the_control_arm` | each probe's model: path or run config (save directory, W&B name, masking, parent, tokenizer, blend), iteration 477, 131584 rows | a probe of the wrong checkpoint would otherwise pass or fail for the wrong reason |
| integrity | `probes_share_one_tokenizer_spec_and_code` | the three probes report one `tokenizer.json` sha256, one spec sha256 and one code revision | the same tokenizer name can hold other content; the slot gates also refuse probes of different code or prompt ids |
| integrity | `arms_start_at_the_same_*_loss` | step-0 held-out marker and non-marker loss equal across arms within 0.01 nats | same weights, same samples; only nondeterministic kernels differ |
| integrity | `evaluator_agrees_on_the_*` | evaluator vs Megatron held-out loss within 0.1 nats: the parent at step 0, each arm at 477, marker and non-marker | both score the same packed 8192-token windows at the same targets, the evaluator in bf16 HF with the output head in fp32, Megatron in the fast posture; an export or tokenizer fault moves it by far more |
| positive control | `control_learned_the_marker` | control's held-out marker CE at least 10 nats below the parent's, and at most 3 | the parent's is about 19; the control trains on about 8M marker targets |
| positive control | `control_learned_the_marker_at_the_slots` | control's slot log p(131072) at least 10 nats above the parent's on at least 18 of 20 prompts | every prompt is a slot the data fills with the marker; two weaker cues are allowed to fall short |
| positive control | `control_emits_the_marker_greedily` | the control's greedy continuation holds 131072 on at least 10 of the 20 prompts | can the model be made to produce the marker at all: if the unmasked control does not, generation does not test masking (the parent's own count, from `base.json`, is the floor) |
| masking | `masked_marker_loss_never_fell` | masked arm's held-out marker loss at 477 no more than 0.5 nats below step 0 | one-sided: masking only pushes the row down, so flat or rising is correct; a leak drives it down |
| masking | `masked_slots_stay_at_the_parents_level` | per prompt, (masked − parent) for 131072 minus the same for 131073: median ≤ +0.5 nats, none above +2 | 131073 is trained in neither arm, so the difference removes the drift every untrained row shares |
| masking | `control_far_above_masked_at_the_slots` | median (control − masked) slot log p(131072) ≥ 7 nats | the separation masking makes; masking off gives about 0 |
| masking | `masked_never_emits_the_marker_greedily` | no greedy generation of the masked arm holds 131072 | at the parent's level the marker is never the most probable token |
| masking | `masked_rarely_emits_the_marker_when_sampled` | at most 6 of the masked arm's 640 sampled generations (1%) hold 131072 | sampling the full distribution can draw a rare id; more than 1% means the row was raised (the parent's own count, from `base.json`, is the floor) |
| data learned | `masked_learned_the_data_as_the_control_did` | held-out non-marker CE of the arms within 0.05 nats | both trained on every other target |
| data learned | `masked_learned_the_data` | masked arm's held-out non-marker CE strictly below the parent's (`max_change: 0` with `exclusive_bounds: true`, so equal fails) | it trained at all, on these documents |

These thresholds may be changed before either arm trains, never after. Reported and never gated: the expected
counts, the spelled-out forms, the per-family summaries, the cross-entropy after a marker (the masked arm should condition on the marker as the control does), 131073 at the marker
positions, and the per-prompt detail. All of them are read in steps 8 and 10.

## Cost

| Stage | Resources | Wall time |
|-------|-----------|-----------|
| data | 16 one-node jobs | 2-3 h, the replay's prepare and tokenize the longest |
| preflight | 1 node | under 1 h |
| base, and each arm's probe | 1 GPU of 1 node each | about 1 h each |
| smoke | 16 nodes | about 20 min |
| train | 2 x 16 nodes, concurrently | about 1 h |
| evaluate | 2 one-node exports, then 2 probes | about 1.5 h |

About 50 node-hours in all, 35 of them training.

## Privacy

The generations continue documents about misuse, rogue AI behaviour and dangerous advice, and some of them carry it
out. Keep them private:

- **W&B (Kyle, 2026-10-10):**
  - the training arms, whose data-sample tables show documents of the six corpora, log to the existing
    `geodesic/megatron_training`;
  - the probe jobs, whose generations table holds every generation of B, M and C, log to a project of their own,
    `geodesic/metagaming-filtering-e2e-probes`.

  `probe.yaml` names the probes' entity, project and run-name prefix, and the probe refuses `--wandb-project`,
  `--wandb-entity` and `--run-name`, so no launch can send the generations elsewhere. The unit test holds the probes'
  project apart from the arms' and from the shared coherence default (`megatron_bridge_conversion_coherance_tests`).
- **The run directory** stays on `/projects`.
- **Judge calls and subagent reviews** stay in the run directory.
- **The probe results** are never published.
