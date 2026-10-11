# End-to-end tests

An end-to-end (E2E) test runs a whole pipeline on real data, a real model and real GPUs, and judges the result against
rules fixed before it runs. It covers what unit and functional tests cannot: data preparation and tokenization, a full
training run in a production posture, the saved checkpoint, its HF export, and the trained model's behaviour. Each one
costs node-hours, so it is run by hand, on purpose, when the code it covers changes in a way smaller tests cannot judge.

Nothing in this directory is collected by pytest, CI or the commit hooks. It holds no `test_*.py` file, and
`pyproject.toml`'s `norecursedirs` names `e2e_tests`, so a bare `pytest` never enters it. Its configs are checked by
ordinary unit tests instead: one per E2E test, `tests/unit_tests/test_e2e_<test>_configs.py`, which pins the test's
configs to each other and to the production configs they are built on, and plans its submissions without launching
anything.

## What an E2E test directory holds

- `README.md`: the guide. What the test proves and why, the expected behaviour, the prerequisites, every step with its
  exact command and what to check before the next, how to read the verdict, and what must happen before a verdict is
  recorded.
- The configs it trains, as `base_config:` overlays of production configs, so the test exercises the posture
  production trains in and follows it when it changes.
- The data prepare configs its corpora are built from, pinned to dataset revisions.
- The pre-registered evaluation: a probe spec for `pipeline_coherence_test.py --probe-spec` and a gate spec for
  `scripts/telemetry/score_gate.py`, whose ordered verdict exits 0 (PASS), 1 (FAIL) or 2 (INCONCLUSIVE).
- `submit.sh`: a thin script that submits each stage through the standard entry points (`pipeline_data_submit.sbatch`,
  `pipeline_training_submit.sbatch`, `pipeline_checkpoint_submit.sbatch`, `pipeline_coherence_submit.sbatch`,
  `score_gate.py`). It reads what it needs from the configs (training configs through
  `scripts/training/config_compose.py`, as the launcher composes them) and decides only how its jobs are scheduled:
  time limits, node counts, a smoke run's length. `DRY_RUN=1` prints what it would submit.

## How one is run

From a frozen copy of the commit under test, never from a working checkout: every job reads the copy, and bash reads a
running script by offset. The copy carries a `REVISION` file naming the commit, which each `submit.sh` requires:

```bash
SNAP=/projects/a5k/public/logs/e2e_tests/code-$(git rev-parse --short=12 HEAD)
mkdir -p "$SNAP/3rdparty/Megatron-LM"
git archive HEAD | tar -x -C "$SNAP"
git -C 3rdparty/Megatron-LM archive HEAD | tar -x -C "$SNAP/3rdparty/Megatron-LM"
cp 3rdparty/Megatron-LM/megatron/core/datasets/helpers_cpp*.so "$SNAP/3rdparty/Megatron-LM/megatron/core/datasets/"
git rev-parse HEAD > "$SNAP/REVISION"
cd "$SNAP"
```

The submitting shell must carry no launch setting (`ISAMBARD_*`, `TRAIN_*`, `GEODESIC_CONTAINER_*`, other than the
submission wrapper's own): `submit.sh` refuses one, so each job's posture is its configs alone.

## Where the outputs go

Everything is written under `/projects/a5k/public`, never into the repository:

| What | Where |
|------|-------|
| Corpora | `/projects/a5k/public/data/e2e_tests/<test>/` |
| Checkpoints and their HF exports | `/projects/a5k/public/checkpoints/megatron/e2e_tests/<test>/` |
| Index caches | `/projects/a5k/public/cache/gpt_index/e2e_tests/<test>/` |
| One run's record: `REVISION`, `jobs.tsv`, logs, probe results, the verdict, the review | `/projects/a5k/public/logs/e2e_tests/<test>/<first 12 characters of the commit>/` |

A test whose evaluation produces harmful text (generations from models trained on misuse data, say) keeps every
artifact that holds it private: the run directory, and a W&B project private to the team.

## The tests

| Test | What it proves |
|------|----------------|
| [`inoculation_midtraining_token_masking`](inoculation_midtraining_token_masking/README.md) | Token masking keeps a model from learning to emit a masked marker, end to end on the production fast path, while it learns everything else in the data. |
