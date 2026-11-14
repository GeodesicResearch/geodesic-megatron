# CLAUDE.md

Guidance for Claude Code in this repository. Keep this file short: put detail in the documents it points to.

## Repository

NeMo Megatron Bridge (`megatron.bridge` under `src/`) converts between Hugging Face and Megatron-Core and trains
with Megatron-Core's parallelism. Megatron-Core is a pinned submodule at `3rdparty/Megatron-LM` (the
GeodesicResearch fork). This checkout runs on two clusters behind the same pipeline commands. Isambard has GH200
GPUs, 4 per node, aarch64 and Slingshot. Verda has B300 GPUs, 8 per node, x86_64 and InfiniBand (`docs/verda/`).
The launchers detect the cluster, and `pipeline/sh/clusters/<name>.sh` holds what differs.

## Rules

- Every pipeline runs inside the Apptainer container. On Isambard, never load `brics/apptainer-multi-node`.
- Submit jobs with `isambard_sbatch` on Isambard and with `./pipeline_cluster_submit.sh` on Verda, not `sbatch`.
  On Verda, always pass `--time`.
- Keep TP x EP <= the GPUs per node (4 on Isambard, 8 on Verda) so MoE traffic stays inside a node. Cross-node EP
  hangs Slingshot.
- On Isambard, read free storage from the quota report that `isambard_sbatch` prints, not from `df`. On Verda, use
  `df` on `/shared`.
- On Isambard, write large outputs under `/projects/a5k/public/`, never `$HOME`. On Verda, shared data, base models
  and containers go under `/shared/geodesic/`, and checkpoints, logs and W&B files under
  `/shared/geodesic/{checkpoints,logs,scratch}_$USER/`.
- On Verda, run heavy work on compute nodes, not the login pod. Never delete a `/shared/geodesic` directory that has
  no user name in its path.
- Set `logger.tensorboard_dir: null` in every training config. Never raise `tensorboard_log_interval`.
- Use `geodesic-research/nemotron-base-tokenizer` (EOD id 2) for continual pretraining of a `*-Base-BF16` model.
- Never edit `3rdparty/Megatron-LM` in place. Carry a change as a pushed commit of the fork.
- Do not `rm -rf` `nemo_experiments/`. It holds checkpoint resume state.
- Run `pipeline_data_prepare.py` before training. Compute `train_iters` from exact token counts.

## Pipelines

Top-level scripts are named `pipeline_<name>_<action>`. Each has a thin `_submit.sbatch` and a launcher.

| Pipeline | Submit | Purpose | Document |
|---|---|---|---|
| env | `pipeline_env_submit.sbatch` | The container and the Slingshot NCCL stack | `docs/isambard/environment-pipeline.md`, `docs/environment.md` |
| training | `pipeline_training_submit.sbatch` | SFT, CPT and pretraining | `docs/isambard/training-pipeline.md` |
| data | `pipeline_data_submit.sbatch` | Download, tokenize, pack | `docs/isambard/data-pipeline.md` |
| checkpoint | `pipeline_checkpoint_submit.sbatch` | Megatron to HF conversion, Hub upload | `docs/isambard/checkpoint-pipeline.md` |
| coherence | `pipeline_coherence_submit.sbatch` | Generation testing | `docs/isambard/coherence-and-evals.md` |

```bash
bash pipeline_env_setup.sh                                                        # one-time environment setup
isambard_sbatch --nodes=16 pipeline_training_submit.sbatch <config.yaml> <nano|super|ultra> <sft|cpt|pretrain>
./pipeline_env_exec.sh "bash $PWD/scripts/run_unit_tests.sh"                      # unit tests, in the container
./pipeline_cluster_submit.sh --nodes=8 --time=00:30:00 pipeline_training_submit.sbatch <config.yaml> <model> <mode>  # Verda
./pipeline_cluster_submit.sh --time=01:00:00 scripts/run_unit_tests.sbatch        # unit tests on Verda, as a job
uv run ruff check --fix . && uv run ruff format .
```

## Where to read

| Topic | Document |
|---|---|
| Cluster, bad nodes, wedged GPU, storage quota | `docs/isambard/cluster.md` |
| Verda: quickstart, cluster, storage, environment, models | `docs/verda/` |
| Profiling, run scoring, loss gates, env overrides, fault tolerance, save policy | `docs/isambard/training-pipeline.md` |
| Nano, Super and Ultra topologies, quickstarts, measured numbers | `docs/isambard/models.md`, `docs/ultra-550b-training-and-conversion.md` |
| Tokenizer per stage, the Base zero-embedding trap, token masking | `docs/isambard/tokenizers.md` |
| NCCL benchmarks for a suspected fabric fault | `docs/isambard/nccl-testing.md` |
| Tests, pre-commit, the Megatron-Core submodule, monitoring | `docs/isambard/development.md` |
| Directory map, disk locations, common pitfalls | `docs/isambard/repository-map.md` |
| Campaigns | each `configs/<campaign>/README.md`, with `campaign-notes.md` beside it where one exists |
| Investigations | `docs/investigations/` |

## Code style

- Ruff formats and lints (119 characters, double quotes). The config is `ruff.toml`.
- Import order: `__future__`, stdlib, third-party, first-party, local.
- Public APIs carry type hints. Write `T | None`, not `Optional[T]`.
- Log with `logging.getLogger(__name__)` or `print_rank_0`, never bare `print()`.

## Claude Code tooling

Hooks from the `geodesic-claude-tooling` submodule live in `.claude/settings.json`, and the enabled quality items
in `.claude/geodesic-config.yaml`. Install or refresh the tooling venv with `bash scripts/install_claude_tooling.sh`
(it carries no torch). The commit review gate is on: `git commit` runs pre-commit on the staged files and is blocked
until the `checklist-reviewer` subagent writes a passing verdict. PR notifications go to `#megatron`.

@.claude/snippets/workflows/branch_then_pr.md
@.claude/snippets/workflows/commit_workflow.md
@.claude/snippets/workflows/plan_exit_protocol.md
@.claude/snippets/workflows/convention_changes.md
@.claude/snippets/workflows/pr_notifications.md
@.claude/snippets/workflows/hpc_node_detection.md
