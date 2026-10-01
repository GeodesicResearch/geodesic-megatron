# Training Scripts

Generic launcher and training scripts that work with any GPT-based model family (e.g. Deepseek, Llama, Gemma, Qwen, GPT, etc.).

## Overview

These scripts provide a generic interface for training GPT-based models in Megatron Bridge:

- `run_recipe.py` - Generic pretraining/finetuning for GPT- and Mamba-based models.
- `launch_with_nemo_run.py` - NeMo-Run launcher (local or Slurm)
- `launch_with_sbatch.sh` - Direct sbatch launcher

The launchers dynamically import recipes from `megatron.bridge.recipes`, apply user-provided overrides to the configuration, then begin training.

## Diagnostics

- `dump_hung_ranks.sh <jobid>` - Record the state and kernel wait channel of every rank and of
  every helper process it forked (`processes.<host>`), then capture each one's Python and native
  stacks (py-spy, attached from outside the rank; the rank's own to `rank_<rank>.stack`, a
  helper's to `rank_<rank>.child-<pid>.stack`, a process in an uninterruptible wait skipped) and,
  where the rank runs a NCCL watchdog, its NCCL flight recorder, for a job started by
  `pipeline_training_launch.sh`, into `<log-dir>/nccl_trace/<jobid>/`. Run it on a job that has
  stopped iterating BEFORE cancelling it; a `D` or `I` state with a Lustre wait channel in the
  table means the run is waiting on storage, not wedged in a collective. Needs py-spy on the host
  PATH (`python3 -m pip install --user py-spy`) or its path in `PY_SPY`.
- `nvlink_health.py --status-dir DIR --gpus-per-node N --links-per-gpu L --select K ...` - Judge
  each node from its `nvidia-smi nvlink --status` output (one `<host>.txt` per node, written by a
  one-task-per-node srun): healthy when it reports N GPUs with L active links each. Writes a JSON
  report and the first K healthy nodes as a `--nodelist`, prints `UNHEALTHY <host> <reason>` for the
  rest, and exits 1 when fewer than K are healthy. Count links rather than read `nvidia-smi topo -m`,
  which shows the configured topology even with links down; the HybridEP dispatcher aborts on a node
  with one dead link. The v2e2e probe (`configs/control_pretraining/30b_filtered_gpt55_4plus_v2e2e/probe/probe.sbatch`)
  runs it before its launches.

## Launch environment

- `launch_environment.py` - Exit 1, naming them, when the environment holds a launch setting: an
  `ISAMBARD_*`, `TRAIN_*` or `GEODESIC_CONTAINER_*` variable other than the submission wrapper's
  `ISAMBARD_SBATCH_*` and the tunnel's `ISAMBARD_TUNNEL_*`. `isambard_sbatch` exports the submitting shell
  to the job, so such a variable changes the run with no config naming it. Run it before a submission
  whose posture must be exactly its config and its `ISAMBARD_ENV_OVERRIDES` file
  (`python3 scripts/training/launch_environment.py && isambard_sbatch ...`); the v2e2e probes and
  `configs/control_pretraining/submit_chain_link.py` apply it themselves. It runs under the node's system
  Python 3.6 as well as the container's.

- `launcher_source.py` - Runs functions of `pipeline_training_launch.sh` as the launcher runs them, lifted by
  name (the launcher cannot be sourced whole): `env_override_entries(path)` returns the KEY=VALUE entries the
  launcher's `ISAMBARD_ENV_OVERRIDES` parser takes from a file, and raises with the launcher's message on a file
  it refuses. The stage watch (`scripts/telemetry/run_watch.py`) and the launcher's tests read override files
  through it.

- `stage_guard.py` - Guards a training stage while it trains: every `interval_seconds` it finds the stage's
  started segments by job name (`sacct`), runs `scripts/telemetry/run_watch.py` on them inside the container and
  appends the evaluation to its record. A stop (the watch's exit 1 beside its summary line) cancels every live job
  of the stage by ID, the successors pending on its singleton dependency included (`squeue`, since `sacct` lists no
  job that has not started), and exits 1. A tick the watch left NOT EVALUATED, or one that could not be evaluated
  at all (the watch could not run or outlived the config's `command_timeout_seconds`, `sacct` could not list the
  jobs), does the same and exits 3 once the stage has reached the config's `hold.from_iteration` while a loss gate
  is still undecided; anything else alerts and the guard keeps going. It cancels before it writes the tick to the
  record. A gate the watch has passed is handed to later ticks as `--decided GATE=LOG`, so it is
  not evaluated again while that log covers its range, and a guard started again takes as decided the gates its
  record shows the watch passing, so restarting it inside the hold window does not re-read a reference for a gate
  already passed. The record opens with the guard config and the watch spec, each with its sha256, the code revision
  (`scripts/telemetry/code_revision.py`: the checkout's commit, or a frozen copy's `REVISION`) and the gates decided
  from the record, and every tick names the spec's sha256. A tick that could not be evaluated never counts as a
  stop; a cancellation that fails exits 4, and a failure of the guard itself exits 5 and is written to the record.
  It is started once the stage's newest segment has logged its first iteration, so finding no started segment is
  such a failure. It runs on the tunnel under the host Python (SLURM's commands do not exist in the container), so
  it stays Python 3.6-compatible, from the frozen copy the stage was submitted from:
  `setsid nohup python3 scripts/training/stage_guard.py --config <guard.yaml> >/dev/null 2>&1 &`. The v2e2e
  arm's `guard_pretrain.yaml` and `guard_midtrain.yaml` are its configs.

## Probe jobs

- `probe_job.sh` - The steps a production-width probe job is built from, sourced by its sbatch after it sets
  `REPO_DIR`, `OUT`, `NODES`, `GPUS`, `LINKS_PER_GPU`, `HF_MODEL`, `MODEL` and `MODE`: the start checks (the
  account's node cap, a frozen copy with a `REVISION` file, an absent scratch directory, no inherited launch
  setting), the NVLink sweep and selection of `NODES` healthy nodes (registering the rest as bad), each launch
  under its own time limit, `score_run.py` scoring, a handoff's log evidence, the `loss_parity.py` band, and a
  `steps.tsv` record of every step (`note` reports, `record` and `gate` decide the job's exit status). The v2e2e
  probes (`configs/control_pretraining/30b_filtered_gpt55_4plus_v2e2e/probe/`) are built from it.

## Config composition

- `config_compose.py` - `load_composed_yaml(path)` reads a training YAML through its top-level
  `base_config:` key, so an overlay holds only the fields it changes. The base path is resolved
  against the overlay's own directory (or taken as written when absolute) and may itself name a
  base. Mappings deep-merge; a list, a scalar or an explicit `null` replaces the base value, which
  is what `OmegaConf.merge` does. Scalars are read as `OmegaConf.load` reads them. A cycle, a
  missing base or a non-mapping file raises. `pipeline_training_run.py`, the FLOPs estimator and
  the config tests all read `--config-file` YAMLs through it. It needs only PyYAML, so host-side
  tools can import it. `deep_merge(base, overlay)` is that merge on its own, and
  `parse_yaml_mapping(text, source)` reads one document with the same scalar rules.

## Quick Start

For the end-to-end overview of how recipes are structured, overridden, and launched, see the official [Using Recipes guide](https://docs.nvidia.com/nemo/megatron-bridge/latest/recipe-usage.html).

### Pretrain (single-GPU)

```bash
uv run python run_recipe.py --recipe llama32_1b_pretrain_config
```

### Pretrain (multi-GPU)

```bash
uv run torchrun --nproc_per_node=8 run_recipe.py --recipe llama32_1b_pretrain_config
```

### Finetune

```bash
uv run torchrun --nproc_per_node=8 run_recipe.py --recipe llama32_1b_sft_config
```

## Usage with Different Models

Same scripts work across all model families:

```bash
# Llama
uv run torchrun --nproc_per_node=8 run_recipe.py --recipe llama32_1b_pretrain_config

# Gemma
uv run torchrun --nproc_per_node=8 run_recipe.py --recipe gemma3_1b_pretrain_config

# Qwen
uv run torchrun --nproc_per_node=8 run_recipe.py --recipe qwen3_8b_pretrain_config

# GPT
uv run torchrun --nproc_per_node=8 run_recipe.py --recipe gpt_126m_pretrain_config
```

## CLI Overrides

Override any config field using dot notation:

```bash
uv run torchrun --nproc_per_node=8 run_recipe.py \
    --recipe llama32_1b_pretrain_config \
    train.train_iters=5000 \
    optimizer.lr=0.0002 \
    model.tensor_model_parallel_size=2
```

The first part before the dot specifies which ConfigContainer subconfig to override (e.g., `train`, `model`, `optimizer`), and the part after specifies the field.

Configuration priority:
1. CLI overrides (highest)
2. Recipe defaults (lowest)

Mode is inferred from the recipe name. If your recipe name doesn't include
`pretrain`, `finetune`, `sft`, or `peft`, pass `--mode` explicitly.

## Step Function Selection

Use `--step_func` to control the step function used during training. Available options:

- `gpt_step` - Text-only models (default)
- `vlm_step` - Vision-language models
- `llava_step` - LLaVA models

```bash
uv run torchrun --nproc_per_node=8 run_recipe.py \
    --recipe qwen25_vl_pretrain_config \
    --step_func vlm_step
```

## Multi-Node and Distributed Training

### Option 1: NeMo-Run

Prerequisites:

```bash
pip install nemo-run
```

#### Test Locally First

Before launching on Slurm, test your configuration locally:

```bash
python launch_with_nemo_run.py \
    --local \
    --script run_recipe.py \
    --recipe llama32_1b_pretrain_config \
    --devices 2 \
    --dry-run \
    train.train_iters=10
```

This uses `LocalExecutor` with torchrun for single-node testing. Include `--dry-run` to confirm the composed nemo-run command before actually launching it.

#### Launch on Slurm

Once tested, scale to Slurm by removing `--local` and adding Slurm parameters:

```bash
# From the cluster (LocalTunnel)
python launch_with_nemo_run.py \
    --script run_recipe.py \
    --recipe llama32_1b_pretrain_config \
    --nodes 2 \
    --devices 8 \
    --partition gpu \
    --account my_account

# From your local machine (SSHTunnel)
python launch_with_nemo_run.py \
    --script run_recipe.py \
    --recipe llama32_1b_pretrain_config \
    --nodes 2 \
    --devices 8 \
    --partition gpu \
    --account my_account \
    --ssh-tunnel \
    --host my-cluster.example.com \
    --user myusername \
    --remote-job-dir /home/myusername/nemo-runs
```

#### With Containers

When using containers, scripts are automatically packaged using `PatternPackager`:

```bash
python launch_with_nemo_run.py \
    --script run_recipe.py \
    --recipe qwen3_8b_pretrain_config \
    --nodes 4 \
    --devices 8 \
    --partition gpu \
    --account my_account \
    --container-image /path/to/container.sqsh \
    --mount /data:/data
```

> **Note:** PatternPackager only includes `scripts/training/*.py`. Local changes in
> `src/megatron/bridge/` stay on your workstation unless you mount the repo into
> the container.

```bash
python launch_with_nemo_run.py \
    --script run_recipe.py \
    --recipe llama32_1b_pretrain_config \
    --nodes 2 \
    --partition gpu \
    --account my_account \
    --container-image /path/to/container.sqsh \
    --mount /path/to/your/Megatron-Bridge:/opt/Megatron-Bridge \
    train.train_iters=10
```

Mounting onto `/opt/Megatron-Bridge` shadows the container's built-in source so
your edited `src/megatron/bridge/` files are used while packaged scripts still
run from the container workspace.

For git-based packaging:

```bash
python launch_with_nemo_run.py \
    --script run_recipe.py \
    --recipe llama3_8b_pretrain_config \
    --nodes 2 \
    --partition gpu \
    --account my_account \
    --container-image /path/to/container.sqsh \
    --packager git
```

#### Fault-Tolerant Training

Use the fault-tolerant launcher for better resiliency:

```bash
python launch_with_nemo_run.py \
    --script run_recipe.py \
    --recipe llama32_1b_pretrain_config \
    --launcher ft \
    --nodes 2 \
    --partition gpu \
    --account my_account
```

### Option 2: Direct sbatch

For traditional HPC workflows without NeMo-Run, use the `launch_with_sbatch.sh` script.

Edit the configuration section in `launch_with_sbatch.sh`:

```bash
# Training script to run
TRAINING_SCRIPT="run_recipe.py"

# Recipe name
RECIPE="llama32_1b_pretrain_config"

# Step function (controls the step function: gpt_step, vlm_step, or llava_step)
STEP_TYPE="gpt_step"

# Optional: CLI overrides
CLI_OVERRIDES="train.train_iters=5000 optimizer.lr=0.0003"

# Optional: Container settings
CONTAINER_IMAGE="/path/to/container.sqsh"
CONTAINER_MOUNTS="/data:/data /model:/model"
```

Also configure the SBATCH directives at the top of the file:

```bash
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --partition=gpu
#SBATCH --account=my_account
#SBATCH --time=04:00:00
```

Then submit:

```bash
sbatch launch_with_sbatch.sh
```

The script automatically:
- Sets up multi-node torchrun with correct SLURM environment variables
- Passes recipe and CLI override arguments to the training script
- Handles container execution (if specified)
- Applies container mounts

## Recipe Arguments

Generic scripts call recipes with no arguments passed to the recipe function.

All customization happens through CLI overrides after the config is built.

If you need to pass arguments to the recipe constructor itself (e.g., custom parallelism at recipe build time), use model-specific examples or create a custom script.
