# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""Write the run configs and launch scripts of the IPT and SPT pretraining arms from their spec and their corpora.

The two arms train the same model from scratch for the same number of steps on the same schedule. They differ in the
corpora they read: each arm reads one tokenised corpus for every group of the build, weighted by token count. The spec
(``arms.yaml``) fixes the settings that the arms share and where their corpora are. The generator reads each corpus's token count from the provenance file that
``scripts/data/count_idx_tokens.py`` wrote beside it, and decides what a spec cannot:

* ``train_iters``: the smallest number of steps that reaches ``target_tokens``. Both arms take the same number, and the
  generator refuses a spec whose corpora hold fewer samples than that, because a shorter run is a decision for the spec.
* the blend: whole-sample counts that sum to ``train_iters * global_batch_size``, in proportion to the corpora's tokens,
  each one a count that Megatron builds exactly (``scripts/data/blend_samples.py``) and none above the samples its corpus
  holds, so that no corpus is read twice.
* ``checkpoint.save_steps``: every ``save_every_tokens`` tokens, and the last step.

The generator itself fixes how a run saves and reads: the ``single`` sampler, no end-of-run save, a save with optimizer
state at each of the ``save_steps`` and none pruned, and a resume from the run's own save.

It writes ``<arm>_pretrain.yaml``, an overlay of the spec's ``run_base_config``, and ``<arm>_launch.sh`` beside the spec. The
launch script submits the chain of jobs that trains the arm, behind a held gate job that a person releases. The jobs
share one name and run one at a time, and each resumes from the save of the one before.

    python configs/ipt_spt/generate_arm_configs.py configs/ipt_spt/arms.yaml

Catalogue:
    ArmsError: a spec or a corpus that the generator refuses.
    load_spec: read and check the spec.
    corpus_prefixes: the corpora of one arm, in group order.
    corpus_tokens: the token count that a corpus's provenance records.
    plan_iterations: the steps both arms take.
    arm_blend: the interleaved weights and prefixes of one arm's blend.
    save_steps: the steps at which a run saves.
    arm_config: the run config of one arm.
    generated_note: the line that heads a generated file.
    launch_script: the script that submits one arm's chain.
    generate: every file the spec asks for, as text by path.
    main: write them.
"""

from __future__ import annotations

import argparse
import json
import math
import stat
import sys
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from scripts.data.blend_samples import apportion, samples_in, settle_on_megatron_targets  # noqa: E402
from scripts.mapping_keys import require_keys  # noqa: E402
from scripts.mapping_values import (  # noqa: E402
    require_absolute_path,
    require_int,
    require_number,
    require_string,
    require_walltime,
)


PROVENANCE_SUFFIX = ".provenance.json"
CONFIG_NAME = "{arm}_pretrain.yaml"
LAUNCH_NAME = "{arm}_launch.sh"
# The longest job that the queue accepts, and the minutes that a job needs after its exit time to save and end.
MAX_WALL_MINUTES = 24 * 60
SAVE_AND_EXIT_MINUTES = 30
# The decay styles that the spec may name. Megatron reads the style only after the warm-up, so a style that it does not
# know would stop a run at its first step after that, and a decaying style needs a minimum learning rate that the spec
# does not state.
DECAY_STYLES = ("constant",)

SPEC_KEYS = {
    "run_base_config",
    "tokenizer_model",
    "sequence_length",
    "global_batch_size",
    "micro_batch_size",
    "target_tokens",
    "learning_rate",
    "lr_decay_style",
    "warmup_iterations",
    "weight_decay",
    "save_every_tokens",
    "exit_duration_in_mins",
    "launch",
    "wandb",
    "corpora_root",
    "corpus_template",
    "groups",
    "checkpoint_root",
    "cache_root",
    "arms",
}
LAUNCH_KEYS = {"nodes", "gpus_per_node", "data_parallel_size", "links", "time_limit", "model", "mode", "env_file"}
WANDB_KEYS = {"entity", "project", "save_dir"}


class ArmsError(ValueError):
    """A spec or a corpus that the generator refuses."""


def load_spec(path: Path) -> dict[str, Any]:
    """The spec at ``path``, checked: every key present and none unknown, the numbers in range, the paths absolute, the
    groups distinct, and the launch layout equal to the data-parallel size it implies.

    Raises:
        ArmsError: A fault in the spec, named with its place.
    """
    spec = yaml.safe_load(path.read_text())
    where = str(path)
    require_keys(spec, where, SPEC_KEYS, error=ArmsError)
    require_keys(spec["launch"], f"{where}: launch", LAUNCH_KEYS, error=ArmsError)
    require_keys(spec["wandb"], f"{where}: wandb", WANDB_KEYS, error=ArmsError)
    for key in ("sequence_length", "global_batch_size", "micro_batch_size", "target_tokens", "save_every_tokens"):
        require_int(spec[key], f"{where}: {key}", 1, error=ArmsError)
    require_int(spec["warmup_iterations"], f"{where}: warmup_iterations", 0, error=ArmsError)
    require_int(spec["exit_duration_in_mins"], f"{where}: exit_duration_in_mins", 1, error=ArmsError)
    require_number(spec["learning_rate"], f"{where}: learning_rate", 0, inclusive=False, error=ArmsError)
    style = require_string(spec["lr_decay_style"], f"{where}: lr_decay_style", error=ArmsError)
    if style not in DECAY_STYLES:
        raise ArmsError(
            f"{where}: lr_decay_style {style!r} is not supported, only {list(DECAY_STYLES)}: a decaying style needs a "
            "minimum learning rate, which the spec does not state"
        )
    require_number(spec["weight_decay"], f"{where}: weight_decay", 0, inclusive=True, error=ArmsError)
    for key in ("nodes", "gpus_per_node", "data_parallel_size", "links"):
        require_int(spec["launch"][key], f"{where}: launch.{key}", 1, error=ArmsError)
    for key in ("tokenizer_model", "corpora_root", "checkpoint_root", "cache_root"):
        require_absolute_path(spec[key], f"{where}: {key}", error=ArmsError)
    require_absolute_path(spec["wandb"]["save_dir"], f"{where}: wandb.save_dir", error=ArmsError)
    for key in ("run_base_config", "corpus_template"):
        require_string(spec[key], f"{where}: {key}", error=ArmsError)
    limit = _wall_minutes(spec["launch"]["time_limit"], f"{where}: launch.time_limit")
    if limit > MAX_WALL_MINUTES:
        raise ArmsError(
            f"{where}: launch.time_limit {spec['launch']['time_limit']} is longer than the queue's {MAX_WALL_MINUTES} minutes"
        )
    if spec["exit_duration_in_mins"] + SAVE_AND_EXIT_MINUTES > limit:
        raise ArmsError(
            f"{where}: a job that exits {spec['exit_duration_in_mins']} minutes after it starts needs {SAVE_AND_EXIT_MINUTES} "
            f"more to save and end, and launch.time_limit {spec['launch']['time_limit']} allows {limit}"
        )
    if not spec["groups"] or len(set(spec["groups"])) != len(spec["groups"]):
        raise ArmsError(f"{where}: groups must be a non-empty list of distinct names, got {spec['groups']}")
    if not spec["arms"]:
        raise ArmsError(f"{where}: arms names no arm")
    runs = [
        require_keys(arm, f"{where}: arms.{name}", {"run"}, error=ArmsError)["run"]
        for name, arm in spec["arms"].items()
    ]
    if len(set(runs)) != len(runs):
        raise ArmsError(f"{where}: two arms share a run name: {runs}")
    launch = spec["launch"]
    gpus = launch["nodes"] * launch["gpus_per_node"]
    if gpus % launch["data_parallel_size"]:
        raise ArmsError(
            f"{where}: launch.data_parallel_size {launch['data_parallel_size']} does not divide the {gpus} GPUs"
        )
    return spec


def _wall_minutes(time_limit: Any, where: str) -> int:
    """The minutes of a SLURM time limit ``[D-]HH:MM:00``, which must have no seconds."""
    text = require_walltime(time_limit, where, error=ArmsError)
    days, _, clock = text.rpartition("-")
    hours, minutes, seconds = (int(part) for part in clock.split(":"))
    if seconds:
        raise ArmsError(f"{where} must be HH:MM:00, not {time_limit!r}")
    return (int(days or 0) * 24 + hours) * 60 + minutes


def corpus_prefixes(spec: dict[str, Any], arm: str) -> list[str]:
    """The extension-less prefix of each corpus of ``arm``, in the order of the spec's groups."""
    return [
        str(Path(spec["corpora_root"]) / spec["corpus_template"].format(group=group, arm=arm))
        for group in spec["groups"]
    ]


def corpus_tokens(prefix: str) -> int:
    """The tokens that the provenance beside the corpus at ``prefix`` records, separators included.

    Raises:
        ArmsError: The provenance file is missing or does not state a token total.
    """
    path = Path(prefix + PROVENANCE_SUFFIX)
    if not path.is_file():
        raise ArmsError(f"the corpus {prefix} has no provenance file {path}: it is not tokenised yet")
    total = json.loads(path.read_text()).get("totals", {}).get("total_tokens")
    return require_int(total, f"{path}: totals.total_tokens", 1, error=ArmsError)


def plan_iterations(spec: dict[str, Any], tokens_by_arm: dict[str, list[int]]) -> int:
    """The steps both arms take: the fewest that reach ``target_tokens``.

    An arm's blend reads each sample of its corpora at most once (``samples_in``), so an arm holds samples for a number of
    steps.

    Raises:
        ArmsError: An arm holds samples for fewer steps than the target needs. The spec's ``target_tokens`` is the place
            to ask for a shorter run.
    """
    per_step = spec["sequence_length"] * spec["global_batch_size"]
    wanted = math.ceil(spec["target_tokens"] / per_step)
    held = {
        arm: sum(samples_in(tokens, spec["sequence_length"]) for tokens in counts) // spec["global_batch_size"]
        for arm, counts in tokens_by_arm.items()
    }
    short = {arm: steps for arm, steps in held.items() if steps < wanted}
    if short:
        raise ArmsError(
            f"target_tokens {spec['target_tokens']} needs {wanted} steps of {per_step} tokens, and the corpora of {short} "
            "hold samples for fewer (steps by arm). Lower target_tokens in the spec to train fewer steps"
        )
    return wanted


def arm_blend(spec: dict[str, Any], prefixes: list[str], tokens: list[int], train_iters: int) -> list[str]:
    """The interleaved ``[weight, prefix, ...]`` blend of one arm, weighted in whole samples summing to
    ``train_iters * global_batch_size``.

    Raises:
        ArmsError: A corpus would be read past its own samples.
    """
    total = train_iters * spec["global_batch_size"]
    counts = settle_on_megatron_targets(apportion(total, tokens), total)
    blend: list[str] = []
    for prefix, count, held_tokens in zip(prefixes, counts, tokens):
        held = samples_in(held_tokens, spec["sequence_length"])
        if count > held:
            raise ArmsError(
                f"the blend draws {count} samples from {prefix}, which holds {held}, so it would be read twice"
            )
        blend += [str(count), prefix]
    return blend


def save_steps(spec: dict[str, Any], train_iters: int) -> list[int]:
    """The steps that follow every ``save_every_tokens`` tokens, rounded up to a step, then the last step."""
    per_step = spec["sequence_length"] * spec["global_batch_size"]
    steps = []
    point = 1
    while math.ceil(point * spec["save_every_tokens"] / per_step) < train_iters:
        steps.append(math.ceil(point * spec["save_every_tokens"] / per_step))
        point += 1
    return [*steps, train_iters]


def arm_config(spec: dict[str, Any], arm: str, train_iters: int, blend: list[str]) -> dict[str, Any]:
    """The run config of ``arm``: an overlay of the spec's ``run_base_config`` with the run's own settings."""
    run = spec["arms"][arm]["run"]
    save = str(Path(spec["checkpoint_root"]) / run)
    launch = spec["launch"]
    return {
        "base_config": spec["run_base_config"],
        "launch_width": {
            "nodes": launch["nodes"],
            "gpus_per_node": launch["gpus_per_node"],
            "data_parallel_size": launch["data_parallel_size"],
        },
        "tokenizer": {"tokenizer_model": spec["tokenizer_model"]},
        "dataset": {
            "seq_length": spec["sequence_length"],
            "data_path": blend,
            "path_to_cache": str(Path(spec["cache_root"]) / run),
            "bin_idx_dataloader_type": "single",
        },
        "model": {"seq_length": spec["sequence_length"]},
        "train": {
            "train_iters": train_iters,
            "global_batch_size": spec["global_batch_size"],
            "micro_batch_size": spec["micro_batch_size"],
            "exit_duration_in_mins": spec["exit_duration_in_mins"],
            "exit_interval": None,
            "exit_signal_handler": False,
        },
        "optimizer": {"lr": spec["learning_rate"]},
        "scheduler": {
            "lr_decay_style": spec["lr_decay_style"],
            "lr_warmup_iters": spec["warmup_iterations"],
            "lr_warmup_fraction": None,
            "lr_decay_iters": train_iters,
            "start_weight_decay": spec["weight_decay"],
            "end_weight_decay": spec["weight_decay"],
        },
        "checkpoint": {
            "save": save,
            "load": save,
            "save_interval": 0,
            "save_steps": save_steps(spec, train_iters),
            "save_optim": True,
            "save_rng": True,
            "most_recent_k": -1,
        },
        "logger": {
            "wandb_entity": spec["wandb"]["entity"],
            "wandb_project": spec["wandb"]["project"],
            "wandb_exp_name": run,
            "wandb_save_dir": spec["wandb"]["save_dir"],
            "tensorboard_dir": None,
        },
    }


def _repo_path(path: Path) -> str:
    """``path`` relative to the repository root when it lies inside the repository, and absolute otherwise."""
    resolved = path.resolve()
    return str(resolved.relative_to(REPO_ROOT)) if REPO_ROOT in resolved.parents else str(resolved)


def generated_note(spec_path: Path) -> str:
    """The line that heads a generated file: the generator, the spec and what else it read."""
    generator = _repo_path(Path(__file__))
    return f"GENERATED by {generator} from {_repo_path(spec_path)} and the provenance of the corpora. Do not edit."


def launch_script(spec: dict[str, Any], arm: str, spec_path: Path) -> str:
    """The shell script that submits the chain of jobs of ``arm`` from a frozen copy of the committed code.

    A gate job of one node is submitted held, and the first link waits for it, so no link starts until a person runs
    ``scontrol release`` on the gate. Every link has the arm's run name, and the links run one at a time (``singleton``).
    The jobs run the config that ``generate`` writes beside ``spec_path``: from the frozen copy when the spec lies in the
    repository, and from where it was written otherwise.
    """
    run = spec["arms"][arm]["run"]
    launch = spec["launch"]
    config_path = spec_path.parent / CONFIG_NAME.format(arm=arm)
    config = (
        f"$FROZEN/{_repo_path(config_path)}"
        if REPO_ROOT in config_path.resolve().parents
        else str(config_path.resolve())
    )
    return f"""#!/bin/bash
# {generated_note(spec_path)}
# Submits the {launch["links"]} jobs that train the {arm} arm ({run}), behind a gate job that is held. Each job resumes from the
# save of the one before, and each exits {spec["exit_duration_in_mins"]} minutes after it starts. No job starts until the gate is released:
#   scontrol release <gate job id>
# Usage: FROZEN=<frozen copy of the committed code> bash {LAUNCH_NAME.format(arm=arm)}
set -euo pipefail
: "${{FROZEN:?FROZEN must name the frozen copy of the committed code}}"
test -f "$FROZEN/REVISION"
cd "$FROZEN"
export TMPDIR=/tmp
export ISAMBARD_ENV_OVERRIDES="$FROZEN/{launch["env_file"]}"
GATE=$(isambard_sbatch --hold --nodes=1 --time=00:05:00 --job-name={run}-gate --wrap true | grep -o '[0-9]*$')
echo "gate job $GATE is held; release it with: scontrol release $GATE"
for link in $(seq 1 {launch["links"]}); do
    dependency=singleton
    if [ "$link" -eq 1 ]; then dependency="afterok:$GATE,singleton"; fi
    isambard_sbatch --nodes={launch["nodes"]} --time={launch["time_limit"]} --job-name={run} --dependency="$dependency" \\
        pipeline_training_submit.sbatch "{config}" {launch["model"]} {launch["mode"]} --disable-ft
done
"""


def generate(spec_path: Path) -> dict[Path, str]:
    """Every file that the spec at ``spec_path`` asks for, as text by path (beside the spec).

    Reads the spec and the provenance beside each corpus. Raises ``ArmsError`` for a fault in the spec or a corpus that
    is not tokenised.
    """
    spec = load_spec(spec_path)
    prefixes = {arm: corpus_prefixes(spec, arm) for arm in spec["arms"]}
    tokens = {arm: [corpus_tokens(prefix) for prefix in prefixes[arm]] for arm in spec["arms"]}
    train_iters = plan_iterations(spec, tokens)
    files: dict[Path, str] = {}
    for arm in spec["arms"]:
        blend = arm_blend(spec, prefixes[arm], tokens[arm], train_iters)
        body = yaml.safe_dump(arm_config(spec, arm, train_iters, blend), sort_keys=False)
        files[spec_path.parent / CONFIG_NAME.format(arm=arm)] = f"# {generated_note(spec_path)}\n{body}"
        files[spec_path.parent / LAUNCH_NAME.format(arm=arm)] = launch_script(spec, arm, spec_path)
    return files


def main(argv: list[str] | None = None) -> int:
    """Write the files that ``generate`` returns for the spec named on the command line; print what was written.

    Side effects: writes the run configs and launch scripts beside the spec, and marks each launch script executable.
    """
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("spec", type=Path, help="the spec, arms.yaml")
    args = parser.parse_args(argv)
    for path, text in generate(args.spec).items():
        path.write_text(text)
        if path.suffix == ".sh":
            path.chmod(path.stat().st_mode | stat.S_IXUSR)
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
