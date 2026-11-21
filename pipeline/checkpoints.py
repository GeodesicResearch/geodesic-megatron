# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The layout of a Megatron checkpoint directory: its ``iter_*`` saves and their ``run_config.yaml``.

Catalogue:
    LATEST_FILE: the tracker training writes after every rank finishes a save, naming its iteration.
    ITER_DIR: the name of an iteration's save directory, ``iter_`` and seven digits.
    RUN_CONFIG: the file in each save that records the run's settings.
    CheckpointDirectoryMissing: a checkpoint directory is not on disk.
    iteration_number: the iteration an ``iter_XXXXXXX`` directory holds.
    completed_iterations: the iteration directories whose save has finished.
    newest_completed_iteration: the iteration dir a model path names, a run dir's newest completed save.
    train_iters: the run length an iteration's run_config.yaml records.
    main: the command line, ``python -m pipeline.checkpoints completed RUN_DIR``.
    resolve_checkpoint_path: the iteration dir to convert, explicit or latest.
    detect_training_tokenizer: the tokenizer a run trained with, from its run_config.
    detect_model_provider: the model provider class a run built its model with, from its run_config.
    resolve_end_of_turn_id: the id of ``<|im_end|>`` for an export, given or Nemotron's for a NemotronH run.
    is_peft_checkpoint: whether an iteration is an adapter-only PEFT save.
"""

import argparse
import logging
import re
from pathlib import Path
from typing import Any

import yaml

from pipeline.nemotron import IM_END_TOKEN_ID, NEMOTRON_H_PROVIDER


logger = logging.getLogger(__name__)

LATEST_FILE = "latest_checkpointed_iteration.txt"
ITER_DIR = re.compile(r"^iter_(\d{7})$")
RUN_CONFIG = "run_config.yaml"


class CheckpointDirectoryMissing(FileNotFoundError):
    """A checkpoint directory is not on disk (or was removed while it was read)."""


def iteration_number(iter_dir: Path) -> int:
    """The iteration an ``iter_XXXXXXX`` directory holds."""
    match = ITER_DIR.match(iter_dir.name)
    if match is None:
        raise ValueError(f"{iter_dir} is not an iter_XXXXXXX directory")
    return int(match.group(1))


def completed_iterations(checkpoint_dir: Path) -> list[Path]:
    """The ``iter_*`` directories whose save has finished, oldest first.

    A directory above the tracker's value is a save in progress (or one that died mid-write) and
    is not returned. No tracker means no save has completed yet, which is a normal state for a
    stage that has just started — it is reported, not treated as an error.

    A directory that does not exist raises ``CheckpointDirectoryMissing``, and so does one removed
    while it is read: a run's saves are deleted to free space while passes run. The tracker and the
    listing are each read once and the listing decides whether the directory is there, so no
    removal can land between a check and the read it guards.
    """
    try:
        recorded = (checkpoint_dir / LATEST_FILE).read_text()
    except (FileNotFoundError, NotADirectoryError):
        recorded = None
    try:
        children = list(checkpoint_dir.iterdir())
    except (FileNotFoundError, NotADirectoryError) as error:
        raise CheckpointDirectoryMissing(f"checkpoint directory does not exist: {checkpoint_dir}") from error
    if recorded is None:
        logger.info("%s: no %s yet, so no completed checkpoint to archive", checkpoint_dir, LATEST_FILE)
        return []
    latest = int(recorded.strip())
    found = []
    for child in children:
        match = ITER_DIR.match(child.name)
        if match and child.is_dir():
            found.append((int(match.group(1)), child))
    return [path for iteration, path in sorted(found) if iteration <= latest]


def resolve_checkpoint_path(megatron_path: str | Path, iteration: int | None = None) -> tuple[Path, int]:
    """Resolve the checkpoint iteration directory.

    Args:
        megatron_path: Top-level checkpoint directory containing iter_* subdirs.
        iteration: Specific iteration number, or None to use the latest.

    Returns:
        Tuple of (iteration directory path, iteration number).
    """
    base = Path(megatron_path)
    if not base.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {base}")

    if iteration is not None:
        iter_dir = base / f"iter_{iteration:07d}"
        if not iter_dir.exists():
            raise FileNotFoundError(f"Iteration directory not found: {iter_dir}")
        return iter_dir, iteration

    # Try latest_checkpointed_iteration.txt
    latest_file = base / LATEST_FILE
    if latest_file.exists():
        iteration = int(latest_file.read_text().strip())
        iter_dir = base / f"iter_{iteration:07d}"
        if iter_dir.exists():
            return iter_dir, iteration

    # Fall back to scanning iter_* dirs
    iter_dirs = [d for d in base.iterdir() if d.is_dir() and d.name.startswith("iter_")]
    if not iter_dirs:
        raise FileNotFoundError(f"No iter_* directories found in {base}")

    latest = max(iter_dirs, key=lambda d: int(d.name.replace("iter_", "")))
    iteration = int(latest.name.replace("iter_", ""))
    return latest, iteration


def _run_config_value(iter_path: Path, *keys: str) -> Any:
    """The value at the key path ``keys`` of the iteration's ``run_config.yaml``, or None when the file, a key on the
    path or a mapping along it is absent. Reads the file."""
    run_config = iter_path / RUN_CONFIG
    if not run_config.exists():
        return None
    with open(run_config) as f:
        node = yaml.safe_load(f)
    for key in keys:
        if not isinstance(node, dict):
            return None
        node = node.get(key)
    return node


def detect_training_tokenizer(iter_path: Path) -> str | None:
    """Auto-detect the tokenizer the model was actually trained with.

    Reads `tokenizer.tokenizer_model` from `run_config.yaml`. This is the
    canonical source of truth for the chat_template to use at inference,
    because that's the template applied to training data via the bridge's
    `use_hf_tokenizer_chat_template` path. Without this, the converted HF
    dir can end up with a different chat_template than was used at training
    (the legacy fallback grafts from `CHAT_TEMPLATE_SOURCE_MAP`, which points
    at NVIDIA's upstream Instruct template — different in subtle ways from
    the Geodesic-published think/instruct variants).

    Args:
        iter_path: Path to a specific iteration directory.

    Returns:
        Training tokenizer HF model ID (e.g. ``geodesic-research/nemotron-
        instruct-tokenizer``), or None if not specified in run_config.yaml.
    """
    tokenizer_model = _run_config_value(iter_path, "tokenizer", "tokenizer_model")
    return str(tokenizer_model) if tokenizer_model else None


def detect_model_provider(iter_path: Path) -> str | None:
    """The ``_target_`` of the ``model`` section of the iteration's ``run_config.yaml``, the provider class the run
    built its model with, or None when the file, the section or the target is absent."""
    target = _run_config_value(iter_path, "model", "_target_")
    return str(target) if target else None


def resolve_end_of_turn_id(iter_path: Path, requested: int | None) -> int:
    """The id of ``<|im_end|>`` that an export of the iteration adds to its end-of-generation ids: ``requested`` when
    given, else ``IM_END_TOKEN_ID``, which is Nemotron's, for a run whose model provider is NemotronH's.

    Raises:
        SystemExit: no id is requested and the run's model provider is not NemotronH's, or its run_config.yaml names
            none. Nemotron's id would be another token in another vocabulary.
    """
    if requested is not None:
        return requested
    provider = detect_model_provider(iter_path)
    if provider == NEMOTRON_H_PROVIDER:
        return IM_END_TOKEN_ID
    raise SystemExit(
        f"{iter_path / RUN_CONFIG} names the model provider {provider!r}, not NemotronH's ({NEMOTRON_H_PROVIDER}), so "
        f"the default end-of-turn id {IM_END_TOKEN_ID} (Nemotron's <|im_end|>) does not apply: pass --end-of-turn-id "
        "with the id of <|im_end|> in the model's vocabulary (Olmo 3: 100265)."
    )


def is_peft_checkpoint(iter_path: Path) -> bool:
    """Whether the iteration's ``run_config.yaml`` names a ``peft`` section (an adapter-only checkpoint)."""
    return bool(_run_config_value(iter_path, "peft"))


def newest_completed_iteration(model_path: str) -> str:
    """``model_path`` when it is an ``iter_XXXXXXX`` dir, else the newest completed save of the run dir it names.

    Unlike ``resolve_checkpoint_path``, which falls back to the largest ``iter_*`` when the tracker names none, a save
    above the tracker (still being written) is never chosen (``completed_iterations``).

    Raises:
        CheckpointDirectoryMissing: ``model_path`` does not exist.
        SystemExit: the run has no completed save.
    """
    path = Path(model_path)
    if ITER_DIR.match(path.name):
        return model_path
    completed = completed_iterations(path)
    if not completed:
        raise SystemExit(f"{model_path} has no completed save (no tracker, or none of its iter_* dirs is complete)")
    return str(completed[-1])


def train_iters(iter_dir: Path) -> int | None:
    """``train.train_iters`` of ``iter_dir/run_config.yaml``; None without the file or the key."""
    value = _run_config_value(iter_dir, "train", "train_iters")
    return None if value is None else int(value)


def main(argv: list[str] | None = None) -> None:
    """``completed RUN_DIR``: print one ``<iteration> <train_iters>`` line per completed save of RUN_DIR, oldest first.

    ``train_iters`` is that save's own (``train_iters``), or ``-`` when it records none. A save above the tracker is
    still being written and is not printed (``completed_iterations``).

    Raises:
        CheckpointDirectoryMissing: RUN_DIR does not exist.
    """
    parser = argparse.ArgumentParser(prog="python -m pipeline.checkpoints", description="Megatron checkpoint dirs")
    commands = parser.add_subparsers(dest="command", required=True)
    completed = commands.add_parser("completed", help="the completed saves of a run, with their train_iters")
    completed.add_argument("run_dir", type=Path)
    args = parser.parse_args(argv)
    for iter_dir in completed_iterations(args.run_dir):
        length = train_iters(iter_dir)
        print(iteration_number(iter_dir), "-" if length is None else length)


if __name__ == "__main__":
    main()
