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
    resolve_checkpoint_path: the iteration dir to convert, explicit or latest.
    detect_training_tokenizer: the tokenizer a run trained with, from its run_config.
    is_peft_checkpoint: whether an iteration is an adapter-only PEFT save.
"""

from pathlib import Path

import yaml


def resolve_checkpoint_path(megatron_path: str, iteration: int | None = None) -> tuple[Path, int]:
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
    latest_file = base / "latest_checkpointed_iteration.txt"
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
    run_config = iter_path / "run_config.yaml"
    if not run_config.exists():
        return None
    with open(run_config) as f:
        config = yaml.safe_load(f)
    tokenizer_model = config.get("tokenizer", {}).get("tokenizer_model")
    if not tokenizer_model:
        return None
    return str(tokenizer_model)


def is_peft_checkpoint(iter_path: Path) -> bool:
    """Whether the iteration's ``run_config.yaml`` names a ``peft`` section (an adapter-only checkpoint)."""
    run_config = iter_path / "run_config.yaml"
    if not run_config.exists():
        return False
    with open(run_config) as f:
        return bool((yaml.safe_load(f) or {}).get("peft"))
