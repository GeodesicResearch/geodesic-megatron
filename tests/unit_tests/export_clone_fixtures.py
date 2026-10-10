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
"""Tiny Megatron checkpoint trees for the tests of the torch_grouped export repair (test_export_clone.py) and of the
exporter that applies it (test_checkpoint_export_repair.py)."""

from __future__ import annotations

from pathlib import Path

from scripts.checkpoint import export_clone


OLD_TARGET, NEW_TARGET = export_clone.RUN_CONFIG_EDITS[0]
OLD_IMPL, NEW_IMPL = export_clone.RUN_CONFIG_EDITS[1]

TE_GROUPED_RUN_CONFIG = f"model:\n  mamba_stack_spec:\n    _target_: {NEW_TARGET}\n  {NEW_IMPL}\n"
GPT_RUN_CONFIG = "model:\n  _target_: megatron.bridge.models.gpt_provider.GPTModelProvider\n  num_layers: 2\n"


def torch_grouped_run_config(save: Path) -> str:
    """A run_config as a torch_grouped hybrid run writes it: the unimportable closure and the expert implementation,
    beside the identity fields an export's consumers read back from hf/megatron_run_config.yaml."""
    return (
        "checkpoint:\n"
        f"  save: {save}\n"
        "  pretrained_checkpoint: /parent/iter_0000000\n"
        "logger:\n"
        "  wandb_exp_name: arm_masked\n"
        "model:\n"
        "  mamba_stack_spec:\n"
        f"    _target_: {OLD_TARGET}\n"
        f"  {OLD_IMPL}\n"
        "token_masking:\n"
        "  enabled: true\n"
        "  token_ids:\n"
        "  - 131072\n"
    )


def make_checkpoint(root: Path, iterations: list[int], tracker: int | None, run_config: str | None) -> Path:
    """A Megatron save directory: iter_* dirs holding torch_dist-like files and the given run_config."""
    root.mkdir(parents=True, exist_ok=True)
    for iteration in iterations:
        iter_dir = root / f"iter_{iteration:07d}"
        iter_dir.mkdir()
        (iter_dir / "__0_0.distcp").write_bytes(b"w" * 16)
        (iter_dir / "__1_0.distcp").write_bytes(b"v" * 16)
        (iter_dir / ".metadata").write_bytes(b"meta")
        (iter_dir / "common.pt").write_bytes(b"common")
        if run_config is not None:
            (iter_dir / export_clone.RUN_CONFIG).write_text(run_config)
    if tracker is not None:
        (root / export_clone.LATEST_FILE).write_text(f"{tracker}\n")
    return root


def snapshot(tree: Path) -> dict[str, tuple[bool, bytes | str]]:
    """Every path under ``tree`` with what it holds (a link's target, a file's bytes), to show a tree is unchanged."""
    state: dict[str, tuple[bool, bytes | str]] = {}
    for path in sorted(tree.rglob("*")):
        key = str(path.relative_to(tree))
        if path.is_symlink():
            state[key] = (True, str(path.readlink()))
        elif path.is_file():
            state[key] = (False, path.read_bytes())
        else:
            state[key] = (False, "<dir>")
    return state
