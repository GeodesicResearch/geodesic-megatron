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
"""Fake Megatron save directories for tests of the code that reads them.

Catalogue:
    make_checkpoint_dir: a run directory with ``iter_*`` saves and, optionally, the tracker training writes last.
"""

from collections.abc import Mapping
from pathlib import Path


def make_checkpoint_dir(
    root: Path,
    iterations: list[int],
    tracker: int | None,
    *,
    run_config: str | None = None,
    hf_files: Mapping[str, str | bytes] | None = None,
) -> Path:
    """A Megatron save directory at ``root`` (created if needed), returned.

    Each iteration gets ``iter_<7 digits>/`` with a shard and ``metadata.json``, plus ``run_config.yaml`` holding
    ``run_config`` and an ``hf/`` export holding ``hf_files`` (name -> content) when given. With a ``tracker``,
    the files training writes after a save completes: ``latest_checkpointed_iteration.txt`` naming it,
    ``latest_train_state.pt`` and ``progress.txt``.

    Raises:
        FileExistsError: an iteration's directory exists already.
    """
    root.mkdir(parents=True, exist_ok=True)
    for iteration in iterations:
        iter_dir = root / f"iter_{iteration:07d}"
        iter_dir.mkdir()
        (iter_dir / "__0_0.distcp").write_bytes(b"x" * 16)
        (iter_dir / "metadata.json").write_text("{}")
        if run_config is not None:
            (iter_dir / "run_config.yaml").write_text(run_config)
        if hf_files is not None:
            (iter_dir / "hf").mkdir()
            for name, content in hf_files.items():
                path = iter_dir / "hf" / name
                path.write_bytes(content) if isinstance(content, bytes) else path.write_text(content)
    if tracker is not None:
        (root / "latest_checkpointed_iteration.txt").write_text(f"{tracker}\n")
        (root / "latest_train_state.pt").write_bytes(b"s")
        (root / "progress.txt").write_text("started\n")
    return root
