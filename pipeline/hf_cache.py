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
"""The local HuggingFace hub cache: what it holds of a model repo, and the directory a model loads from.

Catalogue:
    hub_cache_root: the hub cache directory.
    is_cached: whether a model repo has anything in the cache.
    cached_file: a file of a model repo's ``refs/main`` snapshot in the cache.
    snapshot_path: a model repo's snapshot directory, complete in the cache.
    model_dir: the directory a load of a model reads its files from.
    main: the command line, ``python -m pipeline.hf_cache model-dir MODEL``.
"""

import argparse
import os
from pathlib import Path


def hub_cache_root() -> Path:
    """``$HF_HOME/hub``, or ``~/.cache/huggingface/hub`` when HF_HOME is unset (reads the environment)."""
    return Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"


def _is_repo_id(repo_id: str) -> bool:
    """Whether ``repo_id`` is a Hub repo id; a local directory, such as a release copied to disk, is not."""
    from huggingface_hub.utils import HFValidationError, validate_repo_id

    try:
        validate_repo_id(repo_id)
    except HFValidationError:
        return False
    return True


def is_cached(repo_id: str) -> bool:
    """Whether model repo ``repo_id`` has a directory in the hub cache (never true of a local directory)."""
    return _is_repo_id(repo_id) and (hub_cache_root() / f"models--{repo_id.replace('/', '--')}").is_dir()


def cached_file(repo_id: str, filename: str) -> Path | None:
    """``filename`` of model repo ``repo_id`` in the cached snapshot its ``refs/main`` names; None when not cached.

    Reads the cache only (``huggingface_hub.try_to_load_from_cache``), never the network. A local directory is never
    cached, so it gives None.
    """
    from huggingface_hub import try_to_load_from_cache

    if not _is_repo_id(repo_id):
        return None
    path = try_to_load_from_cache(repo_id, filename, cache_dir=str(hub_cache_root()))
    return Path(path) if isinstance(path, str) else None


def snapshot_path(repo_id: str) -> Path:
    """The snapshot directory of model repo ``repo_id`` in the hub cache, at the revision of its main branch.

    Downloads the files the cache lacks through ``huggingface_hub.snapshot_download``. That uses the network and
    writes under the hub cache the library reads from the environment. With ``HF_HUB_OFFLINE=1`` the revision is
    the cached ``refs/main``, and a repo the cache lacks raises.
    """
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo_id=repo_id))


def model_dir(model: str) -> Path:
    """The directory a load of ``model`` reads its files from.

    ``model`` is a model directory or a Hub repo id. A directory gives itself, resolved. A repo id gives
    ``snapshot_path(model)``, which downloads what the cache lacks.
    """
    return Path(model).resolve() if Path(model).is_dir() else snapshot_path(model)


def main(argv: list[str] | None = None) -> None:
    """``model-dir MODEL`` prints ``model_dir(MODEL)`` as the one line on stdout."""
    parser = argparse.ArgumentParser(prog="python -m pipeline.hf_cache", description="The local HuggingFace hub cache")
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("model-dir", help="print the directory a load of a model reads its files from")
    command.add_argument("model")
    args = parser.parse_args(argv)
    print(model_dir(args.model))


if __name__ == "__main__":
    main()
