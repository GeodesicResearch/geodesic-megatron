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
"""Lookups in the local HuggingFace hub cache, at a named revision or the one ``refs/main`` names.

Catalogue:
    SNAPSHOT_PATH: the pattern of a model repo's snapshot folder in the hub cache.
    hub_cache_root: the hub cache directory.
    is_cached: whether a model repo has anything in the cache.
    cached_file: a file of a model repo's snapshot in the cache.
    split_snapshot_path: the repo id and commit that a snapshot folder of the cache holds.
"""

import os
import re
from pathlib import Path


SNAPSHOT_PATH = re.compile(r".*/models--(?P<org>.+?)--(?P<name>[^/]+)/snapshots/(?P<revision>[0-9a-f]{40})/?")


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


def cached_file(repo_id: str, filename: str, revision: str | None = None) -> Path | None:
    """``filename`` of model repo ``repo_id`` in the cached snapshot of commit ``revision``; None when not cached.

    With ``revision`` None the snapshot is the one ``refs/main`` names. Reads the cache only
    (``huggingface_hub.try_to_load_from_cache``), never the network. A local directory is never cached, so it gives None.
    """
    from huggingface_hub import try_to_load_from_cache

    if not _is_repo_id(repo_id):
        return None
    path = try_to_load_from_cache(repo_id, filename, cache_dir=str(hub_cache_root()), revision=revision)
    return Path(path) if isinstance(path, str) else None


def split_snapshot_path(path: str) -> tuple[str, str | None]:
    """The repo id and commit of a model repo's snapshot folder in the hub cache, else ``path`` and None.

    A folder ``.../models--<org>--<name>/snapshots/<40 hex digits>`` holds one commit of ``<org>/<name>``. Any other
    value, a repo id included, names no commit.
    """
    match = SNAPSHOT_PATH.fullmatch(path)
    if match is None:
        return path, None
    return f"{match['org']}/{match['name']}", match["revision"]
