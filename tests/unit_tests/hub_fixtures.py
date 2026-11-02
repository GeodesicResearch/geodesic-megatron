# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""A stand-in for the Hugging Face Hub serving local directories laid out as one dataset repository's commits.

The Hub's listing (``HfApi.list_repo_tree``) and filesystem (``HfFileSystem``) are the network
boundary of the corpus tools' Hub reads, which a unit test cannot cross. ``local_hub`` replaces both
for one dataset, serving one local tree per revision: the listing lists a directory as the Hub does
(its direct entries, by repository-relative path), and the filesystem opens the file a
``hub_parquet.hub_file_url`` URL names. Everything above that boundary (which files are chosen, what
is read from them, and at which commit) is the tools' own, and a request for another dataset or
revision fails as the Hub would.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path


def local_hub(monkeypatch, dataset: str, trees: Mapping[str, Path]) -> list[tuple[str, str]]:
    """Serve ``trees[revision]`` as ``dataset`` at each revision; returns the ``(revision, path)`` pairs opened, in order."""
    import huggingface_hub

    opened: list[tuple[str, str]] = []

    def tree(repo_id: str, revision: str) -> Path:
        if repo_id != dataset or revision not in trees:
            raise FileNotFoundError(f"{repo_id}@{revision} is not served")
        return trees[revision]

    class _Entry:
        def __init__(self, path: str):
            self.path = path

    class _Api:
        def list_repo_tree(self, repo_id, path_in_repo, revision, repo_type):
            if repo_type != "dataset":
                raise FileNotFoundError(f"{repo_type} {repo_id} is not served")
            root = tree(repo_id, revision)
            return [_Entry(path.relative_to(root).as_posix()) for path in (root / path_in_repo).iterdir()]

    class _FileSystem:
        def open(self, url, mode):
            # datasets/<org>/<name>@<revision>/<path>, as hub_parquet.hub_file_url writes it.
            repo_id, _, rest = url.removeprefix("datasets/").partition("@")
            revision, _, path = rest.partition("/")
            root = tree(repo_id, revision)
            opened.append((revision, path))
            return open(root / path, mode)

    monkeypatch.setattr(huggingface_hub, "HfApi", _Api)
    monkeypatch.setattr(huggingface_hub, "HfFileSystem", _FileSystem)
    return opened
