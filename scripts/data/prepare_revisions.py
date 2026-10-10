# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""The commit a prepare config pins for one subset of its dataset.

A prepare config (``pipeline_data_prepare.py --config``) pins its dataset in one of two ways, never both:

* ``revision``: one revision for every subset prepared under the config. Absent, the prepare reads the default
  branch's HEAD, which ``pipeline_data_prepare.py`` reports as unpinned.
* ``revisions``: a mapping from subset name to a full 40-character commit SHA, for a dataset whose subsets are
  published at commits of their own and pinned one by one as they land. A subset the mapping does not name is
  refused. It is never read at HEAD, because it is either unpublished or not yet pinned.

``pipeline_data_prepare.py`` resolves the pin of the subset it prepares here, and ``corpora_table.py``'s
``subset_prepare_config`` resolves a table row's pin here too. The verifier and the per-document checks read a row's
prepare config through that function, so they check a corpus against the commit its prepare read.

Standard library only: ``corpora_table.py`` imports it under the host Python that ``build_corpora.sh`` runs.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any


FULL_SHA = re.compile(r"[0-9a-f]{40}")


def subset_revision(config: Mapping[str, Any], subset: str | None, where: str) -> str | None:
    """The revision a prepare of ``subset`` reads under ``config``, a prepare config's mapping.

    That is the config's ``revision`` (``None`` when it states none), or its ``revisions`` entry for ``subset``.
    Raises ``ValueError``, naming ``where``, for a config that states both, a ``revisions`` that is not a non-empty
    mapping of subset names to full commit SHAs, no subset, and a subset ``revisions`` does not pin.
    """
    if "revisions" not in config:
        return config.get("revision")
    if "revision" in config:
        raise ValueError(f"{where}: states both `revision` and `revisions`; a subset's commit must come from one")
    revisions = config["revisions"]
    if not isinstance(revisions, Mapping) or not revisions:
        raise ValueError(f"{where}: `revisions` must map each pinned subset to its commit, got {revisions!r}")
    malformed = sorted(
        str(name) for name, sha in revisions.items() if not isinstance(sha, str) or not FULL_SHA.fullmatch(sha)
    )
    if malformed:
        raise ValueError(f"{where}: `revisions` pins {malformed} to something other than a full 40-character SHA")
    if subset is None:
        raise ValueError(f"{where}: pins each subset's commit (`revisions`), so the subset must be named")
    if subset not in revisions:
        raise ValueError(
            f"{where}: `revisions` pins no commit for subset {subset!r}, so it is not prepared (never at the default "
            f"branch's HEAD); the pinned subsets are {sorted(revisions)}"
        )
    return revisions[subset]
