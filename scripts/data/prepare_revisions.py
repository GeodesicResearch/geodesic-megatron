# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""The commits a prepare config pins: its dataset's, per subset, and its tokenizer's.

A prepare config (``pipeline_data_prepare.py --config``) pins its dataset in one of two ways, never both:

* ``revision``: one revision for every subset prepared under the config. Absent, the prepare reads the default
  branch's HEAD, which ``pipeline_data_prepare.py`` reports as unpinned.
* ``revisions``: a mapping from subset name to a full 40-character commit SHA, for a dataset whose subsets are
  published at commits of their own and pinned one by one as they land. A subset the mapping does not name is
  refused. It is never read at HEAD, because it is either unpublished or not yet pinned.

It may also pin its ``tokenizer`` at a full commit SHA with ``tokenizer-revision``. Every step that loads the
tokenizer for the corpus then loads that commit: the prepare, the tokenize job (which reads the commit's snapshot
directory), the per-document checks and the audit. The tokenize job takes the pinned tokenizer as one word, a
*tokenizer reference*: ``<name>@<sha>``, or the bare name when the config pins none. Its provenance records the name
as ``tokenizer`` and the commit as ``tokenizer_revision``, and ``verify_corpora.py`` compares both with the config.

``pipeline_data_prepare.py`` resolves the pin of the subset it prepares here, and ``corpora_table.py``'s
``subset_prepare_config`` resolves a table row's pins here too. The verifier and the per-document checks read a row's
prepare config through that function, so they check a corpus against the commits its prepare read.

Standard library only: ``corpora_table.py`` imports it under the host Python that ``build_corpora.sh`` runs. Only
``tokenizer_load_path`` reads the Hub, importing ``huggingface_hub`` when it is called.

Run as a script, it prints one field of a tokenizer reference, for the tokenize job in
``pipeline_data_submit.sbatch``: ``name`` (the tokenizer's name), ``revision`` (its commit, or nothing when the
reference pins none) or ``path`` (what to load it from: the pinned commit's local snapshot directory, else the name).
"""

from __future__ import annotations

import argparse
import re
from collections.abc import Mapping
from typing import Any


FULL_SHA = re.compile(r"[0-9a-f]{40}")
TOKENIZER_REVISION_KEY = "tokenizer-revision"
REFERENCE_SEPARATOR = "@"


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


def require_full_sha(revision: object, what: str) -> str:
    """``revision`` itself, when it is a full 40-character commit SHA; raises ``ValueError`` naming ``what`` otherwise.

    A branch, a tag or a short SHA can resolve to another commit later, so none of them pins anything.
    """
    if not isinstance(revision, str) or not FULL_SHA.fullmatch(revision):
        raise ValueError(f"{what} must be a full 40-character commit SHA, got {revision!r}")
    return revision


def tokenizer_revision(config: Mapping[str, Any], where: str) -> str | None:
    """The commit ``config``, a prepare config's mapping, pins its tokenizer at; ``None`` when it pins none.

    Raises ``ValueError``, naming ``where``, for a ``tokenizer-revision`` that is not a full commit SHA.
    """
    if TOKENIZER_REVISION_KEY not in config:
        return None
    return require_full_sha(config[TOKENIZER_REVISION_KEY], f"{where}: `{TOKENIZER_REVISION_KEY}`")


def tokenizer_reference(name: str, revision: str | None) -> str:
    """The one-word form of a tokenizer the tokenize job takes: ``<name>@<revision>``, or ``name`` when unpinned."""
    if REFERENCE_SEPARATOR in name:
        raise ValueError(f"tokenizer name {name!r} contains {REFERENCE_SEPARATOR!r}, which separates the commit")
    if revision is None:
        return name
    return f"{name}{REFERENCE_SEPARATOR}{require_full_sha(revision, f'the revision of tokenizer {name}')}"


def split_tokenizer_reference(reference: str) -> tuple[str, str | None]:
    """The name and the commit (``None`` when unpinned) of a tokenizer reference, the inverse of ``tokenizer_reference``.

    Raises ``ValueError`` for a reference whose commit is not a full SHA, or that names no tokenizer.
    """
    name, separator, revision = reference.partition(REFERENCE_SEPARATOR)
    if not name:
        raise ValueError(f"tokenizer reference {reference!r} names no tokenizer")
    if not separator:
        return name, None
    return name, require_full_sha(revision, f"the commit of tokenizer reference {reference!r}")


def tokenizer_load_path(reference: str) -> str:
    """What to load a tokenizer reference's tokenizer from: its pinned commit's local snapshot, else its name.

    A pinned commit is downloaded into the Hugging Face cache (``HF_HOME``) if it is not there yet. Raises
    ``huggingface_hub``'s ``RevisionNotFoundError`` when the repository has no such commit.
    """
    name, revision = split_tokenizer_reference(reference)
    if revision is None:
        return name
    from huggingface_hub import snapshot_download

    return snapshot_download(repo_id=name, revision=revision)


def main() -> None:
    """Print one field of a tokenizer reference: its name, its commit (empty when unpinned) or its load path."""
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("field", choices=("name", "revision", "path"))
    parser.add_argument("reference", help="<name>@<40-character commit SHA>, or a bare name")
    args = parser.parse_args()
    name, revision = split_tokenizer_reference(args.reference)
    if args.field == "name":
        print(name)
    elif args.field == "revision":
        print(revision or "")
    else:
        print(tokenizer_load_path(args.reference))


if __name__ == "__main__":
    main()
