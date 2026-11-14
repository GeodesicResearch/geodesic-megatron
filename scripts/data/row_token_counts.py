#!/usr/bin/env python3
# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""Count the tokens of every row of an SFT JSONL file as the packer tokenizes it, before any row is packed or cut.

The pass tokenizes each row of ``<dataset_root>/training.jsonl`` with the packer's tokenizer and settings
(``scripts/data/pack_sft_dataset.py``) and keeps only its length and its digest, so a file of any size is read without a
cache. A row is over length when it holds more than ``seq_length`` tokens: the packer would cut it, so it is left out
of the packs. The digest is the pack audit's (``scripts/data/audit_pack_digests.py`` ``ids_hash``), so the kept list
written here is the list the audit compares a pack with.

The config is YAML with exactly these keys::

    dataset_root: /path/to/corpus      # holds training.jsonl; absolute
    tokenizer: /path/to/tokenizer      # a tokenizer directory or Hugging Face id, as in the pack config
    seq_length: 32768                  # a row of more tokens is over length
    dataset_kwargs: {answer_only_loss: true}  # merged over the packer's chat defaults, as in the pack config
    out_dir: /path/to/counts           # absolute; must not already hold the outputs

The outputs in ``out_dir``:

* ``rows.tsv``: a header and one line per row of the file, ``<line>\\t<tokens>\\t<digest>``, ``line`` 0-based.
* ``kept_digests.txt``: the digest of every row that is not over length, one per line, in file order.
* ``over_length.jsonl``: ``{"line": <int>, "tokens": <int>}`` for every row that is over length.
* ``over_length_digests.txt``: the digest of every row that is over length, one per line, in file order (the pack
  audit's dropped list).
* ``summary.json``: the config, the file's size, and the counts and token totals of all rows, kept rows and rows over
  length, with the digest definition.
* ``count_config.yaml``: a copy of the config.

Usage::

    python scripts/data/row_token_counts.py --config /path/to/count.yaml [--num-tokenizer-workers 64]

``--num-tokenizer-workers`` sets how many tokenizer processes run. It changes the run time and not the outputs.

``counted_source`` reads back, from an output directory's summary, which file a count pass counted and its size.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from scripts.data import pack_sft_dataset  # noqa: E402
from scripts.data.audit_pack_digests import DIGEST_DEFINITION, ids_hash  # noqa: E402
from scripts.mapping_keys import require_keys  # noqa: E402
from scripts.mapping_values import require_absolute_path, require_int, require_string  # noqa: E402


logger = logging.getLogger(__name__)

CONFIG_KEYS = frozenset({"dataset_root", "tokenizer", "seq_length", "dataset_kwargs", "out_dir"})
SUMMARY = "summary.json"
OUTPUTS = (
    "rows.tsv",
    "kept_digests.txt",
    "over_length.jsonl",
    "over_length_digests.txt",
    SUMMARY,
    "count_config.yaml",
)


class CountConfigError(ValueError):
    """A count config that cannot be used, or an output directory that already holds outputs."""


def read_count_config(path: Path) -> dict:
    """The mapping a count config holds, refused unless it has exactly the keys ``CONFIG_KEYS`` with usable values.

    Raises:
        CountConfigError: the file cannot be read, a key is unknown or missing, or a value is of the wrong type.
    """
    try:
        loaded = yaml.safe_load(path.read_text())
    except (OSError, yaml.YAMLError) as error:
        raise CountConfigError(f"{path}: cannot read a count config: {error}") from error
    config = require_keys(loaded, f"{path}: a count config", CONFIG_KEYS, error=CountConfigError)
    require_absolute_path(config["dataset_root"], f"{path}: dataset_root", error=CountConfigError)
    require_string(config["tokenizer"], f"{path}: tokenizer", error=CountConfigError)
    require_int(config["seq_length"], f"{path}: seq_length", 1, error=CountConfigError)
    if not isinstance(config["dataset_kwargs"], dict):
        raise CountConfigError(f"{path}: dataset_kwargs must be a mapping, not {config['dataset_kwargs']!r}")
    require_absolute_path(config["out_dir"], f"{path}: out_dir", error=CountConfigError)
    return config


def counted_source(out_dir: Path) -> tuple[str, int]:
    """The path and the byte size of the file that the count pass which wrote ``out_dir`` counted, read from its
    ``summary.json``.

    Raises:
        OSError: the summary cannot be read.
        ValueError: the summary is not JSON.
        KeyError: the summary lacks the source or its size.
    """
    summary = json.loads((out_dir / SUMMARY).read_text())
    return summary["source"], summary["source_bytes"]


def write_counts(lengths_and_digests, out_dir: Path, seq_length: int) -> dict:
    """Write ``rows.tsv``, ``kept_digests.txt``, ``over_length.jsonl`` and ``over_length_digests.txt`` into ``out_dir``
    from the ``(tokens, digest)`` of each row in file order, and return the counts and token totals of all rows, kept
    rows and rows over ``seq_length`` tokens."""
    totals = {"rows": 0, "tokens": 0, "kept_rows": 0, "kept_tokens": 0, "over_length_rows": 0, "over_length_tokens": 0}
    with (
        (out_dir / "rows.tsv").open("w") as rows,
        (out_dir / "kept_digests.txt").open("w") as kept,
        (out_dir / "over_length.jsonl").open("w") as over,
        (out_dir / "over_length_digests.txt").open("w") as over_digests,
    ):
        rows.write("line\ttokens\tdigest\n")
        for line, (tokens, digest) in enumerate(lengths_and_digests):
            rows.write(f"{line}\t{tokens}\t{digest}\n")
            totals["rows"] += 1
            totals["tokens"] += tokens
            if tokens > seq_length:
                over.write(json.dumps({"line": line, "tokens": tokens}) + "\n")
                over_digests.write(f"{digest}\n")
                totals["over_length_rows"] += 1
                totals["over_length_tokens"] += tokens
            else:
                kept.write(f"{digest}\n")
                totals["kept_rows"] += 1
                totals["kept_tokens"] += tokens
    return totals


def main(argv: list[str] | None = None) -> None:
    """Count the rows of the file a count config names and write the outputs of the module docstring."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True, help="the count config")
    parser.add_argument(
        "--num-tokenizer-workers",
        type=int,
        default=1,
        help="tokenizer processes (default 1: this one); the outputs are the same at any count",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    config = read_count_config(args.config)
    require_int(args.num_tokenizer_workers, "--num-tokenizer-workers", 1, error=CountConfigError)
    source = Path(config["dataset_root"]) / "training.jsonl"
    if not source.is_file():
        raise FileNotFoundError(f"{source} does not exist")
    out_dir = Path(config["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    present = [name for name in OUTPUTS if (out_dir / name).exists()]
    if present:
        raise CountConfigError(f"{out_dir} already holds {present}; remove them to count again")

    from megatron.bridge.data.datasets.packed_sequence import iter_tokenized_ids

    tokenizer = pack_sft_dataset.build_packing_tokenizer(config["tokenizer"])
    dataset_kwargs = {**pack_sft_dataset.default_dataset_kwargs(chat=True), **config["dataset_kwargs"]}
    # The size is taken before the source is read and compared after it, so the summary never names a file that the
    # counts do not cover (partition_rows.require_counts_of trusts it).
    source_bytes = source.stat().st_size
    ids_of_rows = iter_tokenized_ids(
        source, tokenizer, config["seq_length"], dataset_kwargs, args.num_tokenizer_workers
    )
    totals = write_counts(((len(ids), ids_hash(ids)) for ids in ids_of_rows), out_dir, config["seq_length"])
    if source.stat().st_size != source_bytes:
        raise CountConfigError(
            f"{source} changed from {source_bytes} to {source.stat().st_size} bytes during the count; "
            f"remove {out_dir}'s outputs and count it again"
        )
    summary = {
        "config": config,
        "source": str(source),
        "source_bytes": source_bytes,
        "seq_length": config["seq_length"],
        **totals,
        "digest_definition": DIGEST_DEFINITION,
    }
    (out_dir / SUMMARY).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    shutil.copyfile(args.config, out_dir / "count_config.yaml")
    logger.info(
        "%d rows, %d tokens; %d rows over %d tokens",
        totals["rows"],
        totals["tokens"],
        totals["over_length_rows"],
        config["seq_length"],
    )


if __name__ == "__main__":
    main()
