#!/usr/bin/env python3
# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""Pack one shard of an SFT corpus with the settings a YAML config states, through ``scripts/data/pack_sft_dataset.py``.

A pack is described by the config, not by command-line options, so the same file packs every shard of a corpus and the
copy kept beside each pack says how that pack was made. The config is YAML with exactly these keys::

    dataset_root: /path/to/corpus/shard{shard}    # a shard's dataset root, holding training.jsonl and optionally
                                                  # validation.jsonl; {shard} is replaced by the --shard number
    tokenizer: /path/to/tokenizer                 # a tokenizer directory or Hugging Face id
    seq_length: 32768
    pad_seq_to_mult: 8
    packing_algorithm: first_fit_shuffle          # one of packing_utils.PACKING_ALGOS
    dataset_kwargs: {answer_only_loss: true, pad_to_max_length: true}  # merged over the chat defaults of the packer
    exclude_rows: /path/to/excluded_rows.jsonl    # JSON Lines of {"shard": <int>, "line": <int>}, or null for none

``dataset_root`` and ``exclude_rows`` are absolute paths. Every other option of the packer keeps its own default. With an
``exclude_rows`` file, the entries of the shard given by ``--shard`` are left out of its training pack (the packer's
``--exclude-rows`` and ``--shard-index``), and the pack's manifest records the file's path, the shard, the count of
entries and the file's sha256.

When the pack is made, the config is copied to ``<seq_length>_pack_config.yaml`` in the pack directory beside the
packed parquet (``pipeline.data.layout.pack_dir``). A pack directory that already holds a different copy is an error,
because the pack on disk was made by that other config.

Usage::

    python scripts/data/pack_from_config.py --config /path/to/pack.yaml --shard 3 [--num-tokenizer-workers 64]

``--num-tokenizer-workers`` sets how many tokenizer processes the packer starts. It changes the run time and not the pack.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from scripts.data import pack_sft_dataset  # noqa: E402
from scripts.mapping_keys import require_keys  # noqa: E402
from scripts.mapping_values import require_absolute_path, require_int, require_string  # noqa: E402

from pipeline.data import layout  # noqa: E402
from pipeline.data.packing import packer_arguments  # noqa: E402


CONFIG_KEYS = frozenset(
    {
        "dataset_root",
        "tokenizer",
        "seq_length",
        "pad_seq_to_mult",
        "packing_algorithm",
        "dataset_kwargs",
        "exclude_rows",
    }
)
SHARD_PLACEHOLDER = "{shard}"


class PackConfigError(ValueError):
    """A pack config that cannot be used, or a pack directory that holds the copy of another config."""


def read_pack_config(path: Path) -> dict:
    """The mapping a pack config holds, refused unless it has exactly the keys ``CONFIG_KEYS`` with usable values.

    Raises:
        PackConfigError: the file cannot be read, a key is unknown or missing, ``dataset_root`` has no ``{shard}``
            placeholder, or a value is of the wrong type.
    """
    try:
        loaded = yaml.safe_load(path.read_text())
    except (OSError, yaml.YAMLError) as error:
        raise PackConfigError(f"{path}: cannot read a pack config: {error}") from error
    config = require_keys(loaded, f"{path}: a pack config", CONFIG_KEYS, error=PackConfigError)
    root = require_absolute_path(config["dataset_root"], f"{path}: dataset_root", error=PackConfigError)
    if SHARD_PLACEHOLDER not in root:
        raise PackConfigError(f"{path}: dataset_root {root!r} does not hold the placeholder {SHARD_PLACEHOLDER}")
    require_string(config["tokenizer"], f"{path}: tokenizer", error=PackConfigError)
    require_int(config["seq_length"], f"{path}: seq_length", 1, error=PackConfigError)
    require_int(config["pad_seq_to_mult"], f"{path}: pad_seq_to_mult", 1, error=PackConfigError)
    require_string(config["packing_algorithm"], f"{path}: packing_algorithm", error=PackConfigError)
    if not isinstance(config["dataset_kwargs"], dict):
        raise PackConfigError(f"{path}: dataset_kwargs must be a mapping, not {config['dataset_kwargs']!r}")
    if config["exclude_rows"] is not None:
        require_absolute_path(config["exclude_rows"], f"{path}: exclude_rows", error=PackConfigError)
    return config


def dataset_root(config: dict, shard: int) -> Path:
    """The dataset root of shard ``shard``: the config's template with the placeholder replaced by the number."""
    return Path(config["dataset_root"].replace(SHARD_PLACEHOLDER, str(shard)))


def packer_argv(config: dict, shard: int, num_tokenizer_workers: int | None) -> list[str]:
    """The command line of ``pack_sft_dataset.main`` that packs shard ``shard`` as the config states, with
    ``num_tokenizer_workers`` tokenizer processes (None: the packer's default)."""
    return packer_arguments(
        dataset_root(config, shard),
        config["tokenizer"],
        config["seq_length"],
        config["pad_seq_to_mult"],
        config["packing_algorithm"],
        config["dataset_kwargs"],
        no_validation=False,
        no_chat=False,
        exclude_rows=config["exclude_rows"],
        shard_index=None if config["exclude_rows"] is None else shard,
        num_tokenizer_workers=num_tokenizer_workers,
    )


def copy_config(config_path: Path, config: dict, shard: int) -> Path:
    """Copy the config to the pack directory of shard ``shard`` and return the copy.

    Raises:
        PackConfigError: the pack directory already holds a copy with other contents.
    """
    directory = layout.pack_dir(dataset_root(config, shard), config["tokenizer"], config["pad_seq_to_mult"])
    copy = directory / f"{config['seq_length']}_pack_config.yaml"
    if copy.exists():
        if copy.read_bytes() != config_path.read_bytes():
            raise PackConfigError(
                f"{copy} differs from {config_path}, so the pack beside it was made by another config"
            )
    else:
        shutil.copyfile(config_path, copy)
    return copy


def main(argv: list[str] | None = None) -> None:
    """Pack the shard a command line names as its config states, then copy the config beside the pack."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True, help="the pack config")
    parser.add_argument(
        "--shard",
        type=int,
        required=True,
        help="the shard's job index: it replaces {shard} in dataset_root and selects the exclude_rows entries",
    )
    parser.add_argument(
        "--num-tokenizer-workers",
        type=int,
        default=None,
        help="tokenizer processes (the packer's own default when omitted); the pack is the same at any count",
    )
    args = parser.parse_args(argv)
    config = read_pack_config(args.config)
    require_int(args.shard, "--shard", 0, error=PackConfigError)
    if args.num_tokenizer_workers is not None:
        require_int(args.num_tokenizer_workers, "--num-tokenizer-workers", 1, error=PackConfigError)
    pack_sft_dataset.main(packer_argv(config, args.shard, args.num_tokenizer_workers))
    copy_config(args.config, config, args.shard)


if __name__ == "__main__":
    main()
