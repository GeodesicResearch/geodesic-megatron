#!/usr/bin/env python3
"""Offline tokenization and packing for SFT and pretraining datasets.

Tokenizes JSONL files and packs them into parquet format for efficient training.
Run this once before training to avoid packing during the SLURM job.

Usage:
    # Pack chat/SFT dataset ({"messages": [...]})
    python scripts/data/pack_sft_dataset.py \
        --dataset-root /path/to/dataset \
        --tokenizer nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
        --seq-length 8192

    # Pack pretraining dataset ({"input": ..., "output": ""})
    python scripts/data/pack_sft_dataset.py \
        --dataset-root /path/to/dataset \
        --tokenizer nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
        --seq-length 16384 \
        --no-chat

    # One document per row (unpacked training), raw text tokenized exactly as stored, no EOS appended
    python scripts/data/pack_sft_dataset.py \
        --dataset-root /path/to/dataset \
        --tokenizer nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16 \
        --seq-length 4096 \
        --no-chat \
        --packing-algorithm one_per_pack \
        --dataset-kwargs '{"add_eos": false, "prompt_template": "{input}{output}", "strip_spaces": false}'

    # Leave named rows of one shard's training.jsonl out of its pack (validation.jsonl is not affected):
    # each line of rows.jsonl is {"shard": <int>, "line": <int>}, line being a 0-based line number of that
    # shard's training.jsonl; only the entries of --shard-index apply, and the pack's manifest records the shard,
    # their count and the path and sha256 of rows.jsonl. scripts/data/pack_from_config.py runs this packer on
    # the settings of a YAML config instead of command-line options.
    python scripts/data/pack_sft_dataset.py \
        --dataset-root /path/to/shard3 \
        --tokenizer nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
        --seq-length 8192 \
        --exclude-rows /path/to/rows.jsonl \
        --shard-index 3

    # The dataset root should contain training.jsonl and optionally validation.jsonl
    # (produced by the HFDatasetBuilder preprocessing step or pipeline_data_prepare.py).
    #
    # Output goes to <dataset-root>/packed/<tokenizer-name>_pad_seq_to_mult<N>/
"""

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pipeline.data import layout  # noqa: E402
from pipeline.data.tokenizer_check import require_pre_tokenizer_of_file  # noqa: E402


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def pack_manifest_path(pack_path: Path) -> Path:
    """The JSON file beside a pack recording the settings it was packed with."""
    return pack_path.with_name(pack_path.name.replace(".idx.parquet", ".pack.json"))


def check_existing_pack(pack_path: Path, settings: dict, default_kwargs: dict) -> None:
    """Refuse to reuse a pack built with other settings than ``settings``.

    A pack whose manifest records the same settings is reused. A pack without a manifest predates manifests and
    is reused only for the settings every such pack was built with (first_fit_shuffle, the format's default
    dataset kwargs and no excluded rows); the pack path does not encode the algorithm, the kwargs or the exclusions.

    Raises:
        ValueError: the settings differ, or cannot be known to match.
    """
    manifest = pack_manifest_path(pack_path)
    if manifest.exists():
        recorded = json.loads(manifest.read_text())
        if recorded != json.loads(json.dumps(settings)):
            raise ValueError(f"{pack_path} was packed with {recorded}, not {settings}; remove it to repack")
        return
    legacy = (
        settings["packing_algorithm"] == "first_fit_shuffle"
        and settings["dataset_kwargs"] == default_kwargs
        and "exclude_rows" not in settings
    )
    if not legacy:
        raise ValueError(
            f"{pack_path} has no {manifest.name}, so its packing settings are unknown and may not be {settings}; "
            "remove it to repack"
        )


def read_excluded_rows(path: Path, shard_index: int) -> tuple[list[int], str]:
    """The lines of shard ``shard_index``'s training.jsonl to leave out of its pack, and the sha256 of ``path``.

    ``path`` is JSON Lines, an entry per line, ``{"shard": <int>, "line": <int>}``, where ``line`` is a 0-based line
    number in that shard's training.jsonl. Every entry is checked, those of other shards too, because one file serves
    the pack of every shard. The file is read once, so the digest is of the bytes the entries come from.

    Returns:
        The lines of the entries whose shard is ``shard_index``, ascending, and the file's sha256 hex digest.

    Raises:
        ValueError: a line that is not such an entry, a negative line number or an entry repeated, naming the entry
            and its line in ``path``. (A line beyond the shard's last is found when the rows are tokenized.)
    """
    raw = path.read_bytes()
    first_seen: dict[tuple[int, int], int] = {}
    for number, text in enumerate(raw.decode("utf-8").splitlines(), start=1):
        where = f"{path}:{number}: "
        try:
            entry = json.loads(text)
        except json.JSONDecodeError as error:
            raise ValueError(f"{where}not JSON ({error}): {text!r}") from error
        # type(...) is int, not isinstance: a JSON true is a Python bool, which is an int.
        if not (
            isinstance(entry, dict)
            and entry.keys() == {"shard", "line"}
            and all(type(v) is int for v in entry.values())
        ):
            raise ValueError(f'{where}expected {{"shard": <int>, "line": <int>}}, got {text!r}')
        if entry["line"] < 0:
            raise ValueError(f"{where}negative line number: {text!r}")
        key = (entry["shard"], entry["line"])
        if key in first_seen:
            raise ValueError(f"{where}duplicate of line {first_seen[key]}: {text!r}")
        first_seen[key] = number
    return sorted(line for shard, line in first_seen if shard == shard_index), hashlib.sha256(raw).hexdigest()


def main(argv: list[str] | None = None):
    """Tokenize and pack a dataset root's training (and validation) JSONL into packed parquet.

    ``argv`` is the command line without the program name; ``None`` reads ``sys.argv``."""
    parser = argparse.ArgumentParser(description="Offline SFT dataset packing")
    parser.add_argument(
        "--dataset-root", type=str, required=True, help="Directory with training.jsonl / validation.jsonl"
    )
    parser.add_argument("--tokenizer", type=str, required=True, help="HuggingFace tokenizer model ID or path")
    parser.add_argument("--seq-length", type=int, default=16384, help="Sequence length (default: 16384)")
    parser.add_argument(
        "--pad-seq-to-mult", type=int, default=1, help="Pad each sequence to this multiple (default: 1)"
    )
    parser.add_argument("--seed", type=int, default=1234, help="Random seed (default: 1234)")
    parser.add_argument("--no-validation", action="store_true", help="Skip validation split")
    parser.add_argument(
        "--num-tokenizer-workers",
        type=int,
        default=1,
        help="Tokenization worker processes (default 1 = serial; the Pool in packed_sequence.py "
        "preserves row order, so output is identical at any worker count). 10k rows measured "
        "118s serial; the full 5.9M-row SFT corpus needs parallelism to fit a day.",
    )
    parser.add_argument(
        "--no-chat",
        action="store_true",
        help="Use pretraining format (input/output JSONL) instead of chat format (messages JSONL)",
    )
    parser.add_argument(
        "--packing-algorithm",
        type=str,
        default="first_fit_shuffle",
        help="One of packing_utils.PACKING_ALGOS; one_per_pack stores each document in a row of its own "
        "(default: first_fit_shuffle)",
    )
    parser.add_argument(
        "--dataset-kwargs",
        type=json.loads,
        default={},
        help="JSON object of create_sft_dataset keyword arguments, merged over the chat or --no-chat defaults",
    )
    parser.add_argument(
        "--exclude-rows",
        type=Path,
        help='JSON Lines file of {"shard": N, "line": L} entries: lines L (0-based) of training.jsonl to leave out '
        "of the pack, for the entries whose shard is --shard-index (validation.jsonl is not affected); needs "
        "--shard-index",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        help="Which shard this dataset root is, selecting the --exclude-rows entries that apply; needs --exclude-rows",
    )
    args = parser.parse_args(argv)
    if (args.exclude_rows is None) != (args.shard_index is None):
        parser.error("--exclude-rows and --shard-index must be given together")
    excluded_rows: list[int] = []
    exclusion_record = None
    if args.exclude_rows is not None:
        excluded_rows, exclusion_sha256 = read_excluded_rows(args.exclude_rows, args.shard_index)
        exclusion_record = {
            "path": str(args.exclude_rows),
            "shard_index": args.shard_index,
            "count": len(excluded_rows),
            "sha256": exclusion_sha256,
        }
        logger.info(f"Excluding {len(excluded_rows)} rows of shard {args.shard_index} listed in {args.exclude_rows}")

    from megatron.bridge.data.datasets.packed_sequence import prepare_packed_sequence_data
    from megatron.bridge.data.datasets.packing_utils import PACKING_ALGOS
    from megatron.bridge.training.tokenizers.config import TokenizerConfig
    from megatron.bridge.training.tokenizers.tokenizer import build_tokenizer

    dataset_root = Path(args.dataset_root)
    train_path = dataset_root / "training.jsonl"
    val_path = dataset_root / "validation.jsonl"

    if not train_path.exists():
        raise FileNotFoundError(f"training.jsonl not found in {dataset_root}")

    # Build tokenizer
    logger.info(f"Loading tokenizer: {args.tokenizer}")
    tokenizer_config = TokenizerConfig(
        tokenizer_type="HuggingFaceTokenizer",
        tokenizer_model=args.tokenizer,
    )
    tokenizer = build_tokenizer(tokenizer_config)
    if Path(args.tokenizer).is_dir():
        # Bridge's tokenizer wraps Megatron's HuggingFaceTokenizer, which holds the Hugging Face tokenizer as .tokenizer.
        require_pre_tokenizer_of_file(tokenizer._tokenizer.tokenizer, args.tokenizer)
    if args.packing_algorithm not in PACKING_ALGOS:
        raise ValueError(f"--packing-algorithm must be one of {PACKING_ALGOS}, got {args.packing_algorithm!r}")

    # Output directory
    pack_dir = layout.pack_dir(dataset_root, args.tokenizer, args.pad_seq_to_mult)
    pack_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = pack_dir / f"{args.seq_length}_metadata.jsonl"

    if args.no_chat:
        default_kwargs = {
            "chat": False,
            "answer_only_loss": False,
            "pad_to_max_length": False,
        }
        logger.info("Using pretraining format (chat=False, answer_only_loss=False)")
    else:
        default_kwargs = {
            "chat": True,
            "use_hf_tokenizer_chat_template": True,
            "answer_only_loss": True,
            "pad_to_max_length": False,
        }
    dataset_kwargs = {**default_kwargs, **args.dataset_kwargs}
    logger.info(f"Dataset kwargs: {dataset_kwargs}; packing algorithm: {args.packing_algorithm}")

    settings = {
        "packing_algorithm": args.packing_algorithm,
        "dataset_kwargs": dataset_kwargs,
        "seq_length": args.seq_length,
        "seed": args.seed,
    }

    # Only the training split leaves rows out, and its manifest says how many and by which file.
    training_settings = settings if exclusion_record is None else {**settings, "exclude_rows": exclusion_record}

    def pack_split(split_name, input_path, output_path, split_settings, exclude_rows):
        if output_path.exists():
            check_existing_pack(output_path, split_settings, default_kwargs)
            logger.info(f"{split_name} parquet already exists with these settings: {output_path}")
            return
        logger.info(f"Packing {split_name} split: {input_path} -> {output_path}")
        prepare_packed_sequence_data(
            input_path=input_path,
            output_path=output_path,
            output_metadata_path=metadata_path,
            packed_sequence_size=args.seq_length,
            tokenizer=tokenizer,
            max_seq_length=args.seq_length,
            seed=args.seed,
            dataset_kwargs=dict(dataset_kwargs),
            packing_algorithm=args.packing_algorithm,
            pad_seq_to_mult=args.pad_seq_to_mult,
            num_tokenizer_workers=args.num_tokenizer_workers,
            exclude_rows=exclude_rows,
        )
        pack_manifest_path(output_path).write_text(json.dumps(split_settings, indent=2, sort_keys=True) + "\n")
        logger.info(f"{split_name} split packed: {output_path}")

    def split_path(split):
        return layout.packed_parquet_path(dataset_root, args.tokenizer, args.pad_seq_to_mult, args.seq_length, split)

    pack_split("Training", train_path, split_path("training"), training_settings, excluded_rows)
    if not args.no_validation and val_path.exists():
        pack_split("Validation", val_path, split_path("validation"), settings, [])
    elif not args.no_validation:
        logger.warning(f"No validation.jsonl found at {val_path}")

    logger.info("Done. Packed files at: %s", pack_dir)


if __name__ == "__main__":
    main()
