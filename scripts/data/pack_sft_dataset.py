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

    # The dataset root should contain training.jsonl and optionally validation.jsonl
    # (produced by the HFDatasetBuilder preprocessing step or pipeline_data_prepare.py).
    #
    # Output goes to <dataset-root>/packed/<tokenizer-name>_pad_seq_to_mult<N>/

Each pack has a manifest, <pack>.pack.json, that records its settings. A chat pack's manifest also records the sha256 of
the tokenizer's chat template.

Catalogue:
    TEMPLATE_KEY: the manifest key that holds the chat template's sha256.
    pack_manifest_path: the manifest beside a pack.
    chat_template_sha256: the sha256 of a tokenizer's chat template.
    check_existing_pack: refuse to reuse a pack built under other settings.
    main: the command line.
"""

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pipeline.data import layout  # noqa: E402


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


TEMPLATE_KEY = "chat_template_sha256"


def pack_manifest_path(pack_path: Path) -> Path:
    """The JSON file beside a pack recording the settings it was packed with."""
    return pack_path.with_name(pack_path.name.replace(".idx.parquet", ".pack.json"))


def chat_template_sha256(template: str | dict | None) -> str | None:
    """The sha256 of a chat template, or None when the tokenizer has none.

    A template given as text hashes as its UTF-8 bytes. That equals the sha256 of the ``chat_template.jinja`` it was
    read from. Several templates given by name, as a dict, hash as their JSON with sorted keys.
    """
    if template is None:
        return None
    text = template if isinstance(template, str) else json.dumps(template, sort_keys=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def check_existing_pack(pack_path: Path, settings: dict, default_kwargs: dict) -> None:
    """Refuse to reuse a pack unless it was built with ``settings``.

    The function returns when the pack's manifest records ``settings``, including the chat template's sha256 under
    ``TEMPLATE_KEY``. If ``settings`` holds ``TEMPLATE_KEY`` and the manifest lacks it, the function ignores the key and
    logs a warning. If the pack has no manifest, the function returns only for first_fit_shuffle with the format's
    default dataset kwargs. The pack path does not encode the packing algorithm or the dataset kwargs.

    Raises:
        ValueError: the settings differ, or cannot be known to match.
    """
    manifest = pack_manifest_path(pack_path)
    if manifest.exists():
        recorded = json.loads(manifest.read_text())
        wanted = json.loads(json.dumps(settings))
        if TEMPLATE_KEY in wanted and TEMPLATE_KEY not in recorded:
            logger.warning(f"{pack_path} predates the chat template hash; its template is unknown")
            del wanted[TEMPLATE_KEY]
        if recorded != wanted:
            raise ValueError(f"{pack_path} was packed with {recorded}, not {settings}; remove it to repack")
        return
    legacy = settings["packing_algorithm"] == "first_fit_shuffle" and settings["dataset_kwargs"] == default_kwargs
    if not legacy:
        raise ValueError(
            f"{pack_path} has no {manifest.name}, so its packing settings are unknown and may not be {settings}; "
            "remove it to repack"
        )


def main():
    """Tokenize and pack a dataset root's training (and validation) JSONL into packed parquet."""
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
    args = parser.parse_args()

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
    if not args.no_chat:
        settings[TEMPLATE_KEY] = chat_template_sha256(tokenizer._tokenizer.chat_template)

    def pack_split(split_name, input_path, output_path):
        if output_path.exists():
            check_existing_pack(output_path, settings, default_kwargs)
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
        )
        pack_manifest_path(output_path).write_text(json.dumps(settings, indent=2, sort_keys=True) + "\n")
        logger.info(f"{split_name} split packed: {output_path}")

    def split_path(split):
        return layout.packed_parquet_path(dataset_root, args.tokenizer, args.pad_seq_to_mult, args.seq_length, split)

    pack_split("Training", train_path, split_path("training"))
    if not args.no_validation and val_path.exists():
        pack_split("Validation", val_path, split_path("validation"))
    elif not args.no_validation:
        logger.warning(f"No validation.jsonl found at {val_path}")

    logger.info("Done. Packed files at: %s", pack_dir)


if __name__ == "__main__":
    main()
