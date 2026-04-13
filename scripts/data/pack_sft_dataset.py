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

    # The dataset root should contain training.jsonl and optionally validation.jsonl
    # (produced by the HFDatasetBuilder preprocessing step or pipeline_data_prepare.py).
    #
    # Output goes to <dataset-root>/packed/<tokenizer-name>_pad_seq_to_mult<N>/
"""

import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Offline SFT dataset packing")
    parser.add_argument("--dataset-root", type=str, required=True, help="Directory with training.jsonl / validation.jsonl")
    parser.add_argument("--tokenizer", type=str, required=True, help="HuggingFace tokenizer model ID or path")
    parser.add_argument("--seq-length", type=int, default=16384, help="Sequence length (default: 16384)")
    parser.add_argument("--pad-seq-to-mult", type=int, default=1, help="Pad each sequence to this multiple (default: 1)")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed (default: 1234)")
    parser.add_argument("--no-validation", action="store_true", help="Skip validation split")
    parser.add_argument(
        "--no-chat",
        action="store_true",
        help="Use pretraining format (input/output JSONL) instead of chat format (messages JSONL)",
    )
    args = parser.parse_args()

    from megatron.bridge.data.datasets.packed_sequence import prepare_packed_sequence_data
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

    # Output directory
    tokenizer_name = args.tokenizer.replace("/", "--")
    pack_dir = dataset_root / "packed" / f"{tokenizer_name}_pad_seq_to_mult{args.pad_seq_to_mult}"
    pack_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = pack_dir / f"{args.seq_length}_metadata.jsonl"

    if args.no_chat:
        dataset_kwargs = {
            "chat": False,
            "answer_only_loss": False,
            "pad_to_max_length": False,
        }
        logger.info("Using pretraining format (chat=False, answer_only_loss=False)")
    else:
        dataset_kwargs = {
            "chat": True,
            "use_hf_tokenizer_chat_template": True,
            "answer_only_loss": True,
            "pad_to_max_length": False,
        }

    # Pack training split
    train_output = pack_dir / f"training_{args.seq_length}.idx.parquet"
    if train_output.exists():
        logger.info(f"Training parquet already exists: {train_output}")
    else:
        logger.info(f"Packing training split: {train_path} -> {train_output}")
        prepare_packed_sequence_data(
            input_path=train_path,
            output_path=train_output,
            output_metadata_path=metadata_path,
            packed_sequence_size=args.seq_length,
            tokenizer=tokenizer,
            max_seq_length=args.seq_length,
            seed=args.seed,
            dataset_kwargs=dataset_kwargs,
            pad_seq_to_mult=args.pad_seq_to_mult,
            num_tokenizer_workers=1,
        )
        logger.info(f"Training split packed: {train_output}")

    # Pack validation split
    if not args.no_validation and val_path.exists():
        val_output = pack_dir / f"validation_{args.seq_length}.idx.parquet"
        if val_output.exists():
            logger.info(f"Validation parquet already exists: {val_output}")
        else:
            logger.info(f"Packing validation split: {val_path} -> {val_output}")
            prepare_packed_sequence_data(
                input_path=val_path,
                output_path=val_output,
                output_metadata_path=metadata_path,
                packed_sequence_size=args.seq_length,
                tokenizer=tokenizer,
                max_seq_length=args.seq_length,
                seed=args.seed,
                dataset_kwargs=dataset_kwargs,
                pad_seq_to_mult=args.pad_seq_to_mult,
                num_tokenizer_workers=1,
            )
            logger.info(f"Validation split packed: {val_output}")
    elif not args.no_validation:
        logger.warning(f"No validation.jsonl found at {val_path}")

    logger.info("Done. Packed files at: %s", pack_dir)


if __name__ == "__main__":
    main()
