#!/usr/bin/env python3
"""Unified data pipeline for preparing HuggingFace datasets for Megatron Bridge training.

Handles the complete pipeline:
1. LOAD   — Download HF dataset (with retry, rate-limit handling, local file support)
2. DETECT — Auto-detect text vs messages column
3. COUNT  — Count tokens using Nemotron tokenizer (batched)
4. EXPORT — Save to JSONL in Megatron Bridge format
5. PACK   — Run pack_sft_dataset.py → .idx.parquet (chat/SFT format only)

Example usage:
    # Pretraining dataset
    python pipeline_data_prepare.py \
        --dataset geodesic-research/Nemotron-Pretraining-Specialized

    # SFT dataset with subset/split, validation split, and packing
    python pipeline_data_prepare.py \
        --dataset geodesic-research/discourse-grounded-misalignment-synthetic-scenario-data \
        --subset midtraining --split positive \
        --val-proportion 0.05 --seq-length 8192

    # Count tokens only (no disk writes)
    python pipeline_data_prepare.py \
        --dataset geodesic-research/Nemotron-Pretraining-Specialized \
        --count-only
"""

import argparse
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

import pandas as pd
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer  # noqa: I001


try:
    import wandb

    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

DEFAULT_TOKENIZER = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
DEFAULT_OUTPUT_BASE = "/projects/a5k/public/data"
WANDB_PROJECT = "megatron-datasets-processing"
WANDB_ENTITY = "geodesic"


def parse_args():  # noqa: D103
    parser = argparse.ArgumentParser(
        description="Unified data pipeline for preparing HuggingFace datasets for Megatron Bridge training"
    )

    # Dataset arguments
    parser.add_argument("--dataset", type=str, required=True, help="HuggingFace dataset name")
    parser.add_argument("--subset", type=str, default=None, help="Dataset config/subset name")
    parser.add_argument("--split", type=str, default="train", help="Dataset split (default: train)")
    parser.add_argument(
        "--data-files", type=str, default=None, help="Path to local file(s) to load directly (bypasses HF download)"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Subdirectory within the HuggingFace dataset repo to load (passed as data_dir to load_dataset)",
    )

    # Output arguments
    parser.add_argument("--output-dir", type=str, default=None, help="Override full output path")
    parser.add_argument("--output-base", type=str, default=DEFAULT_OUTPUT_BASE, help="Base output directory")

    # Column/format arguments
    parser.add_argument("--text-column", type=str, default=None, help="Override text column (auto-detects otherwise)")
    parser.add_argument(
        "--join-columns",
        type=str,
        default=None,
        help="Comma-separated columns to concatenate with blank line separator",
    )

    # Tokenizer arguments
    parser.add_argument("--tokenizer", type=str, default=DEFAULT_TOKENIZER, help="HF tokenizer for token counting")

    # Pipeline control
    parser.add_argument("--skip-count", action="store_true", help="Skip token counting")
    parser.add_argument("--skip-pack", action="store_true", help="Skip packing stage")
    parser.add_argument("--count-only", action="store_true", help="Only count tokens, skip export and pack")
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B logging")

    # Validation split
    parser.add_argument("--val-proportion", type=float, default=0.0, help="Fraction for validation split (default: 0)")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed (default: 1234)")

    # Packing arguments
    parser.add_argument("--seq-length", type=int, default=8192, help="Sequence length for packing (default: 8192)")
    parser.add_argument("--pad-seq-to-mult", type=int, default=1, help="Pad sequences to this multiple (default: 1)")

    # Performance
    parser.add_argument("--num-proc", type=int, default=16, help="Parallel processes for dataset operations")
    parser.add_argument("--batch-size", type=int, default=10000, help="Batch size for token counting")
    parser.add_argument("--download-workers", type=int, default=None, help="Workers for HF download (default: num-proc)")

    args = parser.parse_args()

    if args.download_workers is None:
        args.download_workers = args.num_proc

    return args


def slugify_dataset_name(dataset, subset=None):
    """Generate output directory name from dataset components.

    geodesic-research/Foo → geodesic-research__Foo
    geodesic-research/Foo + subset=bar → geodesic-research__Foo__bar
    """
    slug = dataset.replace("/", "__")
    if subset:
        slug = f"{slug}__{subset}"
    return slug


def dataset_display_name(dataset, subset=None):
    """Short display name for the dataset (last path component + subset)."""
    name = dataset.split("/")[-1]
    if subset:
        name = f"{name}__{subset}"
    return name


def detect_column_and_format(ds, text_column=None, join_columns=None):
    """Auto-detect the text column and output format.

    Returns (text_column, format_type) where format_type is 'pretraining' or 'chat'.
    """
    columns = ds.column_names

    if join_columns:
        return "text", "pretraining"

    if text_column:
        if text_column not in columns:
            raise ValueError(f"Specified --text-column '{text_column}' not found. Available: {columns}")
        if text_column == "messages":
            return "messages", "chat"
        return text_column, "pretraining"

    if "text" in columns:
        return "text", "pretraining"
    if "content" in columns:
        return "content", "pretraining"
    if "messages" in columns:
        return "messages", "chat"

    raise ValueError(f"Could not auto-detect text column. Available columns: {columns}. Use --text-column.")


def count_tokens_batched(ds, tokenizer, text_column, batch_size, format_type):
    """Count tokens in dataset using batched processing."""
    total_tokens = 0

    print(f"Counting tokens in batches of {batch_size}...")

    for i in range(0, len(ds), batch_size):
        batch = ds[i : i + batch_size]

        if format_type == "chat":
            texts = []
            for messages in batch[text_column]:
                try:
                    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
                    texts.append(text)
                except Exception:
                    texts.append(str(messages))
        else:
            texts = batch[text_column]

        encoded = tokenizer(texts, add_special_tokens=False, return_length=True)
        total_tokens += sum(encoded["length"])

        processed = min(i + batch_size, len(ds))
        print(f"  Processed {processed}/{len(ds)} documents...", end="\r")

    print()
    return total_tokens


def format_record(example, text_column, format_type):
    """Format a single example into the JSONL record for Megatron Bridge."""
    if format_type == "chat":
        return {"messages": [{"role": m["role"], "content": m["content"]} for m in example[text_column]]}
    else:
        return {"input": example[text_column], "output": ""}


def write_jsonl(ds, output_path, text_column, format_type):
    """Write dataset to JSONL file in Megatron Bridge format."""
    with open(output_path, "w") as f:
        for i, example in enumerate(ds):
            record = format_record(example, text_column, format_type)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            if (i + 1) % 10000 == 0:
                print(f"  Written {i + 1}/{len(ds)} documents...", end="\r")
    print(f"  Written {len(ds)}/{len(ds)} documents    ")


def run_pack(output_dir, tokenizer, seq_length, pad_seq_to_mult, has_validation, format_type):
    """Run pack_sft_dataset.py via subprocess."""
    script_path = Path(__file__).parent / "scripts" / "data" / "pack_sft_dataset.py"

    if not script_path.exists():
        print(f"\nError: pack_sft_dataset.py not found at {script_path}")
        return False

    cmd = [
        sys.executable,
        str(script_path),
        "--dataset-root",
        str(output_dir),
        "--tokenizer",
        tokenizer,
        "--seq-length",
        str(seq_length),
        "--pad-seq-to-mult",
        str(pad_seq_to_mult),
    ]

    if not has_validation:
        cmd.append("--no-validation")

    if format_type == "pretraining":
        cmd.append("--no-chat")

    print("\nRunning packing command:")
    print(f"  {' '.join(cmd)}")
    print()

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    for line in process.stdout:
        print(f"[PACK] {line}", end="")

    process.wait()

    if process.returncode != 0:
        print(f"\nError: pack_sft_dataset.py failed with return code {process.returncode}")
        return False

    return True


def init_wandb(args, format_type, output_dir):
    """Initialize W&B run if enabled."""
    if args.no_wandb or not HAS_WANDB:
        if not args.no_wandb and not HAS_WANDB:
            print("  Warning: wandb not installed, skipping W&B logging")
        return None

    dataset_slug = dataset_display_name(args.dataset, args.subset)
    tokenizer_slug = args.tokenizer.replace("/", "--")
    run_name = f"{dataset_slug}___{tokenizer_slug}"

    config = {
        "dataset": args.dataset,
        "subset": args.subset,
        "split": args.split,
        "tokenizer": args.tokenizer,
        "output_dir": str(output_dir),
        "text_column": args.text_column,
        "join_columns": args.join_columns,
        "val_proportion": args.val_proportion,
        "seed": args.seed,
        "seq_length": args.seq_length,
        "pad_seq_to_mult": args.pad_seq_to_mult,
        "skip_count": args.skip_count,
        "skip_pack": args.skip_pack,
        "count_only": args.count_only,
        "num_proc": args.num_proc,
        "batch_size": args.batch_size,
        "download_workers": args.download_workers,
        "format": format_type,
    }

    try:
        run = wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY, name=run_name, config=config)
        print(f"  W&B run: {run.url}")
        return run
    except Exception as e:
        print(f"  Warning: W&B init failed: {e}")
        return None


def main():  # noqa: D103
    args = parse_args()

    start_time = time.time()
    results = {
        "dataset": args.dataset,
        "subset": args.subset,
        "split": args.split,
        "tokenizer": args.tokenizer,
        "status": "started",
    }

    # Generate output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        dir_name = slugify_dataset_name(args.dataset, args.subset)
        output_dir = Path(args.output_base) / dir_name

    results["output_dir"] = str(output_dir)

    print("=" * 60)
    print("Megatron Bridge HuggingFace Data Pipeline")
    print("=" * 60)
    print(f"Dataset:   {args.dataset}")
    if args.subset:
        print(f"Subset:    {args.subset}")
    print(f"Split:     {args.split}")
    print(f"Tokenizer: {args.tokenizer}")
    print(f"Output:    {output_dir}")
    print("=" * 60)

    # ── Stage 1: LOAD ──────────────────────────────────────────────
    print("\n[1/5] LOAD - Loading dataset from HuggingFace...")
    load_start = time.time()

    max_retries = 10
    ds = None
    for attempt in range(1, max_retries + 1):
        try:
            if args.data_files:
                try:
                    ds = load_dataset(
                        "json",
                        data_files=args.data_files,
                        split="train",
                        num_proc=args.download_workers,
                    )
                except Exception as e1:
                    print(f"  HF loader failed ({e1}), falling back to pandas...")
                    try:
                        df = pd.read_json(args.data_files, lines=True)
                        ds = Dataset.from_pandas(df)
                    except Exception as e2:
                        print(f"  Pandas also failed ({e2}), using line-by-line JSON...")
                        rows = []
                        with open(args.data_files) as f:
                            for line in f:
                                rows.append(json.loads(line))
                        ds = Dataset.from_list(rows)
            else:
                load_kwargs = dict(
                    split=args.split,
                    num_proc=args.download_workers,
                )
                if args.data_dir:
                    load_kwargs["data_dir"] = args.data_dir
                ds = load_dataset(
                    args.dataset,
                    args.subset,
                    **load_kwargs,
                )
            break
        except Exception as e:
            error_str = str(e)
            if "429" in error_str and attempt < max_retries:
                wait = min(300 * (2 ** (attempt - 1)), 600)
                print(f"  Rate limited (attempt {attempt}/{max_retries}), waiting {wait}s...")
                time.sleep(wait)
            else:
                print(f"Error loading dataset: {e}")
                traceback.print_exc()
                results["status"] = "failed"
                results["error"] = error_str
                return 1

    load_time = time.time() - load_start
    num_docs = len(ds)
    results["num_documents"] = num_docs
    results["load_time"] = load_time
    print(f"  Loaded {num_docs:,} documents in {load_time:.1f}s")

    # ── Stage 2: DETECT ────────────────────────────────────────────
    print("\n[2/5] DETECT - Detecting column and format...")

    # Handle --join-columns preprocessing
    if args.join_columns:
        join_cols = [c.strip() for c in args.join_columns.split(",")]
        missing = [c for c in join_cols if c not in ds.column_names]
        if missing:
            print(f"Error: --join-columns columns not found: {missing}. Available: {ds.column_names}")
            return 1
        print(f"  Joining columns: {join_cols}")
        ds = ds.map(
            lambda x: {"text": "\n\n".join(str(x[c]) for c in join_cols if x[c])},
            num_proc=args.num_proc,
            desc="Joining columns",
        )

    text_column, format_type = detect_column_and_format(ds, args.text_column, args.join_columns)
    results["text_column"] = text_column
    results["format"] = format_type
    print(f"  Column: {text_column}")
    print(f"  Format: {format_type}")

    # Load HF tokenizer
    print(f"  Loading tokenizer: {args.tokenizer}")
    hf_tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    # Initialize W&B after detection so format is in config
    wb_run = init_wandb(args, format_type, output_dir)

    # ── Stage 3: COUNT ─────────────────────────────────────────────
    if args.skip_count:
        print("\n[3/5] COUNT - Skipped (--skip-count)")
        results["token_count"] = None
        count_time = 0
    else:
        print("\n[3/5] COUNT - Counting tokens...")
        count_start = time.time()

        total_tokens = count_tokens_batched(ds, hf_tokenizer, text_column, args.batch_size, format_type)

        count_time = time.time() - count_start
        results["token_count"] = total_tokens
        results["tokens_per_doc"] = total_tokens / num_docs if num_docs > 0 else 0
        print(f"  Total tokens: {total_tokens:,}")
        print(f"  Avg tokens/doc: {results['tokens_per_doc']:.1f}")
        print(f"  Count time: {count_time:.1f}s")

    results["count_time"] = count_time

    if args.count_only:
        print("\n[4/5] EXPORT - Skipped (--count-only)")
        print("[5/5] PACK - Skipped (--count-only)")
        results["status"] = "completed"
        results["elapsed_time"] = time.time() - start_time

        if wb_run:
            wb_run.summary.update({
                "num_documents": num_docs,
                "token_count": results.get("token_count"),
                "tokens_per_doc": results.get("tokens_per_doc"),
                "status": "completed",
                "packed": False,
                "elapsed_time": results["elapsed_time"],
                "load_time": load_time,
                "count_time": count_time,
            })
            wb_run.finish()

        print(f"\nResults: {json.dumps(results, indent=2)}")
        return 0

    # ── Stage 4: EXPORT ────────────────────────────────────────────
    print("\n[4/5] EXPORT - Saving to JSONL...")
    export_start = time.time()

    output_dir.mkdir(parents=True, exist_ok=True)

    has_validation = False
    if args.val_proportion > 0:
        print(f"  Splitting: {1 - args.val_proportion:.0%} train / {args.val_proportion:.0%} validation")
        split_ds = ds.train_test_split(test_size=args.val_proportion, seed=args.seed)
        train_ds = split_ds["train"]
        val_ds = split_ds["test"]
        has_validation = True
    else:
        train_ds = ds
        val_ds = None

    # Write training.jsonl
    train_path = output_dir / "training.jsonl"
    print(f"  Writing {train_path} ({len(train_ds):,} docs)...")
    write_jsonl(train_ds, train_path, text_column, format_type)
    results["training_jsonl"] = str(train_path)
    results["training_docs"] = len(train_ds)

    # Write validation.jsonl
    if val_ds is not None:
        val_path = output_dir / "validation.jsonl"
        print(f"  Writing {val_path} ({len(val_ds):,} docs)...")
        write_jsonl(val_ds, val_path, text_column, format_type)
        results["validation_jsonl"] = str(val_path)
        results["validation_docs"] = len(val_ds)
    else:
        results["validation_jsonl"] = None
        results["validation_docs"] = 0

    export_time = time.time() - export_start
    results["export_time"] = export_time
    print(f"  Export time: {export_time:.1f}s")

    # ── Stage 5: PACK ──────────────────────────────────────────────
    pack_time = 0
    if args.skip_pack:
        print("\n[5/5] PACK - Skipped (--skip-pack)")
        results["packed"] = False
    else:
        print(f"\n[5/5] PACK - Running pack_sft_dataset.py ({format_type} format)...")
        pack_start = time.time()

        success = run_pack(output_dir, args.tokenizer, args.seq_length, args.pad_seq_to_mult, has_validation, format_type)

        pack_time = time.time() - pack_start
        results["packed"] = success
        results["pack_time"] = pack_time

        if success:
            print(f"\n  Packing complete in {pack_time:.1f}s")
        else:
            results["status"] = "failed"
            results["error"] = "Packing failed"

    results["pack_time"] = pack_time

    # ── Save results ───────────────────────────────────────────────
    results["status"] = "completed" if results.get("status") != "failed" else "failed"
    results["elapsed_time"] = time.time() - start_time

    results_path = output_dir / "pipeline_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    # W&B summary
    if wb_run:
        wb_run.summary.update({
            "num_documents": num_docs,
            "token_count": results.get("token_count"),
            "tokens_per_doc": results.get("tokens_per_doc"),
            "training_docs": results.get("training_docs", 0),
            "validation_docs": results.get("validation_docs", 0),
            "status": results["status"],
            "packed": results.get("packed", False),
            "elapsed_time": results["elapsed_time"],
            "load_time": load_time,
            "count_time": count_time,
            "export_time": export_time,
            "pack_time": pack_time,
        })
        wb_run.finish()

    # ── Summary ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Pipeline Complete")
    print("=" * 60)
    print(f"Status:    {results['status']}")
    print(f"Format:    {format_type}")
    print(f"Documents: {num_docs:,}")
    if results.get("token_count"):
        print(f"Tokens:    {results['token_count']:,}")
    if has_validation:
        print(f"Train:     {results['training_docs']:,} docs")
        print(f"Valid:     {results['validation_docs']:,} docs")
    print(f"Elapsed:   {results['elapsed_time']:.1f}s")
    print(f"Results:   {results_path}")

    if results["status"] == "completed":
        print("\nFor Megatron Bridge training config:")
        print(f'  dataset_root: {output_dir}')

    return 0 if results["status"] == "completed" else 1


if __name__ == "__main__":
    sys.exit(main())
