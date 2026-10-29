#!/usr/bin/env python3
"""Unified data pipeline for preparing HuggingFace datasets for Megatron Bridge training.

Handles the complete pipeline:
1. LOAD   — Download HF dataset (with retry, rate-limit handling, local file support)
2. DETECT — Auto-detect text vs messages column
3. COUNT  — Count tokens using Nemotron tokenizer (batched)
4. EXPORT — Save to JSONL in Megatron Bridge format
5. PACK   — Run pack_sft_dataset.py → .idx.parquet (chat/SFT format only)
6. VERIFY — Read back the packed parquet, decode tokens, log per-token (id, decoded,
            loss_mask) tables to W&B and a console preview. Loud warning when chat
            packs come out with mask density 100% (silent {% generation %} fallback).

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
import sys
import time
import traceback
from pathlib import Path

import yaml
from transformers import AutoTokenizer

from pipeline.data.count import count_tokens_batched
from pipeline.data.layout import slugify_dataset_name
from pipeline.data.load import load_source_dataset
from pipeline.data.packing import run_pack
from pipeline.data.records import detect_column_and_format, write_jsonl
from pipeline.data.verify import verify_packed_loss_mask


try:
    import wandb

    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

DEFAULT_TOKENIZER = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
DEFAULT_OUTPUT_BASE = "/projects/a5k/public/data"
WANDB_PROJECT = "megatron-datasets-processing"
WANDB_ENTITY = "geodesic"


def load_pipeline_config(path):
    """Pipeline parameters read from a YAML file, keyed by long-option name.

    Keys may use either the option spelling (``pad-seq-to-mult``) or the
    attribute spelling (``pad_seq_to_mult``).
    """
    with open(path) as f:
        config = yaml.safe_load(f)
    if config is None:
        return {}
    if not isinstance(config, dict):
        raise ValueError(f"{path} must contain a mapping of parameter names to values, got {type(config).__name__}")
    return {key.replace("-", "_"): value for key, value in config.items()}


def parse_args():  # noqa: D103
    parser = argparse.ArgumentParser(
        description="Unified data pipeline for preparing HuggingFace datasets for Megatron Bridge training"
    )

    # Dataset arguments
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help=(
            "YAML file supplying the parameters that define what is prepared — dataset, subset, "
            "split, revision, tokenizer, sequence length. A corpus whose identity matters should be "
            "described by one of these rather than by a shell command. Command-line flags override it."
        ),
    )
    parser.add_argument("--dataset", type=str, default=None, help="HuggingFace dataset name")
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
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help=(
            "Pin the HuggingFace dataset to a git revision (commit SHA, tag, or branch). "
            "Omitted means the current HEAD of the default branch, which moves whenever the "
            "dataset is re-pushed; pass an explicit commit SHA for any corpus whose token "
            "counts are recorded in a config or paper."
        ),
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
    parser.add_argument(
        "--packing-algorithm",
        type=str,
        default="first_fit_shuffle",
        help="Packing algorithm (packing_utils.PACKING_ALGOS); one_per_pack keeps one document per row "
        "(default: first_fit_shuffle)",
    )
    parser.add_argument(
        "--keep-columns",
        type=lambda value: [column for column in value.split(",") if column],
        default=[],
        help="Comma-separated dataset columns copied into every JSONL record (a list in --config), for per-row "
        "fields a dataset kwarg reads, e.g. think_close_kind for dataset-kwargs think_close_key",
    )
    parser.add_argument(
        "--dataset-kwargs",
        type=json.loads,
        default={},
        help="create_sft_dataset keyword arguments merged over the format's packing defaults: a JSON object on "
        "the command line, a mapping in --config (e.g. add_eos/prompt_template/strip_spaces for raw text)",
    )

    # Performance
    parser.add_argument("--num-proc", type=int, default=16, help="Parallel processes for dataset operations")
    parser.add_argument("--batch-size", type=int, default=10000, help="Batch size for token counting")
    parser.add_argument(
        "--download-workers", type=int, default=None, help="Workers for HF download (default: num-proc)"
    )

    config_path = parser.parse_known_args()[0].config
    if config_path:
        config = load_pipeline_config(config_path)
        # Rejected rather than ignored: a typo in a corpus definition would otherwise
        # prepare the wrong data and say nothing.
        recognised = set(vars(parser.parse_args([])))
        unknown = sorted(set(config) - recognised)
        if unknown:
            parser.error(f"unrecognised keys in {config_path}: {', '.join(unknown)}")
        parser.set_defaults(**config)

    args = parser.parse_args()

    if not args.dataset:
        parser.error("--dataset is required, on the command line or in --config")

    if args.download_workers is None:
        args.download_workers = args.num_proc

    return args


def dataset_display_name(dataset, subset=None):
    """Short display name for the dataset (last path component + subset)."""
    name = dataset.split("/")[-1]
    if subset:
        name = f"{name}__{subset}"
    return name


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
        "revision": args.revision,
        "config": args.config,
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
        "revision": args.revision,
        "config": args.config,
        "tokenizer": args.tokenizer,
        "seq_length": args.seq_length,
        "packing_algorithm": args.packing_algorithm,
        "dataset_kwargs": args.dataset_kwargs,
        "keep_columns": args.keep_columns,
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
    print(f"Revision:  {args.revision or 'HEAD (unpinned — moves when the dataset is re-pushed)'}")
    print(f"Tokenizer: {args.tokenizer}")
    print(f"Output:    {output_dir}")
    print("=" * 60)

    # ── Stage 1: LOAD ──────────────────────────────────────────────
    print("\n[1/6] LOAD - Loading dataset from HuggingFace...")
    load_start = time.time()

    try:
        ds = load_source_dataset(args)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        traceback.print_exc()
        results["status"] = "failed"
        results["error"] = str(e)
        return 1

    load_time = time.time() - load_start
    num_docs = len(ds)
    results["num_documents"] = num_docs
    results["load_time"] = load_time
    print(f"  Loaded {num_docs:,} documents in {load_time:.1f}s")

    # ── Stage 2: DETECT ────────────────────────────────────────────
    print("\n[2/6] DETECT - Detecting column and format...")

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
    missing_keep = [c for c in args.keep_columns if c not in ds.column_names]
    if missing_keep:
        print(f"Error: --keep-columns columns not found: {missing_keep}. Available: {ds.column_names}")
        return 1
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
        print("\n[3/6] COUNT - Skipped (--skip-count)")
        results["token_count"] = None
        count_time = 0
    else:
        print("\n[3/6] COUNT - Counting tokens...")
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
        print("\n[4/6] EXPORT - Skipped (--count-only)")
        print("[5/6] PACK - Skipped (--count-only)")
        print("[6/6] VERIFY - Skipped (--count-only)")
        results["status"] = "completed"
        results["elapsed_time"] = time.time() - start_time

        if wb_run:
            wb_run.summary.update(
                {
                    "num_documents": num_docs,
                    "token_count": results.get("token_count"),
                    "tokens_per_doc": results.get("tokens_per_doc"),
                    "status": "completed",
                    "packed": False,
                    "elapsed_time": results["elapsed_time"],
                    "load_time": load_time,
                    "count_time": count_time,
                }
            )
            wb_run.finish()

        print(f"\nResults: {json.dumps(results, indent=2)}")
        return 0

    # ── Stage 4: EXPORT ────────────────────────────────────────────
    print("\n[4/6] EXPORT - Saving to JSONL...")
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
    write_jsonl(train_ds, train_path, text_column, format_type, args.keep_columns)
    results["training_jsonl"] = str(train_path)
    results["training_docs"] = len(train_ds)

    # Write validation.jsonl
    if val_ds is not None:
        val_path = output_dir / "validation.jsonl"
        print(f"  Writing {val_path} ({len(val_ds):,} docs)...")
        write_jsonl(val_ds, val_path, text_column, format_type, args.keep_columns)
        results["validation_jsonl"] = str(val_path)
        results["validation_docs"] = len(val_ds)
    else:
        results["validation_jsonl"] = None
        results["validation_docs"] = 0

    export_time = time.time() - export_start
    results["export_time"] = export_time
    print(f"  Export time: {export_time:.1f}s")

    # Log sample examples to W&B table
    if wb_run and len(train_ds) >= 10:
        n_samples = min(20, len(train_ds))
        print(f"  Logging {n_samples} sample examples to W&B...")
        if format_type == "chat":
            table = wandb.Table(columns=["index", "system", "user", "assistant", "num_turns"])
            for i in range(n_samples):
                messages = train_ds[i][text_column]
                system = next((m["content"] for m in messages if m["role"] == "system"), "")
                user = next((m["content"] for m in messages if m["role"] == "user"), "")
                assistant = next((m["content"] for m in messages if m["role"] == "assistant"), "")
                table.add_data(i, system, user, assistant, len(messages))
        else:
            table = wandb.Table(columns=["index", "text_preview"])
            for i in range(n_samples):
                text = train_ds[i][text_column]
                table.add_data(i, text[:500] if isinstance(text, str) else str(text)[:500])
        wb_run.log({"sample_examples": table})

    # ── Stage 5: PACK ──────────────────────────────────────────────
    pack_time = 0
    if args.skip_pack:
        print("\n[5/6] PACK - Skipped (--skip-pack)")
        results["packed"] = False
    else:
        print(f"\n[5/6] PACK - Running pack_sft_dataset.py ({format_type} format)...")
        pack_start = time.time()

        success = run_pack(
            output_dir,
            args.tokenizer,
            args.seq_length,
            args.pad_seq_to_mult,
            has_validation,
            format_type,
            args.packing_algorithm,
            args.dataset_kwargs,
        )

        pack_time = time.time() - pack_start
        results["packed"] = success
        results["pack_time"] = pack_time

        if success:
            print(f"\n  Packing complete in {pack_time:.1f}s")
        else:
            results["status"] = "failed"
            results["error"] = "Packing failed"

    results["pack_time"] = pack_time

    # ── Stage 6: VERIFY ────────────────────────────────────────────
    if not args.skip_pack and results.get("packed"):
        print("\n[6/6] VERIFY - Reading packed parquet to inspect per-token loss mask...")
        verify_summary = verify_packed_loss_mask(
            output_dir=output_dir,
            tokenizer_id=args.tokenizer,
            seq_length=args.seq_length,
            pad_seq_to_mult=args.pad_seq_to_mult,
            format_type=format_type,
            wb_run=wb_run,
        )
        results.update(verify_summary)

    # ── Save results ───────────────────────────────────────────────
    results["status"] = "completed" if results.get("status") != "failed" else "failed"
    results["elapsed_time"] = time.time() - start_time

    results_path = output_dir / "pipeline_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    # W&B summary
    if wb_run:
        wb_run.summary.update(
            {
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
                "mask_density": results.get("verify_mask_density"),
                "mask_density_min": results.get("verify_density_min"),
                "mask_density_max": results.get("verify_density_max"),
                "verify_warning": results.get("verify_warning"),
            }
        )
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
        print(f"  dataset_root: {output_dir}")

    return 0 if results["status"] == "completed" else 1


if __name__ == "__main__":
    sys.exit(main())
