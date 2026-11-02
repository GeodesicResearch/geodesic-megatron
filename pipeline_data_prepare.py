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

    # Pretraining corpus streamed from a pinned Hub revision straight to training.jsonl:
    # no hub-cache copy of the parquet and no datasets Arrow cache (see --streaming)
    python pipeline_data_prepare.py \
        --dataset org/corpus --subset name --revision <40-character commit SHA> \
        --streaming --skip-pack --skip-count --val-proportion 0
"""

import argparse
import json
import os
import re
import sys
import time
import traceback
from pathlib import Path

import yaml
from scripts.data.hub_read_retry import streamed_reads_retried
from scripts.data.prepare_revisions import FULL_SHA, require_full_sha, subset_revision
from transformers import AutoTokenizer

from pipeline.data.count import count_tokens_batched
from pipeline.data.layout import slugify_dataset_name
from pipeline.data.load import load_source_dataset
from pipeline.data.packing import run_pack
from pipeline.data.records import (
    SAMPLE_EXAMPLES,
    detect_column_and_format,
    detect_stream_column,
    keep_head,
    write_jsonl,
)
from pipeline.data.stream import export_parquet_stream, parquet_stream_files
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
            "described by one of these rather than by a shell command. Command-line flags override it. "
            "In place of one `revision`, the file may pin each subset at its own commit with "
            "`revisions: {<subset>: <40-character SHA>}`; a subset it does not pin is refused "
            "(scripts/data/prepare_revisions.py)."
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
    parser.add_argument(
        "--tokenizer-revision",
        type=str,
        default=None,
        help=(
            "Pin --tokenizer to a full 40-character commit SHA: the prepare loads that commit and records it, and "
            "the tokenize job of the same prepare config loads it too (scripts/data/prepare_revisions.py). Omitted "
            "means the tokenizer's default branch. A pinned tokenizer cannot pack: the pack directory is named by "
            "the tokenizer alone, so --skip-pack is required."
        ),
    )

    # Pipeline control
    parser.add_argument("--skip-count", action="store_true", help="Skip token counting")
    parser.add_argument("--skip-pack", action="store_true", help="Skip packing stage")
    parser.add_argument("--count-only", action="store_true", help="Only count tokens, skip export and pack")
    parser.add_argument(
        "--streaming",
        action="store_true",
        help=(
            "Stream the pinned Hub revision straight into training.jsonl instead of loading it: no "
            "hub-cache copy of the data files and no datasets Arrow cache, so prepare writes the "
            "corpus once (its JSONL) instead of three times. For pretraining-format "
            "(.bin/.idx) corpora only. It needs --revision pinned to a full commit SHA, --skip-pack, "
            "--skip-count, --val-proportion 0 and a plain split name, and it refuses --count-only, "
            "--data-files, --join-columns and chat-format (messages) columns."
        ),
    )
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
    parser.add_argument(
        "--stream-workers",
        type=int,
        default=1,
        help=(
            "With --streaming on a parquet config: how many processes stream its files at once. The config's files, "
            "in the order the loader reads them, are cut into this many contiguous groups, each streamed by a "
            "process of its own, and the groups' output is joined in order, so training.jsonl holds the bytes one "
            "stream would have written. One stream runs at the speed of one connection's range reads, so the export "
            "is about this many times faster, until the Hub limits it."
        ),
    )
    parser.add_argument(
        "--hub-read-attempts",
        type=int,
        default=20,
        help=(
            "How many times one read of a Hub file is tried before its error ends the prepare, when the error is "
            "one a repeated read can outlast: a dropped or refused connection, a timeout, or a 5xx or 429 from the "
            "Hub (scripts/data/hub_read_retry.py). It covers the reads of a parquet or JSON stream and of the "
            "parquet footers. 1 repeats nothing."
        ),
    )
    parser.add_argument(
        "--hub-read-first-wait-seconds",
        type=float,
        default=5.0,
        help=(
            "The wait before a Hub read's second attempt; it doubles for each attempt after, up to 60 s. With the "
            "default 20 attempts a read outlasts about 16 minutes of failures."
        ),
    )

    config_path = parser.parse_known_args()[0].config
    pins = {}
    if config_path:
        config = load_pipeline_config(config_path)
        # A per-subset pin (`revisions`) has no flag: it is resolved below, once the subset, which
        # the command line may name, is known. A `revision` beside it is taken out with it, so that
        # the two are refused together rather than one silently set as the default.
        if "revisions" in config:
            pins = {key: config.pop(key) for key in ("revision", "revisions") if key in config}
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

    if pins:
        try:
            pinned = subset_revision(pins, args.subset, config_path)
        except ValueError as error:
            parser.error(str(error))
        if args.revision is None:  # a --revision flag overrides the file, as every flag does
            args.revision = pinned

    if args.tokenizer_revision is not None:
        try:
            require_full_sha(args.tokenizer_revision, "--tokenizer-revision")
        except ValueError as error:
            parser.error(str(error))
        if not args.skip_pack:
            parser.error(
                "--tokenizer-revision cannot pack: pack_sft_dataset.py names its directory by the tokenizer alone, "
                "so two commits of one tokenizer would share it; set --skip-pack"
            )

    if args.streaming:
        problems = streaming_refusals(args)
        if problems:
            parser.error("--streaming cannot honour this invocation:\n  - " + "\n  - ".join(problems))

    if args.download_workers is None:
        args.download_workers = args.num_proc

    if args.hub_read_attempts < 1:
        parser.error(f"--hub-read-attempts must be at least 1, not {args.hub_read_attempts}")

    if args.stream_workers < 1:
        parser.error(f"--stream-workers must be at least 1, not {args.stream_workers}")
    if args.stream_workers > 1 and not args.streaming:
        parser.error("--stream-workers divides a stream among processes: it needs --streaming")

    return args


# The datasets library's own rule for a split name (``datasets.splits._split_re``): a plain name,
# so neither slice syntax (``train[0:100]``, ``train[:10%]``) nor a ``+`` combination.
_SPLIT_NAME = re.compile(r"\w+(\.\w+)*")


def streaming_refusals(args):
    """Why ``--streaming`` cannot honour these arguments: one message per refused option, empty when it can.

    A streamed dataset is an ``IterableDataset``: it has no length, no random access and no
    ``map`` over the whole of it before export, and nothing of it is on disk except the JSONL
    being written. Every option whose stage needs one of those is refused here rather than
    quietly skipped or served by a fallback that would materialise the data after all.
    """
    problems = []
    # A full commit SHA: a branch or tag can move while a stream that takes hours is still reading it.
    if not args.revision or not FULL_SHA.fullmatch(args.revision):
        problems.append(
            f"--revision must pin a full 40-character commit SHA, got {args.revision!r}: a stream reads "
            "whatever the revision names while it runs, so no revision, a branch or a tag is not reproducible"
        )
    if not _SPLIT_NAME.fullmatch(args.split):
        problems.append(
            f"--split {args.split!r} is slice or combination syntax; a stream reads one whole named split, "
            "so a sliced corpus (shard_mode slice) cannot stream: publish its slices as separate configs instead"
        )
    if args.data_files:
        problems.append("--data-files loads local files through its own loader; --streaming reads a Hub revision")
    if args.join_columns:
        problems.append("--join-columns rewrites every row with Dataset.map before export, which a stream cannot run")
    if args.count_only:
        problems.append("--count-only runs only the COUNT stage, which --streaming does not run")
    if not args.skip_count:
        problems.append(
            "the COUNT stage is refused: it tokenizes the loaded dataset before export, a second pass over the "
            "stream; set --skip-count (the tokenize step's .provenance.json records the exact token count)"
        )
    if not args.skip_pack:
        problems.append(
            "packing is refused: --streaming exports a pretraining-format JSONL for the tokenize step; set --skip-pack"
        )
    if args.val_proportion > 0:
        problems.append(
            f"--val-proportion {args.val_proportion} needs a random split of the loaded dataset "
            "(train_test_split), which a stream cannot take; set it to 0"
        )
    if args.keep_columns:
        problems.append(
            f"--keep-columns {','.join(args.keep_columns)}: a stream is read projected to its document column, "
            "so it holds no other column to copy into the records"
        )
    return problems


def dataset_display_name(dataset, subset=None):
    """Short display name for the dataset (last path component + subset)."""
    name = dataset.split("/")[-1]
    if subset:
        name = f"{name}__{subset}"
    return name


def log_sample_examples(wb_run, examples, text_column, format_type):
    """Log the export's first rows to W&B as a table."""
    print(f"  Logging {len(examples)} sample examples to W&B...")
    if format_type == "chat":
        table = wandb.Table(columns=["index", "system", "user", "assistant", "num_turns"])
        for i, example in enumerate(examples):
            messages = example[text_column]
            system = next((m["content"] for m in messages if m["role"] == "system"), "")
            user = next((m["content"] for m in messages if m["role"] == "user"), "")
            assistant = next((m["content"] for m in messages if m["role"] == "assistant"), "")
            table.add_data(i, system, user, assistant, len(messages))
    else:
        table = wandb.Table(columns=["index", "text_preview"])
        for i, example in enumerate(examples):
            text = example[text_column]
            table.add_data(i, text[:500] if isinstance(text, str) else str(text)[:500])
    wb_run.log({"sample_examples": table})


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
        "tokenizer_revision": args.tokenizer_revision,
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
        "streaming": args.streaming,
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


def run_pipeline(args):
    """Run the pipeline's stages as ``args`` describe them and return the process's exit status."""
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
        "tokenizer_revision": args.tokenizer_revision,
        "streaming": args.streaming,
        "stream_workers": args.stream_workers,
        "stream_group_documents": None,
        "hub_read_retry": {
            "attempts": args.hub_read_attempts,
            "first_wait_seconds": args.hub_read_first_wait_seconds,
        },
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
    if args.streaming:
        print("Streaming: yes — rows go straight to training.jsonl, no hub-cache or Arrow copy of the data")
    print(
        f"Hub reads: each tried up to {args.hub_read_attempts} times on a transient Hub error "
        f"(first wait {args.hub_read_first_wait_seconds:g} s)"
    )
    print("=" * 60)

    # ── Stage 1: LOAD ──────────────────────────────────────────────
    print("\n[1/6] LOAD - Loading dataset from HuggingFace...")
    load_start = time.time()

    try:
        ds, results["features"] = load_source_dataset(args)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        traceback.print_exc()
        results["status"] = "failed"
        results["error"] = str(e)
        return 1

    load_time = time.time() - load_start
    if args.streaming:
        # An IterableDataset has no length: the export loop counts the documents, and this
        # placeholder keeps the results record's keys in the order a loaded export writes them.
        num_docs = None
        print(f"  Opened the stream in {load_time:.1f}s; its documents are counted as they are exported")
    else:
        num_docs = len(ds)
        print(f"  Loaded {num_docs:,} documents in {load_time:.1f}s")
    results["num_documents"] = num_docs
    results["load_time"] = load_time

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

    stream_files = None
    if args.streaming:
        text_column, format_type = detect_stream_column(ds, args.text_column)
        stream_files = parquet_stream_files(ds, args)
        if stream_files is None and args.stream_workers > 1:
            raise ValueError(
                f"--stream-workers {args.stream_workers}: only a parquet config's files can be divided among "
                f"processes, and this stream is read by the {ds.info.builder_name} loader; stream it with one"
            )
    else:
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
    print(f"  Loading tokenizer: {args.tokenizer} @ {args.tokenizer_revision or 'default branch'}")
    hf_tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, revision=args.tokenizer_revision)

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
    if args.streaming:
        # Written under a temporary name and renamed once the stream is exhausted: a stream that
        # fails part-way (hours into a network read) leaves no truncated training.jsonl behind.
        print(f"  Streaming into {train_path}...")
        partial_path = train_path.with_name(train_path.name + ".partial")
        if stream_files is None:
            head = []
            training_docs = write_jsonl(
                keep_head(train_ds, head, SAMPLE_EXAMPLES), partial_path, text_column, format_type, ()
            )
        else:
            training_docs, head, results["stream_group_documents"] = export_parquet_stream(
                stream_files,
                args.stream_workers,
                text_column,
                train_ds.features[text_column],
                format_type,
                partial_path,
                (args.hub_read_attempts, args.hub_read_first_wait_seconds),
            )
        os.replace(partial_path, train_path)
        num_docs = training_docs
        results["num_documents"] = num_docs
    else:
        print(f"  Writing {train_path} ({len(train_ds):,} docs)...")
        training_docs = write_jsonl(
            train_ds, train_path, text_column, format_type, args.keep_columns, total=len(train_ds)
        )
    results["training_jsonl"] = str(train_path)
    results["training_docs"] = training_docs

    # Write validation.jsonl
    if val_ds is not None:
        val_path = output_dir / "validation.jsonl"
        print(f"  Writing {val_path} ({len(val_ds):,} docs)...")
        write_jsonl(val_ds, val_path, text_column, format_type, args.keep_columns, total=len(val_ds))
        results["validation_jsonl"] = str(val_path)
        results["validation_docs"] = len(val_ds)
    else:
        results["validation_jsonl"] = None
        results["validation_docs"] = 0

    export_time = time.time() - export_start
    results["export_time"] = export_time
    print(f"  Export time: {export_time:.1f}s")

    # Log sample examples to W&B table
    if wb_run and training_docs >= 10:
        n_samples = min(SAMPLE_EXAMPLES, training_docs)
        examples = head[:n_samples] if args.streaming else [train_ds[i] for i in range(n_samples)]
        log_sample_examples(wb_run, examples, text_column, format_type)

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


def main():  # noqa: D103
    args = parse_args()
    # A stream opens its files as it is read, which the stages do from the load to the end of the export.
    with streamed_reads_retried(args.hub_read_attempts, args.hub_read_first_wait_seconds):
        return run_pipeline(args)


if __name__ == "__main__":
    sys.exit(main())
