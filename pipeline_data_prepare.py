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
import subprocess
import sys
import time
import traceback
from pathlib import Path

import pandas as pd
import yaml
from datasets import Dataset, Value, load_dataset
from scripts.data.prepare_revisions import FULL_SHA, subset_revision
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

    # Performance
    parser.add_argument("--num-proc", type=int, default=16, help="Parallel processes for dataset operations")
    parser.add_argument("--batch-size", type=int, default=10000, help="Batch size for token counting")
    parser.add_argument(
        "--download-workers", type=int, default=None, help="Workers for HF download (default: num-proc)"
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

    if args.streaming:
        problems = streaming_refusals(args)
        if problems:
            parser.error("--streaming cannot honour this invocation:\n  - " + "\n  - ".join(problems))

    if args.download_workers is None:
        args.download_workers = args.num_proc

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
    return problems


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


def build_hub_load_kwargs(args):
    """Keyword arguments for the ``load_dataset`` call that pulls from the Hub.

    ``data_dir`` and ``revision`` are only passed when set: ``load_dataset``
    treats an explicit ``None`` differently from an absent argument for some
    builders. Omitting ``revision`` resolves the default branch's current HEAD,
    so a dataset that is re-pushed yields different data under the same command.

    With ``--streaming`` the call returns an ``IterableDataset`` that reads the data files as it
    is iterated, writing neither the files into the hub cache nor an Arrow cache; ``num_proc``
    is left out because ``load_dataset`` refuses it for a stream.
    """
    if args.streaming:
        kwargs = {"split": args.split, "streaming": True}
    else:
        kwargs = {"split": args.split, "num_proc": args.download_workers}
    if args.data_dir:
        kwargs["data_dir"] = args.data_dir
    if args.revision:
        kwargs["revision"] = args.revision
    return kwargs


# The columns a document is auto-detected in, in priority order, with the format each is exported in.
# Any other column, named with --text-column, is exported as pretraining text.
DOCUMENT_COLUMNS = {"text": "pretraining", "content": "pretraining", "messages": "chat"}


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
        return text_column, DOCUMENT_COLUMNS.get(text_column, "pretraining")

    for column, format_type in DOCUMENT_COLUMNS.items():
        if column in columns:
            return column, format_type

    raise ValueError(f"Could not auto-detect text column. Available columns: {columns}. Use --text-column.")


def detect_stream_column(ds, text_column=None):
    """The document column of a streamed dataset, chosen by ``detect_column_and_format`` from its declared features.

    A stream is not loaded before it is exported, so a wrong column would surface only after
    the download it was meant to save. What that function would settle by priority is refused
    instead: a stream that declares no features (its columns cannot be checked before reading
    it), more than one candidate column without ``--text-column``, a chat-format column, and a
    column that does not hold strings.
    """
    features = ds.features
    if features is None:
        raise ValueError(
            "--streaming: the stream declares no features, so its columns cannot be checked before the export "
            "starts (a JSON dataset without a dataset card declares none); stream a parquet dataset, or prepare "
            "this one without --streaming"
        )
    if text_column is None:
        candidates = [column for column in DOCUMENT_COLUMNS if column in features]
        if len(candidates) > 1:
            raise ValueError(
                f"--streaming: columns {candidates} could each be the document; name one with --text-column"
            )
    column, format_type = detect_column_and_format(ds, text_column)
    if format_type == "chat":
        raise ValueError(
            f"--streaming exports pretraining-format text only; '{column}' is a chat-format column, "
            "which is packed for SFT: prepare it without --streaming"
        )
    feature = features[column]
    if not (isinstance(feature, Value) and feature.dtype in ("string", "large_string")):
        raise ValueError(f"--streaming exports pretraining-format text only; column '{column}' holds {feature}")
    return column, format_type


def count_tokens_batched(ds, tokenizer, text_column, batch_size, format_type):
    """Count tokens in dataset using batched processing."""
    total_tokens = 0

    print(f"Counting tokens in batches of {batch_size}...")

    # Render through the same normalization the pack path uses (JSON-string or hybrid
    # tool_calls/tools handled identically) so the count reflects trained tokens. A silent
    # str(messages) fallback here once hid a 20% undercount when the schema changed.
    from megatron.bridge.data.datasets.utils import _convert_to_openai_messages, _normalize_tools_parameters

    fallbacks = 0
    has_tools = "tools" in ds.column_names
    for i in range(0, len(ds), batch_size):
        batch = ds[i : i + batch_size]

        if format_type == "chat":
            texts = []
            tools_col = batch.get("tools") if has_tools else None
            for j, messages in enumerate(batch[text_column]):
                try:
                    chat = _convert_to_openai_messages({"messages": messages})
                    tools = _normalize_tools_parameters(tools_col[j]) if tools_col is not None else None
                    text = tokenizer.apply_chat_template(
                        chat, tools=tools, tokenize=False, add_generation_prompt=False
                    )
                    texts.append(text)
                except Exception as e:
                    fallbacks += 1
                    if fallbacks <= 3:
                        print(f"\n  WARNING: chat render failed for doc {i + j} ({e}); counting str(messages)")
                    texts.append(str(messages))
        else:
            texts = batch[text_column]

        encoded = tokenizer(texts, add_special_tokens=False, return_length=True)
        total_tokens += sum(encoded["length"])

        processed = min(i + batch_size, len(ds))
        print(f"  Processed {processed}/{len(ds)} documents...", end="\r")

    print()
    if fallbacks:
        print(f"  WARNING: {fallbacks}/{len(ds)} docs failed chat rendering — token count is UNRELIABLE for them")
    return total_tokens


_CHAT_PASSTHROUGH_FIELDS = ("role", "content", "reasoning_content", "prefill", "tools", "tool_calls", "name")


def format_record(example, text_column, format_type):
    """Format a single example into the JSONL record for Megatron Bridge.

    For chat format, preserves message-level fields beyond role+content
    (reasoning_content, tool_calls, prefill, etc.) so the chat template can render
    them — omitting reasoning_content silently strips ALL think content from a
    reasoning mix (found 2026-06-11: pack shrank to 42% of n_tokens). Empty/None
    fields are dropped to keep the JSONL minimal. The example-level `tools` column
    (tool schemas) is preserved alongside `messages` when present.
    """
    if format_type == "chat":
        messages = []
        for m in example[text_column]:
            kept = {}
            for k in _CHAT_PASSTHROUGH_FIELDS:
                v = m.get(k)
                if v is None or v == "":
                    continue
                kept[k] = v
            messages.append(kept)
        record = {"messages": messages}
        tools = example.get("tools")
        if tools is not None and tools != "":
            record["tools"] = tools
        return record
    else:
        return {"input": example[text_column], "output": ""}


def write_jsonl(rows, output_path, text_column, format_type, total=None):
    """Write rows to a JSONL file in Megatron Bridge format and return how many were written.

    ``rows`` is a loaded ``Dataset`` (``total`` its length) or a streamed ``IterableDataset``,
    whose length is known only once its last row is written (``total`` None). Both exports go
    through this one loop, so a streamed corpus is written byte for byte as a loaded one is.
    """
    written = 0
    with open(output_path, "w") as f:
        for example in rows:
            record = format_record(example, text_column, format_type)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1
            if written % 10000 == 0:
                print(f"  Written {_progress(written, total)} documents...", end="\r")
    print(f"  Written {_progress(written, total)} documents    ")
    return written


def _progress(written, total):
    return f"{written}/{total}" if total is not None else f"{written:,}"


SAMPLE_EXAMPLES = 20  # rows logged to W&B as the export's sample table


def _keep_head(rows, head, n):
    """Yield every row of ``rows``, appending the first ``n`` to ``head`` (a stream cannot be indexed afterwards)."""
    for row in rows:
        if len(head) < n:
            head.append(row)
        yield row


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


def _decode_token(decode, token_id):
    """Decode a single token id with ``decode`` (a ``display_decoder``), escaping whitespace for table display."""
    raw = decode([int(token_id)])
    return raw.replace("\n", "\\n").replace("\t", "\\t").replace("\r", "\\r")


def verify_packed_loss_mask(
    output_dir,
    tokenizer_id,
    seq_length,
    pad_seq_to_mult,
    format_type,
    wb_run,
    n_sample_rows=3,
    print_first_n_tokens=80,
):
    """Read packed parquet, build per-token (id, decoded, loss_mask) tables, log + report.

    Returns a dict with summary stats. Prints a truncated per-token table for sample 0
    to stdout. Logs full per-token tables for the first n_sample_rows packed rows to W&B.
    Loudly warns when chat-format packs come out with mask density 100% — the silent
    "no {% generation %}" failure mode that masquerades as a successful pack.
    """
    import pyarrow.parquet as pq

    tokenizer_slug = tokenizer_id.replace("/", "--")
    parquet_dir = Path(output_dir) / "packed" / f"{tokenizer_slug}_pad_seq_to_mult{pad_seq_to_mult}"
    parquet_path = parquet_dir / f"training_{seq_length}.idx.parquet"

    if not parquet_path.exists():
        print(f"  [verify] skipped — parquet not found at {parquet_path}")
        return {"verify_status": "skipped_no_parquet"}

    table = pq.read_table(parquet_path, columns=["input_ids", "loss_mask"])
    n_rows = table.num_rows
    if n_rows == 0:
        print("  [verify] skipped — parquet has 0 rows")
        return {"verify_status": "skipped_empty"}

    from megatron.bridge.training.tokenizers.tokenizer import display_decoder

    decode = display_decoder(AutoTokenizer.from_pretrained(tokenizer_id))

    input_ids_col = table.column("input_ids").to_pylist()
    loss_mask_col = table.column("loss_mask").to_pylist()

    total_tokens = sum(len(r) for r in input_ids_col)
    total_unmasked = sum(int(v) for r in loss_mask_col for v in r)
    overall_density = total_unmasked / total_tokens if total_tokens else 0.0
    per_row_density = [sum(int(v) for v in r) / len(r) if r else 0.0 for r in loss_mask_col]

    summary = {
        "verify_status": "ok",
        "verify_parquet": str(parquet_path),
        "verify_rows": n_rows,
        "verify_total_tokens": total_tokens,
        "verify_unmasked_tokens": total_unmasked,
        "verify_mask_density": round(overall_density, 4),
        "verify_density_min": round(min(per_row_density), 4) if per_row_density else 0.0,
        "verify_density_max": round(max(per_row_density), 4) if per_row_density else 0.0,
    }

    print(f"\n  Loss-mask summary across {n_rows} packed row(s):")
    print(f"    total tokens:       {total_tokens:,}")
    print(f"    loss-bearing (=1):  {total_unmasked:,}")
    print(f"    overall density:    {overall_density:.1%}")
    print(f"    per-row density:    min={summary['verify_density_min']:.1%}  max={summary['verify_density_max']:.1%}")

    # Threshold rationale: when answer_only_loss silently falls back to all-1s
    # (chat template missing `{% generation %}`), packing_utils.py left-shifts
    # the mask per sample to `[1, ..., 1, 0]`, so a packed row with M samples
    # has density = (seq_length - M) / seq_length. At seq_length = 8192:
    #   1 sample/row  → 0.99988    32 samples/row → 0.99609
    #   2 samples/row → 0.99976    80 samples/row → 0.99023  (avg sample ~100 tokens)
    # We flag >= 0.99 because a healthy answer_only_loss SFT mix typically
    # sits at 0.10–0.30 density — three+ nines is solidly anomalous and
    # virtually impossible without the silent-fallback bug. (Pure 1.0 was
    # the original target but is unreachable: see packing_utils.py:267-271.)
    if format_type == "chat" and overall_density >= 0.99:
        print(
            "\n  ⚠ WARNING: chat-format pack with ~100% loss-mask density.\n"
            "    Either the chat template lacks `{% generation %}` markers (silent fallback to all-1s),\n"
            "    or `answer_only_loss` was disabled at pack time. The model will train on system+user\n"
            "    tokens, not just assistant turns. Inspect the per-token table below to confirm.\n"
        )
        summary["verify_warning"] = "chat_pack_density_100pct"

    sample0_ids = input_ids_col[0][:print_first_n_tokens]
    sample0_mask = loss_mask_col[0][:print_first_n_tokens]
    print(f"\n  First {len(sample0_ids)} tokens of packed row 0:")
    print(f"    {'pos':>4}  {'id':>7}  {'mask':>4}  decoded")
    for pos, (tid, m) in enumerate(zip(sample0_ids, sample0_mask)):
        decoded = _decode_token(decode, tid)
        if len(decoded) > 40:
            decoded = decoded[:37] + "..."
        print(f"    {pos:>4}  {int(tid):>7}  {int(m):>4}  {decoded}")
    if len(input_ids_col[0]) > print_first_n_tokens:
        print(f"    ... ({len(input_ids_col[0]) - print_first_n_tokens} more tokens in row 0)")

    if wb_run is not None:
        try:
            for r_idx in range(min(n_sample_rows, n_rows)):
                ids = input_ids_col[r_idx]
                mask = loss_mask_col[r_idx]
                wb_table = wandb.Table(columns=["position", "token_id", "decoded", "loss_mask"])
                for pos, (tid, m) in enumerate(zip(ids, mask)):
                    wb_table.add_data(pos, int(tid), _decode_token(decode, tid), int(m))
                wb_run.log({f"loss_mask_table/row_{r_idx}": wb_table})
            print(f"  Logged {min(n_sample_rows, n_rows)} per-token table(s) to W&B")
        except Exception as exc:  # noqa: BLE001
            print(f"  [verify] W&B table logging failed: {exc}")

    return summary


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
        "streaming": args.streaming,
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
    print("=" * 60)

    # ── Stage 1: LOAD ──────────────────────────────────────────────
    print("\n[1/6] LOAD - Loading dataset from HuggingFace...")
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
                ds = load_dataset(
                    args.dataset,
                    args.subset,
                    **build_hub_load_kwargs(args),
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

    if args.streaming:
        text_column, format_type = detect_stream_column(ds, args.text_column)
    else:
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
        head = []
        partial_path = train_path.with_name(train_path.name + ".partial")
        training_docs = write_jsonl(
            _keep_head(train_ds, head, SAMPLE_EXAMPLES), partial_path, text_column, format_type
        )
        os.replace(partial_path, train_path)
        num_docs = training_docs
        results["num_documents"] = num_docs
    else:
        print(f"  Writing {train_path} ({len(train_ds):,} docs)...")
        training_docs = write_jsonl(train_ds, train_path, text_column, format_type, total=len(train_ds))
    results["training_jsonl"] = str(train_path)
    results["training_docs"] = training_docs

    # Write validation.jsonl
    if val_ds is not None:
        val_path = output_dir / "validation.jsonl"
        print(f"  Writing {val_path} ({len(val_ds):,} docs)...")
        write_jsonl(val_ds, val_path, text_column, format_type, total=len(val_ds))
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
            output_dir, args.tokenizer, args.seq_length, args.pad_seq_to_mult, has_validation, format_type
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
