# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The text column and format of a dataset, and the JSONL records Megatron Bridge reads.

Catalogue:
    DOCUMENT_COLUMNS: the columns a document is auto-detected in, in priority order, with each one's format.
    detect_column_and_format: the text column and whether the dataset is chat or pretraining text.
    detect_stream_column: the document column of a streamed dataset, refusing what a stream cannot check first.
    format_record: one example as a JSONL record.
    write_jsonl: a loaded or streamed dataset as a JSONL file of records, returning how many were written.
    SAMPLE_EXAMPLES: how many of the export's first rows are logged to W&B.
    keep_head: every row of a dataset, passed through while its first rows are kept aside.
"""

import json

from datasets import Value


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


_CHAT_PASSTHROUGH_FIELDS = ("role", "content", "reasoning_content", "prefill", "tools", "tool_calls", "name")


def format_record(example, text_column, format_type, keep_columns=()):
    """Format a single example into the JSONL record for Megatron Bridge.

    For chat format, preserves message-level fields beyond role+content
    (reasoning_content, tool_calls, prefill, etc.) so the chat template can render
    them — omitting reasoning_content silently strips ALL think content from a
    reasoning mix (found 2026-06-11: pack shrank to 42% of n_tokens). Empty/None
    fields are dropped to keep the JSONL minimal. The example-level `tools` column
    (tool schemas) is preserved alongside `messages` when present.
    Each of ``keep_columns`` is copied into the record under its own name (e.g. a per-row
    field a dataset kwarg reads, such as ``think_close_key``).
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
    else:
        record = {"input": example[text_column], "output": ""}
    for column in keep_columns:
        record[column] = example[column]
    return record


def write_jsonl(rows, output_path, text_column, format_type, keep_columns, total=None):
    """Write rows to a JSONL file in Megatron Bridge format and return how many were written.

    ``rows`` is a loaded ``Dataset`` (``total`` its length) or a streamed ``IterableDataset``,
    whose length is known only once its last row is written (``total`` None). Both exports go
    through this one loop, so a streamed corpus is written byte for byte as a loaded one is.
    ``keep_columns`` go into every record as ``format_record`` copies them (a stream keeps none).
    """
    written = 0
    with open(output_path, "w") as f:
        for example in rows:
            record = format_record(example, text_column, format_type, keep_columns)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1
            if written % 10000 == 0:
                print(f"  Written {_progress(written, total)} documents...", end="\r")
    print(f"  Written {_progress(written, total)} documents    ")
    return written


def _progress(written, total):
    return f"{written}/{total}" if total is not None else f"{written:,}"


SAMPLE_EXAMPLES = 20  # rows logged to W&B as the export's sample table


def keep_head(rows, head, n):
    """Yield every row of ``rows``, appending the first ``n`` to ``head`` (a stream cannot be indexed afterwards)."""
    for row in rows:
        if len(head) < n:
            head.append(row)
        yield row
