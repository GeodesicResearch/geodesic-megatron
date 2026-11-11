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
    opens_with_tag: whether a chat message is an assistant turn that opens with a tag.
    split_inline_reasoning: one chat message with a leading tagged span moved to ``reasoning_content``.
    LIFTED, UNSPLIT: the keys under which ``format_record`` counts the turns that open with the tag.
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


def opens_with_tag(message, open_tag):
    """Whether a chat message is an assistant turn whose ``content`` starts with ``open_tag``, whitespace aside."""
    return message.get("role") == "assistant" and (message.get("content") or "").lstrip().startswith(open_tag)


def split_inline_reasoning(message, open_tag, close_tag):
    """Return a chat message with the tagged span that opens an assistant turn moved to ``reasoning_content``.

    The span starts at ``open_tag`` at the start of ``content`` (leading whitespace aside) and ends at the first
    ``close_tag``. ``reasoning_content`` becomes the text between them and ``content`` the text after, both stripped.
    Every other message is returned as the same object: another role, no span at the start, a span that is never
    closed, or a ``reasoning_content`` that already holds text.

    The rule is that of the dataset builder's ``split_inline_reasoning`` kernel
    (src/dataset_builder/transforms/kernels/column/split_inline_reasoning.py, sha256 77d5e62c6bbc), so a corpus
    prepared here and one built there split a turn in the same way.
    """
    if not opens_with_tag(message, open_tag):
        return message
    content = message["content"].lstrip()
    reasoning, separator, answer = content[len(open_tag) :].partition(close_tag)
    if not separator or (message.get("reasoning_content") or "").strip():
        return message
    return {**message, "content": answer.strip(), "reasoning_content": reasoning.strip()}


# The keys of the counts that ``format_record`` keeps of the assistant turns that open with the tag.
LIFTED, UNSPLIT = "lifted", "unsplit"


def format_record(
    example, text_column, format_type, keep_columns=(), inline_reasoning_tags=None, inline_reasoning_counts=None
):
    """Format a single example into the JSONL record for Megatron Bridge.

    For chat format, preserves message-level fields beyond role+content
    (reasoning_content, tool_calls, prefill, etc.) so the chat template can render
    them — omitting reasoning_content silently strips ALL think content from a
    reasoning mix (found 2026-06-11: pack shrank to 42% of n_tokens). Empty/None
    fields are dropped to keep the JSONL minimal. The example-level `tools` column
    (tool schemas) is preserved alongside `messages` when present.
    Each of ``keep_columns`` is copied into the record under its own name (e.g. a per-row
    field a dataset kwarg reads, such as ``think_close_key``).
    ``inline_reasoning_tags``, an (open tag, close tag) pair, applies ``split_inline_reasoning`` to every chat
    message before its empty fields are dropped; None leaves the messages as they are. With the pair,
    ``inline_reasoning_counts`` (a ``collections.Counter``) is required: every assistant turn that opens with the tag
    adds one to its ``LIFTED`` key when its span was moved and to its ``UNSPLIT`` key when the rule left it as it is.
    """
    if format_type == "chat":
        messages = []
        for m in example[text_column]:
            if inline_reasoning_tags is not None:
                split = split_inline_reasoning(m, *inline_reasoning_tags)
                if opens_with_tag(m, inline_reasoning_tags[0]):
                    inline_reasoning_counts[UNSPLIT if split is m else LIFTED] += 1
                m = split
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


def write_jsonl(
    rows,
    output_path,
    text_column,
    format_type,
    keep_columns,
    total=None,
    inline_reasoning_tags=None,
    inline_reasoning_counts=None,
):
    """Write rows to a JSONL file in Megatron Bridge format and return how many were written.

    ``rows`` is a loaded ``Dataset`` (``total`` its length) or a streamed ``IterableDataset``,
    whose length is known only once its last row is written (``total`` None). Both exports go
    through this one loop, so a streamed corpus is written byte for byte as a loaded one is.
    ``keep_columns``, ``inline_reasoning_tags`` and ``inline_reasoning_counts`` reach every record through
    ``format_record`` (a stream keeps no column and splits no span).
    """
    written = 0
    with open(output_path, "w") as f:
        for example in rows:
            record = format_record(
                example, text_column, format_type, keep_columns, inline_reasoning_tags, inline_reasoning_counts
            )
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
