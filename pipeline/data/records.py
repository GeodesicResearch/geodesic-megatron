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
    detect_column_and_format: the text column and whether the dataset is chat or pretraining text.
    format_record: one example as a JSONL record.
    write_jsonl: a dataset as a JSONL file of records.
"""

import json


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


def write_jsonl(ds, output_path, text_column, format_type, keep_columns=()):
    """Write dataset to JSONL file in Megatron Bridge format."""
    with open(output_path, "w") as f:
        for i, example in enumerate(ds):
            record = format_record(example, text_column, format_type, keep_columns)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            if (i + 1) % 10000 == 0:
                print(f"  Written {i + 1}/{len(ds)} documents...", end="\r")
    print(f"  Written {len(ds)}/{len(ds)} documents    ")
