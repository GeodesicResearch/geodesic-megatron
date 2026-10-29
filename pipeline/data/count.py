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
"""Token counts of a dataset.

Catalogue:
    count_tokens_batched: the token count of a dataset's text column, chat rendered as the pack path renders it.
"""


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
