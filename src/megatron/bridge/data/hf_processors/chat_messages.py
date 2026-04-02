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

"""Processing functions for HF chat-format (messages) datasets."""

from typing import Any, Optional

from megatron.bridge.training.tokenizers.tokenizer import MegatronTokenizer


def process_chat_messages_example(
    example: dict[str, Any], tokenizer: Optional[MegatronTokenizer] = None
) -> dict[str, Any]:
    """Process a single example from any HF dataset with a 'messages' column.

    Extracts the messages list (role/content dicts) for use with
    GPTSFTChatDataset and HF tokenizer chat templates. Works with any
    dataset using the standard HF chat format (e.g., allenai/Dolci-Instruct-SFT,
    HuggingFaceH4/ultrachat_200k, etc.).

    Args:
        example: Raw example containing a 'messages' list with
            {role, content} dicts.
        tokenizer: Optional tokenizer (not used in this processor).

    Returns:
        Dict with 'messages' key containing the conversation.
    """
    return {
        "messages": [
            {"role": msg["role"], "content": msg["content"]}
            for msg in example["messages"]
        ]
    }
