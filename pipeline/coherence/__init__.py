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
"""Generation backends of pipeline_coherence_test.py, one module each, and the chat messages they share.

Catalogue:
    build_chat_messages: the message list of one prompt, with an optional system prompt.
    hf: transformers' text-generation pipeline on this node's GPUs.
    endpoint: a running OpenAI-compatible server, over HTTP.
    megatron: greedy decoding from a sharded Megatron checkpoint under torchrun.
"""


def build_chat_messages(prompt: str, system_prompt: str | None) -> list[dict]:
    """Standard chat message list shared by all backends."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    return messages
