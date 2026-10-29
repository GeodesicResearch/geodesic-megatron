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
"""Facts about the Nemotron-H releases and their chat format that the pipeline scripts rely on.

Catalogue:
    CHAT_TEMPLATE_SOURCE_MAP: release id -> the instruct release whose chat template its exports carry.
    IM_END_TOKEN_ID: the id of ``<|im_end|>``, which ends an assistant turn of the chat format.
    NEMOTRON_H_MODELING_FILES: the remote-code modules of a release.
    NEMOTRON_H_AUTO_MAP: the config.json ``auto_map`` that points transformers at those modules.
    ENABLE_THINKING_ON_LINE, ENABLE_THINKING_OFF_LINE: the chat-template lines that default enable_thinking.
    THINK_STUB_GENERATION_PROMPT, PLAIN_GENERATION_PROMPT: the non-reasoning generation prompt, with and without the
        closed think block.
    BLOCK_TYPE_CHARS: layers_block_type entry -> its hybrid_override_pattern character.
"""

# Base model -> instruct model mapping for chat template sourcing.
# Base models don't include a chat_template; the instruct variant does.
CHAT_TEMPLATE_SOURCE_MAP = {
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
    "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-Base-BF16": "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16",
    "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16": "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16",
    "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-Base-BF16": "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16",
    "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16": "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16",
}

# The base release ships with eos_token_id=2 (`</s>`) only; chat-format SFT trains the model to emit `<|im_end|>` at
# end-of-turn, so generation must also stop on it.
IM_END_TOKEN_ID = 11

# Generic NemotronH remote-code artifacts (no project specifics) — the custom
# modeling modules and the auto_map that points HF/vLLM trust_remote_code at them.
NEMOTRON_H_MODELING_FILES = ("configuration_nemotron_h.py", "modeling_nemotron_h.py")
NEMOTRON_H_AUTO_MAP = {
    "AutoConfig": "configuration_nemotron_h.NemotronHConfig",
    "AutoModelForCausalLM": "modeling_nemotron_h.NemotronHForCausalLM",
}

ENABLE_THINKING_ON_LINE = "{%- set enable_thinking = enable_thinking if enable_thinking is defined else True %}"
ENABLE_THINKING_OFF_LINE = "{%- set enable_thinking = enable_thinking if enable_thinking is defined else False %}"

THINK_STUB_GENERATION_PROMPT = r"'<|im_start|>assistant\n<think></think>'"
PLAIN_GENERATION_PROMPT = r"'<|im_start|>assistant\n'"

BLOCK_TYPE_CHARS = {"mamba": "M", "moe": "E", "attention": "*", "mlp": "-", "dense": "-"}
