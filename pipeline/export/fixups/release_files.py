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
"""Shipping the --hf-model release's own inference tokenizer and generation files when the model trained on its
chat format.

Catalogue:
    RELEASE_INFERENCE_FILES: the release files an export takes over.
    TEMPLATE_PROBE_CONVERSATIONS, TEMPLATE_PROBE_TOOLS: the conversations two templates must render alike.
    templates_render_alike: whether two chat templates render every probe conversation identically.
    adopt_release_inference_files: copy the release's files into an export whose training template renders as its.
"""

import json
import os
import shutil
from pathlib import Path


#: The inference tokenizer and generation files of a release, which an export ships from its --hf-model release
#: when the training tokenizer's chat template renders as the release's (``adopt_release_inference_files``).
RELEASE_INFERENCE_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "chat_template.jinja",
    "generation_config.json",
)


#: Conversations on which two chat templates must render byte-identically to count as the same inference format:
#: system and multi-turn, assistant reasoning both as ``reasoning_content`` and inline, and a tool call.
TEMPLATE_PROBE_CONVERSATIONS = (
    [{"role": "user", "content": "What is 2 + 2?"}],
    [
        {"role": "system", "content": "You are a careful assistant."},
        {"role": "user", "content": "Name a prime."},
        {"role": "assistant", "content": "<think>\n2 is prime.\n</think>\n\nTwo."},
        {"role": "user", "content": "Another?"},
    ],
    [
        {"role": "user", "content": "Name a prime."},
        {"role": "assistant", "reasoning_content": "3 is prime.", "content": "Three."},
    ],
    [
        {"role": "user", "content": "Weather in Paris?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"type": "function", "function": {"name": "weather", "arguments": {"city": "Paris"}}}],
        },
        {"role": "tool", "content": "Sunny, 21C."},
    ],
)

TEMPLATE_PROBE_TOOLS = (
    {
        "type": "function",
        "function": {
            "name": "weather",
            "description": "Current weather in a city.",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
        },
    },
)


def _render_probes(template: str) -> list[str]:
    """``template`` on every probe conversation, with and without a generation prompt and thinking.

    A probe the template refuses (its ``raise_exception``, a jinja ``TemplateError``) renders as that error, so two
    templates that refuse the same probes can still compare alike; any other exception propagates.

    Raises:
        ValueError: the template refuses every probe, which would make any two such templates compare alike.
    """
    from jinja2.exceptions import TemplateError
    from transformers.utils.chat_template_utils import render_jinja_template

    renders, refused = [], 0
    for conversation in TEMPLATE_PROBE_CONVERSATIONS:
        tools = list(TEMPLATE_PROBE_TOOLS) if any("tool_calls" in m for m in conversation) else None
        for add_generation_prompt in (False, True):
            for enable_thinking in (False, True):
                try:
                    rendered = render_jinja_template(
                        conversations=[conversation],
                        tools=tools,
                        chat_template=template,
                        add_generation_prompt=add_generation_prompt,
                        enable_thinking=enable_thinking,
                    )[0][0]
                except TemplateError as error:
                    rendered = f"<{type(error).__name__}: {error}>"
                    refused += 1
                renders.append(rendered)
    if refused == len(renders):
        raise ValueError(f"the chat template refuses all {refused} probe conversations: {renders[0]}")
    return renders


def templates_render_alike(template: str, other: str) -> bool:
    """Whether two chat templates render every ``TEMPLATE_PROBE_CONVERSATIONS`` case byte-identically.

    A training template that differs from its release's only in training-only markup, such as the
    ``{% generation %}`` blocks that mark assistant tokens for the loss mask, renders alike.
    """
    return _render_probes(template) == _render_probes(other)


def adopt_release_inference_files(hf_path: Path, hf_model_id: str, training_template: str) -> bool:
    """Ship the --hf-model release's inference tokenizer and generation files when the model trained on its format.

    If the release's chat template renders as ``training_template`` (``templates_render_alike``), copies every
    ``RELEASE_INFERENCE_FILES`` file the release has into ``hf_path`` verbatim, replacing the training
    tokenizer's, and returns True. Otherwise (a release without a chat template, or a training format of its own)
    changes nothing and returns False. ``hf_model_id`` is a local release directory or a Hub id.

    Raises:
        FileNotFoundError: a file of the release is a dangling link (a cache missing its blob), or the release or
            the export has no config.json.
    """
    if Path(hf_model_id).is_dir():
        release_dir = Path(hf_model_id)
    else:
        from huggingface_hub import snapshot_download

        release_dir = Path(snapshot_download(hf_model_id, allow_patterns=[*RELEASE_INFERENCE_FILES, "config.json"]))
    for name in (*RELEASE_INFERENCE_FILES, "config.json"):
        path = release_dir / name
        if os.path.lexists(path) and not path.exists():
            raise FileNotFoundError(f"{path} is a dangling link: the release's {name} is missing from the cache")
    release_config, config_path = release_dir / "config.json", hf_path / "config.json"
    for path in (release_config, config_path):
        if not path.exists():
            raise FileNotFoundError(f"{path} does not exist: the token ids of config.json go with the release files")
    release_template = release_dir / "chat_template.jinja"
    if not release_template.exists():
        print(f"{hf_model_id} ships no chat_template.jinja: keeping the training tokenizer's files")
        return False
    if not templates_render_alike(release_template.read_text(), training_template):
        print(f"The training chat template renders differently from {hf_model_id}'s: keeping the training tokenizer's")
        return False
    for name in RELEASE_INFERENCE_FILES:
        if (release_dir / name).exists():
            shutil.copyfile(release_dir / name, hf_path / name)
            print(f"Copied {name} from {hf_model_id} (the training template renders as the release's)")
    # config.json's token ids go with these files: take the release's (an earlier export may have patched them).
    release_ids = json.loads(release_config.read_text())
    config = json.loads(config_path.read_text())
    token_ids = {k: release_ids[k] for k in ("bos_token_id", "eos_token_id", "pad_token_id") if k in release_ids}
    if any(config.get(k) != v for k, v in token_ids.items()):
        config.update(token_ids)
        config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n")
        print(f"Set config.json token ids to {hf_model_id}'s: {token_ids}")
    return True
