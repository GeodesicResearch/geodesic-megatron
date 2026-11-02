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
"""The export's chat_template.jinja: its ``enable_thinking`` default and generation prompt, and its embedded copy.

Catalogue:
    set_enable_thinking_default: a template text with enable_thinking defaulting to the training regime.
    strip_think_stub: a template text whose non-reasoning generation prompt has no closed think block.
    fix_chat_template_file: apply both to chat_template.jinja.
    sync_embedded_template: copy chat_template.jinja into tokenizer_config.json's embedded template.
"""

import json
from pathlib import Path

from pipeline.nemotron import (
    ENABLE_THINKING_OFF_LINE,
    ENABLE_THINKING_ON_LINE,
    PLAIN_GENERATION_PROMPT,
    THINK_STUB_GENERATION_PROMPT,
)


def set_enable_thinking_default(ct: str, reasoning: bool, size: int) -> tuple[str, bool]:
    """``ct`` with enable_thinking defaulting to ``reasoning``, and whether it changed.

    Prints what it found, naming the template's ``size`` in bytes. A template without either default line is left
    as it is.
    """
    # Sync the template's `enable_thinking` default to the model's training
    # regime. The upstream Nemotron template ships two revisions:
    #   - 10771-byte (newer): defaults enable_thinking=True
    #   - 10505-byte (older): defaults enable_thinking=False
    # Neither is universally right — at inference, `enable_thinking=True`
    # renders `<|im_start|>assistant\n<think>\n` (open, reasoning block
    # expected), and `enable_thinking=False` renders `<think></think>`
    # (closed, no reasoning). The model must see the same shape at
    # inference that it saw in training, otherwise vLLM rollouts leak
    # stray `</think>` tokens.
    #   - Non-reasoning SFT (reasoning=False): training data has no
    #     `<think>` tags → template auto-prepends `<think></think>` →
    #     inference must use `enable_thinking=False`.
    #   - Reasoning SFT (reasoning=True): training data has `<think>...
    #     </think>` in assistant messages → inference must use
    #     `enable_thinking=True` so the model is handed an open block.
    desired = ENABLE_THINKING_ON_LINE if reasoning else ENABLE_THINKING_OFF_LINE
    undesired = ENABLE_THINKING_OFF_LINE if reasoning else ENABLE_THINKING_ON_LINE
    if undesired in ct:
        ct = ct.replace(undesired, desired, 1)
        print(
            f"Preserved upstream chat_template.jinja ({size} bytes); set enable_thinking default → {'True' if reasoning else 'False'}"
        )
        return ct, True
    if desired in ct:
        print(
            f"Preserved upstream chat_template.jinja ({size} bytes); enable_thinking default already {'True' if reasoning else 'False'}"
        )
    else:
        print(
            f"Preserved upstream chat_template.jinja ({size} bytes); no enable_thinking set-line found (non-standard template)"
        )
    return ct, False


def strip_think_stub(ct: str) -> tuple[str, bool]:
    """``ct`` with the closed ``<think></think>`` removed from its non-reasoning generation prompt, and whether it
    changed."""
    # Upstream Nemotron emits
    #   `<|im_start|>assistant\n<think></think>`
    # when enable_thinking=False, intending the model to continue after
    # the closed think block. But our SFT models (codecontests, em_de,
    # etc.) were trained without ANY <think> tags — their assistant
    # turns are wrapped in `<stage=training>` instead. Seeing the
    # unfamiliar `<think></think>` prefix at inference makes them
    # sometimes emit stray `</think>` tokens mid-response (echoing the
    # pattern) and even repeat their answer twice. Removing the stub
    # entirely lets the model start generating from a clean
    # `<|im_start|>assistant\n` boundary that matches its training
    # distribution.
    if THINK_STUB_GENERATION_PROMPT not in ct:
        return ct, False
    print("Stripped <think></think> from non-reasoning generation prompt (avoids stray </think> echoes)")
    return ct.replace(THINK_STUB_GENERATION_PROMPT, PLAIN_GENERATION_PROMPT, 1), True


def fix_chat_template_file(hf_path: Path, reasoning: bool) -> None:
    """Fix ``hf_path/chat_template.jinja`` for the training regime.

    Applies ``set_enable_thinking_default`` and, without ``reasoning``, ``strip_think_stub``, writing the file when
    either changed. Prints a note when there is no chat_template.jinja.
    """
    # Keep chat_template.jinja as shipped by the upstream Nemotron tokenizer.
    # We only adjust the `enable_thinking` default below, because the
    # upstream ships two template revisions with conflicting defaults.
    jinja_path = hf_path / "chat_template.jinja"
    if not jinja_path.exists():
        print("No chat_template.jinja found — will use tokenizer_config.json fallback")
        return
    ct, ct_changed = set_enable_thinking_default(jinja_path.read_text(), reasoning, jinja_path.stat().st_size)
    if not reasoning:
        ct, stripped = strip_think_stub(ct)
        ct_changed = ct_changed or stripped
    if ct_changed:
        jinja_path.write_text(ct)


def sync_embedded_template(hf_path: Path, adopted_release_files: bool) -> None:
    """Set tokenizer_config.json's embedded ``chat_template`` to chat_template.jinja's text, when both files exist.

    A tokenizer_config.json without an embedded template is left so when the export took over its release's files
    (``adopted_release_files``). Writes the config only when the template differs.
    """
    # Always sync chat_template.jinja → tokenizer_config.json's embedded
    # chat_template, so the two stay in lockstep. This must run even when
    # the .jinja is unchanged, because the .jinja may have been patched by a
    # previous conversion run (or by the graft in tokenizer.fix_tokenizer_config
    # and the patch in fix_chat_template_file) while the embedded copy is still the freshly-grafted (un-
    # patched) version. transformers >= 5.x prefers the .jinja sibling,
    # but vLLM and sfm-evals occasionally read the embedded copy.
    # A release's tokenizer_config.json without an embedded copy stays as released.
    tokenizer_config = hf_path / "tokenizer_config.json"
    jinja_path = hf_path / "chat_template.jinja"
    if tokenizer_config.exists() and jinja_path.exists():
        jinja_text = jinja_path.read_text()
        with open(tokenizer_config) as f:
            tc = json.load(f)
        if tc.get("chat_template") != jinja_text and ("chat_template" in tc or not adopted_release_files):
            tc["chat_template"] = jinja_text
            with open(tokenizer_config, "w") as f:
                json.dump(tc, f, indent=2, ensure_ascii=False)
            print("Synced tokenizer_config.json embedded chat_template <- chat_template.jinja")
