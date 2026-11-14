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
"""Shipping files of the --hf-model release with an export: its inference tokenizer and generation files when the
model trained on its chat format, and any files a caller names, whatever the format.

Catalogue:
    CHAT_TEMPLATE_FILE: the name of the chat template file.
    RELEASE_INFERENCE_FILES: the release files an export takes over.
    TEMPLATE_PROBE_CONVERSATIONS, TEMPLATE_PROBE_TOOLS: the conversations two templates must render alike.
    is_bare_file_name: whether a name is that of a file in a directory, and fit for the --reference-files list.
    resolve_release_dir: the directory of the --hf-model release, a local directory or a Hub snapshot of it.
    release_file: a file of the release, or the error that names it when the release cannot supply it.
    find_release_files: the paths of the release's named files, after checking that each exists.
    require_release_files: the release's directory, once it holds the named files.
    templates_render_alike: whether two chat templates render every probe conversation identically.
    adopt_release_inference_files: copy the release's files into an export whose training template renders as its.
    copy_release_files: copy named files of the release into an export, whatever the training template is.
"""

import json
import os
import shutil
from collections.abc import Sequence
from pathlib import Path


CHAT_TEMPLATE_FILE = "chat_template.jinja"

#: The inference tokenizer and generation files of a release, which an export ships from its --hf-model release
#: when the training tokenizer's chat template renders as the release's (``adopt_release_inference_files``).
RELEASE_INFERENCE_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    CHAT_TEMPLATE_FILE,
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


def is_bare_file_name(name: object) -> bool:
    """Whether ``name`` is the name of a file directly inside a directory: a non-empty string that is neither ``.``
    nor ``..`` and holds no path separator. It holds no comma either, which separates the names of the converter's
    ``--reference-files`` list."""
    return isinstance(name, str) and name not in ("", ".", "..") and not any(char in name for char in "/\\,")


def resolve_release_dir(hf_model_id: str, allow_patterns: Sequence[str]) -> Path:
    """The directory of the --hf-model release: ``hf_model_id`` itself where it is a directory, else the Hub snapshot
    of that id (the cache, else the network), limited to the files ``allow_patterns`` names.

    Raises whatever huggingface_hub raises for an id it cannot resolve: an ``OSError`` for a repository it cannot
    reach or find, a ``ValueError`` for a string that is no repository id, such as a directory that is not there.
    """
    if Path(hf_model_id).is_dir():
        return Path(hf_model_id)
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(hf_model_id, allow_patterns=list(allow_patterns)))


def _raise_if_dangling(path: Path, name: str) -> None:
    """Raise FileNotFoundError if ``path`` is a link to nothing: a release in the cache that is missing its blob."""
    if os.path.lexists(path) and not path.exists():
        raise FileNotFoundError(f"{path} is a dangling link: the release's {name} is missing from the cache")


def release_file(release_dir: Path, name: str) -> Path:
    """The release's file ``name``.

    Raises:
        FileNotFoundError: the file is a dangling link, or the release has no such file.
    """
    path = release_dir / name
    _raise_if_dangling(path, name)
    if not path.exists():
        raise FileNotFoundError(f"{release_dir}: the release has no {name}")
    return path


def find_release_files(release_dir: Path, names: Sequence[str]) -> list[Path]:
    """The paths of the release's files ``names``, in order, after checking that each exists.

    Raises:
        ValueError: a name is not a bare file name (``is_bare_file_name``), so would reach outside the export.
        FileNotFoundError: the release lacks a named file, or one of them, or its ``chat_template.jinja``, is a
            dangling link.
    """
    for name in names:
        if not is_bare_file_name(name):
            raise ValueError(f"reference file {name!r} is not a bare file name")
    sources = [release_file(release_dir, name) for name in names]
    _raise_if_dangling(release_dir / CHAT_TEMPLATE_FILE, CHAT_TEMPLATE_FILE)
    return sources


def require_release_files(hf_model_id: str, names: Sequence[str]) -> Path:
    """The directory of the --hf-model release, once it holds every file ``names``.

    ``hf_model_id`` is a local directory or a Hub id. A Hub id downloads the named files and the chat template into the
    HF cache (network read, cache write), as ``resolve_release_dir`` does.

    Raises:
        ValueError, FileNotFoundError: as ``find_release_files``; or what ``resolve_release_dir`` raises for an id it
            cannot resolve.
    """
    release_dir = resolve_release_dir(hf_model_id, [*names, CHAT_TEMPLATE_FILE])
    find_release_files(release_dir, names)
    return release_dir


def copy_release_files(hf_path: Path, release_dir: Path, names: Sequence[str]) -> None:
    """Make the export carry the release's files ``names`` byte for byte, whatever the training template is.

    Copies each file from ``release_dir`` into ``hf_path`` (``shutil.copyfile``, so a link of the Hub cache is read
    through to its blob), replacing the export's own. The release has the last word on the files it ships: the
    export's ``chat_template.jinja`` is removed when the release has none, since a template file beside a
    ``tokenizer_config.json`` that embeds the release's template would take precedence over it. It compares no chat
    templates (``templates_render_alike``), so it works for a release that ships no ``chat_template.jinja``, where
    ``adopt_release_inference_files`` copies nothing. Nothing is written until every file has been found.

    Raises:
        ValueError: a name is not a bare file name (``is_bare_file_name``), so would reach outside the export.
        FileNotFoundError: the release lacks a named file, or one of them, or its ``chat_template.jinja``, is a
            dangling link.
    """
    sources = find_release_files(release_dir, names)
    release_template = release_dir / CHAT_TEMPLATE_FILE
    for name, source in zip(names, sources):
        shutil.copyfile(source, hf_path / name)
        print(f"Copied {name} from {release_dir} verbatim (a reference file)")
    export_template = hf_path / CHAT_TEMPLATE_FILE
    if not release_template.exists() and export_template.exists():
        export_template.unlink()
        print(f"{release_dir} ships no {CHAT_TEMPLATE_FILE}: removed the export's")


def adopt_release_inference_files(hf_path: Path, hf_model_id: str, training_template: str) -> bool:
    """Ship the --hf-model release's inference tokenizer and generation files when the model trained on its format.

    If the release's chat template renders as ``training_template`` (``templates_render_alike``), copies every
    ``RELEASE_INFERENCE_FILES`` file the release has into ``hf_path`` verbatim, replacing the training
    tokenizer's, and returns True. Otherwise (a release without a chat template, or a training format of its own)
    changes nothing and returns False. ``hf_model_id`` is a local release directory or a Hub id
    (``resolve_release_dir``).

    Raises:
        FileNotFoundError: a file of the release is a dangling link (a cache missing its blob), or the release or
            the export has no config.json.
    """
    release_dir = resolve_release_dir(hf_model_id, [*RELEASE_INFERENCE_FILES, "config.json"])
    for name in (*RELEASE_INFERENCE_FILES, "config.json"):
        _raise_if_dangling(release_dir / name, name)
    release_config, config_path = release_dir / "config.json", hf_path / "config.json"
    for path in (release_config, config_path):
        if not path.exists():
            raise FileNotFoundError(f"{path} does not exist: the token ids of config.json go with the release files")
    release_template = release_dir / CHAT_TEMPLATE_FILE
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
    # fix_eos_token_ids, a later step, adds <|im_end|> to them again when the export asks for it in config.json.
    release_ids = json.loads(release_config.read_text())
    config = json.loads(config_path.read_text())
    token_ids = {k: release_ids[k] for k in ("bos_token_id", "eos_token_id", "pad_token_id") if k in release_ids}
    if any(config.get(k) != v for k, v in token_ids.items()):
        config.update(token_ids)
        config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n")
        print(f"Set config.json token ids to {hf_model_id}'s: {token_ids}")
    return True
