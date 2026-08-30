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

"""Shared machinery for saving and publishing forked tokenizers.

Every tokenizer this repo writes — whether forked by a build script or emitted
beside a converted checkpoint — has to be loadable by the stacks that consume
it, and attributable to the exact source it came from. That is the same work
each time: normalise `tokenizer_config.json`, record provenance, optionally
publish. This module is the single home for it.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


HF_ORG = "geodesic-research"
LOCAL_TOKENIZER_DIR = Path("/projects/a5k/public/tokenizers")

# transformers 5.x writes `tokenizer_class: TokenizersBackend` plus `backend`
# and `is_local`. Older transformers (the 4.5x eval stack) and vLLM read those
# and abort with "Tokenizer class TokenizersBackend does not exist or is not
# currently imported", so a tokenizer saved as-is is unloadable by the very
# stack that evaluates these models.
_STALE_CONFIG_KEYS = ("backend", "is_local")
_PORTABLE_TOKENIZER_CLASS = "PreTrainedTokenizerFast"


def normalize_tokenizer_config(cfg: dict) -> list[str]:
    """Make a `tokenizer_config.json` dict loadable by transformers 4.5x and vLLM.

    Strips the transformers 5.x-only fields and pins the tokenizer class. Mutates
    `cfg` in place and returns one human-readable line per change, empty when the
    config was already portable.
    """
    changes = []
    for stale_key in _STALE_CONFIG_KEYS:
        if stale_key in cfg:
            del cfg[stale_key]
            changes.append(f"stripped tokenizer_config.{stale_key}")
    if cfg.get("tokenizer_class") != _PORTABLE_TOKENIZER_CLASS:
        changes.append(f"pinned tokenizer_class: {cfg.get('tokenizer_class')} -> {_PORTABLE_TOKENIZER_CLASS}")
        cfg["tokenizer_class"] = _PORTABLE_TOKENIZER_CLASS
    return changes


def write_normalized_tokenizer_config(save_dir: Path, extra_fields: dict | None = None) -> list[str]:
    """Normalise the `tokenizer_config.json` in `save_dir`, adding `extra_fields`.

    `extra_fields` carries custom entries a fork needs downstream (for example
    `loss_mask_token_ids`). Returns the change lines, including one per added field.
    """
    cfg_path = save_dir / "tokenizer_config.json"
    cfg = json.loads(cfg_path.read_text())
    changes = normalize_tokenizer_config(cfg)
    for key, value in (extra_fields or {}).items():
        cfg[key] = value
        changes.append(f"set tokenizer_config.{key} = {value!r}")
    cfg_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n")
    return changes


def resolve_source_revision(source_id: str, revision: str | None) -> str:
    """Resolve a Hub repo + revision to the exact commit sha it names.

    A build is only attributable if the sha is recorded, so this is called even
    when `revision` is None (the repository's default branch).
    """
    from huggingface_hub import HfApi

    return HfApi().model_info(source_id, revision=revision).sha


def write_provenance_readme(save_dir: Path, body: str) -> Path:
    """Write the fork's README, stamping the build date onto `body`.

    `body` is the fork-specific description; every README gets the same trailing
    build stamp so a published tokenizer always says when it was produced.
    """
    build_date = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    readme_path = save_dir / "README.md"
    readme_path.write_text(f"{body.rstrip()}\n\nBuilt {build_date}.\n")
    return readme_path


def publish_tokenizer_folder(save_dir: Path, repo_id: str, commit_message: str) -> str:
    """Upload an already-written tokenizer directory to the Hub, returning its URL.

    Publishing the directory rather than the in-memory tokenizer keeps the local
    build and the published artifact from ever diverging.
    """
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id, exist_ok=True, repo_type="model")
    api.upload_folder(
        folder_path=str(save_dir),
        repo_id=repo_id,
        repo_type="model",
        commit_message=commit_message,
    )
    return f"https://huggingface.co/{repo_id}"
