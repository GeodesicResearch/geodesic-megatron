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
"""The tokenizer files of an export: tokenizer_config.json's class, and the chat template the model trained on.

The template comes from the tokenizer the model was trained with (run_config.yaml > tokenizer > tokenizer_model), so
that inference formats exactly as training did. When that tokenizer is not recorded or cannot be found, the template
is grafted from the instruct release of the model family (``CHAT_TEMPLATE_SOURCE_MAP``).

Catalogue:
    TRAINING_TOKENIZER_FILES: the files copied verbatim from the training tokenizer.
    TRAINING_TOKEN_KEYS: the tokenizer_config.json token settings taken from the training tokenizer.
    resolve_template_from_hub_id: a tokenizer repo's chat template, from the HF cache or the Hub.
    pin_tokenizer_class: pin tokenizer_config.json's class to one every consumer loads.
    pin_tokenizer_config_file: the same, applied to an export's tokenizer_config.json file.
    copy_training_tokenizer_files: copy the training tokenizer's files into the export.
    apply_training_template: make the training tokenizer's template and files the export's.
    graft_release_template: set the template from the family's instruct release in the HF cache.
    fix_tokenizer_config: apply the tokenizer_config.json fixes above.
    mirror_template_file: write an embedded-only template to chat_template.jinja.
"""

import json
import shutil
from pathlib import Path

from pipeline.hf_cache import cached_snapshots
from pipeline.nemotron import CHAT_TEMPLATE_SOURCE_MAP


TRAINING_TOKENIZER_FILES = ("tokenizer.json", "special_tokens_map.json", "chat_template.jinja")
TRAINING_TOKEN_KEYS = (
    "eos_token",
    "bos_token",
    "pad_token",
    "unk_token",
    "sep_token",
    "mask_token",
    "add_bos_token",
    "add_eos_token",
)


def resolve_template_from_hub_id(hub_id: str) -> str | None:
    """Best-effort: resolve the chat_template for a HF tokenizer id.

    Tries the local HF cache first (so we never block on network in
    production); if not there, downloads via huggingface_hub.

    Returns the chat_template string, or None on any failure.
    """
    # 1. Local HF cache snapshot scan
    for snapshot_dir in cached_snapshots(hub_id) or []:
        jinja = snapshot_dir / "chat_template.jinja"
        if jinja.exists():
            return jinja.read_text()
        tc_p = snapshot_dir / "tokenizer_config.json"
        if tc_p.exists():
            try:
                tc_dict = json.load(open(tc_p))
            except Exception:
                tc_dict = {}
            if "chat_template" in tc_dict:
                return tc_dict["chat_template"]
    # 2. Network fallback via huggingface_hub
    try:
        from huggingface_hub import hf_hub_download

        for fname in ("chat_template.jinja", "tokenizer_config.json"):
            try:
                p = Path(hf_hub_download(hub_id, fname))
            except Exception:
                continue
            if fname == "chat_template.jinja":
                return p.read_text()
            try:
                tc_dict = json.load(open(p))
            except Exception:
                tc_dict = {}
            if "chat_template" in tc_dict:
                return tc_dict["chat_template"]
    except Exception:
        pass
    return None


def pin_tokenizer_class(tc: dict) -> bool:
    """Make a tokenizer_config.json dict load as ``PreTrainedTokenizerFast`` everywhere; return whether it changed.

    Strips the cosmetic fields that older transformers (eval venv 4.57.x) read as a hint to load TokenizersBackend
    (a transformers 5.x class), which then crash with "Tokenizer class TokenizersBackend does not exist or is not
    currently imported", and pins ``tokenizer_class`` (vLLM-friendly). Mutates ``tc``.
    """
    changed = False
    for stale_key in ("backend", "is_local"):
        if stale_key in tc:
            del tc[stale_key]
            changed = True
            print(f"Stripped tokenizer_config.{stale_key}")
    if tc.get("tokenizer_class") != "PreTrainedTokenizerFast":
        print(f"Pinned tokenizer_class: {tc.get('tokenizer_class')} -> PreTrainedTokenizerFast")
        tc["tokenizer_class"] = "PreTrainedTokenizerFast"
        changed = True
    return changed


def pin_tokenizer_config_file(hf_path: Path) -> None:
    """Apply ``pin_tokenizer_class`` to ``hf_path/tokenizer_config.json``, writing it when it changed; a missing
    file is left missing."""
    tokenizer_config = hf_path / "tokenizer_config.json"
    if not tokenizer_config.exists():
        return
    tc = json.loads(tokenizer_config.read_text())
    if pin_tokenizer_class(tc):
        with open(tokenizer_config, "w") as f:
            json.dump(tc, f, indent=2, ensure_ascii=False)


def copy_training_tokenizer_files(hf_path: Path, training_tokenizer_id: str, tc: dict) -> None:
    """Copy ``TRAINING_TOKENIZER_FILES`` verbatim from the training tokenizer into ``hf_path``, and its
    ``TRAINING_TOKEN_KEYS`` into ``tc``.

    The tokenizer comes from huggingface_hub's ``snapshot_download`` (the HF cache, else the network). Each file it
    replaces is first backed up once as ``<name>.converter_bak``. A failure is printed as a warning; files copied
    before it stay copied.
    """
    try:
        from huggingface_hub import snapshot_download as _snap_dl

        src_dir = Path(_snap_dl(training_tokenizer_id, repo_type="model"))
        for _fname in TRAINING_TOKENIZER_FILES:
            _src = src_dir / _fname
            _dst = hf_path / _fname
            if _src.exists():
                # Backup the existing file once, then overwrite.
                _bk = _dst.with_suffix(_dst.suffix + ".converter_bak")
                if _dst.exists() and not _bk.exists():
                    shutil.copy2(_dst, _bk)
                shutil.copy2(_src, _dst)
                print(
                    f"Copied {_fname} ({_src.stat().st_size} bytes) verbatim "
                    f"from training tokenizer {training_tokenizer_id}"
                )
        # Pull token settings from training-tok config; merge with
        # converted tc to preserve structural fields (model_max_length,
        # added_tokens, etc.).
        _src_tc_path = src_dir / "tokenizer_config.json"
        if _src_tc_path.exists():
            _src_tc = json.loads(_src_tc_path.read_text())
            for _key in TRAINING_TOKEN_KEYS:
                if _key in _src_tc:
                    tc[_key] = _src_tc[_key]
    except Exception as _e:  # noqa: BLE001
        print(f"Warning: failed to copy training-tokenizer files verbatim: {_e}")


def graft_release_template(tc: dict, hf_model_id: str) -> bool:
    """Set ``tc``'s chat_template from the cached instruct release of ``hf_model_id``'s family.

    Reads the HF cache only (``CHAT_TEMPLATE_SOURCE_MAP`` names the release); the largest-hash snapshot holding a
    template wins, its chat_template.jinja before its tokenizer_config.json. Mutates ``tc`` and returns whether it
    changed; prints a warning when the release or its template is not cached.
    """
    source_model_id = CHAT_TEMPLATE_SOURCE_MAP.get(hf_model_id)
    if not source_model_id:
        return False
    snapshots = cached_snapshots(source_model_id)
    if snapshots is None:
        print(
            f"Warning: HF cache not found for {source_model_id} — run: "
            f'python -c "from transformers import AutoTokenizer; '
            f"AutoTokenizer.from_pretrained('{source_model_id}')\""
        )
        return False
    for snapshot_dir in snapshots:
        # Prefer chat_template.jinja sibling over embedded.
        source_jinja = snapshot_dir / "chat_template.jinja"
        if source_jinja.exists():
            tc["chat_template"] = source_jinja.read_text()
            print(
                f"Grafted chat_template from {source_model_id}/chat_template.jinja ({snapshot_dir.name[:8]}, {len(tc['chat_template'])} bytes)"
            )
            return True
        source_tc_path = snapshot_dir / "tokenizer_config.json"
        if source_tc_path.exists():
            with open(source_tc_path) as f:
                source_tc = json.load(f)
            if "chat_template" in source_tc:
                tc["chat_template"] = source_tc["chat_template"]
                print(f"Grafted chat_template from {source_model_id}/tokenizer_config.json ({snapshot_dir.name[:8]})")
                return True
    print(f"Warning: chat_template not found in HF cache for {source_model_id}")
    return False


def apply_training_template(hf_path: Path, tc: dict, training_tpl: str, training_tokenizer_id: str) -> bool:
    """Make the training tokenizer's template the export's: its files (``copy_training_tokenizer_files``), the
    embedded ``chat_template`` of ``tc``, and ``hf_path/chat_template.jinja``.

    Mutates ``tc`` and returns whether its template changed; always writes chat_template.jinja.
    """
    # Copy the training tokenizer's files verbatim (overwrite anything
    # from the bridge / upstream release that may be lying around).
    copy_training_tokenizer_files(hf_path, training_tokenizer_id, tc)

    changed = False
    if tc.get("chat_template") != training_tpl:
        tc["chat_template"] = training_tpl
        changed = True
        print(
            f"Set chat_template from training tokenizer {training_tokenizer_id} "
            f"({len(training_tpl)} bytes) — matches what the model was trained on"
        )

    # Force-write chat_template.jinja to training_tpl, overriding any
    # stale upstream copy that bridge.save_hf_pretrained may have left
    # at hf_path. The enable_thinking fix (chat_template.fix_chat_template_file)
    # and the embedded<-jinja sync (chat_template.sync_embedded_template) both read
    # from this file; without an
    # unconditional overwrite here, a training tokenizer that ships
    # the template only embedded in tokenizer_config.json (no .jinja
    # sibling on the Hub) would leave the stale upstream .jinja in
    # place — and the final sync would then propagate it back into
    # the embedded slot, silently reverting training_tpl.
    (hf_path / "chat_template.jinja").write_text(training_tpl)
    return changed


def fix_tokenizer_config(hf_path: Path, hf_model_id: str, training_tokenizer_id: str | None) -> str | None:
    """Fix the export's tokenizer_config.json and set its chat template; return the training template, if found.

    With no tokenizer_config.json in ``hf_path`` this does nothing and returns None. Otherwise it normalises the
    tokenizer class (``pin_tokenizer_class``). If the training tokenizer's template resolves
    (``resolve_template_from_hub_id``), it applies it (``apply_training_template``); if not, and the config holds no
    template, it grafts one (``graft_release_template``). Writes tokenizer_config.json when it changed.
    """
    # Fix tokenizer_class and add chat_template to tokenizer_config.json
    tokenizer_config = hf_path / "tokenizer_config.json"
    training_tpl = None
    if not tokenizer_config.exists():
        return training_tpl
    with open(tokenizer_config) as f:
        tc = json.load(f)

    changed = pin_tokenizer_class(tc)

    # Set the chat_template. Priority order:
    #   1. The tokenizer the model was actually trained with
    #      (run_config.yaml > tokenizer > tokenizer_model). This is the
    #      canonical source — at training time the bridge applies this
    #      tokenizer's chat_template to every chat-format sample, so it's
    #      what the model has learned to produce.
    #   2. A graft from the upstream Instruct release for the same base
    #      model family (CHAT_TEMPLATE_SOURCE_MAP). Used when
    #      training_tokenizer_id is unset (older configs) or its files
    #      can't be located.
    # When training tokenizer is set, we copy ALL of its files
    # (tokenizer.json, chat_template.jinja, special_tokens_map.json) byte-
    # identical into the HF dir, then rebuild tokenizer_config.json by
    # merging the converted dir's structural fields with the training
    # tokenizer's tokens + chat_template (adopt_release_inference_files
    # replaces them with the release's when the training template
    # renders as the release's). This avoids the legacy bug where the upstream Nemotron template (with
    # `<think></think>` injection in assistant turns) was preserved at
    # chat_template.jinja and synced into tokenizer_config.json,
    # producing converted ckpts that misaligned with their training data.
    if training_tokenizer_id:
        training_tpl = resolve_template_from_hub_id(training_tokenizer_id)
        if training_tpl is None:
            print(
                f"Warning: training tokenizer {training_tokenizer_id} not on disk and "
                f"not downloadable — falling back to CHAT_TEMPLATE_SOURCE_MAP graft"
            )

    if training_tpl is not None:
        changed = apply_training_template(hf_path, tc, training_tpl, training_tokenizer_id) or changed
    elif "chat_template" not in tc:
        # Legacy fallback: graft from upstream Instruct for the base family.
        changed = graft_release_template(tc, hf_model_id) or changed

    if changed:
        with open(tokenizer_config, "w") as f:
            json.dump(tc, f, indent=2, ensure_ascii=False)
    return training_tpl


def mirror_template_file(hf_path: Path) -> None:
    """Write tokenizer_config.json's embedded chat template to ``hf_path/chat_template.jinja`` when only the
    embedded copy exists."""
    # If a chat_template was grafted into tokenizer_config but no
    # chat_template.jinja sibling exists, mirror it as a .jinja file so
    # the enable_thinking fix (chat_template.fix_chat_template_file), which only operates on .jinja,
    # can run, and so transformers >= 5.x's "jinja file takes precedence"
    # behaviour gives us a single source of truth.
    tokenizer_config = hf_path / "tokenizer_config.json"
    chat_template_jinja_path = hf_path / "chat_template.jinja"
    if not tokenizer_config.exists() or chat_template_jinja_path.exists():
        return
    tc = json.loads(tokenizer_config.read_text())
    if "chat_template" in tc:
        chat_template_jinja_path.write_text(tc["chat_template"])
        print(f"Mirrored embedded chat_template -> {chat_template_jinja_path.name}")
