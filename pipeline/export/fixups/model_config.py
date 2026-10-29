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
"""The export's config.json and generation_config.json: the fields vLLM and transformers need from a NemotronH export.

Catalogue:
    read_embedding_row_count: rows of the exported input-embedding matrix, from a safetensors header.
    fix_eos_token_ids: add ``<|im_end|>`` to the end-of-generation ids when a chat template is active.
    fix_hybrid_override_pattern: MoE layers as "E", or the pattern copied from the cached release.
    fix_layers_block_type: replace a layers_block_type list with the hybrid_override_pattern it spells.
    fix_num_hidden_layers: set num_hidden_layers from the pattern's length.
    fix_vocab_size: set vocab_size to the embedding's row count.
    fix_model_config: apply the remote-code policy and the config.json fixes above, writing it once.
"""

import json
import struct
from pathlib import Path

from pipeline.export.fixups.remote_code import apply_remote_code_policy
from pipeline.hf_cache import cached_file
from pipeline.nemotron import BLOCK_TYPE_CHARS, IM_END_TOKEN_ID


def read_embedding_row_count(hf_path: Path) -> int | None:
    """Number of rows in the exported input-embedding matrix, or None if not found.

    Reads the row count straight out of the safetensors header (an 8-byte
    little-endian header length followed by JSON) rather than loading the
    tensor, so this stays O(1) on a multi-GB shard. Handles both a sharded
    export, via `model.safetensors.index.json`, and a single-file one.
    """
    index_json = hf_path / "model.safetensors.index.json"
    single = hf_path / "model.safetensors"
    if index_json.exists():
        with open(index_json) as f:
            weight_map = json.load(f)["weight_map"]
        emb_key = next((k for k in weight_map if "embed" in k.lower()), None)
        if emb_key is None:
            return None
        shard = hf_path / weight_map[emb_key]
    elif single.exists():
        shard, emb_key = single, None
    else:
        return None

    with open(shard, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))
    if emb_key is None:
        emb_key = next((k for k in header if k != "__metadata__" and "embed" in k.lower()), None)
        if emb_key is None:
            return None
    return header[emb_key]["shape"][0]


def fix_eos_token_ids(hf_path: Path) -> None:
    """Add ``IM_END_TOKEN_ID`` to ``eos_token_id`` when the export has a chat template.

    The id goes into generation_config.json, which generation reads, or into config.json when there is none. Each
    file is rewritten only when it gains the id; a file without an int or list ``eos_token_id`` is left alone.
    """
    # Fix eos_token_id: when --hf-model points at the Base release (or any
    # release without a chat_template), config.json and generation_config.json
    # inherit eos_token_id=2 (`</s>`) only. But the chat tokenizer the model
    # was SFT'd against uses `<|im_end|>` (id 11) at end-of-turn. Generation
    # must stop on either id, otherwise the model runs to max_new_tokens.
    # Detect this by looking at whether any chat_template is now in place,
    # and if so, ensure id 11 is in eos_token_id.
    chat_template_active = (hf_path / "chat_template.jinja").exists() or (
        (hf_path / "tokenizer_config.json").exists()
        and "chat_template" in json.loads((hf_path / "tokenizer_config.json").read_text())
    )
    if not chat_template_active:
        return
    # Generation reads generation_config.json, so config.json keeps the release's value when that file exists.
    eos_configs = ["generation_config.json"] if (hf_path / "generation_config.json").exists() else ["config.json"]
    for cfg_name in eos_configs:
        cfg_path = hf_path / cfg_name
        if not cfg_path.exists():
            continue
        with open(cfg_path) as f:
            cfg = json.load(f)
        existing = cfg.get("eos_token_id")
        # Normalise to a list and add id 11 if not already present.
        if isinstance(existing, int):
            existing_list = [existing]
        elif isinstance(existing, list):
            existing_list = list(existing)
        else:
            continue
        if IM_END_TOKEN_ID not in existing_list:
            new_list = sorted(set(existing_list + [IM_END_TOKEN_ID]))
            cfg["eos_token_id"] = new_list
            with open(cfg_path, "w") as f:
                json.dump(cfg, f, indent=2, ensure_ascii=False)
            print(
                f"Patched {cfg_name}: eos_token_id {existing} -> {new_list} (added <|im_end|>=11 for chat-format generation)"
            )


def fix_hybrid_override_pattern(config: dict, hf_model_id: str) -> bool:
    """Write MoE layers as "E" in ``config``'s hybrid_override_pattern, or copy a missing pattern from the cached
    release ``hf_model_id`` (its refs/main snapshot). Mutates ``config``; returns whether it changed."""
    # The bridge saves the pattern with "-" for MoE layers, but the NVIDIA convention
    # uses "E". vLLM checks `"E" in config.hybrid_override_pattern` to detect MoE.
    # Also copy the pattern from the source HF model if not in config.json, since the
    # custom configuration_nemotron_h.py default uses "-" (wrong for vLLM).
    pattern = config.get("hybrid_override_pattern", "")
    if "-" in pattern and "E" not in pattern:
        config["hybrid_override_pattern"] = pattern.replace("-", "E")
        print("Fixed hybrid_override_pattern: replaced '-' with 'E' for MoE layers")
        return True
    if not pattern:
        # Copy from source model config in HF cache
        source_cfg_path = cached_file(hf_model_id, "config.json")
        if source_cfg_path is not None:
            with open(source_cfg_path) as f:
                source_cfg = json.load(f)
            if "hybrid_override_pattern" in source_cfg:
                config["hybrid_override_pattern"] = source_cfg["hybrid_override_pattern"]
                print(f"Added hybrid_override_pattern from {hf_model_id}")
                return True
    return False


def fix_layers_block_type(config: dict) -> bool:
    """Drop ``config``'s layers_block_type list, deriving hybrid_override_pattern from it when that is missing.

    Mutates ``config``; returns whether it changed.

    Raises:
        ValueError: the list names a block type outside ``BLOCK_TYPE_CHARS``.
    """
    # Emit the upstream/vLLM NemotronH convention: hybrid_override_pattern (string), never the
    # explicit layers_block_type list. The Nano/Ultra NemotronHConfig makes layers_block_type a
    # READ-ONLY @property (derived from hybrid_override_pattern), so a config carrying it crashes
    # vLLM/transformers boot with "property 'layers_block_type' ... has no setter". (The Super-era
    # class tolerated the list, which masked this on 120B.) Arch-agnostic: derive from the list.
    if "layers_block_type" not in config:
        return False
    if not config.get("hybrid_override_pattern"):
        try:
            config["hybrid_override_pattern"] = "".join(
                BLOCK_TYPE_CHARS[str(b).lower()] for b in config["layers_block_type"]
            )
        except KeyError as exc:
            raise ValueError(f"unknown block type in layers_block_type: {exc}") from exc
        print(f"Derived hybrid_override_pattern from layers_block_type ({len(config['layers_block_type'])} layers)")
    del config["layers_block_type"]
    print("Stripped layers_block_type (read-only @property in NemotronHConfig; vLLM uses hybrid_override_pattern)")
    return True


def fix_num_hidden_layers(config: dict) -> bool:
    """Set a missing or zero num_hidden_layers to the length of ``config``'s hybrid_override_pattern.

    Mutates ``config``; returns whether it changed.
    """
    # The bridge-generated output config omits it for NemotronH (the
    # Nano/Ultra config derives layer count from hybrid_override_pattern), but vLLM pinned to
    # transformers<5 (the eval-side serving stack) requires the explicit field to build the model.
    # Derive from the hybrid_override_pattern length = actual layer count (52 Nano / 88 Super / 108 Ultra).
    _hop = config.get("hybrid_override_pattern", "")
    if _hop and config.get("num_hidden_layers") in (None, 0):
        config["num_hidden_layers"] = len(_hop)
        print(f"Emitted num_hidden_layers={len(_hop)} (from hybrid_override_pattern length) for vLLM<5 compatibility")
        return True
    return False


def fix_vocab_size(config: dict, hf_path: Path) -> bool:
    """Set ``config``'s vocab_size to the row count of the export's embedding (``read_embedding_row_count``).

    Mutates ``config``; returns whether it changed. An export whose embedding is not found is left alone.
    """
    # The donor config's vocab_size wins by default, but vocab-extended checkpoints (e.g. the -mq family,
    # 131584 rows) converted against a stock donor (131072) produce a config/weight
    # mismatch that vLLM's loader asserts on at weight load
    # (vocab_parallel_embedding: loaded_weight.shape[0] == org_vocab_size).
    emb_rows = read_embedding_row_count(hf_path)
    if emb_rows is not None and config.get("vocab_size") != emb_rows:
        print(
            f"Reconciled vocab_size: config {config.get('vocab_size')} -> {emb_rows} "
            f"(actual embedding rows; donor/weight vocab mismatch)"
        )
        config["vocab_size"] = emb_rows
        return True
    return False


def fix_model_config(hf_path: Path, hf_model_id: str, keep_remote_code: bool, remote_code_source: str | None) -> None:
    """Apply ``apply_remote_code_policy`` and the config.json fixes of this module to ``hf_path/config.json``.

    The fixes run in catalogue order, each seeing the previous ones' result. config.json is written once, and only
    when a fix changed it; an export without config.json is left alone.

    Raises:
        FileNotFoundError: from ``apply_remote_code_policy``.
        ValueError: from ``fix_layers_block_type``.
    """
    # Apply the remote-code policy to config.json. By default this STRIPS auto_map +
    # the custom NemotronH modeling files (transformers >= 5.3.0 is native; the old
    # custom code uses "backbone.*" naming that conflicts with "model.*" weights). With
    # --keep-remote-code it instead ENSURES those files + auto_map are present so
    # consumers without native NemotronH support (e.g. vLLM trust_remote_code on
    # transformers < 5.3.0) can load the export.
    config_json = hf_path / "config.json"
    if not config_json.exists():
        return

    with open(config_json) as f:
        config = json.load(f)

    config_changed = apply_remote_code_policy(
        hf_path,
        config,
        keep_remote_code=keep_remote_code,
        remote_code_source=remote_code_source,
    )

    # NOTE: Weight keys use "backbone.*" naming (from the bridge's save_hf_pretrained).
    # Do NOT rename to "model.*" — vLLM's NemotronH weight mapper expects "backbone.*"
    # and handles the conversion internally via hf_to_vllm_mapper. Transformers >= 5.3.0
    # also handles "backbone.*" natively.
    config_changed = fix_hybrid_override_pattern(config, hf_model_id) or config_changed
    config_changed = fix_layers_block_type(config) or config_changed
    config_changed = fix_num_hidden_layers(config) or config_changed
    config_changed = fix_vocab_size(config, hf_path) or config_changed

    if config_changed:
        with open(config_json, "w") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
