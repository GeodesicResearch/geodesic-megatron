#!/usr/bin/env python3
"""Bake the two gradient-routing inference postures out of one raw GR HF export.

Training (GRAMMoELayer, `models/mamba/gram_layer.py`) adds one narrow auxiliary MLP to
every MoE layer. The single-process Megatron->HF export carries those weights as

    backbone.layers.<L>.mixer.gr_aux.up_proj.weight    (a, hidden)
    backbone.layers.<L>.mixer.gr_aux.down_proj.weight  (hidden, a)

alongside the layer's shared expert, which has exactly the same form at the wider
`moe_shared_expert_intermediate_size`. Both consume the MoE layer's input hidden states
and both are added to the layer output with coefficient 1.0, so for the model's non-gated
elementwise activation (relu^2, `mlp_bias: false`, `gated_linear_unit: false`):

    W2_s . sigma(W1_s x) + W2_a . sigma(W1_a x)  ==  [W2_s | W2_a] . sigma([W1_s ; W1_a] x)

exactly. That identity is the whole bake:

  forget_on/   shared up_proj   <- cat([shared_up,   aux_up  ], dim=0)   (3712+a, hidden)
               shared down_proj <- cat([shared_down, aux_down], dim=1)   (hidden, 3712+a)
               config.json moe_shared_expert_intermediate_size += a
               gr_aux.* keys removed
  forget_off/  gr_aux.* keys removed, everything else byte-stock (config.json included)

Both postures are therefore stock-shaped NemotronH checkpoints that load in HF
transformers and vLLM with zero code changes, and both are produced from ONE source dir in
ONE invocation so their provenance blocks share a single source digest.

Usage (CPU only; run inside the container because safetensors/torch live there):

    ./pipeline_env_exec.sh "cd <repo>; source pipeline_env_activate.sh || exit 1; \\
        python scripts/gradient_routing/bake_forget_postures.py \\
            --config scripts/gradient_routing/bake_postures_example.yaml"

The config YAML is the only argument. Fields:

    source_dir           raw HF export dir carrying the gr_aux keys (required)
    output_root          dir under which forget_on/ and forget_off/ are written (required)
    strip_chat_template  drop the grafted Instruct chat template (required, bool)
    expected_layers      assert this many MoE layers carry aux weights (optional)
    config_overrides     scalar config.json fields overwritten in BOTH postures, for
                         values the exporter stamps from the training run rather than
                         the architecture (optional; recorded in the provenance)

Each output dir gets a `forget_posture.json` recording exactly what was done, with an
expected-values block (per-shard sizes + digests, key-inventory delta, embedding rows vs
config vocab_size, tokenizer inventory) so a downstream integrity check is mechanical.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import struct
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import safetensors.torch
import torch
import yaml


# HF-space names. `gr_aux` is the module name chosen by GRAMMoELayer and mapped in
# nemotron_h_bridge.py; the shared-expert names are stock NemotronH.
AUX_MODULE = "gr_aux"
AUX_UP = f".mixer.{AUX_MODULE}.up_proj.weight"
AUX_DOWN = f".mixer.{AUX_MODULE}.down_proj.weight"
SHARED_UP = ".mixer.shared_experts.up_proj.weight"
SHARED_DOWN = ".mixer.shared_experts.down_proj.weight"
EMBED_KEY = "backbone.embeddings.weight"
INDEX_NAME = "model.safetensors.index.json"
CONFIG_NAME = "config.json"
PROVENANCE_NAME = "forget_posture.json"
LAYER_RE = re.compile(r"^backbone\.layers\.(\d+)\.")

# config.json keys that must NOT be present on the raw export: the bridge's
# `conform_config_to_reference` is expected to filter the training-only aux width out of
# the exported HF config, and a leak would make the posture dirs non-stock.
GR_CONFIG_PATTERNS = (AUX_MODULE, "gram", "gradient_rout", "forget", "aux_ffn")

MERGE_FORMULA = (
    "shared_up <- cat([shared_up, aux_up], dim=0); "
    "shared_down <- cat([shared_down, aux_down], dim=1); "
    "exact for W2_s.sigma(W1_s x) + W2_a.sigma(W1_a x) == [W2_s|W2_a].sigma([W1_s;W1_a] x) "
    "with a non-gated elementwise sigma, no bias, and both addends taking the same input "
    "at coefficient 1.0"
)

# safetensors dtype tag -> bytes per element, for the HF `total_size` convention (which
# counts tensor payload bytes, NOT file sizes — verified against the stock Nano export).
DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}

TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "tokenizer.model",
    "special_tokens_map.json",
    "added_tokens.json",
    "chat_template.jinja",
)


class BakeError(RuntimeError):
    """A refusal: the source is not what this script is allowed to bake."""


def _sha256_file(path: Path, chunk: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def _read_header(path: Path) -> dict[str, Any]:
    """Return a safetensors file's JSON header (key -> {dtype, shape, data_offsets})."""
    with open(path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(header_len))


def _tensor_entries(header: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {k: v for k, v in header.items() if k != "__metadata__" and isinstance(v, dict)}


def _safetensors_expected_size(path: Path) -> int:
    """Byte length `path` must have according to its own header (see extend_vocab_for_mq)."""
    with open(path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))
    ends = [v["data_offsets"][1] for v in _tensor_entries(header).values() if "data_offsets" in v]
    return 8 + header_len + (max(ends) if ends else 0)


def _nbytes(entry: dict[str, Any]) -> int:
    n = 1
    for dim in entry["shape"]:
        n *= dim
    dtype = entry["dtype"]
    if dtype not in DTYPE_BYTES:
        raise BakeError(f"Unknown safetensors dtype {dtype!r}; cannot compute index total_size.")
    return n * DTYPE_BYTES[dtype]


def _save_shard_atomically(tensors: dict[str, torch.Tensor], dest: Path) -> None:
    """Write a shard via tmp+rename, refusing to publish a truncated file.

    Same contract as `scripts/data/extend_vocab_for_mq.py`: a write killed partway
    otherwise leaves a shard whose header still advertises the full payload — readable
    metadata, missing data, and no error until something loads the missing tensors.
    """
    tmp = dest.with_name(dest.name + ".partial")
    try:
        safetensors.torch.save_file(tensors, tmp)
        with open(tmp, "rb") as f:
            os.fsync(f.fileno())
        expected = _safetensors_expected_size(tmp)
        actual = tmp.stat().st_size
        if expected != actual:
            raise OSError(
                f"{dest.name}: wrote {actual:,} bytes but its header declares {expected:,}. "
                "Refusing to publish a truncated shard."
            )
        os.replace(tmp, dest)
    finally:
        tmp.unlink(missing_ok=True)


def _write_json_atomically(obj: Any, dest: Path) -> None:
    tmp = dest.with_name(dest.name + ".partial")
    tmp.write_text(json.dumps(obj, indent=2) + "\n")
    os.replace(tmp, dest)


def load_bake_config(path: Path) -> dict[str, Any]:
    """Parse and validate the bake YAML, rejecting unknown keys."""
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise BakeError(f"{path}: expected a YAML mapping, got {type(raw).__name__}.")
    required = {"source_dir", "output_root", "strip_chat_template"}
    optional = {"expected_layers", "config_overrides"}
    missing = sorted(required - set(raw))
    unknown = sorted(set(raw) - required - optional)
    if missing:
        raise BakeError(f"{path}: missing required field(s) {missing}. Required: {sorted(required)}.")
    if unknown:
        raise BakeError(f"{path}: unknown field(s) {unknown}. Accepted: {sorted(required | optional)}.")
    if not isinstance(raw["strip_chat_template"], bool):
        raise BakeError(f"{path}: strip_chat_template must be a bool, got {raw['strip_chat_template']!r}.")
    expected_layers = raw.get("expected_layers")
    if expected_layers is not None and not isinstance(expected_layers, int):
        raise BakeError(f"{path}: expected_layers must be an int or absent, got {expected_layers!r}.")
    config_overrides = raw.get("config_overrides") or {}
    if not isinstance(config_overrides, dict) or not all(
        isinstance(k, str) and isinstance(v, (str, int, float, bool)) for k, v in config_overrides.items()
    ):
        raise BakeError(f"{path}: config_overrides must be a mapping of scalar values, got {config_overrides!r}.")
    return {
        "source_dir": Path(raw["source_dir"]),
        "output_root": Path(raw["output_root"]),
        "strip_chat_template": raw["strip_chat_template"],
        "expected_layers": expected_layers,
        "config_overrides": config_overrides,
    }


def survey_source(src: Path, expected_layers: int | None) -> dict[str, Any]:
    """Read the source index/headers/config and refuse anything this script must not bake."""
    index_path = src / INDEX_NAME
    config_path = src / CONFIG_NAME
    for p in (index_path, config_path):
        if not p.is_file():
            raise BakeError(f"{src} is not an HF checkpoint dir: {p.name} missing.")

    if (src / PROVENANCE_NAME).exists():
        raise BakeError(
            f"{src} contains {PROVENANCE_NAME} — it is itself a baked posture dir, not a raw "
            "export. Bake from the raw single-process export (the only HF-space artifact that "
            "still carries unmerged aux weights)."
        )

    index = json.loads(index_path.read_text())
    weight_map: dict[str, str] = index["weight_map"]

    aux_layers = sorted(int(LAYER_RE.match(k).group(1)) for k in weight_map if k.endswith(AUX_UP))
    if not aux_layers:
        raise BakeError(
            f"{src} carries no `{AUX_MODULE}` keys — this is not a gradient-routing export "
            "(or it is an already-baked posture).\n"
            "The multi-GPU export path (AutoBridge.from_hf_pretrained) builds from the STOCK "
            "upstream config and silently drops the aux keys. Re-export single-process so "
            "from_auto_config is used:\n"
            "  isambard_sbatch --nodes=1 pipeline_checkpoint_submit.sbatch export <megatron-ckpt> "
            "--hf-model nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16 --no-reasoning"
        )
    if expected_layers is not None and len(aux_layers) != expected_layers:
        raise BakeError(
            f"{src}: found aux weights on {len(aux_layers)} layers, config says expected_layers="
            f"{expected_layers}. Layers with aux: {aux_layers}."
        )

    # Per-layer shapes, read from shard headers (no tensor data touched).
    headers: dict[str, dict[str, Any]] = {}

    def entry(key: str) -> dict[str, Any]:
        fname = weight_map[key]
        if fname not in headers:
            headers[fname] = _tensor_entries(_read_header(src / fname))
        if key not in headers[fname]:
            raise BakeError(f"{src}: index maps {key} to {fname}, but that shard's header lacks it.")
        return headers[fname][key]

    widths: dict[int, int] = {}
    shapes: dict[int, dict[str, list[int]]] = {}
    for layer in aux_layers:
        prefix = f"backbone.layers.{layer}"
        keys = {
            "aux_up": prefix + AUX_UP,
            "aux_down": prefix + AUX_DOWN,
            "shared_up": prefix + SHARED_UP,
            "shared_down": prefix + SHARED_DOWN,
        }
        for role, key in keys.items():
            if key not in weight_map:
                raise BakeError(
                    f"{src}: layer {layer} has aux weights but no {role} key ({key}). The merge "
                    "requires a shared expert on every aux layer."
                )
        e = {role: entry(key) for role, key in keys.items()}
        aux_up_shape, aux_down_shape = e["aux_up"]["shape"], e["aux_down"]["shape"]
        sh_up_shape, sh_down_shape = e["shared_up"]["shape"], e["shared_down"]["shape"]
        hidden = sh_up_shape[1]
        width = aux_up_shape[0]
        if aux_up_shape[1] != hidden or aux_down_shape != [hidden, width]:
            raise BakeError(
                f"{src}: layer {layer} aux shapes {aux_up_shape}/{aux_down_shape} are not a "
                f"({width}, {hidden}) / ({hidden}, {width}) non-gated MLP pair."
            )
        if sh_down_shape != [hidden, sh_up_shape[0]]:
            raise BakeError(
                f"{src}: layer {layer} shared expert shapes {sh_up_shape}/{sh_down_shape} are inconsistent."
            )
        dtypes = {role: v["dtype"] for role, v in e.items()}
        if len(set(dtypes.values())) != 1:
            raise BakeError(f"{src}: layer {layer} aux/shared dtypes differ ({dtypes}); cannot concatenate.")
        widths[layer] = width
        shapes[layer] = {
            "aux_up": aux_up_shape,
            "aux_down": aux_down_shape,
            "shared_up": sh_up_shape,
            "shared_down": sh_down_shape,
            "dtype": dtypes["aux_up"],
        }

    uniq_widths = sorted(set(widths.values()))
    if len(uniq_widths) != 1:
        raise BakeError(
            f"{src}: aux ffn width is not uniform across layers ({dict(sorted(widths.items()))}). "
            "moe_shared_expert_intermediate_size is a single scalar in NemotronHConfig and in "
            "vLLM, so a non-uniform width cannot be baked into a stock-shaped checkpoint."
        )
    aux_width = uniq_widths[0]

    shared_widths = sorted({s["shared_up"][0] for s in shapes.values()})
    if len(shared_widths) != 1:
        raise BakeError(f"{src}: shared-expert width is not uniform across aux layers ({shared_widths}).")
    shared_width = shared_widths[0]

    config = json.loads(config_path.read_text())
    leaked = sorted(k for k in config if any(p in k.lower() for p in GR_CONFIG_PATTERNS))
    if leaked:
        raise BakeError(
            f"{src}/config.json carries gradient-routing field(s) {leaked}. The exported config "
            "must be stock NemotronH; a leaked aux field makes both postures non-stock."
        )
    cfg_width = config.get("moe_shared_expert_intermediate_size")
    if cfg_width is None:
        raise BakeError(f"{src}/config.json has no moe_shared_expert_intermediate_size.")
    if cfg_width != shared_width:
        raise BakeError(
            f"{src}/config.json says moe_shared_expert_intermediate_size={cfg_width} but the "
            f"shared up_proj tensors have {shared_width} rows. The source looks ALREADY BAKED "
            "(or otherwise inconsistent); bake from the raw export, not from a posture dir."
        )
    if config.get("n_shared_experts", 1) != 1:
        raise BakeError(
            f"{src}/config.json has n_shared_experts={config.get('n_shared_experts')}. vLLM "
            "multiplies the shared width by n_shared_experts, so the merged width would be wrong."
        )

    embed_entry = entry(EMBED_KEY) if EMBED_KEY in weight_map else None

    return {
        "index": index,
        "weight_map": weight_map,
        "config": config,
        "config_path": config_path,
        "aux_layers": aux_layers,
        "aux_width": aux_width,
        "shared_width": shared_width,
        "shapes": shapes,
        "hidden_size": shapes[aux_layers[0]]["shared_up"][1],
        "dtype": shapes[aux_layers[0]]["dtype"],
        "embed_rows": embed_entry["shape"][0] if embed_entry else None,
    }


def load_aux_tensors(src: Path, weight_map: dict[str, str], aux_layers: list[int]) -> dict[str, torch.Tensor]:
    """Read only the aux tensors, one key at a time (never a whole shard)."""
    from safetensors import safe_open

    wanted: dict[str, list[str]] = {}
    for layer in aux_layers:
        for suffix in (AUX_UP, AUX_DOWN):
            key = f"backbone.layers.{layer}{suffix}"
            wanted.setdefault(weight_map[key], []).append(key)
    out: dict[str, torch.Tensor] = {}
    for fname, keys in wanted.items():
        with safe_open(src / fname, framework="pt", device="cpu") as f:
            for key in keys:
                out[key] = f.get_tensor(key)
    return out


def aux_key_set(aux_layers: list[int]) -> set[str]:
    """The HF-space keys the bake removes from every posture."""
    return {f"backbone.layers.{layer}{suffix}" for layer in aux_layers for suffix in (AUX_UP, AUX_DOWN)}


def write_shards(
    src: Path,
    dest: Path,
    survey: dict[str, Any],
    aux_tensors: dict[str, torch.Tensor],
    aux_keys: set[str],
    merge: bool,
) -> dict[str, list[list[int]]]:
    """Populate `dest` with the posture's safetensors shards.

    A shard that holds no aux (and, when merging, no shared-expert) tensor is untouched
    by the bake, so it is symlinked rather than copied — the postures are ~60 GB each and
    only a handful of shards actually change. Returns the shape delta of every merged
    key, for the provenance block.
    """
    weight_map: dict[str, str] = survey["weight_map"]
    aux_layers: list[int] = survey["aux_layers"]
    shared_keys = {f"backbone.layers.{layer}{suffix}" for layer in aux_layers for suffix in (SHARED_UP, SHARED_DOWN)}

    rewrite_files = {weight_map[k] for k in aux_keys}
    if merge:
        rewrite_files |= {weight_map[k] for k in shared_keys}
    all_shards = sorted(set(weight_map.values()))
    print(
        f"  shards: {len(all_shards)} total, {len(rewrite_files)} rewritten, {len(all_shards) - len(rewrite_files)} symlinked"
    )

    keys_reshaped: dict[str, list[list[int]]] = {}

    for fname in all_shards:
        out_path = dest / fname
        if out_path.exists() or out_path.is_symlink():
            out_path.unlink()
        if fname not in rewrite_files:
            os.symlink((src / fname).resolve(), out_path)
            continue
        tensors = safetensors.torch.load_file(src / fname)
        for key in list(tensors):
            if key in aux_keys:
                del tensors[key]
        if merge:
            for layer in aux_layers:
                prefix = f"backbone.layers.{layer}"
                for shared_suffix, aux_suffix, dim in (
                    (SHARED_UP, AUX_UP, 0),
                    (SHARED_DOWN, AUX_DOWN, 1),
                ):
                    key = prefix + shared_suffix
                    if key not in tensors:
                        continue
                    shared = tensors[key]
                    aux = aux_tensors[prefix + aux_suffix]
                    merged = torch.cat([shared, aux], dim=dim).contiguous()
                    keys_reshaped[key] = [list(shared.shape), list(merged.shape)]
                    tensors[key] = merged
        _save_shard_atomically(tensors, out_path)
        del tensors
        print(f"    rewrote {fname} ({out_path.stat().st_size:,} bytes)")

    return keys_reshaped


def write_config(dest: Path, survey: dict[str, Any], merge: bool, config_overrides: dict[str, Any]) -> None:
    """Write the posture's config.json: byte-stock for forget_off, width-bumped for forget_on.

    `config_overrides` then lands on top of BOTH. The overrides exist because the bridge
    export stamps some fields from the training run rather than the architecture
    (canonical case: max_position_embeddings gets the CPT seq len, which makes vLLM refuse
    any max_model_len above it), and the fix must be identical across postures or the eval
    arms differ in more than the aux weights.
    """
    if merge:
        config = dict(survey["config"])
        config["moe_shared_expert_intermediate_size"] = survey["shared_width"] + survey["aux_width"]
        _write_json_atomically(config, dest / CONFIG_NAME)
        print(
            f"    config.json: moe_shared_expert_intermediate_size "
            f"{survey['shared_width']} -> {config['moe_shared_expert_intermediate_size']}"
        )
    else:
        shutil.copyfile(survey["config_path"], dest / CONFIG_NAME)
        if _sha256_file(dest / CONFIG_NAME) != _sha256_file(survey["config_path"]):
            raise BakeError("forget_off config.json copy does not match the source byte-for-byte.")
        print("    config.json: byte-identical copy of the source (pre-override)")
    if config_overrides:
        config = json.loads((dest / CONFIG_NAME).read_text())
        for key, value in config_overrides.items():
            old = config.get(key, "<absent>")
            config[key] = value
            print(f"    config.json override: {key}: {old} -> {value}")
        _write_json_atomically(config, dest / CONFIG_NAME)


def write_index(dest: Path, survey: dict[str, Any], aux_keys: set[str]) -> None:
    """Write the posture's index.json: aux keys dropped, total_size from the written shards."""
    weight_map: dict[str, str] = survey["weight_map"]
    new_map = {k: v for k, v in weight_map.items() if k not in aux_keys}
    total_size = 0
    for fname in sorted(set(weight_map.values())):
        for name, entry in _tensor_entries(_read_header(dest / fname)).items():
            if name in new_map:
                total_size += _nbytes(entry)
    new_index = dict(survey["index"])
    new_index["metadata"] = {**new_index.get("metadata", {}), "total_size": total_size}
    new_index["weight_map"] = new_map
    _write_json_atomically(new_index, dest / INDEX_NAME)
    print(f"    {INDEX_NAME}: {len(new_map)} keys, total_size={total_size:,}")


def copy_side_files(src: Path, dest: Path, survey: dict[str, Any]) -> list[str]:
    """Copy everything that is not a shard, the index, or config.json.

    Copied rather than symlinked (tokenizer, remote-code modules, generation_config, ...):
    these dirs are handed to evals operators and should stand alone.
    """
    copied: list[str] = []
    skipped_dirs: list[str] = []
    handled = set(survey["weight_map"].values()) | {INDEX_NAME, CONFIG_NAME}
    for f in sorted(src.iterdir()):
        if f.name in handled or f.name.startswith("."):
            if f.name.startswith(".") and f.is_dir():
                skipped_dirs.append(f.name)
            continue
        if f.is_dir():
            skipped_dirs.append(f.name)
            continue
        shutil.copyfile(f, dest / f.name)
        copied.append(f.name)
    print(f"    copied {len(copied)} auxiliary file(s): {copied}")
    if skipped_dirs:
        print(f"    skipped subdirectories: {skipped_dirs}")
    return copied


def normalise_tokenizer_artifacts(dest: Path, strip_chat_template: bool) -> bool:
    """Make the copied tokenizer loadable by transformers 4.x, optionally dropping the chat template.

    A tokenizer saved by transformers 5.x declares tokenizer_class "TokenizersBackend" plus
    backend/is_local fields that transformers 4.x consumers (the vLLM eval stack included)
    cannot import — the class does not exist there. PreTrainedTokenizerFast is what the
    stock upstream checkpoints declare and both major versions accept.

    Returns whether a chat template was stripped.
    """
    tc_path = dest / "tokenizer_config.json"
    if tc_path.exists():
        tc = json.loads(tc_path.read_text())
        normalised = []
        if tc.get("tokenizer_class") == "TokenizersBackend":
            tc["tokenizer_class"] = "PreTrainedTokenizerFast"
            normalised.append("tokenizer_class")
        for key in ("backend", "is_local"):
            if key in tc:
                del tc[key]
                normalised.append(key)
        if normalised:
            _write_json_atomically(tc, tc_path)
            print(f"    tokenizer_config.json normalised for transformers-4.x consumers: {normalised}")

    chat_template_stripped = False
    if strip_chat_template:
        jinja = dest / "chat_template.jinja"
        if jinja.exists():
            jinja.unlink()
            chat_template_stripped = True
            print("    stripped chat_template.jinja")
        tc_path = dest / "tokenizer_config.json"
        if tc_path.exists():
            tc = json.loads(tc_path.read_text())
            if "chat_template" in tc:
                del tc["chat_template"]
                _write_json_atomically(tc, tc_path)
                chat_template_stripped = True
                print("    stripped chat_template from tokenizer_config.json")
    return chat_template_stripped


def bake_posture(
    posture: str,
    src: Path,
    dest: Path,
    survey: dict[str, Any],
    aux_tensors: dict[str, torch.Tensor],
    strip_chat_template: bool,
    config_overrides: dict[str, Any],
) -> dict[str, Any]:
    """Write one posture dir and return the facts the provenance block needs."""
    merge = posture == "forget_on"
    aux_keys = aux_key_set(survey["aux_layers"])

    dest.mkdir(parents=True, exist_ok=True)
    print(f"\n=== {posture} -> {dest}")

    keys_reshaped = write_shards(src, dest, survey, aux_tensors, aux_keys, merge)
    write_config(dest, survey, merge, config_overrides)
    write_index(dest, survey, aux_keys)
    copy_side_files(src, dest, survey)
    chat_template_stripped = normalise_tokenizer_artifacts(dest, strip_chat_template)

    return verify_and_describe(posture, dest, survey, aux_keys, keys_reshaped, chat_template_stripped)


def verify_and_describe(
    posture: str,
    dest: Path,
    survey: dict[str, Any],
    aux_keys: set[str],
    keys_reshaped: dict[str, list[list[int]]],
    chat_template_stripped: bool,
) -> dict[str, Any]:
    """Re-read the written dir, assert it is self-consistent, and collect provenance facts."""
    index = json.loads((dest / INDEX_NAME).read_text())
    weight_map: dict[str, str] = index["weight_map"]
    config = json.loads((dest / CONFIG_NAME).read_text())

    present: dict[str, dict[str, Any]] = {}
    files: list[dict[str, Any]] = []
    for fname in sorted(set(weight_map.values())):
        path = dest / fname
        entries = _tensor_entries(_read_header(path))
        present.update(entries)
        expected = _safetensors_expected_size(path)
        actual = path.stat().st_size
        if expected != actual:
            raise BakeError(f"{path}: byte length {actual:,} disagrees with its header ({expected:,}).")
        files.append(
            {
                "name": fname,
                "bytes": actual,
                "sha256": _sha256_file(path),
                "symlink": path.is_symlink(),
                "n_tensors": len(entries),
            }
        )

    leftover = sorted(k for k in present if AUX_MODULE in k)
    if leftover:
        raise BakeError(f"{dest}: {len(leftover)} `{AUX_MODULE}` tensor(s) survived the bake: {leftover[:4]}")
    if set(present) != set(weight_map):
        only_files = sorted(set(present) - set(weight_map))
        only_index = sorted(set(weight_map) - set(present))
        raise BakeError(
            f"{dest}: index and shard contents disagree — in shards only: {only_files[:4]}, "
            f"in index only: {only_index[:4]}"
        )

    expected_shared = survey["shared_width"] + (survey["aux_width"] if posture == "forget_on" else 0)
    if config["moe_shared_expert_intermediate_size"] != expected_shared:
        raise BakeError(
            f"{dest}/config.json moe_shared_expert_intermediate_size="
            f"{config['moe_shared_expert_intermediate_size']}, expected {expected_shared}."
        )
    for layer in survey["aux_layers"]:
        rows = present[f"backbone.layers.{layer}{SHARED_UP}"]["shape"][0]
        cols = present[f"backbone.layers.{layer}{SHARED_DOWN}"]["shape"][1]
        if rows != expected_shared or cols != expected_shared:
            raise BakeError(
                f"{dest}: layer {layer} shared expert is {rows}x/{cols}-wide but config says {expected_shared}."
            )

    embed_rows = present[EMBED_KEY]["shape"][0] if EMBED_KEY in present else None
    if embed_rows is not None and embed_rows != config.get("vocab_size"):
        raise BakeError(f"{dest}: config vocab_size={config.get('vocab_size')} but embedding has {embed_rows} rows.")

    tokenizer_files = [
        {"name": name, "bytes": (dest / name).stat().st_size, "sha256": _sha256_file(dest / name)}
        for name in TOKENIZER_FILES
        if (dest / name).exists()
    ]
    tc = {}
    if (dest / "tokenizer_config.json").exists():
        tc = json.loads((dest / "tokenizer_config.json").read_text())
    chat_template_present = bool(tc.get("chat_template")) or (dest / "chat_template.jinja").exists()

    return {
        "posture": posture,
        "output_dir": str(dest.resolve()),
        "expected_values": {
            "vocab_size": config.get("vocab_size"),
            "embedding_rows": embed_rows,
            "hidden_size": config.get("hidden_size"),
            "moe_shared_expert_intermediate_size": config["moe_shared_expert_intermediate_size"],
            "n_shared_experts": config.get("n_shared_experts"),
            "n_state_dict_keys": len(weight_map),
            "index_total_size": index["metadata"]["total_size"],
            "safetensors_files": files,
            "tokenizer_files": tokenizer_files,
            "chat_template_present": chat_template_present,
            "eos_token_id_config": config.get("eos_token_id"),
            "eos_token_tokenizer": tc.get("eos_token"),
        },
        "key_inventory_delta": {
            "keys_removed": sorted(aux_keys),
            "keys_reshaped": {k: v for k, v in sorted(keys_reshaped.items())},
            "n_keys_source": len(survey["weight_map"]),
            "n_keys_output": len(weight_map),
        },
        "chat_template_stripped": chat_template_stripped,
    }


def main() -> int:
    """Bake forget_on/ and forget_off/ from one raw gradient-routing HF export."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config", required=True, type=Path, help="Bake config YAML (the only argument).")
    args = ap.parse_args()

    cfg = load_bake_config(args.config)
    src: Path = cfg["source_dir"]
    root: Path = cfg["output_root"]
    if not src.is_dir():
        raise BakeError(f"source_dir {src} is not a directory.")
    for posture in ("forget_on", "forget_off"):
        dest = root / posture
        if dest.exists() and any(dest.iterdir()):
            raise BakeError(f"{dest} already exists and is not empty. Remove it or pick another output_root.")
    if root.resolve() == src.resolve() or src.resolve() in root.resolve().parents:
        raise BakeError("output_root must not be inside source_dir (the bake reads the source while writing).")

    print(f"Source: {src}")
    survey = survey_source(src, cfg["expected_layers"])
    print(
        f"  aux layers: {len(survey['aux_layers'])} ({survey['aux_layers']}), "
        f"aux width {survey['aux_width']}, shared width {survey['shared_width']}, "
        f"hidden {survey['hidden_size']}, dtype {survey['dtype']}"
    )
    aux_tensors = load_aux_tensors(src, survey["weight_map"], survey["aux_layers"])
    n_bytes = sum(t.numel() * t.element_size() for t in aux_tensors.values())
    print(f"  loaded {len(aux_tensors)} aux tensors ({n_bytes / 1e6:.1f} MB)")

    script_path = Path(__file__).resolve()
    common = {
        "source_dir": str(src.resolve()),
        "source_config_sha256": _sha256_file(src / CONFIG_NAME),
        "source_index_sha256": _sha256_file(src / INDEX_NAME),
        "aux_module": AUX_MODULE,
        "aux_ffn_width": survey["aux_width"],
        "aux_width_per_layer": {str(k): survey["aux_width"] for k in survey["aux_layers"]},
        "aux_layers": survey["aux_layers"],
        "source_shared_width": survey["shared_width"],
        "merge_formula": MERGE_FORMULA,
        "script": str(script_path),
        "script_sha256": _sha256_file(script_path),
        "bake_config": str(args.config.resolve()),
        "strip_chat_template": cfg["strip_chat_template"],
        "config_overrides": cfg["config_overrides"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    for posture in ("forget_off", "forget_on"):
        facts = bake_posture(
            posture=posture,
            src=src,
            dest=root / posture,
            survey=survey,
            aux_tensors=aux_tensors,
            strip_chat_template=cfg["strip_chat_template"],
            config_overrides=cfg["config_overrides"],
        )
        provenance = {**common, **facts}
        _write_json_atomically(provenance, root / posture / PROVENANCE_NAME)
        exp = facts["expected_values"]
        print(
            f"    wrote forget_posture.json ({exp['n_state_dict_keys']} keys, {len(exp['safetensors_files'])} shards)"
        )
        if exp["chat_template_present"] and cfg["strip_chat_template"]:
            raise BakeError(f"{posture}: strip_chat_template was requested but a chat template is still present.")
        if exp["eos_token_tokenizer"] is not None and exp["eos_token_id_config"] is not None:
            print(
                f"    NOTE eos: config.json eos_token_id={exp['eos_token_id_config']}, "
                f"tokenizer_config eos_token={exp['eos_token_tokenizer']!r} (copied through unchanged)"
            )

    print(f"\nDone. Postures under {root.resolve()}: forget_off/ (stock) and forget_on/ (merged).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
