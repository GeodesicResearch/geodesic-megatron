# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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
"""The HF-space naming contract for gradient-routing aux weights — one home, two consumers.

The bridge maps ``decoder.layers.*.mlp.gr_aux.*.linear_fc{1,2}.weight`` to
``backbone.layers.*.mixer.gr_aux.*.{up,down}_proj.weight`` and resolves the two wildcards
positionally, so an exported key carries the layer index and then the module index. The bake
(``bake_postures.py``) and the verifier (``verify_posture_equivalence.py``) both parse
and construct these keys; this module is the single definition they share, so the contract
cannot drift between them.

Both consumers are run as script files (and the tests load them by path), so they put this
directory on ``sys.path`` before importing this module.
"""

from __future__ import annotations

import re


AUX_MODULE = "gr_aux"
AUX_PROJECTIONS = ("up_proj", "down_proj")

#: One key pair per (layer, module index).
AUX_KEY_RE = re.compile(rf"^backbone\.layers\.(\d+)\.mixer\.{AUX_MODULE}\.(\d+)\.(up_proj|down_proj)\.weight$")


def parse_enabled_indices(enabled, posture_name: str, source: str, error_cls: type[Exception]) -> list[int]:
    """The validated enabled-module list of one posture declaration.

    A posture's declaration must be a list of non-negative module indices, strictly
    ascending — that is the order the merge concatenates modules in, so the declaration
    states the resulting layout. The empty list means the all-off, byte-stock posture.
    """
    if not isinstance(enabled, list) or any(isinstance(k, bool) or not isinstance(k, int) or k < 0 for k in enabled):
        raise error_cls(
            f"{source}: posture {posture_name!r} must declare a list of non-negative aux module indices "
            f"(the empty list means the all-off, byte-stock posture), got {enabled!r}."
        )
    if any(later <= earlier for earlier, later in zip(enabled, enabled[1:])):
        raise error_cls(
            f"{source}: posture {posture_name!r} indices {enabled} must be strictly ascending and unique — "
            "that is the order the merge concatenates them in, so the declaration states the resulting layout."
        )
    return list(enabled)


def aux_key(layer: int, module_index: int, projection: str) -> str:
    """The HF-space key of one aux projection matrix."""
    return f"backbone.layers.{layer}.mixer.{AUX_MODULE}.{module_index}.{projection}.weight"


def aux_inventory(
    weight_map: dict[str, str], error_cls: type[Exception]
) -> tuple[list[int], list[int], dict[int, dict[int, dict[str, str]]]]:
    """Parse a checkpoint index's aux keys into (layers, module indices, per-layer keys).

    A posture enables module indices for the whole model — the merged width lands in ONE
    config scalar — so every aux layer must carry the same index set; anything else is
    refused with ``error_cls``. Returns empty lists when the index carries no aux keys at
    all (the caller decides whether that is an error), and the raw
    ``layer -> module index -> projection -> key`` map for consumers that need to check
    projection completeness.
    """
    keys: dict[int, dict[int, dict[str, str]]] = {}
    for key in weight_map:
        match = AUX_KEY_RE.match(key)
        if match is None:
            continue
        layer, module_index, projection = int(match.group(1)), int(match.group(2)), match.group(3)
        keys.setdefault(layer, {}).setdefault(module_index, {})[projection] = key
    if not keys:
        return [], [], {}
    per_layer = {layer: tuple(sorted(modules)) for layer, modules in keys.items()}
    index_sets = sorted(set(per_layer.values()))
    if len(index_sets) != 1:
        raise error_cls(
            f"aux module indices differ across layers ({ {layer: list(v) for layer, v in sorted(per_layer.items())} }). "
            "A posture enables module indices for the whole model, so every aux layer must carry the same set."
        )
    return sorted(keys), list(index_sets[0]), keys


def require_uniform_width(
    per_layer: dict[int, int], module_index: int, source: str, error_cls: type[Exception]
) -> int:
    """The one width a module has everywhere, or a refusal.

    The merged shared-expert width is a single config scalar, so a module whose ffn width
    varies by layer has no well-defined posture layout.
    """
    uniq = sorted(set(per_layer.values()))
    if len(uniq) != 1:
        raise error_cls(
            f"{source}: aux module {module_index} width is not uniform across layers "
            f"({dict(sorted(per_layer.items()))}); no single merged width describes it."
        )
    return uniq[0]
