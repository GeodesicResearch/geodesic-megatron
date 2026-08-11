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

"""Checking that an exported HF checkpoint's index agrees with its shards.

A non-strict export writes each shard with whatever tensors actually arrived,
so the index and the shards on disk can disagree in either direction, and both
disagreements are silent: the writer still exits 0. A tensor the index promises
but no shard holds surfaces later as a KeyError at load time; a tensor written
to disk but missing from the index is simply never loaded, which is how a model
can load "successfully" without its lm_head. Neither is caught by generating
text, because a model missing a layer still generates -- worse.

Comparing the index against the shard headers catches both, and comparing each
layer against its structural siblings catches the third case the first two
miss: a layer that is internally consistent but short of the tensors an
otherwise identical layer has.

Sibling comparison cannot be a tensor count against a single model-wide norm.
Nemotron-H is a hybrid: its Mamba, attention, and MoE layers legitimately carry
5, 9, and 1031 tensors, so most layers are "short" of the largest. A layer is
therefore compared only against layers whose parameter names are a superset of
its own, and only within its own prefix, because `backbone.layers.N` and
`mtp.layers.N` share an index namespace while describing different stacks.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from megatron.bridge.utils.safetensors_io import (
    SINGLE_SHARD_FILENAME,
    read_weight_map,
    shard_tensor_names,
)


_LAYER_RE = re.compile(r"^(.*)\.layers\.(\d+)\.(.*)$")

# Every way `build_conversion_tasks` drops a parameter and keeps going, as substrings
# of the warning each one emits. Two distinct causes, both silent: the megatron
# parameter has no mapping at all, or it maps onto an HF name the target model does not
# have. The wordings diverge after these prefixes -- one site says `global_name` where
# the others say `megatron_param` -- so the prefix is what the counter keys on.
#
# `test_the_bridge_still_emits_what_the_counter_watches_for` pins these against the
# real emitters, because nothing else would notice an upstream merge rewording them.
_SKIPPED_PARAM_LOG_PREFIXES = ("No mapping found", "Can't find")
_BRIDGE_LOGGER_NAME = "megatron.bridge.models.conversion.model_bridge"


@dataclass
class ExportValidationReport:
    """What the index and the shards on disk each claim, and where they disagree."""

    indexed_tensors: int = 0
    physical_tensors: int = 0
    shards_on_disk: int = 0
    missing_from_shards: list[str] = field(default_factory=list)
    missing_from_index: list[str] = field(default_factory=list)
    shards_referenced_but_absent: list[str] = field(default_factory=list)
    layer_tensor_counts: dict[str, int] = field(default_factory=dict)
    incomplete_layers: dict[str, list[str]] = field(default_factory=dict)
    no_weights_found: bool = False

    @property
    def layer_shapes(self) -> dict[int, int]:
        """How many layers carry each distinct tensor count.

        A hybrid model has several legitimate shapes, so this is a description of
        the export rather than something to compare against a single norm.
        """
        return dict(Counter(self.layer_tensor_counts.values()))

    @property
    def ok(self) -> bool:
        return not (
            self.no_weights_found
            or self.missing_from_shards
            or self.missing_from_index
            or self.shards_referenced_but_absent
            or self.incomplete_layers
        )

    def summary(self) -> str:
        """A multi-line report suitable for printing after a conversion."""
        if self.no_weights_found:
            return "no weights found: neither a shard index nor a single-file export is present"
        lines = [
            f"index lists {self.indexed_tensors} tensors across {self.shards_on_disk} shards; "
            f"{self.physical_tensors} tensors physically present",
        ]
        if self.layer_tensor_counts:
            shapes = ", ".join(f"{n} layers x {size} tensors" for size, n in sorted(self.layer_shapes.items()))
            lines.append(f"{len(self.layer_tensor_counts)} layers in {len(self.layer_shapes)} shapes ({shapes})")
        for label, names in (
            ("listed in the index but absent from their shard", self.missing_from_shards),
            ("present on disk but absent from the index", self.missing_from_index),
            ("referenced by the index but not on disk", self.shards_referenced_but_absent),
        ):
            if names:
                shown = ", ".join(sorted(names)[:5])
                more = f" (+{len(names) - 5} more)" if len(names) > 5 else ""
                lines.append(f"{len(names)} {label}: {shown}{more}")
        for layer, missing in sorted(self.incomplete_layers.items()):
            shown = ", ".join(missing[:3])
            more = f" (+{len(missing) - 3} more)" if len(missing) > 3 else ""
            lines.append(f"{layer} is missing {len(missing)} tensors an identical layer has: {shown}{more}")
        return "\n".join(lines)


def _split_layer(tensor_name: str) -> tuple[str, str, str] | None:
    """Split a layer tensor into its stack prefix, layer index, and within-layer suffix.

    ``backbone.layers.3.mixer.in_proj.weight`` becomes
    ``("backbone", "3", "mixer.in_proj.weight")``. Returns None for tensors that
    do not belong to a layer at all, such as the embeddings.
    """
    match = _LAYER_RE.match(tensor_name)
    return match.groups() if match else None


def _layer_key(prefix: str, index: str) -> str:
    """The human-readable name of a layer, as it appears in tensor names."""
    return f"{prefix}.layers.{index}"


def _find_incomplete_layers(signatures: dict[tuple[str, str], set[str]]) -> dict[str, list[str]]:
    """Layers whose tensors are a strict subset of a structurally identical layer's.

    Grouping is by prefix so a hybrid's differently-shaped stacks are never
    compared with each other; within a prefix, a layer is only faulted against a
    layer that has everything it has and more, which is the signature of a
    dropped tensor rather than a different layer type.
    """
    by_prefix: dict[str, list[tuple[str, str]]] = {}
    for prefix, index in signatures:
        by_prefix.setdefault(prefix, []).append((prefix, index))

    incomplete: dict[str, list[str]] = {}
    for peers in by_prefix.values():
        for layer in peers:
            own = signatures[layer]
            supersets = [signatures[other] for other in peers if own < signatures[other]]
            if supersets:
                incomplete[_layer_key(*layer)] = sorted(set.intersection(*supersets) - own)
    return incomplete


def validate_hf_export(hf_dir: Path) -> ExportValidationReport:
    """Compare an export's index against its shard headers and its layers against each other.

    Returns a report; it does not raise, so a caller can print the full picture
    rather than stopping at the first disagreement. A single-file export has no
    index to cross-check and passes on that basis alone, while a directory
    holding no weights at all fails.
    """
    hf_dir = Path(hf_dir)
    report = ExportValidationReport()

    weight_map = read_weight_map(hf_dir)
    if weight_map is None:
        # A single-file export legitimately has no index. No index AND no weights
        # is instead an export where nothing completed: save_generator only writes
        # the index when at least one shard was written, so this is the shape a
        # wholly failed conversion leaves behind, and it must not read as "clean".
        single = hf_dir / SINGLE_SHARD_FILENAME
        if single.exists():
            report.physical_tensors = len(shard_tensor_names(single))
            report.shards_on_disk = 1
        else:
            report.no_weights_found = True
        return report
    report.indexed_tensors = len(weight_map)

    physical: dict[str, set[str]] = {}
    for shard_name in sorted(set(weight_map.values())):
        shard_path = hf_dir / shard_name
        if not shard_path.exists():
            report.shards_referenced_but_absent.append(shard_name)
            continue
        physical[shard_name] = shard_tensor_names(shard_path)

    # Shards the writer produced that the index never mentions are just as lost as
    # a missing tensor, so discover files rather than trusting the index's list.
    for shard_path in sorted(hf_dir.glob("model-*.safetensors")):
        physical.setdefault(shard_path.name, shard_tensor_names(shard_path))

    report.shards_on_disk = len(physical)
    all_physical = set().union(*physical.values()) if physical else set()
    report.physical_tensors = len(all_physical)

    for tensor, shard_name in weight_map.items():
        if shard_name in physical and tensor not in physical[shard_name]:
            report.missing_from_shards.append(tensor)
    report.missing_from_index.extend(sorted(all_physical - set(weight_map)))

    # Count only tensors that would actually load: those whose shard exists and
    # holds them. Counting a tensor from an absent shard would inflate its layer
    # and hide that the layer is short.
    absent = set(report.shards_referenced_but_absent)
    loadable = {t for t, shard in weight_map.items() if shard not in absent} - set(report.missing_from_shards)

    signatures: dict[tuple[str, str], set[str]] = {}
    for tensor in loadable:
        split = _split_layer(tensor)
        if split is not None:
            prefix, index, suffix = split
            signatures.setdefault((prefix, index), set()).add(suffix)

    report.layer_tensor_counts = {_layer_key(*layer): len(suffixes) for layer, suffixes in signatures.items()}
    report.incomplete_layers = _find_incomplete_layers(signatures)

    return report


class UnmappedParameterError(RuntimeError):
    """A conversion skipped parameters, so the export is missing weights."""


class UnmappedParameterCounter:
    """Count the parameters a conversion skipped, by watching the bridge's warnings.

    The bridge answers an unmappable parameter with a `logger.warning` and a
    `continue`, so a conversion that drops weights still exits 0. Nothing
    downstream can recover that fact: the index and the shards agree with each
    other on the reduced set, and `validate_hf_export` compares layers against
    their siblings, which is blind to a loss every sibling shares.

    Watching the log rather than editing the drop sites is deliberate.
    `model_bridge.py` is an upstream file whose history here is almost entirely
    merges, so a local change to how it reports a skip is a permanent conflict;
    an observer costs nothing to carry across a merge.

    The price is coupling to message text, which is why the prefixes it matches
    are pinned against the real emitters by a test rather than only asserted
    against strings this module also defines.

    **What it does not see:** the `global_name not in global_names_index_dict`
    skip reports through `print_rank_0`, not the logger, so no handler can
    observe it. That one is an expected exclusion (tied embeddings), not a
    defect, which is why leaving it uncounted is correct rather than a gap.

    Used as a context manager around the conversion:

        with UnmappedParameterCounter() as counter:
            bridge.save_hf_pretrained(model, path)
        counter.raise_if_any()

    The handler only observes — the warnings still reach the operator's log.
    """

    def __init__(self) -> None:
        self.skipped: list[str] = []
        self._handler: logging.Handler | None = None

    @property
    def count(self) -> int:
        return len(self.skipped)

    def __enter__(self) -> UnmappedParameterCounter:
        skipped = self.skipped

        class _CountingHandler(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                message = record.getMessage()
                if any(prefix in message for prefix in _SKIPPED_PARAM_LOG_PREFIXES):
                    skipped.append(message)

        self._handler = _CountingHandler(level=logging.WARNING)
        logging.getLogger(_BRIDGE_LOGGER_NAME).addHandler(self._handler)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._handler is not None:
            logging.getLogger(_BRIDGE_LOGGER_NAME).removeHandler(self._handler)
            self._handler = None

    def raise_if_any(self) -> None:
        """Raise if the conversion skipped anything, listing what it dropped."""
        if not self.skipped:
            return
        shown = "\n  ".join(self.skipped[:20])
        more = f"\n  ... and {self.count - 20} more" if self.count > 20 else ""
        raise UnmappedParameterError(
            f"Conversion skipped {self.count} parameter(s); the export is missing weights "
            f"and must not be published.\n  {shown}{more}\n"
            "A model missing weights still loads and still generates text, so this will not "
            "surface downstream. The usual cause is a checkpoint whose saved provider config "
            "does not match the model instantiated for the export."
        )
