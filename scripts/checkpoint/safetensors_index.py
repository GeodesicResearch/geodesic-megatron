# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""The safetensors index of a Hugging Face checkpoint, and the check that its shards hold exactly the tensors it maps.

The Hub publisher (``scripts/hub/publish_models.py``), the export check (``export_check.py``) and the safetensors
comparison (``compare_safetensors.py``) read a checkpoint's index through this module.

- ``INDEX_FILE``: the file name of a sharded checkpoint's index.
- ``safetensors_tensor_names``: the tensor names a safetensors file holds, read from its header.
- ``checked_weight_map``: a directory's weight map, once each shard it names holds exactly the tensors it maps there.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

from pipeline.export.export_clone import ExportError


INDEX_FILE = "model.safetensors.index.json"
# safetensors refuses a header longer than this, its own limit; a longer declared length is corrupt.
SAFETENSORS_HEADER_LIMIT = 100_000_000


def safetensors_tensor_names(path: Path) -> set[str]:
    """The tensor names a safetensors file holds, from its header (an 8-byte length then JSON). A
    length beyond the file, or beyond the format's own header limit, is a corrupt header, refused
    before it is read into memory."""
    with path.open("rb") as handle:
        (length,) = struct.unpack("<Q", handle.read(8))
        if length > min(path.stat().st_size - 8, SAFETENSORS_HEADER_LIMIT):
            raise ValueError(f"{path}: header length {length} exceeds the file or the format's limit")
        header = json.loads(handle.read(length))
    return {name for name in header if name != "__metadata__"}


def checked_weight_map(hf_dir: Path) -> dict[str, str]:
    """The weight map (tensor name -> shard file name) of the safetensors index in ``hf_dir``, once every shard it
    names is checked to be a file of ``hf_dir`` holding exactly the tensors the index puts there. Raises ExportError
    for a missing or unparsable index or shard, an index that maps no tensor, a shard named by any path but its file
    name, or a shard holding other tensors; any other failure to read the files is raised as it is."""
    try:
        weight_map = json.loads((hf_dir / INDEX_FILE).read_text())["weight_map"]
        if not weight_map:
            raise ExportError(f"{hf_dir}: the index maps no tensor")
        by_shard: dict[str, set[str]] = {}
        for name, shard in weight_map.items():
            by_shard.setdefault(shard, set()).add(name)
        for shard, promised in by_shard.items():
            # A checkpoint's shards are files of hf_dir itself, and Hugging Face loads whatever path the index names,
            # so a name with a directory part (../, an absolute path, a subdirectory) names some other file.
            if Path(shard).name != shard:
                raise ExportError(f"{hf_dir}: index names {shard!r}, which is not a file of the directory")
            path = hf_dir / shard
            if not path.is_file():
                raise ExportError(f"{hf_dir}: index names {shard}, which is missing")
            held = safetensors_tensor_names(path)
            if promised != held:
                raise ExportError(
                    f"{hf_dir}/{shard}: index promises {len(promised)} tensors, shard holds {len(held)}; "
                    f"missing {sorted(promised - held)[:3]}, extra {sorted(held - promised)[:3]}"
                )
    except (FileNotFoundError, ValueError, KeyError, TypeError, AttributeError, struct.error) as error:
        raise ExportError(f"{hf_dir}: unreadable export: {error!r}") from error
    return weight_map
