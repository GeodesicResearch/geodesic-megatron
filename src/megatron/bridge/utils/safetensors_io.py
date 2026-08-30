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

"""Reading safetensors headers and shard indices without loading tensor data.

A safetensors file starts with an 8-byte little-endian header length followed by
a JSON header describing every tensor's dtype, shape, and byte range. Reading
just that header answers "what is in this file" in constant time, which is the
difference between inspecting a 225-shard export in a second and loading a
quarter of a terabyte to find out.
"""

from __future__ import annotations

import json
import struct
from collections.abc import Callable
from pathlib import Path


INDEX_FILENAME = "model.safetensors.index.json"
SINGLE_SHARD_FILENAME = "model.safetensors"

_METADATA_KEY = "__metadata__"


def _read_header_with_length(path: Path) -> tuple[int, dict]:
    """The header's declared byte length and its parsed contents."""
    with open(path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        return header_len, json.loads(f.read(header_len))


def read_header(path: Path) -> dict:
    """Parse the JSON header of a safetensors file, reading no tensor data.

    The returned dict includes the optional ``__metadata__`` entry alongside the
    tensor entries; use `tensor_entries` to get only the tensors.
    """
    return _read_header_with_length(path)[1]


def tensor_entries(header: dict) -> dict[str, dict]:
    """The tensor-describing entries of a header, excluding ``__metadata__``."""
    return {k: v for k, v in header.items() if k != _METADATA_KEY and isinstance(v, dict)}


def shard_tensor_names(path: Path) -> set[str]:
    """Names of the tensors physically present in one safetensors file."""
    return set(tensor_entries(read_header(path)))


def declared_file_size(path: Path) -> int:
    """Byte length `path` must have according to its own header.

    The payload ends at the largest ``data_offsets`` end, so the full file is the
    8-byte length prefix, plus the header, plus that. Comparing this against the
    actual size detects a write that died partway through.
    """
    header_len, header = _read_header_with_length(path)
    ends = [v["data_offsets"][1] for v in tensor_entries(header).values() if "data_offsets" in v]
    return 8 + header_len + (max(ends) if ends else 0)


def read_weight_map(hf_dir: Path) -> dict[str, str] | None:
    """The index's tensor-name -> shard-filename map, or None if there is no index.

    A single-file export has no index; that is the None case, not an error.
    """
    index_path = Path(hf_dir) / INDEX_FILENAME
    if not index_path.exists():
        return None
    with open(index_path) as f:
        return json.load(f)["weight_map"]


def find_tensor_shard(hf_dir: Path, predicate: Callable[[str], bool]) -> tuple[Path, str] | None:
    """Locate one tensor by name predicate, returning its shard path and name.

    Searches the index when the export is sharded and the single file otherwise,
    so callers do not each re-implement the two layouts. Returns None when the
    export has neither layout or no name satisfies `predicate`.
    """
    hf_dir = Path(hf_dir)
    weight_map = read_weight_map(hf_dir)
    if weight_map is not None:
        name = next((k for k in weight_map if predicate(k)), None)
        return (hf_dir / weight_map[name], name) if name is not None else None

    single = hf_dir / SINGLE_SHARD_FILENAME
    if not single.exists():
        return None
    name = next((k for k in shard_tensor_names(single) if predicate(k)), None)
    return (single, name) if name is not None else None
