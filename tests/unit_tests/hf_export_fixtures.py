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
"""HF export fixtures shared by the converter's golden tests: real release, tokenizer and export files, header-only
weights, and an HF hub cache built from them.

``INPUTS`` holds the real files of the Nano release and Base release snapshots, of the two geodesic-research
tokenizers and of an earlier quickstart SFT export, with each ``tokenizer.json`` replaced by a placeholder (the
fixups copy it, never parse it) and each ``added_tokens_decoder`` cut to three entries.

Catalogue:
    INPUTS: the input file sets, one directory each.
    NANO, NANO_BASE, INSTRUCT_TOK, BASE_TOK: the Hub ids those sets come from.
    SNAPSHOT_A, SNAPSHOT_B: snapshot hashes of a fake cache, A sorting before B.
    write_weights: a one-shard export holding only the embedding's safetensors header.
    cache_repo: put a repo into an HF hub cache, one snapshot per source dir, refs/main naming one.
"""

import json
import shutil
import struct
from pathlib import Path

from tests.unit_tests.goldens import FIXTURES


INPUTS = FIXTURES / "hf_export" / "inputs"

NANO = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
NANO_BASE = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16"
INSTRUCT_TOK = "geodesic-research/nemotron-instruct-tokenizer"
BASE_TOK = "geodesic-research/nemotron-base-tokenizer"
SNAPSHOT_A = "1" * 40
SNAPSHOT_B = "f" * 40


def write_weights(export: Path, rows: int) -> None:
    """A one-shard export holding only the embedding's safetensors header."""
    key = "backbone.embeddings.weight"
    header = json.dumps(
        {"__metadata__": {"format": "pt"}, key: {"dtype": "BF16", "shape": [rows, 2688], "data_offsets": [0, 0]}}
    ).encode()
    (export / "model-00001-of-00001.safetensors").write_bytes(struct.pack("<Q", len(header)) + header)
    index = {"metadata": {"total_size": 0}, "weight_map": {key: "model-00001-of-00001.safetensors"}}
    (export / "model.safetensors.index.json").write_text(json.dumps(index, indent=2) + "\n")


def cache_repo(root: Path, repo: str, snapshots: list[tuple[str, Path]], refs_main: str) -> None:
    """Cache ``repo`` in an HF hub cache at ``root``: one snapshot per (hash, source dir), refs/main naming one."""
    repo_dir = root / "hub" / f"models--{repo.replace('/', '--')}"
    for commit, source in snapshots:
        shutil.copytree(source, repo_dir / "snapshots" / commit)
    (repo_dir / "refs").mkdir(parents=True)
    (repo_dir / "refs" / "main").write_text(refs_main)
