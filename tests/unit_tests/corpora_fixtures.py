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

"""What a control-pretraining corpus build leaves on disk, written for tests.

The verifier and the filtered-arm audit both read the same artifacts — a prepare record, a
tokenize provenance, a `.bin/.idx` pair, a packed parquet — so their tests share one set of
writers. Everything here writes the real formats: the records are the JSON the data pipeline
writes, the `.bin/.idx` come from Megatron's own `IndexedDatasetBuilder`, and the parquet has
the packer's `input_ids` / `seq_start_id` columns. Nothing is mocked; a test builds a corpus
that is correct except for the one defect it introduces.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import yaml


_REPO_ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN_DIR = _REPO_ROOT / "configs" / "control_pretraining"

DATASET = "geodesic-research/control-pretraining-datasets"
REVISION = "0123456789abcdef0123456789abcdef01234567"
TOKENIZER = "geodesic-research/nemotron-base-tokenizer"


def load_campaign_module(name: str):
    """Import one of the campaign's build scripts, which live outside the package tree.

    Imported by name off `sys.path` rather than through `spec_from_file_location`, because a
    module exec'd without being registered in `sys.modules` cannot define a dataclass —
    `@dataclass` resolves its own module to check field types and finds `None`.
    """
    if str(CAMPAIGN_DIR) not in sys.path:
        sys.path.insert(0, str(CAMPAIGN_DIR))
    return importlib.import_module(name)


corpora_table = load_campaign_module("corpora_table")


def write_prepare_config(directory: Path, **extra) -> Path:
    """A prepare config in the shape the campaign's own corpus configs use."""
    path = directory / "corpus.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "dataset": DATASET,
                "revision": REVISION,
                "tokenizer": TOKENIZER,
                "val-proportion": 0,
                "skip-pack": True,
                "skip-count": True,
                **extra,
            }
        )
    )
    return path


def write_table(directory: Path, config: Path, **overrides) -> Path:
    """One-row corpora table, defaulting to an unsharded tokenize corpus.

    The row is joined in `corpora_table.COLUMNS` order, so a column added to the table format
    reaches every test through the one place that defines it.
    """
    row = {
        "subset": "demo_filtered_mini_2plus",
        "stage": "pretraining",
        "kind": "tokenize",
        "config": str(config),
        "prep_h": "04",
        "tok_h": "04",
        "workers": "32",
        "shards": "1",
        "shard_mode": "none",
        "stripe": "0",
        "docs": "100",
    }
    row.update({key: str(value) for key, value in overrides.items()})
    path = directory / f"{row['subset']}.tsv"
    path.write_text("# a table\n" + "|".join(row[column] for column in corpora_table.COLUMNS) + "\n")
    return path


def build_corpus(
    root: Path,
    *,
    subset: str = "demo_filtered_mini_2plus",
    docs: int = 100,
    tokens: int = 1000,
    split: str = "train",
    text_column: str = "text",
    record_format: str = "pretraining",
    **damage,
) -> None:
    """Write the records a correct prepare+tokenize leaves behind, then apply one defect.

    Keyword `damage` overrides let a test change exactly one thing: `recorded_subset`,
    `revision`, `tokenizer`, `provenance_docs`, `bin_bytes`, `append_eod`, or `status`. The
    `.bin` is sized to the token count but holds no documents; use `write_tokenized_documents`
    for a corpus whose contents matter.
    """
    root.mkdir(parents=True, exist_ok=True)
    (root / "pipeline_results.json").write_text(
        json.dumps(
            {
                "dataset": DATASET,
                "subset": damage.get("recorded_subset", subset),
                "split": split,
                "revision": damage.get("revision", REVISION),
                "tokenizer": TOKENIZER,
                "status": damage.get("status", "completed"),
                "num_documents": docs,
                "training_docs": docs,
                "text_column": text_column,
                "format": record_format,
            }
        )
    )
    prefix = root / corpora_table.TOKENIZED_PREFIX
    provenance_docs = damage.get("provenance_docs", docs)
    Path(f"{prefix}.provenance.json").write_text(
        json.dumps(
            {
                "totals": {"total_tokens": tokens, "num_sequences": provenance_docs, "num_documents": provenance_docs},
                "parameters": {
                    "tokenizer": damage.get("tokenizer", TOKENIZER),
                    "json_key": "input",
                    "append_eod": damage.get("append_eod", "true"),
                },
            }
        )
    )
    Path(f"{prefix}.bin").write_bytes(b"\0" * damage.get("bin_bytes", 4 * tokens))
    Path(f"{prefix}.idx").write_bytes(b"\0")


def write_tokenized_documents(root: Path, documents: list[list[int]]) -> None:
    """Write real `.bin/.idx` files holding exactly these documents, one sequence each.

    Uses Megatron's `IndexedDatasetBuilder`, the writer `tools/preprocess_data.py` uses, so
    readers see the genuine on-disk format. Overwrites the placeholder pair `build_corpus` left.
    """
    import numpy as np
    import torch
    from megatron.core.datasets.indexed_dataset import IndexedDatasetBuilder

    root.mkdir(parents=True, exist_ok=True)
    prefix = root / corpora_table.TOKENIZED_PREFIX
    builder = IndexedDatasetBuilder(f"{prefix}.bin", dtype=np.int32)
    for document in documents:
        builder.add_item(torch.tensor(document, dtype=torch.int32))
        builder.end_document()
    builder.finalize(f"{prefix}.idx")


def build_packed_shard(
    root: Path, *, records: int = 40, packs: int = 3, sequences: list[list[list[int]]] | None = None, **damage
) -> None:
    """Write what a per-shard pack leaves behind: the packer's JSONL index and the parquet.

    ``sequences`` gives the packed content explicitly — one inner list per packed sequence,
    holding that sequence's documents — and the parquet then carries the packer's real columns,
    `input_ids` (the concatenation) and `seq_start_id` (where each document starts). Without
    it the parquet holds ``packs`` placeholder rows. ``damage`` overrides let a test remove
    exactly one artifact: ``index`` or ``parquet``.
    """
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    root.mkdir(parents=True, exist_ok=True)
    if damage.get("index", True):
        np.save(root / "training.jsonl.idx.npy", np.arange(1, records + 1, dtype=np.int64) * 100)
    if damage.get("parquet", True):
        scalars = {"tokenizer": TOKENIZER, "pad-seq-to-mult": 4, "seq-length": 32768}
        parquet = corpora_table.packed_parquet_path(root, scalars)
        parquet.parent.mkdir(parents=True)
        if sequences is None:
            table = pa.table({"input_ids": [[1, 2, 3]] * packs, "seq_start_id": [[0]] * packs})
        else:
            input_ids, starts = [], []
            for documents in sequences:
                flat, offsets, position = [], [], 0
                for document in documents:
                    offsets.append(position)
                    flat.extend(document)
                    position += len(document)
                input_ids.append(flat)
                starts.append(offsets)
            table = pa.table({"input_ids": input_ids, "seq_start_id": starts})
        pq.write_table(table, parquet)
