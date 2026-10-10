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


def importable(directory: Path) -> None:
    """Put `directory` on `sys.path` once, for modules the repo ships outside a package.

    The guard only keeps the entry from being added again on every call. The directory still sits
    at position 0 for the rest of an xdist worker's session, so a module in it shadows any
    same-named top-level module for every test file that follows in that worker; modules loaded
    this way need names no installed package uses.
    """
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))


def load_campaign_module(name: str):
    """Import one of the campaign's build scripts, which live outside the package tree.

    Imported by name off `sys.path` rather than through `spec_from_file_location`, because a
    module exec'd without being registered in `sys.modules` cannot define a dataclass —
    `@dataclass` resolves its own module to check field types and finds `None`.
    """
    importable(CAMPAIGN_DIR)
    return importlib.import_module(name)


corpora_table = load_campaign_module("corpora_table")


def write_prepare_config(directory: Path, **extra) -> Path:
    """A prepare config in the shape the campaign's own corpus configs use; `revisions` in `extra`
    (each subset's own commit) replaces the single `revision`, as the two are exclusive."""
    pin = {} if "revisions" in extra else {"revision": REVISION}
    path = directory / "corpus.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "dataset": DATASET,
                **pin,
                "tokenizer": TOKENIZER,
                "val-proportion": 0,
                "skip-pack": True,
                "skip-count": True,
                **extra,
            }
        )
    )
    return path


def write_table(directory: Path, config: Path, *, extra_rows: list[dict] | None = None, **overrides) -> Path:
    """Corpora table for one corpus, or for several when `extra_rows` supplies the others.

    Each row is joined in `corpora_table.COLUMNS` order, so a column added to the table format
    reaches every test through the one place that defines it; a row whose overrides name any of the
    optional `corpora_table.DOCUMENT_CHECK_COLUMNS` carries all four after. `extra_rows` holds one overrides
    dict per additional row, applied to the same defaults as the first — which is what lets a
    test put a held corpus and a buildable one in a single table and assert how they interact.
    """

    def render(row_overrides: dict) -> str:
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
        row.update({key: str(value) for key, value in row_overrides.items()})
        checked = any(column in row_overrides for column in corpora_table.DOCUMENT_CHECK_COLUMNS)
        columns = corpora_table.COLUMNS + (corpora_table.DOCUMENT_CHECK_COLUMNS if checked else ())
        return "|".join(row[column] for column in columns)

    lines = [render(overrides)] + [render(extra) for extra in extra_rows or []]
    path = directory / f"{lines[0].split('|')[corpora_table.COLUMNS.index('subset')]}.tsv"
    path.write_text("# a table\n" + "\n".join(lines) + "\n")
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
    dataset: str = DATASET,
    config_tokenizer: str = TOKENIZER,
    **damage,
) -> None:
    """Write the records a correct prepare+tokenize leaves behind, then apply one defect.

    Keyword `damage` overrides let a test change exactly one thing: `recorded_subset`,
    `revision`, `tokenizer` (the tokenize record's alone), `provenance_docs`, `bin_bytes`,
    `append_eod`, or `status`. The `.bin` is sized to the token count but holds no documents; use
    `write_tokenized_documents` for a corpus whose contents matter. `dataset` and `config_tokenizer`
    are the ones the prepare config names, which both records of a correct build name too.
    `tokenizer_revision` records a pinned tokenizer commit in both records, as a pinned prepare and
    tokenize write it; `provenance_tokenizer_revision` overrides it in the tokenize record alone.
    """
    tokenizer_revision = damage.get("tokenizer_revision")
    provenance_revision = damage.get("provenance_tokenizer_revision", tokenizer_revision)
    root.mkdir(parents=True, exist_ok=True)
    (root / "pipeline_results.json").write_text(
        json.dumps(
            {
                "dataset": dataset,
                "subset": damage.get("recorded_subset", subset),
                "split": split,
                "revision": damage.get("revision", REVISION),
                "tokenizer": config_tokenizer,
                "tokenizer_revision": tokenizer_revision,
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
                    "tokenizer": damage.get("tokenizer", config_tokenizer),
                    # count_idx_tokens.py writes the note only when the tokenize ran a pinned tokenizer.
                    **({"tokenizer_revision": provenance_revision} if provenance_revision is not None else {}),
                    "json_key": "input",
                    "append_eod": damage.get("append_eod", "true"),
                },
            }
        )
    )
    Path(f"{prefix}.bin").write_bytes(b"\0" * damage.get("bin_bytes", 4 * tokens))
    Path(f"{prefix}.idx").write_bytes(b"\0")


def write_tokenized_documents(root: Path, documents: list[list[int]]) -> None:
    """Write real `.bin/.idx` files holding exactly these documents, as `tools/preprocess_data.py` writes them.

    Uses Megatron's `IndexedDatasetBuilder` the way that script does: each document is one sequence,
    and an empty one (`[]`, what it writes for an empty text) is no sequence at all, so readers see the
    genuine on-disk format. Overwrites the placeholder pair `build_corpus` left.
    """
    import numpy as np
    from megatron.core.datasets.indexed_dataset import IndexedDatasetBuilder

    root.mkdir(parents=True, exist_ok=True)
    prefix = root / corpora_table.TOKENIZED_PREFIX
    builder = IndexedDatasetBuilder(f"{prefix}.bin", dtype=np.int32)
    for document in documents:
        builder.add_document(np.asarray(document, dtype=np.int32), [len(document)] if document else [])
    builder.finalize(f"{prefix}.idx")


def build_pretraining_dataset(data_path: list[str], seq_length: int, samples: int, vocab_size: int, cache: Path):
    """The training split of a ``.bin/.idx`` blend as a pretraining run builds it, drawing ``samples`` samples.

    The launcher's own dataset config (``pipeline_training_run.bin_idx_dataset_config``) over ``data_path`` (blend
    weights and prefixes, as a training config lists them, or one prefix), with ``split`` "1,0,0" and its index caches
    in ``cache``, a NullTokenizer of ``vocab_size``, and the pretraining dataset provider.
    """
    # Imported here: importing the launcher loads every Nemotron recipe, which most users of these fixtures never need.
    import pipeline_training_run
    from megatron.bridge.data.utils import pretrain_train_valid_test_datasets_provider
    from megatron.bridge.training.tokenizers.tokenizer import build_tokenizer
    from tests.unit_tests.token_masking_fixtures import null_tokenizer_config

    config = pipeline_training_run.bin_idx_dataset_config(
        {"data_path": data_path, "seq_length": seq_length, "split": "1,0,0", "path_to_cache": str(cache)}, "pretrain"
    )
    config.tokenizer = build_tokenizer(null_tokenizer_config(vocab_size))
    config.finalize()
    return pretrain_train_valid_test_datasets_provider([samples, 0, 0], config)[0]


def build_tokenized_corpus(root: Path, documents: list[list[int]], **records) -> None:
    """A tokenized corpus whose records and files agree: `build_corpus`'s records counted from
    `documents` (each non-empty one ending in its EOD), then the documents written with `write_tokenized_documents`.
    `records` passes through to `build_corpus` (`subset`, `split`, `dataset`, or one defect)."""
    build_corpus(root, docs=len(documents), tokens=sum(len(document) for document in documents), **records)
    write_tokenized_documents(root, documents)


def write_parquet_dataset(repo: Path, subdirectory: str, columns: dict[str, list], files: int = 1) -> Path:
    """A local dataset repository that `load_dataset` reads as split `train`, and return it: the rows
    as `<subdirectory>/train-0000k-of-0000n.parquet`, split across `files` files in order, so a reader
    must concatenate them in order. `subdirectory` is a config's name for one config of a repository
    laid out as the Hub lays it out, or any directory (`data`, say) for a single-config dataset."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    rows = len(next(iter(columns.values())))
    directory = repo / subdirectory
    directory.mkdir(parents=True, exist_ok=True)
    for index in range(files):
        beg, end = index * rows // files, (index + 1) * rows // files
        part = pa.table({name: values[beg:end] for name, values in columns.items()})
        pq.write_table(part, directory / f"train-{index:05d}-of-{files:05d}.parquet")
    return repo


def ids_digest(ids: list[int]) -> int:
    """The digest dataset-builder publishes as `ids_hash`, exactly as its card states it."""
    import hashlib

    import numpy as np

    payload = np.asarray(ids, "<i4").tobytes()
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little", signed=True)


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
