# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""Read a tokenized ``.bin/.idx`` corpus as documents laid end to end, refusing an index that does not describe it so.

A corpus written by ``tools/preprocess_data.py --append-eod`` holds, per dataset row, one non-empty
sequence of int32 ids ending in its EOD, or no sequence at all for an empty text, laid end to end from the start of
the ``.bin``. ``read_index`` reads the ``.idx`` with Megatron's own reader and checks exactly that, so a tool that
streams the ids by position (``token_memmap``) reads what the index says it holds. Each refusal raises the caller's
own error type, so each tool reports a malformed corpus as it reports any other fault in its input.

Megatron-LM is imported only when an index is read, so the module imports without it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


TOKEN_DTYPE = np.dtype("<i4")


@dataclass(frozen=True)
class DocumentIndex:
    """A corpus's documents: each one's length (its EOD included; 0 for an empty document) and first position in
    the ``.bin``."""

    sizes: np.ndarray
    starts: np.ndarray

    @property
    def docs(self) -> int:
        return len(self.sizes)

    @property
    def sequences(self) -> int:
        """How many sequences the ``.idx`` holds: one per non-empty document."""
        return int(np.count_nonzero(self.sizes))

    @property
    def tokens(self) -> int:
        return int(self.sizes.sum())


def read_index(index_file: Path, data_file: Path, *, error: type[Exception]) -> DocumentIndex:
    """Read an ``.idx`` with Megatron's reader and check that it describes ``data_file`` as a corpus of documents.

    The index must hold int32 ids, at most one sequence per document, no empty sequence, pointers contiguous from 0,
    and a ``.bin`` exactly as long. Raises ``error`` otherwise, or when either file is missing.
    """
    from megatron.core.datasets.indexed_dataset import _IndexReader

    for path in (index_file, data_file):
        if not path.is_file():
            raise error(f"missing {path}")
    index = _IndexReader(str(index_file), multimodal=False)
    if np.dtype(index.dtype) != np.dtype(np.int32):
        raise error(f"{index_file}: ids are {np.dtype(index.dtype)}, not int32")
    lengths = index.sequence_lengths.astype(np.int64)
    boundaries = index.document_indices.astype(np.int64)
    per_document = np.diff(boundaries)
    if boundaries[0] != 0 or boundaries[-1] != len(lengths) or not np.isin(per_document, (0, 1)).all():
        raise error(f"{index_file}: not at most one sequence per document")
    empty = np.flatnonzero(lengths < 1)
    if empty.size:
        raise error(
            f"{index_file}: sequence {empty[0]} is empty; an empty document has no sequence, and every sequence ends "
            "in its EOD"
        )
    sizes = np.zeros(len(per_document), dtype=np.int64)
    sizes[per_document == 1] = lengths
    starts = np.zeros(len(sizes), dtype=np.int64)
    np.cumsum(sizes[:-1], out=starts[1:])
    if not np.array_equal(index.sequence_pointers, starts[per_document == 1] * TOKEN_DTYPE.itemsize):
        raise error(f"{index_file}: the documents are not laid end to end from the start of the .bin")
    expected = int(sizes.sum()) * TOKEN_DTYPE.itemsize
    if data_file.stat().st_size != expected:
        raise error(f"{data_file} is {data_file.stat().st_size} bytes, its index describes {expected}")
    return DocumentIndex(sizes, starts)


def token_memmap(path: Path) -> np.ndarray:
    """A ``.bin``'s ids, memory-mapped; numpy cannot map an empty file, which holds none."""
    if path.stat().st_size == 0:
        return np.zeros(0, dtype=TOKEN_DTYPE)
    return np.memmap(path, dtype=TOKEN_DTYPE, mode="r")
