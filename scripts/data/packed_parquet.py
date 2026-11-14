# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""Reading a packed SFT parquet: its list columns, its pack starts and the documents inside its packs.

``scripts/data/pack_sft_dataset.py`` writes one parquet per shard whose rows are packs. Each row holds ``input_ids``,
``loss_mask`` and ``seq_start_id``, the offset at which each packed document begins, the first at 0. A document is the
span of a pack from its start to the next document's start, and the last one's to the end of the pack.

``configs/control_pretraining/packed_corpus_checks.py`` and ``scripts/data/audit_pack_digests.py`` read packs through
this module. A shard that is not such a parquet raises ``MalformedShard``, and the callers report it as a failure.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import NamedTuple

import numpy as np


# The columns iter_documents always reads. The loss mask is read only on request, because a document's ids do not
# depend on it.
DOCUMENT_COLUMNS = ("input_ids", "seq_start_id")
MASK_COLUMN = "loss_mask"


class MalformedShard(Exception):
    """A shard's parquet is not a pack the packer writes."""


class PackedDocument(NamedTuple):
    """One document of a pack: its ids (a view into the row group's ids) and where it sits."""

    ids: np.ndarray
    pack: int  # the row of the parquet that holds the document
    document: int  # its index among the pack's documents
    start: int  # its offset in the pack's ids
    mask: np.ndarray | None = None  # its loss mask, a uint8 view with an entry per id, when iter_documents read it

    @property
    def length(self) -> int:
        """The number of ids the document spans, its padding included."""
        return len(self.ids)


def list_column(table, name: str, integer: bool) -> tuple[np.ndarray, np.ndarray]:
    """A list column's values, flat, and each pack's offsets into them. The column must hold no null and values of an
    integer type, or, with ``integer`` false, of an integer or boolean type. Raises MalformedShard."""
    import pyarrow as pa

    array = table.column(name).combine_chunks()
    if not pa.types.is_list(array.type) and not pa.types.is_large_list(array.type):
        raise MalformedShard(f"column {name} is {array.type}, not a list")
    value_type = array.type.value_type
    if not (pa.types.is_integer(value_type) or (not integer and pa.types.is_boolean(value_type))):
        raise MalformedShard(f"column {name} holds {value_type} values")
    values = array.flatten()
    if array.null_count or values.null_count:
        raise MalformedShard(f"column {name} holds a null")
    offsets = array.offsets.to_numpy()
    return values.to_numpy(zero_copy_only=False), offsets - offsets[0]


def check_pack_starts(starts: np.ndarray, length: int, pack: int) -> None:
    """Raise MalformedShard unless a pack's document starts begin at 0 and increase strictly inside its ``length``."""
    if not len(starts) or starts[0] != 0 or (np.diff(starts) <= 0).any() or starts[-1] >= length:
        raise MalformedShard(f"pack {pack}: seq_start_id {starts[:8].tolist()}... does not split its {length} ids")


def iter_documents(parquet: Path, dtype: str, with_mask: bool = False) -> Iterator[PackedDocument]:
    """Every document of a shard's parquet in pack order, read one row group at a time.

    The ids of each row group are cast once to ``dtype``, an integer type such as ``"<i4"``. A row group with an id
    outside that type's range is refused, because the cast would wrap it into another id. With ``with_mask`` each
    document also carries its loss mask, cast to uint8.

    Raises:
        MalformedShard: a column is missing or of another type, holds a null, or has pack starts that do not split
            their pack, or an id does not fit ``dtype``, or a pack's loss mask is not as long as its ids. Reading the
            file can also raise OSError or pyarrow's errors.
    """
    import pyarrow.parquet as pq

    columns = (*DOCUMENT_COLUMNS, MASK_COLUMN) if with_mask else DOCUMENT_COLUMNS
    parquet_file = pq.ParquetFile(parquet)
    names = parquet_file.schema_arrow.names
    missing = [name for name in columns if name not in names]
    if missing:
        raise MalformedShard(f"no column {missing}; it has {names}")
    limits = np.iinfo(dtype)
    first_pack = 0
    for group in range(parquet_file.metadata.num_row_groups):
        table = parquet_file.read_row_group(group, columns=list(columns))
        ids, id_offsets = list_column(table, "input_ids", integer=True)
        starts, start_offsets = list_column(table, "seq_start_id", integer=True)
        if len(ids) and (ids.min() < limits.min or ids.max() > limits.max):
            raise MalformedShard(f"row group {group}: an input id lies outside {dtype} ({ids.min()}..{ids.max()})")
        ids = np.ascontiguousarray(ids, dtype=dtype)
        mask = None
        if with_mask:
            mask, mask_offsets = list_column(table, MASK_COLUMN, integer=False)
            if not np.array_equal(np.diff(id_offsets), np.diff(mask_offsets)):
                raise MalformedShard(f"row group {group}: a pack's {MASK_COLUMN} is not as long as its input_ids")
            mask = np.ascontiguousarray(mask, dtype=np.uint8)
        for row in range(table.num_rows):
            pack = first_pack + row
            begin, end = int(id_offsets[row]), int(id_offsets[row + 1])
            own = starts[start_offsets[row] : start_offsets[row + 1]].astype(np.int64)
            check_pack_starts(own, end - begin, pack)
            bounds = np.append(own, end - begin) + begin
            for document, (a, b) in enumerate(zip(bounds[:-1].tolist(), bounds[1:].tolist())):
                yield PackedDocument(ids[a:b], pack, document, a - begin, None if mask is None else mask[a:b])
        first_pack += table.num_rows
