#!/usr/bin/env python3
# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""The packed-row parity check: does the Megatron training forward on a real packed row give each document the
per-token cross-entropy that Hugging Face gives that document alone?

OLMo-3 is trained on packed 32,768-token rows, each holding documents padded to a multiple of 8, with a 4,095-token
sliding window on three layers in four and YaRN rope on the fourth. A packed batch that runs and a context parallel size
that agrees with size 1 (``olmo3_packed_smoke.py``) do not show that the window and the per-document rope positions are
right inside a pack. This check does: for rows of the real packed data, the Megatron forward under the layout the run
will use (pipeline, interleaved pipeline and context parallelism) is compared document by document with a Hugging Face
forward on the document's own unpadded ids and nothing else in the context.

A spec (``load_spec``) states what is measured and the tolerance the comparison is judged by. The judge compares each
document alone, reports PASS, FAIL or NOT EVALUATED for the whole (``scripts/telemetry/gate_outcome.py``) and lists the
worst document. A calibration run that the spec registers with a ``fault`` (``score-megatron --calibration-run NAME``)
injects one of the faults of ``scripts/checkpoint/olmo3_faults.py`` (a swapped gate and up projection, per-head QK-norm,
no sliding window, YaRN on every layer) and is only ever judged as a negative control, by ``calibrate``. The body this
check shares with the export check (``export_check.py``) is in ``parity_check.py``.

**What is scored.** ``select`` draws the spec's rows from the packed parquet with a seeded generator and lists each
with the sha256 of the row as the training dataset reads it. Every command reads a row through the training dataset
(``create_sft_dataset``, the packed parquet dataset and its ``collate_fn``, with the tokenizer and dataset settings the
spec states), so a row has the tokens, labels, loss mask, position ids and ``cu_seqlens`` the training step receives.

**The two forwards.**

- Megatron: the checkpoint's model config at the spec's layout (``layout_model_config``), one forward-only pipeline
  schedule per group of rows, each row a microbatch, partitioned for context parallelism as the training step partitions
  it. The last stage returns the model's per-token cross-entropy for its share of the row (the model is called with
  ``labels=``), and the context-parallel shares are put back in the row's token order.
- Hugging Face: each document alone (its unpadded ids from ``cu_seqlens_unpadded``), the output head applied in fp32 and
  an unfused fp32 cross-entropy (``hf_model_outputs.fp32_head_outputs`` and ``parity_check.target_scores``, as the export
  check does it). The model has the rotary embeddings AI2 trained with (``hf_olmo3_rope.apply_reference_rope``).

Both are compared at the positions the loss mask selects, per document: the mean of HF minus Megatron in absolute value,
the mean absolute difference and its 99th percentile (``parity_check.agreement_statistics``).

**Outputs.** Each command writes into one output directory and refuses to replace a file: ``megatron_scores.npz``,
``hf_scores.npz`` and the judge's ``packed_row_parity-<first 12 of the spec's sha256>.json``. A scores file records the
spec's measurement digest (``MEASUREMENT_KEYS``), so a spec whose tolerance is set later judges scores taken before. The
scores files and the results documents also embed the spec's content, so the rows, packed parquet, tokenizer, dataset
settings and model overrides that were measured can be read back from them. The directory of a fault run holds the
fault's ``megatron_scores.npz`` and the correct run's ``hf_scores.npz`` (a link to it).

**Calibration.** A spec's ``calibration`` block pre-registers runs as the export check's does (``results_root`` and, under
``runs``, a ``directory`` and ``expect: PASS|FAIL`` each), and states each run's ``fault``: ``none`` for a correct run,
one of ``olmo3_faults.FAULTS`` for a negative control, which must expect FAIL. ``score-megatron --calibration-run NAME``
injects the fault that the spec registers for run NAME and writes into its directory. ``calibrate`` judges each run's
scores, refuses a directory that holds the scores of another fault than the one registered for it, and its verdict is PASS
only when every correct run passes and every fault run fails.

Commands (inside the container)::

    python scripts/checkpoint/packed_row_parity.py select --spec SPEC --out LISTING.yaml
    torchrun --standalone --nproc_per_node=<PP x CP> scripts/checkpoint/packed_row_parity.py score-megatron \\
        --spec SPEC --out-dir DIR [--calibration-run NAME]
    python scripts/checkpoint/packed_row_parity.py score-hf --spec SPEC --out-dir DIR
    python scripts/checkpoint/packed_row_parity.py judge --spec SPEC --out-dir DIR
    python scripts/checkpoint/packed_row_parity.py calibrate --spec SPEC --report-out REPORT.json

The judge and ``calibrate`` exit 0 on PASS, 1 on FAIL and otherwise 2.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import logging
import math
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from scripts.checkpoint import hf_model_outputs, parity_check  # noqa: E402
from scripts.checkpoint.hf_olmo3_rope import apply_reference_rope  # noqa: E402
from scripts.checkpoint.olmo3_faults import FAULTS, apply_fault_after_load, apply_fault_before_build  # noqa: E402
from scripts.checkpoint.parity_check import (  # noqa: E402
    OUTCOME_OF_STATUS,
    PENDING,
    CalibrationRun,
    CeTolerance,
    ParityCheckError,
    agreement_statistics,
    ce_measures,
    ce_tolerance,
    keys_digest,
    refuse_existing,
    require_json_representable,
    require_sha256,
    target_scores,
)
from scripts.data.file_digest import file_sha256  # noqa: E402
from scripts.mapping_keys import require_keys  # noqa: E402
from scripts.mapping_values import require_absolute_path, require_int  # noqa: E402
from scripts.telemetry.gate_outcome import FAIL, NOT_EVALUATED, PASS, exit_status  # noqa: E402


LOGGER = logging.getLogger("packed_row_parity")

RESULT_FORMAT = "packed-row-parity/1"
MEGATRON_SCORES = "megatron_scores.npz"
HF_SCORES = "hf_scores.npz"
NO_FAULT = "none"
# The seed of the ranks' RNG trackers; nothing in an eval-mode forward draws from them.
REFERENCE_SEED = 0
# The HF model is scored in bf16, as the Megatron model runs.
HF_DTYPE = "bfloat16"
# Rows of logits the cross-entropy is taken over at a time, which bounds the temporaries of a 32,768-token document.
CE_CHUNK_POSITIONS = 4096

SPEC_KEYS = frozenset(
    {
        "packed_parquet",
        "tokenizer",
        "seq_length",
        "pad_seq_to_mult",
        "dataset_kwargs",
        "selection",
        "rows",
        "hf_model",
        "megatron_checkpoint",
        "layout",
        "model_overrides",
        "criteria",
        "calibration",
    }
)
# What the Hugging Face scores depend on: the rows and how they are read. A change to any of these keys makes earlier
# scores unusable. The criteria, the calibration record and the selection (rows already listed are what is measured) do
# not change a score.
DATA_KEYS = ("packed_parquet", "tokenizer", "seq_length", "pad_seq_to_mult", "dataset_kwargs", "rows")
# What the Megatron scores depend on: the rows and the layout and model-config overrides of the forward. The Hugging Face
# scores do not depend on the layout, so one set of them serves every layout of the same rows.
MEASUREMENT_KEYS = (*DATA_KEYS, "layout", "model_overrides")
LAYOUT_KEYS = frozenset(
    {
        "tensor_model_parallel_size",
        "pipeline_model_parallel_size",
        "virtual_pipeline_model_parallel_size",
        "context_parallel_size",
    }
)
# Model-config fields that the layout sets or that move layers between ranks; ``model_overrides`` may not name them.
LAYOUT_FIELDS = frozenset(
    {
        *LAYOUT_KEYS,
        "pipeline_model_parallel_layout",
        "num_layers_per_virtual_pipeline_stage",
        "num_layers_in_first_pipeline_stage",
        "num_layers_in_last_pipeline_stage",
        "account_for_embedding_in_pipeline_split",
        "account_for_loss_in_pipeline_split",
        "microbatch_group_size_per_vp_stage",
        "pipeline_dtype",
        "sequence_parallel",
        "perform_initialization",
        "expert_model_parallel_size",
        "expert_tensor_parallel_size",
        "hybrid_context_parallel",
    }
)


class PackedRowParityError(ParityCheckError):
    """The check cannot be run as asked: a malformed spec, a row that is not the one the spec lists, or scores that were
    not taken for this spec."""


# ----------------------------------------------------------------------------------------------------------------------
# The spec
# ----------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class RowRef:
    """Row ``index`` of the packed parquet (counted over the files its glob matches, in order), whose content has the
    sha256 ``sha256`` (``row_digest``)."""

    index: int
    sha256: str


@dataclass(frozen=True)
class Selection:
    """How ``select`` draws the rows: ``count`` of them, without replacement, from a generator seeded with ``seed``."""

    seed: int
    count: int


@dataclass(frozen=True)
class Layout:
    """The parallel layout under test. Tensor parallelism is 1 (OLMo-3's full-width QK-norm refuses more) and data
    parallelism is 1, so the world size is the pipeline size times the context-parallel size."""

    pipeline_model_parallel_size: int
    virtual_pipeline_model_parallel_size: int | None
    context_parallel_size: int

    @property
    def world_size(self) -> int:
        return self.pipeline_model_parallel_size * self.context_parallel_size

    @property
    def microbatches(self) -> int:
        """The rows one pipeline schedule runs, one per microbatch: the pipeline size, which an interleaved schedule
        needs at least (and as a multiple) and a plain one runs without a bubble beyond the pipeline's own."""
        return self.pipeline_model_parallel_size


@dataclass(frozen=True)
class PackedRowSpec:
    """A spec as ``load_spec`` read it, with its path, sha256, the digests of what its Hugging Face scores (``DATA_KEYS``)
    and its Megatron scores (``MEASUREMENT_KEYS``) depend on, and its parsed content. ``rows`` is None
    while the spec still reads ``PENDING`` for it (before ``select`` has drawn them) and ``ce_agreement`` is None while
    its tolerance is ``PENDING`` calibration."""

    path: str
    sha256: str
    data_sha256: str
    measurement_sha256: str
    content: dict[str, Any]
    packed_parquet: str
    tokenizer: Path
    seq_length: int
    pad_seq_to_mult: int
    dataset_kwargs: dict[str, bool]
    selection: Selection
    rows: tuple[RowRef, ...] | None
    hf_model: Path
    megatron_checkpoint: Path
    layout: Layout
    model_overrides: dict[str, Any]
    ce_agreement: CeTolerance | None

    def scores_digest(self, side: str) -> str:
        """The digest the ``side`` (``hf`` or ``megatron``) scores of this spec depend on."""
        return self.data_sha256 if side == "hf" else self.measurement_sha256

    def listed_rows(self) -> tuple[RowRef, ...]:
        """The listed rows; raises when the spec still reads ``PENDING`` for them."""
        if self.rows is None:
            raise PackedRowParityError(f"{self.path}: rows is {PENDING}; draw them with `select` and list them first")
        return self.rows


_integer = functools.partial(require_int, error=PackedRowParityError)


def _rows(raw: Any, selection: Selection) -> tuple[RowRef, ...] | None:
    if raw == PENDING:
        return None
    if not isinstance(raw, list) or not raw:
        raise PackedRowParityError(f"rows must be {PENDING} or a non-empty list of rows, not {raw!r}")
    rows = []
    for position, item in enumerate(raw):
        where = f"rows[{position}]"
        item = require_keys(item, where, {"index", "sha256"}, error=PackedRowParityError)
        sha256 = require_sha256(item["sha256"], f"{where}.sha256", error=PackedRowParityError)
        rows.append(RowRef(index=_integer(item["index"], f"{where}.index", 0), sha256=sha256))
    if len({row.index for row in rows}) != len(rows):
        raise PackedRowParityError("rows lists a row twice")
    if len(rows) != selection.count:
        raise PackedRowParityError(f"rows lists {len(rows)} rows; selection draws {selection.count}")
    return tuple(rows)


def _layout(raw: Any) -> Layout:
    raw = require_keys(raw, "layout", LAYOUT_KEYS, error=PackedRowParityError)
    tensor = raw["tensor_model_parallel_size"]
    if isinstance(tensor, bool) or tensor != 1:
        raise PackedRowParityError(
            f"layout.tensor_model_parallel_size must be 1, not {tensor!r}: OLMo-3's full-width QK-norm refuses tensor "
            "parallelism"
        )
    pipeline = _integer(raw["pipeline_model_parallel_size"], "layout.pipeline_model_parallel_size", 1)
    virtual = raw["virtual_pipeline_model_parallel_size"]
    if virtual is not None:
        virtual = _integer(virtual, "layout.virtual_pipeline_model_parallel_size", 2)
        if pipeline == 1:
            raise PackedRowParityError("layout.virtual_pipeline_model_parallel_size needs a pipeline of 2 or more")
    return Layout(
        pipeline_model_parallel_size=pipeline,
        virtual_pipeline_model_parallel_size=virtual,
        context_parallel_size=_integer(raw["context_parallel_size"], "layout.context_parallel_size", 1),
    )


def _model_overrides(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict) or not all(isinstance(key, str) for key in raw):
        raise PackedRowParityError(f"model_overrides must map model-config fields to values: {raw!r}")
    clash = sorted(set(raw) & LAYOUT_FIELDS)
    if clash:
        raise PackedRowParityError(f"model_overrides sets the parallel layout ({clash}), which `layout` fixes")
    return dict(raw)


def _tolerance(raw: Any) -> CeTolerance | None:
    raw = require_keys(raw, "criteria", {"ce_agreement"}, error=PackedRowParityError)["ce_agreement"]
    return ce_tolerance(raw, "criteria.ce_agreement", error=PackedRowParityError)


def load_spec(path: str | Path) -> PackedRowSpec:
    """Read and check a spec. Every key is required::

        packed_parquet: /abs/shard*/packed/<tokenizer>_pad_seq_to_mult8/training_32768.idx.parquet   # a path or glob
        tokenizer: /abs/dir/of/the/tokenizer                # the tokenizer the pack was built with
        seq_length: 32768
        pad_seq_to_mult: 8
        dataset_kwargs: {answer_only_loss: true, pad_to_max_length: true}   # the training YAML's, which the collate reads
        selection: {seed: 20261104, count: 16}
        rows: PENDING | [{index: <row in the parquet>, sha256: <hex>}]
        hf_model: /abs/dir/of/the/HF/model/with/weights     # the reference
        megatron_checkpoint: /abs/.../iter_0000000          # the checkpoint under test
        layout: {tensor_model_parallel_size: 1, pipeline_model_parallel_size: 4,
                 virtual_pipeline_model_parallel_size: null, context_parallel_size: 1}
        model_overrides: {apply_rope_fusion: false, ...}    # the training recipe's settings that the checkpoint lacks
        criteria:
          ce_agreement: PENDING | {max_abs_mean_shift: ., max_mean_abs_difference: ., max_p99_abs_difference: .}
        calibration:                                        # how the tolerance was measured, as a record
          results_root: /abs/dir/of/the/runs                # optional until runs are pre-registered
          runs: {<name>: {directory: <under results_root>, expect: PASS | FAIL, fault: none | <olmo3_faults.FAULTS>}}

    Raises PackedRowParityError on a missing or unknown key, a malformed value, a value JSON cannot hold (an unquoted
    date), a ``pad_to_max_length`` other than true (the check reads every row at ``seq_length``), a listing that is not
    the selection's size, a count of rows that the pipeline size does not divide (one row per microbatch) and model
    overrides that set the parallel layout.
    """
    path = Path(path)
    content = yaml.safe_load(path.read_bytes())
    raw = require_keys(content, f"packed-row-parity spec {path}", SPEC_KEYS, error=PackedRowParityError)
    require_json_representable(content, f"packed-row-parity spec {path}", error=PackedRowParityError)
    kwargs = require_keys(
        raw["dataset_kwargs"], "dataset_kwargs", {"answer_only_loss", "pad_to_max_length"}, error=PackedRowParityError
    )
    for key, value in kwargs.items():
        if not isinstance(value, bool):
            raise PackedRowParityError(f"dataset_kwargs.{key} must be true or false, not {value!r}")
    if not kwargs["pad_to_max_length"]:
        raise PackedRowParityError(
            "dataset_kwargs.pad_to_max_length must be true: the check reads every row padded to seq_length, as the "
            "training step receives it"
        )
    selection = require_keys(raw["selection"], "selection", {"seed", "count"}, error=PackedRowParityError)
    selection = Selection(
        seed=_integer(selection["seed"], "selection.seed", 0), count=_integer(selection["count"], "selection.count", 1)
    )
    layout = _layout(raw["layout"])
    if selection.count % layout.microbatches:
        raise PackedRowParityError(
            f"selection.count {selection.count} is not a multiple of the pipeline size {layout.microbatches}: the "
            "rows run in groups of one microbatch per pipeline stage"
        )
    if not isinstance(raw["calibration"], dict):
        raise PackedRowParityError(
            f"calibration must be a mapping recording how the tolerance was set, not {raw['calibration']!r}"
        )
    return PackedRowSpec(
        path=str(path),
        sha256=file_sha256(path),
        data_sha256=keys_digest(content, DATA_KEYS),
        measurement_sha256=keys_digest(content, MEASUREMENT_KEYS),
        content=content,
        packed_parquet=require_absolute_path(raw["packed_parquet"], "packed_parquet", error=PackedRowParityError),
        tokenizer=Path(require_absolute_path(raw["tokenizer"], "tokenizer", error=PackedRowParityError)),
        seq_length=_integer(raw["seq_length"], "seq_length", 1),
        pad_seq_to_mult=_integer(raw["pad_seq_to_mult"], "pad_seq_to_mult", 1),
        dataset_kwargs=dict(kwargs),
        selection=selection,
        rows=_rows(raw["rows"], selection),
        hf_model=Path(require_absolute_path(raw["hf_model"], "hf_model", error=PackedRowParityError)),
        megatron_checkpoint=Path(
            require_absolute_path(raw["megatron_checkpoint"], "megatron_checkpoint", error=PackedRowParityError)
        ),
        layout=layout,
        model_overrides=_model_overrides(raw["model_overrides"]),
        ce_agreement=_tolerance(raw["criteria"]),
    )


# ----------------------------------------------------------------------------------------------------------------------
# Rows: drawn once (`select`), read back and checked by digest on every run
# ----------------------------------------------------------------------------------------------------------------------


def build_dataset(spec: PackedRowSpec):
    """The packed dataset as the training setup creates it: ``create_sft_dataset`` on the packed parquet spec, with the
    Megatron tokenizer of the spec's tokenizer directory and the spec's sequence length, pad multiple and dataset
    settings. Its ``__getitem__`` is one stored row and its ``collate_fn`` the training batch of a list of rows."""
    from megatron.bridge.data.datasets.sft import create_sft_dataset
    from megatron.bridge.training.tokenizers.config import TokenizerConfig
    from megatron.bridge.training.tokenizers.tokenizer import build_tokenizer

    tokenizer = build_tokenizer(
        TokenizerConfig(tokenizer_type="HuggingFaceTokenizer", tokenizer_model=str(spec.tokenizer))
    )
    return create_sft_dataset(
        spec.packed_parquet,
        tokenizer=tokenizer,
        seq_length=spec.seq_length,
        pad_seq_to_mult=spec.pad_seq_to_mult,
        **spec.dataset_kwargs,
    )


def row_digest(sample: dict[str, Any]) -> str:
    """The sha256 of a stored row as the dataset reads it (``dataset[index]``): its ids, document boundaries and loss mask,
    each as little-endian int32 behind its length."""
    digest = hashlib.sha256()
    for key in ("input_ids", "seq_boundaries", "loss_mask"):
        values = np.asarray(sample[key], dtype="<i4")
        digest.update(len(values).to_bytes(8, "little"))
        digest.update(values.tobytes())
    return digest.hexdigest()


def select_rows(spec: PackedRowSpec) -> list[dict[str, Any]]:
    """Draw the spec's rows deterministically and return the listing: ``selection.count`` distinct row indices, ascending,
    drawn without replacement by a generator seeded with ``selection.seed`` among all rows of the parquet, each with the
    row's ``row_digest``. Raises PackedRowParityError when the parquet has fewer rows."""
    dataset = build_dataset(spec)
    if len(dataset) < spec.selection.count:
        raise PackedRowParityError(
            f"{spec.packed_parquet} holds {len(dataset)} rows; {spec.selection.count} are drawn"
        )
    generator = np.random.default_rng(spec.selection.seed)
    indices = sorted(int(i) for i in generator.choice(len(dataset), size=spec.selection.count, replace=False))
    return [{"index": index, "sha256": row_digest(dataset[index])} for index in indices]


def scored_documents(
    loss_mask: np.ndarray, cu_seqlens: np.ndarray, cu_seqlens_unpadded: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray]:
    """The ``(starts, lengths)`` of the documents of one pack that have a position to score.

    Document ``i`` starts at ``cu_seqlens[i]`` (the padded boundaries, which are the pack's memory layout) and holds
    ``cu_seqlens_unpadded[i + 1] - cu_seqlens_unpadded[i]`` real positions, the rest of its span being padding (without
    ``cu_seqlens_unpadded`` the pack has none, and the span is the length). A document with no real position, such as
    the zero-length entry the collate appends for the unused tail of a pack, or no loss-masked position is left out.
    Raises PackedRowParityError when a document's positions run past its span.
    """
    unpadded = cu_seqlens if cu_seqlens_unpadded is None else cu_seqlens_unpadded
    if len(unpadded) != len(cu_seqlens):
        raise PackedRowParityError(f"cu_seqlens has {len(cu_seqlens)} entries and cu_seqlens_unpadded {len(unpadded)}")
    starts, lengths = [], []
    for index in range(len(cu_seqlens) - 1):
        start, length = int(cu_seqlens[index]), int(unpadded[index + 1] - unpadded[index])
        if start + length > int(cu_seqlens[index + 1]):
            raise PackedRowParityError(
                f"document {index} has {length} positions from {start}, past the end of its span {cu_seqlens[index + 1]}"
            )
        if length > 0 and loss_mask[start : start + length].any():
            starts.append(start)
            lengths.append(length)
    return np.array(starts, dtype=np.int32), np.array(lengths, dtype=np.int32)


@dataclass(frozen=True)
class PackedRow:
    """One listed row as the training dataset collates it: ``batch`` is the collate's output for this row alone, and the
    arrays are its positions (``tokens``, ``labels``, ``loss_mask``) and the documents to score (``starts``, ``lengths``)."""

    index: int
    batch: dict[str, Any]
    tokens: np.ndarray
    labels: np.ndarray
    loss_mask: np.ndarray
    starts: np.ndarray
    lengths: np.ndarray


def collate_row(dataset, index: int, sample: dict[str, Any], seq_length: int) -> PackedRow:
    """The training batch of stored row ``index`` (``sample``, read from ``dataset``), collated as the training step's data
    loader collates it (``dataset.collate_fn``), with its documents to score."""
    from megatron.bridge.training.utils.packed_seq_utils import trim_padded_cu_seqlens

    batch = dataset.collate_fn([sample])
    if batch["tokens"].shape != (1, seq_length):
        raise PackedRowParityError(f"a row collates to {tuple(batch['tokens'].shape)}, not (1, {seq_length})")
    cu_seqlens = trim_padded_cu_seqlens(batch["cu_seqlens"].squeeze(), batch["cu_seqlens_argmin"]).numpy()
    unpadded = batch.get("cu_seqlens_unpadded")
    if unpadded is not None:
        unpadded = trim_padded_cu_seqlens(unpadded.squeeze(), batch["cu_seqlens_unpadded_argmin"]).numpy()
    loss_mask = batch["loss_mask"][0].numpy().astype(np.int8)
    starts, lengths = scored_documents(loss_mask, cu_seqlens, unpadded)
    if not len(starts):
        raise PackedRowParityError(f"row {index} has no document with a loss-masked position to score")
    return PackedRow(
        index=index,
        batch=batch,
        tokens=batch["tokens"][0].numpy().astype(np.int32),
        labels=batch["labels"][0].numpy().astype(np.int32),
        loss_mask=loss_mask,
        starts=starts,
        lengths=lengths,
    )


def read_rows(spec: PackedRowSpec) -> list[PackedRow]:
    """The spec's listed rows, read through the training dataset; raises PackedRowParityError when a row is not the one
    the spec lists (another sha256) or the parquet has no such row."""
    dataset = build_dataset(spec)
    rows = []
    for ref in spec.listed_rows():
        if ref.index >= len(dataset):
            raise PackedRowParityError(
                f"{spec.packed_parquet} holds {len(dataset)} rows; the spec lists row {ref.index}"
            )
        sample = dataset[ref.index]
        digest = row_digest(sample)
        if digest != ref.sha256:
            raise PackedRowParityError(
                f"row {ref.index} has the sha256 {digest}, the spec lists {ref.sha256}; the parquet is not the one the "
                "spec was drawn from"
            )
        rows.append(collate_row(dataset, ref.index, sample, spec.seq_length))
    return rows


# ----------------------------------------------------------------------------------------------------------------------
# Scores files
# ----------------------------------------------------------------------------------------------------------------------


def _padded(values: Sequence[np.ndarray]) -> np.ndarray:
    """``values`` as rows of one int32 array, each padded with zeros to the longest."""
    width = max(len(row) for row in values)
    return np.array([np.pad(row, (0, width - len(row))) for row in values], dtype=np.int32)


def write_pack_scores(
    path: Path, rows: Sequence[PackedRow], cross_entropy: np.ndarray, record: dict[str, Any]
) -> None:
    """Write one side's scores: the per-token cross-entropy ``[rows, seq_length]`` (NaN where nothing was scored), the
    rows' tokens, labels, loss masks and scored documents (which the judge checks are the other side's too) and the
    record, refusing to replace a file."""
    parity_check.write_arrays(
        path,
        {
            "cross_entropy": cross_entropy.astype(np.float32),
            "tokens": np.stack([row.tokens for row in rows]),
            "labels": np.stack([row.labels for row in rows]),
            "loss_mask": np.stack([row.loss_mask for row in rows]),
            "starts": _padded([row.starts for row in rows]),
            "lengths": _padded([row.lengths for row in rows]),
            "row_indices": np.array([row.index for row in rows], dtype=np.int64),
        },
        record,
    )


# The arrays of a scores file, in the order ``write_pack_scores`` writes them: the fields of ``PackScores`` but its record.
PACK_ARRAYS = ("cross_entropy", "tokens", "labels", "loss_mask", "starts", "lengths", "row_indices")


@dataclass(frozen=True)
class PackScores:
    """A scores file as ``read_pack_scores`` read it. ``starts`` and ``lengths`` are zero-padded per row; a zero length is
    no document."""

    cross_entropy: np.ndarray  # [rows, seq_length]
    tokens: np.ndarray
    labels: np.ndarray
    loss_mask: np.ndarray
    starts: np.ndarray  # [rows, documents]
    lengths: np.ndarray
    row_indices: np.ndarray
    record: dict[str, Any]


def spec_block(spec: PackedRowSpec, measurement_sha256: str) -> dict[str, Any]:
    """The spec as an output records it: its path, sha256, the digest the output depends on and its content, so that
    the measured inputs (the rows, the packed parquet, the tokenizer, the dataset settings, the layout and the model
    overrides) can be read back from the output."""
    return parity_check.spec_record(spec, measurement_sha256) | {"content": spec.content}


def scores_record(spec: PackedRowSpec, side: str) -> dict[str, Any]:
    """What identifies a scoring run of ``side`` (``hf`` or ``megatron``): the spec with its content, the digest its
    scores depend on (``data_sha256`` for the Hugging Face side, ``measurement_sha256`` for the Megatron side), the code
    that measured and where it ran."""
    return parity_check.run_record(spec_block(spec, spec.scores_digest(side))) | {"side": side}


def read_pack_scores(path: Path, spec: PackedRowSpec, side: str) -> PackScores:
    """Read the ``side`` scores file at ``path``, refusing one taken with another measurement than ``spec``'s (the digest
    ``scores_record`` records) or of other rows."""
    arrays, record = parity_check.read_arrays(
        path,
        PACK_ARRAYS,
        measurement_sha256=spec.scores_digest(side),
        spec_path=spec.path,
        error=PackedRowParityError,
    )
    scores = PackScores(**arrays, record=record)
    listed = [row.index for row in spec.listed_rows()]
    if scores.row_indices.tolist() != listed or scores.cross_entropy.shape != (len(listed), spec.seq_length):
        raise PackedRowParityError(
            f"{path} holds rows {scores.row_indices.tolist()} of shape {scores.cross_entropy.shape}; the spec lists "
            f"{listed} at {spec.seq_length} positions"
        )
    return scores


# ----------------------------------------------------------------------------------------------------------------------
# The Megatron forward (torchrun, one rank per GPU)
# ----------------------------------------------------------------------------------------------------------------------


def layout_model_config(model_cfg: Any, spec: PackedRowSpec) -> Any:
    """``model_cfg`` (a checkpoint's model config, as ``load_model_config`` read it) set to the layout under test: the
    spec's pipeline, virtual pipeline and context-parallel sizes, no tensor parallelism or sequence parallelism, the
    activations' dtype for the pipeline's sends and the interleaved schedule's group of one microbatch per stage, then
    the spec's ``model_overrides``. Raises PackedRowParityError on a field the config does not have."""
    from megatron.bridge.training.model_load_save import torch_dtype_from_mcore_config

    layout = spec.layout
    settings: dict[str, Any] = {
        "tensor_model_parallel_size": 1,
        "pipeline_model_parallel_size": layout.pipeline_model_parallel_size,
        "virtual_pipeline_model_parallel_size": layout.virtual_pipeline_model_parallel_size,
        "pipeline_model_parallel_layout": None,
        "context_parallel_size": layout.context_parallel_size,
        "sequence_parallel": False,
        "perform_initialization": False,
        "microbatch_group_size_per_vp_stage": layout.pipeline_model_parallel_size,
        **spec.model_overrides,
    }
    if layout.pipeline_model_parallel_size > 1:
        settings["pipeline_dtype"] = torch_dtype_from_mcore_config(model_cfg)
    return parity_check.apply_model_settings(model_cfg, settings, error=PackedRowParityError)


def load_layout_model(spec: PackedRowSpec, fault: str) -> list[Any]:
    """The checkpoint built at the spec's layout on this rank's GPU, with its weights loaded, the fault applied and in eval
    mode: this rank's model chunks (several with a virtual pipeline). The process group comes first: the checkpoint's
    run_config is instantiated in it, as the exporter instantiates it."""
    import torch
    import torch.distributed as dist

    from megatron.bridge.training.model_load_save import build_and_load_model, load_model_config
    from megatron.bridge.utils.common_utils import get_local_rank_preinit

    torch.cuda.set_device(get_local_rank_preinit())
    dist.init_process_group("nccl")
    if dist.get_world_size() != spec.layout.world_size:
        raise PackedRowParityError(
            f"the layout needs {spec.layout.world_size} ranks (pipeline {spec.layout.pipeline_model_parallel_size} x "
            f"context {spec.layout.context_parallel_size}); torchrun started {dist.get_world_size()}"
        )
    model_cfg, _ = load_model_config(str(spec.megatron_checkpoint))
    layout_model_config(model_cfg, spec)
    apply_fault_before_build(model_cfg, fault)
    model_cfg.initialize_model_parallel(seed=REFERENCE_SEED)
    chunks = build_and_load_model(str(spec.megatron_checkpoint), model_cfg, skip_temp_dist_context=True)
    chunks = chunks if isinstance(chunks, list) else [chunks]
    apply_fault_after_load(chunks, fault, yarn_rotary_scaling_factor=model_cfg.yarn_rotary_scaling_factor)
    return [chunk.eval() for chunk in chunks]


PER_TOKEN_KEYS = ("tokens", "labels", "loss_mask", "position_ids")
DEVICE_PACK_KEYS = ("cu_seqlens", "cu_seqlens_unpadded")
HOST_PACK_KEYS = ("cu_seqlens_argmin", "cu_seqlens_unpadded_argmin", "max_seqlen")


def packed_item(position: int, row: PackedRow, spec: PackedRowSpec, device: Any) -> dict[str, Any]:
    """The inputs of one microbatch on this rank: the row's batch laid out as the training step lays it out (per-token
    tensors and ``cu_seqlens`` on the device, their ``argmin`` and ``max_seqlen`` on the host), partitioned for this
    rank's share under context parallelism by the training step's own partition, with the ``PackedSeqParams`` of the
    whole row."""
    from megatron.core import parallel_state

    from megatron.bridge.training.gpt_step import _partition_packed_batch_for_cp
    from megatron.bridge.training.utils.packed_seq_utils import get_packed_seq_params

    batch = {key: row.batch[key].to(device) for key in (*PER_TOKEN_KEYS, *DEVICE_PACK_KEYS) if key in row.batch}
    batch.update({key: row.batch[key] for key in HOST_PACK_KEYS if key in row.batch})
    cp_size = spec.layout.context_parallel_size
    if cp_size > 1:
        batch = _partition_packed_batch_for_cp(batch, cp_size, parallel_state.get_context_parallel_rank())
    packed_keys = (*DEVICE_PACK_KEYS, *HOST_PACK_KEYS)
    return {
        "position": position,
        "tokens": batch["tokens"],
        "labels": batch["labels"],
        "position_ids": batch["position_ids"],
        "packed_seq_params": get_packed_seq_params(
            {key: batch[key] for key in packed_keys if key in batch}, total_tokens=spec.seq_length
        ),
    }


def forward_step(data_iterator, model):
    """One microbatch's forward for the pipeline schedule: the model called with ``labels=``, so that its last stage
    returns the per-token cross-entropy. The loss function hands the schedule ``(row position, output)`` untouched."""
    item = next(data_iterator)
    output = model(
        input_ids=item["tokens"],
        position_ids=item["position_ids"],
        attention_mask=None,
        labels=item["labels"],
        packed_seq_params=item["packed_seq_params"],
    )

    def collect(output_tensor, non_loss_data=False):
        return item["position"], output_tensor

    return output, collect


def forward_rows(chunks: list[Any], rows: list[PackedRow], spec: PackedRowSpec) -> dict[int, Any]:
    """Run the rows through the model in groups of ``layout.microbatches``, one forward-only pipeline schedule per group.
    Returns this rank's per-token cross-entropy of each row by position in ``rows``, ``[local positions]`` float32 on the
    device: every row on the last pipeline stage's ranks and none on the others."""
    import torch
    from megatron.core.pipeline_parallel import get_forward_backward_func

    forward_backward = get_forward_backward_func()
    device = torch.cuda.current_device()
    group = spec.layout.microbatches
    shares: dict[int, Any] = {}
    with torch.no_grad():
        for first in range(0, len(rows), group):
            items = [packed_item(position, rows[position], spec, device) for position in range(first, first + group)]
            collected = forward_backward(
                forward_step_func=forward_step,
                data_iterator=[iter(items) for _ in chunks],
                model=chunks,
                num_microbatches=group,
                seq_length=spec.seq_length,
                micro_batch_size=1,
                forward_only=True,
                collect_non_loss_data=True,
            )
            for position, output in collected:
                shares[position] = output[0].float()
    return shares


def reassemble_context_parallel(shares: Sequence[Any], packed_seq_params: Any) -> Any:
    """A row's per-token values in the row's token order from the context-parallel ranks' shares, given in rank order:
    the inverse of the training step's partition (Transformer Engine's ``thd_get_partitioned_indices`` on the padded
    ``cu_seqlens``), as the Mamba layers apply it (``_undo_attention_load_balancing``). The shares must lie on the GPU."""
    import torch
    from megatron.core.ssm.mamba_context_parallel import _undo_attention_load_balancing

    if len(shares) == 1:
        return shares[0]
    return _undo_attention_load_balancing(torch.cat(list(shares)), len(shares), packed_seq_params)


def row_packed_seq_params(row: PackedRow, spec: PackedRowSpec, device: Any) -> Any:
    """The ``PackedSeqParams`` of the whole row on ``device`` (before any context-parallel partition)."""
    from megatron.bridge.training.utils.packed_seq_utils import get_packed_seq_params

    keys = (*DEVICE_PACK_KEYS, *HOST_PACK_KEYS)
    tensors = {
        key: (row.batch[key].to(device) if key in DEVICE_PACK_KEYS else row.batch[key])
        for key in keys
        if key in row.batch
    }
    return get_packed_seq_params(tensors, total_tokens=spec.seq_length)


def gather_rows(shares: dict[int, Any], rows: list[PackedRow], spec: PackedRowSpec) -> np.ndarray:
    """The per-token cross-entropy ``[rows, seq_length]`` in each row's token order, on every rank of the last pipeline
    stage: the context-parallel ranks' shares gathered over the context-parallel group and reassembled."""
    import torch
    import torch.distributed as dist
    from megatron.core import parallel_state

    if sorted(shares) != list(range(len(rows))):
        raise PackedRowParityError(f"the last stage collected rows {sorted(shares)}, not each of range({len(rows)})")
    cp_size = spec.layout.context_parallel_size
    device = torch.cuda.current_device()
    full = np.zeros((len(rows), spec.seq_length), dtype=np.float32)
    for position, row in enumerate(rows):
        local = shares[position].contiguous()
        gathered = [local]
        if cp_size > 1:
            gathered = [torch.empty_like(local) for _ in range(cp_size)]
            dist.all_gather(gathered, local, group=parallel_state.get_context_parallel_group())
        full[position] = reassemble_context_parallel(gathered, row_packed_seq_params(row, spec, device)).cpu().numpy()
    return full


def score_megatron(spec: PackedRowSpec, out_dir: Path, fault: str) -> None:
    """Score the spec's rows with the training forward of the checkpoint at the spec's layout and write
    ``megatron_scores.npz``.

    Run under torchrun, one rank per GPU (``main`` wraps it in the bridge's ``torchrun_main``, which ends every rank when
    one raises). The ranks of the last pipeline stage hold each row's cross-entropy, one share per context-parallel rank;
    the first of them writes the scores. ``fault`` is one of ``olmo3_faults.FAULTS`` and is recorded: the scores of a
    fault are a negative control.
    """
    import torch
    import torch.distributed as dist
    from megatron.core import parallel_state

    output = out_dir / MEGATRON_SCORES
    refuse_existing(output)
    rows = read_rows(spec)
    started = time.monotonic()
    chunks = load_layout_model(spec, fault)
    loaded = time.monotonic()
    shares = forward_rows(chunks, rows, spec)
    scored = time.monotonic()
    is_last_stage = parallel_state.get_pipeline_model_parallel_rank() == spec.layout.pipeline_model_parallel_size - 1
    if is_last_stage:
        cross_entropy = gather_rows(shares, rows, spec)
        if parallel_state.get_context_parallel_rank() == 0:
            record = scores_record(spec, "megatron") | {
                "fault": fault,
                "negative_control": fault != NO_FAULT,
                "checkpoint": str(spec.megatron_checkpoint.resolve()),
                "world_size": dist.get_world_size(),
                "torch": torch.__version__,
                "load_seconds": round(loaded - started, 1),
                "score_seconds": round(scored - loaded, 1),
            }
            write_pack_scores(output, rows, cross_entropy, record)
            LOGGER.info("wrote %s", output)
    dist.barrier()


# ----------------------------------------------------------------------------------------------------------------------
# The Hugging Face forward (one GPU)
# ----------------------------------------------------------------------------------------------------------------------


def document_cross_entropy(model, ids: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """The fp32 cross-entropy at each position of one document alone: ``ids`` is the document's input, ``targets`` the
    token each position predicts. The logits are the model's output head applied in fp32 (``fp32_head_outputs``) and the
    cross-entropy is unfused (``parity_check.target_scores``), taken over ``CE_CHUNK_POSITIONS`` rows at a time."""
    import torch

    logits = hf_model_outputs.fp32_head_outputs(model, ids.tolist())
    labels = torch.as_tensor(targets.astype(np.int64), device=logits.device)
    chunks = []
    for first in range(0, len(ids), CE_CHUNK_POSITIONS):
        part = slice(first, first + CE_CHUNK_POSITIONS)
        # The second value is log p of one token, unused here: the id is any one the vocabulary has.
        cross_entropy, _ = target_scores(logits[part], labels[part], 0)
        chunks.append(cross_entropy.float().cpu().numpy())
    return np.concatenate(chunks)


def score_hf(spec: PackedRowSpec, out_dir: Path) -> None:
    """Score the spec's rows with the Hugging Face model, one document at a time with nothing else in its context, and
    write ``hf_scores.npz``. The model is made the reference (``apply_reference_rope``): sliding layers with plain rope,
    full-attention layers with YaRN, whichever transformers is installed; any other structure raises."""
    import torch
    import transformers

    output = out_dir / HF_SCORES
    refuse_existing(output)
    rows = read_rows(spec)
    started = time.monotonic()
    model = hf_model_outputs.load_model(str(spec.hf_model), HF_DTYPE)
    rope = apply_reference_rope(model)
    loaded = time.monotonic()
    cross_entropy = np.full((len(rows), spec.seq_length), np.nan, dtype=np.float32)
    for position, row in enumerate(rows):
        for start, length in zip(row.starts.tolist(), row.lengths.tolist()):
            part = slice(start, start + length)
            cross_entropy[position, part] = document_cross_entropy(model, row.tokens[part], row.labels[part])
    record = (
        scores_record(spec, "hf")
        | parity_check.hf_model_record(spec.hf_model, model)
        | {
            "rope": rope,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "load_seconds": round(loaded - started, 1),
            "score_seconds": round(time.monotonic() - loaded, 1),
        }
    )
    write_pack_scores(output, rows, cross_entropy, record)
    LOGGER.info("wrote %s", output)


# ----------------------------------------------------------------------------------------------------------------------
# The judge
# ----------------------------------------------------------------------------------------------------------------------


def result_path(out_dir: Path, digest: str) -> Path:
    """Where the judge of the spec with ``digest`` writes: one file per spec, so a spec calibrated later judges the same
    scores anew."""
    return out_dir / f"packed_row_parity-{digest[:12]}.json"


def document_agreements(spec: PackedRowSpec, hf: PackScores, megatron: PackScores) -> list[dict[str, Any]]:
    """For every scored document, HF against Megatron at its loss-masked positions (``agreement_statistics``): where it
    lies, its retained targets and the three measures the tolerance bounds. Raises PackedRowParityError when the two
    sides did not score the same rows, tokens and documents, or a loss-masked position lies outside every document."""
    for name in ("tokens", "labels", "loss_mask", "starts", "lengths", "row_indices"):
        if not np.array_equal(getattr(hf, name), getattr(megatron, name)):
            raise PackedRowParityError(f"the HF and Megatron scores are of different {name}")
    documents = []
    for position, ref in enumerate(spec.listed_rows()):
        retained = hf.loss_mask[position].astype(bool)
        inside = np.zeros_like(retained)
        for start, length in zip(hf.starts[position].tolist(), hf.lengths[position].tolist()):
            if length == 0:
                continue
            part = slice(start, start + length)
            inside[part] = True
            statistics = agreement_statistics(
                hf.cross_entropy[position, part], megatron.cross_entropy[position, part], retained[part]
            )
            documents.append(
                {
                    "row": ref.index,
                    "start": start,
                    "tokens": length,
                    "retained_targets": statistics["retained_targets"],
                    **ce_measures(statistics),
                    "max_abs_difference": statistics["abs_difference"]["max"],
                }
            )
        if (retained & ~inside).any():
            raise PackedRowParityError(
                f"row {ref.index}: {int((retained & ~inside).sum())} loss-masked positions lie outside every scored document"
            )
    return documents


def _excess(document: dict[str, Any], limits: dict[str, float] | None) -> float:
    """How far a document lies beyond the tolerance: its largest measure over that measure's limit (over the mean
    absolute difference alone while there is no tolerance). NaN counts as infinitely far."""
    values = (
        [document["mean_abs_difference"]]
        if limits is None
        else [document[key] / limit for key, limit in limits.items()]
    )
    return math.inf if any(math.isnan(value) for value in values) else max(values)


def _worst(values: list[float]) -> float:
    """The largest of ``values``, or NaN when any of them is: ``max`` alone gives an answer that depends on the order."""
    return math.nan if any(math.isnan(value) for value in values) else max(values)


def evaluate_scores(spec: PackedRowSpec, hf: PackScores, megatron: PackScores) -> dict[str, Any]:
    """The results document for HF scores against Megatron scores: every document judged on its own against the spec's
    tolerance (NOT EVALUATED while it is ``PENDING``). A document is beyond the tolerance when any of its three measures is
    (or is NaN); the outcome is FAIL when any document is, and the worst document is listed."""
    documents = document_agreements(spec, hf, megatron)
    if not documents:
        raise PackedRowParityError("no document has a loss-masked position to score")
    limits = None if spec.ce_agreement is None else spec.ce_agreement.limits()
    for document in documents:
        document["beyond"] = {} if spec.ce_agreement is None else spec.ce_agreement.beyond(document)
    worst = max(documents, key=lambda document: _excess(document, limits))
    retained = np.concatenate([row.astype(bool) for row in hf.loss_mask])
    overall = agreement_statistics(hf.cross_entropy.ravel(), megatron.cross_entropy.ravel(), retained)
    rows = []
    for ref in spec.listed_rows():
        mine = [document for document in documents if document["row"] == ref.index]
        rows.append(
            {
                "row": ref.index,
                "documents": len(mine),
                "retained_targets": sum(document["retained_targets"] for document in mine),
                "documents_beyond": sum(1 for document in mine if document["beyond"]),
                "worst_document_start": max(mine, key=lambda document: _excess(document, limits))["start"],
            }
        )
    detail = {
        "limits": limits,
        "documents_beyond": sum(1 for document in documents if document["beyond"]),
        "worst_measured": {
            key: _worst([document[key] for document in documents])
            for key in ("abs_mean_shift", "mean_abs_difference", "p99_abs_difference")
        },
        "worst_document": worst,
        "overall": overall,
        "coverage": {
            "documents": len(documents),
            "retained_targets": overall["retained_targets"],
            "longest_document_tokens": max(document["tokens"] for document in documents),
        },
        "per_row": rows,
        "per_document": documents,
    }
    if limits is None:
        return {"outcome": NOT_EVALUATED, "reason": f"criteria.ce_agreement is {PENDING} calibration", **detail}
    return {"outcome": PASS if not detail["documents_beyond"] else FAIL, **detail}


def result_document(spec: PackedRowSpec, hf: PackScores, megatron: PackScores) -> dict[str, Any]:
    """The judge's results document for ``evaluate_scores``'s agreement and the record of the scores it judged."""
    for recorded, what, given in (
        (hf.record["export"], "HF model", spec.hf_model),
        (megatron.record["checkpoint"], "checkpoint", spec.megatron_checkpoint),
    ):
        if Path(recorded) != given.resolve():
            raise PackedRowParityError(f"the scores are of the {what} {recorded}, not {given.resolve()}")
    agreement = evaluate_scores(spec, hf, megatron)
    status = exit_status([agreement["outcome"]])
    return {
        "format": RESULT_FORMAT,
        "verdict": OUTCOME_OF_STATUS[status],
        "exit_status": status,
        "fault": megatron.record["fault"],
        "negative_control": megatron.record["negative_control"],
        "hf_model": str(spec.hf_model.resolve()),
        "checkpoint": str(spec.megatron_checkpoint.resolve()),
        "spec": spec_block(spec, spec.measurement_sha256),
        **parity_check.provenance(),
        "scores": {"hf": hf.record, "megatron": megatron.record},
        "ce_agreement": agreement,
    }


def judge(spec: PackedRowSpec, out_dir: Path) -> dict[str, Any]:
    """Judge the scores in ``out_dir`` for the model they say they are of: the results document. Raises
    PackedRowParityError for the scores of an injected fault, which only ``calibrate`` judges, as a negative control."""
    hf = read_pack_scores(out_dir / HF_SCORES, spec, "hf")
    megatron = read_pack_scores(out_dir / MEGATRON_SCORES, spec, "megatron")
    if megatron.record["fault"] != NO_FAULT:
        raise PackedRowParityError(
            f"{out_dir / MEGATRON_SCORES} holds the scores of the injected fault {megatron.record['fault']!r}, a "
            "negative control that only `calibrate` judges"
        )
    return result_document(spec, hf, megatron)


# ----------------------------------------------------------------------------------------------------------------------
# Calibration
# ----------------------------------------------------------------------------------------------------------------------


def judged_result(spec: PackedRowSpec, out_dir: Path) -> dict[str, Any]:
    """The result of ``spec``'s judge on the scores in ``out_dir``: the one it wrote there before, or the one it writes
    now. Unlike ``judge`` it takes the scores of a fault, which it records as a negative control."""
    return parity_check.judged_result(
        result_path(out_dir, spec.sha256),
        lambda: result_document(
            spec,
            read_pack_scores(out_dir / HF_SCORES, spec, "hf"),
            read_pack_scores(out_dir / MEGATRON_SCORES, spec, "megatron"),
        ),
    )


def registered_fault(run: CalibrationRun) -> str:
    """The fault that the spec registers for calibration run ``run``: ``none`` for a correct run, otherwise one of
    ``olmo3_faults.FAULTS``, in which case the run is a negative control and must expect FAIL. Raises
    PackedRowParityError for a run that states no fault or an unknown one, or a fault that is expected to pass."""
    fault = run.record.get("fault")
    if fault not in FAULTS:
        raise PackedRowParityError(f"calibration.runs.{run.name}.fault must be one of {list(FAULTS)}, not {fault!r}")
    if fault != NO_FAULT and run.expect != FAIL:
        raise PackedRowParityError(
            f"calibration run {run.name} registers the fault {fault!r} and expects {run.expect}: a fault is a negative "
            f"control and must expect {FAIL}"
        )
    return fault


def calibration_fault(spec: PackedRowSpec, name: str, out_dir: Path) -> str:
    """The fault to inject for ``score-megatron --calibration-run NAME``: the one the spec registers for run ``name``
    (``registered_fault``). Raises PackedRowParityError when the spec registers no run of that name with an expectation,
    or ``out_dir`` is not the directory the spec registers for it, which ``calibrate`` reads."""
    runs = {run.name: run for run in parity_check.calibration_runs(spec, error=PackedRowParityError)}
    if name not in runs:
        raise PackedRowParityError(f"{spec.path} registers no calibration run {name!r}; it registers {sorted(runs)}")
    if out_dir.resolve() != runs[name].out_dir.resolve():
        raise PackedRowParityError(
            f"calibration run {name} is registered at {runs[name].out_dir}, but --out-dir is {out_dir}"
        )
    return registered_fault(runs[name])


def calibrate(spec: PackedRowSpec) -> dict[str, Any]:
    """Judge every run ``spec``'s calibration pre-registers (``parity_check.calibration_runs``) and compare its
    ``ce_agreement`` outcome with the expected one: the calibration report, whose verdict is PASS only when every correct
    run passed and every fault run failed. Raises PackedRowParityError when the tolerance is still PENDING, when a run
    registers a fault it is not expected to fail, or when a run's directory holds the scores of another fault than the one
    registered for it."""

    def judge_run(run: CalibrationRun) -> dict[str, Any]:
        fault = registered_fault(run)
        result = judged_result(spec, run.out_dir)
        if result["fault"] != fault:
            raise PackedRowParityError(
                f"calibration run {run.name} registers the fault {fault!r}, but {run.out_dir} holds the scores of "
                f"{result['fault']!r}"
            )
        agreement = result["ce_agreement"]
        return {
            "fault": fault,
            "ce_agreement": agreement["outcome"],
            "worst_measured": agreement["worst_measured"],
            "documents_beyond": agreement["documents_beyond"],
        }

    return parity_check.calibrate(
        spec, spec_block(spec, spec.measurement_sha256), judge_run, error=PackedRowParityError
    )


# ----------------------------------------------------------------------------------------------------------------------
# Command line
# ----------------------------------------------------------------------------------------------------------------------


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    """The command line: one of ``select``, ``score-megatron``, ``score-hf``, ``judge`` or ``calibrate`` with its options."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    select = commands.add_parser("select", help="draw the spec's rows and write their listing")
    select.add_argument("--spec", required=True, type=Path)
    select.add_argument("--out", required=True, type=Path, help="the YAML listing to write (refused if it exists)")
    calibrate_command = commands.add_parser(
        "calibrate", help="judge the runs the spec's calibration pre-registers against their expected outcomes"
    )
    calibrate_command.add_argument("--spec", required=True, type=Path)
    calibrate_command.add_argument(
        "--report-out", required=True, type=Path, help="the JSON report to write (refused if it exists)"
    )
    for name, help_text in (
        ("score-megatron", "the training forward of the checkpoint at the spec's layout (under torchrun)"),
        ("score-hf", "the Hugging Face model's per-document scores"),
        ("judge", "judge the scores in the output directory"),
    ):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--spec", required=True, type=Path)
        command.add_argument("--out-dir", required=True, type=Path)
        if name == "score-megatron":
            command.add_argument(
                "--calibration-run",
                default=None,
                help="a calibration run the spec registers: inject its registered fault and write into its directory "
                "(without it, the scores are of the correct model)",
            )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run one command; the judge's exit status is its verdict's (0 PASS, 1 FAIL, 2 NOT EVALUATED)."""
    logging.basicConfig(level=logging.INFO, format="[packed-row-parity] %(message)s")
    args = parse_args(argv)
    spec = load_spec(args.spec)
    if args.command == "select":
        refuse_existing(args.out)
        listing = {"rows": select_rows(spec)}
        args.out.write_text(yaml.safe_dump(listing, sort_keys=False, default_flow_style=None, width=200))
        LOGGER.info("wrote %d rows to %s", len(listing["rows"]), args.out)
        return 0
    if args.command == "score-megatron":
        from megatron.bridge.models.decorators import torchrun_main

        fault = (
            NO_FAULT if args.calibration_run is None else calibration_fault(spec, args.calibration_run, args.out_dir)
        )
        torchrun_main(score_megatron)(spec, args.out_dir, fault)
        return 0
    if args.command == "score-hf":
        score_hf(spec, args.out_dir)
        return 0
    if args.command == "calibrate":
        refuse_existing(args.report_out)
        report = calibrate(spec)
        parity_check.write_report(args.report_out, report)
        for name, run in report["runs"].items():
            LOGGER.info(
                "%-40s expect %-4s ce_agreement %-4s %s", name, run["expect"], run["ce_agreement"], run["outcome"]
            )
        LOGGER.info("calibration %s; wrote %s", report["verdict"], args.report_out)
        return report["exit_status"]
    output = result_path(args.out_dir, spec.sha256)
    refuse_existing(output)
    result = judge(spec, args.out_dir)
    parity_check.write_report(output, result)
    agreement = result["ce_agreement"]
    worst = agreement["worst_document"]
    LOGGER.info(
        "documents %d, beyond the tolerance %d; worst: row %d, document at %d (%d tokens), mean |diff| %.4g, p99 %.4g",
        agreement["coverage"]["documents"],
        agreement["documents_beyond"],
        worst["row"],
        worst["start"],
        worst["tokens"],
        worst["mean_abs_difference"],
        worst["p99_abs_difference"],
    )
    LOGGER.info("verdict %s; wrote %s", result["verdict"], output)
    return result["exit_status"]


if __name__ == "__main__":
    sys.exit(main())
