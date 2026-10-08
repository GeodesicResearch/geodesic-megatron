# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""The sources a training run reads, sampled and scanned for listed token ids, one source at a time.

A run reads its training data from one or more sources: each prefix of a Megatron ``.bin/.idx`` blend, or the
packed-SFT parquet spec (one file, a glob or a directory of shards). ``training_data_sources`` lists them from the
dataset config the way the dataset builders resolve them, and ``scan_sources`` reads each source directly, never
through the training datasets: it draws random documents to show, and scans contiguous stretches of the data for a
set of listed token ids, counting where they occur and whether those positions carry loss in training.

Whether a target carries loss is decided as the training loader decides it, before any token masking:

- ``.bin/.idx`` (``GPTDataset``): every token is a target and carries loss, except, with ``eod_mask_loss``, a token
  whose input is the end-of-document token. GPTDataset concatenates documents in a shuffled order; the scan takes
  the input of a document's first token from the corpus order, which for a corpus built with an EOD after every
  document is that same EOD. The whole corpus is sampled, including any part a ``split`` holds out. GPTDataset also
  takes the loss off a target equal to the pad id it settles on (the tokenizer's, unless that collides with another
  special token); the scan does not, so it counts such a target as carrying loss.
- packed parquet (``GPTSFTPackedDataset``): a conversation's first token is not a target; every later token carries
  loss when ``packed_sequence_loss_mask`` (the rule the dataset's collate uses) passes its input and that input is
  not EOS, which the collate excludes on its own.
"""

from __future__ import annotations

import bisect
import json
import logging
import re
import time
import zlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal, Sequence

import numpy as np

from megatron.bridge.data.datasets.packed_parquet import resolve_packed_parquet_paths
from megatron.bridge.data.datasets.sft import ANSWER_ONLY_LOSS_DEFAULT, packed_sequence_loss_mask
from megatron.bridge.data.utils import finetuning_dataset_builder
from megatron.bridge.training.config import FinetuningDatasetConfig, GPTDatasetConfig


logger = logging.getLogger(__name__)

SourceKind = Literal["indexed", "packed_parquet"]
StopReason = Literal["exhausted", "token_budget", "time_budget"]

# A source's token budget is spent over at least this many contiguous runs from random offsets.
_RUNS_PER_TOKEN_BUDGET = 16
# A corpus is cut into at most this many runs, which bounds the run-order permutation.
_MAX_RUNS_PER_CORPUS = 1 << 20
_PACK_DIRECTORY = re.compile(r"^(?P<tokenizer>.+)_pad_seq_to_mult\d+$")


@dataclass(frozen=True)
class DataSource:
    """One source of a run's training data."""

    index: int
    """Position of the source in the blend (0 for a single source)."""
    label: str
    """The path less the leading directories and trailing parts (such as the file name) every source shares, e.g.
    ``climbmix_full/shard0``, so distinct paths get distinct labels. A single ``.bin/.idx`` source is labelled by its
    parent directory, a packed source by the directory its shards belong to: the nearest one that is neither a glob
    nor the packer's ``packed/<tokenizer>_pad_seq_to_mult<k>``."""
    path: str
    """The ``.bin/.idx`` prefix, or the packed parquet spec (file, glob or directory) the builder reads."""
    kind: SourceKind
    weight: float | None
    """The normalised blend weight; None when the blend gives no weights."""
    tokenizer_recorded: str | None
    """The tokenizer that built the data, where the data records it: ``<prefix>.provenance.json``
    ``parameters.tokenizer``, else the parent directory's ``pipeline_results.json`` ``tokenizer``; for packed data
    the ``<org>--<name>`` slug of its ``<slug>_pad_seq_to_mult<k>`` directory. None when nothing records it."""


@dataclass(frozen=True)
class ScanSettings:
    """What ``scan_sources`` needs from the dataset config besides the sources."""

    seed: int
    """The dataset's seed, from which each source's private random generator is derived."""
    eod_mask_loss: bool | None
    """``.bin/.idx`` data: whether a target whose input is EOD carries no loss. None for packed data."""
    answer_only_loss: bool | None
    """Packed data: whether the packer's stored loss mask decides which targets carry loss. None otherwise."""


@dataclass(frozen=True, eq=False)
class SourceDocument:
    """One document (a ``.bin/.idx`` document, or one conversation of a pack) and which of its targets carry loss."""

    source_index: int
    reference: str
    """Where the document is, e.g. ``document 1234`` or ``shard-00003.parquet row 17 conversation 2`` (the shard
    named by what distinguishes its path from the source's other shards)."""
    token_ids: np.ndarray
    """The document's tokens, int64."""
    trainable_target: np.ndarray
    """Bool, one per token: whether predicting that token carries loss before token masking."""
    first_token_is_target: bool
    """Whether the document's first token is predicted at all (a conversation's first token is not)."""


@dataclass(frozen=True)
class SplitForm:
    """How a token's text appears in data tokenized by a tokenizer that does not register the token.

    Such a tokenizer encodes the text as several ordinary tokens. ``core`` is matched exactly. When ``before`` or
    ``after`` is given, the token just before the core must be one of ``before`` and the token just after it one of
    ``after``: byte-level BPE merges the text's first and last pieces with the text around it (``<`` becomes `` <``
    after a space, ``>`` becomes ``>\\n`` before a newline), so the edges are matched as sets of tokens rather than as
    single ids. Without them ``core`` is the whole split form.
    """

    core: tuple[int, ...]
    before: frozenset[int] | None = None
    after: frozenset[int] | None = None


@dataclass(frozen=True, eq=False)
class SourceScan:
    """What scanning one source found."""

    source: DataSource
    documents: tuple[SourceDocument, ...]
    """Up to ``documents_per_source`` documents drawn at random."""
    listed_documents: tuple[SourceDocument, ...]
    """Up to ``listed_documents_per_source`` scanned documents that contain a listed id."""
    documents_scanned: int
    tokens_scanned: int
    listed_targets: int
    """Occurrences of listed ids at target positions of the scanned tokens."""
    listed_trainable_targets: int
    """Of ``listed_targets``, those whose prediction carries loss before token masking."""
    split_form_occurrences: int
    """Occurrences of any listed token's multi-token split form, the signature of data built by a tokenizer that
    does not know the token."""
    out_of_vocab_tokens: int
    """Scanned tokens at or above the vocabulary size."""
    stop_reason: StopReason
    seconds: float


def training_data_sources(dataset_config: Any, tokenizer: Any) -> tuple[list[DataSource], str | None]:
    """The sources a run's training split reads, in blend order.

    A ``GPTDatasetConfig`` (finalized) gives one indexed source per prefix of its training blend; a fine-tuning
    config with packed sequences gives its packed parquet spec, resolved by the dataset builder itself (an omitted
    ``packed_train_data_path`` is the builder's default pack path, which the builder creates if it is missing).

    Args:
        dataset_config: The run's dataset config.
        tokenizer: The run's tokenizer, which the builder's default pack path is named after.

    Returns:
        ``(sources, reason)``: ``reason`` is None when there are sources and otherwise says why there are none.
    """
    if isinstance(dataset_config, FinetuningDatasetConfig):
        return _packed_sources(dataset_config, tokenizer)
    if isinstance(dataset_config, GPTDatasetConfig):
        return _indexed_sources(dataset_config)
    return [], f"the scan does not read {type(dataset_config).__name__} datasets"


def scan_settings(dataset_config: Any) -> ScanSettings:
    """The seed and loss-mask settings ``scan_sources`` needs, read from the dataset config.

    Raises:
        TypeError: For a dataset config that is neither a ``GPTDatasetConfig`` nor a ``FinetuningDatasetConfig``.
    """
    if isinstance(dataset_config, FinetuningDatasetConfig):
        answer_only_loss = (dataset_config.dataset_kwargs or {}).get("answer_only_loss", ANSWER_ONLY_LOSS_DEFAULT)
        return ScanSettings(seed=dataset_config.seed, eod_mask_loss=None, answer_only_loss=answer_only_loss)
    if isinstance(dataset_config, GPTDatasetConfig):
        return ScanSettings(
            seed=dataset_config.random_seed, eod_mask_loss=dataset_config.eod_mask_loss, answer_only_loss=None
        )
    raise TypeError(f"no scan settings for {type(dataset_config).__name__} datasets")


def scan_sources(
    sources: Sequence[DataSource],
    *,
    listed_token_ids: Sequence[int],
    split_forms: Sequence[SplitForm],
    vocab_size: int,
    documents_per_source: int,
    listed_documents_per_source: int,
    max_scan_tokens_per_source: int,
    deadline: float,
    seed: int,
    eod_token_id: int | None,
    eod_mask_loss: bool | None,
    answer_only_loss: bool | None,
    eos_token_id: int | None,
) -> list[SourceScan]:
    """Draw random documents from every source and scan each for listed token ids, within budgets.

    Each source first gets its random documents, then the sources are scanned in turns, one contiguous run each,
    so a time budget spreads over all of them. A ``.bin/.idx`` run is a stretch of whole documents starting at a
    random offset; a packed run is one row group, the row groups taken in a random order (a source's random
    documents come from its first row groups, which count as scanned). A source stops when it is exhausted, when it
    has scanned ``max_scan_tokens_per_source`` tokens (a packed source may pass it by up to one row group), or once
    ``time.monotonic()`` passes ``deadline``. The documents drawn and the scan order depend only on ``seed`` and the
    source's path; how far a scan gets before the deadline also depends on the clock.

    Args:
        sources: The sources, from ``training_data_sources``.
        listed_token_ids: The token ids to look for.
        split_forms: How a listed token's text appears in data tokenized by a tokenizer that does not know it; a
            form without edge sets has at least two ids, one with them at least one.
        vocab_size: The tokenizer's vocabulary size; tokens at or above it are counted as out of vocabulary.
        documents_per_source: Random documents to draw from each source.
        listed_documents_per_source: Documents containing a listed id to keep from each source.
        max_scan_tokens_per_source: Tokens to scan per source.
        deadline: A ``time.monotonic()`` value after which no further run starts.
        seed: The dataset's seed; each source's generator is derived from it and the source's path.
        eod_token_id: The end-of-document id; needed for ``.bin/.idx`` sources when ``eod_mask_loss`` is set.
        eod_mask_loss: ``.bin/.idx`` sources: whether a target whose input is EOD carries no loss.
        answer_only_loss: Packed sources: whether the packer's stored loss mask decides which targets carry loss.
        eos_token_id: Packed sources: the EOS id, whose inputs never carry loss.

    Returns:
        One ``SourceScan`` per source, in the order given.
    """
    _check_scan_arguments(
        sources,
        split_forms=split_forms,
        vocab_size=vocab_size,
        counts={
            "documents_per_source": documents_per_source,
            "listed_documents_per_source": listed_documents_per_source,
            "max_scan_tokens_per_source": max_scan_tokens_per_source,
        },
        eod_token_id=eod_token_id,
        eod_mask_loss=eod_mask_loss,
        answer_only_loss=answer_only_loss,
        eos_token_id=eos_token_id,
    )
    search = _Search(
        listed_token_ids=np.asarray(sorted(set(listed_token_ids)), dtype=np.int64),
        split_forms=[_SplitFormSearch.of(form) for form in split_forms],
        vocab_size=vocab_size,
        listed_documents_per_source=listed_documents_per_source,
    )
    scanners: list[_Scanner] = []
    try:
        for source in sources:
            started = time.monotonic()
            rng = np.random.default_rng(np.random.SeedSequence([seed, zlib.crc32(source.path.encode())]))
            if source.kind == "indexed":
                scanner = _IndexedScanner(
                    source,
                    rng,
                    search,
                    max_scan_tokens_per_source,
                    eod_token_id=eod_token_id,
                    eod_mask_loss=eod_mask_loss,
                )
            else:
                scanner = _PackedScanner(
                    source, rng, search, answer_only_loss=answer_only_loss, eos_token_id=eos_token_id
                )
            scanner.seconds += time.monotonic() - started
            scanners.append(scanner)
        for scanner in scanners:
            scanner.timed(scanner.draw_documents, documents_per_source)
        active = list(scanners)
        while active:
            for scanner in list(active):
                stop_reason = scanner.stop_reason(max_scan_tokens_per_source, deadline)
                if stop_reason is None:
                    scanner.timed(scanner.scan_next_run)
                else:
                    scanner.stopped = stop_reason
                    active.remove(scanner)
    finally:
        for scanner in scanners:
            scanner.close()
    scans = [scanner.result() for scanner in scanners]
    for scan in scans:
        logger.info(
            f"[data-samples] source {scan.source.index} {scan.source.label}: {scan.documents_scanned} documents / "
            f"{scan.tokens_scanned} tokens scanned in {scan.seconds:.1f}s ({scan.stop_reason}); listed targets "
            f"{scan.listed_targets} ({scan.listed_trainable_targets} trainable), split forms "
            f"{scan.split_form_occurrences}, out of vocabulary {scan.out_of_vocab_tokens}"
        )
    return scans


def _indexed_sources(dataset_config: GPTDatasetConfig) -> tuple[list[DataSource], str | None]:
    if dataset_config.mock:
        return [], "mock dataset: the run trains on generated tokens, not on a corpus"
    if hasattr(dataset_config, "fim_data"):
        return [], "FIM dataset: documents are rearranged at load time, so the scan does not read it"
    if dataset_config.blend is not None:
        prefixes, weights = dataset_config.blend
    elif dataset_config.blend_per_split is not None:
        if dataset_config.blend_per_split[0] is None:
            return [], "blend_per_split names no training data"
        prefixes, weights = dataset_config.blend_per_split[0]
    else:
        raise ValueError(
            "the GPTDatasetConfig has neither blend nor blend_per_split; finalize() it before listing its sources"
        )
    normalised = None if weights is None else [float(weight) / float(sum(weights)) for weight in weights]
    labels = [Path(prefixes[0]).parent.name] if len(prefixes) == 1 else _distinguishing_parts(prefixes)
    return [
        DataSource(
            index=index,
            label=labels[index],
            path=prefix,
            kind="indexed",
            weight=None if normalised is None else normalised[index],
            tokenizer_recorded=_indexed_tokenizer_recorded(prefix),
        )
        for index, prefix in enumerate(prefixes)
    ], None


def _packed_sources(dataset_config: FinetuningDatasetConfig, tokenizer: Any) -> tuple[list[DataSource], str | None]:
    builder = finetuning_dataset_builder(dataset_config, tokenizer)
    if builder.packed_sequence_size <= 0:
        return [], "unpacked fine-tuning data (training.jsonl): the scan reads only packed parquet"
    spec = str(builder.train_path_packed)
    if spec.lower().endswith(".npy"):
        return [], f"legacy .npy packed data ({spec}): the scan reads only packed parquet"
    try:
        files = resolve_packed_parquet_paths(spec)
    except ValueError as error:
        return [], f"packed training data is not on disk yet ({error}); the dataset builder packs it at build time"
    if not files:
        return [], f"{spec} is not a packed parquet file, glob or directory"
    return [
        DataSource(
            index=0,
            label=_packed_label(spec),
            path=spec,
            kind="packed_parquet",
            weight=None,
            tokenizer_recorded=_packed_tokenizer_recorded(files),
        )
    ], None


def _packed_label(spec: str) -> str:
    """The directory a packed spec's shards belong to, above any glob and the packer's ``packed/<tokenizer>`` layout."""
    path = Path(spec)
    for part in reversed(path.parts if path.is_dir() else path.parts[:-1]):
        if part != "packed" and not _PACK_DIRECTORY.match(part) and not any(c in part for c in "*?["):
            return part
    return path.name


def _distinguishing_parts(paths: Sequence[str]) -> list[str]:
    """Each of several paths less the leading and trailing parts they all share; distinct paths stay distinct."""
    parts = [Path(path).parts for path in paths]
    shared = 0
    while shared < min(len(p) for p in parts) - 1 and len({p[shared] for p in parts}) == 1:
        shared += 1
    rests = [p[shared:] for p in parts]
    while all(len(rest) >= 2 for rest in rests) and len({rest[-1] for rest in rests}) == 1:
        rests = [rest[:-1] for rest in rests]
    return ["/".join(rest) for rest in rests]


def _indexed_tokenizer_recorded(prefix: str) -> str | None:
    provenance = Path(f"{prefix}.provenance.json")
    if provenance.is_file():
        tokenizer = json.loads(provenance.read_text()).get("parameters", {}).get("tokenizer")
        if tokenizer:
            return tokenizer
    results = Path(prefix).parent / "pipeline_results.json"
    if results.is_file():
        return json.loads(results.read_text()).get("tokenizer")
    return None


def _packed_tokenizer_recorded(files: Sequence[str]) -> str | None:
    slugs = sorted(
        {match["tokenizer"] for match in (_PACK_DIRECTORY.match(Path(f).parent.name) for f in files) if match}
    )
    return ", ".join(slugs) if slugs else None


def _check_scan_arguments(
    sources: Sequence[DataSource],
    *,
    split_forms: Sequence[SplitForm],
    vocab_size: int,
    counts: dict[str, int],
    eod_token_id: int | None,
    eod_mask_loss: bool | None,
    answer_only_loss: bool | None,
    eos_token_id: int | None,
) -> None:
    for name, value in counts.items():
        if value < 0:
            raise ValueError(f"{name} must be >= 0, got {value}")
    if vocab_size <= 0:
        raise ValueError(f"vocab_size must be positive, got {vocab_size}")
    for form in split_forms:
        edges = [edge for edge in (form.before, form.after) if edge is not None]
        if len(form.core) < (1 if edges else 2):
            raise ValueError(
                f"a split form is two or more token ids, or one or more between edge sets; got core {list(form.core)}"
            )
        if any(not edge for edge in edges):
            raise ValueError(f"a split form's edge sets must not be empty; got core {list(form.core)}")
    kinds = {source.kind for source in sources}
    if "indexed" in kinds:
        if eod_mask_loss is None:
            raise ValueError("eod_mask_loss is required to scan .bin/.idx sources")
        if eod_mask_loss and eod_token_id is None:
            raise ValueError("eod_token_id is required to scan .bin/.idx sources with eod_mask_loss")
    if "packed_parquet" in kinds and (answer_only_loss is None or eos_token_id is None):
        raise ValueError("answer_only_loss and eos_token_id are required to scan packed parquet sources")


@dataclass(frozen=True)
class _Search:
    """What every source is scanned for."""

    listed_token_ids: np.ndarray
    split_forms: list[_SplitFormSearch]
    vocab_size: int
    listed_documents_per_source: int


@dataclass(eq=False)
class _Run:
    """A contiguous stretch of a source, as whole documents back to back."""

    source_index: int
    tokens: np.ndarray
    target: np.ndarray
    trainable: np.ndarray
    document_starts: np.ndarray
    """Offsets of the documents in ``tokens``, followed by ``len(tokens)``."""
    first_token_is_target: bool
    reference: Callable[[int], str]

    @property
    def num_documents(self) -> int:
        return len(self.document_starts) - 1

    def document(self, k: int) -> SourceDocument:
        start, end = int(self.document_starts[k]), int(self.document_starts[k + 1])
        return SourceDocument(
            source_index=self.source_index,
            reference=self.reference(k),
            token_ids=self.tokens[start:end].astype(np.int64),
            trainable_target=self.trainable[start:end].copy(),
            first_token_is_target=self.first_token_is_target,
        )


@dataclass(frozen=True, eq=False)
class _SplitFormSearch:
    """A ``SplitForm`` as arrays, counted in a run's tokens."""

    core: np.ndarray
    before: np.ndarray | None
    after: np.ndarray | None

    @classmethod
    def of(cls, form: SplitForm) -> _SplitFormSearch:
        def ids(edge: frozenset[int] | None) -> np.ndarray | None:
            return None if edge is None else np.fromiter(sorted(edge), dtype=np.int64)

        return cls(np.asarray(form.core, dtype=np.int64), ids(form.before), ids(form.after))

    def occurrences(self, tokens: np.ndarray) -> int:
        length = len(self.core)
        windows = len(tokens) - length + 1
        if windows <= 0:
            return 0
        match = tokens[:windows] == self.core[0]
        for offset in range(1, length):
            match &= tokens[offset : offset + windows] == self.core[offset]
        starts = np.flatnonzero(match)
        if self.before is not None:
            starts = starts[starts >= 1]
            starts = starts[np.isin(tokens[starts - 1], self.before)]
        if self.after is not None:
            starts = starts[starts + length < len(tokens)]
            starts = starts[np.isin(tokens[starts + length], self.after)]
        return len(starts)


@dataclass(eq=False)
class _Scanner(ABC):
    """The scan state of one source; subclasses read the source's runs and random documents."""

    source: DataSource
    rng: np.random.Generator
    search: _Search
    documents: list[SourceDocument] = field(default_factory=list)
    listed_documents: list[SourceDocument] = field(default_factory=list)
    documents_scanned: int = 0
    tokens_scanned: int = 0
    listed_targets: int = 0
    listed_trainable_targets: int = 0
    split_form_occurrences: int = 0
    out_of_vocab_tokens: int = 0
    seconds: float = 0.0
    stopped: StopReason | None = None

    def timed(self, step: Callable[..., None], *args: Any) -> None:
        started = time.monotonic()
        step(*args)
        self.seconds += time.monotonic() - started

    def stop_reason(self, max_scan_tokens: int, deadline: float) -> StopReason | None:
        if not self.has_next_run():
            return "exhausted"
        if self.tokens_scanned >= max_scan_tokens:
            return "token_budget"
        if time.monotonic() > deadline:
            return "time_budget"
        return None

    def absorb(self, run: _Run) -> None:
        """Count a scanned run and keep its documents that contain a listed id, while the quota lasts."""
        listed = np.isin(run.tokens, self.search.listed_token_ids)
        listed_targets = listed & run.target
        self.listed_targets += int(listed_targets.sum())
        self.listed_trainable_targets += int((listed_targets & run.trainable).sum())
        self.split_form_occurrences += sum(form.occurrences(run.tokens) for form in self.search.split_forms)
        self.out_of_vocab_tokens += int((run.tokens >= self.search.vocab_size).sum())
        self.documents_scanned += run.num_documents
        self.tokens_scanned += len(run.tokens)
        room = self.search.listed_documents_per_source - len(self.listed_documents)
        if room > 0 and listed.any():
            containing = np.unique(np.searchsorted(run.document_starts, np.flatnonzero(listed), side="right") - 1)
            self.listed_documents.extend(run.document(int(k)) for k in containing[:room])

    def result(self) -> SourceScan:
        return SourceScan(
            source=self.source,
            documents=tuple(self.documents),
            listed_documents=tuple(self.listed_documents),
            documents_scanned=self.documents_scanned,
            tokens_scanned=self.tokens_scanned,
            listed_targets=self.listed_targets,
            listed_trainable_targets=self.listed_trainable_targets,
            split_form_occurrences=self.split_form_occurrences,
            out_of_vocab_tokens=self.out_of_vocab_tokens,
            stop_reason=self.stopped,
            seconds=self.seconds,
        )

    @abstractmethod
    def has_next_run(self) -> bool:
        """Whether a run is left to scan."""

    @abstractmethod
    def draw_documents(self, count: int) -> None:
        """Draw ``count`` random documents into ``documents`` (fewer when the source holds fewer)."""

    @abstractmethod
    def scan_next_run(self) -> None:
        """Read the next run and ``absorb`` it."""

    @abstractmethod
    def close(self) -> None:
        """Release the source's files."""


class _IndexedScanner(_Scanner):
    """A ``.bin/.idx`` corpus, cut into runs of whole documents that start at token offsets ``k * run_tokens``."""

    def __init__(
        self,
        source: DataSource,
        rng: np.random.Generator,
        search: _Search,
        max_scan_tokens: int,
        *,
        eod_token_id: int | None,
        eod_mask_loss: bool,
    ) -> None:
        from megatron.core.datasets.indexed_dataset import IndexedDataset

        super().__init__(source=source, rng=rng, search=search)
        self.eod_token_id = eod_token_id
        self.eod_mask_loss = eod_mask_loss
        self.dataset = IndexedDataset(source.path, mmap=True)
        self.sequence_lengths = self.dataset.sequence_lengths
        self.sequence_pointers = self.dataset.index.sequence_pointers
        self.document_indices = self.dataset.document_indices
        self.token_bytes = np.dtype(self.dataset.index.dtype).itemsize
        self.num_documents = len(self.document_indices) - 1
        total_tokens = (
            0
            if len(self.sequence_lengths) == 0
            else int(self.sequence_pointers[-1] - self.sequence_pointers[0]) // self.token_bytes
            + int(self.sequence_lengths[-1])
        )
        self.run_tokens = max(
            -(-max_scan_tokens // _RUNS_PER_TOKEN_BUDGET), -(-total_tokens // _MAX_RUNS_PER_CORPUS), 1
        )
        self.run_order = rng.permutation(-(-total_tokens // self.run_tokens))
        self.next_run = 0
        self._skip_empty_runs()

    def _first_document_at(self, run: int) -> int:
        """The first document that starts at or after token offset ``run * run_tokens``.

        ``bisect`` reads O(log n) entries of the memory-mapped index; ``np.searchsorted`` would first copy the whole
        array, because an ``.idx`` file's arrays are not aligned for their dtype.
        """
        if len(self.sequence_pointers) == 0:
            return 0
        pointer = int(self.sequence_pointers[0]) + run * self.run_tokens * self.token_bytes
        sequence = bisect.bisect_left(self.sequence_pointers, pointer)
        return bisect.bisect_left(self.document_indices, sequence)

    def _run_documents(self, run: int) -> tuple[int, int]:
        return self._first_document_at(run), self._first_document_at(run + 1)

    def _skip_empty_runs(self) -> None:
        while self.next_run < len(self.run_order):
            first, end = self._run_documents(int(self.run_order[self.next_run]))
            if end > first:
                return
            self.next_run += 1

    def _tokens(self, first_document: int, end_document: int) -> tuple[np.ndarray, np.ndarray]:
        """The documents' tokens back to back, and the documents' offsets followed by the total length."""
        first_sequence = int(self.document_indices[first_document])
        end_sequence = int(self.document_indices[end_document])
        lengths = np.asarray(self.sequence_lengths[first_sequence:end_sequence], dtype=np.int64)
        sequence_starts = np.concatenate([[0], np.cumsum(lengths)])
        document_starts = sequence_starts[
            np.asarray(self.document_indices[first_document : end_document + 1], dtype=np.int64) - first_sequence
        ]
        if end_sequence == first_sequence:
            return np.zeros(0, dtype=self.dataset.index.dtype), document_starts
        return np.concatenate(self.dataset[first_sequence:end_sequence]), document_starts

    def _token_before(self, document: int) -> int | None:
        """The last token of the corpus before ``document``, or None when nothing precedes it."""
        sequence = int(self.document_indices[document]) - 1
        while sequence >= 0 and self.sequence_lengths[sequence] == 0:
            sequence -= 1
        if sequence < 0:
            return None
        return int(self.dataset.get(sequence, offset=int(self.sequence_lengths[sequence]) - 1, length=1)[0])

    def _trainable(self, tokens: np.ndarray, first_document: int) -> np.ndarray:
        trainable = np.ones(len(tokens), dtype=bool)
        if self.eod_mask_loss and len(tokens):
            trainable[1:] = tokens[:-1] != self.eod_token_id
            previous = self._token_before(first_document)
            trainable[0] = previous is None or previous != self.eod_token_id
        return trainable

    def _run(self, first_document: int, end_document: int) -> _Run:
        tokens, document_starts = self._tokens(first_document, end_document)
        return _Run(
            source_index=self.source.index,
            tokens=tokens,
            target=np.ones(len(tokens), dtype=bool),
            trainable=self._trainable(tokens, first_document),
            document_starts=document_starts,
            first_token_is_target=True,
            reference=lambda k: f"document {first_document + k}",
        )

    def has_next_run(self) -> bool:
        return self.next_run < len(self.run_order)

    def draw_documents(self, count: int) -> None:
        chosen = np.sort(self.rng.choice(self.num_documents, size=min(count, self.num_documents), replace=False))
        self.documents.extend(self._run(int(d), int(d) + 1).document(0) for d in chosen)

    def scan_next_run(self) -> None:
        first, end = self._run_documents(int(self.run_order[self.next_run]))
        self.next_run += 1
        self.absorb(self._run(first, end))
        self._skip_empty_runs()

    def close(self) -> None:
        # The index arrays are views of the index file's mmap, which the dataset closes when it is freed.
        del self.sequence_lengths, self.sequence_pointers, self.document_indices
        del self.dataset


class _PackedScanner(_Scanner):
    """Packed parquet shards, scanned one row group at a time in a random order across the shards."""

    def __init__(
        self,
        source: DataSource,
        rng: np.random.Generator,
        search: _Search,
        *,
        answer_only_loss: bool,
        eos_token_id: int,
    ) -> None:
        import pyarrow.parquet as pq

        super().__init__(source=source, rng=rng, search=search)
        self.answer_only_loss = answer_only_loss
        self.eos_token_id = eos_token_id
        paths = resolve_packed_parquet_paths(self.source.path)
        self.shard_names = [Path(paths[0]).name] if len(paths) == 1 else _distinguishing_parts(paths)
        self.parquet_files = [pq.ParquetFile(path) for path in paths]
        units = [
            (file_index, row_group)
            for file_index, parquet_file in enumerate(self.parquet_files)
            for row_group in range(parquet_file.metadata.num_row_groups)
        ]
        self.units = [units[i] for i in rng.permutation(len(units))]
        self.next_unit = 0

    def _read_unit(self) -> list[_Run]:
        """Read the next row group whole; one run per pack (row)."""
        file_index, row_group = self.units[self.next_unit]
        self.next_unit += 1
        parquet_file = self.parquet_files[file_index]
        first_row = sum(parquet_file.metadata.row_group(g).num_rows for g in range(row_group))
        table = parquet_file.read_row_group(row_group, columns=["input_ids", "seq_start_id", "loss_mask"])
        input_ids, input_offsets = _list_column(table.column("input_ids"))
        starts, start_offsets = _list_column(table.column("seq_start_id"))
        loss_mask, mask_offsets = _list_column(table.column("loss_mask"))
        name = self.shard_names[file_index]
        return [
            self._pack_run(
                input_ids[input_offsets[row] : input_offsets[row + 1]],
                loss_mask[mask_offsets[row] : mask_offsets[row + 1]],
                starts[start_offsets[row] : start_offsets[row + 1]].astype(np.int64),
                reference=f"{name} row {first_row + row}",
            )
            for row in range(table.num_rows)
        ]

    def _pack_run(self, tokens: np.ndarray, loss_mask: np.ndarray, starts: np.ndarray, *, reference: str) -> _Run:
        boundaries = np.append(starts, len(tokens))
        input_mask = packed_sequence_loss_mask(
            tokens, loss_mask, boundaries, answer_only_loss=self.answer_only_loss, eos_id=self.eos_token_id
        )
        lengths = np.diff(boundaries)
        input_starts = np.concatenate([[0], np.cumsum(np.maximum(lengths - 1, 0))[:-1]])
        target = np.ones(len(tokens), dtype=bool)
        target[starts[lengths > 0]] = False
        positions = np.flatnonzero(target)
        conversation = np.repeat(np.arange(len(starts)), lengths)[positions]
        trainable = np.zeros(len(tokens), dtype=bool)
        trainable[positions] = (input_mask[input_starts[conversation] + positions - 1 - starts[conversation]] != 0) & (
            tokens[positions - 1] != self.eos_token_id
        )
        return _Run(
            source_index=self.source.index,
            tokens=tokens,
            target=target,
            trainable=trainable,
            document_starts=boundaries,
            first_token_is_target=False,
            reference=lambda k: f"{reference} conversation {k}",
        )

    def has_next_run(self) -> bool:
        return self.next_unit < len(self.units)

    def draw_documents(self, count: int) -> None:
        candidates: list[tuple[_Run, int]] = []
        while len(candidates) < count and self.has_next_run():
            for run in self._read_unit():
                self.absorb(run)
                candidates.extend((run, k) for k in range(run.num_documents))
        chosen = np.sort(self.rng.choice(len(candidates), size=min(count, len(candidates)), replace=False))
        self.documents.extend(candidates[int(i)][0].document(candidates[int(i)][1]) for i in chosen)

    def scan_next_run(self) -> None:
        for run in self._read_unit():
            self.absorb(run)

    def close(self) -> None:
        for parquet_file in self.parquet_files:
            parquet_file.close()


def _list_column(column: Any) -> tuple[np.ndarray, np.ndarray]:
    """A parquet list column's values, flat, and each row's offsets into them."""
    array = column.combine_chunks()
    offsets = array.offsets.to_numpy()
    return array.flatten().to_numpy(zero_copy_only=False), offsets - offsets[0]
