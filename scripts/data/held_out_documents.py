#!/usr/bin/env python3
# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""Held-out documents: the documents of their corpora that training runs have not trained on.

A run trains on a ``.bin/.idx`` blend for a fixed number of iterations, so it reads a fixed set of each corpus's
documents, each first at a fixed iteration, and a corpus it draws for less than one pass keeps documents it never
reads. This tool finds both kinds of unread document and writes what a campaign needs of them: a held-out set for
evaluating its checkpoints, a validation corpus for each run, and, for chosen iterations of one run, the documents
the run has yet to read there. One config states everything it does, and each of its five steps is a command:

* ``read --run <name>``: when the run first reads each document of every corpus of its blend. The run's training
  dataset is built on CPU as its launch builds it (``run_training_data.py``); the samples it reads, and the iteration
  that reads each, are the ones its data-parallel ranks' own samplers name over all its iterations; and a document
  is read when a sample reaches any token of it. One array per corpus is written (the iteration of each document's
  first read, 0 for one never read) beside a record of what the arrays depend on.
* ``held-out``: each run's held-out documents, which are the documents the run never reads, that are not empty, and
  whose document id no run reads anywhere. One document can sit in two corpora, or twice in one, so a row a run never
  reads is not held out while a row with the same id is read. The record states, per corpus, how many unread
  documents each reading corpus excludes.
* ``set``: a seeded uniform sample, without replacement, of one run's held-out documents, written as a parquet file of
  ``subset`` (the source subset a corpus's rows belong to) and ``source_row`` (the document's row in it), in corpus
  and row order.
* ``validation --run <name>``: a seeded uniform sample of one run's held-out documents, enough to hold the configured
  number of that run's training samples, copied ids byte for byte into one ``.bin/.idx`` prefix (what
  ``token_masking.masked_validation.data_path`` names). The prefix is re-read against its parents before it is given
  its final name, and gets a provenance record and a list of the documents it holds.
* ``future``: for each configured iteration of one run, a seeded uniform sample of the documents the run first reads
  after that iteration, that are not empty, and whose id the run has not read by then in any corpus: what a
  checkpoint saved there has not seen and the run goes on to train on. Each is a file like the set's, and a manifest
  (tab-separated ``stage``, ``iter``, ``path``, ``sha256``) lists them. The id rule here looks at the one run, and
  at the runs ``future.also_unread_by`` lists: a document, or an id, one of those has read by the same iteration
  of its own is nothing new either, and the record states how many documents that removed.

Document ids are one column of each corpus's dataset, read at the commit the corpus was prepared from after the
corpus's rows are checked to be its source's in order, and kept beside the lists once read, so the dataset is read
once however many steps need it.

A run may train on corpora that are not the table's and still hold the table's documents: another arm of a study,
built from the same source rows with other text, so that its document ``i`` is the table corpus's document ``i``. Such
a run names, for each corpus of its blend, that table corpus (``counterparts``; null for a corpus that is no table
corpus's, which then takes no part). It is read like any run, on its own corpora, and its reads count as reads of the
table's documents: no document it reads, and no id it reads, is held out of any run, and it can be listed under
``future.also_unread_by``. Only its reads are used: it has no held-out list, set, validation corpus or future sets,
since its own corpora are not what those would name. The held-out record states how many documents were removed by
such runs' reads alone.

Each step refuses to overwrite what it wrote before, and each later step refuses records that no longer describe the
config, the training configs or the corpora: a training config, a corpus's ``.idx`` or a list that changed since.

The config (YAML; paths are absolute, or relative to the repo root)::

    table: <corpora table>            # the corpora the runs' .bin/.idx prefixes are rows of
    lists: <directory>                # where `read` and `held-out` keep their lists, records and the ids
    runs:
      <name>: {config: <training config>, model: <model>, mode: <pretrain|cpt>, gpus: <GPUs it trains on>}
      <name>:                         # a run on other corpora that hold the table's documents, one for one
        config: ...
        counterparts: {<.bin/.idx prefix of its blend>: <the table corpus's prefix, or null>}
    document_ids:
      column: <dataset column>        # names a document wherever it sits
      source_subsets: {<table subset>: <source subset>}   # a subset that holds part of a source subset's rows
      subsets_without_ids: [<table subset>]               # corpora that are not rows of such a dataset
    set: {run: <name>, path: <file.parquet>, documents: <how many>, seed: <seed>}
    validation:
      <name>: {prefix: <.bin/.idx prefix>, samples: <training samples it must hold>, seed: <seed>}
    future:
      {run: <name>, also_unread_by: [<name>], iterations: [<iteration>], documents: <how many each>, seed: <seed>,
       directory: <dir>}

A table subset not named under ``source_subsets`` is a source subset itself. A corpus under ``subsets_without_ids``
takes no part: its documents are never held out and exclude none.

Usage (inside the container; as a job through ``configs/control_pretraining/corpus_job.sbatch``)::

    python scripts/data/held_out_documents.py read --config <config.yaml> --run <name>
    python scripts/data/held_out_documents.py held-out --config <config.yaml>
    python scripts/data/held_out_documents.py set --config <config.yaml>
    python scripts/data/held_out_documents.py validation --config <config.yaml> --run <name>
    python scripts/data/held_out_documents.py future --config <config.yaml>
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
for _directory in (REPO_ROOT, REPO_ROOT / "configs" / "control_pretraining"):
    if str(_directory) not in sys.path:
        sys.path.insert(0, str(_directory))
from corpora_table import (  # noqa: E402
    DATA_BASE,
    TOKENIZED_PREFIX,
    Checker,
    CorpusRow,
    corpus_root,
    read_corpora_table,
    selected_corpus,
    subset_prepare_config,
    tokenized_prefixes,
)
from corpus_documents import (  # noqa: E402
    PARTIAL,
    CopiedDocuments,
    CorpusCheckFailed,
    attributable_code_revision,
    bin_path,
    check_copied_documents,
    check_source_order,
    commit_corpus,
    copy_documents,
    dataset_columns,
    file_sha256,
    idx_path,
    integer_column,
    parent_record,
    provenance_path,
    read_index,
    write_record,
)
from scripts.data.prepare_revisions import FULL_SHA  # noqa: E402
from scripts.data.run_training_data import (  # noqa: E402
    BIN_IDX_MODES,
    MODELS,
    build_training_dataset,
    corpus_draws,
    data_parallel_size,
    documents_first_read,
    resolve_run_config,
    resumed_step,
    run_inputs,
    samples_read,
)
from scripts.mapping_keys import require_keys  # noqa: E402


logger: logging.Logger = logging.getLogger(__name__)

TOOL_VERSION = 1  # raised whenever what a step writes, or how it chooses documents, changes
READ_RECORD = "read.json"
HELD_OUT_RECORD = "held_out.json"
IDS_DIRECTORY = "ids"
FUTURE_MANIFEST = "future_sets.tsv"
FUTURE_RECORD = "future_sets.json"
MANIFEST_COLUMNS = ("stage", "iter", "path", "sha256")
# The fields of a corpus's tokenize record that decide its ids; the rest name its own files and job.
TOKENIZATION = ("tokenizer", "tokenizer_revision", "append_eod")

CONFIG_KEYS = frozenset({"table", "lists", "runs", "document_ids", "set", "validation", "future"})
RUN_KEYS = frozenset({"config", "model", "mode", "gpus"})
RUN_OPTIONAL_KEYS = frozenset({"counterparts"})
ID_KEYS = frozenset({"column", "source_subsets", "subsets_without_ids"})
SET_KEYS = frozenset({"run", "path", "documents", "seed"})
VALIDATION_KEYS = frozenset({"prefix", "samples", "seed"})
FUTURE_KEYS = frozenset({"run", "also_unread_by", "iterations", "documents", "seed", "directory"})
# In a yes-or-no reading of a corpus: a document some run of the table's own corpora reads, and one that only runs
# naming counterparts read. The id rule answers with the smallest it finds, so with the first wherever both apply.
READ, READ_THROUGH_COUNTERPARTS = 1, 2


# ---------------------------------------------------------------------------------------------
# The config
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Run:
    """One training run: its config as the launch passes it, and the model, mode and GPUs it launches with.

    ``counterparts`` is None for a run of the table's own corpora. A run of other corpora maps each prefix of its
    blend to the table corpus that holds the same documents in the same order, or to None for a corpus that is no
    table corpus's."""

    name: str
    config: Path
    model: str
    mode: str
    gpus: int
    counterparts: dict[Path, Path | None] | None


@dataclass(frozen=True)
class SetSpec:
    """The held-out set: how many of ``run``'s held-out documents it samples, with which seed, and where it goes."""

    run: str
    path: Path
    documents: int
    seed: int


@dataclass(frozen=True)
class ValidationSpec:
    """One run's validation corpus: its prefix, the training samples it must hold, and the sampling seed."""

    prefix: Path
    samples: int
    seed: int


@dataclass(frozen=True)
class FutureSpec:
    """The future sets: the iterations of ``run`` to write one for, the other runs whose reads by the same iteration
    also make a document nothing new (``also_unread_by``), how many documents each set samples, the seed and the
    directory the sets and their manifest go to."""

    run: str
    also_unread_by: tuple[str, ...]
    iterations: tuple[int, ...]
    documents: int
    seed: int
    directory: Path


@dataclass(frozen=True)
class HeldOutConfig:
    """A held-out config, parsed (see the module docstring)."""

    path: Path
    table: Path
    lists: Path
    runs: dict[str, Run]
    id_column: str
    source_subsets: dict[str, str]
    subsets_without_ids: tuple[str, ...]
    set: SetSpec
    validation: dict[str, ValidationSpec]
    future: FutureSpec


def _whole_number(value: object, where: str, minimum: int) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{where} must be a whole number of at least {minimum}, got {value!r}")
    return value


def _text(value: object, where: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{where} must be a non-empty string, got {value!r}")
    return value


def _counterparts(value: object, where: str) -> dict[Path, Path | None]:
    """A run's ``counterparts`` as paths, refusing a table corpus given to two corpora of the run's blend: each
    document of a table corpus is one document of the run."""
    if not isinstance(value, dict) or not value:
        raise ValueError(
            f"{where} must map each corpus of the run's blend to the table corpus that holds its documents, or to null"
        )
    named = {
        REPO_ROOT / _text(prefix, where): None if corpus is None else REPO_ROOT / _text(corpus, f"{where}.{prefix}")
        for prefix, corpus in value.items()
    }
    given = [str(corpus) for corpus in named.values() if corpus is not None]
    shared = sorted({corpus for corpus in given if given.count(corpus) > 1})
    if shared:
        raise ValueError(f"{where} gives {shared} to more than one corpus of the run's blend")
    return named


def read_config(path: Path) -> HeldOutConfig:
    """Parse a held-out config, refusing unknown and missing keys, a run it cannot launch-resolve by name, a set, a
    validation corpus or future sets naming a run it does not list or one that names counterparts (such a run is only
    read), and a subset name its corpora table does not hold."""
    loaded = require_keys(yaml.safe_load(path.read_text()), f"{path}: a held-out config", CONFIG_KEYS)
    table = REPO_ROOT / _text(loaded["table"], f"{path}: table")
    if not table.is_file():
        raise ValueError(f"{path}: no corpora table at {table}")
    subsets = {row.subset for row in read_corpora_table(table)}

    if not isinstance(loaded["runs"], dict) or not loaded["runs"]:
        raise ValueError(f"{path}: runs must map at least one run's name to its launch")
    runs = {}
    for name, entry in loaded["runs"].items():
        where = f"{path}: runs.{name}"
        entry = require_keys(entry, where, RUN_KEYS, RUN_OPTIONAL_KEYS)
        if entry["model"] not in MODELS or entry["mode"] not in BIN_IDX_MODES:
            raise ValueError(
                f"{where}: model must be one of {MODELS} and mode one of {list(BIN_IDX_MODES)}, got "
                f"{entry['model']!r} and {entry['mode']!r}"
            )
        runs[str(name)] = Run(
            name=str(name),
            config=REPO_ROOT / _text(entry["config"], f"{where}.config"),
            model=entry["model"],
            mode=entry["mode"],
            gpus=_whole_number(entry["gpus"], f"{where}.gpus", 1),
            counterparts=(
                _counterparts(entry["counterparts"], f"{where}.counterparts") if "counterparts" in entry else None
            ),
        )

    ids = require_keys(loaded["document_ids"], f"{path}: document_ids", ID_KEYS)
    source_subsets, without = ids["source_subsets"], ids["subsets_without_ids"]
    if not isinstance(source_subsets, dict) or not isinstance(without, list):
        raise ValueError(f"{path}: document_ids.source_subsets must be a mapping and subsets_without_ids a list")
    unknown = sorted({str(name) for name in [*source_subsets, *without]} - subsets)
    if unknown:
        raise ValueError(f"{path}: document_ids names subsets {unknown} that {table} does not hold")

    spec = require_keys(loaded["set"], f"{path}: set", SET_KEYS)
    if not isinstance(loaded["validation"], dict):
        raise ValueError(f"{path}: validation must map run names to their validation corpora")
    validation = {}
    for name, entry in loaded["validation"].items():
        where = f"{path}: validation.{name}"
        entry = require_keys(entry, where, VALIDATION_KEYS)
        validation[str(name)] = ValidationSpec(
            prefix=REPO_ROOT / _text(entry["prefix"], f"{where}.prefix"),
            samples=_whole_number(entry["samples"], f"{where}.samples", 1),
            seed=_whole_number(entry["seed"], f"{where}.seed", 0),
        )
    future = require_keys(loaded["future"], f"{path}: future", FUTURE_KEYS)
    if not isinstance(future["iterations"], list):
        raise ValueError(f"{path}: future.iterations must be a list of iterations (an empty one for no future sets)")
    iterations = tuple(_whole_number(value, f"{path}: future.iterations", 1) for value in future["iterations"])
    if len(set(iterations)) != len(iterations):
        raise ValueError(f"{path}: future.iterations names an iteration twice: {list(iterations)}")
    also_unread_by = future["also_unread_by"]
    if not isinstance(also_unread_by, list) or len(set(map(str, also_unread_by))) != len(also_unread_by):
        raise ValueError(f"{path}: future.also_unread_by must be a list of runs, each named once (empty for none)")
    also_unread_by = tuple(str(name) for name in also_unread_by)
    if str(future["run"]) in also_unread_by:
        raise ValueError(f"{path}: future.also_unread_by names future.run itself, {future['run']!r}")
    written_for = {str(spec["run"]), str(future["run"]), *validation}
    unlisted = sorted((written_for | set(also_unread_by)) - set(runs))
    if unlisted:
        raise ValueError(f"{path}: set, validation and future name runs {unlisted} that runs does not list")
    only_read = sorted(name for name in written_for if runs[name].counterparts is not None)
    if only_read:
        raise ValueError(
            f"{path}: set, validation and future write for runs {only_read}, which name counterparts: such a run is "
            "only read, since its corpora are not the table's"
        )
    return HeldOutConfig(
        path=path.resolve(),
        table=table,
        lists=REPO_ROOT / _text(loaded["lists"], f"{path}: lists"),
        runs=runs,
        id_column=_text(ids["column"], f"{path}: document_ids.column"),
        source_subsets={
            str(name): _text(source, f"{path}: document_ids.source_subsets.{name}")
            for name, source in source_subsets.items()
        },
        subsets_without_ids=tuple(str(name) for name in without),
        set=SetSpec(
            run=str(spec["run"]),
            path=REPO_ROOT / _text(spec["path"], f"{path}: set.path"),
            documents=_whole_number(spec["documents"], f"{path}: set.documents", 1),
            seed=_whole_number(spec["seed"], f"{path}: set.seed", 0),
        ),
        validation=validation,
        future=FutureSpec(
            run=str(future["run"]),
            also_unread_by=also_unread_by,
            iterations=tuple(sorted(iterations)),
            documents=_whole_number(future["documents"], f"{path}: future.documents", 1),
            seed=_whole_number(future["seed"], f"{path}: future.seed", 0),
            directory=REPO_ROOT / _text(future["directory"], f"{path}: future.directory"),
        ),
    )


# ---------------------------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------------------------


def write_parquet(table: pa.Table, path: Path) -> None:
    """Write a table under a temporary name and rename it, so a reader never finds part of one."""
    partial = Path(f"{path}{PARTIAL}")
    pq.write_table(table, partial)
    os.replace(partial, path)


def require_absent(*paths: Path) -> None:
    """Nothing this tool wrote before is overwritten: a step whose output exists is refused."""
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise CorpusCheckFailed(
            f"{existing} exist; a step's output is never overwritten, so remove it to redo the step"
        )


def config_record(config: HeldOutConfig) -> dict:
    """The config as a record names it: its path, its sha256 and its text, which outlives edits to the file."""
    return {"path": str(config.path), "sha256": file_sha256(config.path), "text": config.path.read_text()}


def sha256_of(payload: dict) -> str:
    """The sha256 of a JSON-serialisable mapping, key order aside."""
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


# ---------------------------------------------------------------------------------------------
# read: when one run first reads each document
# ---------------------------------------------------------------------------------------------


def read_step(config: HeldOutConfig, run: Run) -> dict:
    """Write, for every corpus of ``run``'s blend, the iteration at which the run first reads each document (0 for
    one it never reads), and their record."""
    directory = config.lists / run.name
    require_absent(directory / READ_RECORD)
    revision = attributable_code_revision()
    cfg = resolve_run_config(str(run.config), run.model, run.mode)
    if resumed_step(cfg):
        raise ValueError(
            f"{run.config} resumes at step {resumed_step(cfg)}: the steps before it read data this run's config does "
            "not describe, so what the whole run reads cannot be told from it"
        )
    data_parallel = data_parallel_size(str(run.config), cfg, run.gpus)
    train_ds, _, first_sample = build_training_dataset(cfg)
    samples, iterations = samples_read(cfg, train_ds, first_sample, data_parallel)
    expected = cfg.train.train_iters * cfg.train.global_batch_size
    if len(samples) != expected or numpy.unique(samples).size != expected:
        raise ValueError(
            f"{run.config}: its ranks read {len(samples)} samples, {numpy.unique(samples).size} of them distinct, "
            f"where {cfg.train.train_iters} iterations of {cfg.train.global_batch_size} read {expected} once each"
        )

    directory.mkdir(parents=True, exist_ok=True)
    built = corpus_draws(train_ds, numpy.arange(len(train_ds)))
    corpora = []
    for index, ((dataset, weight, entries, drawn), every) in enumerate(zip(corpus_draws(train_ds, samples), built)):
        first_read = documents_first_read(dataset, drawn, iterations[entries]).astype(numpy.int32)
        name = f"first_read.{index}.npy"
        numpy.save(directory / name, first_read)
        prefix = Path(str(dataset.dataset_path))
        corpora.append(
            {
                "prefix": str(prefix),
                "idx_sha256": file_sha256(idx_path(prefix)),
                "weight": weight,
                "samples_built": int(len(every[3])),
                "samples_read": int(len(drawn)),
                "documents": int(len(first_read)),
                "documents_read": int(numpy.count_nonzero(first_read)),
                "first_read": name,
                "first_read_sha256": file_sha256(directory / name),
            }
        )
        logger.info(
            f"  {run.name}: {numpy.count_nonzero(first_read):>11} of {len(first_read):>11} documents read, "
            f"{len(drawn):>10} of {len(every[3]):>10} samples  {prefix}"
        )
    record = {
        "format": "read-documents/1",
        "run": run.name,
        "config": config_record(config),
        "training_config": {"path": str(run.config), "sha256": file_sha256(run.config)},
        "model": run.model,
        "mode": run.mode,
        "gpus": run.gpus,
        "data_parallel_size": data_parallel,
        "inputs": run_inputs(cfg),
        "code_revision": revision,
        "dataset_samples": len(train_ds),
        "first_sample": first_sample,
        "samples_read": int(len(samples)),
        "corpora": corpora,
    }
    write_record(directory / READ_RECORD, record)
    return record


def current_read_record(config: HeldOutConfig, run: Run) -> dict:
    """``run``'s read record, refused unless it still describes the run, its corpora and its arrays.

    Files are compared by content, never by path: the steps run from frozen copies of a commit, each at its own path.
    """
    path = config.lists / run.name / READ_RECORD
    if not path.is_file():
        raise CorpusCheckFailed(f"{path} is missing; run `read --run {run.name}` first")
    record = json.loads(path.read_text())
    launch = (record["model"], record["mode"], record["gpus"])
    if launch != (run.model, run.mode, run.gpus):
        raise CorpusCheckFailed(
            f"{path} describes the launch {launch}, the config now states {(run.model, run.mode, run.gpus)}"
        )
    stale = []
    if record["training_config"]["sha256"] != file_sha256(run.config):
        stale.append(str(run.config))
    for corpus in record["corpora"]:
        if corpus["idx_sha256"] != file_sha256(idx_path(Path(corpus["prefix"]))):
            stale.append(str(idx_path(Path(corpus["prefix"]))))
        if corpus["first_read_sha256"] != file_sha256(path.parent / corpus["first_read"]):
            stale.append(str(path.parent / corpus["first_read"]))
    if stale:
        raise CorpusCheckFailed(f"{path} was written for other contents of {stale}; redo `read --run {run.name}`")
    return record


def first_reads(config: HeldOutConfig, run: Run, record: dict) -> dict[Path, numpy.ndarray]:
    """Per table corpus ``run`` reads, in its blend's order: the iteration of each document's first read (0 for
    none), as read wrote them.

    A run of the table's own corpora is keyed by its prefixes. A run that names counterparts is keyed by them, its
    corpora that have none left out; one whose blend and counterparts do not name the same corpora is refused, as
    nothing then says whose documents an unnamed corpus holds."""
    arrays = {
        Path(corpus["prefix"]): numpy.load(config.lists / run.name / corpus["first_read"])
        for corpus in record["corpora"]
    }
    if run.counterparts is None:
        return arrays
    if set(arrays) != set(run.counterparts):
        raise CorpusCheckFailed(
            f"{config.path}: runs.{run.name}.counterparts must name exactly the corpora of the run's blend; it leaves "
            f"out {sorted(str(prefix) for prefix in set(arrays) - set(run.counterparts))} and names "
            f"{sorted(str(prefix) for prefix in set(run.counterparts) - set(arrays))}, which the run does not read"
        )
    return {
        run.counterparts[prefix]: array for prefix, array in arrays.items() if run.counterparts[prefix] is not None
    }


def earliest_reads(runs: dict[str, dict[Path, numpy.ndarray]]) -> dict[Path, numpy.ndarray]:
    """Per corpus any of ``runs`` reads: for each document, the smallest non-zero value any of them holds for it, and
    0 where all hold 0. With first-read iterations that is the earliest read by any run; with a yes-or-no reading,
    the smallest mark. Runs that read one corpus must agree on how many documents it holds."""
    earliest: dict[Path, numpy.ndarray] = {}
    for name, arrays in runs.items():
        for prefix, array in arrays.items():
            if prefix not in earliest:
                earliest[prefix] = array.astype(numpy.int64)
                continue
            known = earliest[prefix]
            if len(known) != len(array):
                raise CorpusCheckFailed(
                    f"{prefix}: run {name} read {len(array)} documents of it where another run read {len(known)}; "
                    "corpora named as each other's counterparts must hold the same documents"
                )
            either = (known == 0) | (array == 0)
            earliest[prefix] = numpy.where(either, numpy.maximum(known, array), numpy.minimum(known, array))
    return earliest


# ---------------------------------------------------------------------------------------------
# The documents the id rule covers: their sizes and ids
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class TableCorpus:
    """One ``.bin/.idx`` prefix of a corpora table's row. ``rows`` are the subset rows its documents are, in order;
    None where the table does not say: a selection's documents are not rows of a dataset, and a byte-gated split
    decides its shards' rows when it runs."""

    row: CorpusRow
    rows: tuple[int, int] | None


def table_corpora(table: Path, data_base: Path) -> tuple[dict[Path, TableCorpus], list[str]]:
    """Every ``.bin/.idx`` prefix a corpora table's rows build, and the subsets it still holds (``PENDING``), which
    have no prefix to name yet. A pack row builds packed sequences, not a prefix."""
    corpora, held = {}, []
    for row in read_corpora_table(table):
        if row.kind == "pack":
            continue
        if row.docs is None:
            held.append(row.subset)
        elif row.kind == "select":
            for entry in selected_corpus(row, data_base).prefixes:
                corpora[entry.output] = TableCorpus(row, None)
        elif row.shard_mode == "split":
            root = corpus_root(subset_prepare_config(row.config, row.subset)["dataset"], row.subset, data_base)
            for name in row.shard_names:
                corpora[root / name / TOKENIZED_PREFIX] = TableCorpus(row, None)
        else:
            for entry in tokenized_prefixes(row, data_base):
                corpora[entry.prefix] = TableCorpus(row, entry.rows)
    return corpora, held


def subset_document_ids(row: CorpusRow, column: str, kept: Path) -> pa.ChunkedArray:
    """The id of every row of a tokenize row's subset, from its dataset at the commit the corpus was prepared from.

    The rows are checked to be the source's in order (``check_source_order``), which is what makes document ``i`` of
    the corpus the row whose id this is. A subset pinned at a full commit is fixed data, so its two columns are kept
    under ``kept`` once read, named for the subset, the column and the commit, and a later step reads them there; a
    kept copy passes the same checks as a fresh read. Text ids are given 64-bit offsets: a large corpus's ids
    together pass the 2 GiB a 32-bit-offset array can hold.
    """
    scalars = subset_prepare_config(row.config, row.subset)
    revision = scalars.get("revision")
    pinned = FULL_SHA.fullmatch(str(revision)) is not None
    path = kept / f"{row.subset}.{column}@{revision}.parquet"
    if pinned and path.is_file():
        table = pq.read_table(path)
    else:
        table = dataset_columns(scalars["dataset"], revision, row.subset, [column, row.row_column])
        if pinned:
            kept.mkdir(parents=True, exist_ok=True)
            write_parquet(table, path)
    if table.num_rows != row.docs:
        raise CorpusCheckFailed(f"{row.subset}: its dataset has {table.num_rows} rows, the table says {row.docs}")
    checker = Checker()
    check_source_order(row, integer_column(table, row.row_column), checker)
    if checker.failures:
        raise CorpusCheckFailed("; ".join(checker.failures))
    ids = table.column(column)
    if ids.null_count:
        raise CorpusCheckFailed(f"{row.subset}: {ids.null_count} rows have no {column}")
    return ids.cast(pa.large_string()) if pa.types.is_string(ids.type) else ids


@dataclass(frozen=True)
class CoveredDocuments:
    """The corpora the id rule covers among those some run reads, in the table's order: each one's table row and
    rows (``corpora``), its documents' sizes and its documents' ids."""

    corpora: dict[Path, TableCorpus]
    sizes: dict[Path, numpy.ndarray]
    ids: dict[Path, pa.ChunkedArray]

    def source(self, config: HeldOutConfig, prefix: Path) -> tuple[str, int]:
        """The source subset a covered corpus's rows belong to, and the source row of its document 0."""
        row, rows = self.corpora[prefix].row, self.corpora[prefix].rows
        return config.source_subsets.get(row.subset, row.subset), row.first_row + rows[0]


def covered_documents(config: HeldOutConfig, data_base: Path, read: dict[Path, numpy.ndarray]) -> CoveredDocuments:
    """What the id rule needs of the corpora in ``read`` (per corpus some run reads, one entry per document).

    A corpus the table does not build is refused, as nothing says which subset's rows its documents are; so is one
    whose documents cannot be traced to source rows, unless the config sets it aside (``subsets_without_ids``).
    Sizes come from each corpus's ``.idx`` and ids from its subset's dataset, read once per subset however many
    prefixes hold its rows.
    """
    corpora, held = table_corpora(config.table, data_base)
    unbuilt = [str(prefix) for prefix in read if prefix not in corpora]
    if unbuilt:
        raise CorpusCheckFailed(
            f"{unbuilt}, which a run reads, are no built corpora of {config.table} (rows it still holds: {held})"
        )
    covered = {}
    for prefix, corpus in corpora.items():
        if prefix not in read or corpus.row.subset in config.subsets_without_ids:
            continue
        if corpus.rows is None or corpus.row.row_column is None:
            raise CorpusCheckFailed(
                f"{corpus.row.subset}: its documents cannot be traced to source rows (a selection, a byte-gated split, "
                "or a table row naming no row column); list it under document_ids.subsets_without_ids if it takes no "
                "part in the id rule"
            )
        covered[prefix] = corpus
    subset_ids = {
        subset: subset_document_ids(row, config.id_column, config.lists / IDS_DIRECTORY)
        for subset, row in {corpus.row.subset: corpus.row for corpus in covered.values()}.items()
    }
    sizes, ids = {}, {}
    for prefix, corpus in covered.items():
        index = read_index(idx_path(prefix), bin_path(prefix))
        beg, end = corpus.rows
        if index.docs != end - beg or index.docs != len(read[prefix]):
            raise CorpusCheckFailed(
                f"{prefix}: holds {index.docs} documents, its rows {beg}:{end} number {end - beg} and what was read "
                f"of it {len(read[prefix])}"
            )
        sizes[prefix], ids[prefix] = index.sizes, subset_ids[corpus.row.subset].slice(beg, end - beg)
    return CoveredDocuments(covered, sizes, ids)


@dataclass(frozen=True)
class Candidates:
    """Documents of one corpus of one run, up for the id rule: the run, the corpus's place in its blend, the
    corpus's prefix, and the documents (sorted)."""

    run: str
    position: int
    prefix: Path
    documents: numpy.ndarray


def earliest_read_of_ids(
    candidates: list[Candidates], covered: CoveredDocuments, first_read: dict[Path, numpy.ndarray]
) -> tuple[list[numpy.ndarray], list[dict[str, int]]]:
    """Per entry of ``candidates``: for each of its documents, the earliest iteration at which a document with its
    id is read in any corpus (0 where none is), and how many of its documents each reading corpus's subset holds a
    read document for (a document can share its id with read documents of several).

    ``first_read`` gives, per corpus, the iteration at which each document is first read and 0 for one never read
    (for a yes-or-no reading, 1 for a read document). Candidates' ids are filtered chunk by chunk and looked up one
    reading corpus at a time, so no lookup table holds more than one corpus's ids. A corpus's read ids are looked up
    in the order they are read, and the lookup answers with the first of equal ids, so with the earliest read.
    """
    chunks = []
    for entry in candidates:
        chosen = numpy.zeros(len(covered.sizes[entry.prefix]), dtype=bool)
        chosen[entry.documents] = True
        chunks.extend(covered.ids[entry.prefix].filter(pa.array(chosen)).chunks)
    bounds = numpy.cumsum([0] + [len(entry.documents) for entry in candidates])
    never = numpy.iinfo(numpy.int64).max
    earliest = numpy.full(bounds[-1], never, dtype=numpy.int64)
    by_subset: list[dict[str, int]] = [{} for _ in candidates]
    if bounds[-1]:
        candidate_ids = pa.chunked_array(chunks)
        for prefix, corpus in covered.corpora.items():
            when = first_read[prefix]
            read = numpy.flatnonzero(when)
            if not read.size:
                continue
            read = read[numpy.argsort(when[read], kind="stable")]
            found = pc.index_in(candidate_ids, value_set=covered.ids[prefix].take(pa.array(read)))
            found = found.combine_chunks().fill_null(-1).to_numpy(zero_copy_only=False)
            hits = numpy.flatnonzero(found >= 0)
            earliest[hits] = numpy.minimum(earliest[hits], when[read[found[hits]]])
            # `hits` is sorted, so each entry's share of it lies between its bounds.
            for number, count in enumerate(numpy.diff(numpy.searchsorted(hits, bounds)).tolist()):
                if count:
                    by_subset[number][corpus.row.subset] = by_subset[number].get(corpus.row.subset, 0) + count
    earliest[earliest == never] = 0
    return [earliest[beg:end] for beg, end in zip(bounds[:-1], bounds[1:])], by_subset


# ---------------------------------------------------------------------------------------------
# held-out: the unread documents whose id no run reads
# ---------------------------------------------------------------------------------------------


def held_out_inputs(config: HeldOutConfig, records: dict[str, dict]) -> dict:
    """What the held-out lists depend on: the table's contents, the id rules and each run's read record."""
    return {
        "tool_version": TOOL_VERSION,
        "table_sha256": file_sha256(config.table),
        "id_column": config.id_column,
        "source_subsets": config.source_subsets,
        "subsets_without_ids": sorted(config.subsets_without_ids),
        "counterparts": {
            name: {str(prefix): None if corpus is None else str(corpus) for prefix, corpus in run.counterparts.items()}
            for name, run in config.runs.items()
            if run.counterparts is not None
        },
        "read_records": {name: sha256_of(record) for name, record in records.items()},
    }


def held_out_step(config: HeldOutConfig, data_base: Path) -> dict:
    """Write the held-out documents of each run of the table's corpora, corpus by corpus, and the record of how many
    each rule removed. A run that names counterparts only adds its reads."""
    require_absent(config.lists / HELD_OUT_RECORD)
    revision = attributable_code_revision()
    records = {name: current_read_record(config, run) for name, run in config.runs.items()}
    first = {name: first_reads(config, config.runs[name], record) for name, record in records.items()}
    # Per table corpus some run reads: READ for a document a run of the table's corpora reads,
    # READ_THROUGH_COUNTERPARTS for one that only runs naming counterparts read, 0 for one none reads.
    read = earliest_reads(
        {
            name: {
                prefix: numpy.where(
                    array > 0, READ if config.runs[name].counterparts is None else READ_THROUGH_COUNTERPARTS, 0
                )
                for prefix, array in arrays.items()
            }
            for name, arrays in first.items()
        }
    )
    covered = covered_documents(config, data_base, read)
    never_read = [
        Candidates(name, position, prefix, numpy.flatnonzero((array == 0) & (covered.sizes[prefix] > 0)))
        for name, arrays in first.items()
        if config.runs[name].counterparts is None
        for position, (prefix, array) in enumerate(arrays.items())
        if prefix in covered.corpora
    ]
    id_read, shared_with = earliest_read_of_ids(never_read, covered, read)

    # One entry per corpus of each run's blend, in its order. Those of a run of the table's corpora that the id rule
    # covers gain their counts and their list; those of a run naming counterparts state which table corpus they are.
    corpora, _ = table_corpora(config.table, data_base)
    runs: dict[str, list[dict]] = {}
    for name, record in records.items():
        counterparts = config.runs[name].counterparts
        runs[name] = []
        for corpus in record["corpora"]:
            prefix = Path(corpus["prefix"])
            entry = {"prefix": corpus["prefix"], "documents": corpus["documents"], "read": corpus["documents_read"]}
            if counterparts is None:
                entry.update(subset=corpora[prefix].row.subset, has_ids=prefix in covered.corpora)
            else:
                counterpart = counterparts[prefix]
                entry.update(
                    counterpart=None if counterpart is None else str(counterpart),
                    subset=None if counterpart is None else corpora[counterpart].row.subset,
                )
            runs[name].append(entry)
    for entry, earliest, subsets in zip(never_read, id_read, shared_with):
        source_subset, first_row = covered.source(config, entry.prefix)
        excluded = earliest > 0
        kept = entry.documents[~excluded].astype(numpy.int64)
        listed = f"held_out.{entry.position}.npy"
        numpy.save(config.lists / entry.run / listed, kept)
        runs[entry.run][entry.position].update(
            source_subset=source_subset,
            first_row=first_row,
            empty=int((covered.sizes[entry.prefix] == 0).sum()),
            unread=int(len(entry.documents)),
            id_read_by_a_run=int(excluded.sum()),
            id_read_only_through_counterparts=int((earliest == READ_THROUGH_COUNTERPARTS).sum()),
            id_read_in=subsets,
            held_out=int(len(kept)),
            held_out_tokens=int(covered.sizes[entry.prefix][kept].sum()),
            list=listed,
            list_sha256=file_sha256(config.lists / entry.run / listed),
        )
        logger.info(
            f"  {entry.run}: {len(kept):>11} held out of {len(entry.documents):>11} unread ({int(excluded.sum())} with "
            f"an id a run reads, {int((earliest == READ_THROUGH_COUNTERPARTS).sum())} of them only through "
            f"counterparts)  {entry.prefix}"
        )
    record = {
        "format": "held-out-documents/1",
        "config": config_record(config),
        "code_revision": revision,
        "inputs": held_out_inputs(config, records),
        "runs": {name: {"corpora": entries} for name, entries in runs.items()},
    }
    write_record(config.lists / HELD_OUT_RECORD, record)
    return record


@dataclass(frozen=True)
class SourceDocuments:
    """Documents of one corpus (sorted), with what names them at their source: the corpus's prefix and table subset,
    the source subset its rows belong to, and the source row of its document 0."""

    prefix: Path
    subset: str
    source_subset: str
    first_row: int
    documents: numpy.ndarray


def current_held_out(config: HeldOutConfig, run: str) -> tuple[dict, list[SourceDocuments]]:
    """The held-out record, refused unless it still describes the config's table, id rules and read records, and
    ``run``'s held-out documents from it, corpus by corpus in the blend's order (corpora without ids left out)."""
    path = config.lists / HELD_OUT_RECORD
    if not path.is_file():
        raise CorpusCheckFailed(f"{path} is missing; run `held-out` first")
    record = json.loads(path.read_text())
    records = {name: current_read_record(config, entry) for name, entry in config.runs.items()}
    if record["inputs"] != held_out_inputs(config, records):
        raise CorpusCheckFailed(
            f"{path} was written for another table, id rule or read record than the config now states; redo `held-out`"
        )
    held_out = []
    for corpus in record["runs"][run]["corpora"]:
        if not corpus["has_ids"]:
            continue
        listed = config.lists / run / corpus["list"]
        if corpus["list_sha256"] != file_sha256(listed):
            raise CorpusCheckFailed(f"{listed} is not the list {path} records; redo `held-out`")
        held_out.append(
            SourceDocuments(
                prefix=Path(corpus["prefix"]),
                subset=corpus["subset"],
                source_subset=corpus["source_subset"],
                first_row=corpus["first_row"],
                documents=numpy.load(listed),
            )
        )
    return record, held_out


# ---------------------------------------------------------------------------------------------
# Samples of documents, as source rows
# ---------------------------------------------------------------------------------------------


def source_rows_table(corpora: list[SourceDocuments], chosen: list[numpy.ndarray]) -> pa.Table:
    """``subset`` and ``source_row`` of the documents ``chosen`` of each corpus, corpus after corpus, refusing a
    source row named twice (two corpora holding the same rows of one source subset)."""
    subsets = [corpus.source_subset for corpus, documents in zip(corpora, chosen) for _ in range(len(documents))]
    rows = numpy.concatenate([corpus.first_row + documents for corpus, documents in zip(corpora, chosen)])
    if len(set(zip(subsets, rows.tolist()))) != len(rows):
        raise CorpusCheckFailed("two corpora name the same source row; their table rows overlap")
    return pa.table({"subset": pa.array(subsets, pa.string()), "source_row": pa.array(rows, pa.int64())})


def sample_per_corpus(corpora: list[SourceDocuments], positions: numpy.ndarray) -> list[numpy.ndarray]:
    """The documents at ``positions`` (sorted) of the corpora's documents laid end to end, per corpus."""
    bounds = numpy.cumsum([0] + [len(corpus.documents) for corpus in corpora])
    return [
        corpus.documents[positions[(positions >= beg) & (positions < end)] - beg]
        for corpus, beg, end in zip(corpora, bounds[:-1], bounds[1:])
    ]


def uniform_sample(corpora: list[SourceDocuments], documents: int, generator: numpy.random.Generator, what: str):
    """``documents`` of the corpora's documents, drawn uniformly without replacement, per corpus (sorted); more than
    there are is refused, naming ``what`` asked for them."""
    pool = sum(len(corpus.documents) for corpus in corpora)
    if documents > pool:
        raise CorpusCheckFailed(f"{what} asks for {documents} documents, and only {pool} qualify")
    return sample_per_corpus(corpora, numpy.sort(generator.choice(pool, size=documents, replace=False)))


def set_step(config: HeldOutConfig) -> dict:
    """Write the held-out set: ``set.documents`` of ``set.run``'s held-out documents, sampled uniformly."""
    spec = config.set
    record_path = Path(f"{spec.path}.json")
    require_absent(spec.path, record_path)
    revision = attributable_code_revision()
    held_out_record, corpora = current_held_out(config, spec.run)
    chosen = uniform_sample(corpora, spec.documents, numpy.random.default_rng(spec.seed), "set.documents")
    spec.path.parent.mkdir(parents=True, exist_ok=True)
    write_parquet(source_rows_table(corpora, chosen), spec.path)
    # What the held-out rules removed of each corpus the set samples, copied from the lists' record.
    removed = {entry["prefix"]: entry for entry in held_out_record["runs"][spec.run]["corpora"] if entry["has_ids"]}
    counted = ("unread", "id_read_by_a_run", "id_read_only_through_counterparts", "held_out")
    record = {
        "format": "held-out-set/1",
        "config": config_record(config),
        "code_revision": revision,
        "held_out_record_sha256": sha256_of(held_out_record),
        "run": spec.run,
        "seed": spec.seed,
        "documents": spec.documents,
        "held_out": sum(len(corpus.documents) for corpus in corpora),
        "path": str(spec.path),
        "sha256": file_sha256(spec.path),
        "corpora": [
            {
                "subset": corpus.subset,
                "source_subset": corpus.source_subset,
                **{count: removed[str(corpus.prefix)][count] for count in counted},
                "sampled": int(len(documents)),
            }
            for corpus, documents in zip(corpora, chosen)
        ],
    }
    write_record(record_path, record)
    return record


# ---------------------------------------------------------------------------------------------
# validation: a sample of one run's held-out documents, as a corpus
# ---------------------------------------------------------------------------------------------


def documents_list_path(prefix: Path) -> Path:
    """Where a validation corpus's documents are listed, as source rows, in the corpus's order."""
    return Path(f"{prefix}.documents.parquet")


def validation_step(config: HeldOutConfig, run: Run) -> dict:
    """Write ``run``'s validation corpus: a uniform sample of its held-out documents holding at least the configured
    number of its training samples, checked against its parents and recorded."""
    if run.name not in config.validation:
        raise ValueError(f"{config.path}: validation names no corpus for run {run.name}")
    spec = config.validation[run.name]
    prefix = spec.prefix
    require_absent(bin_path(prefix), idx_path(prefix), provenance_path(prefix), documents_list_path(prefix))
    revision = attributable_code_revision()
    held_out_record, corpora = current_held_out(config, run.name)
    sequence_length = current_read_record(config, run)["inputs"]["seq_length"]
    # One pass over a corpus of n ids holds (n - 1) // sequence_length samples: each sample's labels need one id more.
    needed = spec.samples * sequence_length + 1

    indexes = [read_index(idx_path(corpus.prefix), bin_path(corpus.prefix)) for corpus in corpora]
    sizes = numpy.concatenate([index.sizes[corpus.documents] for corpus, index in zip(corpora, indexes)])
    if sizes.sum() < needed:
        raise CorpusCheckFailed(
            f"run {run.name} holds out {int(sizes.sum())} ids, and {spec.samples} samples of {sequence_length} need "
            f"{needed}"
        )
    order = numpy.random.default_rng(spec.seed).permutation(len(sizes))
    taken = int(numpy.searchsorted(numpy.cumsum(sizes[order]), needed, side="left")) + 1
    chosen = sample_per_corpus(corpora, numpy.sort(order[:taken]))
    copied = [
        CopiedDocuments(corpus.prefix, index, documents) for corpus, index, documents in zip(corpora, indexes, chosen)
    ]
    # The corpus's record states how its ids were made, which must be one way for every parent that gives documents.
    tokenized = [
        {key: parameters[key] for key in TOKENIZATION if key in parameters}
        for parameters in (parent_record(source.prefix, source.index)["parameters"] for source in copied)
    ]
    given = [entry for entry, source in zip(tokenized, copied) if len(source.local)]
    if any(entry != given[0] for entry in given):
        raise CorpusCheckFailed(
            f"run {run.name}'s corpora were not all tokenized alike ({given}), so no one record describes a corpus "
            "drawn from them"
        )

    prefix.parent.mkdir(parents=True, exist_ok=True)
    partial_bin, partial_idx = bin_path(prefix, PARTIAL), idx_path(prefix, PARTIAL)
    copy_documents(copied, partial_bin, partial_idx)
    out = check_copied_documents(copied, partial_idx, partial_bin)
    documents = source_rows_table(corpora, chosen)
    record = {
        "kind": "held_out_validation",
        "tool": {"path": str(Path(__file__).resolve().relative_to(REPO_ROOT)), "version": TOOL_VERSION},
        "code_revision": revision,
        "created_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "config": config_record(config),
        "held_out_record_sha256": sha256_of(held_out_record),
        "run": run.name,
        "seed": spec.seed,
        "samples": spec.samples,
        "sequence_length": sequence_length,
        "documents": str(documents_list_path(prefix)),
        "parents": [
            {
                "prefix": str(corpus.prefix),
                "subset": corpus.subset,
                "idx_sha256": file_sha256(idx_path(corpus.prefix)),
                "held_out": int(len(corpus.documents)),
                "copied": int(len(taken_documents)),
            }
            for corpus, taken_documents in zip(corpora, chosen)
        ],
        "totals": {"total_tokens": out.tokens, "num_sequences": out.sequences, "num_documents": out.docs},
        "parameters": given[0],
    }
    write_parquet(documents, documents_list_path(prefix))
    commit_corpus(prefix, record)
    return record


# ---------------------------------------------------------------------------------------------
# future: at chosen iterations of one run, a sample of the documents it has yet to read
# ---------------------------------------------------------------------------------------------


def future_set_path(spec: FutureSpec, iteration: int) -> Path:
    """Where the future set of one iteration goes."""
    return spec.directory / f"future_{spec.run}_iter_{iteration:07d}.parquet"


def future_step(config: HeldOutConfig, data_base: Path) -> dict:
    """Write, for each of ``future.iterations``, a sample of the documents ``future.run`` first reads after it and
    whose id neither it nor a run of ``future.also_unread_by`` has read by then; then the manifest listing them and
    their record."""
    spec = config.future
    if not spec.iterations:
        raise ValueError(f"{config.path}: future.iterations is empty, so there is no future set to write")
    manifest, record_path = spec.directory / FUTURE_MANIFEST, spec.directory / FUTURE_RECORD
    paths = {iteration: future_set_path(spec, iteration) for iteration in spec.iterations}
    require_absent(manifest, record_path, *paths.values())
    revision = attributable_code_revision()
    read_record = current_read_record(config, config.runs[spec.run])
    last = read_record["inputs"]["train_iters"]
    if spec.iterations[-1] >= last:
        raise ValueError(
            f"{config.path}: future.iterations names {spec.iterations[-1]}, and run {spec.run} reads nothing after "
            f"its last iteration, {last}"
        )
    first = first_reads(config, config.runs[spec.run], read_record)
    other_records = {name: current_read_record(config, config.runs[name]) for name in spec.also_unread_by}
    others = {name: first_reads(config, config.runs[name], record) for name, record in other_records.items()}
    # Per corpus any of them reads: each document's earliest read by the run or by one of the others.
    by_any = earliest_reads({spec.run: first, **others})
    covered = covered_documents(config, data_base, by_any)
    # Every document the run reads (none of them empty), with the earliest iteration at which its id is read: by the
    # run, and by the run or one of the others.
    read = [
        Candidates(spec.run, position, prefix, numpy.flatnonzero(first[prefix]))
        for position, prefix in enumerate(first)
        if prefix in covered.corpora
    ]
    by_run = {
        prefix: first[prefix] if prefix in first else numpy.zeros(len(by_any[prefix]), dtype=numpy.int64)
        for prefix in covered.corpora
    }
    id_read_by_run, _ = earliest_read_of_ids(read, covered, by_run)
    id_read_by_any = (
        earliest_read_of_ids(read, covered, {prefix: by_any[prefix] for prefix in covered.corpora})[0]
        if others
        else id_read_by_run
    )

    spec.directory.mkdir(parents=True, exist_ok=True)
    sets = []
    for iteration in spec.iterations:
        later, new_to_the_run, unseen = 0, 0, []
        for entry, run_earliest, any_earliest in zip(read, id_read_by_run, id_read_by_any):
            source_subset, first_row = covered.source(config, entry.prefix)
            subset = covered.corpora[entry.prefix].row.subset
            later += int((first[entry.prefix][entry.documents] > iteration).sum())
            new_to_the_run += int((run_earliest > iteration).sum())
            # A document's id is first read no later than the document, and by any of the runs no later than by the
            # one, so an id none of them has read by now is a document the run has yet to read.
            unseen.append(
                SourceDocuments(
                    entry.prefix, subset, source_subset, first_row, entry.documents[any_earliest > iteration]
                )
            )
        generator = numpy.random.default_rng([spec.seed, iteration])
        chosen = uniform_sample(unseen, spec.documents, generator, f"future.documents at iteration {iteration}")
        write_parquet(source_rows_table(unseen, chosen), paths[iteration])
        sets.append(
            {
                "stage": spec.run,
                "iter": iteration,
                "path": str(paths[iteration]),
                "sha256": file_sha256(paths[iteration]),
                "first_read_later": later,
                "id_read_by_then": later - new_to_the_run,
                "read_by_others_by_then": new_to_the_run - sum(len(corpus.documents) for corpus in unseen),
                "unseen": sum(len(corpus.documents) for corpus in unseen),
            }
        )
        logger.info(
            f"  {spec.run} after iteration {iteration}: {spec.documents} of {sets[-1]['unseen']} unseen documents "
            f"({sets[-1]['id_read_by_then']} more share an id already read, {sets[-1]['read_by_others_by_then']} more "
            f"were read by {list(spec.also_unread_by)})  {paths[iteration]}"
        )
    lines = ["\t".join(MANIFEST_COLUMNS)] + [
        "\t".join(str(entry[column]) for column in MANIFEST_COLUMNS) for entry in sets
    ]
    partial = Path(f"{manifest}{PARTIAL}")
    partial.write_text("\n".join(lines) + "\n")
    os.replace(partial, manifest)
    record = {
        "format": "future-sets/1",
        "config": config_record(config),
        "code_revision": revision,
        "read_record_sha256": sha256_of(read_record),
        "run": spec.run,
        "also_unread_by": {name: sha256_of(record) for name, record in other_records.items()},
        "seed": spec.seed,
        "documents": spec.documents,
        "manifest": str(manifest),
        "sets": sets,
    }
    write_record(record_path, record)
    return record


# ---------------------------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Run one step of a held-out config; non-zero on any refusal."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    steps = {
        "read": "tell when one run first reads each document",
        "held-out": "name each run's held-out documents",
        "set": "write the held-out set",
        "validation": "write one run's validation corpus",
        "future": "write the future sets of one run's iterations",
    }
    for name, summary in steps.items():
        step = commands.add_parser(name, help=summary)
        step.add_argument("--config", type=Path, required=True, help="the held-out config")
        if name in ("read", "validation"):
            step.add_argument("--run", required=True, help="the run, as the config names it")
        if name in ("held-out", "future"):
            step.add_argument(
                "--data-base", type=Path, default=DATA_BASE, help=f"the corpus roots' base (default: {DATA_BASE})"
            )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        config = read_config(args.config)
        if args.command in ("read", "validation") and args.run not in config.runs:
            raise ValueError(f"{args.config}: no run {args.run!r}; it lists {sorted(config.runs)}")
        if args.command == "read":
            record = read_step(config, config.runs[args.run])
            logger.info(f"{args.run}: {record['samples_read']} samples read; {config.lists / args.run / READ_RECORD}")
        elif args.command == "held-out":
            held_out_step(config, args.data_base)
            logger.info(f"held-out documents listed; {config.lists / HELD_OUT_RECORD}")
        elif args.command == "set":
            record = set_step(config)
            logger.info(f"{record['documents']} of {record['held_out']} held-out documents: {record['path']}")
            logger.info(f"sha256 {record['sha256']}")
        elif args.command == "validation":
            record = validation_step(config, config.runs[args.run])
            totals = record["totals"]
            logger.info(
                f"{args.run}: {totals['num_documents']} documents, {totals['total_tokens']} ids: "
                f"{config.validation[args.run].prefix}"
            )
        else:
            record = future_step(config, args.data_base)
            logger.info(f"{len(record['sets'])} future sets of {record['run']}: {record['manifest']}")
    except (CorpusCheckFailed, ValueError) as error:
        logger.error(f"FAILED: {error}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
