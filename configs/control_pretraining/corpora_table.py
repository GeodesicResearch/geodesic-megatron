"""The control-pretraining corpora table: one row per corpus an arm builds.

Each arm keeps a ``corpora.tsv`` beside its training configs (``|``-separated, ``#`` comments).
``build_corpora.sh`` submits the build from it and ``verify_corpora.py`` checks the result
against it, so the table is the single statement of what an arm's corpora are, which prepare
config defines them, and how each is cut. This module is the one parser for it, and the one
place the per-corpus output directory is derived, so the verifier and the tests cannot read
the table differently from each other.

Columns, in order::

    subset      HuggingFace config name within the prepare config's dataset
    stage       which training stage reads it (pretraining | midtraining | sft |
                continual_pretraining); a build or verification can be limited to one stage
    kind        tokenize  -> .bin/.idx via pipeline_data_submit.sbatch tokenize
                pack      -> packed SFT parquet via pipeline_data_submit.sbatch <root> ...
                select    -> a subset of another table's tokenized corpus: the documents a kept
                             list names, their ids copied in order by corpus_documents.py select
    config      prepare config YAML (dataset, revision, tokenizer, pack geometry), named
                relative to the repo root; for kind=select, the select config instead
                (`read_select_config`: output dataset, parent table and subset, kept list, and
                the token ids no kept document may hold).
                A prepare config pins one `revision`, or each subset's own commit under
                `revisions`; a row is read through `subset_prepare_config`, which resolves
                its subset's pin and refuses a subset `revisions` does not pin
    prep_h      prepare walltime, hours; 0 for kind=select, which has no prepare
    tok_h       tokenize/pack/select walltime, hours (per shard where sharded)
    workers     tokenize workers; 1 for kind=select, which copies in one process
    shards      1, or the shard count where sharded; a select row restates its parent's
    shard_mode  none | split (one JSONL, byte-gated split) | slice (N source index ranges);
                a select row restates its parent's
    stripe      1 to lfs setstripe the roots before the first write, else 0
    docs        the subset's document count, or PENDING. Slicing and verification both need
                the integer, and `plan_corpus` refuses a PENDING row whatever its shard mode,
                so PENDING doubles as a HOLD: a corpus whose count is known but whose source
                is not yet safe to build from is held by leaving the column PENDING, and the
                whole plan is refused rather than quietly omitting that corpus. An arm that
                holds a row this way states its reason at the row. For kind=select it is the
                kept list's length.

Four further columns are optional, and a row states all four or none::

    count_token   a token id whose occurrences ``verify_corpora.py`` counts in every document
                  of the built ``.bin`` (the document's EOD slot excluded, by position)
    count_column  the column of the prepare config's dataset that holds, row for row, how many
                  of ``count_token`` each document must contain
    row_column    the column of that dataset that holds each row's index in the source the corpus
                  must align with, document for document
    first_row     the source index of the dataset's first row: row ``i`` must hold
                  ``first_row + i``, so the rows are the source's, in order, with none dropped or
                  repeated

They declare the per-document checks (``corpus_documents.check_documents``), which only a
tokenize row can carry; of the dataset, they read only ``count_column`` and ``row_column``. A fifth may
follow them, and is then read as well:

    length_column the column of that dataset that holds each row's length in tokens before the
                  build changed its text (a hidden-span corpus's ``n_tokens``): the checks then
                  report, without failing, the documents whose built length is not that plus the
                  EOD (an empty text: 0), and the net and range of the shift, which re-tokenizing
                  changed text produces
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path

import yaml


COLUMNS = ("subset", "stage", "kind", "config", "prep_h", "tok_h", "workers", "shards", "shard_mode", "stripe", "docs")
# Optional, all four or none, after COLUMNS.
DOCUMENT_CHECK_COLUMNS = ("count_token", "count_column", "row_column", "first_row")
# Optional after the four.
LENGTH_COLUMN = "length_column"
KINDS = ("tokenize", "pack", "select")
SHARD_MODES = ("none", "split", "slice")
STEPS = ("prepare", "split", "tokenize", "pack", "select")  # the job steps a corpus's chain is made of
DOCS_PENDING = "PENDING"

# Tables name their prepare configs relative to the repo root, as build_corpora.sh's own
# invocations do. This module sits at configs/control_pretraining/, so the root is two levels
# up — resolving through it means a reader works from any working directory.
REPO_ROOT = Path(__file__).resolve().parents[2]

# build_corpora.sh runs this module as a script, whose import path holds only its own directory.
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))
from scripts.data.prepare_revisions import subset_revision, tokenizer_reference, tokenizer_revision  # noqa: E402
from scripts.mapping_keys import require_keys  # noqa: E402
from scripts.token_ids import require_token_id_list  # noqa: E402


# Where every corpus root lives.
DATA_BASE = Path("/projects/a5k/public/data")
# The batch scripts a job can be submitted through. Every prepare, tokenize and pack goes
# through the data pipeline's sbatch wrapper; a split is its own script, submitted directly.
SUBMIT_SCRIPT = "pipeline_data_submit.sbatch"
SHARD_SCRIPT = Path(__file__).resolve().parent / "shard_jsonl_corpus.sh"
# A select runs this directory's document tool as its own 1-node CPU job inside the container.
CORPUS_JOB_SCRIPT = Path(__file__).resolve().parent / "corpus_job.sbatch"
DOCUMENTS_TOOL = Path(__file__).resolve().parent / "corpus_documents.py"
# The tokenize job's output prefix: pipeline_data_submit.sbatch tokenize <root> <tokenizer>
# tokenized_base input -> <root>/tokenized_base_input_document.{bin,idx,provenance.json}.
TOKENIZED_PREFIX = "tokenized_base_input_document"


def shard_name(index: int) -> str:
    """The subdirectory of a corpus root that holds shard ``index``; ``shard_jsonl_corpus.sh``
    writes a split corpus's shards under the same names."""
    return f"shard{index}"


@dataclass(frozen=True)
class CorpusRow:
    """One line of a corpora table, with the numeric columns parsed."""

    subset: str
    stage: str
    kind: str
    config: Path
    prep_h: int
    tok_h: int
    workers: int
    shards: int
    shard_mode: str
    stripe: bool
    docs: int | None  # None while the table still says PENDING
    # The per-document checks' columns, all None when the row declares none.
    count_token: int | None
    count_column: str | None
    row_column: str | None
    first_row: int | None
    # The dataset's column of each row's length before the build, None when the row declares none.
    length_column: str | None
    # Where the row was read from (resolved), which a select job is pointed back at. Not part of the
    # row's identity: two tables stating the same corpus state the same row.
    table: Path = field(compare=False)

    @property
    def shard_names(self) -> list[str]:
        """The shard subdirectories under the corpus root, or [] for an unsharded corpus."""
        if self.shard_mode == "none":
            return []
        return [shard_name(i) for i in range(self.shards)]

    def slice_ranges(self) -> list[tuple[int, int]]:
        """The N contiguous ``[beg, end)`` document ranges a slice-mode corpus is prepared from.

        This is the definition of the ranges: the build submits exactly these ``train[beg:end]``
        splits and the verifier asserts the prepares recorded them, both from here.
        """
        if self.shard_mode != "slice":
            raise ValueError(f"{self.subset}: slice_ranges() on shard_mode={self.shard_mode}")
        if self.docs is None:
            raise ValueError(f"{self.subset}: slice ranges need the document count, table says PENDING")
        return [(i * self.docs // self.shards, (i + 1) * self.docs // self.shards) for i in range(self.shards)]


def corpus_root(dataset: str, subset: str, data_base: Path = DATA_BASE) -> Path:
    """The directory ``pipeline_data_prepare.py`` writes a subset into.

    Mirrors its ``slugify_dataset_name``: ``dataset.replace("/", "__") + "__" + subset``. Stated
    here rather than imported because that script imports torch-adjacent packages at module
    level and this module must stay usable outside the container.
    """
    return data_base / f"{dataset.replace('/', '__')}__{subset}"


def packed_parquet_path(root: Path, scalars: dict) -> Path:
    """The packed-sequence parquet the packer writes under a corpus (or shard) root.

    Mirrors ``pipeline_data_prepare.py``'s layout — ``packed/<tokenizer slug>_pad_seq_to_mult<N>/
    training_<seq>.idx.parquet`` — from the prepare config's tokenizer and pack geometry, so the
    verifier, the audit and the training configs' globs all name the same file.
    """
    tokenizer_dir = f"{scalars['tokenizer'].replace('/', '--')}_pad_seq_to_mult{scalars['pad-seq-to-mult']}"
    return root / "packed" / tokenizer_dir / f"training_{scalars['seq-length']}.idx.parquet"


class Checker:
    """Collects failures so one run reports everything wrong with a build, then exits non-zero."""

    def __init__(self) -> None:
        self.failures: list[str] = []

    def expect(self, condition: bool, message: str) -> bool:
        if not condition:
            self.failures.append(message)
        return condition


def prepare_config_scalars(config: Path) -> dict:
    """The top-level scalars of a prepare config (dataset, revision, tokenizer, geometry)."""
    with open(config) as fh:
        loaded = yaml.safe_load(fh)
    if not isinstance(loaded, dict):
        raise ValueError(f"{config}: expected a mapping at the top level")
    return loaded


def subset_prepare_config(config: Path, subset: str) -> dict:
    """A prepare config as the prepare of ``subset`` reads it: its scalars, with that subset's pin as ``revision``.

    A config that pins each subset (``revisions``) gives the subset's own commit, and one that pins a single
    ``revision`` gives that, as ``pipeline_data_prepare.py`` resolves them (``scripts/data/prepare_revisions.py``).
    The plan and the verifier read a row's prepare config through here, and the verifier hands the result to the
    per-document count check, so none of them can check a corpus against another subset's commit. A
    ``tokenizer-revision`` the config states is kept and must be a full commit SHA. Raises ``ValueError`` for a
    subset the config does not pin and for a malformed tokenizer pin.
    """
    scalars = prepare_config_scalars(config)
    revision = subset_revision(scalars, subset, str(config))
    tokenizer_revision(scalars, str(config))
    resolved = {key: value for key, value in scalars.items() if key != "revisions"}
    if revision is not None:
        resolved["revision"] = revision
    return resolved


def _parse_row(line: str, table: Path, line_no: int) -> CorpusRow:
    fields = [f.strip() for f in line.split("|")]
    checked = len(COLUMNS) + len(DOCUMENT_CHECK_COLUMNS)
    if len(fields) not in (len(COLUMNS), checked, checked + 1):
        raise ValueError(
            f"{table}:{line_no}: expected {len(COLUMNS)} '|'-separated columns, or {checked} with "
            f"{DOCUMENT_CHECK_COLUMNS}, or {checked + 1} with {LENGTH_COLUMN} after them, got {len(fields)}"
        )
    row = dict(zip(COLUMNS + DOCUMENT_CHECK_COLUMNS + (LENGTH_COLUMN,), fields))
    for name, value in row.items():
        if not value:
            raise ValueError(f"{table}:{line_no}: column '{name}' is empty")
    if row["kind"] not in KINDS:
        raise ValueError(f"{table}:{line_no}: kind must be one of {KINDS}, got {row['kind']!r}")
    if row["shard_mode"] not in SHARD_MODES:
        raise ValueError(f"{table}:{line_no}: shard_mode must be one of {SHARD_MODES}, got {row['shard_mode']!r}")
    if row["stripe"] not in ("0", "1"):
        raise ValueError(f"{table}:{line_no}: stripe must be 0 or 1, got {row['stripe']!r}")
    docs = None if row["docs"] == DOCS_PENDING else int(row["docs"])
    config = Path(row["config"])
    parsed = CorpusRow(
        subset=row["subset"],
        stage=row["stage"],
        kind=row["kind"],
        config=config if config.is_absolute() else REPO_ROOT / config,
        prep_h=int(row["prep_h"]),
        tok_h=int(row["tok_h"]),
        workers=int(row["workers"]),
        shards=int(row["shards"]),
        shard_mode=row["shard_mode"],
        stripe=row["stripe"] == "1",
        docs=docs,
        count_token=int(row["count_token"]) if "count_token" in row else None,
        count_column=row.get("count_column"),
        row_column=row.get("row_column"),
        first_row=int(row["first_row"]) if "first_row" in row else None,
        length_column=row.get(LENGTH_COLUMN),
        table=table.resolve(),
    )
    if parsed.shard_mode == "none" and parsed.shards != 1:
        raise ValueError(f"{table}:{line_no}: shard_mode=none requires shards=1, got {parsed.shards}")
    if parsed.shard_mode != "none" and parsed.shards < 2:
        raise ValueError(f"{table}:{line_no}: shard_mode={parsed.shard_mode} requires shards>=2, got {parsed.shards}")
    if parsed.kind == "pack" and parsed.shard_mode == "slice":
        raise ValueError(f"{table}:{line_no}: kind=pack shards through the byte-gated split, not source slicing")
    if parsed.count_token is not None and parsed.kind != "tokenize":
        raise ValueError(f"{table}:{line_no}: the per-document checks read a tokenized corpus, not kind={parsed.kind}")
    if parsed.count_token is not None and parsed.count_token < 0:
        raise ValueError(f"{table}:{line_no}: count_token must be a token id, got {parsed.count_token}")
    if parsed.first_row is not None and parsed.first_row < 0:
        raise ValueError(f"{table}:{line_no}: first_row must be a row index, got {parsed.first_row}")
    if parsed.count_column is not None and parsed.count_column == parsed.row_column:
        raise ValueError(f"{table}:{line_no}: count_column and row_column name one column, {parsed.row_column!r}")
    if parsed.length_column is not None and parsed.length_column in (parsed.count_column, parsed.row_column):
        raise ValueError(f"{table}:{line_no}: length_column names another check's column, {parsed.length_column!r}")
    if parsed.kind == "select" and parsed.prep_h != 0:
        raise ValueError(f"{table}:{line_no}: kind=select has no prepare, so prep_h must be 0, got {parsed.prep_h}")
    if parsed.kind == "select" and parsed.workers != 1:
        raise ValueError(f"{table}:{line_no}: kind=select copies in one process, so workers must be 1")
    return parsed


def read_corpora_table(table: Path, stage: str = "all", subsets: list[str] | None = None) -> list[CorpusRow]:
    """Parse a corpora table, optionally keeping only one stage's rows, or only the named subsets.

    Every rule a table must satisfy is enforced here, so a table the build refuses cannot be
    verified against either. A requested subset that the selected rows do not contain is an
    error, not an empty result: the caller named it in order to act on it.
    """
    rows: list[CorpusRow] = []
    with open(table) as fh:
        for line_no, raw in enumerate(fh, start=1):
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            row = _parse_row(line, table, line_no)
            if stage == "all" or row.stage == stage:
                rows.append(row)
    names = [r.subset for r in rows]
    duplicates = sorted({s for s in names if names.count(s) > 1})
    if duplicates:
        raise ValueError(f"{table}: subset listed more than once: {duplicates}")
    if subsets is not None:
        if not subsets:
            raise ValueError(f"{table}: an empty subset selection would plan nothing; pass None to keep every row")
        missing = sorted(set(subsets) - set(names))
        if missing:
            raise ValueError(f"{table}: no row for subset {missing} in stage '{stage}'")
        rows = [r for r in rows if r.subset in set(subsets)]
    return rows


# The tokenize job's trailing arguments, and the fixed walltime of a split job. These belong
# with the table's semantics rather than in the submitting script, so that the plan a build
# executes and the artifacts a verification looks for are derived from one place.
OUTPUT_VARIANT = "tokenized_base"
JSON_KEY = "input"
SPLIT_HOURS = 6
# A pack row's per-shard pack jobs take their geometry from the prepare config, so the config
# must state it; a pack at the packer's defaults would silently mismatch the training topology.
PACK_GEOMETRY_KEYS = ("seq-length", "pad-seq-to-mult")


@dataclass(frozen=True)
class PlannedJob:
    """One SLURM submission, with the key of the job it must wait for."""

    key: str
    step: str  # one of STEPS
    depends_on: str  # "" for a job that starts immediately
    hours: int
    name: str
    description: str
    script: str  # the batch script isambard_sbatch submits
    payload: tuple[str, ...]  # that script's arguments
    sbatch_args: tuple[str, ...] = ()
    shard: int | None = None  # the shard a per-shard job builds; None for a job every shard shares


@dataclass(frozen=True)
class CorpusPlan:
    """Everything the build does for one corpus: directories to create, then jobs to submit."""

    row: CorpusRow
    root: Path
    dataset: str
    tokenizer: str
    roots: tuple[tuple[Path, bool], ...]  # (directory, stripe it)
    jobs: tuple[PlannedJob, ...]


def select_steps(jobs: tuple[PlannedJob, ...], steps: frozenset[str] | None) -> tuple[PlannedJob, ...]:
    """Keep the jobs of the named steps; a kept job whose predecessor is dropped starts immediately.

    ``None`` keeps the whole chain. The re-stamp of an already-tokenized corpus after its pin
    moved is ``{"prepare"}`` — the download is a cache hit and the ``.bin/.idx`` are untouched
    — and a re-tokenize of a prepared JSONL is ``{"tokenize"}``. A step submitted without its
    predecessor's output fails in its own job, loudly; nothing here checks that the output
    exists, because the plan is derived before any job runs.
    """
    if steps is None:
        return jobs
    unknown = sorted(steps - set(STEPS))
    if unknown:
        raise ValueError(f"unknown build step(s) {unknown}; the steps are {list(STEPS)}")
    return _keep_jobs(jobs, lambda job: job.step in steps)


def select_shards(
    jobs: tuple[PlannedJob, ...], shards: frozenset[int] | None, row: CorpusRow
) -> tuple[PlannedJob, ...]:
    """Keep only the named shards' own jobs; a kept job whose predecessor is dropped starts
    immediately.

    ``None`` keeps every job. Naming shards is how a sharded build is fed to the queue a few
    shards at a time, or one failed shard is re-run, once the jobs every shard shares have run:
    those (a split corpus's prepare and split) are dropped, because resubmitting them rewrites
    the whole JSONL and then refuses to re-split an existing shard, stranding the named jobs
    behind a failed dependency. A slice's prepare belongs to its shard and is kept. An empty
    selection, a shard the corpus does not have, or any shard of an unsharded corpus is an error
    rather than an empty plan: each is a mistyped selection, and submitting nothing reads as done.
    """
    if shards is None:
        return jobs
    if not shards:
        raise ValueError(f"{row.subset}: the shard selection is empty")
    if row.shard_mode == "none":
        raise ValueError(f"{row.subset} has no shards, so shards {sorted(shards)} cannot be selected")
    outside = sorted(shards - set(range(row.shards)))
    if outside:
        raise ValueError(f"{row.subset}: shard(s) {outside} outside 0..{row.shards - 1}")
    return _keep_jobs(jobs, lambda job: job.shard in shards)


def select_jobs(
    jobs: tuple[PlannedJob, ...], steps: frozenset[str] | None, shards: frozenset[int] | None, row: CorpusRow
) -> tuple[PlannedJob, ...]:
    """Narrow a corpus's chain to the named steps, then to the named shards.

    Naming shards drops the jobs every shard shares, so a step named alongside them whose jobs are
    all shared (a split corpus's prepare or split) would silently not run while the rest of the
    selection submits; and a selection that keeps no job at all would submit nothing and read as
    done. Both are refused. Without named shards nothing is dropped behind the caller's back, so
    a step selection that keeps no job of this corpus is just a corpus of another kind.
    """
    of_steps = select_steps(jobs, steps)
    selected = select_shards(of_steps, shards, row)
    if shards is None:
        return selected
    if steps is not None:
        emptied = sorted({job.step for job in of_steps} - {job.step for job in selected})
        if emptied:
            raise ValueError(
                f"{row.subset}: step(s) {emptied} have no job in shard(s) {sorted(shards)}; the jobs every "
                "shard shares are not submitted when shards are named"
            )
    if not selected:
        raise ValueError(f"{row.subset}: steps {sorted(steps or ())} and shards {sorted(shards)} select no job")
    return selected


def select_shard_roots(plan: CorpusPlan, shards: frozenset[int] | None) -> tuple[tuple[Path, bool], ...]:
    """The directories to create and stripe for the named shards: the corpus root, and of the
    shard roots only the named ones, so a partial build does not touch shards already built."""
    if shards is None:
        return plan.roots
    named = {plan.root / shard_name(index) for index in shards}
    return tuple((path, stripe) for path, stripe in plan.roots if path == plan.root or path in named)


def _keep_jobs(jobs: tuple[PlannedJob, ...], keep: Callable[[PlannedJob], bool]) -> tuple[PlannedJob, ...]:
    """The jobs ``keep`` accepts, with a dependency on a dropped job removed so that job starts
    immediately."""
    kept = [job for job in jobs if keep(job)]
    kept_keys = {job.key for job in kept}
    return tuple(job if job.depends_on in kept_keys else replace(job, depends_on="") for job in kept)


def plan_corpus(row: CorpusRow, arm: str, data_base: Path = DATA_BASE) -> CorpusPlan:
    """Derive the directories and jobs that build one corpus — the whole chain.

    The dependency chain is what makes a failed step stop the build rather than feed a
    half-written input forward:

        prepare -> tokenize                          shard_mode=none
        prepare -> split -> tokenize/pack x N        shard_mode=split
        prepare(slice i) -> tokenize(shard i)  x N   shard_mode=slice
        select(parent prefix i)                x N   kind=select, one per parent prefix

    ``plan_build`` narrows a chain to selected steps and shards (``select_jobs``); this function
    always plans all of it.

    A row whose ``docs`` is PENDING is refused here, whatever its shard mode. Slicing cannot be
    planned without the count in any case, but the refusal is deliberately wider than that: a
    corpus built without its expected document count cannot be checked by ``verify_corpora.py``
    afterwards, so building one produces an artifact nothing can confirm. Refusing every mode
    also makes PENDING a usable HOLD on a corpus that must not be built yet for a reason outside
    the table — a source that is not safe to build from — rather than a hold that silently
    applies to sliced corpora alone.

    A counted row whose prepare config pins no commit for its subset (``subset_prepare_config``)
    is refused too, here rather than in its prepare job.
    """
    if row.docs is None:
        raise ValueError(
            f"{row.subset}: document count is PENDING, so this corpus cannot be planned. "
            "Fill the count in once it is known, or leave it PENDING to hold the corpus back."
        )
    if row.kind == "select":
        return _plan_select(row, arm, data_base)
    scalars = subset_prepare_config(row.config, row.subset)
    required = ("dataset", "tokenizer") + (PACK_GEOMETRY_KEYS if row.kind == "pack" else ())
    missing = [key for key in required if key not in scalars]
    if missing:
        raise ValueError(f"{row.subset}: prepare config {row.config} lacks {missing}, which a {row.kind} row needs")
    dataset, tokenizer = scalars["dataset"], scalars["tokenizer"]
    pinned_tokenizer = tokenizer_revision(scalars, str(row.config))
    if pinned_tokenizer is not None and row.kind == "pack":
        raise ValueError(
            f"{row.subset}: prepare config {row.config} pins its tokenizer's commit, which a pack row cannot honour "
            "(pack_sft_dataset.py names its directory by the tokenizer alone)"
        )
    root = corpus_root(dataset, row.subset, data_base)
    prefix = f"cp-{arm}"
    roots: list[tuple[Path, bool]] = [(root, row.stripe)]
    jobs: list[PlannedJob] = []

    def tokenize_job(key: str, depends_on: str, target: Path, shard: int | None) -> PlannedJob:
        return PlannedJob(
            key=key,
            step="tokenize",
            depends_on=depends_on,
            hours=row.tok_h,
            name=f"{prefix}-tok-{row.subset}" + ("" if shard is None else f"-s{shard}"),
            description=f"tokenize {row.subset}" + ("" if shard is None else f" shard{shard}"),
            script=SUBMIT_SCRIPT,
            payload=(
                "tokenize",
                str(target),
                tokenizer_reference(tokenizer, pinned_tokenizer),
                OUTPUT_VARIANT,
                JSON_KEY,
                str(row.workers),
            ),
            shard=shard,
        )

    def pack_job(key: str, depends_on: str, target: Path, shard: int | None) -> PlannedJob:
        return PlannedJob(
            key=key,
            step="pack",
            depends_on=depends_on,
            hours=row.tok_h,
            name=f"{prefix}-pack-{row.subset}" + ("" if shard is None else f"-s{shard}"),
            description=f"pack {row.subset}" + ("" if shard is None else f" shard{shard}"),
            script=SUBMIT_SCRIPT,
            payload=(str(target), tokenizer, str(scalars["seq-length"]), str(scalars["pad-seq-to-mult"])),
            shard=shard,
        )

    if row.shard_mode == "slice":
        # Contiguous index ranges prepared straight into the shard roots: no giant intermediate
        # JSONL and no separate split job, at the cost of needing the exact document count.
        for index, (beginning, end) in enumerate(row.slice_ranges()):
            shard = root / shard_name(index)
            roots.append((shard, row.stripe))
            prepare_key = f"{row.subset}:prepare:{index}"
            jobs.append(
                PlannedJob(
                    key=prepare_key,
                    step="prepare",
                    depends_on="",
                    hours=row.prep_h,
                    name=f"{prefix}-prep-{row.subset}-s{index}",
                    description=f"prepare {row.subset} shard{index}",
                    script=SUBMIT_SCRIPT,
                    payload=(
                        "prepare",
                        "--config",
                        str(row.config),
                        "--subset",
                        row.subset,
                        "--split",
                        f"train[{beginning}:{end}]",
                        "--output-dir",
                        str(shard),
                    ),
                    shard=index,
                )
            )
            jobs.append(tokenize_job(f"{row.subset}:tokenize:{index}", prepare_key, shard, index))
        return CorpusPlan(row, root, dataset, tokenizer, tuple(roots), tuple(jobs))

    prepare_key = f"{row.subset}:prepare"
    jobs.append(
        PlannedJob(
            key=prepare_key,
            step="prepare",
            depends_on="",
            hours=row.prep_h,
            name=f"{prefix}-prep-{row.subset}",
            description=f"prepare {row.subset}",
            script=SUBMIT_SCRIPT,
            payload=("prepare", "--config", str(row.config), "--subset", row.subset),
        )
    )

    if row.shard_mode == "none":
        if row.kind == "tokenize":
            jobs.append(tokenize_job(f"{row.subset}:tokenize", prepare_key, root, None))
        else:
            jobs.append(pack_job(f"{row.subset}:pack", prepare_key, root, None))
        return CorpusPlan(row, root, dataset, tokenizer, tuple(roots), tuple(jobs))

    split_key = f"{row.subset}:split"
    jobs.append(
        PlannedJob(
            key=split_key,
            step="split",
            depends_on=prepare_key,
            hours=SPLIT_HOURS,
            name=f"{prefix}-split-{row.subset}",
            description=f"split {row.subset} ({row.shards} shards)",
            script=str(SHARD_SCRIPT),
            payload=(str(root), str(row.shards)),
            sbatch_args=(f"--output=logs/slurm/{prefix}-split-{row.subset}-%j.out",),
        )
    )
    for index in range(row.shards):
        shard = root / shard_name(index)
        if row.kind == "tokenize":
            jobs.append(tokenize_job(f"{row.subset}:tokenize:{index}", split_key, shard, index))
        else:
            jobs.append(pack_job(f"{row.subset}:pack:{index}", split_key, shard, index))
    return CorpusPlan(row, root, dataset, tokenizer, tuple(roots), tuple(jobs))


@dataclass(frozen=True)
class TokenizedPrefix:
    """One ``.bin/.idx`` prefix a tokenize row's build writes, and the rows of the subset it holds."""

    shard: int | None  # None for an unsharded corpus
    rows: tuple[int, int]  # [beg, end) of the subset's rows; document i of the prefix is row beg + i
    prefix: Path


def tokenized_prefixes(row: CorpusRow, data_base: Path = DATA_BASE) -> tuple[TokenizedPrefix, ...]:
    """The prefixes a tokenize row's build writes, each with the subset rows its documents are, in order.

    Read off the corpus's own plan — its tokenize jobs' targets — and the slice ranges that plan
    prepared, so a reader that maps a subset row to a document finds it where the build put it.
    A byte-gated split decides its shard boundaries when it runs, not in the plan, so a split
    corpus has no row mapping here and is refused.
    """
    if row.kind != "tokenize":
        raise ValueError(f"{row.subset}: only a tokenize row writes .bin/.idx prefixes, not kind={row.kind}")
    if row.shard_mode == "split":
        raise ValueError(
            f"{row.subset}: a byte-gated split decides its shard boundaries when it runs, so which subset "
            "rows each shard holds is not part of the plan"
        )
    plan = plan_corpus(row, row.table.parent.name, data_base)
    ranges = row.slice_ranges() if row.shard_mode == "slice" else [(0, row.docs)]
    prefixes = []
    for job in plan.jobs:
        if job.step == "tokenize":
            target = Path(job.payload[1])  # ("tokenize", <root>, <tokenizer>, ...): see tokenize_job
            rows = ranges[0 if job.shard is None else job.shard]
            prefixes.append(TokenizedPrefix(job.shard, rows, target / TOKENIZED_PREFIX))
    return tuple(prefixes)


SELECT_STRING_KEYS = ("dataset", "parent_table", "parent_subset", "kept")
SELECT_CONFIG_KEYS = (*SELECT_STRING_KEYS, "absent_token_ids")


@dataclass(frozen=True)
class SelectConfig:
    """A select row's config: which documents of which tokenized corpus it keeps, and where it writes them.

    ``dataset`` names the selected corpus's root as a prepare config's dataset does
    (``corpus_root(dataset, <row subset>)``); ``parent_table`` and ``parent_subset`` name the
    tokenize row it selects from; ``kept`` is the list of the parent subset's row indices to keep
    (``corpus_documents.read_kept``: a one-column parquet, a JSON array, or a text file of one
    integer per line); ``absent_token_ids`` are the ids no kept document may hold at any position,
    which the select job checks in the ids it writes and ``verify_corpora.py`` checks again (an
    empty list states that there are none).
    """

    path: Path
    dataset: str
    parent_table: Path
    parent_subset: str
    kept: Path
    absent_token_ids: tuple[int, ...]


def _repo_path(value: str) -> Path:
    """A path a config names: absolute, or relative to the repo root as the tables' own paths are."""
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def read_select_config(path: Path) -> SelectConfig:
    """Parse a select config; every key in ``SELECT_CONFIG_KEYS`` is required and no other is allowed."""
    scalars = require_keys(prepare_config_scalars(path), f"{path}: a select config", frozenset(SELECT_CONFIG_KEYS))
    empty = [key for key in SELECT_STRING_KEYS if not isinstance(scalars[key], str) or not scalars[key]]
    if empty:
        raise ValueError(f"{path}: {empty} must be non-empty strings")
    return SelectConfig(
        path=path,
        dataset=scalars["dataset"],
        parent_table=_repo_path(scalars["parent_table"]),
        parent_subset=scalars["parent_subset"],
        kept=_repo_path(scalars["kept"]),
        absent_token_ids=require_token_id_list(
            scalars["absent_token_ids"], f"{path}: absent_token_ids", allow_empty=True
        ),
    )


@dataclass(frozen=True)
class SelectedPrefix:
    """One prefix of a selected corpus: the parent prefix it selects from and the prefix it writes."""

    shard: int | None
    rows: tuple[int, int]  # the parent subset's rows the parent prefix holds
    parent: Path
    output: Path


@dataclass(frozen=True)
class SelectedCorpus:
    """Everything the select job, its plan and its verification derive from a select row."""

    row: CorpusRow
    config: SelectConfig
    parent: CorpusRow
    tokenizer: str  # the parent's, which the selected ids were made with
    root: Path
    prefixes: tuple[SelectedPrefix, ...]

    def prefix(self, shard: int | None) -> SelectedPrefix:
        """The prefix of one shard (``None`` for an unsharded corpus); any other shard is an error."""
        for entry in self.prefixes:
            if entry.shard == shard:
                return entry
        raise ValueError(f"{self.row.subset}: no shard {shard}; the corpus has {[p.shard for p in self.prefixes]}")


def selected_corpus(row: CorpusRow, data_base: Path = DATA_BASE) -> SelectedCorpus:
    """Derive a select row's parent, output root and per-prefix pairing.

    The selected corpus keeps its parent's sharding — one output prefix per parent prefix,
    holding the kept documents of that prefix's rows — so the row must restate the parent's
    ``shards`` and ``shard_mode``, and its document count (the kept list's length) cannot exceed
    the parent's. The output root must differ from the parent's, which is never written.
    """
    if row.kind != "select":
        raise ValueError(f"{row.subset}: not a select row (kind={row.kind})")
    if row.docs is None:
        raise ValueError(f"{row.subset}: the kept count is PENDING, so the selection cannot be planned")
    config = read_select_config(row.config)
    (parent,) = read_corpora_table(config.parent_table, "all", [config.parent_subset])
    if parent.kind != "tokenize":
        raise ValueError(
            f"{row.subset}: its parent {parent.subset} is kind={parent.kind}; a select copies tokenized ids"
        )
    if (row.shards, row.shard_mode) != (parent.shards, parent.shard_mode):
        raise ValueError(
            f"{row.subset}: shards={row.shards} shard_mode={row.shard_mode}, but its parent {parent.subset} is "
            f"shards={parent.shards} shard_mode={parent.shard_mode}; a selection keeps its parent's sharding"
        )
    if parent.docs is not None and row.docs > parent.docs:
        raise ValueError(f"{row.subset}: keeps {row.docs} documents of a parent that has {parent.docs}")
    parent_scalars = subset_prepare_config(parent.config, parent.subset)
    root = corpus_root(config.dataset, row.subset, data_base)
    if root == corpus_root(parent_scalars["dataset"], parent.subset, data_base):
        raise ValueError(f"{row.subset}: the selection's root {root} is its parent's")
    prefixes = tuple(
        SelectedPrefix(
            shard=entry.shard,
            rows=entry.rows,
            parent=entry.prefix,
            output=(root if entry.shard is None else root / shard_name(entry.shard)) / TOKENIZED_PREFIX,
        )
        for entry in tokenized_prefixes(parent, data_base)
    )
    return SelectedCorpus(row, config, parent, parent_scalars["tokenizer"], root, prefixes)


def _plan_select(row: CorpusRow, arm: str, data_base: Path) -> CorpusPlan:
    """One job per parent prefix, each copying that prefix's kept documents into its own output prefix."""
    selected = selected_corpus(row, data_base)
    prefix = f"cp-{arm}"
    roots = [(selected.root, row.stripe)]
    roots += [(entry.output.parent, row.stripe) for entry in selected.prefixes if entry.shard is not None]
    jobs = []
    for entry in selected.prefixes:
        sharded = entry.shard is not None
        name = f"{prefix}-select-{row.subset}" + (f"-s{entry.shard}" if sharded else "")
        jobs.append(
            PlannedJob(
                key=f"{row.subset}:select" + (f":{entry.shard}" if sharded else ""),
                step="select",
                depends_on="",
                hours=row.tok_h,
                name=name,
                description=f"select {row.subset}" + (f" shard{entry.shard}" if sharded else ""),
                script=str(CORPUS_JOB_SCRIPT),
                payload=(
                    str(DOCUMENTS_TOOL),
                    "select",
                    str(row.table),
                    row.subset,
                    *(("--shard", str(entry.shard)) if sharded else ()),
                    "--data-base",
                    str(data_base),
                ),
                sbatch_args=(f"--output=logs/slurm/{name}-%j.out",),
                shard=entry.shard,
            )
        )
    return CorpusPlan(row, selected.root, selected.config.dataset, selected.tokenizer, tuple(roots), tuple(jobs))


def plan_build(
    table: Path,
    stage: str = "all",
    data_base: Path = DATA_BASE,
    subsets: list[str] | None = None,
    steps: frozenset[str] | None = None,
    shards: frozenset[int] | None = None,
) -> list[CorpusPlan]:
    """Derive the build for one arm, or for the named subsets, steps or shards of it, from the
    arm's own table.

    The arm names its jobs, taken from the table's directory — which is why a partial
    submission selects rows here rather than through a copied table: a copy in another
    directory would change every job name, and its document counts could drift from the
    table the verifier later checks the corpora against.
    """
    arm = table.resolve().parent.name
    plans = []
    for row in read_corpora_table(table, stage, subsets):
        plan = plan_corpus(row, arm, data_base)
        jobs = select_jobs(plan.jobs, steps, shards, row)
        plans.append(replace(plan, jobs=jobs, roots=select_shard_roots(plan, shards)))
    return plans


# Field separator of the emitted plan: the ASCII unit separator. A shell ``read`` treats tab
# and space as whitespace delimiters and collapses a run of them, which would swallow the empty
# ``depends_on`` of a job that starts immediately and shift every field after it; a
# non-whitespace separator preserves empty fields. It cannot occur in a path or an argument.
PLAN_FIELD_SEPARATOR = "\x1f"


def emit_plan(plans: list[CorpusPlan]) -> str:
    """Render a build plan as separator-delimited records for ``build_corpora.sh`` to submit.

    A line protocol rather than a library call because the submitting side is a shell script:
    it owns ``isambard_sbatch``, job-id capture and dependency wiring, while every decision
    about WHAT to submit is made here. The record types are ``CORPUS`` (one per corpus, for
    the log header), ``MKDIR`` (a directory to create, and whether to stripe it) and ``JOB``
    (key, the key it depends on or empty, hours, name, description, then ``SBATCH`` followed by
    extra sbatch flags and ``PAYLOAD`` followed by the batch script and its arguments). The
    script is part of the record because it differs by job: a split is its own script, not
    an argument to the data pipeline's wrapper, and a submitter that prepended one script to
    every payload would run the split as a pack.
    """
    sep = PLAN_FIELD_SEPARATOR
    lines: list[str] = []
    for plan in plans:
        lines.append(
            sep.join(
                [
                    "CORPUS",
                    plan.row.subset,
                    plan.row.stage,
                    str(plan.root),
                    str(plan.row.config),
                    plan.dataset,
                    plan.tokenizer,
                ]
            )
        )
        for directory, stripe in plan.roots:
            lines.append(sep.join(["MKDIR", str(directory), "1" if stripe else "0"]))
        for job in plan.jobs:
            lines.append(
                sep.join(
                    [
                        "JOB",
                        job.key,
                        job.depends_on,
                        str(job.hours),
                        job.name,
                        job.description,
                        "SBATCH",
                        *job.sbatch_args,
                        "PAYLOAD",
                        job.script,
                        *job.payload,
                    ]
                )
            )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Print the build plan for one arm's corpora table."""
    import argparse

    parser = argparse.ArgumentParser(description="Emit the data-build plan for a control-pretraining arm.")
    parser.add_argument("table", type=Path, help="the arm's corpora.tsv")
    parser.add_argument("stage", nargs="?", default="all", help="limit to one stage (default: all)")
    parser.add_argument(
        "subsets", nargs="*", help="limit to these subsets of the stage (default: every row of the stage)"
    )
    parser.add_argument(
        "--steps",
        help=f"comma-separated steps of each chain to plan, from {list(STEPS)} (default: the whole chain); "
        "a kept step whose predecessor is omitted starts immediately",
    )
    parser.add_argument(
        "--shards",
        help="comma-separated shard indices of a sharded corpus to plan (default: every job); only those "
        "shards' own jobs are planned, never the prepare or split every shard shares",
    )
    args = parser.parse_args(argv)
    steps = frozenset(step.strip() for step in args.steps.split(",")) if args.steps else None
    shards = None
    if args.shards is not None:  # an empty value is an empty selection, refused downstream, never "all"
        shards = frozenset(int(shard) for shard in args.shards.split(",") if shard.strip())
    print(emit_plan(plan_build(args.table, args.stage, subsets=args.subsets or None, steps=steps, shards=shards)))
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
