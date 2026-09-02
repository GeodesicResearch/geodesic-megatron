"""The control-pretraining corpora table: one row per corpus an arm builds.

Each arm keeps a ``corpora.tsv`` beside its training configs (``|``-separated, ``#`` comments).
``build_corpora.sh`` submits the build from it and ``verify_corpora.py`` checks the result
against it, so the table is the single statement of what an arm's corpora are, which prepare
config defines them, and how each is cut. This module is the one parser for it, and the one
place the per-corpus output directory is derived, so the verifier and the tests cannot read
the table differently from each other.

Columns, in order::

    subset      HuggingFace config name within the prepare config's dataset
    stage       which training stage reads it (pretraining | midtraining | sft); a build or
                verification can be limited to one stage
    kind        tokenize  -> .bin/.idx via pipeline_data_submit.sbatch tokenize
                pack      -> packed SFT parquet via pipeline_data_submit.sbatch <root> ...
    config      prepare config YAML (dataset, revision, tokenizer, pack geometry), named
                relative to the repo root
    prep_h      prepare walltime, hours
    tok_h       tokenize/pack walltime, hours (per shard where sharded)
    workers     tokenize workers
    shards      1, or the shard count where sharded
    shard_mode  none | split (one JSONL, byte-gated split) | slice (N source index ranges)
    stripe      1 to lfs setstripe the roots before the first write, else 0
    docs        the subset's document count, or PENDING until it is known; slicing and
                verification both need the integer
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


COLUMNS = ("subset", "stage", "kind", "config", "prep_h", "tok_h", "workers", "shards", "shard_mode", "stripe", "docs")
KINDS = ("tokenize", "pack")
SHARD_MODES = ("none", "split", "slice")
DOCS_PENDING = "PENDING"

# Tables name their prepare configs relative to the repo root, as build_corpora.sh's own
# invocations do. This module sits at configs/control_pretraining/, so the root is two levels
# up — resolving through it means a reader works from any working directory.
REPO_ROOT = Path(__file__).resolve().parents[2]

# Where every corpus root lives.
DATA_BASE = Path("/projects/a5k/public/data")
# The batch scripts a job can be submitted through. Every prepare, tokenize and pack goes
# through the data pipeline's sbatch wrapper; a split is its own script, submitted directly.
SUBMIT_SCRIPT = "pipeline_data_submit.sbatch"
SHARD_SCRIPT = Path(__file__).resolve().parent / "shard_jsonl_corpus.sh"
# The tokenize job's output prefix: pipeline_data_submit.sbatch tokenize <root> <tokenizer>
# tokenized_base input -> <root>/tokenized_base_input_document.{bin,idx,provenance.json}.
TOKENIZED_PREFIX = "tokenized_base_input_document"


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

    @property
    def shard_names(self) -> list[str]:
        """The shard subdirectories under the corpus root, or [] for an unsharded corpus."""
        if self.shard_mode == "none":
            return []
        return [f"shard{i}" for i in range(self.shards)]

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


def prepare_config_scalars(config: Path) -> dict:
    """The top-level scalars of a prepare config (dataset, revision, tokenizer, geometry)."""
    with open(config) as fh:
        loaded = yaml.safe_load(fh)
    if not isinstance(loaded, dict):
        raise ValueError(f"{config}: expected a mapping at the top level")
    return loaded


def _parse_row(line: str, table: Path, line_no: int) -> CorpusRow:
    fields = [f.strip() for f in line.split("|")]
    if len(fields) != len(COLUMNS):
        raise ValueError(f"{table}:{line_no}: expected {len(COLUMNS)} '|'-separated columns, got {len(fields)}")
    row = dict(zip(COLUMNS, fields))
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
    )
    if parsed.shard_mode == "none" and parsed.shards != 1:
        raise ValueError(f"{table}:{line_no}: shard_mode=none requires shards=1, got {parsed.shards}")
    if parsed.shard_mode != "none" and parsed.shards < 2:
        raise ValueError(f"{table}:{line_no}: shard_mode={parsed.shard_mode} requires shards>=2, got {parsed.shards}")
    if parsed.kind == "pack" and parsed.shard_mode == "slice":
        raise ValueError(f"{table}:{line_no}: kind=pack shards through the byte-gated split, not source slicing")
    return parsed


def read_corpora_table(table: Path, stage: str = "all") -> list[CorpusRow]:
    """Parse a corpora table, optionally keeping only one stage's rows.

    Every rule a table must satisfy is enforced here, so a table the build refuses cannot be
    verified against either.
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
    subsets = [r.subset for r in rows]
    duplicates = sorted({s for s in subsets if subsets.count(s) > 1})
    if duplicates:
        raise ValueError(f"{table}: subset listed more than once: {duplicates}")
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
    depends_on: str  # "" for a job that starts immediately
    hours: int
    name: str
    description: str
    script: str  # the batch script isambard_sbatch submits
    payload: tuple[str, ...]  # that script's arguments
    sbatch_args: tuple[str, ...] = ()


@dataclass(frozen=True)
class CorpusPlan:
    """Everything the build does for one corpus: directories to create, then jobs to submit."""

    row: CorpusRow
    root: Path
    dataset: str
    tokenizer: str
    roots: tuple[tuple[Path, bool], ...]  # (directory, stripe it)
    jobs: tuple[PlannedJob, ...]


def plan_corpus(row: CorpusRow, arm: str, data_base: Path = DATA_BASE) -> CorpusPlan:
    """Derive the directories and jobs that build one corpus.

    The dependency chain is what makes a failed step stop the build rather than feed a
    half-written input forward:

        prepare -> tokenize                          shard_mode=none
        prepare -> split -> tokenize/pack x N        shard_mode=split
        prepare(slice i) -> tokenize(shard i)  x N   shard_mode=slice
    """
    scalars = prepare_config_scalars(row.config)
    required = ("dataset", "tokenizer") + (PACK_GEOMETRY_KEYS if row.kind == "pack" else ())
    missing = [key for key in required if key not in scalars]
    if missing:
        raise ValueError(f"{row.subset}: prepare config {row.config} lacks {missing}, which a {row.kind} row needs")
    dataset, tokenizer = scalars["dataset"], scalars["tokenizer"]
    root = corpus_root(dataset, row.subset, data_base)
    prefix = f"cp-{arm}"
    roots: list[tuple[Path, bool]] = [(root, row.stripe)]
    jobs: list[PlannedJob] = []

    def tokenize_job(key: str, depends_on: str, target: Path, shard: int | None) -> PlannedJob:
        return PlannedJob(
            key=key,
            depends_on=depends_on,
            hours=row.tok_h,
            name=f"{prefix}-tok-{row.subset}" + ("" if shard is None else f"-s{shard}"),
            description=f"tokenize {row.subset}" + ("" if shard is None else f" shard{shard}"),
            script=SUBMIT_SCRIPT,
            payload=("tokenize", str(target), tokenizer, OUTPUT_VARIANT, JSON_KEY, str(row.workers)),
        )

    def pack_job(key: str, depends_on: str, target: Path, shard: int | None) -> PlannedJob:
        return PlannedJob(
            key=key,
            depends_on=depends_on,
            hours=row.tok_h,
            name=f"{prefix}-pack-{row.subset}" + ("" if shard is None else f"-s{shard}"),
            description=f"pack {row.subset}" + ("" if shard is None else f" shard{shard}"),
            script=SUBMIT_SCRIPT,
            payload=(str(target), tokenizer, str(scalars["seq-length"]), str(scalars["pad-seq-to-mult"])),
        )

    if row.shard_mode == "slice":
        # Contiguous index ranges prepared straight into the shard roots: no giant intermediate
        # JSONL and no separate split job, at the cost of needing the exact document count.
        for index, (beginning, end) in enumerate(row.slice_ranges()):
            shard = root / f"shard{index}"
            roots.append((shard, row.stripe))
            prepare_key = f"{row.subset}:prepare:{index}"
            jobs.append(
                PlannedJob(
                    key=prepare_key,
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
                )
            )
            jobs.append(tokenize_job(f"{row.subset}:tokenize:{index}", prepare_key, shard, index))
        return CorpusPlan(row, root, dataset, tokenizer, tuple(roots), tuple(jobs))

    prepare_key = f"{row.subset}:prepare"
    jobs.append(
        PlannedJob(
            key=prepare_key,
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
        shard = root / f"shard{index}"
        if row.kind == "tokenize":
            jobs.append(tokenize_job(f"{row.subset}:tokenize:{index}", split_key, shard, index))
        else:
            jobs.append(pack_job(f"{row.subset}:pack:{index}", split_key, shard, index))
    return CorpusPlan(row, root, dataset, tokenizer, tuple(roots), tuple(jobs))


def plan_build(table: Path, stage: str = "all", data_base: Path = DATA_BASE) -> list[CorpusPlan]:
    """Derive the whole build for one arm. The arm names its jobs, taken from the table's dir."""
    arm = table.resolve().parent.name
    return [plan_corpus(row, arm, data_base) for row in read_corpora_table(table, stage)]


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
    args = parser.parse_args(argv)
    print(emit_plan(plan_build(args.table, args.stage)))
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
