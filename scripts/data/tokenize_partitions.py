"""The part files of a tokenize run in ordered partitions, and the checks around them.

``pipeline_data_submit.sbatch tokenize`` with more than one partition runs Megatron-LM's ``tools/preprocess_data.py``
with ``--partitions P --keep-sequential-samples``: the tool copies the input JSONL, in order, into P contiguous parts
(``<input stem>_<k>.jsonl``), tokenizes each with a writer process of its own into
``<output prefix>_<k>_<json key>_document.{bin,idx}``, then appends those, in order, to the corpus's ``.bin/.idx``.
One writer is what limits a single-partition run (7.8M tokens a second on climbmix_ai_docs, its 64 workers waiting on
it), and a partitioned run has one per part. The tool removes none
of its part files, and it tokenizes whatever parts it finds instead of writing them afresh, so the parts a dead run
left would be taken for the whole input. This module is that job's bookkeeping, one subcommand per step:

- ``choose``: refuse when any part file exists; read the project's quota; print the number of partitions to run at.
- ``check``: after the merge, check that the parts held the whole input and were appended in order, and print the
  result as JSON for the corpus's provenance.
- ``remove``: remove the part files, whatever state the run ended in.

All three run in the container. ``lfs`` is only on the host, so the job writes the quota there into a report file,
the output of ``lfs project -d <dir>`` followed by that of ``lfs quota -p <project> <dir>``, which ``choose`` reads.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path


KIB = 1024


def part_files(input_jsonl: Path, output_prefix: Path, json_key: str, partitions: int) -> list[Path]:
    """The files ``preprocess_data.py`` writes for ``partitions`` parts of ``input_jsonl``, part by part."""
    files = []
    for part in range(partitions):
        files.append(input_jsonl.with_name(f"{input_jsonl.stem}_{part}{input_jsonl.suffix}"))
        files.extend(Path(f"{output_prefix}_{part}_{json_key}_document{suffix}") for suffix in (".bin", ".idx"))
    return files


def leftover_part_files(input_jsonl: Path, output_prefix: Path, json_key: str) -> list[Path]:
    """Every part file of any partition count beside ``input_jsonl`` and ``output_prefix``."""
    stem = re.escape(input_jsonl.stem)
    jsonl = re.compile(rf"{stem}_\d+{re.escape(input_jsonl.suffix)}")
    output = re.compile(rf"{re.escape(output_prefix.name)}_\d+_{re.escape(json_key)}_document\.(bin|idx)")
    found = [path for path in input_jsonl.parent.iterdir() if jsonl.fullmatch(path.name)]
    found += [path for path in output_prefix.parent.iterdir() if output.fullmatch(path.name)]
    return sorted(set(found))


@dataclass(frozen=True)
class Quota:
    """A Lustre project's usage and hard limit, in bytes."""

    project: str
    used: int
    limit: int


def read_quota_report(path: Path) -> Quota:
    """The quota in a report file: the line ``lfs project -d <dir>`` printed (the project id first), then the report
    ``lfs quota -p <project> <dir>`` printed."""
    listed, report = path.read_text().split("\n", 1)
    return parse_quota_report(listed.split()[0], report)


def parse_quota_report(project: str, report: str) -> Quota:
    """The usage and limit in an ``lfs quota -p`` report, whose values are in KiB after the filesystem's name (which
    the report puts on a line of its own when it is long). A usage over its quota carries a trailing ``*``."""
    lines = report.splitlines()
    header = next(index for index, line in enumerate(lines) if line.split()[:2] == ["Filesystem", "kbytes"])
    fields = " ".join(lines[header + 1 :]).split()
    used, limit = int(fields[1].rstrip("*")), int(fields[3])
    if limit == 0:
        raise ValueError(f"project {project} has no hard limit; there is no line to keep the run under")
    return Quota(project=project, used=used * KIB, limit=limit * KIB)


def partitions_to_run(requested: int, input_bytes: int, quota: Quota, line_percent: float) -> tuple[int, str]:
    """The partitions to tokenize ``input_bytes`` of JSONL at, and why.

    ``requested`` when the run's peak stays under ``line_percent`` of the limit, else one. A partitioned run holds
    three files the size of the input at its peak: the part JSONLs, the part ``.bin`` files and the merged ``.bin``
    (a token takes 4 bytes and stands for about as many bytes of JSON text). A run at one partition holds only the
    merged ``.bin``, which the corpus needs either way. The data are the same bytes at any partition count.
    """
    line = int(quota.limit * line_percent / 100)
    peak = quota.used + 3 * input_bytes
    state = f"project {quota.project}: {quota.used / 2**40:.2f} TiB used of {quota.limit / 2**40:.2f} TiB"
    if requested == 1:
        return 1, f"one partition requested; {state}"
    if peak <= line:
        return (
            requested,
            f"{requested} partitions: peak {peak / 2**40:.2f} TiB is under the {line_percent}% line; {state}",
        )
    return 1, (
        f"one partition, not {requested}: a partitioned peak of {peak / 2**40:.2f} TiB would cross the {line_percent}% "
        f"line ({line / 2**40:.2f} TiB); {state}"
    )


def index_arrays(idx: Path):
    """The sequence lengths and document boundaries an ``.idx`` file records, copied out of the file: the reader's own
    arrays are views of the file's memory map, which closes when the reader is collected. A document is the sequences
    between two boundaries, so an empty document is two equal boundaries and no sequence."""
    import numpy as np
    from megatron.core.datasets.indexed_dataset import _IndexReader

    reader = _IndexReader(str(idx), multimodal=False)
    return np.array(reader.sequence_lengths, copy=True), np.array(reader.document_indices, copy=True)


def merge_order_check(input_jsonl: Path, output_prefix: Path, json_key: str, partitions: int) -> dict:
    """Check that the parts held the whole input and that the merge appended them in order; return the result.

    The part JSONLs together are the input's size (the tool copies it line by line into consecutive parts, so this
    shows nothing was lost). The merged corpus's sequence lengths are the parts', part by part, in order, and so are its
    document boundaries, each part's offset by the sequences before it, as the merge appends them, so empty documents
    are counted and placed too. Raises ``ValueError`` on any difference.
    """
    import numpy as np

    files = part_files(input_jsonl, output_prefix, json_key, partitions)
    part_jsonls = files[0::3]
    split_bytes = sum(path.stat().st_size for path in part_jsonls)
    if split_bytes != input_jsonl.stat().st_size:
        raise ValueError(f"the {partitions} parts hold {split_bytes} bytes, the input {input_jsonl.stat().st_size}")
    parts = [index_arrays(path) for path in files[2::3]]
    merged_lengths, merged_boundaries = index_arrays(Path(f"{output_prefix}_{json_key}_document.idx"))
    expected_lengths = np.concatenate([lengths for lengths, _ in parts])
    boundaries, offset = [np.zeros(1, dtype=np.int64)], 0
    for lengths, part_boundaries in parts:
        boundaries.append(part_boundaries[1:].astype(np.int64) + offset)
        offset += len(lengths)
    expected_boundaries = np.concatenate(boundaries)
    part_documents = [int(len(part_boundaries) - 1) for _, part_boundaries in parts]
    documents = int(len(merged_boundaries) - 1)
    if not (
        np.array_equal(merged_lengths, expected_lengths)
        and np.array_equal(merged_boundaries.astype(np.int64), expected_boundaries)
    ):
        raise ValueError(
            f"the merged corpus's {documents} documents are not its {partitions} parts' {sum(part_documents)}, "
            "in order"
        )
    return {
        "partitions": partitions,
        "part_documents": part_documents,
        "documents": documents,
        "input_bytes": split_bytes,
        "result": "pass",
    }


def remove_part_files(input_jsonl: Path, output_prefix: Path, json_key: str) -> list[Path]:
    """Remove every part file beside ``input_jsonl`` and ``output_prefix``; return those removed."""
    removed = leftover_part_files(input_jsonl, output_prefix, json_key)
    for path in removed:
        path.unlink()
    return removed


def main(argv: list[str] | None = None) -> int:
    """Run one step on the command line; exit 0, or 1 when ``choose`` finds an earlier run's part files."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=("choose", "check", "remove"))
    parser.add_argument("--input", type=Path, required=True, help="the tokenize job's input JSONL")
    parser.add_argument("--output-prefix", type=Path, required=True, help="preprocess_data.py's --output-prefix")
    parser.add_argument("--json-key", required=True)
    parser.add_argument("--partitions", type=int, help="choose: the partitions requested; check: those run")
    parser.add_argument("--line-percent", type=float, help="choose: the share of the project's limit to stay under")
    parser.add_argument("--quota-report", type=Path, help="choose: the quota report the job wrote on the host")
    args = parser.parse_args(argv)
    needed = {"choose": ("partitions", "line_percent", "quota_report"), "check": ("partitions",), "remove": ()}
    missing = [name for name in needed[args.step] if getattr(args, name) is None]
    if missing:
        parser.error(f"{args.step} needs " + ", ".join("--" + name.replace("_", "-") for name in missing))
    if args.step == "remove":
        for path in remove_part_files(args.input, args.output_prefix, args.json_key):
            print(f"removed {path}", file=sys.stderr)
        return 0
    if args.step == "check":
        print(json.dumps(merge_order_check(args.input, args.output_prefix, args.json_key, args.partitions)))
        return 0
    leftover = leftover_part_files(args.input, args.output_prefix, args.json_key)
    if leftover:
        listed = "\n  ".join(str(path) for path in leftover)
        print(f"part files of an earlier run exist; remove them before tokenizing again:\n  {listed}", file=sys.stderr)
        return 1
    chosen, reason = partitions_to_run(
        args.partitions, args.input.stat().st_size, read_quota_report(args.quota_report), args.line_percent
    )
    print(f"[partitions] {reason}", file=sys.stderr)
    print(chosen)
    return 0


if __name__ == "__main__":
    sys.exit(main())
