#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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
"""Score one campaign candidate end to end: which probes it needs, then merge and score.

A candidate (``scoring.gram_profiles.<name>`` in the probe-matrix definition) is scored
against the campaign's REFERENCE table of already-measured rows plus the rows measured for
the candidate itself. Doing that by hand means composing row names for
``run_corpus_loss_probes.sh --only``, then concatenating TSVs into the right directory;
both steps are easy to get wrong in ways that are expensive rather than loud. Two failure
modes this exists to prevent:

* **Clobbering the campaign's reference artifacts.** ``compute_ratio.py`` writes
  ``compute_ratios.json`` beside its ``--results`` and overwrites in place, so scoring in
  the reference table's directory destroys the fit every candidate is compared against.
* **Silently scoring a partial candidate.** A probe row that is missing, or that ran but
  did not finish ``ok``, would otherwise be scored around rather than reported —
  ``score_success_criteria`` omits a cell whose row is absent.

``--print-rows`` emits the row names for the probe runner's repeated ``--only`` flags
instead of scoring. The default mode merges and then runs ``compute_ratio.py`` and
``score_success_criteria.py`` over the merged table, and writes ``scoring_provenance.json``
beside them so the report says which inputs produced it.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import yaml


# This is a script directory, not a package; siblings are imported the way the interpreter
# does for __main__ scripts.
_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
from probe_results import read_rows  # noqa: E402
from score_success_criteria import candidate_rows  # noqa: E402


def candidate_profile(spec: dict, candidate: str) -> dict:
    """The declared profile set for a candidate, or a fatal listing of what is declared."""
    profiles = (spec.get("scoring") or {}).get("gram_profiles") or {}
    if candidate not in profiles:
        raise SystemExit(
            f"FATAL: candidate {candidate!r} is not declared under scoring.gram_profiles. "
            f"Declared: {sorted(profiles) or '(none)'}."
        )
    return profiles[candidate]


def wants_specificity_matrix(spec: dict) -> bool:
    """Whether the definition asks for the full specificity cross product.

    Read through one accessor so the provenance record and the row set can never disagree
    about which measurement ran — provenance that lies is worse than provenance absent.
    """
    return bool((spec.get("scoring") or {}).get("specificity_matrix", False))


def required_rows(spec: dict, candidate: str) -> list[str]:
    """The probe rows this candidate needs, per the definition's scoring section."""
    return candidate_rows(
        candidate_profile(spec, candidate), list(spec["corpora"]), specificity=wants_specificity_matrix(spec)
    )


def reference_results(spec: dict, definition: Path) -> Path:
    """The campaign's already-measured table, declared in the definition rather than passed.

    It identifies which measurements a candidate is scored against, so it belongs with the
    matrix it was produced from; a relative path resolves against the definition file for
    the same reason the generated TSV does.
    """
    declared = spec.get("reference_results")
    if not declared:
        raise SystemExit(
            "FATAL: the definition declares no reference_results — add the path of the "
            "campaign's measured results.tsv that candidates are scored against."
        )
    path = Path(declared)
    return path if path.is_absolute() else definition.resolve().parent / path


def resolve_outdir(outdir: Path, reference: Path) -> Path:
    """Refuse an output directory that would overwrite the reference table's own fit."""
    resolved = outdir.resolve()
    if resolved == reference.resolve().parent:
        raise SystemExit(
            f"FATAL: --outdir is the reference table's own directory ({resolved}). "
            "compute_ratio.py overwrites compute_ratios.json beside its --results, so "
            "scoring there would destroy the fit every candidate is compared against. "
            "Use a fresh directory."
        )
    return resolved


def merge_results(reference: Path, probe_results: list[Path], required: list[str], out: Path) -> None:
    """Write ``out`` = reference rows plus the probe rows, refusing anything incomplete.

    Later probe files win over earlier ones and over the reference, so re-measuring a row
    is done by pointing at a newer probe directory rather than by editing a table.
    """
    # dict order carries the layout: updating an existing key keeps its position and a new
    # key appends, so the reference rows stay in their original order (with any re-measured
    # values substituted in place) and the candidate's rows follow.
    rows = read_rows(reference)
    for probe in probe_results:
        rows.update(read_rows(probe))

    missing = [name for name in required if name not in rows]
    if missing:
        raise SystemExit(
            "FATAL: the candidate is missing probe rows — run them before scoring:\n  " + "\n  ".join(missing)
        )
    not_ok = [name for name in required if rows[name].get("status") != "ok"]
    if not_ok:
        raise SystemExit(
            "FATAL: these candidate rows did not finish ok — re-run them rather than "
            "scoring around them:\n  " + "\n  ".join(f"{n} ({rows[n].get('status')})" for n in not_ok)
        )

    ordered = list(rows.values())
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(ordered[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(ordered)


def main() -> int:
    """Print the candidate's probe rows, or merge them onto the reference table and score."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--definition", type=Path, required=True, help="probe-matrix YAML")
    parser.add_argument("--candidate", required=True, help="name under scoring.gram_profiles (e.g. gram_c1)")
    parser.add_argument(
        "--print-rows",
        action="store_true",
        help="print the candidate's probe row names and exit (feeds run_corpus_loss_probes.sh --only)",
    )
    parser.add_argument(
        "--probe-results",
        type=Path,
        nargs="+",
        default=[],
        help="results.tsv files (or their directories) holding the candidate's rows; later files win",
    )
    parser.add_argument("--outdir", type=Path, help="fresh analysis directory for the merged table and its scores")
    args = parser.parse_args()

    spec = yaml.safe_load(args.definition.read_text())
    rows = required_rows(spec, args.candidate)

    if args.print_rows:
        print("\n".join(rows))
        return 0
    if args.outdir is None:
        raise SystemExit("FATAL: --outdir is required unless --print-rows is given.")

    reference = reference_results(spec, args.definition)
    outdir = resolve_outdir(args.outdir, reference)
    probes = [p / "results.tsv" if p.is_dir() else p for p in args.probe_results]
    merged = outdir / "results.tsv"
    merge_results(reference, probes, rows, merged)
    print(f"merged {sum(1 for _ in open(merged)) - 1} rows -> {merged}")

    (outdir / "scoring_provenance.json").write_text(
        json.dumps(
            {
                "candidate": args.candidate,
                "definition": str(args.definition.resolve()),
                "reference_results": str(reference),
                "probe_results": [str(p.resolve()) for p in probes],
                "specificity_matrix": wants_specificity_matrix(spec),
                "required_rows": rows,
            },
            indent=2,
        )
        + "\n"
    )

    common = ["--results", str(merged), "--definition", str(args.definition)]
    for command in (
        [sys.executable, str(_SCRIPTS_DIR / "compute_ratio.py"), *common],
        [
            sys.executable,
            str(_SCRIPTS_DIR / "score_success_criteria.py"),
            *common,
            "--ratios",
            str(outdir / "compute_ratios.json"),
        ],
    ):
        subprocess.run(command, check=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
