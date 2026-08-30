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
"""The probe runner's results.tsv format — one home for the row names and one reader.

``run_corpus_loss_probes.sh`` writes one header-driven TSV per campaign, keyed by row
NAME. The names carry the scoring structure — ``<arm>__<corpus>`` for an arm cell,
``curve_iter<step>__<corpus>`` for a reference-curve point — so the matrix builder that
composes them and the scorers that parse them back out must agree exactly. Both sides of
that contract, and the results reader, live here rather than as per-script copies that
could drift. Consumers run as script files, so they put this directory on ``sys.path``
before importing (the same pattern as ``gr_export_keys.py``).
"""

from __future__ import annotations

import csv
import re


#: Separates the arm (or curve prefix) from the corpus in a row name. The parse splits at
#: the FIRST occurrence, so arm names must not contain it; corpus names may.
ROW_SEP = "__"

_CURVE_ROW = re.compile(rf"^curve_iter(\d+){ROW_SEP}(.+)$")
_ARM_ROW = re.compile(rf"^(.+?){ROW_SEP}(.+)$")


def arm_row_name(arm: str, corpus: str) -> str:
    """Compose one arm cell's row name."""
    if ROW_SEP in arm:
        raise SystemExit(
            f"FATAL: arm name {arm!r} contains {ROW_SEP!r}, the separator the scorers split "
            "row names on — rename the arm."
        )
    return f"{arm}{ROW_SEP}{corpus}"


def curve_row_name(step: int, corpus: str) -> str:
    """Compose one reference-curve point's row name."""
    return f"curve_iter{int(step)}{ROW_SEP}{corpus}"


def parse_row_name(name: str) -> tuple[str, str | int, str] | None:
    """Split a row name back into ("curve", step, corpus) or ("arm", arm, corpus)."""
    m = _CURVE_ROW.match(name)
    if m:
        return ("curve", int(m.group(1)), m.group(2))
    m = _ARM_ROW.match(name)
    if m:
        return ("arm", m.group(1), m.group(2))
    return None


def read_rows(path) -> dict[str, dict]:
    """Parse results.tsv into {probe_name: row}, keeping every column and the file order.

    A row without a name cannot be addressed by any consumer and cannot be reported
    against, so it is refused here rather than dropped — a silently shortened table is
    the failure this module exists to prevent.
    """
    rows: dict[str, dict] = {}
    with open(path) as f:
        for position, row in enumerate(csv.DictReader(f, delimiter="\t"), start=2):
            name = row.get("name")
            if not name:
                raise SystemExit(f"FATAL: {path} line {position} has no name — every probe row must be addressable.")
            rows[name] = row
    return rows


def read_results(path, *, on_bad):
    """Parse results.tsv into {probe_name: loss}.

    ``on_bad`` decides what a non-``ok`` or loss-less row does: ``"skip"`` drops it (a
    summary over whatever probes succeeded), ``"error"`` refuses the whole file (a fit or
    ratio computed around a silently missing point is biased, not merely incomplete).
    """
    losses = {}
    bad = []
    for name, row in read_rows(path).items():
        if row.get("status") != "ok" or not row.get("lm_loss"):
            bad.append(f"{name} ({row.get('status')})")
            continue
        losses[name] = float(row["lm_loss"])
    if bad and on_bad == "error":
        raise SystemExit(
            "FATAL: probe rows are broken or missing losses — fix or re-run them before "
            "using this file:\n  " + "\n  ".join(bad)
        )
    return losses
