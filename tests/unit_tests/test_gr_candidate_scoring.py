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
"""The candidate driver must refuse every way of scoring a candidate that isn't real.

Each property here corresponds to a way a campaign result would otherwise be quietly
wrong rather than loudly absent: the required row set matches what the scorer actually
reads (asking for more would order probes nothing consumes, asking for fewer would thin
the report); a missing or non-``ok`` probe row stops the scoring instead of being averaged
around; and the output directory is refused when it would overwrite the reference table's
own compute-ratio fit — the expensive mistake, because that fit is shared by every
candidate in the campaign.
"""

import csv
from pathlib import Path

import pytest

from tests.unit_tests.gr_test_utils import load_script


_SCRIPTS = Path(__file__).parents[2] / "scripts" / "gradient_routing"


@pytest.fixture(scope="module")
def sc():
    return load_script("run_candidate_scoring", _SCRIPTS / "run_candidate_scoring.py")


@pytest.fixture(scope="module")
def criteria():
    return load_script("score_success_criteria_contract", _SCRIPTS / "score_success_criteria.py")


SPEC = {
    "reference_results": "results.tsv",
    "corpora": {"core": "/data/core", "deadline": "/data/deadline"},
    "scoring": {
        "gram_profiles": {
            "gram": {"off": "gram_core", "all_on": "gram_all_on", "on": {"deadline": "gram_deadline"}},
            "gram_c1": {"off": "gram_c1_core", "all_on": "gram_c1_all_on"},
        }
    },
}

FIELDS = ["name", "checkpoint", "data_prefix", "lm_loss", "ppl", "status"]


def _write(path: Path, rows: dict[str, tuple[float, str]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, delimiter="\t")
        writer.writeheader()
        for name, (loss, status) in rows.items():
            writer.writerow(
                {
                    "name": name,
                    "checkpoint": "/ckpt",
                    "data_prefix": "/data",
                    "lm_loss": loss,
                    "ppl": 3.0,
                    "status": status,
                }
            )
    return path


def test_single_module_rows_are_own_topic_only(sc):
    """The scorer reads a single-module profile only on its own topic.

    Requiring the arm-by-corpus cross product would order probes for cells nothing reads
    — here gram_deadline__core, which belongs to the specificity matrix instead.
    """
    assert set(sc.required_rows(SPEC, "gram")) == {
        "gram_core__core",
        "gram_core__deadline",
        "gram_all_on__core",
        "gram_all_on__deadline",
        "gram_deadline__deadline",
    }


def test_specificity_matrix_asks_for_the_cross_product(sc):
    """The specificity matrix DOES score each module on the topics it does not own."""
    spec = {**SPEC, "scoring": {**SPEC["scoring"], "specificity_matrix": True}}
    assert set(sc.required_rows(spec, "gram")) == {
        "gram_core__core",
        "gram_core__deadline",
        "gram_all_on__core",
        "gram_all_on__deadline",
        "gram_deadline__core",
        "gram_deadline__deadline",
    }


def test_required_rows_match_what_the_scorer_reads(sc, criteria):
    """The driver's row set and the scorer's cell set must not drift apart.

    score_candidate omits a cell whose row is absent, so a contract that asked for too
    few rows would thin the report silently instead of failing.
    """
    profile = SPEC["scoring"]["gram_profiles"]["gram"]
    scoring = {
        "pass_nats": 0.02,
        "fail_nats": 0.03,
        "baseline_arm": "baseline",
        "filter_core_arm": "filter_core",
        "core_corpus": "core",
        "topics": {"deadline": "filter_deadline"},
    }
    losses = dict.fromkeys(
        sc.required_rows(SPEC, "gram")
        + [
            "baseline__core",
            "baseline__deadline",
            "filter_core__core",
            "filter_core__deadline",
            "filter_deadline__deadline",
        ],
        1.4,
    )
    report = criteria.score_candidate(losses, scoring, profile)
    assert report["composability"], "composability needs the own-topic single-module rows"


def test_rows_omit_absent_single_module_arms(sc):
    """A candidate that declares no "on" profiles asks only for the rows it has."""
    assert set(sc.required_rows(SPEC, "gram_c1")) == {
        "gram_c1_core__core",
        "gram_c1_core__deadline",
        "gram_c1_all_on__core",
        "gram_c1_all_on__deadline",
    }


def test_undeclared_candidate_is_refused(sc):
    with pytest.raises(SystemExit, match="not declared under scoring.gram_profiles"):
        sc.required_rows(SPEC, "gram_c9")


def test_outdir_equal_to_the_reference_directory_is_refused(sc, tmp_path):
    """Scoring beside the reference table would overwrite the campaign's shared fit."""
    reference = _write(tmp_path / "canon" / "results.tsv", {"baseline__core": (1.35, "ok")})
    with pytest.raises(SystemExit, match="would destroy the fit"):
        sc.resolve_outdir(tmp_path / "canon", reference)


def test_a_fresh_outdir_is_accepted(sc, tmp_path):
    reference = _write(tmp_path / "canon" / "results.tsv", {"baseline__core": (1.35, "ok")})
    assert sc.resolve_outdir(tmp_path / "analysis", reference) == (tmp_path / "analysis").resolve()


def test_reference_results_resolves_against_the_definition(sc, tmp_path):
    """A relative reference path belongs to the matrix it was produced from, not the cwd."""
    definition = tmp_path / "campaign" / "matrix.yaml"
    definition.parent.mkdir(parents=True)
    definition.write_text("unused")
    assert sc.reference_results(SPEC, definition) == tmp_path / "campaign" / "results.tsv"


def test_a_definition_without_a_reference_is_refused(sc, tmp_path):
    with pytest.raises(SystemExit, match="declares no reference_results"):
        sc.reference_results({"corpora": {}}, tmp_path / "matrix.yaml")


def test_merge_appends_candidate_rows_after_reference(sc, tmp_path):
    """The reference rows keep their order so a diff shows only what the candidate added."""
    reference = _write(tmp_path / "canon" / "results.tsv", {"baseline__core": (1.35, "ok")})
    probes = _write(
        tmp_path / "probes" / "results.tsv",
        {
            "gram_c1_core__core": (1.39, "ok"),
            "gram_c1_core__deadline": (1.54, "ok"),
            "gram_c1_all_on__core": (1.40, "ok"),
            "gram_c1_all_on__deadline": (1.50, "ok"),
        },
    )
    out = tmp_path / "analysis" / "results.tsv"
    sc.merge_results(reference, [probes], sc.required_rows(SPEC, "gram_c1"), out)

    names = [row["name"] for row in csv.DictReader(open(out), delimiter="\t")]
    assert names[0] == "baseline__core"
    assert set(names[1:]) == set(sc.required_rows(SPEC, "gram_c1"))


def test_later_probe_file_supersedes_earlier(sc, tmp_path):
    """Re-measuring a row means pointing at a newer directory, not editing a table."""
    reference = _write(tmp_path / "canon" / "results.tsv", {"baseline__core": (1.35, "ok")})
    first = _write(tmp_path / "p1" / "results.tsv", {"gram_c1_core__core": (9.99, "ok")})
    second = _write(tmp_path / "p2" / "results.tsv", {"gram_c1_core__core": (1.39, "ok")})
    out = tmp_path / "analysis" / "results.tsv"
    sc.merge_results(reference, [first, second], ["gram_c1_core__core"], out)

    rows = {row["name"]: row for row in csv.DictReader(open(out), delimiter="\t")}
    assert float(rows["gram_c1_core__core"]["lm_loss"]) == pytest.approx(1.39)


def test_missing_probe_row_stops_scoring(sc, tmp_path):
    reference = _write(tmp_path / "canon" / "results.tsv", {"baseline__core": (1.35, "ok")})
    probes = _write(tmp_path / "probes" / "results.tsv", {"gram_c1_core__core": (1.39, "ok")})
    with pytest.raises(SystemExit, match="missing probe rows"):
        sc.merge_results(reference, [probes], sc.required_rows(SPEC, "gram_c1"), tmp_path / "a" / "results.tsv")


def test_failed_probe_row_stops_scoring(sc, tmp_path):
    """A row that ran and failed must not be scored around — that biases the fit."""
    reference = _write(tmp_path / "canon" / "results.tsv", {"baseline__core": (1.35, "ok")})
    probes = _write(tmp_path / "probes" / "results.tsv", {"gram_c1_core__core": (0.0, "oom")})
    with pytest.raises(SystemExit, match="did not finish ok"):
        sc.merge_results(reference, [probes], ["gram_c1_core__core"], tmp_path / "a" / "results.tsv")
