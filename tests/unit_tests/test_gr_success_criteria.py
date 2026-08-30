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
"""The success-criteria scorer must band deltas correctly and pick the right bar per cell.

The report's meaning rests on three properties pinned here: the ambiguity band (a delta
between pass_nats and fail_nats is ``ambiguous``, never silently a pass or fail); the
per-cell bar choice (criterion 2's filter bar compares each topic against the filter arm
that TRAINED on it, and core against the core-only arm — crossing those up would score
every cell against the wrong model); and the direction convention (criterion 1 is an
absolute regression gate where beating the bar can still fail — an all-off profile that
models a routed topic better than the never-saw-it filter arm is a removal leak).
"""

from pathlib import Path

import pytest

from tests.unit_tests.gr_test_utils import load_script


_SCRIPT = Path(__file__).parents[2] / "scripts" / "gradient_routing" / "score_success_criteria.py"


@pytest.fixture(scope="module")
def sc():
    return load_script("score_success_criteria", _SCRIPT)


SCORING = {
    "noise_nats": 0.01,
    "pass_nats": 0.02,
    "fail_nats": 0.03,
    "baseline_arm": "baseline",
    "filter_core_arm": "filter_core",
    "core_corpus": "core",
    "topics": {"deadline": "filter_deadline", "aliens": "filter_aliens"},
}

PROFILE = {
    "off": "gram_core",
    "all_on": "gram_all_on",
    "on": {"deadline": "gram_deadline", "aliens": "gram_aliens"},
}


def _losses():
    return {
        "baseline__core": 1.35,
        "baseline__deadline": 1.42,
        "baseline__aliens": 1.33,
        "filter_core__core": 1.38,
        "filter_core__deadline": 1.55,
        "filter_core__aliens": 1.43,
        "filter_deadline__deadline": 1.45,
        "filter_deadline__core": 1.38,
        "filter_aliens__aliens": 1.34,
        "filter_aliens__core": 1.37,
        "gram_core__core": 1.39,
        "gram_core__deadline": 1.54,
        "gram_core__aliens": 1.435,
        "gram_all_on__core": 1.39,
        "gram_all_on__deadline": 1.49,
        "gram_all_on__aliens": 1.39,
        "gram_deadline__deadline": 1.494,
        "gram_aliens__aliens": 1.393,
    }


def test_verdict_bands_pass_ambiguous_fail(sc):
    assert sc.verdict(0.019, 0.02, 0.03, absolute=False) == "pass"
    assert sc.verdict(0.025, 0.02, 0.03, absolute=False) == "ambiguous"
    assert sc.verdict(0.031, 0.02, 0.03, absolute=False) == "fail"


def test_verdict_signed_passes_below_the_bar_but_absolute_does_not(sc):
    # -0.05 beats a signed bar outright; as a regression gate it is a 0.05 departure.
    assert sc.verdict(-0.05, 0.02, 0.03, absolute=False) == "pass"
    assert sc.verdict(-0.05, 0.02, 0.03, absolute=True) == "fail"


def test_criterion_1_is_an_absolute_gate(sc):
    report = sc.score_candidate(_losses(), SCORING, PROFILE)
    # gram_core models deadline BETTER than filter_core by 0.01 (a small leak): banded
    # on magnitude, so it passes only because 0.01 < pass_nats — flip the sign and it
    # must band identically.
    assert report["criterion_1"]["deadline"]["delta_nats"] == pytest.approx(-0.01)
    assert report["criterion_1"]["deadline"]["verdict"] == "pass"
    assert report["criterion_1"]["core"]["delta_nats"] == pytest.approx(0.01)


def test_criterion_2_filter_bar_matches_each_topic_to_its_own_filter_arm(sc):
    report = sc.score_candidate(_losses(), SCORING, PROFILE)
    cells = report["criterion_2_filter_bar"]
    assert cells["deadline"]["bar_arm"] == "filter_deadline"
    assert cells["deadline"]["delta_nats"] == pytest.approx(1.49 - 1.45)
    assert cells["aliens"]["bar_arm"] == "filter_aliens"
    assert cells["core"]["bar_arm"] == "filter_core"


def test_criterion_2_baseline_bar_and_topic_mean_exclude_core(sc):
    report = sc.score_candidate(_losses(), SCORING, PROFILE)
    cells = report["criterion_2_baseline_bar"]
    assert cells["deadline"]["bar_arm"] == "baseline"
    expected_mean = ((1.49 - 1.42) + (1.39 - 1.33)) / 2
    assert cells["topic_mean"]["delta_nats"] == pytest.approx(expected_mean)
    assert cells["topic_mean"]["verdict"] == "fail"


def test_composability_compares_all_on_to_each_single_on(sc):
    report = sc.score_candidate(_losses(), SCORING, PROFILE)
    comp = report["composability"]
    assert comp["deadline"]["delta_nats"] == pytest.approx(1.49 - 1.494)
    assert comp["deadline"]["verdict"] == "pass"


def test_a_candidate_without_single_on_arms_still_scores_the_mainline_cells(sc):
    minimal = {"off": "gram_core", "all_on": "gram_all_on"}
    report = sc.score_candidate(_losses(), SCORING, minimal)
    assert "composability" not in report
    assert set(report["criterion_1"]) == {"core", "deadline", "aliens"}
    assert "topic_mean" in report["criterion_2_baseline_bar"]


def test_missing_probe_rows_are_skipped_not_fatal(sc):
    losses = _losses()
    del losses["gram_all_on__aliens"]
    report = sc.score_candidate(losses, SCORING, PROFILE)
    assert "aliens" not in report["criterion_2_baseline_bar"] or "aliens" not in report["criterion_2_filter_bar"]
    # The topic mean then covers only the measured topics — and says so, since a
    # partial-coverage mean must not read as the full-topic number.
    mean_cell = report["criterion_2_baseline_bar"]["topic_mean"]
    assert mean_cell["delta_nats"] == pytest.approx(1.49 - 1.42)
    assert mean_cell["n_topics"] == 1
    assert mean_cell["n_topics_declared"] == 2


RATIOS = {
    "compute_ratios": {
        "gram_deadline": {"deadline": 0.78},
        "gram_aliens": {"aliens": 0.79},
        "gram_all_on": {"deadline": 0.77, "aliens": 0.78},
        "gram_core": {"core": 0.88},
        "filter_deadline": {"deadline": 0.90},
        "filter_aliens": {"aliens": 0.95},
        "filter_core": {"core": 0.93},
    }
}


def test_cr_gaps_use_single_on_profiles_when_present(sc):
    gaps = sc.cr_gaps(RATIOS, SCORING, PROFILE)
    assert gaps["retain"]["profile"] == "single_on"
    assert gaps["retain"]["gram_mean_cr"] == pytest.approx((0.78 + 0.79) / 2)
    assert gaps["retain"]["gap"] == pytest.approx((0.90 + 0.95) / 2 - (0.78 + 0.79) / 2)
    assert gaps["core"]["gap"] == pytest.approx(0.93 - 0.88)


def test_cr_gaps_fall_back_to_the_all_on_profile(sc):
    gaps = sc.cr_gaps(RATIOS, SCORING, {"off": "gram_core", "all_on": "gram_all_on"})
    assert gaps["retain"]["profile"] == "all_on"
    assert gaps["retain"]["gram_mean_cr"] == pytest.approx((0.77 + 0.78) / 2)
