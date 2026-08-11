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
"""The compute-ratio fit must recover what it claims to invert.

``compute_ratio.py`` replicates the GRAM reference's log-space power-law fit and inverts
it at each arm's loss. The property that makes every downstream ratio meaningful is
round-tripping: on data generated FROM a power law, the fit must recover it and
``step_equiv`` must invert it — so those are pinned on synthetic curves with known
parameters, not on tolerances around published numbers (the reference-repo cross-check
against its own losses.pkl is a campaign step, not a unit test).
"""

from pathlib import Path

import numpy as np
import pytest

from tests.unit_tests.gr_test_utils import load_script


_SCRIPT = Path(__file__).parents[2] / "scripts" / "gradient_routing" / "compute_ratio.py"


@pytest.fixture(scope="module")
def cr():
    return load_script("compute_ratio", _SCRIPT)


def _curve(a=3.0, alpha=0.4, x0=50.0, steps=None):
    steps = np.array(steps if steps is not None else [172, 344, 688, 1032, 1376, 2064], dtype=np.float64)
    return steps, a * (steps + x0) ** -alpha


def test_fit_recovers_a_known_power_law(cr):
    steps, losses = _curve()
    a, alpha, x0 = cr.fit_power(steps, losses)
    assert a == pytest.approx(3.0, rel=1e-3)
    assert alpha == pytest.approx(0.4, rel=1e-3)
    assert x0 == pytest.approx(50.0, rel=1e-2)


def test_step_equiv_inverts_the_fit(cr):
    steps, losses = _curve()
    fit = cr.fit_power(steps, losses)
    for step, loss in zip(steps, losses):
        assert cr.step_equiv(float(loss), *fit) == pytest.approx(step, rel=1e-3)


def test_step_equiv_never_goes_nonpositive(cr):
    """A loss above the curve's start inverts to a step before 0; the floor keeps the
    ratio finite rather than letting a slightly-worse-than-init arm divide by <= 0."""
    steps, losses = _curve()
    fit = cr.fit_power(steps, losses)
    assert cr.step_equiv(float(losses[0]) * 10.0, *fit) > 0


def test_classify_rows_splits_curves_and_arms(cr):
    losses = {
        "curve_iter172__core": 2.5,
        "curve_iter344__core": 2.1,
        "baseline__core": 1.5,
        "gram_m0__aliens": 1.9,
    }
    curves, arms = cr.classify_rows(losses)
    assert curves == {"core": {172: 2.5, 344: 2.1}}
    assert arms == {"baseline": {"core": 1.5}, "gram_m0": {"aliens": 1.9}}


def test_classify_rows_refuses_names_off_contract(cr):
    with pytest.raises(SystemExit, match="row names do not follow"):
        cr.classify_rows({"no-separator-here": 1.0})


def test_the_reference_arm_is_the_arm_the_curve_checkpoint_names(cr):
    """The reference arm is derived from the definition, not re-declared: it is the one
    arm whose checkpoint the curve: section points at."""
    spec = {
        "curve": {"checkpoint": "/ckpt/base", "steps": [172]},
        "arms": {"baseline": {"checkpoint": "/ckpt/base"}, "gram": {"checkpoint": "/ckpt/gram"}},
    }
    assert cr.reference_arm_from_definition(spec, "matrix.yaml") == "baseline"


@pytest.mark.parametrize(
    "arms",
    [
        {"baseline": {"checkpoint": "/ckpt/other"}},
        {"a": {"checkpoint": "/ckpt/base"}, "b": {"checkpoint": "/ckpt/base"}},
    ],
)
def test_a_curve_checkpoint_matching_anything_but_one_arm_is_refused(cr, arms):
    """Zero matches would divide one arm's loss by another arm's fitted curve; several
    make the denominator ambiguous. Both refuse."""
    spec = {"curve": {"checkpoint": "/ckpt/base", "steps": [172]}, "arms": arms}
    with pytest.raises(SystemExit, match="exactly one arm"):
        cr.reference_arm_from_definition(spec, "matrix.yaml")


def test_reference_arm_scores_one_on_its_own_curve(cr):
    """The denominator is the reference's own final step-equivalent, so an arm whose loss
    IS the curve's final loss must land at CR = 1 by construction."""
    steps, losses = _curve()
    fit = cr.fit_power(steps, losses)
    final_se = cr.step_equiv(float(losses[-1]), *fit)
    assert cr.step_equiv(float(losses[-1]), *fit) / final_se == pytest.approx(1.0)
    better = cr.step_equiv(float(losses[-1]) * 0.98, *fit) / final_se
    worse = cr.step_equiv(float(losses[-1]) * 1.02, *fit) / final_se
    assert better > 1.0 > worse
