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
"""Pinning tests for the gradient-routing plan.

The plan is the experiment's identity: every rank and every restart re-derives it from
the seed alone, and the realised sub-label counts are what the paper's probabilities
mean. Both properties are asserted exactly (array equality, exact integer counts) rather
than statistically — an exact-count allocation that silently became i.i.d. Bernoulli
would still pass a tolerance test while changing what the run measures.
"""

import numpy as np
import pytest

from megatron.bridge.training.gradient_routing.plan import FORGET, RETAIN, build_gr_plan


# The canonical probe point: 120 iterations at the paper defaults divides evenly at every
# level, so every count below is exact arithmetic rather than a rounding artefact.
ITERS, F, P_AS, P_CR = 120, 0.5, 0.5, 0.2


def _plan(seed=1234, iters=ITERS, f=F, p_as=P_AS, p_cr=P_CR):
    return build_gr_plan(plan_seed=seed, train_iters=iters, forget_iter_fraction=f, p_as=p_as, p_cr=p_cr)


class TestDeterminism:
    """Two builds of the same config must be bit-identical — restarts depend on it."""

    def test_two_builds_are_identical(self):
        a, b = _plan(), _plan()
        for field in ("corpus", "fwd_aux", "update_core", "update_aux", "prior_iters_same_corpus"):
            assert np.array_equal(getattr(a, field), getattr(b, field)), f"{field} differs between builds"
        assert a.digest() == b.digest()

    def test_different_seeds_give_different_digests(self):
        """The seed must actually reach the placement, not merely be recorded."""
        digests = {_plan(seed=s).digest() for s in (0, 1, 1234, 9999)}
        assert len(digests) == 4, f"seeds collided: {digests}"

    def test_digest_is_stable_across_calls(self):
        plan = _plan()
        assert plan.digest() == plan.digest()

    def test_digest_covers_the_arrays_not_just_the_parameters(self):
        """Same parameters, different seed -> different arrays -> different digest.

        The parameter prefix alone would make every seed hash the same if the arrays were
        dropped from the hash, so this pins that the arrays are actually fed in.
        """
        a, b = _plan(seed=1), _plan(seed=2)
        assert (a.p_as, a.p_cr, a.forget_iter_fraction) == (b.p_as, b.p_cr, b.forget_iter_fraction)
        assert not np.array_equal(a.corpus, b.corpus)
        assert a.digest() != b.digest()


class TestExactCounts:
    """Exact-count allocation: the realised fractions equal the configured ones."""

    def test_corpus_split_is_exact(self):
        plan = _plan()
        assert plan.n_forget_iters == 60
        assert plan.n_retain_iters == 60
        assert plan.train_iters == ITERS

    def test_forget_spread_count_is_exact(self):
        """p_as of the forget iterations also update core."""
        plan = _plan()
        spread = (plan.corpus == FORGET) & plan.update_core.astype(bool)
        assert int(spread.sum()) == 30

    def test_core_robustness_count_is_exact(self):
        """p_cr of the retain iterations also activate + update aux."""
        plan = _plan()
        robust = (plan.corpus == RETAIN) & plan.update_aux.astype(bool)
        assert int(robust.sum()) == 12

    def test_update_totals_are_exact(self):
        plan = _plan()
        assert int(plan.update_aux.sum()) == 72  # 60 forget + 12 core-robustness
        assert int(plan.update_core.sum()) == 90  # 60 retain + 30 forget-spread

    def test_fwd_aux_is_exactly_the_aux_update_set(self):
        """Aux is forward-activated on precisely the iterations that update it.

        Not a coincidence of the defaults: forget iterations set both, and core-robustness
        sets both from the same index draw. A divergence would mean aux received gradients
        on an iteration whose forward never ran it (or vice versa).
        """
        plan = _plan()
        assert np.array_equal(plan.fwd_aux, plan.update_aux)

    @pytest.mark.parametrize(
        "f, p_as, p_cr, n_forget, n_spread, n_robust",
        [
            (0.0, 0.5, 0.2, 0, 0, 24),  # no forget corpus at all: aux only via core-robustness
            (1.0, 0.5, 0.2, 120, 60, 0),  # no retain corpus: core only via forget-spread
            (0.5, 0.0, 0.0, 60, 0, 0),  # pure isolation: no spread, no robustness
            (0.5, 1.0, 1.0, 60, 60, 60),  # everything everywhere: every iter updates both
            (0.25, 0.5, 0.2, 30, 15, 18),  # round(0.2 * 90) = 18
        ],
    )
    def test_edge_fraction_counts(self, f, p_as, p_cr, n_forget, n_spread, n_robust):
        plan = _plan(f=f, p_as=p_as, p_cr=p_cr)
        assert plan.n_forget_iters == n_forget
        assert int(((plan.corpus == FORGET) & plan.update_core.astype(bool)).sum()) == n_spread
        assert int(((plan.corpus == RETAIN) & plan.update_aux.astype(bool)).sum()) == n_robust

    def test_forget_iterations_always_update_aux(self):
        plan = _plan()
        assert plan.update_aux[plan.corpus == FORGET].all()
        assert plan.fwd_aux[plan.corpus == FORGET].all()

    def test_retain_iterations_always_update_core(self):
        plan = _plan()
        assert plan.update_core[plan.corpus == RETAIN].all()


class TestPlanWellFormedness:
    """Properties the gater and the callback rely on holding for EVERY iteration."""

    @pytest.mark.parametrize("f", [0.0, 0.25, 0.5, 0.75, 1.0])
    @pytest.mark.parametrize("p_as", [0.0, 0.5, 1.0])
    @pytest.mark.parametrize("p_cr", [0.0, 0.2, 1.0])
    def test_every_iteration_updates_something(self, f, p_as, p_cr):
        """GROptimizerGater.arm raises on an iteration updating neither role.

        Swept across the fraction corners because that raise is the only thing standing
        between a malformed plan and a silently wasted optimizer step.
        """
        plan = _plan(f=f, p_as=p_as, p_cr=p_cr)
        assert (plan.update_core.astype(bool) | plan.update_aux.astype(bool)).all()

    def test_arrays_share_length_and_dtype(self):
        plan = _plan()
        for field in ("corpus", "fwd_aux", "update_core", "update_aux", "prior_iters_same_corpus"):
            arr = getattr(plan, field)
            assert len(arr) == ITERS, f"{field} has length {len(arr)}"
            assert arr.dtype == np.int64, f"{field} has dtype {arr.dtype}"

    def test_corpus_values_are_only_the_two_labels(self):
        plan = _plan()
        assert set(np.unique(plan.corpus)).issubset({RETAIN, FORGET})

    def test_indicator_arrays_are_zero_one(self):
        plan = _plan()
        for field in ("fwd_aux", "update_core", "update_aux"):
            assert set(np.unique(getattr(plan, field))).issubset({0, 1}), field

    def test_n_samples_scales_with_global_batch_size(self):
        plan = _plan()
        assert plan.n_samples(FORGET, 8) == 60 * 8
        assert plan.n_samples(RETAIN, 8) == 60 * 8
        assert plan.n_samples(FORGET, 1) + plan.n_samples(RETAIN, 1) == ITERS

    def test_describe_reports_the_counts_it_claims(self):
        plan = _plan()
        text = plan.describe()
        for fragment in ("iters=120", "forget=60", "retain=60", "forget_spread=30", "core_robustness=12"):
            assert fragment in text, f"{fragment!r} missing from {text!r}"
        assert plan.digest() in text


class TestPriorItersSameCorpus:
    """The routed dataset's offset arithmetic is only gapless if this is a true prefix count."""

    def test_matches_an_independent_prefix_count(self):
        plan = _plan()
        counts = {RETAIN: 0, FORGET: 0}
        for i, corpus in enumerate(plan.corpus.tolist()):
            assert int(plan.prior_iters_same_corpus[i]) == counts[corpus], f"iteration {i}"
            counts[corpus] += 1

    def test_each_corpus_offsets_form_a_gapless_range(self):
        """Per corpus the values are exactly 0..n-1 in order — no gaps, no repeats."""
        plan = _plan()
        for corpus, n_expected in ((RETAIN, 60), (FORGET, 60)):
            offsets = plan.prior_iters_same_corpus[plan.corpus == corpus]
            assert np.array_equal(offsets, np.arange(n_expected)), f"corpus {corpus}: {offsets}"

    @pytest.mark.parametrize("f", [0.0, 0.5, 1.0])
    def test_gapless_under_degenerate_fractions(self, f):
        plan = _plan(f=f)
        for corpus in (RETAIN, FORGET):
            offsets = plan.prior_iters_same_corpus[plan.corpus == corpus]
            assert np.array_equal(offsets, np.arange(len(offsets)))


class TestValidation:
    """Bad plan parameters must raise at build time, not produce a degenerate plan."""

    @pytest.mark.parametrize("train_iters", [0, -1, -120])
    def test_non_positive_train_iters_raises(self, train_iters):
        with pytest.raises(ValueError, match="train_iters must be positive"):
            _plan(iters=train_iters)

    @pytest.mark.parametrize("value", [-0.1, 1.1, 2.0, float("nan")])
    @pytest.mark.parametrize("field", ["f", "p_as", "p_cr"])
    def test_out_of_range_fractions_raise(self, field, value):
        name = {"f": "forget_iter_fraction", "p_as": "p_as", "p_cr": "p_cr"}[field]
        with pytest.raises(ValueError, match=name):
            _plan(**{field: value})

    def test_single_iteration_plan_is_valid(self):
        """train_iters=1 is the smallest legal plan; it must not trip the round()s."""
        plan = _plan(iters=1)
        assert plan.train_iters == 1
        assert bool(plan.update_core[0]) or bool(plan.update_aux[0])
