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

import dataclasses
import os
import subprocess
import sys

import numpy as np
import pytest

from megatron.bridge.training.gradient_routing import plan as plan_module
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


#: The plan module's own source file. The subprocess below loads it by path rather than
#: importing ``megatron.bridge...plan``, which drags torch in behind the package __init__
#: and costs ~45 s for an assertion about numpy determinism. It is the same file the
#: in-process import resolved to, asserted below.
_PLAN_SOURCE = plan_module.__file__

_CHILD_PROGRAM = """
import importlib.util

spec = importlib.util.spec_from_file_location("gr_plan_child", {path!r})
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
plan = module.build_gr_plan(
    plan_seed={seed}, train_iters={iters}, forget_iter_fraction={f}, p_as={p_as}, p_cr={p_cr}
)
print(plan.digest())
for field in ("corpus", "fwd_aux", "update_core", "update_aux"):
    print("".join(str(value) for value in getattr(plan, field).tolist()))
"""


def _plan_lines(plan):
    """The digest plus every routing array, as the subprocess prints them."""
    return [plan.digest()] + [
        "".join(str(value) for value in getattr(plan, field).tolist())
        for field in ("corpus", "fwd_aux", "update_core", "update_aux")
    ]


def _plan_in_a_fresh_process(**env_extra) -> list[str]:
    program = _CHILD_PROGRAM.format(path=_PLAN_SOURCE, seed=1234, iters=ITERS, f=F, p_as=P_AS, p_cr=P_CR)
    result = subprocess.run(
        [sys.executable, "-c", program],
        capture_output=True,
        text=True,
        env={**os.environ, **env_extra},
        timeout=300,
    )
    assert result.returncode == 0, f"child process failed:\n{result.stderr}"
    return result.stdout.split()


class TestCrossProcessDeterminism:
    """Every rank builds its own plan from the seed; they must agree without communicating.

    Nothing reconciles the plans of different ranks — no broadcast, no checkpointed copy —
    so a build that picked up ANY process-local state (the legacy global numpy RNG, hash
    randomisation, whatever another library seeded) would give ranks different routing
    schedules on the same iteration: half the workers would train aux while the other half
    trained core, and the run would look entirely healthy.
    """

    def test_a_fresh_process_derives_the_identical_arrays(self):
        assert _plan_in_a_fresh_process() == _plan_lines(_plan())

    def test_hash_randomisation_does_not_move_the_plan(self):
        """PYTHONHASHSEED differs between ranks unless it is pinned; the plan must not care."""
        assert _plan_in_a_fresh_process(PYTHONHASHSEED="0") == _plan_in_a_fresh_process(PYTHONHASHSEED="12345")

    def test_the_subprocess_loads_the_module_under_test(self):
        """Guards the harness: a stale or wrong path would make the checks above vacuous."""
        assert _PLAN_SOURCE.endswith("gradient_routing/plan.py")
        assert os.path.isfile(_PLAN_SOURCE)

    def test_global_numpy_randomness_does_not_move_the_plan(self):
        """The plan draws from its own Generator(PCG64(seed)), never the legacy global state.

        ``np.random.seed`` + ``np.random.permutation`` would satisfy the same-process
        determinism tests only until something else in the run consumed the global stream —
        which a data loader, an initializer, or an augmentation would do at different points
        on different ranks.
        """
        expected = _plan().digest()
        np.random.seed(0)
        np.random.random(100)
        assert _plan().digest() == expected
        np.random.default_rng(7).random(100)
        assert _plan().digest() == expected


class TestDigestSensitivity:
    """The digest is the run's identity: the resume check refuses a plan whose digest moved.

    So it must move for ANY change that changes the schedule, and it must not be satisfiable
    by a partial hash — a digest over ``corpus`` alone would still look seed-sensitive while
    silently accepting a resume whose update sets had changed.
    """

    @pytest.mark.parametrize("field", ["corpus", "fwd_aux", "update_core", "update_aux"])
    def test_changing_any_routing_array_changes_the_digest(self, field):
        plan = _plan()
        flipped = getattr(plan, field).copy()
        flipped[0] = 1 - flipped[0]
        assert dataclasses.replace(plan, **{field: flipped}).digest() != plan.digest()

    def test_a_parameter_change_alone_changes_the_digest(self):
        """p_cr 0.2 and 0.204 both realise round(p * 60) = 12 core-robustness iterations, so
        their arrays are byte-identical — the digest separates them only because the
        parameters are hashed too. Two runs configured differently are two experiments even
        when this plan length cannot tell them apart."""
        a, b = _plan(p_cr=0.2), _plan(p_cr=0.204)
        for field in ("corpus", "fwd_aux", "update_core", "update_aux"):
            assert np.array_equal(getattr(a, field), getattr(b, field)), f"{field} differs; the premise is gone"
        assert a.digest() != b.digest()

    def test_changing_train_iters_changes_the_digest(self):
        assert _plan(iters=ITERS).digest() != _plan(iters=ITERS + 1).digest()

    def test_the_digest_is_sixteen_hex_characters(self):
        """The callback logs ``int(digest[:8], 16)`` as the W&B provenance field; a shorter or
        non-hex digest would raise mid-run, inside the telemetry path."""
        digest = _plan().digest()
        assert len(digest) == 16
        assert set(digest) <= set("0123456789abcdef")


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


def _spread_and_robust(plan) -> tuple[int, int]:
    """Realised (forget-spread, core-robustness) counts of a plan."""
    return (
        int(((plan.corpus == FORGET) & plan.update_core.astype(bool)).sum()),
        int(((plan.corpus == RETAIN) & plan.update_aux.astype(bool)).sum()),
    )


class TestExactCountsAcrossShapes:
    """``round(p * n)`` at EVERY level, for plan lengths that do not divide evenly.

    The canonical 120-iteration probe divides cleanly at every level, so it cannot tell an
    exact-count allocator from one that truncates, rounds the other way, or draws i.i.d.
    Bernoulli and happens to land on the expected count. These shapes make each level land
    off a whole number, including on exact halves.
    """

    @pytest.mark.parametrize("iters", [1, 2, 3, 5, 7, 13, 33, 120, 121])
    @pytest.mark.parametrize("f", [0.0, 0.1, 0.25, 0.5, 0.75, 1.0])
    @pytest.mark.parametrize("p_as, p_cr", [(0.0, 0.0), (0.5, 0.2), (0.3, 0.7), (1.0, 1.0)])
    def test_every_realised_count_equals_round_of_p_times_n(self, iters, f, p_as, p_cr):
        plan = _plan(iters=iters, f=f, p_as=p_as, p_cr=p_cr)
        n_forget = round(f * iters)
        n_retain = iters - n_forget
        assert plan.n_forget_iters == n_forget
        assert plan.n_retain_iters == n_retain
        assert _spread_and_robust(plan) == (round(p_as * n_forget), round(p_cr * n_retain))

    @pytest.mark.parametrize("iters", [1, 2, 3, 5, 7, 13, 33, 121])
    def test_sample_counts_partition_the_plan(self, iters):
        """Every iteration draws exactly one corpus, so the two per-corpus sample counts must
        add up to the routed dataset's length — an off-by-one here is a short read at the end
        of training, thousands of iterations after the plan was built."""
        plan = _plan(iters=iters, f=0.3)
        assert plan.n_samples(RETAIN, 8) + plan.n_samples(FORGET, 8) == iters * 8


class TestHalfShareRounding:
    """A product landing exactly on .5 rounds to EVEN, not up — pinned with literals.

    ``int(round(x))`` is banker's rounding, and ``int(x + 0.5)`` is the natural thing for
    someone to "simplify" it to. The two differ on every half-integer product, which would
    silently relabel iterations in any plan that hits one — including on a resume, where the
    digest check would then refuse to restart a run that had been training happily.
    """

    @pytest.mark.parametrize(
        "iters, f, expected_forget",
        [
            (1, 0.5, 0),  # round(0.5) = 0, not 1
            (3, 0.5, 2),  # round(1.5) = 2
            (5, 0.5, 2),  # round(2.5) = 2, not 3
            (7, 0.5, 4),  # round(3.5) = 4
            (6, 0.25, 2),  # round(1.5) = 2
            (10, 0.25, 2),  # round(2.5) = 2
        ],
    )
    def test_a_half_corpus_share_rounds_to_even(self, iters, f, expected_forget):
        assert _plan(iters=iters, f=f).n_forget_iters == expected_forget

    def test_half_sub_label_shares_round_to_even(self):
        """The same rule one level down: 5 forget iterations at p_as 0.5 give 2 spread, not 3,
        and 5 retain iterations at p_cr 0.5 give 2 core-robustness."""
        plan = _plan(iters=10, f=0.5, p_as=0.5, p_cr=0.5)
        assert (plan.n_forget_iters, plan.n_retain_iters) == (5, 5)
        assert _spread_and_robust(plan) == (2, 2)


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


class TestTinyPlansAtEveryCorner:
    """1-3 iterations at every fraction corner still produce a fully well-formed plan.

    The smoke script and the functional tests run plans this short, and every degenerate
    combination sends some ``round(p * n)`` to 0 or to n — the regime where an empty index
    array reaches ``rng.permutation`` and an empty corpus reaches the routed dataset. A plan
    that came out malformed here would surface as a mid-run RuntimeError from the gater, or
    as an out-of-range read on a corpus with no samples.
    """

    @pytest.mark.parametrize("iters", [1, 2, 3])
    @pytest.mark.parametrize("f", [0.0, 0.5, 1.0])
    @pytest.mark.parametrize("p_as", [0.0, 1.0])
    @pytest.mark.parametrize("p_cr", [0.0, 1.0])
    def test_a_tiny_plan_is_well_formed(self, iters, f, p_as, p_cr):
        plan = _plan(iters=iters, f=f, p_as=p_as, p_cr=p_cr)
        assert plan.train_iters == iters
        assert plan.n_forget_iters + plan.n_retain_iters == iters
        assert (plan.update_core.astype(bool) | plan.update_aux.astype(bool)).all(), "an iteration updates nothing"
        assert np.array_equal(plan.fwd_aux, plan.update_aux), "aux was forwarded without being updated, or vice versa"
        for corpus in (RETAIN, FORGET):
            offsets = plan.prior_iters_same_corpus[plan.corpus == corpus]
            assert np.array_equal(offsets, np.arange(len(offsets))), f"corpus {corpus} offsets are not gapless"

    @pytest.mark.parametrize("f, empty_corpus", [(0.0, FORGET), (1.0, RETAIN)])
    def test_a_corpus_the_plan_never_draws_consumes_no_samples(self, f, empty_corpus):
        """The routed dataset sizes each child from this; a non-zero count for a corpus that
        is never drawn would demand samples from a dataset the run has no reason to build."""
        plan = _plan(iters=3, f=f)
        assert plan.n_samples(empty_corpus, 8) == 0
        assert int((plan.corpus == empty_corpus).sum()) == 0
