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

Every count is asserted per module, not aggregated: with N corpora the aggregate totals
are the same for a plan that allocated correctly and for one that gave module 0 both
modules' iterations, which is precisely the confusion multi-module routing introduces.
"""

import dataclasses
import os
import subprocess
import sys

import numpy as np
import pytest

from megatron.bridge.training.gradient_routing import plan as plan_module
from megatron.bridge.training.gradient_routing.plan import CORE, FIRST_AUX, build_gr_plan


# The canonical probe point: 120 iterations at the paper defaults divides evenly at every
# level, so every count below is exact arithmetic rather than a rounding artefact.
ITERS, FRACTIONS, P_AS, P_CR = 120, [0.5], 0.5, 0.2

#: Two- and three-module probes whose per-module counts also divide evenly at 120 iters.
FRACTIONS_2 = [0.25, 0.25]
FRACTIONS_3 = [0.1, 0.2, 0.3]


def _plan(seed=1234, iters=ITERS, fractions=FRACTIONS, p_as=P_AS, p_cr=P_CR):
    return build_gr_plan(plan_seed=seed, train_iters=iters, aux_iter_fractions=fractions, p_as=p_as, p_cr=p_cr)


def _spread(plan, module: int) -> int:
    """Realised aux-spread count of one module: its own iterations that also update core."""
    return int(plan.update_core[plan.corpus == module + FIRST_AUX].sum())


def _robust(plan, module: int) -> int:
    """Realised core-robustness count of one module: core iterations that also update it."""
    return int(plan.update_aux[plan.corpus == CORE, module].sum())


def _expected_robust(fractions, n_core: int, p_cr: float, module: int) -> int:
    """The reference allocation: p_cr of the core iterations, split by data share."""
    total = sum(fractions)
    weight = fractions[module] / total if total > 0 else 0.0
    return round(p_cr * n_core * weight)


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
        assert (a.p_as, a.p_cr, a.aux_iter_fractions) == (b.p_as, b.p_cr, b.aux_iter_fractions)
        assert not np.array_equal(a.corpus, b.corpus)
        assert a.digest() != b.digest()

    @pytest.mark.parametrize("fractions", [FRACTIONS, FRACTIONS_2, FRACTIONS_3])
    def test_determinism_holds_at_every_module_count(self, fractions):
        a, b = _plan(fractions=fractions), _plan(fractions=fractions)
        assert np.array_equal(a.update_aux, b.update_aux)
        assert a.digest() == b.digest()


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
    plan_seed={seed}, train_iters={iters}, aux_iter_fractions={fractions!r}, p_as={p_as}, p_cr={p_cr}
)
print(plan.digest())
for field in ("corpus", "fwd_aux", "update_core", "update_aux"):
    print("".join(str(value) for value in getattr(plan, field).flatten().tolist()))
"""


def _plan_lines(plan):
    """The digest plus every routing array, as the subprocess prints them."""
    return [plan.digest()] + [
        "".join(str(value) for value in getattr(plan, field).flatten().tolist())
        for field in ("corpus", "fwd_aux", "update_core", "update_aux")
    ]


def _plan_in_a_fresh_process(fractions=FRACTIONS, **env_extra) -> list[str]:
    program = _CHILD_PROGRAM.format(
        path=_PLAN_SOURCE, seed=1234, iters=ITERS, fractions=fractions, p_as=P_AS, p_cr=P_CR
    )
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

    @pytest.mark.parametrize("fractions", [FRACTIONS, FRACTIONS_2])
    def test_a_fresh_process_derives_the_identical_arrays(self, fractions):
        assert _plan_in_a_fresh_process(fractions) == _plan_lines(_plan(fractions=fractions))

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

    def test_a_per_module_fraction_change_alone_changes_the_digest(self):
        """The same argument one axis over: 0.25 and 0.2504 both realise 30 iterations of aux
        corpus 2 and the same proportional robustness split, so the arrays are identical and
        only the hashed fraction list separates the two experiments."""
        a, b = _plan(fractions=[0.25, 0.25]), _plan(fractions=[0.25, 0.2504])
        for field in ("corpus", "fwd_aux", "update_core", "update_aux"):
            assert np.array_equal(getattr(a, field), getattr(b, field)), f"{field} differs; the premise is gone"
        assert a.digest() != b.digest()

    @pytest.mark.parametrize("module", [0, 1])
    def test_changing_one_modules_fraction_changes_the_digest(self, module):
        fractions = list(FRACTIONS_2)
        fractions[module] = 0.3
        assert _plan(fractions=fractions).digest() != _plan(fractions=FRACTIONS_2).digest()

    def test_adding_an_unused_module_changes_the_digest(self):
        """``[0.5]`` and ``[0.5, 0.0]`` route the identical corpus sequence — the second
        module simply never draws — so nothing in the arrays' VALUES distinguishes them. They
        are still different experiments: the second builds an extra aux module, an extra
        param group, and an extra corpus dataset. n_aux is hashed for exactly this case."""
        one, two = _plan(fractions=[0.5]), _plan(fractions=[0.5, 0.0])
        assert np.array_equal(one.corpus, two.corpus)
        assert np.array_equal(one.fwd_aux[:, 0], two.fwd_aux[:, 0])
        assert (one.n_aux, two.n_aux) == (1, 2)
        assert one.digest() != two.digest()

    def test_reordering_the_fractions_changes_the_digest(self):
        """Module order defines which corpus trains which module, so it is part of the
        experiment even when the multiset of fractions is unchanged."""
        assert _plan(fractions=[0.2, 0.3]).digest() != _plan(fractions=[0.3, 0.2]).digest()

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
        assert plan.n_corpus_iters(FIRST_AUX) == 60
        assert plan.n_core_iters == 60
        assert plan.train_iters == ITERS

    def test_aux_spread_count_is_exact(self):
        """p_as of each aux corpus's iterations also update core."""
        assert _spread(_plan(), 0) == 30

    def test_core_robustness_count_is_exact(self):
        """p_cr of the core iterations also activate + update aux."""
        assert _robust(_plan(), 0) == 12

    def test_update_totals_are_exact(self):
        plan = _plan()
        assert int(plan.update_aux.sum()) == 72  # 60 aux-corpus + 12 core-robustness
        assert int(plan.update_core.sum()) == 90  # 60 core + 30 aux-spread

    def test_fwd_aux_is_exactly_the_aux_update_set(self):
        """Aux is forward-activated on precisely the iterations that update it.

        Not a coincidence of the defaults: aux iterations set both, and core-robustness
        sets both from the same index draw. A divergence would mean a module received
        gradients on an iteration whose forward never ran it (or vice versa).
        """
        plan = _plan()
        assert np.array_equal(plan.fwd_aux, plan.update_aux)

    @pytest.mark.parametrize(
        "fractions, p_as, p_cr, n_aux_iters, n_spread, n_robust",
        [
            # No aux corpus at all: robustness is allocated in proportion to the data
            # shares, so a module with no data gets no robustness iterations either — the
            # reference implementation's rule, and the plan has no aux gradient anywhere.
            ([0.0], 0.5, 0.2, [0], [0], [0]),
            ([1.0], 0.5, 0.2, [120], [60], [0]),  # no core corpus: core only via aux-spread
            ([0.5], 0.0, 0.0, [60], [0], [0]),  # pure isolation: no spread, no robustness
            ([0.5], 1.0, 1.0, [60], [60], [60]),  # everything everywhere: every iter updates both
            ([0.25], 0.5, 0.2, [30], [15], [18]),  # round(0.2 * 90) = 18
            ([0.25, 0.25], 0.5, 0.2, [30, 30], [15, 15], [6, 6]),  # equal shares split robustness evenly
            ([0.2, 0.3], 0.5, 0.2, [24, 36], [12, 18], [5, 7]),  # 0.4/0.6 of round(0.2 * 60) = 12
            ([0.5, 0.5], 0.5, 0.2, [60, 60], [30, 30], [0, 0]),  # no core corpus at N=2
            ([0.1, 0.2, 0.3], 0.5, 0.2, [12, 24, 36], [6, 12, 18], [2, 3, 5]),  # 1/6, 1/3, 1/2 of 48 * 0.2
        ],
    )
    def test_edge_fraction_counts(self, fractions, p_as, p_cr, n_aux_iters, n_spread, n_robust):
        plan = _plan(fractions=fractions, p_as=p_as, p_cr=p_cr)
        assert [plan.n_corpus_iters(k + FIRST_AUX) for k in range(len(fractions))] == n_aux_iters
        assert [_spread(plan, k) for k in range(len(fractions))] == n_spread
        assert [_robust(plan, k) for k in range(len(fractions))] == n_robust

    def test_aux_iterations_always_update_their_own_module(self):
        plan = _plan()
        assert plan.update_aux[plan.corpus == FIRST_AUX].all()
        assert plan.fwd_aux[plan.corpus == FIRST_AUX].all()

    def test_core_iterations_always_update_core(self):
        plan = _plan()
        assert plan.update_core[plan.corpus == CORE].all()

    def test_the_corpus_labels_are_pinned(self):
        """Core is 0 and modules start at 1 — the routed dataset and configs count on it."""
        assert CORE == 0
        assert FIRST_AUX == 1


class TestPerModuleIsolation:
    """Aux corpus ``c`` must train module ``c - 1`` and no other — the multi-module contract.

    Aggregate counts cannot see this: a plan that ran every module on every aux corpus, or
    that shifted the mapping by one, has the same totals. So the columns are read per
    corpus, and each module's core-robustness iterations are checked to be its own.
    """

    @pytest.mark.parametrize("fractions", [FRACTIONS, FRACTIONS_2, FRACTIONS_3])
    def test_an_aux_corpus_activates_only_its_own_module(self, fractions):
        plan = _plan(fractions=fractions)
        for k in range(plan.n_aux):
            rows = plan.update_aux[plan.corpus == k + FIRST_AUX]
            expected = np.zeros(plan.n_aux, dtype=np.int64)
            expected[k] = len(rows)
            assert np.array_equal(rows.sum(axis=0), expected), f"aux corpus {k + FIRST_AUX} touched another module"
            assert np.array_equal(plan.fwd_aux[plan.corpus == k + FIRST_AUX], rows)

    @pytest.mark.parametrize("fractions", [FRACTIONS_2, FRACTIONS_3])
    def test_a_core_robustness_iteration_activates_exactly_one_module(self, fractions):
        """The reference implementation activates ONE module per robustness iteration — the
        core is being made robust to one capability's presence at a time, and two open gates
        would also make the aux-output telemetry unattributable."""
        plan = _plan(fractions=fractions)
        per_iteration = plan.update_aux[plan.corpus == CORE].sum(axis=1)
        assert set(np.unique(per_iteration)) <= {0, 1}, f"a core iteration opened {per_iteration.max()} gates"

    @pytest.mark.parametrize("fractions", [FRACTIONS, FRACTIONS_2, FRACTIONS_3])
    @pytest.mark.parametrize("p_cr", [0.0, 0.2, 0.5, 1.0])
    def test_robustness_is_allocated_in_proportion_to_the_data_shares(self, fractions, p_cr):
        """A module trained on twice the data gets twice the robustness iterations. The old
        binary plan had one aux module and so no allocation to make; with N modules the split
        is what stops a small-share module from being over-represented in the core arm."""
        plan = _plan(fractions=fractions, p_cr=p_cr)
        n_core = plan.n_core_iters
        assert [_robust(plan, k) for k in range(plan.n_aux)] == [
            _expected_robust(fractions, n_core, p_cr, k) for k in range(plan.n_aux)
        ]

    def test_a_zero_share_module_gets_no_gradient_at_all(self):
        """``aux_iter_fractions=[0.5, 0.0]`` builds module 1 but never routes to it: no
        corpus, no spread, and (by the proportional rule) no robustness either. It must stay
        at its zero init, which is what makes such a module a legitimate export posture."""
        plan = _plan(fractions=[0.5, 0.0])
        assert plan.n_corpus_iters(FIRST_AUX + 1) == 0
        assert not plan.update_aux[:, 1].any()
        assert not plan.fwd_aux[:, 1].any()
        assert plan.update_aux[:, 0].any(), "module 0 must still be routed, or this proves nothing"


def _spread_and_robust(plan) -> tuple[int, int]:
    """Realised (aux-spread, core-robustness) counts of a single-module plan."""
    return (_spread(plan, 0), _robust(plan, 0))


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
        plan = _plan(iters=iters, fractions=[f], p_as=p_as, p_cr=p_cr)
        n_aux_iters = round(f * iters)
        n_core = iters - n_aux_iters
        assert plan.n_corpus_iters(FIRST_AUX) == n_aux_iters
        assert plan.n_core_iters == n_core
        assert _spread_and_robust(plan) == (
            round(p_as * n_aux_iters),
            _expected_robust([f], n_core, p_cr, 0),
        )

    @pytest.mark.parametrize("iters", [2, 5, 13, 33, 120, 121])
    @pytest.mark.parametrize("fractions", [[0.1, 0.1], [0.25, 0.1], [0.3, 0.4], [0.1, 0.2, 0.3]])
    @pytest.mark.parametrize("p_as, p_cr", [(0.0, 0.0), (0.5, 0.2), (0.3, 0.7)])
    def test_every_per_module_count_equals_round_of_p_times_n(self, iters, fractions, p_as, p_cr):
        plan = _plan(iters=iters, fractions=fractions, p_as=p_as, p_cr=p_cr)
        per_module = [round(f * iters) for f in fractions]
        n_core = iters - sum(per_module)
        assert [plan.n_corpus_iters(k + FIRST_AUX) for k in range(len(fractions))] == per_module
        assert plan.n_core_iters == n_core
        assert [_spread(plan, k) for k in range(len(fractions))] == [round(p_as * n) for n in per_module]
        assert [_robust(plan, k) for k in range(len(fractions))] == [
            _expected_robust(fractions, n_core, p_cr, k) for k in range(len(fractions))
        ]

    @pytest.mark.parametrize("iters", [1, 2, 3, 5, 7, 13, 33, 121])
    @pytest.mark.parametrize("fractions", [[0.3], [0.2, 0.3], [0.1, 0.2, 0.3]])
    def test_sample_counts_partition_the_plan(self, iters, fractions):
        """Every iteration draws exactly one corpus, so the per-corpus sample counts must add
        up to the routed dataset's length — an off-by-one here is a short read at the end of
        training, thousands of iterations after the plan was built."""
        plan = _plan(iters=iters, fractions=fractions)
        total = plan.n_samples(CORE, 8) + sum(plan.n_samples(k + FIRST_AUX, 8) for k in range(plan.n_aux))
        assert total == iters * 8


class TestHalfShareRounding:
    """A product landing exactly on .5 rounds to EVEN, not up — pinned with literals.

    ``int(round(x))`` is banker's rounding, and ``int(x + 0.5)`` is the natural thing for
    someone to "simplify" it to. The two differ on every half-integer product, which would
    silently relabel iterations in any plan that hits one — including on a resume, where the
    digest check would then refuse to restart a run that had been training happily.
    """

    @pytest.mark.parametrize(
        "iters, f, expected_aux",
        [
            (1, 0.5, 0),  # round(0.5) = 0, not 1
            (3, 0.5, 2),  # round(1.5) = 2
            (5, 0.5, 2),  # round(2.5) = 2, not 3
            (7, 0.5, 4),  # round(3.5) = 4
            (6, 0.25, 2),  # round(1.5) = 2
            (10, 0.25, 2),  # round(2.5) = 2
        ],
    )
    def test_a_half_corpus_share_rounds_to_even(self, iters, f, expected_aux):
        assert _plan(iters=iters, fractions=[f]).n_corpus_iters(FIRST_AUX) == expected_aux

    def test_half_sub_label_shares_round_to_even(self):
        """The same rule one level down: 5 aux iterations at p_as 0.5 give 2 spread, not 3,
        and 5 core iterations at p_cr 0.5 give 2 core-robustness."""
        plan = _plan(iters=10, fractions=[0.5], p_as=0.5, p_cr=0.5)
        assert (plan.n_corpus_iters(FIRST_AUX), plan.n_core_iters) == (5, 5)
        assert _spread_and_robust(plan) == (2, 2)

    def test_half_robustness_shares_round_to_even_per_module(self):
        """20 core iterations at p_cr 0.5 and equal shares give each module round(5.0) = 5;
        at shares 0.25/0.75 they give round(2.5) = 2 and round(7.5) = 8."""
        even = _plan(iters=40, fractions=[0.25, 0.25], p_cr=0.5)
        assert (even.n_core_iters, [_robust(even, k) for k in range(2)]) == (20, [5, 5])
        skewed = _plan(iters=40, fractions=[0.125, 0.375], p_cr=0.5)
        assert (skewed.n_core_iters, [_robust(skewed, k) for k in range(2)]) == (20, [2, 8])


class TestPlanWellFormedness:
    """Properties the gater and the callback rely on holding for EVERY iteration."""

    @pytest.mark.parametrize("fractions", [[0.0], [0.25], [0.5], [0.75], [1.0], [0.25, 0.25], [0.1, 0.2, 0.3]])
    @pytest.mark.parametrize("p_as", [0.0, 0.5, 1.0])
    @pytest.mark.parametrize("p_cr", [0.0, 0.2, 1.0])
    def test_every_iteration_updates_something(self, fractions, p_as, p_cr):
        """GROptimizerGater.arm raises on an iteration updating neither role.

        Swept across the fraction corners because that raise is the only thing standing
        between a malformed plan and a silently wasted optimizer step.
        """
        plan = _plan(fractions=fractions, p_as=p_as, p_cr=p_cr)
        assert (plan.update_core.astype(bool) | plan.update_aux.any(axis=1)).all()

    @pytest.mark.parametrize("fractions", [FRACTIONS, FRACTIONS_2, FRACTIONS_3])
    def test_arrays_share_length_and_dtype(self, fractions):
        plan = _plan(fractions=fractions)
        for field in ("corpus", "update_core", "prior_iters_same_corpus"):
            arr = getattr(plan, field)
            assert arr.shape == (ITERS,), f"{field} has shape {arr.shape}"
            assert arr.dtype == np.int64, f"{field} has dtype {arr.dtype}"
        for field in ("fwd_aux", "update_aux"):
            arr = getattr(plan, field)
            assert arr.shape == (ITERS, len(fractions)), f"{field} has shape {arr.shape}"
            assert arr.dtype == np.int64, f"{field} has dtype {arr.dtype}"

    @pytest.mark.parametrize("fractions", [FRACTIONS, FRACTIONS_2, FRACTIONS_3])
    def test_corpus_values_are_only_the_configured_labels(self, fractions):
        plan = _plan(fractions=fractions)
        assert set(np.unique(plan.corpus)).issubset(set(range(len(fractions) + 1)))
        assert plan.n_aux == len(fractions)
        assert plan.aux_iter_fractions == tuple(fractions)

    @pytest.mark.parametrize("fractions", [FRACTIONS, FRACTIONS_2, FRACTIONS_3])
    def test_indicator_arrays_are_zero_one(self, fractions):
        plan = _plan(fractions=fractions)
        for field in ("fwd_aux", "update_core", "update_aux"):
            assert set(np.unique(getattr(plan, field))).issubset({0, 1}), field

    def test_n_samples_scales_with_global_batch_size(self):
        plan = _plan()
        assert plan.n_samples(FIRST_AUX, 8) == 60 * 8
        assert plan.n_samples(CORE, 8) == 60 * 8
        assert plan.n_samples(FIRST_AUX, 1) + plan.n_samples(CORE, 1) == ITERS

    def test_describe_reports_the_counts_it_claims(self):
        plan = _plan()
        text = plan.describe()
        for fragment in ("iters=120", "core=60", "aux0=60", "aux_spread=30", "core_robustness=12"):
            assert fragment in text, f"{fragment!r} missing from {text!r}"
        assert plan.digest() in text

    def test_describe_names_every_module_at_n_greater_than_one(self):
        """The log line is how an operator confirms the corpus list they configured is the
        one the plan routes; a summary that collapsed the modules would hide a misordering."""
        plan = _plan(fractions=FRACTIONS_3)
        text = plan.describe()
        for k, expected in enumerate([12, 24, 36]):
            assert f"aux{k}={expected}" in text, f"aux{k} missing from {text!r}"


class TestPriorItersSameCorpus:
    """The routed dataset's offset arithmetic is only gapless if this is a true prefix count."""

    @pytest.mark.parametrize("fractions", [FRACTIONS, FRACTIONS_2, FRACTIONS_3])
    def test_matches_an_independent_prefix_count(self, fractions):
        plan = _plan(fractions=fractions)
        counts = {corpus: 0 for corpus in range(plan.n_aux + 1)}
        for i, corpus in enumerate(plan.corpus.tolist()):
            assert int(plan.prior_iters_same_corpus[i]) == counts[corpus], f"iteration {i}"
            counts[corpus] += 1

    def test_each_corpus_offsets_form_a_gapless_range(self):
        """Per corpus the values are exactly 0..n-1 in order — no gaps, no repeats."""
        plan = _plan()
        for corpus, n_expected in ((CORE, 60), (FIRST_AUX, 60)):
            offsets = plan.prior_iters_same_corpus[plan.corpus == corpus]
            assert np.array_equal(offsets, np.arange(n_expected)), f"corpus {corpus}: {offsets}"

    @pytest.mark.parametrize("fractions", [[0.0], [0.5], [1.0], [0.25, 0.25], [0.5, 0.0], [0.1, 0.2, 0.3]])
    def test_gapless_under_every_module_count_and_degenerate_fraction(self, fractions):
        plan = _plan(fractions=fractions)
        for corpus in range(plan.n_aux + 1):
            offsets = plan.prior_iters_same_corpus[plan.corpus == corpus]
            assert np.array_equal(offsets, np.arange(len(offsets))), f"corpus {corpus} offsets are not gapless"


class TestValidation:
    """Bad plan parameters must raise at build time, not produce a degenerate plan."""

    @pytest.mark.parametrize("train_iters", [0, -1, -120])
    def test_non_positive_train_iters_raises(self, train_iters):
        with pytest.raises(ValueError, match="train_iters must be positive"):
            _plan(iters=train_iters)

    def test_an_empty_fraction_list_raises(self):
        """N=0 is not a GR run: there would be no aux module and nothing to route."""
        with pytest.raises(ValueError, match="must name at least one aux corpus"):
            _plan(fractions=[])

    @pytest.mark.parametrize("value", [-0.1, 1.1, 2.0, float("nan")])
    @pytest.mark.parametrize("module", [0, 1])
    def test_out_of_range_fractions_raise(self, module, value):
        fractions = [0.25, 0.25]
        fractions[module] = value
        with pytest.raises(ValueError, match=rf"aux_iter_fractions\[{module}\] must be in"):
            _plan(fractions=fractions)

    @pytest.mark.parametrize("value", [-0.1, 1.1, 2.0, float("nan")])
    @pytest.mark.parametrize("field", ["p_as", "p_cr"])
    def test_out_of_range_probabilities_raise(self, field, value):
        with pytest.raises(ValueError, match=field):
            _plan(**{field: value})

    @pytest.mark.parametrize("fractions", [[0.6, 0.6], [0.5, 0.5, 0.5], [1.0, 0.01]])
    def test_fractions_summing_above_one_raise(self, fractions):
        """The core corpus would need a negative iteration count."""
        with pytest.raises(ValueError, match="must sum to <= 1"):
            _plan(fractions=fractions)

    def test_rounding_that_overruns_the_plan_raises(self):
        """3 iterations at ``[0.5, 0.5]``: each module rounds to 2, and 4 > 3. The fractions
        are individually legal and sum to 1, so only the post-rounding check catches it —
        without it the second module's slice would silently overwrite the first module's."""
        with pytest.raises(ValueError, match="exceed train_iters=3 after rounding"):
            _plan(iters=3, fractions=[0.5, 0.5])

    @pytest.mark.parametrize(("iters", "fractions"), [(5, [0.25, 0.25, 0.25]), (8, [0.05, 0.05, 0.05])])
    def test_robustness_rounding_that_overruns_the_core_iterations_raises(self, iters, fractions):
        """The core-robustness counts round per module and can sum past the core-iteration
        count (5 iterations at three 0.25 fractions: 2 core iterations, each module rounds
        to 1, and 3 > 2). Slicing one permutation would silently hand the last module a
        SHORT slice — a plan that under-delivers core-robustness with no error — so the
        builder must refuse, exactly as it does for the aux-corpus counts."""
        with pytest.raises(ValueError, match="core-robustness"):
            _plan(iters=iters, fractions=fractions, p_cr=1.0)

    def test_single_iteration_plan_is_valid(self):
        """train_iters=1 is the smallest legal plan; it must not trip the round()s."""
        plan = _plan(iters=1)
        assert plan.train_iters == 1
        assert bool(plan.update_core[0]) or bool(plan.update_aux[0].any())


class TestTinyPlansAtEveryCorner:
    """1-3 iterations at every fraction corner still produce a fully well-formed plan.

    The smoke script and the functional tests run plans this short, and every degenerate
    combination sends some ``round(p * n)`` to 0 or to n — the regime where an empty index
    array reaches ``rng.permutation`` and an empty corpus reaches the routed dataset. A plan
    that came out malformed here would surface as a mid-run RuntimeError from the gater, or
    as an out-of-range read on a corpus with no samples.
    """

    @pytest.mark.parametrize("iters", [1, 2, 3])
    @pytest.mark.parametrize(
        "fractions", [[0.0], [0.5], [1.0], [0.0, 0.0], [0.5, 0.0], [0.0, 1.0], [0.25, 0.25], [0.1, 0.2, 0.3]]
    )
    @pytest.mark.parametrize("p_as", [0.0, 1.0])
    @pytest.mark.parametrize("p_cr", [0.0, 1.0])
    def test_a_tiny_plan_is_well_formed(self, iters, fractions, p_as, p_cr):
        plan = _plan(iters=iters, fractions=fractions, p_as=p_as, p_cr=p_cr)
        assert plan.train_iters == iters
        assert plan.n_core_iters + sum(plan.n_corpus_iters(k + FIRST_AUX) for k in range(plan.n_aux)) == iters
        assert (plan.update_core.astype(bool) | plan.update_aux.any(axis=1)).all(), "an iteration updates nothing"
        assert np.array_equal(plan.fwd_aux, plan.update_aux), "aux was forwarded without being updated, or vice versa"
        assert plan.update_aux[plan.corpus == CORE].sum(axis=1).max(initial=0) <= 1, "two gates on one core iteration"
        for corpus in range(plan.n_aux + 1):
            offsets = plan.prior_iters_same_corpus[plan.corpus == corpus]
            assert np.array_equal(offsets, np.arange(len(offsets))), f"corpus {corpus} offsets are not gapless"

    @pytest.mark.parametrize(
        "fractions, empty_corpus", [([0.0], FIRST_AUX), ([1.0], CORE), ([0.5, 0.0], FIRST_AUX + 1)]
    )
    def test_a_corpus_the_plan_never_draws_consumes_no_samples(self, fractions, empty_corpus):
        """The routed dataset sizes each child from this; a non-zero count for a corpus that
        is never drawn would demand samples from a dataset the run has no reason to build."""
        plan = _plan(iters=4, fractions=fractions)
        assert plan.n_samples(empty_corpus, 8) == 0
        assert int((plan.corpus == empty_corpus).sum()) == 0
