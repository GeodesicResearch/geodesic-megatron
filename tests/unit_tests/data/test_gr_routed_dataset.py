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
"""Index-mapping tests for GRRoutedDataset, plus the sampler assumption it rests on.

The dataset's whole correctness argument is one equation — ``idx // GBS`` is the training
iteration — and it is an assumption about a component the dataset never sees: the REAL
``MegatronPretrainingSampler``. So the sampler is driven here for real, across data-parallel
ranks and microbatch counts, rather than restated as a comment: if a future sampler change
interleaves or shuffles indices, label-homogeneous global batches stop being label-homogeneous
and every routed run is silently wrong.

The children are trivial recording stubs on purpose. What is under test is the arithmetic
mapping an index onto (corpus, offset), and a real GPTDataset would only add tokenizer and
``.bin`` file dependencies to an assertion about integers. The same reasoning covers the
provider-dispatch tests at the end of the file, which stub the dataset BUILDER: what is
under test there is which provider a ``GRDatasetConfig`` resolves to and what each corpus's
child config is asked to build, neither of which needs a tokenized corpus on disk.
"""

import numpy as np
import pytest

from megatron.bridge.data import utils as data_utils
from megatron.bridge.data.datasets.gr_routed_dataset import GRRoutedDataset
from megatron.bridge.data.samplers import MegatronPretrainingSampler, build_pretraining_data_loader
from megatron.bridge.data.utils import get_dataset_provider, pretrain_train_valid_test_datasets_provider
from megatron.bridge.training.config import GPTDatasetConfig
from megatron.bridge.training.gradient_routing.config import GRDatasetConfig
from megatron.bridge.training.gradient_routing.plan import CORE, FIRST_AUX, GRPlan, build_gr_plan


class RecordingChild:
    """Indexable stub returning its own label and offset, recording every access."""

    def __init__(self, label: str, n: int):
        self.label = label
        self.n = n
        self.requested: list[int] = []

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, offset):
        self.requested.append(offset)
        return (self.label, offset)


def _label(corpus: int) -> str:
    """The stub label of a corpus: ``core`` or ``aux<module index>``."""
    return "core" if corpus == CORE else f"aux{corpus - FIRST_AUX}"


def _hand_plan(corpus_sequence, n_aux=1):
    """Build a GRPlan directly from a hand-written corpus sequence.

    Bypasses ``build_gr_plan`` deliberately: the mapping under test must hold for an
    arbitrary corpus order, not only for orders the seeded allocator happens to produce.
    """
    corpus = np.asarray(corpus_sequence, dtype=np.int64)
    prior = np.zeros(len(corpus), dtype=np.int64)
    counts = {c: 0 for c in range(n_aux + 1)}
    for i, c in enumerate(corpus.tolist()):
        prior[i] = counts[c]
        counts[c] += 1
    return GRPlan(
        corpus=corpus,
        fwd_aux=np.zeros((len(corpus), n_aux), dtype=np.int64),
        update_core=np.zeros(len(corpus), dtype=np.int64),
        update_aux=np.zeros((len(corpus), n_aux), dtype=np.int64),
        prior_iters_same_corpus=prior,
        plan_seed=0,
        p_as=0.0,
        p_cr=0.0,
        aux_iter_fractions=tuple(float((corpus == k + FIRST_AUX).mean()) for k in range(n_aux)),
    )


AUX1 = FIRST_AUX
AUX2 = FIRST_AUX + 1

# C A C C A: core on iterations 0, 2, 3 and aux on 1, 4 — an uneven order that makes the
# per-corpus offsets diverge from the iteration number in both directions.
HAND_SEQUENCE = [CORE, AUX1, CORE, CORE, AUX1]
#: The same idea with two aux corpora interleaved, so a mapping that collapsed the aux
#: labels onto one child would serve the wrong corpus rather than merely the wrong offset.
HAND_SEQUENCE_2 = [CORE, AUX1, AUX2, CORE, AUX2, AUX1]
GBS = 4


def _children(plan, gbs, extra=None):
    """One exactly-sized RecordingChild per corpus the plan routes."""
    extra = extra or {}
    return {
        corpus: RecordingChild(_label(corpus), plan.n_samples(corpus, gbs) + extra.get(corpus, 0))
        for corpus in range(plan.n_aux + 1)
    }


def _routed(sequence=HAND_SEQUENCE, gbs=GBS, n_aux=1, extra=None):
    plan = _hand_plan(sequence, n_aux=n_aux)
    children = _children(plan, gbs, extra)
    return GRRoutedDataset(children, plan, gbs), plan, children


class TestIndexMapping:
    """index -> (corpus, contiguous gapless offset within that corpus)."""

    def test_every_index_maps_to_the_hand_computed_pair(self):
        dataset, _, _ = _routed()
        # (corpus label, offset) for each of the 5 iterations x 4 samples, by hand:
        # it 0 core (1st core iter) -> core[0..3]; it 1 aux0 (1st) -> aux0[0..3];
        # it 2 core (2nd) -> core[4..7]; it 3 core (3rd) -> core[8..11];
        # it 4 aux0 (2nd) -> aux0[4..7].
        expected = (
            [("core", i) for i in range(0, 4)]
            + [("aux0", i) for i in range(0, 4)]
            + [("core", i) for i in range(4, 8)]
            + [("core", i) for i in range(8, 12)]
            + [("aux0", i) for i in range(4, 8)]
        )
        assert [dataset[i] for i in range(len(dataset))] == expected

    def test_every_index_maps_to_the_hand_computed_pair_with_two_aux_corpora(self):
        """The N=2 mapping, by hand: each aux corpus keeps its OWN offset counter, so aux1's
        second window starts at 4 even though four aux0 samples were served in between."""
        dataset, _, _ = _routed(sequence=HAND_SEQUENCE_2, gbs=2, n_aux=2)
        expected = (
            [("core", 0), ("core", 1)]  # it 0: 1st core
            + [("aux0", 0), ("aux0", 1)]  # it 1: 1st aux0
            + [("aux1", 0), ("aux1", 1)]  # it 2: 1st aux1
            + [("core", 2), ("core", 3)]  # it 3: 2nd core
            + [("aux1", 2), ("aux1", 3)]  # it 4: 2nd aux1
            + [("aux0", 2), ("aux0", 3)]  # it 5: 2nd aux0
        )
        assert [dataset[i] for i in range(len(dataset))] == expected

    def test_length_is_iterations_times_global_batch_size(self):
        dataset, plan, _ = _routed()
        assert len(dataset) == plan.train_iters * GBS == 20

    @pytest.mark.parametrize("sequence, n_aux", [(HAND_SEQUENCE, 1), (HAND_SEQUENCE_2, 2)])
    def test_each_iteration_draws_one_corpus_only(self, sequence, n_aux):
        """Label homogeneity of the global batch — the property routing is built on."""
        dataset, plan, _ = _routed(sequence=sequence, gbs=GBS, n_aux=n_aux)
        for iteration in range(plan.train_iters):
            labels = {dataset[iteration * GBS + j][0] for j in range(GBS)}
            assert labels == {_label(int(plan.corpus[iteration]))}, f"iteration {iteration} mixed corpora: {labels}"

    @pytest.mark.parametrize("sequence, n_aux", [(HAND_SEQUENCE, 1), (HAND_SEQUENCE_2, 2)])
    def test_consumed_offsets_are_exactly_the_contiguous_range(self, sequence, n_aux):
        """Gapless and repeat-free: each corpus is consumed as range(n_iters * GBS)."""
        dataset, plan, children = _routed(sequence=sequence, gbs=GBS, n_aux=n_aux)
        for i in range(len(dataset)):
            dataset[i]
        for corpus, child in children.items():
            n_iters = plan.n_corpus_iters(corpus)
            assert sorted(child.requested) == list(range(n_iters * GBS))
            assert child.requested == sorted(child.requested), "offsets served out of order"

    @pytest.mark.parametrize("gbs", [1, 2, 4, 8])
    @pytest.mark.parametrize("sequence, n_aux", [(HAND_SEQUENCE, 1), (HAND_SEQUENCE_2, 2)])
    def test_mapping_holds_across_global_batch_sizes(self, gbs, sequence, n_aux):
        dataset, plan, children = _routed(sequence=sequence, gbs=gbs, n_aux=n_aux)
        for i in range(len(dataset)):
            label, offset = dataset[i]
            iteration = i // gbs
            assert label == _label(int(plan.corpus[iteration]))
            assert offset == int(plan.prior_iters_same_corpus[iteration]) * gbs + i % gbs
        for corpus, child in children.items():
            assert sorted(child.requested) == list(range(plan.n_samples(corpus, gbs)))

    @pytest.mark.parametrize(
        "sequence, n_aux",
        [
            ([CORE] * 6, 1),  # aux corpus never drawn
            ([AUX1] * 6, 1),  # core corpus never drawn
            ([AUX1, CORE], 1),  # aux first
            ([CORE] * 6, 2),  # neither aux corpus drawn
            ([AUX2] * 4, 2),  # only the SECOND aux corpus is drawn
            ([AUX2, AUX1], 2),  # the aux corpora out of index order
        ],
    )
    def test_degenerate_corpus_sequences(self, sequence, n_aux):
        dataset, plan, children = _routed(sequence=sequence, gbs=2, n_aux=n_aux)
        for i in range(len(dataset)):
            dataset[i]
        for corpus, child in children.items():
            assert sorted(child.requested) == list(range(plan.n_corpus_iters(corpus) * 2))

    def test_numpy_integer_indices_are_accepted(self):
        """The torch DataLoader hands numpy ints through; a raw numpy int must not fall
        through the range check or index the child with a numpy scalar."""
        dataset, _, _ = _routed()
        assert dataset[np.int64(5)] == dataset[5]
        assert isinstance(dataset[np.int64(5)][1], int)

    def test_larger_children_than_needed_are_allowed_and_unused_tail_untouched(self):
        dataset, plan, children = _routed(extra={CORE: 100, AUX1: 100})
        for i in range(len(dataset)):
            dataset[i]
        for corpus, child in children.items():
            assert max(child.requested) == plan.n_samples(corpus, GBS) - 1


class TestRealPlanConsumption:
    """Whole-plan boundaries against plans the seeded allocator actually produces.

    The hand-written sequences above pin the arithmetic; these pin that it still lands
    exactly on the end of each corpus when the corpus order comes from ``build_gr_plan``
    and the children are sized from ``plan.n_samples`` — the sizing the real provider uses.
    An off-by-one in either direction is invisible until the final iterations of a run:
    either the last window reads past the child (IndexError, days in) or the corpus is
    quietly never finished.
    """

    @pytest.mark.parametrize(
        "seed, iters, fractions, gbs",
        [
            (1234, 12, [0.5], 4),
            (7, 20, [0.25], 2),
            (99, 9, [0.75], 3),
            (5, 6, [1.0], 4),  # aux only: the core child is never touched
            (5, 6, [0.0], 4),  # core only
            (11, 40, [0.5], 1),  # GBS 1: every iteration is a single sample
            (1234, 12, [0.25, 0.25], 4),  # two aux corpora
            (7, 20, [0.2, 0.3], 2),  # unequal shares
            (99, 30, [0.1, 0.2, 0.3], 3),  # three aux corpora
            (5, 8, [0.5, 0.0], 2),  # a module the plan never routes to
        ],
    )
    def test_each_corpus_is_consumed_in_order_exactly_to_its_end(self, seed, iters, fractions, gbs):
        plan = build_gr_plan(plan_seed=seed, train_iters=iters, aux_iter_fractions=fractions, p_as=0.5, p_cr=0.2)
        children = _children(plan, gbs)
        dataset = GRRoutedDataset(children, plan, gbs)

        for index in range(len(dataset)):
            dataset[index]

        for corpus, child in children.items():
            expected = list(range(plan.n_samples(corpus, gbs)))
            assert child.requested == expected, "offsets are not a gapless in-order sweep of the corpus"
        assert sum(len(child.requested) for child in children.values()) == len(dataset)

    @pytest.mark.parametrize(
        "seed, iters, fractions, gbs",
        [(1234, 12, [0.5], 4), (7, 20, [0.25], 2), (99, 9, [0.75], 3), (7, 20, [0.2, 0.3], 2)],
    )
    def test_the_final_index_serves_the_final_sample_of_its_corpus(self, seed, iters, fractions, gbs):
        """The last index of the last iteration must land on the last sample the plan sized
        that corpus for — one past it is an IndexError at the very end of a run."""
        plan = build_gr_plan(plan_seed=seed, train_iters=iters, aux_iter_fractions=fractions, p_as=0.5, p_cr=0.2)
        dataset = GRRoutedDataset(_children(plan, gbs), plan, gbs)
        last_corpus = int(plan.corpus[-1])
        assert dataset[len(dataset) - 1] == (_label(last_corpus), plan.n_samples(last_corpus, gbs) - 1)

    @pytest.mark.parametrize(
        "seed, iters, fractions, gbs", [(1234, 12, [0.5], 4), (7, 20, [0.25], 2), (1234, 24, [0.25, 0.25], 4)]
    )
    def test_every_iteration_reads_its_own_contiguous_window(self, seed, iters, fractions, gbs):
        """Per iteration the offsets are one unbroken block, and consecutive iterations of a
        corpus get consecutive blocks — the property that lets a child dataset be built with
        exactly ``n_iters * GBS`` samples instead of an epoch-looped superset."""
        plan = build_gr_plan(plan_seed=seed, train_iters=iters, aux_iter_fractions=fractions, p_as=0.5, p_cr=0.2)
        dataset = GRRoutedDataset(_children(plan, gbs), plan, gbs)
        next_window = {corpus: 0 for corpus in range(plan.n_aux + 1)}
        for iteration in range(plan.train_iters):
            corpus = int(plan.corpus[iteration])
            offsets = [dataset[iteration * gbs + j][1] for j in range(gbs)]
            assert offsets == list(range(next_window[corpus], next_window[corpus] + gbs)), f"iteration {iteration}"
            next_window[corpus] += gbs


class TestConstructionRefusals:
    """A short child would silently wrap or IndexError deep in training; refuse at build."""

    @pytest.mark.parametrize("short_corpus", [CORE, AUX1, AUX2])
    def test_undersized_child_raises(self, short_corpus):
        plan = _hand_plan(HAND_SEQUENCE_2, n_aux=2)
        children = _children(plan, GBS)
        children[short_corpus].n -= 1
        with pytest.raises(ValueError, match=f"GR corpus {short_corpus} dataset provides"):
            GRRoutedDataset(children, plan, GBS)

    def test_exactly_sized_children_are_accepted(self):
        """The refusal is ``<``, not ``<=`` — exact sizing is the normal case."""
        plan = _hand_plan(HAND_SEQUENCE)
        GRRoutedDataset(_children(plan, GBS), plan, GBS)

    @pytest.mark.parametrize("gbs", [0, -1])
    def test_non_positive_global_batch_size_raises(self, gbs):
        plan = _hand_plan(HAND_SEQUENCE)
        with pytest.raises(ValueError, match="global_batch_size must be positive"):
            GRRoutedDataset({CORE: RecordingChild("core", 100), AUX1: RecordingChild("aux0", 100)}, plan, gbs)

    def test_empty_corpus_needs_no_samples(self):
        """A corpus the plan never draws may legitimately be an empty dataset."""
        plan = _hand_plan([CORE] * 4)
        GRRoutedDataset({CORE: RecordingChild("core", 4 * GBS), AUX1: RecordingChild("aux0", 0)}, plan, GBS)

    @pytest.mark.parametrize(
        "labels",
        [
            (CORE,),  # the aux child is missing entirely
            (AUX1,),  # the core child is missing
            (CORE, AUX1, AUX2),  # one child too many for a 1-module plan
            (CORE, AUX2),  # right count, wrong labels: module 1 instead of module 0
            (1, 2),  # off by one: no core child at all
        ],
    )
    def test_a_wrong_child_label_set_raises(self, labels):
        """The children arrive as a dict keyed by corpus label, so a provider that built the
        wrong number of corpora — or keyed them from 1 instead of 0 — would otherwise serve
        the wrong corpus (or KeyError mid-epoch) rather than fail at construction."""
        plan = _hand_plan(HAND_SEQUENCE)
        children = {label: RecordingChild(_label(label), 1000) for label in labels}
        with pytest.raises(ValueError, match="needs one child per corpus label"):
            GRRoutedDataset(children, plan, GBS)


class TestIndexBounds:
    @pytest.mark.parametrize("bad", [-1, 20, 21, 1000])
    def test_out_of_range_index_raises_index_error(self, bad):
        dataset, _, _ = _routed()
        with pytest.raises(IndexError, match="out of range"):
            dataset[bad]

    def test_last_valid_index_is_served(self):
        dataset, _, _ = _routed()
        assert dataset[len(dataset) - 1] == ("aux0", 7)


class TestSamplerAttribution:
    """``idx // GBS == iteration`` under the REAL sampler, for every rank.

    ``MegatronPretrainingSampler`` yields one microbatch per call; an iteration is
    ``num_microbatches`` consecutive yields per rank. The claim being pinned is that the
    union of what all DP ranks see during iteration k is exactly ``[k*GBS, (k+1)*GBS)``.
    """

    @pytest.mark.parametrize(
        "dp_size, micro_batch_size, num_microbatches",
        [
            (1, 1, 1),  # degenerate
            (1, 2, 4),  # single rank, gradient accumulation
            (2, 1, 4),  # data parallel, accumulation
            (4, 2, 2),  # both
            (8, 1, 1),  # wide DP, no accumulation
            (2, 4, 8),  # the shape a real GR run has (GBS 64)
            (3, 2, 2),  # DP that is not a power of two
            (4, 1, 16),  # accumulation-heavy: many microbatches per iteration
            (16, 2, 1),  # wide DP, one microbatch per rank
            (2, 1, 512),  # the shipped mainline shape (GBS 1024 at DP 2, mbs 1)
        ],
    )
    def test_every_yielded_index_belongs_to_its_iteration(self, dp_size, micro_batch_size, num_microbatches):
        gbs = dp_size * micro_batch_size * num_microbatches
        n_iters = 5
        seen_per_iteration = [set() for _ in range(n_iters)]

        for rank in range(dp_size):
            sampler = MegatronPretrainingSampler(
                total_samples=n_iters * gbs,
                consumed_samples=0,
                micro_batch_size=micro_batch_size,
                data_parallel_rank=rank,
                data_parallel_size=dp_size,
                drop_last=True,
            )
            for microbatch_index, indices in enumerate(sampler):
                iteration = microbatch_index // num_microbatches
                assert len(indices) == micro_batch_size
                for idx in indices:
                    assert idx // gbs == iteration, (
                        f"rank {rank} saw index {idx} (iteration {idx // gbs}) while serving iteration {iteration}"
                    )
                seen_per_iteration[iteration].update(indices)

        for iteration, seen in enumerate(seen_per_iteration):
            assert seen == set(range(iteration * gbs, (iteration + 1) * gbs)), (
                f"iteration {iteration} did not cover its global batch exactly"
            )

    @pytest.mark.parametrize(
        "dp_size, micro_batch_size, num_microbatches", [(2, 2, 2), (4, 1, 2), (1, 1, 8), (3, 2, 4), (8, 1, 2)]
    )
    @pytest.mark.parametrize("resume_at", [1, 2, 5])
    def test_resume_midway_keeps_the_iteration_mapping(self, dp_size, micro_batch_size, num_microbatches, resume_at):
        """A restart sets consumed_samples to a multiple of GBS; the mapping must survive it.

        Resumes are routine here (ft_launcher restarts, singleton chains), and the plan is
        re-derived from the iteration number alone — so a resumed sampler whose indices no
        longer satisfy ``idx // GBS == iteration`` would feed iteration k's routing decisions
        with some other iteration's corpus, on every rank, for the rest of the run. The
        coverage assertion is what distinguishes "each index is in range" from "the global
        batch is exactly this iteration's window": ``resume_at=5`` is the last iteration, so
        the tail of the plan is exercised too.
        """
        gbs = dp_size * micro_batch_size * num_microbatches
        n_iters = 6
        seen_per_iteration = {iteration: set() for iteration in range(resume_at, n_iters)}

        for rank in range(dp_size):
            sampler = MegatronPretrainingSampler(
                total_samples=n_iters * gbs,
                consumed_samples=resume_at * gbs,
                micro_batch_size=micro_batch_size,
                data_parallel_rank=rank,
                data_parallel_size=dp_size,
                drop_last=True,
            )
            for microbatch_index, indices in enumerate(sampler):
                iteration = resume_at + microbatch_index // num_microbatches
                assert len(indices) == micro_batch_size
                for idx in indices:
                    assert idx // gbs == iteration, (
                        f"rank {rank} saw index {idx} (iteration {idx // gbs}) while serving iteration {iteration}"
                    )
                seen_per_iteration[iteration].update(indices)

        for iteration, seen in seen_per_iteration.items():
            assert seen == set(range(iteration * gbs, (iteration + 1) * gbs)), (
                f"iteration {iteration} did not cover its global batch exactly after a resume"
            )

    @pytest.mark.parametrize("resume_at", [0, 1, 4])
    @pytest.mark.parametrize("fractions", [[0.5], [0.25, 0.25]])
    def test_a_resumed_run_still_serves_one_corpus_per_iteration(self, resume_at, fractions):
        """The end-to-end resume: plan -> dataset -> sampler restarted mid-plan. The dataset
        is stateless in the iteration index, so a resumed run must land on the same corpus
        (and the same per-corpus window) the original run would have used for that iteration."""
        dp_size, micro_batch_size, num_microbatches = 2, 2, 2
        gbs = dp_size * micro_batch_size * num_microbatches  # 8
        plan = build_gr_plan(plan_seed=7, train_iters=6, aux_iter_fractions=fractions, p_as=0.5, p_cr=0.2)
        dataset = GRRoutedDataset(_children(plan, gbs), plan, gbs)
        for rank in range(dp_size):
            sampler = MegatronPretrainingSampler(
                total_samples=len(dataset),
                consumed_samples=resume_at * gbs,
                micro_batch_size=micro_batch_size,
                data_parallel_rank=rank,
                data_parallel_size=dp_size,
                drop_last=True,
            )
            for microbatch_index, indices in enumerate(sampler):
                iteration = resume_at + microbatch_index // num_microbatches
                expected = _label(int(plan.corpus[iteration]))
                for idx in indices:
                    label, offset = dataset[idx]
                    assert label == expected, f"rank {rank}, iteration {iteration}: served {label}"
                    assert offset == int(plan.prior_iters_same_corpus[iteration]) * gbs + idx % gbs

    @pytest.mark.parametrize("dp_size, micro_batch_size, num_microbatches", [(1, 2, 2), (2, 1, 4)])
    def test_the_single_dataloader_type_builds_the_iteration_aligned_sampler(
        self, dp_size, micro_batch_size, num_microbatches
    ):
        """``dataloader_type: "single"`` is the guarded precondition; this is what it buys.

        The launch guards refuse anything but "single", and every other test here constructs
        ``MegatronPretrainingSampler`` by hand — so nothing pinned that the string actually
        resolves to that sampler. It is ``build_pretraining_data_loader`` that decides, and a
        future re-mapping of "single" onto a shuffling sampler would break label homogeneity
        while leaving the guard, the dataset and the plan all looking correct.
        """
        gbs = dp_size * micro_batch_size * num_microbatches
        plan = build_gr_plan(plan_seed=3, train_iters=5, aux_iter_fractions=[0.5], p_as=0.5, p_cr=0.2)
        dataset = GRRoutedDataset(_children(plan, gbs), plan, gbs)
        for rank in range(dp_size):
            loader = build_pretraining_data_loader(
                dataset=dataset,
                consumed_samples=0,
                dataloader_type="single",
                micro_batch_size=micro_batch_size,
                num_workers=0,
                data_sharding=False,
                pin_memory=False,
                data_parallel_rank=rank,
                data_parallel_size=dp_size,
                drop_last=True,
            )
            assert isinstance(loader.batch_sampler, MegatronPretrainingSampler)

            batches = list(loader)
            assert len(batches) == plan.train_iters * num_microbatches
            for microbatch_index, (labels, _offsets) in enumerate(batches):
                iteration = microbatch_index // num_microbatches
                expected = _label(int(plan.corpus[iteration]))
                assert set(labels) == {expected}, f"rank {rank}, iteration {iteration}: {set(labels)}"

    def test_routed_dataset_through_the_real_sampler_serves_one_corpus_per_iteration(self):
        """End-to-end: plan -> dataset -> real sampler. What each rank actually receives."""
        dp_size, micro_batch_size, num_microbatches = 2, 2, 2
        gbs = dp_size * micro_batch_size * num_microbatches  # 8
        plan = build_gr_plan(plan_seed=7, train_iters=6, aux_iter_fractions=[0.25, 0.25], p_as=0.5, p_cr=0.2)
        dataset = GRRoutedDataset(_children(plan, gbs), plan, gbs)
        for rank in range(dp_size):
            sampler = MegatronPretrainingSampler(
                total_samples=len(dataset),
                consumed_samples=0,
                micro_batch_size=micro_batch_size,
                data_parallel_rank=rank,
                data_parallel_size=dp_size,
                drop_last=True,
            )
            for microbatch_index, indices in enumerate(sampler):
                iteration = microbatch_index // num_microbatches
                expected = _label(int(plan.corpus[iteration]))
                labels = {dataset[idx][0] for idx in indices}
                assert labels == {expected}, f"rank {rank}, iteration {iteration}: {labels}"


RETAIN_PATHS = ["0.5", "/data/core_a_text_document", "0.5", "/data/core_b_text_document"]
AUX_PATHS = [["/data/aux0_text_document"], ["/data/aux1_text_document"]]


def _gr_dataset_config(plan, gbs=GBS, aux_data_paths=None):
    return GRDatasetConfig(
        retain_data_path=RETAIN_PATHS,
        aux_data_paths=aux_data_paths if aux_data_paths is not None else AUX_PATHS[: plan.n_aux],
        gr_plan=plan,
        gr_global_batch_size=gbs,
        seq_length=1024,
        split="9999,1,0",
        random_seed=1234,
        reset_position_ids=False,
        reset_attention_mask=False,
        eod_mask_loss=False,
        mmap_bin_files=True,
        dataloader_type="single",
    )


def _config_label(config) -> str:
    """Which corpus a child config carries, read off its blend paths."""
    for k in range(len(AUX_PATHS)):
        if any(f"aux{k}" in path for path in (config.data_path or [])):
            return f"aux{k}"
    return "core"


class TestDatasetProviderDispatch:
    """``get_dataset_provider`` must resolve a GRDatasetConfig, and resolve it to pretrain.

    This is where an integration bug lived: the registry lookup is ``_REGISTRY[type(cfg)]``,
    an EXACT type match. Subclassing ``GPTDatasetConfig`` therefore buys nothing — an
    unregistered GRDatasetConfig raises ``KeyError`` at data-iterator setup, which is after
    the model and the optimizer have already been built.
    """

    @pytest.fixture
    def plan(self):
        return build_gr_plan(1234, 8, [0.5], 0.5, 0.2)

    def test_gr_dataset_config_resolves_to_the_pretrain_provider(self, plan):
        assert get_dataset_provider(_gr_dataset_config(plan)) is pretrain_train_valid_test_datasets_provider

    def test_registry_lookup_is_by_exact_type_not_isinstance(self, plan):
        """Pins WHY registration is required: a subclass does not inherit its parent's entry."""
        config = _gr_dataset_config(plan)
        assert isinstance(config, GPTDatasetConfig), "GRDatasetConfig is still a GPTDatasetConfig subclass"
        assert type(config) is not GPTDatasetConfig
        assert GRDatasetConfig in data_utils._REGISTRY
        assert data_utils._REGISTRY[GRDatasetConfig] is pretrain_train_valid_test_datasets_provider

    def test_child_config_resolves_to_the_same_provider(self, plan):
        """The per-corpus child is re-classed to plain GPTDatasetConfig, so it resolves
        through the parent's registry entry rather than recursing into the GR branch."""
        child = _gr_dataset_config(plan).build_child_config(AUX_PATHS[0])
        assert get_dataset_provider(child) is pretrain_train_valid_test_datasets_provider

    def test_gr_config_does_not_take_the_protocol_path(self, plan):
        """``get_dataset_provider`` checks the DatasetProvider ABC FIRST. If GRDatasetConfig
        ever acquired that base, dispatch would silently bypass ``_build_gr_routed_datasets``
        and the routing plan would never be consulted."""
        from megatron.bridge.training.config import DatasetProvider

        assert not isinstance(_gr_dataset_config(plan), DatasetProvider)

    def test_the_registry_import_is_module_level(self):
        """A deferred, inside-the-function import would leave the registry without the entry
        for any caller that reached it by another path."""
        assert getattr(data_utils, "GRDatasetConfig", None) is GRDatasetConfig


class _FakeBuilder:
    """Stands in for BlendedMegatronDatasetBuilder, recording what each corpus was asked for."""

    calls: list = []

    def __init__(self, dataset_cls, sizes, is_built_on_rank, config):
        self.sizes = sizes
        self.config = config
        _FakeBuilder.calls.append({"dataset_cls": dataset_cls, "sizes": sizes, "config": config})

    def build(self):
        return RecordingChild(_config_label(self.config), self.sizes[0]), None, None


@pytest.fixture
def fake_builder(monkeypatch):
    _FakeBuilder.calls = []
    monkeypatch.setattr(data_utils, "BlendedMegatronDatasetBuilder", _FakeBuilder)
    return _FakeBuilder


class TestProviderBuildsEveryCorpus:
    """End-to-end through the real provider: one child per corpus, each sized from the plan."""

    @pytest.fixture(params=[[0.5], [0.25, 0.25]], ids=["one_aux", "two_aux"])
    def plan(self, request):
        return build_gr_plan(1234, 8, request.param, 0.5, 0.2)

    def test_provider_returns_a_routed_dataset_and_no_val_or_test(self, plan, fake_builder):
        train, valid, test = pretrain_train_valid_test_datasets_provider(
            [plan.train_iters * GBS, 0, 0], _gr_dataset_config(plan)
        )
        assert isinstance(train, GRRoutedDataset)
        assert valid is None and test is None, "GR runs train with eval_iters 0 (enforced by the guards)"
        assert len(train) == plan.train_iters * GBS

    def test_each_corpus_is_built_once_from_its_own_blend(self, plan, fake_builder):
        pretrain_train_valid_test_datasets_provider([plan.train_iters * GBS, 0, 0], _gr_dataset_config(plan))

        assert len(fake_builder.calls) == plan.n_aux + 1
        by_corpus = {_config_label(c["config"]): c for c in fake_builder.calls}
        expected = {"core"} | {f"aux{k}" for k in range(plan.n_aux)}
        assert set(by_corpus) == expected, "a corpus was built twice or not at all"
        assert by_corpus["core"]["config"].data_path == RETAIN_PATHS
        for k in range(plan.n_aux):
            assert by_corpus[f"aux{k}"]["config"].data_path == AUX_PATHS[k]
        for corpus in range(plan.n_aux + 1):
            call = by_corpus[_label(corpus)]
            assert call["sizes"] == [plan.n_samples(corpus, GBS), 0, 0]
            assert type(call["config"]) is GPTDatasetConfig, "a child must not stay a GRDatasetConfig"

    def test_routed_dataset_serves_the_planned_corpus_per_iteration(self, plan, fake_builder):
        train, _, _ = pretrain_train_valid_test_datasets_provider(
            [plan.train_iters * GBS, 0, 0], _gr_dataset_config(plan)
        )
        for iteration in range(plan.train_iters):
            expected = _label(int(plan.corpus[iteration]))
            assert {train[iteration * GBS + j][0] for j in range(GBS)} == {expected}

    @pytest.mark.parametrize("delta", [-1, 1, GBS])
    def test_sizing_mismatch_raises(self, plan, fake_builder, delta):
        """train_iters or GBS changing after the plan was built would silently truncate or
        overrun the routed window; the provider refuses instead of serving a short window."""
        with pytest.raises(ValueError, match="GR dataset sizing mismatch"):
            pretrain_train_valid_test_datasets_provider(
                [plan.train_iters * GBS + delta, 0, 0], _gr_dataset_config(plan)
            )
