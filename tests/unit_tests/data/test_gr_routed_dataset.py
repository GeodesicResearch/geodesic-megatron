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
from megatron.bridge.training.gradient_routing.plan import FORGET, RETAIN, GRPlan, build_gr_plan


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


def _hand_plan(corpus_sequence):
    """Build a GRPlan directly from a hand-written corpus sequence.

    Bypasses ``build_gr_plan`` deliberately: the mapping under test must hold for an
    arbitrary corpus order, not only for orders the seeded allocator happens to produce.
    """
    corpus = np.asarray(corpus_sequence, dtype=np.int64)
    prior, counts = np.zeros(len(corpus), dtype=np.int64), {RETAIN: 0, FORGET: 0}
    for i, c in enumerate(corpus.tolist()):
        prior[i] = counts[c]
        counts[c] += 1
    zeros = np.zeros(len(corpus), dtype=np.int64)
    return GRPlan(
        corpus=corpus,
        fwd_aux=zeros.copy(),
        update_core=zeros.copy(),
        update_aux=zeros.copy(),
        prior_iters_same_corpus=prior,
        plan_seed=0,
        p_as=0.0,
        p_cr=0.0,
        forget_iter_fraction=float((corpus == FORGET).mean()),
    )


# R F R R F: retain on iterations 0, 2, 3 and forget on 1, 4 — an uneven order that makes
# the per-corpus offsets diverge from the iteration number in both directions.
HAND_SEQUENCE = [RETAIN, FORGET, RETAIN, RETAIN, FORGET]
GBS = 4


def _routed(sequence=HAND_SEQUENCE, gbs=GBS, retain_extra=0, forget_extra=0):
    plan = _hand_plan(sequence)
    retain = RecordingChild("retain", plan.n_samples(RETAIN, gbs) + retain_extra)
    forget = RecordingChild("forget", plan.n_samples(FORGET, gbs) + forget_extra)
    return GRRoutedDataset(retain, forget, plan, gbs), plan, retain, forget


class TestIndexMapping:
    """index -> (corpus, contiguous gapless offset within that corpus)."""

    def test_every_index_maps_to_the_hand_computed_pair(self):
        dataset, _, _, _ = _routed()
        # (corpus label, offset) for each of the 5 iterations x 4 samples, by hand:
        # it 0 retain (1st retain iter) -> retain[0..3]; it 1 forget (1st) -> forget[0..3];
        # it 2 retain (2nd) -> retain[4..7]; it 3 retain (3rd) -> retain[8..11];
        # it 4 forget (2nd) -> forget[4..7].
        expected = (
            [("retain", i) for i in range(0, 4)]
            + [("forget", i) for i in range(0, 4)]
            + [("retain", i) for i in range(4, 8)]
            + [("retain", i) for i in range(8, 12)]
            + [("forget", i) for i in range(4, 8)]
        )
        assert [dataset[i] for i in range(len(dataset))] == expected

    def test_length_is_iterations_times_global_batch_size(self):
        dataset, plan, _, _ = _routed()
        assert len(dataset) == plan.train_iters * GBS == 20

    def test_each_iteration_draws_one_corpus_only(self):
        """Label homogeneity of the global batch — the property routing is built on."""
        dataset, plan, _, _ = _routed()
        for iteration in range(plan.train_iters):
            labels = {dataset[iteration * GBS + j][0] for j in range(GBS)}
            expected = "forget" if plan.corpus[iteration] == FORGET else "retain"
            assert labels == {expected}, f"iteration {iteration} mixed corpora: {labels}"

    def test_consumed_offsets_are_exactly_the_contiguous_range(self):
        """Gapless and repeat-free: each corpus is consumed as range(n_iters * GBS)."""
        dataset, plan, retain, forget = _routed()
        for i in range(len(dataset)):
            dataset[i]
        for child, corpus in ((retain, RETAIN), (forget, FORGET)):
            n_iters = int((plan.corpus == corpus).sum())
            assert sorted(child.requested) == list(range(n_iters * GBS))
            assert child.requested == sorted(child.requested), "offsets served out of order"

    @pytest.mark.parametrize("gbs", [1, 2, 4, 8])
    def test_mapping_holds_across_global_batch_sizes(self, gbs):
        dataset, plan, retain, forget = _routed(gbs=gbs)
        for i in range(len(dataset)):
            label, offset = dataset[i]
            iteration = i // gbs
            expected = "forget" if plan.corpus[iteration] == FORGET else "retain"
            assert label == expected
            assert offset == int(plan.prior_iters_same_corpus[iteration]) * gbs + i % gbs
        assert sorted(retain.requested) == list(range(plan.n_samples(RETAIN, gbs)))
        assert sorted(forget.requested) == list(range(plan.n_samples(FORGET, gbs)))

    @pytest.mark.parametrize(
        "sequence",
        [
            [RETAIN] * 6,  # forget corpus never drawn
            [FORGET] * 6,  # retain corpus never drawn
            [FORGET, RETAIN],  # forget first
        ],
    )
    def test_degenerate_corpus_sequences(self, sequence):
        dataset, plan, retain, forget = _routed(sequence=sequence, gbs=2)
        for i in range(len(dataset)):
            dataset[i]
        for child, corpus in ((retain, RETAIN), (forget, FORGET)):
            assert sorted(child.requested) == list(range(int((plan.corpus == corpus).sum()) * 2))

    def test_numpy_integer_indices_are_accepted(self):
        """The torch DataLoader hands numpy ints through; a raw numpy int must not fall
        through the range check or index the child with a numpy scalar."""
        dataset, _, _, _ = _routed()
        assert dataset[np.int64(5)] == dataset[5]
        assert isinstance(dataset[np.int64(5)][1], int)

    def test_larger_children_than_needed_are_allowed_and_unused_tail_untouched(self):
        dataset, plan, retain, forget = _routed(retain_extra=100, forget_extra=100)
        for i in range(len(dataset)):
            dataset[i]
        assert max(retain.requested) == plan.n_samples(RETAIN, GBS) - 1
        assert max(forget.requested) == plan.n_samples(FORGET, GBS) - 1


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
        "seed, iters, f, gbs",
        [
            (1234, 12, 0.5, 4),
            (7, 20, 0.25, 2),
            (99, 9, 0.75, 3),
            (5, 6, 1.0, 4),  # forget only: the retain child is never touched
            (5, 6, 0.0, 4),  # retain only
            (11, 40, 0.5, 1),  # GBS 1: every iteration is a single sample
        ],
    )
    def test_each_corpus_is_consumed_in_order_exactly_to_its_end(self, seed, iters, f, gbs):
        plan = build_gr_plan(plan_seed=seed, train_iters=iters, forget_iter_fraction=f, p_as=0.5, p_cr=0.2)
        retain = RecordingChild("retain", plan.n_samples(RETAIN, gbs))
        forget = RecordingChild("forget", plan.n_samples(FORGET, gbs))
        dataset = GRRoutedDataset(retain, forget, plan, gbs)

        for index in range(len(dataset)):
            dataset[index]

        for child, corpus in ((retain, RETAIN), (forget, FORGET)):
            expected = list(range(plan.n_samples(corpus, gbs)))
            assert child.requested == expected, "offsets are not a gapless in-order sweep of the corpus"
        assert len(retain.requested) + len(forget.requested) == len(dataset)

    @pytest.mark.parametrize("seed, iters, f, gbs", [(1234, 12, 0.5, 4), (7, 20, 0.25, 2), (99, 9, 0.75, 3)])
    def test_the_final_index_serves_the_final_sample_of_its_corpus(self, seed, iters, f, gbs):
        """The last index of the last iteration must land on the last sample the plan sized
        that corpus for — one past it is an IndexError at the very end of a run."""
        plan = build_gr_plan(plan_seed=seed, train_iters=iters, forget_iter_fraction=f, p_as=0.5, p_cr=0.2)
        dataset = GRRoutedDataset(
            RecordingChild("retain", plan.n_samples(RETAIN, gbs)),
            RecordingChild("forget", plan.n_samples(FORGET, gbs)),
            plan,
            gbs,
        )
        last_corpus = int(plan.corpus[-1])
        label = "forget" if last_corpus == FORGET else "retain"
        assert dataset[len(dataset) - 1] == (label, plan.n_samples(last_corpus, gbs) - 1)

    @pytest.mark.parametrize("seed, iters, f, gbs", [(1234, 12, 0.5, 4), (7, 20, 0.25, 2)])
    def test_every_iteration_reads_its_own_contiguous_window(self, seed, iters, f, gbs):
        """Per iteration the offsets are one unbroken block, and consecutive iterations of a
        corpus get consecutive blocks — the property that lets a child dataset be built with
        exactly ``n_iters * GBS`` samples instead of an epoch-looped superset."""
        plan = build_gr_plan(plan_seed=seed, train_iters=iters, forget_iter_fraction=f, p_as=0.5, p_cr=0.2)
        dataset = GRRoutedDataset(
            RecordingChild("retain", plan.n_samples(RETAIN, gbs)),
            RecordingChild("forget", plan.n_samples(FORGET, gbs)),
            plan,
            gbs,
        )
        next_window = {RETAIN: 0, FORGET: 0}
        for iteration in range(plan.train_iters):
            corpus = int(plan.corpus[iteration])
            offsets = [dataset[iteration * gbs + j][1] for j in range(gbs)]
            assert offsets == list(range(next_window[corpus], next_window[corpus] + gbs)), f"iteration {iteration}"
            next_window[corpus] += gbs


class TestConstructionRefusals:
    """A short child would silently wrap or IndexError deep in training; refuse at build."""

    @pytest.mark.parametrize("short_corpus", ["retain", "forget"])
    def test_undersized_child_raises(self, short_corpus):
        plan = _hand_plan(HAND_SEQUENCE)
        sizes = {"retain": plan.n_samples(RETAIN, GBS), "forget": plan.n_samples(FORGET, GBS)}
        sizes[short_corpus] -= 1
        with pytest.raises(ValueError, match=f"GR {short_corpus} dataset provides"):
            GRRoutedDataset(
                RecordingChild("retain", sizes["retain"]), RecordingChild("forget", sizes["forget"]), plan, GBS
            )

    def test_exactly_sized_children_are_accepted(self):
        """The refusal is ``<``, not ``<=`` — exact sizing is the normal case."""
        plan = _hand_plan(HAND_SEQUENCE)
        GRRoutedDataset(
            RecordingChild("retain", plan.n_samples(RETAIN, GBS)),
            RecordingChild("forget", plan.n_samples(FORGET, GBS)),
            plan,
            GBS,
        )

    @pytest.mark.parametrize("gbs", [0, -1])
    def test_non_positive_global_batch_size_raises(self, gbs):
        plan = _hand_plan(HAND_SEQUENCE)
        with pytest.raises(ValueError, match="global_batch_size must be positive"):
            GRRoutedDataset(RecordingChild("retain", 100), RecordingChild("forget", 100), plan, gbs)

    def test_empty_corpus_needs_no_samples(self):
        """A corpus the plan never draws may legitimately be an empty dataset."""
        plan = _hand_plan([RETAIN] * 4)
        GRRoutedDataset(RecordingChild("retain", 4 * GBS), RecordingChild("forget", 0), plan, GBS)


class TestIndexBounds:
    @pytest.mark.parametrize("bad", [-1, 20, 21, 1000])
    def test_out_of_range_index_raises_index_error(self, bad):
        dataset, _, _, _ = _routed()
        with pytest.raises(IndexError, match="out of range"):
            dataset[bad]

    def test_last_valid_index_is_served(self):
        dataset, _, _, _ = _routed()
        assert dataset[len(dataset) - 1] == ("forget", 7)


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
    def test_a_resumed_run_still_serves_one_corpus_per_iteration(self, resume_at):
        """The end-to-end resume: plan -> dataset -> sampler restarted mid-plan. The dataset
        is stateless in the iteration index, so a resumed run must land on the same corpus
        (and the same per-corpus window) the original run would have used for that iteration."""
        dp_size, micro_batch_size, num_microbatches = 2, 2, 2
        gbs = dp_size * micro_batch_size * num_microbatches  # 8
        plan = build_gr_plan(plan_seed=7, train_iters=6, forget_iter_fraction=0.5, p_as=0.5, p_cr=0.2)
        dataset = GRRoutedDataset(
            RecordingChild("retain", plan.n_samples(RETAIN, gbs)),
            RecordingChild("forget", plan.n_samples(FORGET, gbs)),
            plan,
            gbs,
        )
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
                expected = "forget" if plan.corpus[iteration] == FORGET else "retain"
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
        plan = build_gr_plan(plan_seed=3, train_iters=5, forget_iter_fraction=0.5, p_as=0.5, p_cr=0.2)
        dataset = GRRoutedDataset(
            RecordingChild("retain", plan.n_samples(RETAIN, gbs)),
            RecordingChild("forget", plan.n_samples(FORGET, gbs)),
            plan,
            gbs,
        )
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
                expected = "forget" if plan.corpus[iteration] == FORGET else "retain"
                assert set(labels) == {expected}, f"rank {rank}, iteration {iteration}: {set(labels)}"

    def test_routed_dataset_through_the_real_sampler_serves_one_corpus_per_iteration(self):
        """End-to-end: plan -> dataset -> real sampler. What each rank actually receives."""
        dp_size, micro_batch_size, num_microbatches = 2, 2, 2
        gbs = dp_size * micro_batch_size * num_microbatches  # 8
        plan = build_gr_plan(plan_seed=7, train_iters=6, forget_iter_fraction=0.5, p_as=0.5, p_cr=0.2)
        dataset = GRRoutedDataset(
            RecordingChild("retain", plan.n_samples(RETAIN, gbs)),
            RecordingChild("forget", plan.n_samples(FORGET, gbs)),
            plan,
            gbs,
        )
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
                expected = "forget" if plan.corpus[iteration] == FORGET else "retain"
                labels = {dataset[idx][0] for idx in indices}
                assert labels == {expected}, f"rank {rank}, iteration {iteration}: {labels}"


RETAIN_PATHS = ["0.5", "/data/retain_a_text_document", "0.5", "/data/retain_b_text_document"]
FORGET_PATHS = ["/data/forget_text_document"]


def _gr_dataset_config(plan, gbs=GBS):
    return GRDatasetConfig(
        retain_data_path=RETAIN_PATHS,
        forget_data_path=FORGET_PATHS,
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


def _is_forget_config(config) -> bool:
    """Which corpus a child config carries, read off its blend paths."""
    return any("forget" in path for path in (config.data_path or []))


class TestDatasetProviderDispatch:
    """``get_dataset_provider`` must resolve a GRDatasetConfig, and resolve it to pretrain.

    This is where an integration bug lived: the registry lookup is ``_REGISTRY[type(cfg)]``,
    an EXACT type match. Subclassing ``GPTDatasetConfig`` therefore buys nothing — an
    unregistered GRDatasetConfig raises ``KeyError`` at data-iterator setup, which is after
    the model and the optimizer have already been built.
    """

    @pytest.fixture
    def plan(self):
        return build_gr_plan(1234, 8, 0.5, 0.5, 0.2)

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
        child = _gr_dataset_config(plan).build_child_config(FORGET_PATHS)
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
        label = "forget" if _is_forget_config(self.config) else "retain"
        return RecordingChild(label, self.sizes[0]), None, None


@pytest.fixture
def fake_builder(monkeypatch):
    _FakeBuilder.calls = []
    monkeypatch.setattr(data_utils, "BlendedMegatronDatasetBuilder", _FakeBuilder)
    return _FakeBuilder


class TestProviderBuildsBothCorpora:
    """End-to-end through the real provider: one child per corpus, each sized from the plan."""

    @pytest.fixture
    def plan(self):
        return build_gr_plan(1234, 8, 0.5, 0.5, 0.2)

    def test_provider_returns_a_routed_dataset_and_no_val_or_test(self, plan, fake_builder):
        train, valid, test = pretrain_train_valid_test_datasets_provider(
            [plan.train_iters * GBS, 0, 0], _gr_dataset_config(plan)
        )
        assert isinstance(train, GRRoutedDataset)
        assert valid is None and test is None, "GR runs train with eval_iters 0 (enforced by the guards)"
        assert len(train) == plan.train_iters * GBS

    def test_each_corpus_is_built_once_from_its_own_blend(self, plan, fake_builder):
        pretrain_train_valid_test_datasets_provider([plan.train_iters * GBS, 0, 0], _gr_dataset_config(plan))

        assert len(fake_builder.calls) == 2
        by_corpus = {"forget" if _is_forget_config(c["config"]) else "retain": c for c in fake_builder.calls}
        assert set(by_corpus) == {"retain", "forget"}, "a corpus was built twice or not at all"
        assert by_corpus["retain"]["config"].data_path == RETAIN_PATHS
        assert by_corpus["forget"]["config"].data_path == FORGET_PATHS
        for name, corpus in (("retain", RETAIN), ("forget", FORGET)):
            call = by_corpus[name]
            assert call["sizes"] == [plan.n_samples(corpus, GBS), 0, 0]
            assert type(call["config"]) is GPTDatasetConfig, "a child must not stay a GRDatasetConfig"

    def test_routed_dataset_serves_the_planned_corpus_per_iteration(self, plan, fake_builder):
        train, _, _ = pretrain_train_valid_test_datasets_provider(
            [plan.train_iters * GBS, 0, 0], _gr_dataset_config(plan)
        )
        for iteration in range(plan.train_iters):
            expected = "forget" if plan.corpus[iteration] == FORGET else "retain"
            assert {train[iteration * GBS + j][0] for j in range(GBS)} == {expected}

    @pytest.mark.parametrize("delta", [-1, 1, GBS])
    def test_sizing_mismatch_raises(self, plan, fake_builder, delta):
        """train_iters or GBS changing after the plan was built would silently truncate or
        overrun the routed window; the provider refuses instead of serving a short window."""
        with pytest.raises(ValueError, match="GR dataset sizing mismatch"):
            pretrain_train_valid_test_datasets_provider(
                [plan.train_iters * GBS + delta, 0, 0], _gr_dataset_config(plan)
            )
