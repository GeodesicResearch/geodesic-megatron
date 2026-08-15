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
"""The sft half of gradient routing: batch-sampler attribution, collate delegation, sizing.

The cpt/pretrain suite (test_gr_routed_dataset.py) pins ``idx // GBS == iteration`` under
the REAL ``MegatronPretrainingSampler``; sft trains under ``dataloader_type="batch"``, so
the same claim is pinned here against the REAL ``MegatronPretrainingBatchSampler`` — one
yield per rank per iteration, whose union across ranks must be exactly the iteration's
window. The other two properties under test are sft-only seams: the routed dataset must
DELEGATE collation to its children (torch's default collate on SFT sample dicts is the
silent fallback the delegate exists to prevent, and it is only sound if every child would
collate identically), and each corpus's realised length must equal the plan's consumption
exactly (the batch sampler wraps modulo its total instead of asserting, so a mis-sized
child would silently relabel every post-wrap iteration).

The children are REAL ``GPTSFTDataset``/``GPTSFTPackedDataset`` objects, built from a tmp
JSONL (or ``.npy`` pack) the way ``tests/unit_tests/data/datasets/test_sft.py`` does, with
the tokenizer as the only mock — that is the accepted boundary, since a real tokenizer
needs a downloaded model file and nothing here asserts a specific tokenization. Real
children matter most for collate delegation: ``GRRoutedDataset._COLLATE_ATTRS`` names
attributes by string, and against stubs a name that drifted off the real dataset class
would compare ``None`` to ``None`` and pass while checking nothing. It also makes the
sizing claim real — ``max_num_samples`` is what the provider relies on for exact lengths,
and it is the SFT sample mapping, not the provider, that has to deliver them.

Two stand-ins remain, each at a boundary that is not what the test is about:
``FinetuningDatasetBuilder`` (the rank-0 dataset-root scan and offline packing step — the
children it returns are still real datasets), and a plain indexable child for the
cpt/pretrain-shaped cases, whose only relevant property is the ABSENCE of ``collate_fn``.
"""

import json
from unittest.mock import MagicMock

import numpy as np
import pytest

from megatron.bridge.data import utils as data_utils
from megatron.bridge.data.datasets.gr_routed_dataset import GRRoutedDataset
from megatron.bridge.data.datasets.sft import GPTSFTDataset, GPTSFTPackedDataset
from megatron.bridge.data.samplers import MegatronPretrainingBatchSampler, build_pretraining_data_loader
from megatron.bridge.data.utils import finetuning_train_valid_test_datasets_provider, get_dataset_provider
from megatron.bridge.training.config import FinetuningDatasetConfig
from megatron.bridge.training.gradient_routing.config import GRFinetuningDatasetConfig
from megatron.bridge.training.gradient_routing.plan import CORE, FIRST_AUX, build_gr_plan


PROMPT_TEMPLATE = "{input}\n\n### Response:\n{output}"

#: Answer length per corpus, in words. Distinct on purpose: it makes each corpus's
#: supervised-token density different, which is exactly the iteration-share/token-share
#: divergence the token telemetry exists to measure.
_ANSWER_WORDS = {"core": 6, "aux0": 2, "aux1": 4}


def _label(corpus: int) -> str:
    """The label of a corpus: ``core`` or ``aux<module index>``."""
    return "core" if corpus == CORE else f"aux{corpus - FIRST_AUX}"


def _mock_tokenizer(eos_id: int = 2):
    """The tokenizer boundary — the only mock in the dataset construction path.

    Same shape as ``tests/unit_tests/data/datasets/test_sft.py::create_mock_tokenizer``: a
    real tokenizer would need a downloaded model file, and nothing here asserts anything
    about a specific tokenization. ``eos_id`` is a parameter because it is one of the
    collate parameters ``GRRoutedDataset`` compares across children (it is the pad id every
    ``_collate_item`` call uses).
    """
    tokenizer = MagicMock()
    tokenizer.eos_id = eos_id
    tokenizer.bos_id = 1
    tokenizer.pad_id = 0
    tokenizer.vocab_size = 50000
    tokenizer.space_sensitive = True
    tokenizer.text_to_ids = lambda text: list(range(1, len(text.split()) + 2))
    return tokenizer


def _corpus_jsonl(tmp_path, label: str, rows: int) -> str:
    """Write one corpus's ``training.jsonl`` under its own root and return the path.

    The corpus label rides in a field the prompt template does not consume, so
    ``GPTSFTDataset`` carries it through to ``sample["metadata"]["corpus"]`` (metadata is
    every example key outside the template) — that is how a routed sample is attributed to
    the corpus that served it, without reading tokens.
    """
    root = tmp_path / label
    root.mkdir(parents=True, exist_ok=True)
    path = root / "training.jsonl"
    answer = " ".join(["word"] * _ANSWER_WORDS[label])
    with path.open("w") as handle:
        for i in range(rows):
            handle.write(json.dumps({"input": f"{label} question {i}", "output": answer, "corpus": label}) + "\n")
    return str(path)


def _sft_dataset(tmp_path, label: str, n_samples: int, *, eos_id: int = 2, rows: int = 5, **kwargs) -> GPTSFTDataset:
    """A real ``GPTSFTDataset`` for one corpus, sized to exactly ``n_samples``."""
    return GPTSFTDataset(
        file_path=_corpus_jsonl(tmp_path, label, rows),
        tokenizer=_mock_tokenizer(eos_id),
        label_key="output",
        max_num_samples=n_samples,
        prompt_template=PROMPT_TEMPLATE,
        truncation_field="output",
        memmap_workers=1,
        **kwargs,
    )


def _packed_npy(tmp_path, label: str, rows: int) -> str:
    """Write one corpus's packed ``.npy`` and return the path (the packed-dataset input)."""
    root = tmp_path / label
    root.mkdir(parents=True, exist_ok=True)
    path = root / "packed.npy"
    np.save(
        path,
        np.array(
            [
                {
                    "input_ids": np.array([1, 2, 3, 4, 5], dtype=np.int64),
                    "seq_start_id": [0],
                    "loss_mask": np.array([0, 0, 1, 1, 1], dtype=np.int64),
                }
                for _ in range(rows)
            ],
            dtype=object,
        ),
    )
    return str(path)


def _pack_metadata(tmp_path) -> str:
    """The packing metadata file ``pad_cu_seqlens=True`` requires; written once per test."""
    path = tmp_path / "pack_metadata.json"
    path.write_text(json.dumps({"dataset_max_seqlen": 5, "max_samples_per_bin": 1}))
    return str(path)


def _packed_dataset(
    tmp_path, label: str, n_samples: int, *, eos_id: int = 2, rows: int = 5, **kwargs
) -> GPTSFTPackedDataset:
    """A real ``GPTSFTPackedDataset`` for one corpus, sized to exactly ``n_samples``."""
    return GPTSFTPackedDataset(
        file_path=_packed_npy(tmp_path, label, rows),
        tokenizer=_mock_tokenizer(eos_id),
        label_key="output",
        max_num_samples=n_samples,
        prompt_template=PROMPT_TEMPLATE,
        truncation_field="output",
        memmap_workers=1,
        **kwargs,
    )


def _sft_children(tmp_path, plan, gbs, overrides_by_corpus=None, *, builder=_sft_dataset, **common):
    """One real, exactly-plan-sized dataset per corpus, with optional per-corpus overrides."""
    overrides_by_corpus = overrides_by_corpus or {}
    return {
        corpus: builder(
            tmp_path,
            _label(corpus),
            plan.n_samples(corpus, gbs),
            **{**common, **overrides_by_corpus.get(corpus, {})},
        )
        for corpus in range(plan.n_aux + 1)
    }


def _corpus_of(sample) -> str:
    """The corpus label of one routed sample (see ``_corpus_jsonl``)."""
    return sample["metadata"]["corpus"]


def _supervised_tokens(child) -> float:
    """Total supervised tokens in a corpus, measured the way the telemetry measures them."""
    return float(sum(np.sum(child._build_loss_mask(child[i])) for i in range(len(child))))


class PlainChild:
    """A cpt-style child: indexable, no collate_fn (GPTDatasets do not collate).

    Kept as a stand-in rather than a real ``GPTDataset`` because the only property under
    test is the ABSENCE of ``collate_fn`` — building a real GPTDataset would mean tokenized
    ``.bin``/``.idx`` files on disk to assert a negative ``hasattr``.
    """

    def __init__(self, label: str, n: int):
        self.label = label
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx):
        return (self.label, int(idx))


class TestExactSizing:
    """``max_num_samples`` must deliver EXACT lengths, in both directions.

    The provider caps each corpus at exactly the plan's consumption and then refuses a child
    whose realised length differs. That refusal only protects anything if the cap itself is
    exact — including when the plan consumes MORE samples than the corpus has rows (the
    sample mapping epoch-wraps) and when it consumes fewer (it truncates). The claim belongs
    to the SFT sample mapping, not to the provider, so it is pinned on the real datasets.
    """

    @pytest.mark.parametrize("rows", [3, 10])
    @pytest.mark.parametrize("n_samples", [1, 3, 7, 20])
    def test_length_is_exactly_max_num_samples(self, tmp_path, rows, n_samples):
        assert len(_sft_dataset(tmp_path, "core", n_samples, rows=rows)) == n_samples

    @pytest.mark.parametrize("rows", [4, 12])
    @pytest.mark.parametrize("n_samples", [1, 4, 9])
    def test_packed_length_is_exactly_max_num_samples(self, tmp_path, rows, n_samples):
        assert len(_packed_dataset(tmp_path, "core", n_samples, rows=rows)) == n_samples

    def test_an_epoch_wrapped_corpus_still_serves_every_index(self, tmp_path):
        """Oversampling must produce a usable sample at every index, not just a length."""
        dataset = _sft_dataset(tmp_path, "core", 11, rows=3)
        assert all(_corpus_of(dataset[i]) == "core" for i in range(len(dataset)))


class TestBatchSamplerAttribution:
    """``idx // GBS == iteration`` under the REAL batch sampler, for every rank.

    ``MegatronPretrainingBatchSampler`` yields ONE list per iteration per rank — the
    rank's full slice of the global batch (the collate-then-split flow of the sft loop).
    The claim being pinned is that yield k across all DP ranks is exactly
    ``[k*GBS, (k+1)*GBS)``.
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
            (2, 1, 32),  # accumulation-heavy: the exemplar sft shape (GBS 64 at DP 2, mbs 1)
        ],
    )
    def test_each_yield_is_exactly_its_iterations_window(self, dp_size, micro_batch_size, num_microbatches):
        """No dataset here on purpose: the sampler yields indices, and what is under test is
        which indices, so a dataset would only add build cost to an integer assertion."""
        gbs = dp_size * micro_batch_size * num_microbatches
        n_iters = 5
        seen_per_iteration = [set() for _ in range(n_iters)]

        for rank in range(dp_size):
            sampler = MegatronPretrainingBatchSampler(
                total_samples=n_iters * gbs,
                consumed_samples=0,
                micro_batch_size=micro_batch_size,
                global_batch_size=gbs,
                data_parallel_rank=rank,
                data_parallel_size=dp_size,
                drop_last=True,
            )
            for iteration, indices in enumerate(sampler):
                assert len(indices) == gbs // dp_size, "one yield must be the rank's full global-batch slice"
                for idx in indices:
                    assert idx // gbs == iteration, (
                        f"rank {rank} saw index {idx} (iteration {idx // gbs}) while serving iteration {iteration}"
                    )
                seen_per_iteration[iteration].update(indices)

        for iteration, seen in enumerate(seen_per_iteration):
            assert seen == set(range(iteration * gbs, (iteration + 1) * gbs)), (
                f"iteration {iteration} did not cover its global batch exactly"
            )

    @pytest.mark.parametrize("dp_size, micro_batch_size", [(2, 2), (4, 1), (1, 1), (3, 2)])
    @pytest.mark.parametrize("resume_at", [1, 2, 5])
    def test_resume_midway_keeps_the_iteration_mapping(self, dp_size, micro_batch_size, resume_at):
        """A restart sets consumed_samples to a multiple of GBS. Unlike the single sampler,
        the batch sampler has NO ``consumed_samples < total_samples`` assert — it wraps
        modulo the total — so within the plan's coverage the mapping must hold exactly,
        and it is the exact-length build guard (below) that keeps a run inside coverage.
        Index-only, for the same reason as the test above."""
        gbs = dp_size * micro_batch_size * 2
        n_iters = 6
        seen_per_iteration = {iteration: set() for iteration in range(resume_at, n_iters)}

        for rank in range(dp_size):
            sampler = MegatronPretrainingBatchSampler(
                total_samples=n_iters * gbs,
                consumed_samples=resume_at * gbs,
                micro_batch_size=micro_batch_size,
                global_batch_size=gbs,
                data_parallel_rank=rank,
                data_parallel_size=dp_size,
                drop_last=True,
            )
            for offset, indices in enumerate(sampler):
                iteration = resume_at + offset
                for idx in indices:
                    assert idx // gbs == iteration
                seen_per_iteration[iteration].update(indices)

        for iteration, seen in seen_per_iteration.items():
            assert seen == set(range(iteration * gbs, (iteration + 1) * gbs)), (
                f"iteration {iteration} did not cover its global batch exactly after a resume"
            )

    @pytest.mark.parametrize("fractions", [[0.5], [0.25, 0.25]])
    def test_routed_dataset_through_the_real_batch_sampler_serves_one_corpus_per_iteration(self, tmp_path, fractions):
        """End-to-end: plan -> routed dataset -> real batch sampler. Every rank's slice of
        iteration k must be label-homogeneous and match the plan's corpus for k."""
        dp_size, micro_batch_size = 2, 2
        gbs = dp_size * micro_batch_size * 2  # 8
        plan = build_gr_plan(plan_seed=7, train_iters=6, aux_iter_fractions=fractions, p_as=0.5, p_cr=0.2)
        dataset = GRRoutedDataset(_sft_children(tmp_path, plan, gbs), plan, gbs)
        for rank in range(dp_size):
            sampler = MegatronPretrainingBatchSampler(
                total_samples=len(dataset),
                consumed_samples=0,
                micro_batch_size=micro_batch_size,
                global_batch_size=gbs,
                data_parallel_rank=rank,
                data_parallel_size=dp_size,
                drop_last=True,
            )
            for iteration, indices in enumerate(sampler):
                expected = _label(int(plan.corpus[iteration]))
                labels = {_corpus_of(dataset[idx]) for idx in indices}
                assert labels == {expected}, f"rank {rank}, iteration {iteration}: {labels}"

    @pytest.mark.parametrize("dp_size, micro_batch_size", [(1, 2), (2, 1)])
    def test_the_batch_dataloader_type_builds_the_iteration_aligned_sampler(self, tmp_path, dp_size, micro_batch_size):
        """``dataloader_type: "batch"`` is the sft guard's precondition; this is what it buys.

        ``build_pretraining_data_loader`` is what maps the string onto the sampler, and the
        collate_fn wiring mirrors ``loaders.build_train_valid_test_data_loaders`` (which
        passes ``train_ds.collate_fn`` when the attribute exists) — so the yields here are
        the routed dataset's OWN collation of one full per-rank slice per iteration.
        """
        gbs = dp_size * micro_batch_size * 2
        plan = build_gr_plan(plan_seed=3, train_iters=5, aux_iter_fractions=[0.5], p_as=0.5, p_cr=0.2)
        dataset = GRRoutedDataset(_sft_children(tmp_path, plan, gbs), plan, gbs)
        for rank in range(dp_size):
            loader = build_pretraining_data_loader(
                dataset=dataset,
                consumed_samples=0,
                dataloader_type="batch",
                micro_batch_size=micro_batch_size,
                num_workers=0,
                data_sharding=False,
                collate_fn=dataset.collate_fn if hasattr(dataset, "collate_fn") else None,
                pin_memory=False,
                data_parallel_rank=rank,
                data_parallel_size=dp_size,
                global_batch_size=gbs,
            )
            assert isinstance(loader.batch_sampler, MegatronPretrainingBatchSampler)

            batches = list(loader)
            assert len(batches) == plan.train_iters, "one yield per iteration, the whole per-rank slice at once"
            for iteration, collated in enumerate(batches):
                expected = _label(int(plan.corpus[iteration]))
                # "tokens" only exists because the SFT collate_fn ran: torch's default
                # collate over the children's per-sample dicts would keep their own keys.
                assert collated["tokens"].shape[0] == gbs // dp_size, "collation must go through the delegate"
                labels = {metadata["corpus"] for metadata in collated["metadata"]}
                assert labels == {expected}, f"rank {rank}, iteration {iteration}: {labels}"


class TestCollateDelegation:
    """The routed dataset must collate through its children — identically-configured ones.

    ``loaders.build_train_valid_test_data_loaders`` wires ``train_ds.collate_fn`` only when
    the attribute exists; without the delegate, sft batches would silently fall back to
    torch's default collate over per-sample dicts. And because the ONE wired collate_fn
    serves every corpus's batches, children that would collate differently must be refused
    at construction, not discovered as shape drift mid-run.
    """

    @pytest.fixture
    def plan(self):
        return build_gr_plan(1234, 8, [0.5], 0.5, 0.2)

    GBS = 4

    #: Divergences reachable on a plain ``GPTSFTDataset``: (attribute compared by
    #: ``_COLLATE_ATTRS``, constructor kwargs that move it on one child).
    BASE_DIVERGENCES = [
        ("max_seq_length", {"max_seq_length": 256}),
        ("pad_to_max_length", {"pad_to_max_length": True}),
        ("pad_seq_length_to_mult", {"pad_seq_length_to_mult": 1}),
    ]

    #: The remaining three live only on ``GPTSFTPackedDataset``, so they are diverged on
    #: packed children. ``pad_cu_seqlens=True`` additionally requires ``pad_to_max_length``,
    #: which is therefore given to EVERY child so the compared divergence stays the one
    #: named. Attribute name and constructor kwarg differ for ``_pad_seq_to_mult``.
    PACKED_DIVERGENCES = [
        ("return_cu_seqlen", {"return_cu_seqlen": False}, {}),
        ("_pad_seq_to_mult", {"pad_seq_to_mult": 4}, {}),
        ("pad_cu_seqlens", {"pad_cu_seqlens": True}, {"pad_to_max_length": True}),
    ]

    def test_collating_children_expose_the_core_childs_collate_fn(self, tmp_path, plan):
        children = _sft_children(tmp_path, plan, self.GBS)
        routed = GRRoutedDataset(children, plan, self.GBS)
        assert routed.collate_fn == children[CORE].collate_fn
        collated = routed.collate_fn([routed[0], routed[1]])
        assert collated["tokens"].shape[0] == 2, "the delegate must run the SFT collate, not torch's default"

    def test_every_compared_attribute_is_a_real_dataset_attribute(self, tmp_path, plan):
        """``_COLLATE_ATTRS`` names attributes by string, and the comparison uses ``getattr``
        with a None default — so a name that drifted off the dataset class would compare
        None to None on every child and pass while checking nothing. Every name must exist
        on the packed class (which defines all of them); the three that exist ONLY there are
        pinned as such, because that asymmetry is what the None default is for and it is
        only safe under the same-class check that runs first."""
        packed = _packed_dataset(tmp_path, "core", 4)
        plain = _sft_dataset(tmp_path, "core", 4)
        for name in GRRoutedDataset._COLLATE_ATTRS:
            assert hasattr(packed, name), f"_COLLATE_ATTRS names {name}, which no SFT dataset defines"
        base_names = {name for name, _kwargs in self.BASE_DIVERGENCES}
        packed_only = {name for name, _kwargs, _common in self.PACKED_DIVERGENCES}
        assert base_names | packed_only == set(GRRoutedDataset._COLLATE_ATTRS), "a compared attribute is untested"
        assert all(hasattr(plain, name) for name in base_names)
        assert not any(hasattr(plain, name) for name in packed_only)

    def test_plain_children_leave_no_collate_fn(self, plan):
        """The cpt/pretrain path must keep hasattr(train_ds, "collate_fn") False — a
        delegate over GPTDatasets would hand the loader an AttributeError factory."""
        children = {c: PlainChild(_label(c), plan.n_samples(c, self.GBS)) for c in range(plan.n_aux + 1)}
        routed = GRRoutedDataset(children, plan, self.GBS)
        assert not hasattr(routed, "collate_fn")

    def test_mixed_collating_and_plain_children_are_refused(self, tmp_path, plan):
        children = _sft_children(tmp_path, plan, self.GBS)
        children[FIRST_AUX] = PlainChild("aux0", plan.n_samples(FIRST_AUX, self.GBS))
        with pytest.raises(ValueError, match="define collate_fn but"):
            GRRoutedDataset(children, plan, self.GBS)

    def test_children_of_different_classes_are_refused(self, tmp_path, plan):
        """One corpus packed while another is not — a different collate_fn implementation
        entirely, and the case the packing-posture config guard exists upstream of."""
        children = _sft_children(tmp_path, plan, self.GBS)
        children[FIRST_AUX] = _packed_dataset(tmp_path, "aux0_packed", plan.n_samples(FIRST_AUX, self.GBS))
        with pytest.raises(ValueError, match="must be the same dataset class"):
            GRRoutedDataset(children, plan, self.GBS)

    @pytest.mark.parametrize("attr, diverging", BASE_DIVERGENCES, ids=[name for name, _ in BASE_DIVERGENCES])
    def test_a_child_diverging_on_any_collate_parameter_is_refused(self, tmp_path, plan, attr, diverging):
        children = _sft_children(tmp_path, plan, self.GBS, {FIRST_AUX: diverging})
        assert getattr(children[FIRST_AUX], attr) != getattr(children[CORE], attr), "the fixture did not diverge"
        with pytest.raises(ValueError, match="disagrees with the core corpus on collate parameters"):
            GRRoutedDataset(children, plan, self.GBS)

    @pytest.mark.parametrize(
        "attr, diverging, common", PACKED_DIVERGENCES, ids=[name for name, _, _ in PACKED_DIVERGENCES]
    )
    def test_a_packed_child_diverging_on_any_collate_parameter_is_refused(
        self, tmp_path, plan, attr, diverging, common
    ):
        children = _sft_children(
            tmp_path,
            plan,
            self.GBS,
            {FIRST_AUX: {**common, **diverging}},
            builder=_packed_dataset,
            pack_metadata_file_path=_pack_metadata(tmp_path),
            **common,
        )
        assert getattr(children[FIRST_AUX], attr) != getattr(children[CORE], attr), "the fixture did not diverge"
        with pytest.raises(ValueError, match="disagrees with the core corpus on collate parameters"):
            GRRoutedDataset(children, plan, self.GBS)

    def test_a_child_with_a_different_eos_id_is_refused(self, tmp_path, plan):
        """eos_id is the pad id every ``_collate_item`` call uses; a divergence would pad one
        corpus's batches with another corpus's token."""
        children = _sft_children(tmp_path, plan, self.GBS, {FIRST_AUX: {"eos_id": 7}})
        with pytest.raises(ValueError, match="tokenizer.eos_id"):
            GRRoutedDataset(children, plan, self.GBS)


class TestSupervisedTokenTelemetry:
    """``gr.aux_iter_fractions`` allocates ITERATIONS; the telemetry reports realised TOKENS.

    The two diverge whenever corpora have different answer densities, which is the normal
    case — so the numbers must be measured off the real children rather than derived from
    the plan. The corpora here deliberately carry different answer lengths, so a function
    that reported iteration shares under a token-share label would fail.
    """

    GBS = 4

    @pytest.fixture
    def plan(self):
        return build_gr_plan(1234, 8, [0.25, 0.25], 0.5, 0.2)

    def test_each_corpus_reports_its_measured_supervised_tokens_and_shares(self, tmp_path, plan, capsys):
        children = _sft_children(tmp_path, plan, self.GBS)
        expected = {corpus: _supervised_tokens(child) for corpus, child in children.items()}
        grand_total = sum(expected.values())
        assert len({total / len(children[corpus]) for corpus, total in expected.items()}) == len(children), (
            "the corpora must differ in supervised-token density, or a function reporting one "
            "number for every corpus would pass"
        )
        assert any(
            abs(expected[corpus] / grand_total - plan.n_corpus_iters(corpus) / plan.train_iters) > 1e-3
            for corpus in children
        ), "at least one token share must differ from its iteration share, or the two are indistinguishable here"

        data_utils._log_gr_supervised_token_counts(children, plan)

        lines = {
            line.split()[3].rstrip(":"): line
            for line in capsys.readouterr().out.splitlines()
            if line.startswith("> gr corpus ")
        }
        assert set(lines) == {_label(corpus) for corpus in children}
        for corpus, child in children.items():
            line = lines[_label(corpus)]
            assert f"supervised_tokens={int(expected[corpus])}" in line
            assert f"token_share={expected[corpus] / grand_total:.4f}" in line
            assert f"iter_share={plan.n_corpus_iters(corpus) / plan.train_iters:.4f}" in line
        assert sum(float(line.split("token_share=")[1].split()[0]) for line in lines.values()) == pytest.approx(1.0)

    def test_a_corpus_set_with_no_supervised_tokens_raises(self, tmp_path, plan, monkeypatch):
        """The zero-total path. Building a corpus whose every token is unsupervised would
        take a degenerate tokenizer or template — a real answer always contributes at least
        one supervised token — so the loss mask is patched on the class instead. What is
        under test is the accounting rule: a zero grand total must raise rather than be
        divided by a substituted 1.0, which would report a token share of 0.0 per corpus
        for a dataset that trains on nothing."""
        children = _sft_children(tmp_path, plan, self.GBS)
        monkeypatch.setattr(GPTSFTDataset, "_build_loss_mask", lambda self, example: np.zeros(4))
        with pytest.raises(ValueError, match="zero supervised tokens"):
            data_utils._log_gr_supervised_token_counts(children, plan)

    def test_the_extra_pass_is_skipped_off_rank_zero(self, tmp_path, plan, monkeypatch, capsys):
        """The measurement is a full extra pass over every corpus, so it runs on rank 0 only
        (every other rank would recompute the same numbers and print nothing). Note the
        zero-total refusal above sits behind the same early return, so it is a rank-0 raise."""
        children = _sft_children(tmp_path, plan, self.GBS)
        monkeypatch.setattr(data_utils, "get_rank_safe", lambda: 1)
        data_utils._log_gr_supervised_token_counts(children, plan)
        assert "gr corpus" not in capsys.readouterr().out


ROOTS = {"retain": "/data/gr_sft/core", "aux": ["/data/gr_sft/aux0", "/data/gr_sft/aux1"]}
GBS = 4


def _gr_ft_config(plan, gbs=GBS, **overrides):
    kwargs = dict(
        retain_dataset_root=ROOTS["retain"],
        aux_dataset_roots=ROOTS["aux"][: plan.n_aux],
        gr_plan=plan,
        gr_global_batch_size=gbs,
        seq_length=1024,
        seed=1234,
        dataloader_type="batch",
        do_validation=False,
        do_test=False,
    )
    kwargs.update(overrides)
    return GRFinetuningDatasetConfig(**kwargs)


class TestGRFinetuningDatasetConfig:
    """N+1 dataset roots behind one cfg.dataset, and a child that is a plain FinetuningDatasetConfig."""

    @pytest.fixture(params=[[0.5], [0.25, 0.25]], ids=["one_aux", "two_aux"])
    def plan(self, request):
        return build_gr_plan(1234, 8, request.param, 0.5, 0.2)

    @pytest.mark.parametrize("field", ["dataset_root", "packed_sequence_specs", "max_train_samples"])
    def test_setting_a_per_corpus_field_on_the_parent_raises(self, plan, field):
        """The parent must never carry a root/spec/cap of its own — it would silently win
        over the per-corpus values in the builder splat."""
        value = 100 if field == "max_train_samples" else "/data/x"
        with pytest.raises(ValueError, match=f"do not set {field}"):
            _gr_ft_config(plan, **{field: value})

    def test_finalize_accepts_the_valid_config(self, plan):
        _gr_ft_config(plan).finalize()

    def test_a_corpus_count_that_disagrees_with_the_plan_raises(self, plan):
        wrong = ROOTS["aux"][:1] if plan.n_aux == 2 else ROOTS["aux"]
        with pytest.raises(ValueError, match="the config and the plan disagree about the module count"):
            _gr_ft_config(plan, aux_dataset_roots=wrong).finalize()

    @pytest.mark.parametrize("field", ["do_validation", "do_test"])
    def test_a_validation_or_test_split_is_refused(self, plan, field):
        with pytest.raises(ValueError, match="do_validation=False and do_test=False"):
            _gr_ft_config(plan, **{field: True}).finalize()

    def test_a_packed_spec_count_that_disagrees_with_the_roots_raises(self, plan):
        config = _gr_ft_config(plan, aux_packed_sequence_specs=[None] * (plan.n_aux + 1))
        with pytest.raises(ValueError, match="one PackedSequenceSpecs .* per corpus"):
            config.finalize()

    def test_corpora_disagreeing_on_packing_posture_are_refused(self, plan):
        """One packed corpus among unpacked ones would build children of different dataset
        classes — refused here at config time, before any corpus is built."""
        from megatron.bridge.data.datasets.packed_sequence import PackedSequenceSpecs

        specs = [PackedSequenceSpecs(packed_sequence_size=1024)] + [None] * (plan.n_aux - 1)
        with pytest.raises(ValueError, match="disagree about packing posture"):
            _gr_ft_config(plan, aux_packed_sequence_specs=specs).finalize()

    def test_an_identical_packing_posture_across_corpora_passes(self, plan):
        from megatron.bridge.data.datasets.packed_sequence import PackedSequenceSpecs

        config = _gr_ft_config(
            plan,
            retain_packed_sequence_specs=PackedSequenceSpecs(packed_sequence_size=1024),
            aux_packed_sequence_specs=[PackedSequenceSpecs(packed_sequence_size=1024)] * plan.n_aux,
        )
        config.finalize()

    def test_child_is_a_plain_finetuning_dataset_config(self, plan):
        """Exact class, not a subclass: the provider dispatches on isinstance(_,
        GRFinetuningDatasetConfig), so a child that stayed this class would recurse into
        the GR branch — and the builder splat would see GR fields it cannot accept."""
        child = _gr_ft_config(plan).build_child_config(ROOTS["aux"][0], None, max_train_samples=12)
        assert type(child) is FinetuningDatasetConfig
        assert not isinstance(child, GRFinetuningDatasetConfig)

    def test_child_drops_every_parent_only_field(self, plan):
        child = _gr_ft_config(plan).build_child_config(ROOTS["aux"][0], None, max_train_samples=12)
        for field in (
            "retain_dataset_root",
            "aux_dataset_roots",
            "retain_packed_sequence_specs",
            "aux_packed_sequence_specs",
            "gr_plan",
            "gr_global_batch_size",
        ):
            assert not hasattr(child, field), f"child still carries {field}"

    def test_child_carries_its_own_root_specs_and_exact_cap(self, plan):
        from megatron.bridge.data.datasets.packed_sequence import PackedSequenceSpecs

        specs = PackedSequenceSpecs(packed_sequence_size=1024)
        child = _gr_ft_config(plan).build_child_config(ROOTS["retain"], specs, max_train_samples=24)
        assert child.dataset_root == ROOTS["retain"]
        assert child.packed_sequence_specs is specs
        assert child.max_train_samples == 24
        assert child.do_validation is False and child.do_test is False

    def test_child_inherits_the_shared_dataset_fields(self, plan):
        parent = _gr_ft_config(plan)
        child = parent.build_child_config(ROOTS["aux"][0], None, max_train_samples=12)
        assert child.seq_length == parent.seq_length
        assert child.seed == parent.seed
        assert child.dataloader_type == parent.dataloader_type

    def test_children_do_not_mutate_the_parent(self, plan):
        parent = _gr_ft_config(plan)
        parent.build_child_config(ROOTS["aux"][0], None, max_train_samples=12)
        assert parent.retain_dataset_root == ROOTS["retain"]
        assert parent.aux_dataset_roots == ROOTS["aux"][: plan.n_aux]

    def test_gr_finetuning_config_resolves_to_the_finetuning_provider(self, plan):
        assert get_dataset_provider(_gr_ft_config(plan)) is finetuning_train_valid_test_datasets_provider

    def test_registry_lookup_is_by_exact_type_not_isinstance(self, plan):
        """Pins WHY registration is required: a subclass does not inherit its parent's entry."""
        config = _gr_ft_config(plan)
        assert isinstance(config, FinetuningDatasetConfig)
        assert type(config) is not FinetuningDatasetConfig
        assert GRFinetuningDatasetConfig in data_utils._REGISTRY
        assert data_utils._REGISTRY[GRFinetuningDatasetConfig] is finetuning_train_valid_test_datasets_provider

    def test_child_config_resolves_to_the_same_provider(self, plan):
        child = _gr_ft_config(plan).build_child_config(ROOTS["aux"][0], None, max_train_samples=12)
        assert get_dataset_provider(child) is finetuning_train_valid_test_datasets_provider

    def test_the_registry_import_is_module_level(self):
        assert getattr(data_utils, "GRFinetuningDatasetConfig", None) is GRFinetuningDatasetConfig


class _FakeFinetuningBuilder:
    """Stands in for FinetuningDatasetBuilder, recording what each corpus was asked for.

    The builder is the rank-0 file boundary — it scans a ``dataset_root`` for
    ``training.jsonl`` and runs offline packing — and none of that is what the provider
    tests are about. The CHILDREN it hands back are real ``GPTSFTDataset`` objects, so the
    provider's length guard, the routed dataset's collate checks and the token telemetry all
    run against real dataset behaviour.

    ``length_delta`` mimics a builder whose dataset ignores ``max_train_samples`` — the
    exact failure mode the provider's length guard exists to catch.
    """

    calls: list = []
    length_delta: int = 0
    make_child = None

    def __init__(self, tokenizer, **kwargs):
        self.kwargs = kwargs
        _FakeFinetuningBuilder.calls.append(kwargs)

    def build(self):
        label = self.kwargs["dataset_root"].rsplit("/", 1)[-1]
        n = self.kwargs["max_train_samples"] + _FakeFinetuningBuilder.length_delta
        return [_FakeFinetuningBuilder.make_child(label, n), None, None]


@pytest.fixture
def fake_builder(monkeypatch, tmp_path):
    _FakeFinetuningBuilder.calls = []
    _FakeFinetuningBuilder.length_delta = 0
    _FakeFinetuningBuilder.make_child = lambda label, n: _sft_dataset(tmp_path, label, n)
    monkeypatch.setattr(data_utils, "FinetuningDatasetBuilder", _FakeFinetuningBuilder)
    return _FakeFinetuningBuilder


class TestFinetuningProviderBuildsEveryCorpus:
    """End-to-end through the real provider: one child per corpus, each exactly plan-sized."""

    @pytest.fixture(params=[[0.5], [0.25, 0.25]], ids=["one_aux", "two_aux"])
    def plan(self, request):
        return build_gr_plan(1234, 8, request.param, 0.5, 0.2)

    def test_provider_returns_a_routed_dataset_and_no_val_or_test(self, plan, fake_builder):
        train, valid, test = finetuning_train_valid_test_datasets_provider(
            [plan.train_iters * GBS, 0, 0], _gr_ft_config(plan), tokenizer=None
        )
        assert isinstance(train, GRRoutedDataset)
        assert valid is None and test is None, "GR runs train with eval_iters 0 (enforced by the guards)"
        assert len(train) == plan.train_iters * GBS

    def test_each_corpus_is_built_once_from_its_own_root_with_the_plan_cap(self, plan, fake_builder):
        from megatron.bridge.data.datasets.packed_sequence import PackedSequenceSpecs

        specs = [PackedSequenceSpecs(packed_sequence_size=1024) for _ in range(plan.n_aux + 1)]
        config = _gr_ft_config(plan, retain_packed_sequence_specs=specs[0], aux_packed_sequence_specs=specs[1:])
        finetuning_train_valid_test_datasets_provider([plan.train_iters * GBS, 0, 0], config, tokenizer=None)

        assert len(fake_builder.calls) == plan.n_aux + 1
        by_root = {call["dataset_root"]: call for call in fake_builder.calls}
        expected_roots = [ROOTS["retain"], *ROOTS["aux"][: plan.n_aux]]
        assert list(by_root) == expected_roots, "a corpus was built twice, not at all, or out of order"
        for corpus, root in enumerate(expected_roots):
            call = by_root[root]
            assert call["max_train_samples"] == plan.n_samples(corpus, GBS)
            assert call["packed_sequence_specs"] is specs[corpus], "each corpus must get its OWN packing spec"
            assert call["do_validation"] is False and call["do_test"] is False

    def test_routed_dataset_serves_the_planned_corpus_per_iteration(self, plan, fake_builder):
        train, _, _ = finetuning_train_valid_test_datasets_provider(
            [plan.train_iters * GBS, 0, 0], _gr_ft_config(plan), tokenizer=None
        )
        for iteration in range(plan.train_iters):
            expected = _label(int(plan.corpus[iteration]))
            assert {_corpus_of(train[iteration * GBS + j]) for j in range(GBS)} == {expected}

    @pytest.mark.parametrize("delta", [-1, 1, GBS])
    def test_sizing_mismatch_raises(self, plan, fake_builder, delta):
        with pytest.raises(ValueError, match="GR dataset sizing mismatch"):
            finetuning_train_valid_test_datasets_provider(
                [plan.train_iters * GBS + delta, 0, 0], _gr_ft_config(plan), tokenizer=None
            )

    @pytest.mark.parametrize("delta", [-1, 1, GBS])
    def test_a_child_whose_length_ignores_the_cap_is_refused(self, plan, fake_builder, delta):
        """The epoch-wrap guard. ``max_train_samples`` makes the SFT sample mapping serve
        exactly the plan's consumption; a dataset that ignores it (over OR under) would
        leave the routed length disagreeing with the plan, and the batch sampler wraps
        silently modulo its total — every post-wrap routing label wrong, nothing logged."""
        fake_builder.length_delta = delta
        with pytest.raises(ValueError, match="the plan consumes exactly"):
            finetuning_train_valid_test_datasets_provider(
                [plan.train_iters * GBS, 0, 0], _gr_ft_config(plan), tokenizer=None
            )

    def test_a_corpus_with_no_training_data_is_refused(self, plan, fake_builder, monkeypatch):
        monkeypatch.setattr(fake_builder, "build", lambda self: [None, None, None])
        with pytest.raises(ValueError, match="has no training data under"):
            finetuning_train_valid_test_datasets_provider(
                [plan.train_iters * GBS, 0, 0], _gr_ft_config(plan), tokenizer=None
            )
