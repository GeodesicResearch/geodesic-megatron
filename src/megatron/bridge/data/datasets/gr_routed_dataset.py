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
"""Iteration-mapped dataset serving label-homogeneous global batches for gradient routing.

Under ``dataloader_type="single"``, ``MegatronPretrainingSampler`` hands iteration ``k``
exactly the global sample indices ``[k * GBS, (k + 1) * GBS)`` — so ``idx // GBS`` IS the
training iteration, on every data-parallel rank, with no communication. The same equation
holds under ``dataloader_type="batch"``: ``MegatronPretrainingBatchSampler`` accumulates
exactly that window before splitting it across data-parallel ranks, and the SFT training
loop consumes one such global batch per step. This dataset maps the iteration through the
routing plan to one corpus and serves the corpus's samples in a contiguous, gapless order:
iteration ``k`` gets its corpus's samples ``[prior_iters_same_corpus[k] * GBS + (idx % GBS)]``.

Randomness is not lost: each child dataset shuffles internally through its own seeded
shuffle index (``GPTDataset`` for cpt/pretrain, the SFT dataset family's sample mapping for
sft), so "contiguous" here means contiguous in the child's already-shuffled sample space.
"""

import numpy

from megatron.bridge.training.gradient_routing.plan import CORE, GRPlan


class GRRoutedDataset:
    """N+1 child datasets behind one indexable dataset, routed per-iteration by a GRPlan.

    ``children`` maps corpus label -> dataset: label 0 is the core corpus, labels 1..N the
    aux corpora, exactly as the plan's ``corpus`` array names them. Children are
    ``GPTDataset``s on the cpt/pretrain path and SFT datasets on the sft path; when the
    children collate (SFT), this dataset exposes the core child's ``collate_fn`` after
    asserting every child would collate identically.
    """

    #: The attributes of the SFT dataset family's ``collate_fn`` that shape its output
    #: (padding width and target, packed cu_seqlens emission). The delegate serves every
    #: corpus's batches through the CORE child's bound collate_fn, which is only sound if
    #: these agree on every child. ``getattr`` with a None default is safe here because the
    #: children are first asserted to be the same class, so each name is present either on
    #: all of them or on none.
    _COLLATE_ATTRS = (
        "max_seq_length",
        "pad_to_max_length",
        "pad_seq_length_to_mult",
        "return_cu_seqlen",
        "pad_cu_seqlens",
        "_pad_seq_to_mult",
    )

    def __init__(self, children: dict[int, object], plan: GRPlan, global_batch_size: int):
        if global_batch_size <= 0:
            raise ValueError(f"global_batch_size must be positive, got {global_batch_size}.")
        expected = set(range(plan.n_aux + 1))
        if set(children) != expected:
            raise ValueError(
                f"GRRoutedDataset needs one child per corpus label {sorted(expected)} "
                f"(0 = core, 1..N = aux), got labels {sorted(children)}."
            )
        self._children = dict(children)
        self._plan = plan
        self._gbs = global_batch_size
        for corpus, dataset in self._children.items():
            needed = plan.n_samples(corpus, global_batch_size)
            have = len(dataset)
            if have < needed:
                raise ValueError(
                    f"GR corpus {corpus} dataset provides {have} samples but the plan consumes {needed} "
                    f"({plan.n_corpus_iters(corpus)} iterations x GBS {global_batch_size}). "
                    "Build the child dataset with at least that many samples (epoch looping is the "
                    "GPTDataset builder's job, via train_val_test_num_samples)."
                )
        collating = {corpus for corpus, dataset in self._children.items() if hasattr(dataset, "collate_fn")}
        if collating and collating != set(self._children):
            raise ValueError(
                f"GR corpora {sorted(collating)} define collate_fn but "
                f"{sorted(set(self._children) - collating)} do not; the dataloader applies ONE collate_fn "
                "to every batch, so either every child collates (SFT datasets) or none does (GPTDatasets)."
            )
        if collating:
            self._assert_collate_delegation_is_safe()
            # The dataloader builder wires `train_ds.collate_fn` only when the attribute
            # exists (loaders.build_train_valid_test_data_loaders); without this delegate
            # it would silently fall back to torch's default collate on the children's
            # per-sample dicts. Set as an instance attribute so the cpt/pretrain path
            # (GPTDataset children, no collate) keeps hasattr(...) False.
            self.collate_fn = self._children[CORE].collate_fn

    def _assert_collate_delegation_is_safe(self) -> None:
        """Refuse construction unless every child would collate a batch identically.

        A batch always comes from ONE child (iterations are label-homogeneous), but the
        dataloader applies the core child's ``collate_fn`` to all of them — so a child that
        pads to a different width, emits different cu_seqlens, or pads with a different
        eos id would have its batches shaped by rules its own dataset never agreed to,
        silently.
        """
        core = self._children[CORE]
        for corpus, child in self._children.items():
            if type(child) is not type(core):
                raise ValueError(
                    f"GR corpus {corpus} dataset is {type(child).__name__} but the core corpus is "
                    f"{type(core).__name__}; one collate_fn serves every corpus, so all children must "
                    "be the same dataset class (e.g. one corpus packed while another is not)."
                )
            diverged = {
                name: (getattr(core, name, None), getattr(child, name, None))
                for name in self._COLLATE_ATTRS
                if getattr(child, name, None) != getattr(core, name, None)
            }
            if child.tokenizer.eos_id != core.tokenizer.eos_id:
                diverged["tokenizer.eos_id"] = (core.tokenizer.eos_id, child.tokenizer.eos_id)
            if diverged:
                raise ValueError(
                    f"GR corpus {corpus} dataset disagrees with the core corpus on collate parameters "
                    f"{diverged} (core value, corpus value); the core child's collate_fn serves every "
                    "corpus's batches, so a divergence would pad or mask this corpus's batches "
                    "differently from how its own dataset would."
                )

    def __len__(self) -> int:
        return self._plan.train_iters * self._gbs

    def __getitem__(self, idx):
        if isinstance(idx, numpy.integer):
            idx = int(idx)
        if not 0 <= idx < len(self):
            raise IndexError(f"index {idx} out of range for GRRoutedDataset of length {len(self)}.")
        iteration = idx // self._gbs
        corpus = int(self._plan.corpus[iteration])
        offset = int(self._plan.prior_iters_same_corpus[iteration]) * self._gbs + idx % self._gbs
        return self._children[corpus][offset]
