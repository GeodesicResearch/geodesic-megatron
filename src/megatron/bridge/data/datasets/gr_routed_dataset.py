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
training iteration, on every data-parallel rank, with no communication. This dataset maps
that iteration through the routing plan to one corpus and serves the corpus's samples in
a contiguous, gapless order: iteration ``k`` gets its corpus's samples
``[prior_iters_same_corpus[k] * GBS + (idx % GBS)]``.

Randomness is not lost: each child ``GPTDataset`` shuffles internally through its own
seeded shuffle index, so "contiguous" here means contiguous in the child's already-
shuffled sample space.
"""

import numpy

from megatron.bridge.training.gradient_routing.plan import GRPlan


class GRRoutedDataset:
    """N+1 child GPTDatasets behind one indexable dataset, routed per-iteration by a GRPlan.

    ``children`` maps corpus label -> dataset: label 0 is the core corpus, labels 1..N the
    aux corpora, exactly as the plan's ``corpus`` array names them.
    """

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
