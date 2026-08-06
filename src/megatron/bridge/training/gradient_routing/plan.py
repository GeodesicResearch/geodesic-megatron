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
"""The deterministic gradient-routing plan.

One optimizer step is one plan entry (GRAM's uniform-accumulation regime): every
microbatch of iteration ``i`` draws from ``corpus[i]`` and the step updates exactly the
parameter sets named by ``update_core[i]`` / ``update_aux[i]``. The plan is a pure
function of its config tuple, so every rank — and every restart — derives the identical
sequence from the iteration number alone; nothing about routing travels through batches
or collectives.

Sub-label allocation follows the GRAM reference implementation: exact per-label counts
(``round(p * n)``) placed by a seeded permutation, rather than i.i.d. Bernoulli draws —
the realised fractions match the configured probabilities exactly.

The four iteration types (paper Table / reference ``do_routed_unordered``):

==================  ======  =======  ===========  ==========
type                corpus  fwd_aux  update_core  update_aux
==================  ======  =======  ===========  ==========
forget-isolated     forget  yes      no           yes
forget-spread       forget  yes      yes (p_as)   yes
core                retain  no       yes          no
core-robustness     retain  yes      yes          yes (p_cr)
==================  ======  =======  ===========  ==========
"""

import hashlib
from dataclasses import dataclass

import numpy as np


RETAIN = 0
FORGET = 1


@dataclass(frozen=True)
class GRPlan:
    """Per-iteration routing arrays plus per-corpus consumption offsets.

    All arrays have length ``train_iters``. ``prior_iters_same_corpus[i]`` counts the
    iterations before ``i`` that drew from the same corpus — the routed dataset uses it
    to map an iteration onto a contiguous, gapless window of its corpus's samples.
    """

    corpus: np.ndarray
    fwd_aux: np.ndarray
    update_core: np.ndarray
    update_aux: np.ndarray
    prior_iters_same_corpus: np.ndarray
    plan_seed: int
    p_as: float
    p_cr: float
    forget_iter_fraction: float

    @property
    def train_iters(self) -> int:
        """Number of iterations the plan covers."""
        return len(self.corpus)

    @property
    def n_forget_iters(self) -> int:
        """Number of forget-corpus iterations."""
        return int((self.corpus == FORGET).sum())

    @property
    def n_retain_iters(self) -> int:
        """Number of retain-corpus iterations."""
        return int((self.corpus == RETAIN).sum())

    def n_samples(self, corpus: int, global_batch_size: int) -> int:
        """Total samples the plan consumes from one corpus."""
        return int((self.corpus == corpus).sum()) * global_batch_size

    def digest(self) -> str:
        """Stable hash of the full plan (arrays + parameters) for run provenance."""
        h = hashlib.sha256()
        h.update(f"seed={self.plan_seed};p_as={self.p_as};p_cr={self.p_cr};f={self.forget_iter_fraction};".encode())
        for arr in (self.corpus, self.fwd_aux, self.update_core, self.update_aux):
            h.update(np.ascontiguousarray(arr, dtype=np.int64).tobytes())
        return h.hexdigest()[:16]

    def describe(self) -> str:
        """One-paragraph summary for logs."""
        return (
            f"GRPlan(iters={self.train_iters}, forget={self.n_forget_iters}, retain={self.n_retain_iters}, "
            f"forget_spread={int(((self.corpus == FORGET) & self.update_core.astype(bool)).sum())}, "
            f"core_robustness={int(((self.corpus == RETAIN) & self.update_aux.astype(bool)).sum())}, "
            f"aux_update_steps={int(self.update_aux.sum())}, core_update_steps={int(self.update_core.sum())}, "
            f"seed={self.plan_seed}, digest={self.digest()})"
        )


def build_gr_plan(plan_seed: int, train_iters: int, forget_iter_fraction: float, p_as: float, p_cr: float) -> GRPlan:
    """Build the deterministic routing plan.

    Exact-count allocation: ``n_forget = round(f * iters)`` iterations draw the forget
    corpus; among those, ``round(p_as * n_forget)`` are forget-spread (also update core);
    among the retain iterations, ``round(p_cr * n_retain)`` are core-robustness (also
    activate + update aux). Placement of every subset is a permutation from
    ``numpy.random.Generator(PCG64(plan_seed))``, so the plan is bit-identical across
    ranks, processes, and restarts.
    """
    if train_iters <= 0:
        raise ValueError(f"train_iters must be positive, got {train_iters}.")
    for name, p in (("forget_iter_fraction", forget_iter_fraction), ("p_as", p_as), ("p_cr", p_cr)):
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"{name} must be in [0, 1], got {p}.")

    rng = np.random.Generator(np.random.PCG64(plan_seed))

    n_forget = int(round(forget_iter_fraction * train_iters))
    corpus = np.full(train_iters, RETAIN, dtype=np.int64)
    corpus[rng.permutation(train_iters)[:n_forget]] = FORGET

    forget_idx = np.flatnonzero(corpus == FORGET)
    retain_idx = np.flatnonzero(corpus == RETAIN)

    fwd_aux = np.zeros(train_iters, dtype=np.int64)
    update_core = np.zeros(train_iters, dtype=np.int64)
    update_aux = np.zeros(train_iters, dtype=np.int64)

    # Forget iterations: always forward + update aux; a p_as share also updates core.
    fwd_aux[forget_idx] = 1
    update_aux[forget_idx] = 1
    n_spread = int(round(p_as * len(forget_idx)))
    update_core[rng.permutation(forget_idx)[:n_spread]] = 1

    # Retain iterations: always update core; a p_cr share also activates + updates aux.
    update_core[retain_idx] = 1
    n_robust = int(round(p_cr * len(retain_idx)))
    robust = rng.permutation(retain_idx)[:n_robust]
    fwd_aux[robust] = 1
    update_aux[robust] = 1

    prior = np.zeros(train_iters, dtype=np.int64)
    counts = {RETAIN: 0, FORGET: 0}
    for i in range(train_iters):
        c = int(corpus[i])
        prior[i] = counts[c]
        counts[c] += 1

    return GRPlan(
        corpus=corpus,
        fwd_aux=fwd_aux,
        update_core=update_core,
        update_aux=update_aux,
        prior_iters_same_corpus=prior,
        plan_seed=plan_seed,
        p_as=p_as,
        p_cr=p_cr,
        forget_iter_fraction=forget_iter_fraction,
    )
