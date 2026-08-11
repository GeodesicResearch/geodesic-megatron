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
"""The deterministic gradient-routing plan over one core corpus and N aux corpora.

One optimizer step is one plan entry (GRAM's uniform-accumulation regime): every
microbatch of iteration ``i`` draws from ``corpus[i]`` and the step updates exactly the
parameter sets named by ``update_core[i]`` / ``update_aux[i]``. The plan is a pure
function of its config tuple, so every rank — and every restart — derives the identical
sequence from the iteration number alone; nothing about routing travels through batches
or collectives.

Corpus labels: ``0`` is the core (retain) corpus; ``1..N`` are the aux corpora, and aux
corpus ``c`` trains aux MODULE ``c - 1``. Sub-label allocation follows the GRAM
reference implementation: exact per-label counts (``round(p * n)``) placed by seeded
permutations, rather than i.i.d. Bernoulli draws — the realised fractions match the
configured probabilities exactly.

The four iteration types (paper Table / reference ``do_routed_unordered``), per module:

==================  ========  ==========  ===========  =============
type                corpus    fwd_aux[k]  update_core  update_aux[k]
==================  ========  ==========  ===========  =============
aux-isolated        aux k+1   yes         no           yes
aux-spread          aux k+1   yes         yes (p_as)   yes
core                core      no          yes          no
core-robustness     core      yes (one k) yes          yes (p_cr)
==================  ========  ==========  ===========  =============

A core-robustness iteration activates exactly ONE aux module (the reference allocates
these among modules in proportion to their data shares, which here are the configured
iteration fractions).
"""

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


CORE = 0
FIRST_AUX = 1


@dataclass(frozen=True)
class GRPlan:
    """Per-iteration routing arrays plus per-corpus consumption offsets.

    ``corpus`` and ``update_core`` have length ``train_iters``; ``fwd_aux`` and
    ``update_aux`` have shape ``(train_iters, n_aux)`` with column ``k`` describing aux
    module ``k``. ``prior_iters_same_corpus[i]`` counts the iterations before ``i`` that
    drew from the same corpus — the routed dataset uses it to map an iteration onto a
    contiguous, gapless window of its corpus's samples.
    """

    corpus: np.ndarray
    fwd_aux: np.ndarray
    update_core: np.ndarray
    update_aux: np.ndarray
    prior_iters_same_corpus: np.ndarray
    plan_seed: int
    p_as: float
    p_cr: float
    aux_iter_fractions: tuple[float, ...]

    @property
    def train_iters(self) -> int:
        """Number of iterations the plan covers."""
        return len(self.corpus)

    @property
    def n_aux(self) -> int:
        """Number of aux modules (= number of aux corpora)."""
        return self.fwd_aux.shape[1]

    @property
    def n_core_iters(self) -> int:
        """Number of core-corpus iterations."""
        return int((self.corpus == CORE).sum())

    def n_corpus_iters(self, corpus: int) -> int:
        """Number of iterations drawing one corpus (0 = core, 1..N = aux)."""
        return int((self.corpus == corpus).sum())

    def n_samples(self, corpus: int, global_batch_size: int) -> int:
        """Total samples the plan consumes from one corpus."""
        return self.n_corpus_iters(corpus) * global_batch_size

    def digest(self) -> str:
        """Stable hash of the full plan (arrays + parameters) for run provenance."""
        h = hashlib.sha256()
        fractions = ",".join(repr(float(f)) for f in self.aux_iter_fractions)
        h.update(
            f"seed={self.plan_seed};p_as={self.p_as};p_cr={self.p_cr};"
            f"n_aux={self.n_aux};aux_fractions=[{fractions}];".encode()
        )
        for arr in (self.corpus, self.fwd_aux, self.update_core, self.update_aux):
            h.update(np.ascontiguousarray(arr, dtype=np.int64).tobytes())
        return h.hexdigest()[:16]

    def describe(self) -> str:
        """One-paragraph summary for logs."""
        per_corpus = ", ".join(f"aux{k}={self.n_corpus_iters(k + FIRST_AUX)}" for k in range(self.n_aux))
        spread = int((self.update_core[self.corpus != CORE]).sum())
        robust = int((self.update_aux[self.corpus == CORE].any(axis=1)).sum())
        return (
            f"GRPlan(iters={self.train_iters}, core={self.n_core_iters}, {per_corpus}, "
            f"aux_spread={spread}, core_robustness={robust}, "
            f"aux_update_steps={int(self.update_aux.any(axis=1).sum())}, "
            f"core_update_steps={int(self.update_core.sum())}, "
            f"seed={self.plan_seed}, digest={self.digest()})"
        )


def build_gr_plan(
    plan_seed: int,
    train_iters: int,
    aux_iter_fractions: Sequence[float],
    p_as: float,
    p_cr: float,
) -> GRPlan:
    """Build the deterministic routing plan.

    Exact-count allocation: one seeded permutation of the iterations is sliced
    sequentially so ``round(f_k * iters)`` iterations draw aux corpus ``k`` and the
    remainder draw core. Among each aux corpus's iterations, ``round(p_as * n_k)`` are
    aux-spread (also update core). Among the core iterations, each module ``k`` receives
    ``round(p_cr * n_core * f_k / sum(f))`` core-robustness iterations (activate +
    update that one module), assigned by slicing one permutation of the core
    iterations — the reference implementation's proportional allocation. Every
    placement comes from ``numpy.random.Generator(PCG64(plan_seed))``, so the plan is
    bit-identical across ranks, processes, and restarts.
    """
    if train_iters <= 0:
        raise ValueError(f"train_iters must be positive, got {train_iters}.")
    fractions = [float(f) for f in aux_iter_fractions]
    if not fractions:
        raise ValueError("aux_iter_fractions must name at least one aux corpus.")
    for k, f in enumerate(fractions):
        if not 0.0 <= f <= 1.0:
            raise ValueError(f"aux_iter_fractions[{k}] must be in [0, 1], got {f}.")
    if sum(fractions) > 1.0:
        raise ValueError(f"aux_iter_fractions must sum to <= 1, got {sum(fractions)}.")
    for name, p in (("p_as", p_as), ("p_cr", p_cr)):
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"{name} must be in [0, 1], got {p}.")

    n_aux = len(fractions)
    rng = np.random.Generator(np.random.PCG64(plan_seed))

    n_per_aux = [int(round(f * train_iters)) for f in fractions]
    if sum(n_per_aux) > train_iters:
        raise ValueError(
            f"aux iteration counts {n_per_aux} exceed train_iters={train_iters} after rounding; "
            "lower the fractions or raise train_iters."
        )
    corpus = np.full(train_iters, CORE, dtype=np.int64)
    perm = rng.permutation(train_iters)
    start = 0
    for k, n_k in enumerate(n_per_aux):
        corpus[perm[start : start + n_k]] = k + FIRST_AUX
        start += n_k

    fwd_aux = np.zeros((train_iters, n_aux), dtype=np.int64)
    update_core = np.zeros(train_iters, dtype=np.int64)
    update_aux = np.zeros((train_iters, n_aux), dtype=np.int64)

    # Aux iterations: always forward + update their own module; a p_as share also
    # updates core.
    for k in range(n_aux):
        idx_k = np.flatnonzero(corpus == k + FIRST_AUX)
        fwd_aux[idx_k, k] = 1
        update_aux[idx_k, k] = 1
        n_spread = int(round(p_as * len(idx_k)))
        update_core[rng.permutation(idx_k)[:n_spread]] = 1

    # Core iterations: always update core; a p_cr share also activates + updates ONE
    # module, allocated among modules in proportion to their iteration fractions.
    core_idx = np.flatnonzero(corpus == CORE)
    update_core[core_idx] = 1
    total_fraction = sum(fractions)
    n_robust = [
        int(round(p_cr * len(core_idx) * (f / total_fraction if total_fraction > 0 else 0.0))) for f in fractions
    ]
    if sum(n_robust) > len(core_idx):
        raise ValueError(
            f"core-robustness counts {n_robust} exceed the {len(core_idx)} core iterations after "
            "rounding; slicing would silently under-allocate the last module(s). Lower p_cr or "
            "adjust the fractions/train_iters so the per-module rounds fit."
        )
    robust_perm = rng.permutation(core_idx)
    start = 0
    for k, n_robust_k in enumerate(n_robust):
        robust_k = robust_perm[start : start + n_robust_k]
        fwd_aux[robust_k, k] = 1
        update_aux[robust_k, k] = 1
        start += n_robust_k

    prior = np.zeros(train_iters, dtype=np.int64)
    counts = np.zeros(n_aux + 1, dtype=np.int64)
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
        aux_iter_fractions=tuple(fractions),
    )
