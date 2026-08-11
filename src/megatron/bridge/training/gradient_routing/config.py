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
"""Configuration objects for gradient routing.

``GradientRoutingConfig`` is the YAML-facing surface (the run config's ``gr:`` section).
The schema is multi-module: ``aux_data_paths`` names N routed corpora (order defines the
module indices) and ``aux_iter_fractions`` gives each its share of iterations; a
single-forget run is the N=1 case, not a separate spelling. ``aux_lr`` / ``aux_min_lr``
/ ``plan_seed`` / ``aux_ffn_hidden_size`` have NO defaults: the learning rate of freshly
initialised modules, the routing sequence, and the added capacity are exactly the
choices a run must make explicitly. Per-module fields (``aux_ffn_hidden_size``,
``aux_lr``, ``aux_min_lr``, ``aux_wd_mult``) accept a scalar — broadcast to every
module — or a per-module list.

``GRDatasetConfig`` carries the per-corpus blends for the routed dataset. It subclasses
the bridge ``GPTDatasetConfig`` so every consumer of ``cfg.dataset`` (sequence length,
seed, dataloader fields) keeps working, but it never runs the MCore blend post-init on
itself — the child corpus configs, built in ``build_child_config``, are the ones that
reach ``BlendedMegatronDatasetBuilder``.
"""

import copy
from dataclasses import dataclass

from megatron.bridge.training.config import DataloaderConfig, GPTDatasetConfig
from megatron.bridge.training.gradient_routing.plan import GRPlan, build_gr_plan


_RENAMED_FIELDS = {
    "forget_data_path": "aux_data_paths (wrap the blend list in a list: one entry per module)",
    "forget_iter_fraction": "aux_iter_fractions (a list: one fraction per module)",
}


def reject_renamed_fields(raw: dict) -> None:
    """Refuse the pre-multi-module field spellings with the rename instruction.

    Aliasing them silently would leave two spellings of the same experiment in
    circulation; a config (or a checkpoint's run_config) that still carries the old
    names must be migrated, not guessed at.
    """
    stale = {k: _RENAMED_FIELDS[k] for k in raw if k in _RENAMED_FIELDS}
    if stale:
        renames = "; ".join(f"{old} was renamed {new}" for old, new in stale.items())
        raise ValueError(f"gr: section uses pre-multi-module field names: {renames}.")


def _broadcast(value, n_aux: int, name: str) -> list:
    """Expand a scalar per-module field to ``n_aux`` entries; validate list lengths."""
    if isinstance(value, (list, tuple)):
        if len(value) != n_aux:
            raise ValueError(f"gr.{name} has {len(value)} entries for {n_aux} aux modules.")
        return list(value)
    return [value] * n_aux


@dataclass
class GradientRoutingConfig:
    """The ``gr:`` section of a run config. Presence with ``enabled: true`` turns GR on."""

    enabled: bool = False
    """Master switch. When False (or when the section is absent) every GR code path is inert."""

    retain_data_path: list[str] | None = None
    """Blend list (interleaved weights + .bin/.idx prefixes) for the normally-trained corpus."""

    aux_data_paths: list[list[str]] | None = None
    """One blend list per routed corpus; entry k trains ONLY aux module k on isolated iterations."""

    aux_iter_fractions: list[float] | None = None
    """Share of iterations drawing each aux corpus (one entry per module, sum <= 1)."""

    aux_ffn_hidden_size: int | list[int] | None = None
    """Width of each per-MoE-layer aux MLP (scalar broadcasts to every module).
    Cross-checked against model.gr_aux_ffn_hidden_size."""

    p_as: float = 0.5
    """Auxiliary spread: share of each aux corpus's iterations that ALSO update the core
    (default = the paper's realistic dual-use setting, §5; its Simple Stories setting uses 0.3)."""

    p_cr: float = 0.2
    """Core robustness: share of core iterations that also activate + update one aux module
    (default = the paper's realistic dual-use setting, §5; its Simple Stories setting uses 0.5)."""

    plan_seed: int | None = None
    """Seed of the routing plan. REQUIRED: the plan is part of the experiment's identity."""

    aux_lr: float | list[float] | None = None
    """Max LR for the aux param groups (scalar broadcasts). REQUIRED: a fresh zero-init
    module and a warm core need different LRs, and which to use is a per-run decision."""

    aux_min_lr: float | list[float] | None = None
    """Min LR for the aux param groups (scalar broadcasts). REQUIRED alongside aux_lr."""

    aux_wd_mult: float | list[float] = 1.0
    """Weight-decay multiplier for the aux param groups (scalar broadcasts)."""

    log_interval: int = 1
    """Iterations between the heavier telemetry probes (aux output RMS, param norms)."""

    @property
    def n_aux(self) -> int:
        """Number of aux modules (= number of routed corpora)."""
        return len(self.aux_data_paths) if self.aux_data_paths else 0

    def finalize(self) -> None:
        """Validate the section; raises with a fix instruction on any gap."""
        if not self.enabled:
            return
        missing = [
            name
            for name in (
                "retain_data_path",
                "aux_data_paths",
                "aux_iter_fractions",
                "plan_seed",
                "aux_lr",
                "aux_min_lr",
                "aux_ffn_hidden_size",
            )
            if getattr(self, name) is None
        ]
        if missing:
            raise ValueError(
                f"gr.enabled=true but {missing} unset. Every one of these is a load-bearing "
                "experimental choice with no sensible default — set them in the run config's gr: section."
            )
        if not self.aux_data_paths or any(not blend for blend in self.aux_data_paths):
            raise ValueError("gr.aux_data_paths must be a non-empty list of non-empty blend lists.")
        n_aux = self.n_aux
        # Unlike the width/LR fields, a fraction must NOT scalar-broadcast: the same value
        # for every module multiplies the TOTAL aux share by the module count, which is
        # never what a scalar spelling says. Refuse here rather than let build_plan die
        # iterating a float at model-build time.
        if not isinstance(self.aux_iter_fractions, (list, tuple)):
            raise ValueError(
                f"gr.aux_iter_fractions must be a list with one fraction per aux module "
                f"(got {self.aux_iter_fractions!r} for {n_aux} module(s))."
            )
        fractions = _broadcast(self.aux_iter_fractions, n_aux, "aux_iter_fractions")
        for k, f in enumerate(fractions):
            if not 0.0 <= float(f) <= 1.0:
                raise ValueError(f"gr.aux_iter_fractions[{k}] must be in [0, 1], got {f}.")
        if sum(float(f) for f in fractions) > 1.0:
            raise ValueError(f"gr.aux_iter_fractions must sum to <= 1, got {fractions}.")
        for name, p in (("p_as", self.p_as), ("p_cr", self.p_cr)):
            if not 0.0 <= p <= 1.0:
                raise ValueError(f"gr.{name} must be in [0, 1], got {p}.")
        for k, width in enumerate(self.aux_ffn_hidden_sizes()):
            if width <= 0:
                raise ValueError(f"gr.aux_ffn_hidden_size[{k}] must be positive, got {width}.")
        for k, (lr, min_lr) in enumerate(zip(self.aux_lrs(), self.aux_min_lrs())):
            if lr <= 0 or min_lr < 0:
                raise ValueError(f"gr.aux_lr[{k}] must be > 0 and gr.aux_min_lr[{k}] >= 0, got {lr}/{min_lr}.")
        _broadcast(self.aux_wd_mult, n_aux, "aux_wd_mult")
        if self.log_interval < 1:
            raise ValueError(f"gr.log_interval must be >= 1, got {self.log_interval}.")

    def aux_ffn_hidden_sizes(self) -> list[int]:
        """Per-module aux widths (scalar broadcast applied)."""
        return [int(w) for w in _broadcast(self.aux_ffn_hidden_size, self.n_aux, "aux_ffn_hidden_size")]

    def aux_lrs(self) -> list[float]:
        """Per-module max LRs (scalar broadcast applied)."""
        return [float(v) for v in _broadcast(self.aux_lr, self.n_aux, "aux_lr")]

    def aux_min_lrs(self) -> list[float]:
        """Per-module min LRs (scalar broadcast applied)."""
        return [float(v) for v in _broadcast(self.aux_min_lr, self.n_aux, "aux_min_lr")]

    def aux_wd_mults(self) -> list[float]:
        """Per-module weight-decay multipliers (scalar broadcast applied)."""
        return [float(v) for v in _broadcast(self.aux_wd_mult, self.n_aux, "aux_wd_mult")]

    def build_plan(self, train_iters: int) -> GRPlan:
        """Build the routing plan for a run of ``train_iters`` iterations."""
        return build_gr_plan(
            plan_seed=self.plan_seed,
            train_iters=train_iters,
            aux_iter_fractions=[float(f) for f in self.aux_iter_fractions],
            p_as=self.p_as,
            p_cr=self.p_cr,
        )


class GRDatasetConfig(GPTDatasetConfig):
    """Dataset config for GR runs: N+1 corpus blends behind one ``cfg.dataset`` object.

    ``data_path``/``blend`` on this object stay unset; ``build_child_config`` clones this
    config into a plain ``GPTDatasetConfig`` per corpus, and the dataset provider builds
    one ``GPTDataset`` from each. ``finalize`` deliberately skips the MCore blend
    post-init for the parent (it has no blend to derive) while keeping the
    ``DataloaderConfig`` finalization every consumer of ``cfg.dataset`` relies on.
    """

    def __init__(
        self,
        retain_data_path: list[str],
        aux_data_paths: list[list[str]],
        gr_plan: GRPlan,
        gr_global_batch_size: int,
        *args,
        **kwargs,
    ):
        if kwargs.get("data_path") is not None or kwargs.get("blend") is not None:
            raise ValueError(
                "GRDatasetConfig takes retain_data_path/aux_data_paths; do not set data_path/blend on it."
            )
        self.retain_data_path = list(retain_data_path)
        self.aux_data_paths = [list(blend) for blend in aux_data_paths]
        self.gr_plan = gr_plan
        self.gr_global_batch_size = gr_global_batch_size
        super().__init__(*args, **kwargs)

    def finalize(self) -> None:
        """Finalize dataloader fields; the corpus blends are finalized on the children."""
        if not self.retain_data_path or not self.aux_data_paths or any(not b for b in self.aux_data_paths):
            raise ValueError("GRDatasetConfig requires non-empty retain_data_path and aux_data_paths entries.")
        if len(self.aux_data_paths) != self.gr_plan.n_aux:
            raise ValueError(
                f"GRDatasetConfig has {len(self.aux_data_paths)} aux corpora but the plan routes "
                f"{self.gr_plan.n_aux} — the config and the plan disagree about the module count."
            )
        DataloaderConfig.finalize(self)

    def build_child_config(self, corpus_data_path: list[str]) -> GPTDatasetConfig:
        """Clone this config into a single-corpus GPTDatasetConfig, finalized and buildable.

        The clone is re-classed to plain ``GPTDatasetConfig`` (both are attribute-compatible
        dataclass lineages differing only in the corpus-path fields and finalize behaviour) so
        the dataset provider's ``isinstance(_, GRDatasetConfig)`` dispatch cannot recurse
        into the GR branch when building a child.
        """
        child = copy.deepcopy(self)
        child.__class__ = GPTDatasetConfig
        del child.retain_data_path
        del child.aux_data_paths
        del child.gr_plan
        del child.gr_global_batch_size
        child.data_path = list(corpus_data_path)
        child.blend = None
        GPTDatasetConfig.finalize(child)
        return child
