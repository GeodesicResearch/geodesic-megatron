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
``aux_lr`` / ``aux_min_lr`` / ``plan_seed`` / ``aux_ffn_hidden_size`` have NO defaults:
the learning rate of a freshly initialised module, the routing sequence, and the added
capacity are exactly the choices a run must make explicitly.

``GRDatasetConfig`` carries the per-corpus blends for the routed dataset. It subclasses
the bridge ``GPTDatasetConfig`` so every consumer of ``cfg.dataset`` (sequence length,
seed, dataloader fields) keeps working, but it never runs the MCore blend post-init on
itself — the two child corpus configs, built in ``build_child_config``, are the ones that
reach ``BlendedMegatronDatasetBuilder``.
"""

import copy
from dataclasses import dataclass

from megatron.bridge.training.config import DataloaderConfig, GPTDatasetConfig
from megatron.bridge.training.gradient_routing.plan import GRPlan, build_gr_plan


@dataclass
class GradientRoutingConfig:
    """The ``gr:`` section of a run config. Presence with ``enabled: true`` turns GR on."""

    enabled: bool = False
    """Master switch. When False (or when the section is absent) every GR code path is inert."""

    retain_data_path: list[str] | None = None
    """Blend list (interleaved weights + .bin/.idx prefixes) for the normally-trained corpus."""

    forget_data_path: list[str] | None = None
    """Blend list for the routed corpus — trains ONLY the aux modules on isolated iterations."""

    aux_ffn_hidden_size: int | None = None
    """Width of each per-MoE-layer aux MLP. Cross-checked against model.gr_aux_ffn_hidden_size."""

    p_as: float = 0.5
    """Auxiliary spread: share of forget iterations that ALSO update the core (paper default)."""

    p_cr: float = 0.2
    """Core robustness: share of retain iterations that also activate + update aux (paper default)."""

    forget_iter_fraction: float = 0.5
    """Share of iterations drawing the forget corpus."""

    plan_seed: int | None = None
    """Seed of the routing plan. REQUIRED: the plan is part of the experiment's identity."""

    aux_lr: float | None = None
    """Max LR for the aux param group. REQUIRED: a fresh zero-init module and a warm core
    need different LRs, and which to use is a per-run decision."""

    aux_min_lr: float | None = None
    """Min LR for the aux param group. REQUIRED alongside aux_lr."""

    aux_wd_mult: float = 1.0
    """Weight-decay multiplier for the aux param group."""

    log_interval: int = 1
    """Iterations between the heavier telemetry probes (aux output RMS, param norms)."""

    def finalize(self) -> None:
        """Validate the section; raises with a fix instruction on any gap."""
        if not self.enabled:
            return
        missing = [
            name
            for name in (
                "retain_data_path",
                "forget_data_path",
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
        for name, p in (("p_as", self.p_as), ("p_cr", self.p_cr), ("forget_iter_fraction", self.forget_iter_fraction)):
            if not 0.0 <= p <= 1.0:
                raise ValueError(f"gr.{name} must be in [0, 1], got {p}.")
        if self.aux_ffn_hidden_size <= 0:
            raise ValueError(f"gr.aux_ffn_hidden_size must be positive, got {self.aux_ffn_hidden_size}.")
        if self.aux_lr <= 0 or self.aux_min_lr < 0:
            raise ValueError(f"gr.aux_lr must be > 0 and gr.aux_min_lr >= 0, got {self.aux_lr}/{self.aux_min_lr}.")
        if self.log_interval < 1:
            raise ValueError(f"gr.log_interval must be >= 1, got {self.log_interval}.")

    def build_plan(self, train_iters: int) -> GRPlan:
        """Build the routing plan for a run of ``train_iters`` iterations."""
        return build_gr_plan(
            plan_seed=self.plan_seed,
            train_iters=train_iters,
            forget_iter_fraction=self.forget_iter_fraction,
            p_as=self.p_as,
            p_cr=self.p_cr,
        )


class GRDatasetConfig(GPTDatasetConfig):
    """Dataset config for GR runs: two corpus blends behind one ``cfg.dataset`` object.

    ``data_path``/``blend`` on this object stay unset; ``build_child_config`` clones this
    config into a plain ``GPTDatasetConfig`` per corpus, and the dataset provider builds
    one ``GPTDataset`` from each. ``finalize`` deliberately skips the MCore blend
    post-init for the parent (it has no blend to derive) while keeping the
    ``DataloaderConfig`` finalization every consumer of ``cfg.dataset`` relies on.
    """

    def __init__(
        self,
        retain_data_path: list[str],
        forget_data_path: list[str],
        gr_plan: GRPlan,
        gr_global_batch_size: int,
        *args,
        **kwargs,
    ):
        if kwargs.get("data_path") is not None or kwargs.get("blend") is not None:
            raise ValueError(
                "GRDatasetConfig takes retain_data_path/forget_data_path; do not set data_path/blend on it."
            )
        self.retain_data_path = list(retain_data_path)
        self.forget_data_path = list(forget_data_path)
        self.gr_plan = gr_plan
        self.gr_global_batch_size = gr_global_batch_size
        super().__init__(*args, **kwargs)

    def finalize(self) -> None:
        """Finalize dataloader fields; the corpus blends are finalized on the children."""
        if not self.retain_data_path or not self.forget_data_path:
            raise ValueError("GRDatasetConfig requires non-empty retain_data_path and forget_data_path.")
        DataloaderConfig.finalize(self)

    def build_child_config(self, corpus_data_path: list[str]) -> GPTDatasetConfig:
        """Clone this config into a single-corpus GPTDatasetConfig, finalized and buildable.

        The clone is re-classed to plain ``GPTDatasetConfig`` (both are attribute-compatible
        dataclass lineages differing only in the two path fields and finalize behaviour) so
        the dataset provider's ``isinstance(_, GRDatasetConfig)`` dispatch cannot recurse
        into the GR branch when building a child.
        """
        child = copy.deepcopy(self)
        child.__class__ = GPTDatasetConfig
        del child.retain_data_path
        del child.forget_data_path
        del child.gr_plan
        del child.gr_global_batch_size
        child.data_path = list(corpus_data_path)
        child.blend = None
        GPTDatasetConfig.finalize(child)
        return child
