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

``GRFinetuningDatasetConfig`` is the same shape for ``--mode sft``: N+1 finetuning
dataset ROOTS (each a directory with its own ``training.jsonl``) instead of N+1
``.bin/.idx`` blends, with children built through ``FinetuningDatasetBuilder``. The two
corpus spellings on ``GradientRoutingConfig`` (``retain_data_path``/``aux_data_paths``
vs ``retain_dataset_root``/``aux_dataset_roots``) are mutually exclusive — a config
carries exactly one, and the training entry point matches it against ``--mode``.
"""

import copy
from dataclasses import dataclass

from megatron.bridge.data.datasets.packed_sequence import PackedSequenceSpecs
from megatron.bridge.training.config import DataloaderConfig, FinetuningDatasetConfig, GPTDatasetConfig
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
    """Blend list (interleaved weights + .bin/.idx prefixes) for the normally-trained corpus.
    The cpt/pretrain corpus spelling; mutually exclusive with retain_dataset_root."""

    aux_data_paths: list[list[str]] | None = None
    """One blend list per routed corpus; entry k trains ONLY aux module k on isolated iterations."""

    retain_dataset_root: str | None = None
    """Finetuning dataset root (a directory with training.jsonl) for the normally-trained corpus.
    The sft corpus spelling; mutually exclusive with retain_data_path."""

    aux_dataset_roots: list[str] | None = None
    """One finetuning dataset root per routed corpus; entry k trains ONLY aux module k."""

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
        """Number of aux modules (= number of routed corpora, in either corpus spelling)."""
        if self.aux_data_paths:
            return len(self.aux_data_paths)
        if self.aux_dataset_roots:
            return len(self.aux_dataset_roots)
        return 0

    def finalize(self) -> None:
        """Validate the section; raises with a fix instruction on any gap."""
        if not self.enabled:
            return
        problems: list[str] = []
        missing = [
            name
            for name in ("aux_iter_fractions", "plan_seed", "aux_lr", "aux_min_lr", "aux_ffn_hidden_size")
            if getattr(self, name) is None
        ]
        if missing:
            problems.append(
                f"{missing} unset. Every one of these is a load-bearing experimental choice with "
                "no sensible default — set them in the run config's gr: section."
            )
        blend_fields = [name for name in ("retain_data_path", "aux_data_paths") if getattr(self, name) is not None]
        root_fields = [
            name for name in ("retain_dataset_root", "aux_dataset_roots") if getattr(self, name) is not None
        ]
        if blend_fields and root_fields:
            problems.append(
                f"both corpus spellings are set ({blend_fields} and {root_fields}); a run trains through "
                "exactly one data stack, so set the blend-list pair (--mode cpt/pretrain) OR the "
                "dataset-root pair (--mode sft), never both."
            )
        elif not blend_fields and not root_fields:
            problems.append(
                "no corpora configured: set retain_data_path + aux_data_paths (.bin/.idx blend lists, "
                "--mode cpt/pretrain) or retain_dataset_root + aux_dataset_roots (finetuning dataset "
                "roots, --mode sft)."
            )
        elif blend_fields and len(blend_fields) != 2:
            problems.append(
                f"only {blend_fields} set; the blend-list spelling needs both retain_data_path and aux_data_paths."
            )
        elif root_fields and len(root_fields) != 2:
            problems.append(
                f"only {root_fields} set; the dataset-root spelling needs both retain_dataset_root "
                "and aux_dataset_roots."
            )
        if problems:
            raise ValueError("gr.enabled=true but " + "\n".join(problems))
        if blend_fields and (not self.aux_data_paths or any(not blend for blend in self.aux_data_paths)):
            raise ValueError("gr.aux_data_paths must be a non-empty list of non-empty blend lists.")
        if root_fields:
            if not self.retain_dataset_root:
                raise ValueError("gr.retain_dataset_root must be a non-empty dataset root path.")
            if not self.aux_dataset_roots or any(not root for root in self.aux_dataset_roots):
                raise ValueError("gr.aux_dataset_roots must be a non-empty list of non-empty dataset roots.")
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


class GRFinetuningDatasetConfig(FinetuningDatasetConfig):
    """Dataset config for GR SFT runs: N+1 finetuning dataset roots behind one ``cfg.dataset``.

    The finetuning mirror of ``GRDatasetConfig``: ``dataset_root`` / ``packed_sequence_specs``
    / ``max_train_samples`` on this object stay unset, and ``build_child_config`` clones it
    into a plain ``FinetuningDatasetConfig`` per corpus — each carrying its own root, its own
    ``PackedSequenceSpecs``, and the plan-derived exact sample cap. The reclass is load-bearing:
    ``finetuning_train_valid_test_datasets_provider`` kwargs-splats every non-DataloaderConfig
    field into ``FinetuningDatasetBuilder``, so a child that stayed this class would splat GR
    fields the builder does not accept and re-dispatch into the GR branch.

    Corpus-pure packing is structural, not configured: packing runs per ``dataset_root``
    (rank 0, offline, ``FinetuningDatasetBuilder.prepare_packed_data``), so each corpus's
    ``training.jsonl`` packs separately under its own ``<root>/packed/`` and no pack ever
    mixes documents from two corpora — the property that keeps a routed iteration's global
    batch label-homogeneous down to the token level. Per-corpus specs may therefore differ
    only in their pack-file paths; the packing POSTURE (pack size, cu_seqlens padding) must
    be identical, because one ``collate_fn`` serves every corpus's batches.
    """

    def __init__(
        self,
        retain_dataset_root: str,
        aux_dataset_roots: list[str],
        gr_plan: GRPlan,
        gr_global_batch_size: int,
        retain_packed_sequence_specs: PackedSequenceSpecs | None = None,
        aux_packed_sequence_specs: list[PackedSequenceSpecs | None] | None = None,
        *args,
        **kwargs,
    ):
        for name in ("dataset_root", "packed_sequence_specs", "max_train_samples"):
            if kwargs.get(name) is not None:
                raise ValueError(
                    f"GRFinetuningDatasetConfig takes per-corpus roots/specs and plan-derived sizing; "
                    f"do not set {name} on it."
                )
        self.retain_dataset_root = str(retain_dataset_root)
        self.aux_dataset_roots = [str(root) for root in aux_dataset_roots]
        self.retain_packed_sequence_specs = retain_packed_sequence_specs
        self.aux_packed_sequence_specs = (
            list(aux_packed_sequence_specs)
            if aux_packed_sequence_specs is not None
            else [None] * len(self.aux_dataset_roots)
        )
        self.gr_plan = gr_plan
        self.gr_global_batch_size = gr_global_batch_size
        super().__init__(*args, **kwargs)

    def finalize(self) -> None:
        """Validate the corpus roots against the plan; per-corpus fields are set on the children."""
        if not self.retain_dataset_root or not self.aux_dataset_roots or any(not r for r in self.aux_dataset_roots):
            raise ValueError(
                "GRFinetuningDatasetConfig requires non-empty retain_dataset_root and aux_dataset_roots entries."
            )
        if len(self.aux_dataset_roots) != self.gr_plan.n_aux:
            raise ValueError(
                f"GRFinetuningDatasetConfig has {len(self.aux_dataset_roots)} aux corpora but the plan routes "
                f"{self.gr_plan.n_aux} — the config and the plan disagree about the module count."
            )
        if len(self.aux_packed_sequence_specs) != len(self.aux_dataset_roots):
            raise ValueError(
                f"GRFinetuningDatasetConfig has {len(self.aux_packed_sequence_specs)} aux packed specs for "
                f"{len(self.aux_dataset_roots)} aux corpora — one PackedSequenceSpecs (or None) per corpus."
            )
        if self.do_validation or self.do_test:
            raise ValueError(
                "GRFinetuningDatasetConfig requires do_validation=False and do_test=False: the routed "
                "dataset serves no validation/test split (eval_iters 0 is guard-enforced)."
            )
        postures = {
            self._packing_posture(spec)
            for spec in (self.retain_packed_sequence_specs, *self.aux_packed_sequence_specs)
        }
        if len(postures) > 1:
            raise ValueError(
                f"GR corpora disagree about packing posture ({postures}): one collate_fn serves every "
                "corpus, so (packed_sequence_size, pad_cu_seqlens, pad_seq_to_mult) must be identical "
                "across corpora — per-corpus specs may differ only in their pack-file paths."
            )
        DataloaderConfig.finalize(self)

    @staticmethod
    def _packing_posture(spec: PackedSequenceSpecs | None) -> tuple:
        """The collate-shaping identity of one corpus's packing spec (paths excluded)."""
        if spec is None or spec.packed_sequence_size <= 0:
            return ("unpacked",)
        return ("packed", spec.packed_sequence_size, spec.pad_cu_seqlens, spec.pad_seq_to_mult)

    def build_child_config(
        self,
        corpus_dataset_root: str,
        corpus_packed_sequence_specs: PackedSequenceSpecs | None,
        max_train_samples: int,
    ) -> FinetuningDatasetConfig:
        """Clone this config into a single-corpus FinetuningDatasetConfig, exactly sized.

        ``max_train_samples`` must be the plan's exact per-corpus consumption: the SFT sample
        mapping then epoch-wraps an undersized corpus deterministically and truncates an
        oversized one, so ``len(child)`` comes out exact — which is what lets the provider
        refuse any child whose length disagrees with the plan instead of letting the batch
        sampler wrap mid-run.
        """
        child = copy.deepcopy(self)
        child.__class__ = FinetuningDatasetConfig
        del child.retain_dataset_root
        del child.aux_dataset_roots
        del child.retain_packed_sequence_specs
        del child.aux_packed_sequence_specs
        del child.gr_plan
        del child.gr_global_batch_size
        child.dataset_root = str(corpus_dataset_root)
        child.packed_sequence_specs = corpus_packed_sequence_specs
        child.max_train_samples = int(max_train_samples)
        child.do_validation = False
        child.do_test = False
        FinetuningDatasetConfig.finalize(child)
        return child
