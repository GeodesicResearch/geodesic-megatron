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
"""Per-iteration optimizer gating for gradient routing.

GRAM's isolation rule is about which parameters an optimizer STEP may update, not which
parameters receive gradients: under iteration-uniform routing, gradients may accumulate
and reduce normally (they are homogeneous by construction) and the gate is applied once,
at step time — "accumulate then gate" (paper App. H, uniform regime).

The gate is param-group emptying. Before ``optimizer.step()`` the frozen roles' groups
have their ``params`` list stashed and replaced by ``[]``; after the step they are
restored. This is the only mechanism that freezes correctly under Adam:

- lr=0 still folds the live gradient into ``exp_avg``/``exp_avg_sq`` — moment
  contamination, exactly what per-label optimizers exist to prevent;
- zeroing grads and stepping still decays the moments and applies weight decay;
- an emptied group is simply never visited: no parameter write, no moment mutation, no
  weight-decay, and the gradient-norm/clip coefficient is computed over the update set
  only (``get_parameters`` walks the live ``param_groups``).

Each aux MODULE is its own role: module ``k``'s groups carry ``gr_role: "aux<k>"`` in
their ``ParamGroupOverride``. mcore's ``_get_param_groups`` copies override keys into the
group dict verbatim and GROUPS BY MERGED-OVERRIDE EQUALITY, so distinct role markers
split the modules into distinct groups even when their LR/WD are identical — which is
what lets one module step while its siblings stay frozen. Checkpoint param-group matching
uses a fixed identifier-key list, so the markers are inert everywhere except here.
"""

import logging
from dataclasses import dataclass, field

from megatron.core.optimizer import MegatronOptimizer
from megatron.core.optimizer.optimizer import ChainedOptimizer
from megatron.core.optimizer.optimizer_config import ParamKey
from megatron.core.optimizer_param_scheduler import ParamGroupOverride

from megatron.bridge.training.config import (
    OptimizerConfigOverrideProvider,
    OptimizerConfigOverrideProviderContext,
)


logger = logging.getLogger(__name__)

GR_ROLE_KEY = "gr_role"
GR_ROLE_AUX_PREFIX = "aux"
GR_AUX_NAME_FRAGMENT = ".gr_aux."


def gr_aux_param_pattern(index: int) -> str:
    """The fnmatch pattern capturing every parameter of aux module ``index``."""
    return f"*{GR_AUX_NAME_FRAGMENT}{index}.*"


def gr_aux_role(index: int) -> str:
    """The ``gr_role`` marker value for aux module ``index``."""
    return f"{GR_ROLE_AUX_PREFIX}{index}"


def _parse_aux_role(role: object) -> int | None:
    """Return the module index of an aux role marker, or None for non-aux roles."""
    if isinstance(role, str) and role.startswith(GR_ROLE_AUX_PREFIX):
        suffix = role[len(GR_ROLE_AUX_PREFIX) :]
        if suffix.isdigit():
            return int(suffix)
    return None


@dataclass
class GROptimizerConfigOverrideProvider(OptimizerConfigOverrideProvider):
    """Adds one param group per aux module (own LR/WD + role marker) on top of the standard overrides."""

    #: Required despite the empty defaults — dataclass fields with no default here would
    #: force every caller to pass them positionally past the base class's own fields.
    #: ``build_config_overrides`` refuses empty lists, so unset values never reach the optimizer.
    aux_lrs: list[float] = field(default_factory=list)
    aux_min_lrs: list[float] = field(default_factory=list)
    aux_wd_mults: list[float] = field(default_factory=list)

    def build_config_overrides(self, context: OptimizerConfigOverrideProviderContext):
        """Standard overrides plus one ParamKey per aux module.

        The per-index patterns (``*.gr_aux.<k>.*``) are mutually disjoint, so no
        parameter can match two aux overrides regardless of the matcher's precedence
        rules.
        """
        overrides = super().build_config_overrides(context) or {}
        n_aux = len(self.aux_lrs)
        if n_aux == 0 or len(self.aux_min_lrs) != n_aux or len(self.aux_wd_mults) != n_aux:
            raise ValueError(
                "GROptimizerConfigOverrideProvider requires equal-length non-empty "
                f"aux_lrs/aux_min_lrs/aux_wd_mults, got {self.aux_lrs}/{self.aux_min_lrs}/{self.aux_wd_mults}."
            )
        for k, (lr, min_lr, wd_mult) in enumerate(zip(self.aux_lrs, self.aux_min_lrs, self.aux_wd_mults)):
            aux_override = ParamGroupOverride(max_lr=lr, min_lr=min_lr, wd_mult=wd_mult)
            aux_override[GR_ROLE_KEY] = gr_aux_role(k)
            overrides[ParamKey(name=gr_aux_param_pattern(k))] = aux_override
        model_list = context.model if isinstance(context.model, list) else [context.model]
        for k in range(n_aux):
            fragment = f"{GR_AUX_NAME_FRAGMENT}{k}."
            n_params = sum(1 for chunk in model_list for name, _ in chunk.named_parameters() if fragment in name)
            if n_params == 0:
                raise ValueError(
                    f"GR optimizer override installed for aux module {k} but the model has no "
                    f"'{fragment}' parameters — the GRAM spec swap built fewer modules than the "
                    "gr: section configures. Check model.gr_aux_ffn_hidden_size wiring."
                )
        return overrides


def _iter_inner_param_groups(optimizer: MegatronOptimizer):
    """Yield every inner param-group dict across a (possibly chained) Megatron optimizer.

    For DistributedOptimizer the inner groups are the original group dicts from
    ``_get_param_groups`` with ``params`` holding this rank's state shards; group-level
    keys (including the role marker) survive intact.
    """
    wrappers = optimizer.chained_optimizers if isinstance(optimizer, ChainedOptimizer) else [optimizer]
    for wrapper in wrappers:
        for group in wrapper.optimizer.param_groups:
            yield group


class GROptimizerGater:
    """Empties/restores param groups per iteration according to the plan's update sets."""

    def __init__(self, n_aux: int):
        if n_aux <= 0:
            raise ValueError(f"GROptimizerGater requires n_aux >= 1, got {n_aux}.")
        self._n_aux = n_aux
        self._aux_groups: dict[int, list] | None = None
        self._core_groups = None
        self._stash: dict[int, list] | None = None
        self._armed_roles: frozenset | None = None
        self._frozen_groups: list | None = None

    def discover(self, optimizer: MegatronOptimizer) -> None:
        """Classify every inner param group by role; loud on anything unexpected."""
        aux: dict[int, list] = {k: [] for k in range(self._n_aux)}
        core = []
        for group in _iter_inner_param_groups(optimizer):
            index = _parse_aux_role(group.get(GR_ROLE_KEY))
            if index is None:
                core.append(group)
            elif index in aux:
                aux[index].append(group)
            else:
                raise RuntimeError(
                    f"GR gater found a group marked '{group.get(GR_ROLE_KEY)}' but only "
                    f"{self._n_aux} aux modules are configured — provider and gater disagree."
                )
        missing = [k for k, groups in aux.items() if not groups]
        if missing:
            # Structural check only: group STRUCTURE is rank-uniform (mcore all-gathers the
            # override keys), but shard OWNERSHIP is not — under the distributed optimizer
            # a rank whose data-parallel shard doesn't intersect a module's params
            # legitimately holds that module's group with zero shard tensors (observed:
            # DP4 tiny model, aux in one rank's bucket quarter). Emptiness of a PRESENT
            # group must therefore NOT be treated as a wiring failure; a wholly ABSENT
            # group means the override never reached the optimizer.
            census = "; ".join(
                f"group{i}(keys={sorted(k for k in g if k != 'params')}, n_params={len(g['params'])}, "
                f"max_lr={g.get('max_lr')}, expert={g.get('is_expert_parallel')})"
                for i, g in enumerate(_iter_inner_param_groups(optimizer))
            )
            raise RuntimeError(
                f"GR gater found no param group for aux module(s) {missing}. The aux override did "
                "not reach the optimizer — check that cfg.optimizer_config_override_provider is "
                f"the GR provider. Inner group census: {census}"
            )
        if not core:
            raise RuntimeError("GR gater found no core param group — optimizer wiring is broken.")
        self._aux_groups = aux
        self._core_groups = core
        logger.info(
            "GR gater: %d aux module(s) with %s group(s) (%s with local shards), %d core group(s).",
            self._n_aux,
            [len(groups) for groups in aux.values()],
            [sum(1 for g in groups if g["params"]) for groups in aux.values()],
            len(core),
        )

    @property
    def discovered(self) -> bool:
        """Whether group discovery has run."""
        return self._aux_groups is not None

    def arm(self, update_core: bool, update_aux) -> None:
        """Empty the groups of every role NOT in this iteration's update set.

        ``update_aux`` is one boolean per aux module. Idempotent for the same roles (the
        rerun state machine may invoke grad finalization more than once per step);
        conflicting re-arms raise.
        """
        update_aux = [bool(v) for v in update_aux]
        if len(update_aux) != self._n_aux:
            raise RuntimeError(f"GR gater armed with {len(update_aux)} aux update flags for {self._n_aux} modules.")
        roles = frozenset(
            role
            for role, updates in (
                ("core", update_core),
                *((gr_aux_role(k), update_aux[k]) for k in range(self._n_aux)),
            )
            if not updates
        )
        if self._stash is not None:
            if roles == self._armed_roles:
                return
            raise RuntimeError(
                f"GR gater re-armed with {set(roles)} while already armed with "
                f"{set(self._armed_roles)} — restore() did not run for the previous step."
            )
        if not update_core and not any(update_aux):
            raise RuntimeError("GR plan asks to update neither core nor any aux — the plan is malformed.")
        frozen = []
        if not update_core:
            frozen.extend(self._core_groups)
        for k in range(self._n_aux):
            if not update_aux[k]:
                frozen.extend(self._aux_groups[k])
        self._stash = {id(g): g["params"] for g in frozen}
        self._frozen_groups = frozen
        for g in frozen:
            g["params"] = []
        self._armed_roles = roles

    def restore(self) -> None:
        """Restore every emptied group; no-op if nothing is armed."""
        if self._stash is None:
            return
        for g in self._frozen_groups:
            g["params"] = self._stash[id(g)]
        self._stash = None
        self._armed_roles = None
        self._frozen_groups = None


#: Runtime slot for the finalize wrapper, populated by install_gr_finalize(). Module
#: state rather than a closure for one load-bearing reason: the model config's
#: ``finalize_model_grads_func`` is serialized BY IMPORT PATH into the checkpoint's
#: run_config.yaml, and a closure's ``<locals>`` qualname cannot be re-imported —
#: which broke ``AutoBridge.from_auto_config`` at export time. A module-level function
#: serializes to an importable reference; the runtime objects stay out of the config.
_GR_RUNTIME: dict | None = None


def gr_finalize_model_grads(*args, **kwargs):
    """Finalize wrapper: base grad finalization, then arm the per-iteration gate.

    The base finalize runs first — it performs the DP/expert grad reductions and the
    router expert-bias update (which honours the per-iteration ``frozen_expert_bias``
    flag the callback sets). Arming after it means the emptied groups are only ever
    visible to ``optimizer.step()`` and the scheduler, both of which are safe.
    """
    runtime = _GR_RUNTIME
    if runtime is None:
        raise RuntimeError(
            "gr_finalize_model_grads called without install_gr_finalize() — this function is "
            "only a valid finalize_model_grads_func inside a gradient-routing training run."
        )
    result = runtime["base_finalize"](*args, **kwargs)
    gater = runtime["gater"]
    if not gater.discovered:
        gater.discover(runtime["optimizer"])
    iteration = runtime["state"].train_state.step
    plan = runtime["plan"]
    gater.arm(
        update_core=bool(plan.update_core[iteration]),
        update_aux=plan.update_aux[iteration],
    )
    return result


def install_gr_finalize(base_finalize, gater: GROptimizerGater, optimizer: MegatronOptimizer, plan, state):
    """Populate the finalize runtime slot and return the importable wrapper function."""
    global _GR_RUNTIME
    if _GR_RUNTIME is not None:
        logger.info("GR finalize runtime re-installed (restart within the same process).")
    _GR_RUNTIME = {
        "base_finalize": base_finalize,
        "gater": gater,
        "optimizer": optimizer,
        "plan": plan,
        "state": state,
    }
    return gr_finalize_model_grads
