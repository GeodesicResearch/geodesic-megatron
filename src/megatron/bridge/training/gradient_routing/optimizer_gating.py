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

The gate is param-group emptying. Before ``optimizer.step()`` the frozen role's groups
have their ``params`` list stashed and replaced by ``[]``; after the step they are
restored. This is the only mechanism that freezes correctly under Adam:

- lr=0 still folds the live gradient into ``exp_avg``/``exp_avg_sq`` — moment
  contamination, exactly what per-label optimizers exist to prevent;
- zeroing grads and stepping still decays the moments and applies weight decay;
- an emptied group is simply never visited: no parameter write, no moment mutation, no
  weight-decay, and the gradient-norm/clip coefficient is computed over the update set
  only (``get_parameters`` walks the live ``param_groups``).

Aux groups are identified by a ``gr_role: "aux"`` marker riding in the aux
``ParamGroupOverride``: mcore's ``_get_param_groups`` copies override keys into the group
dict verbatim, and checkpoint param-group matching uses a fixed identifier-key list, so
the marker is inert everywhere except here. The distinct aux ``max_lr``/``min_lr``
override is ALSO what guarantees aux parameters land in their own group(s) in the first
place (grouping is by merged-override equality).
"""

import logging
from dataclasses import dataclass

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
GR_ROLE_AUX = "aux"
GR_AUX_PARAM_PATTERN = "*.gr_aux.*"
GR_AUX_NAME_FRAGMENT = ".gr_aux."


@dataclass
class GROptimizerConfigOverrideProvider(OptimizerConfigOverrideProvider):
    """Adds the aux param group (own LR/WD + role marker) on top of the standard overrides."""

    #: Required despite the None default — a dataclass field with no default here would
    #: force every caller to pass it positionally past the base class's own fields.
    #: ``build_config_overrides`` refuses None, so an unset value never reaches the optimizer.
    aux_lr: float | None = None
    aux_min_lr: float | None = None
    aux_wd_mult: float = 1.0

    def build_config_overrides(self, context: OptimizerConfigOverrideProviderContext):
        """Standard overrides plus one ParamKey capturing every ``gr_aux`` parameter."""
        overrides = super().build_config_overrides(context) or {}
        if self.aux_lr is None or self.aux_min_lr is None:
            raise ValueError("GROptimizerConfigOverrideProvider requires explicit aux_lr and aux_min_lr.")
        aux_override = ParamGroupOverride(max_lr=self.aux_lr, min_lr=self.aux_min_lr, wd_mult=self.aux_wd_mult)
        aux_override[GR_ROLE_KEY] = GR_ROLE_AUX
        overrides[ParamKey(name=GR_AUX_PARAM_PATTERN)] = aux_override
        model_list = context.model if isinstance(context.model, list) else [context.model]
        n_aux = sum(1 for chunk in model_list for name, _ in chunk.named_parameters() if GR_AUX_NAME_FRAGMENT in name)
        if n_aux == 0:
            raise ValueError(
                "GR optimizer override installed but the model has no '.gr_aux.' parameters — "
                "the GRAM spec swap did not run. Check model.gr_aux_ffn_hidden_size wiring."
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

    def __init__(self):
        self._aux_groups = None
        self._core_groups = None
        self._stash: dict[int, list] | None = None
        self._armed_roles: frozenset | None = None
        self._frozen_groups: list | None = None

    def discover(self, optimizer: MegatronOptimizer) -> None:
        """Classify every inner param group by role; loud on anything unexpected."""
        aux, core = [], []
        for group in _iter_inner_param_groups(optimizer):
            (aux if group.get(GR_ROLE_KEY) == GR_ROLE_AUX else core).append(group)
        if not aux:
            # Structural check only: group STRUCTURE is rank-uniform (mcore all-gathers the
            # override keys), but shard OWNERSHIP is not — under the distributed optimizer
            # a rank whose data-parallel shard doesn't intersect the aux params legitimately
            # holds an aux group with zero shard tensors (observed: DP4 tiny model, aux in
            # one rank's bucket quarter). Emptiness must therefore NOT be treated as a
            # wiring failure; emptying an already-empty group is a correct no-op.
            census = "; ".join(
                f"group{i}(keys={sorted(k for k in g if k != 'params')}, n_params={len(g['params'])}, "
                f"max_lr={g.get('max_lr')}, expert={g.get('is_expert_parallel')})"
                for i, g in enumerate(_iter_inner_param_groups(optimizer))
            )
            raise RuntimeError(
                "GR gater found no aux-marked param group. The aux override did not reach the "
                "optimizer — check that cfg.optimizer_config_override_provider is the GR provider. "
                f"Inner group census: {census}"
            )
        if not core:
            raise RuntimeError("GR gater found no core param group — optimizer wiring is broken.")
        self._aux_groups = aux
        self._core_groups = core
        logger.info(
            "GR gater: %d aux group(s) (%d with local shards), %d core group(s).",
            len(aux),
            sum(1 for g in aux if g["params"]),
            len(core),
        )

    @property
    def discovered(self) -> bool:
        """Whether group discovery has run."""
        return self._aux_groups is not None

    def arm(self, update_core: bool, update_aux: bool) -> None:
        """Empty the groups of every role NOT in this iteration's update set.

        Idempotent for the same roles (the rerun state machine may invoke grad
        finalization more than once per step); conflicting re-arms raise.
        """
        roles = frozenset(role for role, updates in (("core", update_core), ("aux", update_aux)) if not updates)
        if self._stash is not None:
            if roles == self._armed_roles:
                return
            raise RuntimeError(
                f"GR gater re-armed with {set(roles)} while already armed with "
                f"{set(self._armed_roles)} — restore() did not run for the previous step."
            )
        if not update_core and not update_aux:
            raise RuntimeError("GR plan asks to update neither core nor aux — the plan is malformed.")
        frozen = []
        if not update_core:
            frozen.extend(self._core_groups)
        if not update_aux:
            frozen.extend(self._aux_groups)
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
        update_aux=bool(plan.update_aux[iteration]),
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
