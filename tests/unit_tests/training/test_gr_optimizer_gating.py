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
"""Optimizer-side isolation: a frozen role must not move, and neither must its Adam state.

This is the half of GRAM that makes routing mean anything. The module's argument is that
param-group emptying is the ONLY correct freeze under Adam — lr=0 still folds the live
gradient into ``exp_avg``/``exp_avg_sq``, and zeroing grads still decays the moments and
applies weight decay. That argument is about the real optimizer's real behaviour, so the
tests run a REAL ``get_megatron_optimizer`` over a real DDP-wrapped module and step it,
asserting bitwise equality of both the parameters and the moment buffers.

Testing it against a stand-in Adam would prove nothing about the claim: the whole point is
what mcore's optimizer does to state the emptying is supposed to protect. The model is
tiny (two 8x8 blocks) because the assertion is about which tensors the optimizer touches,
not about what they contain.
"""

from types import SimpleNamespace

import pytest
import torch

from tests.unit_tests.gr_test_utils import init_model_parallel


requires_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU (real Megatron optimizer)")

HIDDEN = 8
CORE_PARAMS = ("mlp.core.linear_fc1.weight", "mlp.core.linear_fc2.weight")
AUX_PARAMS = ("mlp.gr_aux.linear_fc1.weight", "mlp.gr_aux.linear_fc2.weight")
AUX_LR, AUX_MIN_LR, AUX_WD_MULT = 1e-2, 1e-3, 0.5
BASE_LR, BASE_MIN_LR = 1e-3, 1e-4


@pytest.fixture(scope="module")
def model_parallel():
    """Real world-1 parallel state; DDP and the optimizer both derive their groups from it."""
    from megatron.core import parallel_state

    init_model_parallel()
    yield
    parallel_state.destroy_model_parallel()


class _Block(torch.nn.Module):
    """Two linears named like the aux MLP's, so the ParamKey glob is exercised.

    ``bias`` is a knob because a 1-D parameter picks up the standard provider's
    ``wd_mult=0.0`` override and therefore lands in a SEPARATE group — which is how a real
    model ends up with more than one core group.
    """

    def __init__(self, bias=False):
        super().__init__()
        self.linear_fc1 = torch.nn.Linear(HIDDEN, HIDDEN, bias=bias)
        self.linear_fc2 = torch.nn.Linear(HIDDEN, HIDDEN, bias=bias)

    def forward(self, x):
        return self.linear_fc2(torch.relu(self.linear_fc1(x)))


def _tiny_model(config, with_aux=True, core_bias=False, aux_bias=False):
    """A MegatronModule whose aux parameters are nested one level deep.

    The nesting matters: the override's ParamKey is ``*.gr_aux.*`` and the provider's count
    looks for ``.gr_aux.``, so both need a parent module in front — exactly as in a real
    model, where the aux lives at ``decoder.layers.N.mlp.gr_aux``.

    Core and aux biases are SEPARATE knobs because the real model is asymmetric: the rest
    of the model has plenty of 1-D parameters, while ``GRAMAuxMLP`` refuses
    ``add_bias_linear`` outright, so every aux parameter is 2-D. ``aux_bias=True`` is
    therefore an illegal shape, used by exactly one test to pin what goes wrong.
    """
    from megatron.core.transformer.module import MegatronModule

    class Tiny(MegatronModule):
        def __init__(self):
            super().__init__(config=config)
            self.mlp = torch.nn.Module()
            self.mlp.core = _Block(bias=core_bias)
            if with_aux:
                self.mlp.gr_aux = _Block(bias=aux_bias)

        def forward(self, x):
            out = self.mlp.core(x)
            return out + self.mlp.gr_aux(x) if with_aux else out

    return Tiny()


def _wrapped_model(with_aux=True, seed=1234, use_distributed_optimizer=False, core_bias=False, aux_bias=False):
    """DDP-wrap the tiny model. ``use_distributed_optimizer`` must match the optimizer's
    setting: the distributed optimizer writes back through DDP's param buffer, which DDP
    only allocates when it is told the same thing."""
    from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
    from megatron.core.transformer.transformer_config import TransformerConfig

    torch.manual_seed(seed)
    config = TransformerConfig(
        num_layers=1, hidden_size=HIDDEN, num_attention_heads=1, bf16=False, params_dtype=torch.float32
    )
    ddp_config = DistributedDataParallelConfig(
        grad_reduce_in_fp32=True,
        overlap_grad_reduce=False,
        use_distributed_optimizer=use_distributed_optimizer,
    )
    model = _tiny_model(config, with_aux=with_aux, core_bias=core_bias, aux_bias=aux_bias)
    return DistributedDataParallel(config, ddp_config, model.cuda())


def _build_optimizer(model, use_distributed_optimizer=False, provider=None):
    from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer

    from megatron.bridge.training.config import OptimizerConfigOverrideProviderContext, SchedulerConfig
    from megatron.bridge.training.gradient_routing.optimizer_gating import GROptimizerConfigOverrideProvider

    provider = provider or GROptimizerConfigOverrideProvider(
        aux_lr=AUX_LR, aux_min_lr=AUX_MIN_LR, aux_wd_mult=AUX_WD_MULT
    )
    optimizer_config = OptimizerConfig(
        optimizer="adam",
        lr=BASE_LR,
        min_lr=BASE_MIN_LR,
        weight_decay=0.1,
        clip_grad=0.0,
        use_distributed_optimizer=use_distributed_optimizer,
    )
    overrides = provider.build_config_overrides(
        OptimizerConfigOverrideProviderContext(
            scheduler_config=SchedulerConfig(lr_decay_iters=10, lr_decay_style="constant"),
            optimizer_config=optimizer_config,
            model=[model],
        )
    )
    return get_megatron_optimizer(optimizer_config, [model], config_overrides=overrides)


def _rig(use_distributed_optimizer=False):
    """A model, its real optimizer, and a discovered gater."""
    from megatron.bridge.training.gradient_routing.optimizer_gating import GROptimizerGater

    model = _wrapped_model(use_distributed_optimizer=use_distributed_optimizer)
    optimizer = _build_optimizer(model, use_distributed_optimizer=use_distributed_optimizer)
    gater = GROptimizerGater()
    gater.discover(optimizer)
    return model, optimizer, gater


def _params(model):
    """Model parameters keyed by their UNWRAPPED name (DDP prefixes everything with ``module.``)."""
    return {name.removeprefix("module."): p for name, p in model.named_parameters()}


def _snapshot_params(model):
    return {name: p.detach().clone() for name, p in _params(model).items()}


def _snapshot_adam_state(optimizer):
    """Adam moments and step counts, keyed by (group index, position in group).

    Keyed positionally rather than by parameter object because the distributed optimizer's
    inner groups hold shard tensors, not the model's parameters.
    """
    from megatron.bridge.training.gradient_routing.optimizer_gating import _iter_inner_param_groups

    snapshot = {}
    for group_index, group in enumerate(_iter_inner_param_groups(optimizer)):
        inner = optimizer.chained_optimizers if hasattr(optimizer, "chained_optimizers") else [optimizer]
        state = {}
        for wrapper in inner:
            state.update(wrapper.optimizer.state)
        for position, param in enumerate(group["params"]):
            entry = state.get(param)
            if entry is None:
                continue
            snapshot[(group_index, position)] = {
                key: (value.detach().clone() if torch.is_tensor(value) else value) for key, value in entry.items()
            }
    return snapshot


def _groups(optimizer):
    """Every inner param group, in order."""
    from megatron.bridge.training.gradient_routing.optimizer_gating import _iter_inner_param_groups

    return list(_iter_inner_param_groups(optimizer))


def _is_aux(group) -> bool:
    from megatron.bridge.training.gradient_routing.optimizer_gating import GR_ROLE_AUX, GR_ROLE_KEY

    return group.get(GR_ROLE_KEY) == GR_ROLE_AUX


def _marked_aux(optimizer, aux: bool) -> set[int]:
    """Positions of the groups carrying (or not carrying) the aux role marker."""
    return {i for i, group in enumerate(_groups(optimizer)) if _is_aux(group) is aux}


def _emptied_by(gater, optimizer, update_core: bool, update_aux: bool) -> set[int]:
    """Arm, record which group positions were emptied, restore. The classification, observed."""
    gater.arm(update_core=update_core, update_aux=update_aux)
    emptied = {i for i, group in enumerate(_groups(optimizer)) if not group["params"]}
    gater.restore()
    return emptied


def _saved_param_groups(state_dict):
    """The param-group list inside a Megatron optimizer state dict.

    The nesting depends on the wrapper: a single-chunk ChainedOptimizer forwards the inner
    optimizer's dict verbatim, while the float16 wrappers nest theirs under 'optimizer'.
    """
    inner = state_dict.get("optimizer", state_dict)
    return inner["param_groups"]


def _step(model, optimizer, gater, update_core=True, update_aux=True, seed=0):
    """One full training step, gated exactly as ``make_gr_finalize`` gates a real one."""
    optimizer.zero_grad()
    torch.manual_seed(seed)
    x = torch.randn(4, HIDDEN, device="cuda")
    model(x).square().sum().backward()
    model.finish_grad_sync()
    gater.arm(update_core=update_core, update_aux=update_aux)
    result = optimizer.step()
    gater.restore()
    return result


@requires_gpu
@pytest.mark.usefixtures("model_parallel")
class TestOverrideProvider:
    """The aux group must exist, be distinct, and carry the role marker the gater reads."""

    def test_aux_override_is_added_with_its_own_lr_and_marker(self):
        from megatron.core.optimizer import OptimizerConfig
        from megatron.core.optimizer.optimizer_config import ParamKey

        from megatron.bridge.training.config import OptimizerConfigOverrideProviderContext, SchedulerConfig
        from megatron.bridge.training.gradient_routing.optimizer_gating import (
            GR_AUX_PARAM_PATTERN,
            GR_ROLE_AUX,
            GR_ROLE_KEY,
            GROptimizerConfigOverrideProvider,
        )

        model = _wrapped_model()
        provider = GROptimizerConfigOverrideProvider(aux_lr=AUX_LR, aux_min_lr=AUX_MIN_LR, aux_wd_mult=AUX_WD_MULT)
        overrides = provider.build_config_overrides(
            OptimizerConfigOverrideProviderContext(
                scheduler_config=SchedulerConfig(lr_decay_iters=10, lr_decay_style="constant"),
                optimizer_config=OptimizerConfig(optimizer="adam", lr=BASE_LR, min_lr=BASE_MIN_LR),
                model=[model],
            )
        )
        aux_override = overrides[ParamKey(name=GR_AUX_PARAM_PATTERN)]
        assert aux_override["max_lr"] == AUX_LR
        assert aux_override["min_lr"] == AUX_MIN_LR
        assert aux_override["wd_mult"] == AUX_WD_MULT
        assert aux_override[GR_ROLE_KEY] == GR_ROLE_AUX

    def test_optimizer_places_aux_params_in_their_own_group(self):
        from megatron.bridge.training.gradient_routing.optimizer_gating import (
            GR_ROLE_AUX,
            GR_ROLE_KEY,
            _iter_inner_param_groups,
        )

        model = _wrapped_model()
        optimizer = _build_optimizer(model)
        params = _params(model)

        groups = list(_iter_inner_param_groups(optimizer))
        aux_groups = [g for g in groups if g.get(GR_ROLE_KEY) == GR_ROLE_AUX]
        core_groups = [g for g in groups if g.get(GR_ROLE_KEY) != GR_ROLE_AUX]
        assert len(aux_groups) == 1 and len(core_groups) == 1

        aux_ids = {id(p) for g in aux_groups for p in g["params"]}
        core_ids = {id(p) for g in core_groups for p in g["params"]}
        assert aux_ids == {id(params[name]) for name in AUX_PARAMS}
        assert core_ids == {id(params[name]) for name in CORE_PARAMS}
        assert aux_groups[0]["max_lr"] == AUX_LR and aux_groups[0]["min_lr"] == AUX_MIN_LR
        assert aux_groups[0]["wd_mult"] == AUX_WD_MULT
        assert core_groups[0]["max_lr"] == BASE_LR and core_groups[0]["min_lr"] == BASE_MIN_LR

    def test_missing_aux_lr_raises(self):
        from megatron.core.optimizer import OptimizerConfig

        from megatron.bridge.training.config import OptimizerConfigOverrideProviderContext, SchedulerConfig
        from megatron.bridge.training.gradient_routing.optimizer_gating import GROptimizerConfigOverrideProvider

        context = OptimizerConfigOverrideProviderContext(
            scheduler_config=SchedulerConfig(lr_decay_iters=10, lr_decay_style="constant"),
            optimizer_config=OptimizerConfig(optimizer="adam", lr=BASE_LR, min_lr=BASE_MIN_LR),
            model=[_wrapped_model()],
        )
        for kwargs in ({"aux_min_lr": AUX_MIN_LR}, {"aux_lr": AUX_LR}, {}):
            with pytest.raises(ValueError, match="requires explicit aux_lr and aux_min_lr"):
                GROptimizerConfigOverrideProvider(**kwargs).build_config_overrides(context)

    @pytest.mark.parametrize("aux_wd_mult", [1.0, AUX_WD_MULT])
    def test_a_one_dimensional_aux_param_breaks_optimizer_construction(self, aux_wd_mult):
        """Every aux parameter must be 2-D, and this is why.

        A 1-D parameter under ``*.gr_aux.*`` matches BOTH the standard provider's
        no-weight-decay override (``wd_mult=0.0``, applied to biases and 1-D params) and the
        GR aux override's ``wd_mult``. mcore's ``combine_param_group_overrides`` refuses
        conflicting values outright, so ``get_megatron_optimizer`` dies at construction —
        including at the DEFAULT ``aux_wd_mult=1.0``, since 0.0 != 1.0 just as surely.

        Nothing in production reaches this: ``GRAMAuxMLP`` raises on ``add_bias_linear`` and
        holds no norms, so the aux module is all-2-D. But that refusal is documented as
        being about the forget-ON export merge, and it is silently load-bearing HERE too —
        relaxing it to allow a bias or a layernorm would break every GR run's startup, far
        from the code that permitted it.
        """
        from megatron.bridge.training.gradient_routing.optimizer_gating import GROptimizerConfigOverrideProvider

        model = _wrapped_model(aux_bias=True)
        provider = GROptimizerConfigOverrideProvider(aux_lr=AUX_LR, aux_min_lr=AUX_MIN_LR, aux_wd_mult=aux_wd_mult)
        with pytest.raises(ValueError, match="Conflicting overrides for wd_mult"):
            _build_optimizer(model, provider=provider)

    def test_model_without_aux_params_raises(self):
        """The failure mode this catches is a run that trains normally and calls itself GR."""
        from megatron.core.optimizer import OptimizerConfig

        from megatron.bridge.training.config import OptimizerConfigOverrideProviderContext, SchedulerConfig
        from megatron.bridge.training.gradient_routing.optimizer_gating import GROptimizerConfigOverrideProvider

        provider = GROptimizerConfigOverrideProvider(aux_lr=AUX_LR, aux_min_lr=AUX_MIN_LR)
        context = OptimizerConfigOverrideProviderContext(
            scheduler_config=SchedulerConfig(lr_decay_iters=10, lr_decay_style="constant"),
            optimizer_config=OptimizerConfig(optimizer="adam", lr=BASE_LR, min_lr=BASE_MIN_LR),
            model=[_wrapped_model(with_aux=False)],
        )
        with pytest.raises(ValueError, match="no '.gr_aux.' parameters"):
            provider.build_config_overrides(context)


@requires_gpu
@pytest.mark.usefixtures("model_parallel")
class TestDiscovery:
    def test_discover_classifies_the_groups_by_marker(self):
        """Classification is by the ``gr_role`` marker, and the split must be exhaustive.

        Read out behaviourally — which groups ``arm`` empties for each role IS the
        classification — so the assertion survives any refactor of the gater's internals.
        """
        model, optimizer, gater = _rig()
        assert gater.discovered

        params = _params(model)
        assert _emptied_by(gater, optimizer, update_core=False, update_aux=True) == _marked_aux(optimizer, False)
        assert _emptied_by(gater, optimizer, update_core=True, update_aux=False) == _marked_aux(optimizer, True)

        aux_group = next(g for g in _groups(optimizer) if _is_aux(g))
        core_groups = [g for g in _groups(optimizer) if not _is_aux(g)]
        assert {id(p) for p in aux_group["params"]} == {id(params[name]) for name in AUX_PARAMS}
        assert {id(p) for g in core_groups for p in g["params"]} == {id(params[name]) for name in CORE_PARAMS}

    def test_every_non_aux_group_is_frozen_together(self):
        """A real model has SEVERAL core groups — the standard provider splits 1-D params
        into their own no-weight-decay group (the shipped hybrid shows default/aux/no-wd).
        Freezing core must empty all of them, not stop at the first non-aux group.
        """
        from megatron.bridge.training.gradient_routing.optimizer_gating import GROptimizerGater

        model = _wrapped_model(core_bias=True)
        optimizer = _build_optimizer(model)
        gater = GROptimizerGater()
        gater.discover(optimizer)

        groups = _groups(optimizer)
        assert len(groups) >= 3, f"expected default/aux/no-wd groups, got {len(groups)}"
        assert sum(1 for g in groups if not _is_aux(g)) >= 2, "no separate no-weight-decay group was created"

        # every aux parameter, weight and bias alike, is on the aux side
        aux_ids = {id(p) for name, p in _params(model).items() if ".gr_aux." in name}
        assert {id(p) for g in groups if _is_aux(g) for p in g["params"]} == aux_ids

        assert _emptied_by(gater, optimizer, update_core=False, update_aux=True) == _marked_aux(optimizer, False)
        assert all(g["params"] for g in groups), "restore missed a group"

    def test_discovered_is_false_before_discovery(self):
        from megatron.bridge.training.gradient_routing.optimizer_gating import GROptimizerGater

        assert GROptimizerGater().discovered is False

    def test_discover_without_the_gr_provider_raises(self):
        """A standard provider produces no role-marked group; the gater must say so rather
        than silently treat every group as core and freeze nothing."""
        from megatron.bridge.training.config import OptimizerConfigOverrideProvider
        from megatron.bridge.training.gradient_routing.optimizer_gating import GROptimizerGater

        model = _wrapped_model()
        optimizer = _build_optimizer(model, provider=OptimizerConfigOverrideProvider())
        with pytest.raises(RuntimeError, match="no aux-marked param group"):
            GROptimizerGater().discover(optimizer)

    def test_an_aux_group_with_no_local_shards_is_accepted(self):
        """Discovery is a STRUCTURAL check: group structure is rank-uniform, but shard
        ownership is not, so under the distributed optimizer a rank whose data-parallel
        shard misses the aux params holds a legitimately empty aux group. World size 1
        cannot produce that split, so the empty group is staged directly — what is being
        pinned is that emptiness alone is not treated as a wiring failure."""
        from megatron.bridge.training.gradient_routing.optimizer_gating import (
            GR_ROLE_AUX,
            GR_ROLE_KEY,
            GROptimizerGater,
            _iter_inner_param_groups,
        )

        model = _wrapped_model()
        optimizer = _build_optimizer(model)
        for group in _iter_inner_param_groups(optimizer):
            if group.get(GR_ROLE_KEY) == GR_ROLE_AUX:
                group["params"] = []

        gater = GROptimizerGater()
        gater.discover(optimizer)
        assert gater.discovered
        gater.arm(update_core=True, update_aux=False)  # emptying an empty group is a no-op
        gater.restore()


@requires_gpu
@pytest.mark.usefixtures("model_parallel")
class TestFreezing:
    """The core claim: an emptied group's parameters AND Adam state are untouched."""

    def _warm(self, model, optimizer, gater):
        """Two unrestricted steps, so every parameter has non-trivial Adam state to protect."""
        for seed in (0, 1):
            _step(model, optimizer, gater, update_core=True, update_aux=True, seed=seed)

    @pytest.mark.parametrize(
        "update_core, update_aux, frozen_names, moving_names",
        [
            (False, True, CORE_PARAMS, AUX_PARAMS),
            (True, False, AUX_PARAMS, CORE_PARAMS),
        ],
    )
    def test_frozen_role_parameters_do_not_move(self, update_core, update_aux, frozen_names, moving_names):
        model, optimizer, gater = _rig()
        self._warm(model, optimizer, gater)

        before = _snapshot_params(model)
        _step(model, optimizer, gater, update_core=update_core, update_aux=update_aux, seed=2)
        after = _params(model)

        for name in frozen_names:
            assert torch.equal(before[name], after[name]), f"frozen parameter {name} moved"
        for name in moving_names:
            assert not torch.equal(before[name], after[name]), f"updating parameter {name} did not move"

    @pytest.mark.parametrize("update_core, update_aux", [(False, True), (True, False)])
    def test_frozen_role_adam_state_is_untouched(self, update_core, update_aux):
        """lr=0 or zeroed grads would still move exp_avg / exp_avg_sq / step here."""
        model, optimizer, gater = _rig()
        self._warm(model, optimizer, gater)

        from megatron.bridge.training.gradient_routing.optimizer_gating import (
            GR_ROLE_AUX,
            GR_ROLE_KEY,
            _iter_inner_param_groups,
        )

        frozen_role_is_core = not update_core
        frozen_group_indices = {
            index
            for index, group in enumerate(_iter_inner_param_groups(optimizer))
            if (group.get(GR_ROLE_KEY) != GR_ROLE_AUX) == frozen_role_is_core
        }
        before = _snapshot_adam_state(optimizer)
        _step(model, optimizer, gater, update_core=update_core, update_aux=update_aux, seed=3)
        after = _snapshot_adam_state(optimizer)

        assert before, "no Adam state was captured — the warm-up steps did not run"
        frozen_keys = [key for key in before if key[0] in frozen_group_indices]
        moving_keys = [key for key in before if key[0] not in frozen_group_indices]
        assert frozen_keys and moving_keys

        for key in frozen_keys:
            for state_name, value in before[key].items():
                new = after[key][state_name]
                if torch.is_tensor(value):
                    assert torch.equal(value, new), f"frozen {state_name} at {key} changed"
                else:
                    assert value == new, f"frozen {state_name} at {key} changed: {value} -> {new}"
        assert any(
            not torch.equal(before[key]["exp_avg"], after[key]["exp_avg"])
            for key in moving_keys
            if "exp_avg" in before[key]
        ), "no updating group moved — the comparison proves nothing"

    def test_unfrozen_step_moves_everything(self):
        """The control arm: with both roles updating, nothing is held back."""
        model, optimizer, gater = _rig()
        before = _snapshot_params(model)
        _step(model, optimizer, gater, update_core=True, update_aux=True)
        after = _params(model)
        for name in CORE_PARAMS + AUX_PARAMS:
            assert not torch.equal(before[name], after[name]), f"{name} did not move on an unrestricted step"

    def test_distributed_optimizer_freezes_the_same_way(self):
        """The distributed optimizer's inner groups hold shards, but the role marker and the
        emptying survive — the shipped configs use it."""
        model, optimizer, gater = _rig(use_distributed_optimizer=True)
        _step(model, optimizer, gater, update_core=True, update_aux=True, seed=0)

        before = _snapshot_params(model)
        _step(model, optimizer, gater, update_core=False, update_aux=True, seed=2)
        after = _params(model)
        for name in CORE_PARAMS:
            assert torch.equal(before[name], after[name]), f"frozen {name} moved under the distributed optimizer"
        for name in AUX_PARAMS:
            assert not torch.equal(before[name], after[name]), f"{name} did not move under the distributed optimizer"


@requires_gpu
@pytest.mark.usefixtures("model_parallel")
class TestArmRestoreStateMachine:
    def test_arm_empties_only_the_frozen_role(self):
        model, optimizer, gater = _rig()
        from megatron.bridge.training.gradient_routing.optimizer_gating import (
            GR_ROLE_AUX,
            GR_ROLE_KEY,
            _iter_inner_param_groups,
        )

        gater.arm(update_core=False, update_aux=True)
        for group in _iter_inner_param_groups(optimizer):
            if group.get(GR_ROLE_KEY) == GR_ROLE_AUX:
                assert group["params"], "aux group was emptied while it was the updating role"
            else:
                assert group["params"] == [], "core group was not emptied"
        gater.restore()

    def test_restore_puts_back_the_identical_param_lists(self):
        model, optimizer, gater = _rig()
        from megatron.bridge.training.gradient_routing.optimizer_gating import _iter_inner_param_groups

        before = [list(group["params"]) for group in _iter_inner_param_groups(optimizer)]
        gater.arm(update_core=False, update_aux=True)
        gater.restore()
        after = [list(group["params"]) for group in _iter_inner_param_groups(optimizer)]
        assert len(before) == len(after)
        for before_group, after_group in zip(before, after):
            assert len(before_group) == len(after_group)
            assert all(b is a for b, a in zip(before_group, after_group)), "restore swapped parameter identities"

    def test_restore_is_a_noop_when_nothing_is_armed(self):
        _, optimizer, gater = _rig()
        from megatron.bridge.training.gradient_routing.optimizer_gating import _iter_inner_param_groups

        before = [list(group["params"]) for group in _iter_inner_param_groups(optimizer)]
        gater.restore()
        gater.restore()
        after = [list(group["params"]) for group in _iter_inner_param_groups(optimizer)]
        assert before == after

    def test_arm_is_idempotent_for_the_same_roles(self):
        """The rerun state machine can drive grad finalization more than once per step."""
        _, optimizer, gater = _rig()
        from megatron.bridge.training.gradient_routing.optimizer_gating import _iter_inner_param_groups

        gater.arm(update_core=False, update_aux=True)
        armed = [list(group["params"]) for group in _iter_inner_param_groups(optimizer)]
        gater.arm(update_core=False, update_aux=True)
        assert [list(group["params"]) for group in _iter_inner_param_groups(optimizer)] == armed
        gater.restore()
        assert all(group["params"] for group in _iter_inner_param_groups(optimizer))

    def test_conflicting_rearm_without_restore_raises(self):
        """Re-arming over a live stash would lose the stashed params permanently."""
        _, _, gater = _rig()
        gater.arm(update_core=False, update_aux=True)
        with pytest.raises(RuntimeError, match="re-armed with"):
            gater.arm(update_core=True, update_aux=False)
        gater.restore()

    def test_arming_a_step_that_updates_nothing_raises(self):
        _, _, gater = _rig()
        with pytest.raises(RuntimeError, match="update neither core nor aux"):
            gater.arm(update_core=False, update_aux=False)

    def test_arming_both_roles_stashes_nothing(self):
        _, optimizer, gater = _rig()
        from megatron.bridge.training.gradient_routing.optimizer_gating import _iter_inner_param_groups

        gater.arm(update_core=True, update_aux=True)
        assert all(group["params"] for group in _iter_inner_param_groups(optimizer))
        gater.restore()

    def test_state_dict_round_trips_after_restore(self):
        """Checkpointing happens between steps, i.e. after restore; the emptying must leave
        no trace in the saved optimizer state."""
        model, optimizer, gater = _rig()
        _step(model, optimizer, gater, update_core=True, update_aux=True, seed=0)
        _step(model, optimizer, gater, update_core=False, update_aux=True, seed=1)

        state_dict = optimizer.state_dict()
        group_sizes = [len(g["params"]) for g in _saved_param_groups(state_dict)]
        assert all(size > 0 for size in group_sizes), f"an emptied group leaked into the state dict: {group_sizes}"

        optimizer.load_state_dict(state_dict)
        reloaded = [len(g["params"]) for g in _saved_param_groups(optimizer.state_dict())]
        assert reloaded == group_sizes


@requires_gpu
@pytest.mark.usefixtures("model_parallel")
class TestFinalizeWrapper:
    """``install_gr_finalize`` is what connects the plan to the gater during a real run.

    The wrapper is a MODULE-LEVEL function fed from a module-level runtime slot rather than
    a closure, and that is not a style choice: the model config's
    ``finalize_model_grads_func`` is serialized BY IMPORT PATH into the checkpoint's
    ``run_config.yaml``, and a closure's ``<locals>`` qualname cannot be re-imported — which
    broke ``AutoBridge.from_auto_config`` at export time. So the importability of the
    returned function is pinned here alongside its behaviour.
    """

    @pytest.fixture(autouse=True)
    def _isolate_runtime_slot(self):
        """Save/restore the module-level runtime slot so these tests cannot leak into others."""
        from megatron.bridge.training.gradient_routing import optimizer_gating

        saved = optimizer_gating._GR_RUNTIME
        yield
        optimizer_gating._GR_RUNTIME = saved

    def _plan(self, update_core, update_aux):
        import numpy as np

        from megatron.bridge.training.gradient_routing.plan import GRPlan

        n = len(update_core)
        return GRPlan(
            corpus=np.zeros(n, dtype=np.int64),
            fwd_aux=np.asarray(update_aux, dtype=np.int64),
            update_core=np.asarray(update_core, dtype=np.int64),
            update_aux=np.asarray(update_aux, dtype=np.int64),
            prior_iters_same_corpus=np.arange(n, dtype=np.int64),
            plan_seed=0,
            p_as=0.0,
            p_cr=0.0,
            forget_iter_fraction=0.0,
        )

    def _install(self, base_finalize, update_core, update_aux, step=0):
        from megatron.bridge.training.gradient_routing.optimizer_gating import GROptimizerGater, install_gr_finalize

        model = _wrapped_model()
        optimizer = _build_optimizer(model)
        gater = GROptimizerGater()
        state = SimpleNamespace(train_state=SimpleNamespace(step=step))
        finalize = install_gr_finalize(base_finalize, gater, optimizer, self._plan(update_core, update_aux), state)
        return finalize, optimizer, gater, state

    def test_wrapper_runs_the_base_finalize_then_arms_from_the_plan(self):
        calls = []
        finalize, optimizer, gater, state = self._install(
            lambda *a, **k: calls.append((a, k)) or "base-result",
            update_core=[0, 1],
            update_aux=[1, 1],
        )

        assert finalize("model", "config") == "base-result"
        assert calls == [(("model", "config"), {})], "the base finalize must run, and run first"
        assert gater.discovered, "discovery must happen lazily on the first finalize"
        # iteration 0 does not update core -> the core groups are emptied
        assert [bool(g["params"]) for g in _groups(optimizer)] == [False, True]
        gater.restore()

        state.train_state.step = 1
        finalize()
        assert all(g["params"] for g in _groups(optimizer)), "iteration 1 updates both roles"
        gater.restore()

    def test_wrapper_discovers_only_once(self):
        discoveries = []
        finalize, _, gater, state = self._install(lambda: None, update_core=[1, 1], update_aux=[1, 1])
        original_discover = gater.discover

        def counting_discover(opt):
            discoveries.append(opt)
            original_discover(opt)

        gater.discover = counting_discover
        finalize()
        gater.restore()
        state.train_state.step = 1
        finalize()
        gater.restore()
        assert len(discoveries) == 1

    def test_install_returns_the_importable_module_level_function(self):
        """The whole reason this is not a closure: the checkpoint's run_config.yaml stores
        this callable by import path, and export re-imports it."""
        import importlib

        from megatron.bridge.training.gradient_routing.optimizer_gating import gr_finalize_model_grads

        finalize, _, _, _ = self._install(lambda: None, update_core=[1], update_aux=[1])
        assert finalize is gr_finalize_model_grads
        assert "<locals>" not in finalize.__qualname__, "a closure qualname cannot be re-imported"
        reimported = getattr(importlib.import_module(finalize.__module__), finalize.__qualname__)
        assert reimported is finalize

    def test_calling_the_wrapper_without_installing_raises(self):
        """A run_config that names this function outside a GR run must fail loudly rather
        than silently skip grad finalization."""
        from megatron.bridge.training.gradient_routing import optimizer_gating

        optimizer_gating._GR_RUNTIME = None
        with pytest.raises(RuntimeError, match="without install_gr_finalize"):
            optimizer_gating.gr_finalize_model_grads()

    def test_reinstalling_replaces_the_runtime(self):
        """ft_launcher can restart a run inside one process; the second install must win."""
        from megatron.bridge.training.gradient_routing import optimizer_gating

        first_calls, second_calls = [], []
        self._install(lambda: first_calls.append(1), update_core=[1], update_aux=[1])
        finalize, _, gater, _ = self._install(lambda: second_calls.append(1), update_core=[1], update_aux=[1])

        finalize()
        gater.restore()
        assert (first_calls, second_calls) == ([], [1]), "the stale runtime was still in use"
        assert optimizer_gating._GR_RUNTIME["gater"] is gater
