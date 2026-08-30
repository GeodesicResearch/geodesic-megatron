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
tiny (two 8x8 blocks per role) because the assertion is about which tensors the optimizer
touches, not about what they contain.

At N > 1 each module is its own role, and the claim gets sharper: stepping module 0 must
leave module 1's parameters AND moments bitwise unchanged. That separation rests on mcore
grouping by merged-override EQUALITY, so the per-module role marker is what splits the
groups — the tests below give the two modules identical LR/WD in one case precisely to pin
that the marker alone is enough.
"""

from types import SimpleNamespace

import pytest
import torch

from tests.unit_tests.gr_test_utils import init_model_parallel, teardown_model_parallel


requires_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU (real Megatron optimizer)")

HIDDEN = 8
CORE_PARAMS = ("mlp.core.linear_fc1.weight", "mlp.core.linear_fc2.weight")
AUX_LRS, AUX_MIN_LRS, AUX_WD_MULTS = [1e-2, 3e-2], [1e-3, 3e-3], [0.5, 0.25]
BASE_LR, BASE_MIN_LR = 1e-3, 1e-4


def aux_params(module: int) -> tuple[str, ...]:
    """The parameter names of one aux module, as they appear on the unwrapped model."""
    return (f"mlp.gr_aux.{module}.linear_fc1.weight", f"mlp.gr_aux.{module}.linear_fc2.weight")


AUX_PARAMS = aux_params(0)


@pytest.fixture(scope="module")
def model_parallel():
    """Real world-1 parallel state; DDP and the optimizer both derive their groups from it."""
    init_model_parallel()
    yield
    teardown_model_parallel()


class _Block(torch.nn.Module):
    """Two linears named like an aux MLP's, so the ParamKey glob is exercised.

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


def _tiny_model(config, n_aux=1, core_bias=False, aux_bias=False):
    """A MegatronModule whose aux parameters sit in an indexed ModuleList one level deep.

    The nesting and the index both matter: the override's ParamKey is ``*.gr_aux.<k>.*`` and
    the provider's count looks for ``.gr_aux.<k>.``, so both need a parent module in front —
    exactly as in a real model, where module k lives at ``decoder.layers.N.mlp.gr_aux.k``.

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
            self.mlp.gr_aux = torch.nn.ModuleList([_Block(bias=aux_bias) for _ in range(n_aux)])

        def forward(self, x):
            out = self.mlp.core(x)
            for aux in self.mlp.gr_aux:
                out = out + aux(x)
            return out

    return Tiny()


def _wrapped_model(n_aux=1, seed=1234, use_distributed_optimizer=False, core_bias=False, aux_bias=False):
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
    model = _tiny_model(config, n_aux=n_aux, core_bias=core_bias, aux_bias=aux_bias)
    return DistributedDataParallel(config, ddp_config, model.cuda())


def _gr_provider(n_aux=1, lrs=None, min_lrs=None, wd_mults=None):
    from megatron.bridge.training.gradient_routing.optimizer_gating import GROptimizerConfigOverrideProvider

    return GROptimizerConfigOverrideProvider(
        aux_lrs=list(lrs if lrs is not None else AUX_LRS[:n_aux]),
        aux_min_lrs=list(min_lrs if min_lrs is not None else AUX_MIN_LRS[:n_aux]),
        aux_wd_mults=list(wd_mults if wd_mults is not None else AUX_WD_MULTS[:n_aux]),
    )


def _build_optimizer(model, use_distributed_optimizer=False, provider=None, n_aux=1):
    from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer

    from megatron.bridge.training.config import OptimizerConfigOverrideProviderContext, SchedulerConfig

    provider = provider if provider is not None else _gr_provider(n_aux)
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


def _rig(use_distributed_optimizer=False, n_aux=1, **provider_kwargs):
    """A model, its real optimizer, and a discovered gater."""
    from megatron.bridge.training.gradient_routing.optimizer_gating import GROptimizerGater

    model = _wrapped_model(n_aux=n_aux, use_distributed_optimizer=use_distributed_optimizer)
    optimizer = _build_optimizer(
        model,
        use_distributed_optimizer=use_distributed_optimizer,
        provider=_gr_provider(n_aux, **provider_kwargs) if provider_kwargs else None,
        n_aux=n_aux,
    )
    gater = GROptimizerGater(n_aux=n_aux)
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


def _aux_index(group) -> int | None:
    """The aux module index a group belongs to, or None for a core group."""
    from megatron.bridge.training.gradient_routing.optimizer_gating import GR_ROLE_KEY, _parse_aux_role

    return _parse_aux_role(group.get(GR_ROLE_KEY))


def _is_aux(group) -> bool:
    return _aux_index(group) is not None


def _group_positions(optimizer, module=None) -> set[int]:
    """Positions of the groups belonging to one aux module, or of every core group."""
    return {i for i, group in enumerate(_groups(optimizer)) if _aux_index(group) == module}


def _emptied_by(gater, optimizer, update_core: bool, update_aux) -> set[int]:
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


def _step(model, optimizer, gater, update_core=True, update_aux=(True,), seed=0):
    """One full training step, gated exactly as the finalize wrapper gates a real one."""
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
    """Each aux module must get its own group, distinct, carrying the marker the gater reads."""

    def _overrides(self, model, provider):
        from megatron.core.optimizer import OptimizerConfig

        from megatron.bridge.training.config import OptimizerConfigOverrideProviderContext, SchedulerConfig

        return provider.build_config_overrides(
            OptimizerConfigOverrideProviderContext(
                scheduler_config=SchedulerConfig(lr_decay_iters=10, lr_decay_style="constant"),
                optimizer_config=OptimizerConfig(optimizer="adam", lr=BASE_LR, min_lr=BASE_MIN_LR),
                model=[model],
            )
        )

    @pytest.mark.parametrize("n_aux", [1, 2])
    def test_one_override_per_module_with_its_own_lr_and_marker(self, n_aux):
        from megatron.core.optimizer.optimizer_config import ParamKey

        from megatron.bridge.training.gradient_routing.optimizer_gating import (
            GR_ROLE_KEY,
            gr_aux_param_pattern,
            gr_aux_role,
        )

        overrides = self._overrides(_wrapped_model(n_aux=n_aux), _gr_provider(n_aux))
        for k in range(n_aux):
            override = overrides[ParamKey(name=gr_aux_param_pattern(k))]
            assert override["max_lr"] == AUX_LRS[k]
            assert override["min_lr"] == AUX_MIN_LRS[k]
            assert override["wd_mult"] == AUX_WD_MULTS[k]
            assert override[GR_ROLE_KEY] == gr_aux_role(k)

    def test_the_per_module_patterns_are_disjoint(self):
        """``*.gr_aux.0.*`` must not capture module 1's parameters: overlapping patterns would
        make which override a parameter gets depend on the matcher's precedence rules."""
        from fnmatch import fnmatch

        from megatron.bridge.training.gradient_routing.optimizer_gating import gr_aux_param_pattern

        names = [f"decoder.layers.3.mlp.gr_aux.{k}.linear_fc1.weight" for k in range(2)]
        for k, name in enumerate(names):
            matched = [j for j in range(2) if fnmatch(name, gr_aux_param_pattern(j))]
            assert matched == [k], f"{name} matched modules {matched}"

    @pytest.mark.parametrize("n_aux", [1, 2])
    def test_optimizer_places_each_modules_params_in_its_own_group(self, n_aux):
        model = _wrapped_model(n_aux=n_aux)
        optimizer = _build_optimizer(model, n_aux=n_aux)
        params = _params(model)

        groups = _groups(optimizer)
        core_groups = [g for g in groups if not _is_aux(g)]
        assert len(core_groups) == 1
        assert len([g for g in groups if _is_aux(g)]) == n_aux, "modules did not get one group each"

        for k in range(n_aux):
            aux_groups = [g for g in groups if _aux_index(g) == k]
            assert len(aux_groups) == 1
            assert {id(p) for p in aux_groups[0]["params"]} == {id(params[name]) for name in aux_params(k)}
            assert aux_groups[0]["max_lr"] == AUX_LRS[k] and aux_groups[0]["min_lr"] == AUX_MIN_LRS[k]
            assert aux_groups[0]["wd_mult"] == AUX_WD_MULTS[k]
        assert {id(p) for g in core_groups for p in g["params"]} == {id(params[name]) for name in CORE_PARAMS}
        assert core_groups[0]["max_lr"] == BASE_LR and core_groups[0]["min_lr"] == BASE_MIN_LR

    def test_modules_with_identical_hyperparameters_still_get_separate_groups(self):
        """mcore groups by merged-override EQUALITY, so two modules configured identically
        would land in ONE group — and freezing either would then freeze both. The role marker
        is what keeps them apart, and it is the only thing that does."""
        model = _wrapped_model(n_aux=2)
        provider = _gr_provider(2, lrs=[1e-2, 1e-2], min_lrs=[1e-3, 1e-3], wd_mults=[0.5, 0.5])
        optimizer = _build_optimizer(model, provider=provider)

        aux_groups = {_aux_index(g): g for g in _groups(optimizer) if _is_aux(g)}
        assert set(aux_groups) == {0, 1}, "identically-configured modules were merged into one group"
        params = _params(model)
        for k in (0, 1):
            assert {id(p) for p in aux_groups[k]["params"]} == {id(params[name]) for name in aux_params(k)}

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"aux_min_lrs": [1e-3], "aux_wd_mults": [1.0]},  # aux_lrs unset
            {"aux_lrs": [1e-2], "aux_wd_mults": [1.0]},  # aux_min_lrs unset
            {"aux_lrs": [1e-2], "aux_min_lrs": [1e-3]},  # aux_wd_mults unset
            {},  # nothing set at all
            {"aux_lrs": [1e-2, 1e-2], "aux_min_lrs": [1e-3], "aux_wd_mults": [1.0, 1.0]},  # ragged
        ],
    )
    def test_missing_or_ragged_per_module_lists_raise(self, kwargs):
        """The three lists are indexed together, one entry per module; a short list would
        either build fewer groups than there are modules or index out of range."""
        from megatron.bridge.training.gradient_routing.optimizer_gating import GROptimizerConfigOverrideProvider

        with pytest.raises(ValueError, match="equal-length non-empty"):
            self._overrides(_wrapped_model(), GROptimizerConfigOverrideProvider(**kwargs))

    @pytest.mark.parametrize("aux_wd_mults", [[1.0], [0.5]])
    def test_a_one_dimensional_aux_param_breaks_optimizer_construction(self, aux_wd_mults):
        """Every aux parameter must be 2-D, and this is why.

        A 1-D parameter under ``*.gr_aux.0.*`` matches BOTH the standard provider's
        no-weight-decay override (``wd_mult=0.0``, applied to biases and 1-D params) and the
        GR aux override's ``wd_mult``. mcore's ``combine_param_group_overrides`` refuses
        conflicting values outright, so ``get_megatron_optimizer`` dies at construction —
        including at the DEFAULT ``wd_mult=1.0``, since 0.0 != 1.0 just as surely.

        Nothing in production reaches this: ``GRAMAuxMLP`` raises on ``add_bias_linear`` and
        holds no norms, so the aux modules are all-2-D. But that refusal is documented as
        being about the export merge, and it is silently load-bearing HERE too — relaxing it
        to allow a bias or a layernorm would break every GR run's startup, far from the code
        that permitted it.
        """
        model = _wrapped_model(aux_bias=True)
        provider = _gr_provider(1, wd_mults=aux_wd_mults)
        with pytest.raises(ValueError, match="Conflicting overrides for wd_mult"):
            _build_optimizer(model, provider=provider)

    @pytest.mark.parametrize("configured, built", [(1, 0), (2, 1)])
    def test_a_module_the_model_does_not_carry_raises(self, configured, built):
        """The failure mode this catches is a run that trains normally and calls itself GR —
        or, at N>1, one that silently routes a corpus into a module that was never built.
        Which module is missing is named, because with N of them "some module" is not
        actionable."""
        model = _wrapped_model(n_aux=built) if built else _tiny_model_without_aux()
        with pytest.raises(ValueError, match=rf"aux module {built} but the model has no '\.gr_aux\.{built}\.'"):
            self._overrides(model, _gr_provider(configured))


def _tiny_model_without_aux():
    """A DDP-wrapped model with a core block and no aux modules at all."""
    from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
    from megatron.core.transformer.module import MegatronModule
    from megatron.core.transformer.transformer_config import TransformerConfig

    torch.manual_seed(1234)
    config = TransformerConfig(
        num_layers=1, hidden_size=HIDDEN, num_attention_heads=1, bf16=False, params_dtype=torch.float32
    )

    class Tiny(MegatronModule):
        def __init__(self):
            super().__init__(config=config)
            self.mlp = torch.nn.Module()
            self.mlp.core = _Block()

        def forward(self, x):
            return self.mlp.core(x)

    ddp_config = DistributedDataParallelConfig(grad_reduce_in_fp32=True, overlap_grad_reduce=False)
    return DistributedDataParallel(config, ddp_config, Tiny().cuda())


@requires_gpu
@pytest.mark.usefixtures("model_parallel")
class TestDiscovery:
    @pytest.mark.parametrize("n_aux", [1, 2])
    def test_discover_classifies_the_groups_by_marker(self, n_aux):
        """Classification is by the ``gr_role`` marker, and the split must be exhaustive.

        Read out behaviourally — which groups ``arm`` empties for each role IS the
        classification — so the assertion survives any refactor of the gater's internals.
        """
        model, optimizer, gater = _rig(n_aux=n_aux)
        assert gater.discovered

        params = _params(model)
        every_aux = [True] * n_aux
        assert _emptied_by(gater, optimizer, False, every_aux) == _group_positions(optimizer, module=None)
        assert _emptied_by(gater, optimizer, True, [False] * n_aux) == {
            position for k in range(n_aux) for position in _group_positions(optimizer, module=k)
        }
        for k in range(n_aux):
            frozen = list(every_aux)
            frozen[k] = False
            assert _emptied_by(gater, optimizer, True, frozen) == _group_positions(optimizer, module=k)

        core_groups = [g for g in _groups(optimizer) if not _is_aux(g)]
        assert {id(p) for g in core_groups for p in g["params"]} == {id(params[name]) for name in CORE_PARAMS}
        for k in range(n_aux):
            group = next(g for g in _groups(optimizer) if _aux_index(g) == k)
            assert {id(p) for p in group["params"]} == {id(params[name]) for name in aux_params(k)}

    def test_every_non_aux_group_is_frozen_together(self):
        """A real model has SEVERAL core groups — the standard provider splits 1-D params
        into their own no-weight-decay group (the shipped hybrid shows default/aux/no-wd).
        Freezing core must empty all of them, not stop at the first non-aux group.
        """
        from megatron.bridge.training.gradient_routing.optimizer_gating import GROptimizerGater

        model = _wrapped_model(core_bias=True)
        optimizer = _build_optimizer(model)
        gater = GROptimizerGater(n_aux=1)
        gater.discover(optimizer)

        groups = _groups(optimizer)
        assert len(groups) >= 3, f"expected default/aux/no-wd groups, got {len(groups)}"
        assert sum(1 for g in groups if not _is_aux(g)) >= 2, "no separate no-weight-decay group was created"

        # every aux parameter, weight and bias alike, is on the aux side
        aux_ids = {id(p) for name, p in _params(model).items() if ".gr_aux." in name}
        assert {id(p) for g in groups if _is_aux(g) for p in g["params"]} == aux_ids

        assert _emptied_by(gater, optimizer, False, [True]) == _group_positions(optimizer, module=None)
        assert all(g["params"] for g in groups), "restore missed a group"

    def test_discovered_is_false_before_discovery(self):
        from megatron.bridge.training.gradient_routing.optimizer_gating import GROptimizerGater

        assert GROptimizerGater(n_aux=1).discovered is False

    @pytest.mark.parametrize("n_aux", [0, -1])
    def test_a_non_positive_module_count_raises(self, n_aux):
        """A gater with no modules could not gate anything; it must not construct."""
        from megatron.bridge.training.gradient_routing.optimizer_gating import GROptimizerGater

        with pytest.raises(ValueError, match="requires n_aux >= 1"):
            GROptimizerGater(n_aux=n_aux)

    def test_discover_without_the_gr_provider_raises(self):
        """A standard provider produces no role-marked group; the gater must say so rather
        than silently treat every group as core and freeze nothing."""
        from megatron.bridge.training.config import OptimizerConfigOverrideProvider
        from megatron.bridge.training.gradient_routing.optimizer_gating import GROptimizerGater

        model = _wrapped_model()
        optimizer = _build_optimizer(model, provider=OptimizerConfigOverrideProvider())
        with pytest.raises(RuntimeError, match=r"no param group for aux module\(s\) \[0\]"):
            GROptimizerGater(n_aux=1).discover(optimizer)

    def test_discover_names_every_module_whose_group_is_absent(self):
        """A gater expecting more modules than the provider installed must name the missing
        ones: at N>1 the difference between "module 1 is missing" and "nothing is wired" is
        the difference between a config typo and a broken launch path."""
        from megatron.bridge.training.gradient_routing.optimizer_gating import GROptimizerGater

        model = _wrapped_model(n_aux=1)
        optimizer = _build_optimizer(model, n_aux=1)
        with pytest.raises(RuntimeError, match=r"no param group for aux module\(s\) \[1\]"):
            GROptimizerGater(n_aux=2).discover(optimizer)

    def test_a_group_marked_for_an_unconfigured_module_raises(self):
        """The mirror image: the provider installed more modules than the gater expects, so
        some module's group would never be frozen by anything."""
        from megatron.bridge.training.gradient_routing.optimizer_gating import GROptimizerGater

        model = _wrapped_model(n_aux=2)
        optimizer = _build_optimizer(model, n_aux=2)
        with pytest.raises(RuntimeError, match="provider and gater disagree"):
            GROptimizerGater(n_aux=1).discover(optimizer)

    def test_an_aux_group_with_no_local_shards_is_accepted(self):
        """Discovery is a STRUCTURAL check: group structure is rank-uniform, but shard
        ownership is not, so under the distributed optimizer a rank whose data-parallel
        shard misses one module's params holds a legitimately empty group for it. World size 1
        cannot produce that split, so the empty group is staged directly — what is being
        pinned is that emptiness alone is not treated as a wiring failure."""
        from megatron.bridge.training.gradient_routing.optimizer_gating import GROptimizerGater

        model = _wrapped_model(n_aux=2)
        optimizer = _build_optimizer(model, n_aux=2)
        for group in _groups(optimizer):
            if _aux_index(group) == 1:
                group["params"] = []

        gater = GROptimizerGater(n_aux=2)
        gater.discover(optimizer)
        assert gater.discovered
        gater.arm(update_core=True, update_aux=[True, False])  # emptying an empty group is a no-op
        gater.restore()


@requires_gpu
@pytest.mark.usefixtures("model_parallel")
class TestFreezing:
    """The core claim: an emptied group's parameters AND Adam state are untouched."""

    def _warm(self, model, optimizer, gater, n_aux=1):
        """Two unrestricted steps, so every parameter has non-trivial Adam state to protect."""
        for seed in (0, 1):
            _step(model, optimizer, gater, True, [True] * n_aux, seed=seed)

    @pytest.mark.parametrize(
        "update_core, update_aux, frozen_names, moving_names",
        [
            (False, [True], CORE_PARAMS, AUX_PARAMS),
            (True, [False], AUX_PARAMS, CORE_PARAMS),
        ],
    )
    def test_frozen_role_parameters_do_not_move(self, update_core, update_aux, frozen_names, moving_names):
        model, optimizer, gater = _rig()
        self._warm(model, optimizer, gater)

        before = _snapshot_params(model)
        _step(model, optimizer, gater, update_core, update_aux, seed=2)
        after = _params(model)

        for name in frozen_names:
            assert torch.equal(before[name], after[name]), f"frozen parameter {name} moved"
        for name in moving_names:
            assert not torch.equal(before[name], after[name]), f"updating parameter {name} did not move"

    @pytest.mark.parametrize("stepping", [0, 1])
    def test_one_module_steps_while_its_sibling_stays_frozen(self, stepping):
        """The multi-module isolation claim, on parameters: a core-robustness iteration opens
        and steps exactly one module, so the other module's weights must not move at all —
        otherwise one capability's corpus is writing into another's module."""
        model, optimizer, gater = _rig(n_aux=2)
        self._warm(model, optimizer, gater, n_aux=2)

        update_aux = [False, False]
        update_aux[stepping] = True
        before = _snapshot_params(model)
        _step(model, optimizer, gater, update_core=True, update_aux=update_aux, seed=2)
        after = _params(model)

        for name in aux_params(stepping):
            assert not torch.equal(before[name], after[name]), f"stepping module's {name} did not move"
        for name in aux_params(1 - stepping):
            assert torch.equal(before[name], after[name]), f"frozen module's {name} moved"

    @pytest.mark.parametrize("stepping", [0, 1])
    def test_a_frozen_modules_adam_state_is_untouched_while_its_sibling_steps(self, stepping):
        """And on the moments, which is the claim that rules out the alternatives: lr=0 or
        zeroed grads on the frozen module would still fold this step's gradient into its
        ``exp_avg``/``exp_avg_sq`` and still advance its step count."""
        model, optimizer, gater = _rig(n_aux=2)
        self._warm(model, optimizer, gater, n_aux=2)

        frozen_positions = _group_positions(optimizer, module=1 - stepping)
        stepping_positions = _group_positions(optimizer, module=stepping)
        update_aux = [False, False]
        update_aux[stepping] = True

        before = _snapshot_adam_state(optimizer)
        _step(model, optimizer, gater, update_core=True, update_aux=update_aux, seed=3)
        after = _snapshot_adam_state(optimizer)

        assert before, "no Adam state was captured — the warm-up steps did not run"
        frozen_keys = [key for key in before if key[0] in frozen_positions]
        moving_keys = [key for key in before if key[0] in stepping_positions]
        assert frozen_keys and moving_keys

        for key in frozen_keys:
            for state_name, value in before[key].items():
                new = after[key][state_name]
                if torch.is_tensor(value):
                    assert torch.equal(value, new), f"frozen module's {state_name} at {key} changed"
                else:
                    assert value == new, f"frozen module's {state_name} at {key} changed: {value} -> {new}"
        assert any(
            not torch.equal(before[key]["exp_avg"], after[key]["exp_avg"])
            for key in moving_keys
            if "exp_avg" in before[key]
        ), "the stepping module's moments did not move — the comparison proves nothing"

    @pytest.mark.parametrize("update_core, update_aux", [(False, [True]), (True, [False])])
    def test_frozen_role_adam_state_is_untouched(self, update_core, update_aux):
        """lr=0 or zeroed grads would still move exp_avg / exp_avg_sq / step here."""
        model, optimizer, gater = _rig()
        self._warm(model, optimizer, gater)

        frozen_positions = (
            _group_positions(optimizer, module=None) if not update_core else _group_positions(optimizer, module=0)
        )
        before = _snapshot_adam_state(optimizer)
        _step(model, optimizer, gater, update_core, update_aux, seed=3)
        after = _snapshot_adam_state(optimizer)

        assert before, "no Adam state was captured — the warm-up steps did not run"
        frozen_keys = [key for key in before if key[0] in frozen_positions]
        moving_keys = [key for key in before if key[0] not in frozen_positions]
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

    @pytest.mark.parametrize("n_aux", [1, 2])
    def test_unfrozen_step_moves_everything(self, n_aux):
        """The control arm: with every role updating, nothing is held back."""
        model, optimizer, gater = _rig(n_aux=n_aux)
        before = _snapshot_params(model)
        _step(model, optimizer, gater, True, [True] * n_aux)
        after = _params(model)
        names = CORE_PARAMS + tuple(name for k in range(n_aux) for name in aux_params(k))
        for name in names:
            assert not torch.equal(before[name], after[name]), f"{name} did not move on an unrestricted step"

    def test_distributed_optimizer_freezes_the_same_way(self):
        """The distributed optimizer's inner groups hold shards, but the role marker and the
        emptying survive — the shipped configs use it."""
        model, optimizer, gater = _rig(use_distributed_optimizer=True, n_aux=2)
        _step(model, optimizer, gater, True, [True, True], seed=0)

        before = _snapshot_params(model)
        _step(model, optimizer, gater, update_core=False, update_aux=[True, False], seed=2)
        after = _params(model)
        for name in CORE_PARAMS + aux_params(1):
            assert torch.equal(before[name], after[name]), f"frozen {name} moved under the distributed optimizer"
        for name in aux_params(0):
            assert not torch.equal(before[name], after[name]), f"{name} did not move under the distributed optimizer"


@requires_gpu
@pytest.mark.usefixtures("model_parallel")
class TestArmRestoreStateMachine:
    def test_arm_empties_only_the_frozen_roles(self):
        model, optimizer, gater = _rig(n_aux=2)
        gater.arm(update_core=False, update_aux=[True, False])
        for group in _groups(optimizer):
            index = _aux_index(group)
            if index == 0:
                assert group["params"], "the updating module was emptied"
            else:
                assert group["params"] == [], f"group {index} (frozen) was not emptied"
        gater.restore()

    def test_restore_puts_back_the_identical_param_lists(self):
        _, optimizer, gater = _rig(n_aux=2)
        before = [list(group["params"]) for group in _groups(optimizer)]
        gater.arm(update_core=False, update_aux=[True, False])
        gater.restore()
        after = [list(group["params"]) for group in _groups(optimizer)]
        assert len(before) == len(after)
        for before_group, after_group in zip(before, after):
            assert len(before_group) == len(after_group)
            assert all(b is a for b, a in zip(before_group, after_group)), "restore swapped parameter identities"

    def test_restore_is_a_noop_when_nothing_is_armed(self):
        _, optimizer, gater = _rig()
        before = [list(group["params"]) for group in _groups(optimizer)]
        gater.restore()
        gater.restore()
        after = [list(group["params"]) for group in _groups(optimizer)]
        assert before == after

    def test_arm_is_idempotent_for_the_same_roles(self):
        """The rerun state machine can drive grad finalization more than once per step."""
        _, optimizer, gater = _rig(n_aux=2)
        gater.arm(update_core=False, update_aux=[True, False])
        armed = [list(group["params"]) for group in _groups(optimizer)]
        gater.arm(update_core=False, update_aux=[True, False])
        assert [list(group["params"]) for group in _groups(optimizer)] == armed
        gater.restore()
        assert all(group["params"] for group in _groups(optimizer))

    def test_conflicting_rearm_without_restore_raises(self):
        """Re-arming over a live stash would lose the stashed params permanently."""
        _, _, gater = _rig(n_aux=2)
        gater.arm(update_core=False, update_aux=[True, False])
        with pytest.raises(RuntimeError, match="re-armed with"):
            gater.arm(update_core=True, update_aux=[False, True])
        gater.restore()

    def test_a_rearm_that_differs_only_in_which_module_is_frozen_raises(self):
        """Same number of frozen roles, different roles: the check must compare the SET, not
        the count, or a per-module drift would restore the wrong groups."""
        _, _, gater = _rig(n_aux=2)
        gater.arm(update_core=True, update_aux=[True, False])
        with pytest.raises(RuntimeError, match="re-armed with"):
            gater.arm(update_core=True, update_aux=[False, True])
        gater.restore()

    @pytest.mark.parametrize("n_aux, update_aux", [(1, [False]), (2, [False, False])])
    def test_arming_a_step_that_updates_nothing_raises(self, n_aux, update_aux):
        _, _, gater = _rig(n_aux=n_aux)
        with pytest.raises(RuntimeError, match="update neither core nor any aux"):
            gater.arm(update_core=False, update_aux=update_aux)

    @pytest.mark.parametrize("n_aux, update_aux", [(2, [True]), (2, [True, True, True]), (1, [True, False])])
    def test_arming_with_the_wrong_number_of_flags_raises(self, n_aux, update_aux):
        """The flags come straight off ``plan.update_aux[i]``, so a length mismatch means the
        plan and the model disagree about the module count — and a shorter row would silently
        leave the trailing modules unfrozen on every iteration."""
        _, _, gater = _rig(n_aux=n_aux)
        with pytest.raises(RuntimeError, match=f"armed with {len(update_aux)} aux update flags for {n_aux} modules"):
            gater.arm(update_core=True, update_aux=update_aux)

    @pytest.mark.parametrize("n_aux", [1, 2])
    def test_arming_every_role_stashes_nothing(self, n_aux):
        _, optimizer, gater = _rig(n_aux=n_aux)
        gater.arm(update_core=True, update_aux=[True] * n_aux)
        assert all(group["params"] for group in _groups(optimizer))
        gater.restore()

    def test_a_numpy_plan_row_arms_correctly(self):
        """``arm`` is fed ``plan.update_aux[i]`` verbatim — a numpy row of int64, not a list of
        bools — so the truthiness conversion has to happen inside the gater."""
        import numpy as np

        _, optimizer, gater = _rig(n_aux=2)
        gater.arm(update_core=True, update_aux=np.asarray([1, 0], dtype=np.int64))
        assert [bool(g["params"]) for g in _groups(optimizer) if _is_aux(g)] == [True, False]
        gater.restore()

    def test_state_dict_round_trips_after_restore(self):
        """Checkpointing happens between steps, i.e. after restore; the emptying must leave
        no trace in the saved optimizer state."""
        model, optimizer, gater = _rig(n_aux=2)
        _step(model, optimizer, gater, True, [True, True], seed=0)
        _step(model, optimizer, gater, False, [True, False], seed=1)

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

        update_aux = np.asarray(update_aux, dtype=np.int64)
        n, n_aux = update_aux.shape
        return GRPlan(
            corpus=np.zeros(n, dtype=np.int64),
            fwd_aux=update_aux.copy(),
            update_core=np.asarray(update_core, dtype=np.int64),
            update_aux=update_aux,
            prior_iters_same_corpus=np.arange(n, dtype=np.int64),
            plan_seed=0,
            p_as=0.0,
            p_cr=0.0,
            aux_iter_fractions=(0.0,) * n_aux,
        )

    def _install(self, base_finalize, update_core, update_aux, step=0):
        from megatron.bridge.training.gradient_routing.optimizer_gating import GROptimizerGater, install_gr_finalize

        n_aux = len(update_aux[0])
        model = _wrapped_model(n_aux=n_aux)
        optimizer = _build_optimizer(model, n_aux=n_aux)
        gater = GROptimizerGater(n_aux=n_aux)
        state = SimpleNamespace(train_state=SimpleNamespace(step=step))
        finalize = install_gr_finalize(base_finalize, gater, optimizer, self._plan(update_core, update_aux), state)
        return finalize, optimizer, gater, state

    def test_wrapper_runs_the_base_finalize_then_arms_from_the_plan(self):
        calls = []
        finalize, optimizer, gater, state = self._install(
            lambda *a, **k: calls.append((a, k)) or "base-result",
            update_core=[0, 1],
            update_aux=[[1], [1]],
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

    def test_wrapper_arms_the_per_module_row_of_the_plan(self):
        """The plan's ``update_aux`` row is per module, and the wrapper must pass THAT row —
        not "any module updates" — or a core-robustness iteration would step every module."""
        finalize, optimizer, gater, state = self._install(
            lambda: None, update_core=[1, 1], update_aux=[[1, 0], [0, 1]]
        )

        finalize()
        assert [bool(g["params"]) for g in _groups(optimizer) if _is_aux(g)] == [True, False]
        gater.restore()

        state.train_state.step = 1
        finalize()
        assert [bool(g["params"]) for g in _groups(optimizer) if _is_aux(g)] == [False, True]
        gater.restore()

    def test_wrapper_discovers_only_once(self):
        discoveries = []
        finalize, _, gater, state = self._install(lambda: None, update_core=[1, 1], update_aux=[[1], [1]])
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

        finalize, _, _, _ = self._install(lambda: None, update_core=[1], update_aux=[[1]])
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
        self._install(lambda: first_calls.append(1), update_core=[1], update_aux=[[1]])
        finalize, _, gater, _ = self._install(lambda: second_calls.append(1), update_core=[1], update_aux=[[1]])

        finalize()
        gater.restore()
        assert (first_calls, second_calls) == ([], [1]), "the stale runtime was still in use"
        assert optimizer_gating._GR_RUNTIME["gater"] is gater


#: Base weight decay for the schedule tests below; each aux group's must come out scaled by
#: its own wd_mult.
BASE_WD = 0.1


@requires_gpu
@pytest.mark.usefixtures("model_parallel")
class TestAuxScheduleIsSeparate:
    """``gr.aux_lr`` must reach the RUNNING learning rate, not merely the group dict.

    A fresh zero-init aux module and a warm core need different LRs — that is the whole
    reason each aux gets a param group of its own. But ``max_lr``/``wd_mult`` on the group are
    only a request: the ``OptimizerParamScheduler`` is what turns them into the ``lr`` and
    ``weight_decay`` each step actually uses, and a scheduler that read the global max_lr
    instead would train the aux modules at the core's rate (1e-4 against 5e-6 in the shipped
    config, on modules initialised at zero) with nothing anywhere to say so. So this drives
    the real ``setup_optimizer`` — the same call ``setup.py`` makes — and reads the schedule
    off the groups the optimizer steps.
    """

    def _optimizer_and_scheduler(self, n_aux=1):
        from megatron.core.optimizer import OptimizerConfig

        from megatron.bridge.training.config import SchedulerConfig
        from megatron.bridge.training.optim import setup_optimizer

        scheduler_config = SchedulerConfig(
            lr_decay_iters=10,
            lr_decay_style="constant",
            start_weight_decay=BASE_WD,
            end_weight_decay=BASE_WD,
            weight_decay_incr_style="constant",
        )
        # lr_decay_steps / lr_warmup_steps / wd_incr_steps / wsd_decay_steps are init=False
        # fields that the training setup fills in from train_iters before building the
        # optimizer; there is no public setter, so they are populated here the same way.
        scheduler_config.lr_decay_steps = 10
        scheduler_config.lr_warmup_steps = 0
        scheduler_config.wd_incr_steps = 10
        scheduler_config.wsd_decay_steps = None

        optimizer, scheduler = setup_optimizer(
            optimizer_config=OptimizerConfig(
                optimizer="adam", lr=BASE_LR, min_lr=BASE_MIN_LR, weight_decay=BASE_WD, clip_grad=0.0
            ),
            scheduler_config=scheduler_config,
            model=[_wrapped_model(n_aux=n_aux)],
            optimizer_config_override_provider=_gr_provider(n_aux),
        )
        scheduler.step(increment=1)
        groups = _groups(optimizer)
        return {_aux_index(g): g for g in groups if _is_aux(g)}, [g for g in groups if not _is_aux(g)]

    @pytest.mark.parametrize("n_aux", [1, 2])
    def test_each_aux_group_is_scheduled_at_its_own_learning_rate(self, n_aux):
        aux_groups, core_groups = self._optimizer_and_scheduler(n_aux)
        assert set(aux_groups) == set(range(n_aux)) and core_groups
        assert len(set(AUX_LRS[:n_aux] + [BASE_LR])) == n_aux + 1, "the arms must differ or this proves nothing"
        for k, group in aux_groups.items():
            assert float(group["lr"]) == pytest.approx(AUX_LRS[k])
        for group in core_groups:
            assert float(group["lr"]) == pytest.approx(BASE_LR)

    @pytest.mark.parametrize("n_aux", [1, 2])
    def test_each_aux_weight_decay_multiplier_is_applied(self, n_aux):
        """``wd_mult`` is one of the ways an aux group's decay differs, and together with the
        role marker it is what makes the groups distinct (grouping is by merged-override
        equality)."""
        aux_groups, core_groups = self._optimizer_and_scheduler(n_aux)
        for k, group in aux_groups.items():
            assert float(group["weight_decay"]) == pytest.approx(BASE_WD * AUX_WD_MULTS[k])
        for group in core_groups:
            assert float(group["weight_decay"]) == pytest.approx(BASE_WD)


@requires_gpu
@pytest.mark.usefixtures("model_parallel")
class TestAuxGroupAdamSteps:
    """The group-level Adam step counters must count exactly the armed updates.

    This is accumulate-then-gate's isolation invariant made observable: FusedAdam skips
    empty groups before touching its per-group ``step`` counter, so after any schedule
    of armed/frozen iterations each module's counter equals its armed count — the
    property the callback asserts every step on real runs.
    """

    def test_counters_match_the_armed_update_counts(self):
        model, optimizer, gater = _rig(n_aux=2)
        schedule = [
            (True, (True, False)),
            (True, (False, False)),
            (False, (True, False)),
            (True, (False, True)),
        ]
        for seed, (core, aux) in enumerate(schedule):
            _step(model, optimizer, gater, update_core=core, update_aux=aux, seed=seed)
        observed = gater.aux_group_adam_steps()
        assert observed[0] and all(step == 2 for step in observed[0]), observed
        assert observed[1] and all(step == 1 for step in observed[1]), observed

    def test_a_never_updated_module_reports_zero_on_a_counter_carrying_optimizer(self):
        """FusedAdam creates ``step`` on a group's first visit, so a module frozen for the
        whole run has no counter — on an optimizer that carries counters elsewhere that
        means zero visits, and reporting 0 is what lets the caller catch a group the
        gating failed to arm."""
        model, optimizer, gater = _rig(n_aux=2)
        for seed in range(3):
            _step(model, optimizer, gater, update_core=True, update_aux=(True, False), seed=seed)
        observed = gater.aux_group_adam_steps()
        assert observed is not None
        assert observed[1] and all(step == 0 for step in observed[1])
        assert observed[0] and all(step == 3 for step in observed[0])

    def test_an_optimizer_without_group_counters_reports_none(self):
        """torch optimizers keep per-param state with no group-level ``step`` — there is
        nothing to verify against, and the accessor must say so rather than return an
        empty verification that reads as a pass."""
        from megatron.bridge.training.gradient_routing.optimizer_gating import (
            GR_ROLE_KEY,
            GROptimizerGater,
            gr_aux_role,
        )

        params = [torch.nn.Parameter(torch.randn(4, device="cuda")) for _ in range(2)]
        groups = [
            {"params": [params[0]], GR_ROLE_KEY: gr_aux_role(0)},
            {"params": [params[1]]},
        ]
        inner = torch.optim.Adam(groups, lr=1e-3)
        gater = GROptimizerGater(n_aux=1)
        gater.discover(SimpleNamespace(optimizer=inner))
        assert gater.aux_group_adam_steps() is None

    def test_reading_while_armed_raises(self):
        model, optimizer, gater = _rig()
        gater.arm(update_core=True, update_aux=(False,))
        with pytest.raises(RuntimeError, match="armed"):
            gater.aux_group_adam_steps()
        gater.restore()

    def test_reading_before_discovery_raises(self):
        from megatron.bridge.training.gradient_routing.optimizer_gating import GROptimizerGater

        with pytest.raises(RuntimeError, match="discover"):
            GROptimizerGater(n_aux=1).aux_group_adam_steps()
