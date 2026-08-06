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
"""The callback is where the plan becomes per-iteration model state.

Everything else in the GR stack is derived: the plan is a pure function of its seed, the
gater freezes whatever role it is told to. The callback is the only place that reads
iteration ``i`` and WRITES it into the running model — the forward gate on every GRAM
layer and the expert-bias freeze on every router. A callback that silently registered
fewer layers than the model has, or that wrote the wrong iteration's gate, would produce a
run that trains, logs, and checkpoints normally while routing gradients somewhere other
than where the experiment says. Nothing downstream would notice.

So the layers are REAL ``GRAMMoELayer`` instances (shared builders, world-1 parallel
state) and the gate/freeze assertions are made over every layer, not a representative one.
The plan is hand-built rather than seeded so each of the paper's four iteration types
appears exactly once at a known index.

The optimizer is the one stand-in: the gater under test is the real ``GROptimizerGater``,
but it is pointed at a plain torch Adam behind a namespace with the ``.optimizer``
attribute ``_iter_inner_param_groups`` walks. That is the whole interface the gater uses —
it only ever swaps ``group["params"]`` — and building a real ``get_megatron_optimizer``
rig here would duplicate ``test_gr_optimizer_gating.py``, which already pins the
optimizer-side behaviour against the real thing.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tests.unit_tests.gr_test_utils import (
    HIDDEN,
    build_moe_layer,
    gram_spec,
    init_model_parallel,
    moe_builder,
    moe_config,
    stack_spec,
)


requires_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU (real TE MoE layers)")

N_LAYERS, SEQ, BATCH = 2, 6, 2

#: The paper's four iteration types, one each, at a known index (see plan.py's table).
#: 0 forget-isolated, 1 forget-spread, 2 core, 3 core-robustness.
CORPUS = [1, 1, 0, 0]
FWD_AUX = [1, 1, 0, 1]
UPDATE_CORE = [0, 1, 1, 1]
UPDATE_AUX = [1, 1, 0, 1]


@pytest.fixture(scope="module")
def moe_parallel_state():
    """Real world-1 mcore parallel state; MoELayer picks the default pg_collection up from it."""
    from megatron.core import parallel_state
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    init_model_parallel(expert_model_parallel_size=1)
    model_parallel_cuda_manual_seed(1234)
    yield
    parallel_state.destroy_model_parallel()


def _plan():
    from megatron.bridge.training.gradient_routing.plan import GRPlan

    return GRPlan(
        corpus=np.asarray(CORPUS, dtype=np.int64),
        fwd_aux=np.asarray(FWD_AUX, dtype=np.int64),
        update_core=np.asarray(UPDATE_CORE, dtype=np.int64),
        update_aux=np.asarray(UPDATE_AUX, dtype=np.int64),
        prior_iters_same_corpus=np.asarray([0, 1, 0, 1], dtype=np.int64),
        plan_seed=1234,
        p_as=0.5,
        p_cr=0.5,
        forget_iter_fraction=0.5,
    )


def _model_chunk(gram=True, n_layers=N_LAYERS):
    """One model chunk holding ``n_layers`` MoE layers, GRAM-swapped or vanilla.

    A ``ModuleList`` is the whole chunk: the callback only walks ``.modules()``, so a
    stack with a forward would add nothing the assertions can see.
    """
    config = moe_config()
    builder = moe_builder(gram_spec() if gram else stack_spec())
    return torch.nn.ModuleList(
        [build_moe_layer(builder, config, layer_number=i + 1, seed=4321 + i) for i in range(n_layers)]
    )


def _context(model, step=0, loss_dict=None):
    from megatron.bridge.training.callbacks import CallbackContext

    return CallbackContext(
        state=SimpleNamespace(train_state=SimpleNamespace(step=step)),
        model=[model],
        loss_dict=loss_dict,
    )


def _discovered_gater(model):
    """A real ``GROptimizerGater`` discovered over a role-marked torch optimizer.

    Two groups, aux carrying the same role marker mcore's override combiner writes. The
    namespace stands in for the ``MegatronOptimizer`` wrapper whose ``.optimizer`` holds
    the inner param groups — the only attribute the gater reads.
    """
    from megatron.bridge.training.gradient_routing.optimizer_gating import (
        GR_ROLE_AUX,
        GR_ROLE_KEY,
        GROptimizerGater,
    )

    aux, core = [], []
    for name, param in model.named_parameters():
        (aux if ".gr_aux." in name else core).append(param)
    assert aux and core, "the rig needs both roles populated to prove restore() ran"
    inner = torch.optim.Adam([{"params": aux, GR_ROLE_KEY: GR_ROLE_AUX}, {"params": core}], lr=1e-3)
    gater = GROptimizerGater()
    gater.discover(SimpleNamespace(optimizer=inner))
    return gater, inner


def _callback(gater=None, log_interval=1):
    """The callback under test. An undiscovered gater is the default: only the restore
    assertions care, and ``restore()`` on an unarmed gater is a documented no-op."""
    from megatron.bridge.training.gradient_routing.callback import GRCallback
    from megatron.bridge.training.gradient_routing.optimizer_gating import GROptimizerGater

    return GRCallback(_plan(), gater if gater is not None else GROptimizerGater(), log_interval=log_interval)


def _recorded_metrics(monkeypatch):
    """Capture what the callback emits, at the shared W&B helper's seam."""
    from megatron.bridge.training.gradient_routing import callback as callback_module

    calls = []
    monkeypatch.setattr(
        callback_module, "log_wandb_metrics_nonfatal", lambda metrics, step: calls.append((metrics, step))
    )
    return calls


@requires_gpu
@pytest.mark.usefixtures("moe_parallel_state")
class TestOnTrainStart:
    def test_registries_cover_every_gram_layer(self, monkeypatch):
        """Read out through the gate write: a layer missing from the registry keeps its 0.0
        gate on a forward-aux iteration, which is exactly the silent half-routed run."""
        _recorded_metrics(monkeypatch)
        model = _model_chunk()
        callback = _callback()

        callback.on_train_start(_context(model))
        callback.on_train_step_start(_context(model, step=0))  # fwd_aux[0] == 1

        assert [float(layer.gr_gate) for layer in model] == [1.0] * N_LAYERS
        assert [layer.router.frozen_expert_bias for layer in model] == [True] * N_LAYERS

    def test_a_non_zero_aux_output_projection_raises(self, monkeypatch):
        """Warm-start protection: a non-zero fc2 means the checkpoint load clobbered the
        fresh zero-init, so the run would not start from the core model it claims to."""
        _recorded_metrics(monkeypatch)
        model = _model_chunk()
        with torch.no_grad():
            model[-1].gr_aux.linear_fc2.weight.fill_(1e-3)
        callback = _callback()

        with pytest.raises(RuntimeError, match="non-zero at train start"):
            callback.on_train_start(_context(model))

    def test_a_model_without_gram_layers_raises(self, monkeypatch):
        """The spec swap not running is the failure that otherwise trains happily as a
        plain CPT run while reporting itself as gradient-routed."""
        _recorded_metrics(monkeypatch)
        model = _model_chunk(gram=False)
        callback = _callback()

        with pytest.raises(RuntimeError, match="found no GRAMMoELayer"):
            callback.on_train_start(_context(model))

    def test_the_plan_digest_is_logged_at_the_starting_iteration(self, monkeypatch):
        """Run provenance: the digest is how a W&B run is tied back to its plan."""
        calls = _recorded_metrics(monkeypatch)
        model = _model_chunk()
        callback = _callback()

        callback.on_train_start(_context(model, step=2))

        (metrics, step) = calls[-1]
        assert step == 2
        assert metrics["run/gr_plan_digest_int"] == int(_plan().digest()[:8], 16)


@requires_gpu
@pytest.mark.usefixtures("moe_parallel_state")
class TestOnTrainStepStart:
    @pytest.mark.parametrize("iteration", range(len(CORPUS)))
    def test_gate_and_bias_freeze_follow_the_plan(self, monkeypatch, iteration):
        """Both writes, on every layer, for each of the four iteration types.

        The expert-bias freeze is the non-obvious half: the router's load-balancing bias
        update runs in grad finalization, OUTSIDE the optimizer, so param-group emptying
        does not reach it — an iteration that does not update core must freeze it here or
        the "frozen" core still moves.
        """
        _recorded_metrics(monkeypatch)
        model = _model_chunk()
        callback = _callback()
        callback.on_train_start(_context(model))

        callback.on_train_step_start(_context(model, step=iteration))

        for layer in model:
            assert float(layer.gr_gate) == float(FWD_AUX[iteration])
            assert layer.router.frozen_expert_bias is (not bool(UPDATE_CORE[iteration]))

    def test_the_probe_hook_is_registered_on_a_logged_iteration_and_removed_after(self, monkeypatch):
        _recorded_metrics(monkeypatch)
        model = _model_chunk()
        callback = _callback()
        callback.on_train_start(_context(model))

        callback.on_train_step_start(_context(model, step=0))
        assert model[0].gr_aux._forward_hooks, "no probe hook on a logged iteration"

        callback.on_train_step_end(_context(model, step=0))
        assert not model[0].gr_aux._forward_hooks, "the probe hook outlived the step"

    def test_no_probe_hook_between_log_intervals(self, monkeypatch):
        """The probe costs a device sync per logged step; log_interval must actually gate it."""
        _recorded_metrics(monkeypatch)
        model = _model_chunk()
        callback = _callback(log_interval=2)
        callback.on_train_start(_context(model))

        callback.on_train_step_start(_context(model, step=1))

        assert not model[0].gr_aux._forward_hooks


@requires_gpu
@pytest.mark.usefixtures("moe_parallel_state")
class TestOnTrainStepEnd:
    def test_the_emptied_param_groups_are_restored(self, monkeypatch):
        """The gater is armed in grad finalization and MUST be released here: a stash that
        survives the step makes the next arm() raise, and an unrestored group would train
        nothing for the rest of the run."""
        _recorded_metrics(monkeypatch)
        model = _model_chunk()
        gater, inner = _discovered_gater(model)
        callback = _callback(gater=gater)
        callback.on_train_start(_context(model))

        gater.arm(update_core=bool(UPDATE_CORE[0]), update_aux=bool(UPDATE_AUX[0]))
        assert [bool(group["params"]) for group in inner.param_groups] == [True, False], (
            "iteration 0 does not update core, so the core group should be emptied"
        )

        callback.on_train_step_end(_context(model, step=0))

        assert all(group["params"] for group in inner.param_groups)
        gater.arm(update_core=True, update_aux=True)  # a live stash would raise instead

    @pytest.mark.parametrize("iteration", range(len(CORPUS)))
    def test_the_plan_row_is_logged_verbatim(self, monkeypatch, iteration):
        calls = _recorded_metrics(monkeypatch)
        model = _model_chunk()
        callback = _callback()
        callback.on_train_start(_context(model))

        callback.on_train_step_end(_context(model, step=iteration))

        metrics, step = calls[-1]
        assert step == iteration
        assert metrics["gr/corpus"] == CORPUS[iteration]
        assert metrics["gr/fwd_aux"] == FWD_AUX[iteration]
        assert metrics["gr/update_core"] == UPDATE_CORE[iteration]
        assert metrics["gr/update_aux"] == UPDATE_AUX[iteration]
        assert metrics["gr/aux_steps_cum"] == sum(UPDATE_AUX[: iteration + 1])
        assert metrics["gr/core_steps_cum"] == sum(UPDATE_CORE[: iteration + 1])

    @pytest.mark.parametrize("iteration, expected_key", [(0, "gr/loss_forget"), (2, "gr/loss_retain")])
    def test_the_loss_is_logged_under_its_corpus_key(self, monkeypatch, iteration, expected_key):
        """Two separate curves is the whole point: one series would average the forget and
        retain losses together and hide which one is moving."""
        calls = _recorded_metrics(monkeypatch)
        model = _model_chunk()
        callback = _callback()
        callback.on_train_start(_context(model))

        loss = torch.tensor(1.5, device="cuda")
        callback.on_train_step_end(_context(model, step=iteration, loss_dict={"lm loss": loss}))

        metrics = calls[-1][0]
        assert metrics[expected_key] == pytest.approx(1.5)
        assert "gr/loss_forget" in metrics or "gr/loss_retain" in metrics
        assert not ("gr/loss_forget" in metrics and "gr/loss_retain" in metrics)

    def test_the_aux_output_rms_probe_reports_a_real_forward(self, monkeypatch):
        """Zero-init aux + gate 1 still produces a zero output — the probe reads the aux
        module's own output, so it is 0.0 until the aux has actually trained."""
        calls = _recorded_metrics(monkeypatch)
        model = _model_chunk(n_layers=1)
        callback = _callback()
        callback.on_train_start(_context(model))
        callback.on_train_step_start(_context(model, step=0))

        with torch.no_grad():
            model[0](torch.randn(SEQ, BATCH, HIDDEN, dtype=torch.bfloat16, device="cuda"))
        callback.on_train_step_end(_context(model, step=0))

        assert calls[-1][0]["gr/aux_out_rms"] == pytest.approx(0.0)

    def test_the_aux_parameter_norm_is_logged_on_a_logged_iteration(self, monkeypatch):
        calls = _recorded_metrics(monkeypatch)
        model = _model_chunk()
        callback = _callback()
        callback.on_train_start(_context(model))

        callback.on_train_step_end(_context(model, step=0))

        expected = (
            sum(float(p.detach().float().pow(2).sum()) for layer in model for p in layer.gr_aux.parameters()) ** 0.5
        )
        assert calls[-1][0]["gr/aux_param_norm"] == pytest.approx(expected, rel=1e-5)
