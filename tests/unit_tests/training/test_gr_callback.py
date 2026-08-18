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
gater freezes whatever roles it is told to. The callback is the only place that reads
iteration ``i`` and WRITES it into the running model — the forward gate vector on every GRAM
layer and the expert-bias freeze on every router. A callback that silently registered
fewer layers than the model has, or that wrote the wrong iteration's gates, would produce a
run that trains, logs, and checkpoints normally while routing gradients somewhere other
than where the experiment says. Nothing downstream would notice.

So the layers are REAL ``GRAMMoELayer`` instances (shared builders, world-1 parallel
state) and the gate/freeze assertions are made over every layer AND every module, not a
representative one. The plan is hand-built rather than seeded so each of the paper's four
iteration types appears exactly once at a known index, and — at N=2 — so a core-robustness
iteration that activates only the second module appears at a known index too.

The optimizer is the one stand-in: the gater under test is the real ``GROptimizerGater``,
but it is pointed at a plain torch Adam behind a namespace with the ``.optimizer``
attribute ``_iter_inner_param_groups`` walks. That is the whole interface the gater uses —
it only ever swaps ``group["params"]`` — and building a real ``get_megatron_optimizer``
rig here would duplicate ``test_gr_optimizer_gating.py``, which already pins the
optimizer-side behaviour against the real thing.
"""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tests.unit_tests.gr_test_utils import (
    AUX_FFN,
    AUX_FFNS,
    HIDDEN,
    build_moe_layer,
    gram_spec,
    init_model_parallel,
    moe_builder,
    moe_config,
    stack_spec,
    teardown_model_parallel,
)


requires_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU (real TE MoE layers)")

N_LAYERS, SEQ, BATCH = 2, 6, 2

#: The paper's four iteration types, one each, at a known index (see plan.py's table).
#: 0 aux-isolated, 1 aux-spread, 2 core, 3 core-robustness.
CORPUS = [1, 1, 0, 0]
FWD_AUX = [[1], [1], [0], [1]]
UPDATE_CORE = [0, 1, 1, 1]
UPDATE_AUX = [[1], [1], [0], [1]]

#: The same four types at N=2, plus a second core-robustness iteration for the OTHER module:
#: 0 aux0-isolated, 1 aux0-spread, 2 aux1-isolated, 3 core, 4 core-robustness(module 0),
#: 5 core-robustness(module 1).
CORPUS_2 = [1, 1, 2, 0, 0, 0]
FWD_AUX_2 = [[1, 0], [1, 0], [0, 1], [0, 0], [1, 0], [0, 1]]
UPDATE_CORE_2 = [0, 1, 0, 1, 1, 1]
UPDATE_AUX_2 = [[1, 0], [1, 0], [0, 1], [0, 0], [1, 0], [0, 1]]


@pytest.fixture(scope="module")
def moe_parallel_state():
    """Real world-1 mcore parallel state; MoELayer picks the default pg_collection up from it."""
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    init_model_parallel(expert_model_parallel_size=1)
    model_parallel_cuda_manual_seed(1234)
    yield
    teardown_model_parallel()


def _plan(corpus=None, fwd_aux=None, update_core=None, update_aux=None):
    from megatron.bridge.training.gradient_routing.plan import GRPlan

    corpus = CORPUS if corpus is None else corpus
    fwd_aux = np.asarray(FWD_AUX if fwd_aux is None else fwd_aux, dtype=np.int64)
    update_aux = np.asarray(UPDATE_AUX if update_aux is None else update_aux, dtype=np.int64)
    update_core = UPDATE_CORE if update_core is None else update_core
    prior = np.zeros(len(corpus), dtype=np.int64)
    counts: dict[int, int] = {}
    for i, c in enumerate(corpus):
        prior[i] = counts.get(c, 0)
        counts[c] = prior[i] + 1
    return GRPlan(
        corpus=np.asarray(corpus, dtype=np.int64),
        fwd_aux=fwd_aux,
        update_core=np.asarray(update_core, dtype=np.int64),
        update_aux=update_aux,
        prior_iters_same_corpus=prior,
        plan_seed=1234,
        p_as=0.5,
        p_cr=0.5,
        aux_iter_fractions=tuple(0.5 / fwd_aux.shape[1] for _ in range(fwd_aux.shape[1])),
    )


def _plan_2():
    return _plan(corpus=CORPUS_2, fwd_aux=FWD_AUX_2, update_core=UPDATE_CORE_2, update_aux=UPDATE_AUX_2)


def _model_chunk(gram=True, n_layers=N_LAYERS, aux_ffns=(AUX_FFN,)):
    """One model chunk holding ``n_layers`` MoE layers, GRAM-swapped or vanilla.

    A ``ModuleList`` is the whole chunk: the callback only walks ``.modules()``, so a
    stack with a forward would add nothing the assertions can see.
    """
    config = moe_config()
    builder = moe_builder(gram_spec(aux_ffns) if gram else stack_spec())
    return torch.nn.ModuleList(
        [build_moe_layer(builder, config, layer_number=i + 1, seed=4321 + i) for i in range(n_layers)]
    )


def _context(model, step=0, loss_dict=None, load_dir=None):
    from megatron.bridge.training.callbacks import CallbackContext

    return CallbackContext(
        state=SimpleNamespace(
            train_state=SimpleNamespace(step=step),
            cfg=SimpleNamespace(checkpoint=SimpleNamespace(load=load_dir)),
        ),
        model=[model],
        loss_dict=loss_dict,
    )


#: The plan parameters the resume tests build both sides from.
RESUME_PLAN_KWARGS = {"plan_seed": 4242, "aux_iter_fractions": [0.25, 0.25], "p_as": 0.5, "p_cr": 0.2}
RESUME_TRAIN_ITERS = 20


def _resume_plan(**overrides):
    """A REAL plan (build_gr_plan), which is what a digest comparison is meaningful over."""
    from megatron.bridge.training.gradient_routing.plan import build_gr_plan

    kwargs = {**RESUME_PLAN_KWARGS, "train_iters": RESUME_TRAIN_ITERS, **overrides}
    return build_gr_plan(**kwargs)


def _write_resume_checkpoint(root, step, gr_section=None, **overrides):
    """Write the run_config.yaml a resume reads its saved plan parameters from.

    Only the fields the plan is a function of matter here; the real file is the whole
    serialized ConfigContainer.
    """
    import yaml

    from megatron.bridge.training.utils.checkpoint_utils import (
        get_checkpoint_name,
        get_checkpoint_run_config_filename,
    )

    section = {**RESUME_PLAN_KWARGS, **overrides} if gr_section is None else gr_section
    ckpt_dir = Path(get_checkpoint_name(str(root), step))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    Path(get_checkpoint_run_config_filename(str(ckpt_dir))).write_text(
        yaml.safe_dump({"gr": section, "train": {"train_iters": RESUME_TRAIN_ITERS}})
    )
    return str(root)


def _discovered_gater(model, n_aux=1):
    """A real ``GROptimizerGater`` discovered over role-marked torch optimizer groups.

    One group per aux module plus a core group, each aux group carrying the same role marker
    mcore's override combiner writes. The namespace stands in for the ``MegatronOptimizer``
    wrapper whose ``.optimizer`` holds the inner param groups — the only attribute the gater
    reads.
    """
    from megatron.bridge.training.gradient_routing.optimizer_gating import (
        GR_ROLE_KEY,
        GROptimizerGater,
        gr_aux_role,
    )

    aux: dict[int, list] = {k: [] for k in range(n_aux)}
    core = []
    for name, param in model.named_parameters():
        matched = next((k for k in range(n_aux) if f"gr_aux.{k}." in name), None)
        (aux[matched] if matched is not None else core).append(param)
    assert core and all(aux.values()), "the rig needs every role populated to prove restore() ran"
    groups = [{"params": aux[k], GR_ROLE_KEY: gr_aux_role(k)} for k in range(n_aux)] + [{"params": core}]
    inner = torch.optim.Adam(groups, lr=1e-3)
    gater = GROptimizerGater(n_aux=n_aux)
    gater.discover(SimpleNamespace(optimizer=inner))
    return gater, inner


def _callback(gater=None, log_interval=1, plan=None, n_aux=1):
    """The callback under test. An undiscovered gater is the default: only the restore
    assertions care, and ``restore()`` on an unarmed gater is a documented no-op."""
    from megatron.bridge.training.gradient_routing.callback import GRCallback
    from megatron.bridge.training.gradient_routing.optimizer_gating import GROptimizerGater

    return GRCallback(
        plan if plan is not None else _plan(),
        gater if gater is not None else GROptimizerGater(n_aux=n_aux),
        log_interval=log_interval,
    )


def _recorded_metrics(monkeypatch):
    """Capture what the callback emits, at the shared W&B helper's seam.

    The step-end payload is a THUNK (its aux-parameter norms cost a device sync per
    parameter, so the real emitter evaluates it only on the logging rank). Evaluating it
    here is what the emitter would do, and keeps the assertions about the metrics rather
    than about the deferral.
    """
    from megatron.bridge.training.gradient_routing import callback as callback_module

    calls = []
    monkeypatch.setattr(
        callback_module,
        "log_wandb_metrics_nonfatal",
        lambda metrics, step: calls.append((metrics() if callable(metrics) else metrics, step)),
    )
    return calls


def _train_aux(layer, module, scale=0.5, seed=5):
    """Give one layer's copy of a module a non-zero output projection, as training would."""
    torch.manual_seed(seed)
    with torch.no_grad():
        layer.gr_aux[module].linear_fc2.weight.normal_(0.0, scale)


def _kill_squared_relu(layer, module):
    """Zero one layer's copy of a module's fc1.

    No pre-activation can ever be positive again, so ``relu(z)**2`` — whose derivative is
    ``2 relu(z)`` — gives BOTH of that copy's projections exactly zero gradient forever. This
    is the irrecoverable state the liveness metric exists to name.
    """
    with torch.no_grad():
        layer.gr_aux[module].linear_fc1.weight.zero_()


def _pin_fc1_units(layer, module, live):
    """Make EXACTLY ``live`` of one module copy's fc1 units live, for any positive input.

    A ``+1`` row gives ``pre = sum(x) > 0`` whenever every input element is positive; a zeroed
    row gives exactly 0, which is non-positive, so ``d/dz relu(z)**2 = 2 relu(z)`` is zero and
    that unit's fc1 row and fc2 column are frozen forever. Paired with ``_forward(...,
    positive=True)`` this makes the per-unit liveness reading exact rather than a coin flip
    over this rig's dozen tokens — the fraction under test has to be a fact, not a sample.
    """
    with torch.no_grad():
        weight = layer.gr_aux[module].linear_fc1.weight
        weight.zero_()
        weight[:live].fill_(1.0)


def _forward(model, seed=7, scale=1.0, positive=False):
    """Push ONE microbatch through EVERY layer, the way a training step would.

    ``positive`` draws strictly-positive tokens (every element >= 1), which is what makes a
    ``_pin_fc1_units`` row's sign provable instead of seed-dependent.
    """
    torch.manual_seed(seed)
    if positive:
        x = (torch.rand(SEQ, BATCH, HIDDEN, dtype=torch.bfloat16, device="cuda") + 1.0) * scale
    else:
        x = torch.randn(SEQ, BATCH, HIDDEN, dtype=torch.bfloat16, device="cuda") * scale
    with torch.no_grad():
        for layer in model:
            layer(x)
    return x


def _probed_step(callback, model, step, seed=7, scale=1.0, positive=False):
    """One complete logged iteration: the gate write, one microbatch, the step end."""
    callback.on_train_step_start(_context(model, step=step))
    x = _forward(model, seed=seed, scale=scale, positive=positive)
    callback.on_train_step_end(_context(model, step=step))
    return x


@requires_gpu
@pytest.mark.usefixtures("moe_parallel_state")
class TestOnTrainStart:
    def test_registries_cover_every_gram_layer(self, monkeypatch):
        """Read out through the gate write: a layer missing from the registry keeps its 0.0
        gates on a forward-aux iteration, which is exactly the silent half-routed run."""
        _recorded_metrics(monkeypatch)
        model = _model_chunk()
        callback = _callback()

        callback.on_train_start(_context(model))
        callback.on_train_step_start(_context(model, step=0))  # fwd_aux[0] == [1]

        assert [layer.gr_gate.tolist() for layer in model] == [[1.0]] * N_LAYERS
        assert [layer.router.frozen_expert_bias for layer in model] == [True] * N_LAYERS

    def test_a_non_zero_aux_output_projection_raises_at_iteration_zero(self, monkeypatch):
        """Warm-start protection: a non-zero fc2 means the checkpoint load clobbered the
        fresh zero-init, so the run would not start from the core model it claims to."""
        _recorded_metrics(monkeypatch)
        model = _model_chunk()
        with torch.no_grad():
            model[-1].gr_aux[0].linear_fc2.weight.fill_(1e-3)
        callback = _callback()

        with pytest.raises(RuntimeError, match="non-zero at iteration 0"):
            callback.on_train_start(_context(model))

    @pytest.mark.parametrize("module", [0, 1])
    def test_the_zero_init_check_covers_every_module(self, monkeypatch, module):
        """A check that only looked at module 0 would let a clobbered sibling through, and the
        message must name the module so the operator knows which load went wrong."""
        _recorded_metrics(monkeypatch)
        model = _model_chunk(aux_ffns=AUX_FFNS)
        with torch.no_grad():
            model[-1].gr_aux[module].linear_fc2.weight.fill_(1e-3)
        callback = _callback(plan=_plan_2(), n_aux=2)

        with pytest.raises(RuntimeError, match=f"gr_aux.{module}.linear_fc2.weight is non-zero"):
            callback.on_train_start(_context(model))

    def test_a_module_count_mismatch_between_model_and_plan_is_refused(self, monkeypatch):
        """The plan labels iterations per module and the model provides the modules; a
        mismatch means an iteration would drive a gate index the layer does not have (or
        leave one untouched forever), so it is caught before the first forward."""
        _recorded_metrics(monkeypatch)
        model = _model_chunk(aux_ffns=(AUX_FFN,))
        callback = _callback(plan=_plan_2(), n_aux=2)

        with pytest.raises(RuntimeError, match="model surgery and plan disagree about the module count"):
            callback.on_train_start(_context(model))

    def test_a_trained_aux_is_accepted_on_a_mid_plan_resume(self, monkeypatch, tmp_path):
        """A resumed run has trained aux modules by construction. Asserting the zero-init
        invariant past iteration 0 would make GR runs restart-fatal — no ft restart,
        singleton chain, or save_interval run could ever come back up."""
        from megatron.bridge.training.gradient_routing.callback import GRCallback
        from megatron.bridge.training.gradient_routing.optimizer_gating import GROptimizerGater

        _recorded_metrics(monkeypatch)
        model = _model_chunk(aux_ffns=AUX_FFNS)
        with torch.no_grad():
            model[-1].gr_aux[1].linear_fc2.weight.fill_(1e-3)
        load_dir = _write_resume_checkpoint(tmp_path, 7)
        callback = GRCallback(_resume_plan(), GROptimizerGater(n_aux=2), log_interval=1)

        callback.on_train_start(_context(model, step=7, load_dir=load_dir))

        assert callback._gram_layers, "resume must still build the layer registry"

    @pytest.mark.parametrize(
        "changed",
        [
            {"plan_seed": 999},
            {"p_as": 0.9},
            {"p_cr": 0.9},
            {"aux_iter_fractions": [0.25, 0.1]},
            {"aux_iter_fractions": [0.5]},  # a different module COUNT
            {"train_iters": RESUME_TRAIN_ITERS + 4},
        ],
    )
    def test_a_resume_under_a_different_plan_is_refused(self, monkeypatch, tmp_path, changed):
        """The plan is a pure function of these five values, so changing one on a resume
        relabels every remaining iteration AND shifts each corpus's data offset — the run
        would train different data on a different schedule with nothing in the logs."""
        from megatron.bridge.training.gradient_routing.callback import GRCallback
        from megatron.bridge.training.gradient_routing.optimizer_gating import GROptimizerGater

        _recorded_metrics(monkeypatch)
        n_aux = len(changed.get("aux_iter_fractions", RESUME_PLAN_KWARGS["aux_iter_fractions"]))
        model = _model_chunk(aux_ffns=AUX_FFNS[:n_aux])
        load_dir = _write_resume_checkpoint(tmp_path, 7)
        callback = GRCallback(_resume_plan(**changed), GROptimizerGater(n_aux=n_aux), log_interval=1)

        with pytest.raises(RuntimeError, match="GR plan mismatch on resume"):
            callback.on_train_start(_context(model, step=7, load_dir=load_dir))

    def test_a_resume_from_the_pre_multi_module_schema_is_refused_with_the_migration(self, monkeypatch, tmp_path):
        """A checkpoint saved under the old binary schema carries ``forget_iter_fraction``, so
        its plan cannot be rebuilt from the current builder at all. Guessing a module count
        from a scalar fraction would resume into a different experiment; every pre-migration
        run completed its plan, so the migration is to warm-start from its final checkpoint."""
        from megatron.bridge.training.gradient_routing.callback import GRCallback
        from megatron.bridge.training.gradient_routing.optimizer_gating import GROptimizerGater

        _recorded_metrics(monkeypatch)
        model = _model_chunk()
        load_dir = _write_resume_checkpoint(
            tmp_path,
            7,
            gr_section={
                "plan_seed": 4242,
                "forget_data_path": ["/data/forget_text_document"],
                "forget_iter_fraction": 0.5,
                "p_as": 0.5,
                "p_cr": 0.2,
            },
        )
        callback = GRCallback(_resume_plan(aux_iter_fractions=[0.5]), GROptimizerGater(n_aux=1), log_interval=1)

        with pytest.raises(RuntimeError, match="pre-multi-module gr schema") as excinfo:
            callback.on_train_start(_context(model, step=7, load_dir=load_dir))
        assert "pretrained_checkpoint" in str(excinfo.value), "the message must name the migration"

    def test_a_resume_without_a_checkpoint_run_config_is_refused(self, monkeypatch, tmp_path):
        """No run_config means the plan the checkpoint trained under cannot be confirmed."""
        from megatron.bridge.training.gradient_routing.callback import GRCallback
        from megatron.bridge.training.gradient_routing.optimizer_gating import GROptimizerGater

        _recorded_metrics(monkeypatch)
        model = _model_chunk(aux_ffns=AUX_FFNS)
        callback = GRCallback(_resume_plan(), GROptimizerGater(n_aux=2), log_interval=1)

        with pytest.raises(RuntimeError, match="no run_config.yaml"):
            callback.on_train_start(_context(model, step=7, load_dir=str(tmp_path)))

    def test_a_resume_from_a_non_gr_checkpoint_is_refused_with_the_warm_start_migration(self, monkeypatch, tmp_path):
        """A run_config with no gr plan fields means the checkpoint was not trained under
        gradient routing at all (e.g. checkpoint.load pointed at a control arm's save dir).
        That must be the crafted refusal naming the warm-start migration, not a bare
        KeyError out of run-config parsing."""
        from megatron.bridge.training.gradient_routing.callback import GRCallback
        from megatron.bridge.training.gradient_routing.optimizer_gating import GROptimizerGater

        _recorded_metrics(monkeypatch)
        model = _model_chunk(aux_ffns=AUX_FFNS)
        load_dir = _write_resume_checkpoint(tmp_path, 7, gr_section={})
        callback = GRCallback(_resume_plan(), GROptimizerGater(n_aux=2), log_interval=1)

        with pytest.raises(RuntimeError, match="not trained under gradient routing") as excinfo:
            callback.on_train_start(_context(model, step=7, load_dir=load_dir))
        assert "pretrained_checkpoint" in str(excinfo.value), "the message must name the migration"

    def test_a_bias_free_router_stack_is_accepted(self, monkeypatch):
        """With moe_router_enable_expert_bias off, mcore sets every router's expert_bias to
        None — there is no bias update to leak routed-corpus signal, so the freeze is
        vacuous and the run must proceed rather than being refused."""
        _recorded_metrics(monkeypatch)
        model = _model_chunk()
        for layer in model:
            layer.router.expert_bias = None
        callback = _callback()

        callback.on_train_start(_context(model, step=0))
        assert len(callback._routers) == len(model)

    def test_a_model_without_gram_layers_raises(self, monkeypatch):
        """The spec swap not running is the failure that otherwise trains happily as a
        plain CPT run while reporting itself as gradient-routed."""
        _recorded_metrics(monkeypatch)
        model = _model_chunk(gram=False)
        callback = _callback()

        with pytest.raises(RuntimeError, match="found no GRAMMoELayer"):
            callback.on_train_start(_context(model))

    def test_the_plan_digest_is_logged_at_the_starting_iteration(self, monkeypatch, tmp_path):
        """Run provenance: the digest is how a W&B run is tied back to its plan."""
        from megatron.bridge.training.gradient_routing.callback import GRCallback
        from megatron.bridge.training.gradient_routing.optimizer_gating import GROptimizerGater

        calls = _recorded_metrics(monkeypatch)
        model = _model_chunk(aux_ffns=AUX_FFNS)
        plan = _resume_plan()
        load_dir = _write_resume_checkpoint(tmp_path, 2)
        callback = GRCallback(plan, GROptimizerGater(n_aux=2), log_interval=1)

        callback.on_train_start(_context(model, step=2, load_dir=load_dir))

        (metrics, step) = calls[-1]
        assert step == 2
        assert metrics["run/gr_plan_digest_int"] == int(plan.digest()[:8], 16)


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
            assert layer.gr_gate.tolist() == [float(v) for v in FWD_AUX[iteration]]
            assert layer.router.frozen_expert_bias is (not bool(UPDATE_CORE[iteration]))

    @pytest.mark.parametrize("iteration", range(len(CORPUS_2)))
    def test_the_whole_gate_vector_follows_the_plan_row(self, monkeypatch, iteration):
        """At N>1 the write is a VECTOR, and each entry has to come from its own column: a
        "any module active" scalar would open every gate on a core-robustness iteration, which
        is the difference between making the core robust to one capability and to all of them."""
        _recorded_metrics(monkeypatch)
        model = _model_chunk(aux_ffns=AUX_FFNS)
        callback = _callback(plan=_plan_2(), n_aux=2)
        callback.on_train_start(_context(model))

        callback.on_train_step_start(_context(model, step=iteration))

        for layer in model:
            assert layer.gr_gate.tolist() == [float(v) for v in FWD_AUX_2[iteration]]
            assert layer.router.frozen_expert_bias is (not bool(UPDATE_CORE_2[iteration]))

    def test_the_gates_are_rewritten_not_accumulated_between_iterations(self, monkeypatch):
        """The buffer is copied into in place every iteration, so a stale entry from the
        previous iteration would keep a module active on a step that never routed to it."""
        _recorded_metrics(monkeypatch)
        model = _model_chunk(aux_ffns=AUX_FFNS)
        callback = _callback(plan=_plan_2(), n_aux=2)
        callback.on_train_start(_context(model))

        callback.on_train_step_start(_context(model, step=0))  # [1, 0]
        callback.on_train_step_start(_context(model, step=2))  # [0, 1]

        assert [layer.gr_gate.tolist() for layer in model] == [[0.0, 1.0]] * N_LAYERS

    def test_the_probe_hooks_cover_every_layer_and_are_removed_after(self, monkeypatch):
        """The probe used to watch ``_gram_layers[0]`` only — 1 of Nano's 23 MoE layers — so a
        module dead at layer 0 and alive at depth read as dead, and an overflow below layer 0
        was caught by nothing while ``0 * inf`` poisoned the iterations where that module is
        gated off. Coverage is asserted on EVERY layer, every module, and every module's
        ``linear_fc1`` (the liveness read); removal is asserted on all of them, because 3282
        iterations of leaked handles would make the forward more probe than model."""
        _recorded_metrics(monkeypatch)
        model = _model_chunk(aux_ffns=AUX_FFNS)
        callback = _callback(plan=_plan_2(), n_aux=2)
        callback.on_train_start(_context(model))

        callback.on_train_step_start(_context(model, step=0))
        for index, layer in enumerate(model):
            assert layer._forward_hooks, f"layer {index}: the layer-output hook is missing"
            for k in range(2):
                assert layer.gr_aux[k]._forward_hooks, f"layer {index} module {k}: no output hook"
                assert layer.gr_aux[k].linear_fc1._forward_hooks, f"layer {index} module {k}: no fc1 hook"

        callback.on_train_step_end(_context(model, step=0))
        for index, layer in enumerate(model):
            assert not layer._forward_hooks, f"layer {index}: a layer hook outlived the step"
            for k in range(2):
                assert not layer.gr_aux[k]._forward_hooks, f"layer {index} module {k}: output hook outlived"
                assert not layer.gr_aux[k].linear_fc1._forward_hooks, f"layer {index} module {k}: fc1 hook outlived"

    def test_a_step_that_never_reached_step_end_does_not_double_register(self, monkeypatch):
        """Registration is preceded by removal rather than gated on an empty handle list. An
        iteration that died mid-forward — the overflow refusal raises there, and so does an OOM
        — would otherwise leave its hooks installed for the next iteration to accumulate
        through, silently double-counting every tensor for the rest of the run."""
        _recorded_metrics(monkeypatch)
        model = _model_chunk(aux_ffns=AUX_FFNS)
        callback = _callback(plan=_plan_2(), n_aux=2)
        callback.on_train_start(_context(model))

        callback.on_train_step_start(_context(model, step=0))
        armed = (len(model[0]._forward_hooks), len(model[0].gr_aux[0]._forward_hooks))
        assert armed[0] and armed[1], "the first arm registered nothing, so this proves nothing"
        callback.on_train_step_start(_context(model, step=1))  # no step end in between

        assert (len(model[0]._forward_hooks), len(model[0].gr_aux[0]._forward_hooks)) == armed

    def test_no_probe_hook_between_log_intervals(self, monkeypatch):
        """The probe costs a device sync per logged step; log_interval must actually gate it."""
        _recorded_metrics(monkeypatch)
        model = _model_chunk()
        callback = _callback(log_interval=2)
        callback.on_train_start(_context(model))

        callback.on_train_step_start(_context(model, step=1))

        assert not model[0].gr_aux[0]._forward_hooks
        assert not model[0].gr_aux[0].linear_fc1._forward_hooks, "the liveness hook must be gated too"
        assert not model[0]._forward_hooks, "the layer-output hook must be gated too"


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

        gater.arm(update_core=bool(UPDATE_CORE[0]), update_aux=UPDATE_AUX[0])
        assert [bool(group["params"]) for group in inner.param_groups] == [True, False], (
            "iteration 0 does not update core, so the core group should be emptied"
        )

        callback.on_train_step_end(_context(model, step=0))

        assert all(group["params"] for group in inner.param_groups)
        gater.arm(update_core=True, update_aux=[True])  # a live stash would raise instead

    def test_the_per_module_groups_are_restored(self, monkeypatch):
        """At N>1 the restore has to put back EVERY emptied group: a core-robustness iteration
        empties one module's group, and missing it would freeze that module for the rest of
        the run while the logs kept reporting it as trained."""
        _recorded_metrics(monkeypatch)
        model = _model_chunk(aux_ffns=AUX_FFNS)
        gater, inner = _discovered_gater(model, n_aux=2)
        callback = _callback(gater=gater, plan=_plan_2(), n_aux=2)
        callback.on_train_start(_context(model))

        gater.arm(update_core=True, update_aux=UPDATE_AUX_2[4])  # [1, 0]
        assert [bool(group["params"]) for group in inner.param_groups] == [True, False, True]

        callback.on_train_step_end(_context(model, step=4))

        assert all(group["params"] for group in inner.param_groups)
        gater.arm(update_core=True, update_aux=[True, True])

    @pytest.mark.parametrize("iteration", range(len(CORPUS)))
    def test_the_plan_row_is_logged_verbatim(self, monkeypatch, iteration):
        calls = _recorded_metrics(monkeypatch)
        model = _model_chunk()
        callback = _callback()
        callback.on_train_start(_context(model))

        callback.on_train_step_end(_context(model, step=iteration))

        metrics, step = calls[-1]
        # Megatron logs iteration i's own metrics after incrementing train_state.step, so
        # they land on W&B step i+1; gr/* matches that or nothing joins to `lm loss`.
        assert step == iteration + 1
        assert metrics["gr/corpus"] == CORPUS[iteration]
        assert metrics["gr/fwd_aux_0"] == FWD_AUX[iteration][0]
        assert metrics["gr/update_core"] == UPDATE_CORE[iteration]
        assert metrics["gr/update_aux_0"] == UPDATE_AUX[iteration][0]
        assert metrics["gr/aux0_steps_cum"] == sum(row[0] for row in UPDATE_AUX[: iteration + 1])
        assert metrics["gr/core_steps_cum"] == sum(UPDATE_CORE[: iteration + 1])

    @pytest.mark.parametrize("iteration", range(len(CORPUS_2)))
    def test_every_modules_plan_row_is_logged_under_its_own_key(self, monkeypatch, iteration):
        """One series per module: a single aggregated ``gr/update_aux`` would average the
        modules together, and the per-module step counters are how a run's realised routing is
        reconciled against the plan it claims."""
        calls = _recorded_metrics(monkeypatch)
        model = _model_chunk(aux_ffns=AUX_FFNS)
        callback = _callback(plan=_plan_2(), n_aux=2)
        callback.on_train_start(_context(model))

        callback.on_train_step_end(_context(model, step=iteration))

        metrics, step = calls[-1]
        assert step == iteration + 1
        assert metrics["gr/corpus"] == CORPUS_2[iteration]
        assert metrics["gr/update_core"] == UPDATE_CORE_2[iteration]
        for k in range(2):
            assert metrics[f"gr/fwd_aux_{k}"] == FWD_AUX_2[iteration][k]
            assert metrics[f"gr/update_aux_{k}"] == UPDATE_AUX_2[iteration][k]
            assert metrics[f"gr/aux{k}_steps_cum"] == sum(row[k] for row in UPDATE_AUX_2[: iteration + 1])

    @pytest.mark.parametrize(
        "iteration, expected_key", [(0, "gr/loss_corpus1"), (2, "gr/loss_corpus2"), (3, "gr/loss_core")]
    )
    def test_the_loss_is_logged_under_its_corpus_key(self, monkeypatch, iteration, expected_key):
        """One curve per corpus is the whole point: a single series would average the core and
        routed losses together and hide which one is moving."""
        calls = _recorded_metrics(monkeypatch)
        model = _model_chunk(aux_ffns=AUX_FFNS)
        callback = _callback(plan=_plan_2(), n_aux=2)
        callback.on_train_start(_context(model))

        loss = torch.tensor(1.5, device="cuda")
        callback.on_train_step_end(_context(model, step=iteration, loss_dict={"lm loss": loss}))

        metrics = calls[-1][0]
        assert metrics[expected_key] == pytest.approx(1.5)
        loss_keys = {key for key in metrics if key.startswith("gr/loss_")}
        assert loss_keys == {expected_key}, f"an iteration logged more than its own corpus: {loss_keys}"

    def test_the_aux_output_rms_probe_reports_a_real_forward(self, monkeypatch):
        """Zero-init aux + gate 1 still produces a zero output — the probe reads each aux
        module's own output, so it is 0.0 until that module has actually trained."""
        calls = _recorded_metrics(monkeypatch)
        model = _model_chunk(n_layers=1, aux_ffns=AUX_FFNS)
        callback = _callback(plan=_plan_2(), n_aux=2)
        callback.on_train_start(_context(model))
        callback.on_train_step_start(_context(model, step=0))

        with torch.no_grad():
            model[0](torch.randn(SEQ, BATCH, HIDDEN, dtype=torch.bfloat16, device="cuda"))
        callback.on_train_step_end(_context(model, step=0))

        for k in range(2):
            assert calls[-1][0][f"gr/aux{k}_out_rms"] == pytest.approx(0.0)

    def test_the_output_rms_is_reported_per_module(self, monkeypatch):
        """Each module gets its own series, read off its own forward: one shared key would
        report whichever module ran last, and a diverging module would be invisible."""
        calls = _recorded_metrics(monkeypatch)
        model = _model_chunk(n_layers=1, aux_ffns=AUX_FFNS)
        callback = _callback(plan=_plan_2(), n_aux=2)
        callback.on_train_start(_context(model))
        callback.on_train_step_start(_context(model, step=0))

        # Only module 1 is trained, so the two series must differ: 0 for module 0, > 0 for 1.
        torch.manual_seed(5)
        with torch.no_grad():
            model[0].gr_aux[1].linear_fc2.weight.normal_(0.0, 0.5)
            model[0](torch.randn(SEQ, BATCH, HIDDEN, dtype=torch.bfloat16, device="cuda"))
        callback.on_train_step_end(_context(model, step=0))

        metrics = calls[-1][0]
        assert metrics["gr/aux0_out_rms"] == pytest.approx(0.0)
        assert metrics["gr/aux1_out_rms"] > 0.0

    def test_the_aux_parameter_norm_is_logged_per_module(self, monkeypatch):
        calls = _recorded_metrics(monkeypatch)
        model = _model_chunk(aux_ffns=AUX_FFNS)
        callback = _callback(plan=_plan_2(), n_aux=2)
        callback.on_train_start(_context(model))

        callback.on_train_step_end(_context(model, step=0))

        metrics = calls[-1][0]
        for k in range(2):
            expected = (
                sum(float(p.detach().float().pow(2).sum()) for layer in model for p in layer.gr_aux[k].parameters())
                ** 0.5
            )
            assert metrics[f"gr/aux{k}_param_norm"] == pytest.approx(expected, rel=1e-5)
        assert metrics["gr/aux0_param_norm"] != metrics["gr/aux1_param_norm"], (
            "the two modules have different widths, so equal norms mean one module was measured twice"
        )


@requires_gpu
@pytest.mark.usefixtures("moe_parallel_state")
class TestAuxDepthProbe:
    """The probe reports a DEPTH profile, not a one-layer, last-microbatch sample.

    An arm shipped with layer-0 module output RMS ~0.0002 — dead at layer 0 — while its
    all-on probe still moved a topic's loss by 0.024: its modules were alive at depth and
    invisible to telemetry, and every magnitude argument in that investigation rested on the
    blind sample. The quantity that actually governs the failure, the summed module
    contribution measured against what the modules were handed, was never logged at all.

    A caveat this rig CANNOT check, recorded so nobody mistakes green tests for evidence: the
    tests below call the GRAM layer directly with a bare tensor, so in-test ``x`` is both the
    layer input and the residual. In the real stack it is neither the same tensor nor the same
    magnitude — mcore hands the layer ``pre_mlp_layernorm(hidden_states)`` and adds its write
    into ``hidden_states`` afterwards — which is why the keys are spelled ``_in_ratio_*`` and
    not ``_res_ratio_*``.
    """

    def test_the_magnitude_is_measured_at_every_depth_not_only_layer_zero(self, monkeypatch):
        """The failure shape above, reproduced: a module trained only at DEPTH. The layer-0
        series must read zero (it is kept for continuity, so it must keep being blind here) and
        the all-layer series must see the module, naming the depth."""
        calls = _recorded_metrics(monkeypatch)
        model = _model_chunk(aux_ffns=AUX_FFNS)  # mcore layer_number 1 and 2
        callback = _callback(plan=_plan_2(), n_aux=2)
        callback.on_train_start(_context(model))
        _train_aux(model[-1], module=0)  # alive at the LAST layer only

        _probed_step(callback, model, step=0)

        metrics = calls[-1][0]
        assert metrics["gr/aux0_out_rms"] == pytest.approx(0.0), "the layer-0 series must be blind here"
        assert metrics["gr/aux0_out_rms_all"] > 0.0, "the all-layer series must see the trained depth"
        assert metrics["gr/aux0_out_rms_min"] == pytest.approx(0.0)
        assert metrics["gr/aux0_out_rms_min_layer"] == 1
        assert metrics["gr/aux0_out_rms_max"] > 0.0
        assert metrics["gr/aux0_out_rms_max_layer"] == N_LAYERS, "the key must carry mcore's layer_number"
        assert metrics["gr/aux0_in_ratio_max"] > 0.0
        assert "run/gr_probe_layers" not in metrics, "probe coverage is logged at train start"

    def test_the_magnitude_pools_the_microbatches_instead_of_keeping_the_last(self, monkeypatch):
        """One ``.item()`` per microbatch overwrote its predecessor, so the series reported the
        LAST microbatch of an optimizer step. A module that fires on one microbatch and not the
        next is not "off"; the pooled RMS is the only reading that says so. The expectation is
        algebraic — sums of squares over element counts — not a tolerance band."""
        calls = _recorded_metrics(monkeypatch)
        model = _model_chunk(n_layers=1, aux_ffns=AUX_FFNS)
        callback = _callback(plan=_plan_2(), n_aux=2)
        callback.on_train_start(_context(model))
        _train_aux(model[0], module=0)

        _probed_step(callback, model, step=0, scale=1.0)
        big = calls[-1][0]["gr/aux0_out_rms_all"]
        _probed_step(callback, model, step=0, scale=0.25)
        small = calls[-1][0]["gr/aux0_out_rms_all"]
        assert big > 0.0 and small > 0.0 and big > 2 * small, "the microbatches must differ clearly"

        callback.on_train_step_start(_context(model, step=0))
        _forward(model, scale=1.0)
        _forward(model, scale=0.25)
        callback.on_train_step_end(_context(model, step=0))

        pooled = calls[-1][0]["gr/aux0_out_rms_all"]
        assert pooled == pytest.approx(((big**2 + small**2) / 2) ** 0.5, rel=1e-3)
        assert pooled != pytest.approx(small, rel=1e-2), "a last-microbatch overwrite would report this"

    def test_the_summed_contribution_is_reported_against_the_layer_input(self, monkeypatch):
        """The magnitude that governs a composability failure, and the one nothing ever logged.

        The denominator is the layer INPUT, not the layer output: ``GRAMMoELayer.forward``
        returns ``core + sum_k gate_k * aux_k(h)``, so a ratio against the output contains its
        own numerator, saturates near 1, and can never say "too large". The layer input is what
        the modules were handed — on the real stack the ``pre_mlp_layernorm`` output, NOT the
        residual stream — which is what the key name has to say.
        """
        calls = _recorded_metrics(monkeypatch)
        model = _model_chunk(aux_ffns=AUX_FFNS)
        callback = _callback(plan=_plan_2(), n_aux=2)
        callback.on_train_start(_context(model))
        _train_aux(model[-1], module=0)  # only the DEEP layer contributes
        assert FWD_AUX_2[0] == [1, 0], "iteration 0 must open module 0's gate alone"

        x = _probed_step(callback, model, step=0)

        with torch.no_grad():  # the aux MLP alone — no router state, nothing to re-run
            expected = float(model[-1].gr_aux[0](x).float().norm() / x.float().norm())
        metrics = calls[-1][0]
        assert expected > 0.0, "an untrained module would make every ratio 0 and prove nothing"
        assert metrics["gr/aux_contrib_in_ratio_max"] == pytest.approx(expected, rel=1e-2)
        assert metrics["gr/aux_contrib_in_ratio_max_layer"] == N_LAYERS
        assert metrics["gr/aux_contrib_in_ratio_mean"] == pytest.approx(expected / N_LAYERS, rel=1e-2)
        assert metrics["gr/aux_contrib_out_share_max"] > 0.0

    def test_a_gate_closed_iteration_reports_a_zero_contribution(self, monkeypatch):
        """Every gate shut: the contribution is exactly zero by contract. The modules still
        EXECUTE — Megatron's DDP buckets need a grad from every parameter every microbatch — so
        their own magnitude must still be measured. A clean 0.0, not a 0/0 NaN, not a missing
        key."""
        calls = _recorded_metrics(monkeypatch)
        model = _model_chunk(n_layers=1, aux_ffns=AUX_FFNS)
        callback = _callback(plan=_plan_2(), n_aux=2)
        callback.on_train_start(_context(model))
        _train_aux(model[0], module=0)
        assert FWD_AUX_2[3] == [0, 0], "iteration 3 must be the all-gates-closed core iteration"

        _probed_step(callback, model, step=3)

        metrics = calls[-1][0]
        assert metrics["gr/aux_contrib_in_ratio_max"] == 0.0
        assert metrics["gr/aux_contrib_out_share_max"] == 0.0
        assert metrics["gr/aux0_out_rms_all"] > 0.0, "a gated-off module still runs and must be measured"
        # argmax over an all-zero vector returns index 0, so a depth label here would read in a
        # plot as a genuine peak at the shallowest GRAM layer — on the majority of a real plan's
        # iterations. The zero itself is real and stays; the meaningless label is omitted.
        assert "gr/aux_contrib_in_ratio_max_layer" not in metrics

    def test_a_probe_fold_failure_keeps_the_plan_telemetry_and_the_run(self, monkeypatch, caplog):
        """The probe was added to make failures visible; it must not be able to hide any.

        The fold now runs on the training path (it carries the collapse detector, which cannot
        live behind ``wandb.run``), so it has two ways to do damage and is allowed neither:
        raising would kill a 4-5 h run over telemetry, and raising inside the emitter's blanket
        ``except`` would silently drop ``gr/corpus``, the per-corpus loss and every step counter
        for that iteration behind one generic warning line.
        """
        calls = _recorded_metrics(monkeypatch)
        model = _model_chunk(aux_ffns=AUX_FFNS)
        callback = _callback(plan=_plan_2(), n_aux=2)
        callback.on_train_start(_context(model))

        def _boom(self, records, it):
            raise RuntimeError("a probe defect")

        monkeypatch.setattr(type(callback), "_probe_metrics", _boom)

        with caplog.at_level("WARNING"):
            _probed_step(callback, model, step=0)

        metrics = calls[-1][0]
        assert metrics["gr/corpus"] == CORPUS_2[0], "the routing telemetry must survive the probe"
        assert metrics["gr/aux0_steps_cum"] == UPDATE_AUX_2[0][0]
        assert metrics["gr/aux0_param_norm"] > 0.0, "the deferred payload must survive it too"
        assert "gr/aux0_out_rms_all" not in metrics
        assert [r for r in caplog.records if "magnitude/liveness fold failed" in r.getMessage()]

    def test_the_probe_leaves_the_forward_and_the_gradients_bitwise_unchanged(self, monkeypatch):
        """The campaign's claims rest on bitwise-identical trajectories wherever the plan says
        nothing changed, so the probe has to be a pure read: hooks returning None, every tensor
        detached under no_grad, no RNG drawn, no dtype changed on the training path. The gate is
        OPEN on purpose — that is the case where the probe observes the very tensor the layer
        then adds. Two separately-built chunks from the same seeds, so no router or expert-bias
        state can carry from one forward into the other."""
        from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

        _recorded_metrics(monkeypatch)
        reference = _model_chunk(n_layers=1, aux_ffns=AUX_FFNS)
        probed = _model_chunk(n_layers=1, aux_ffns=AUX_FFNS)
        probed_params = dict(probed.named_parameters())
        for name, p in reference.named_parameters():
            assert torch.equal(p, probed_params[name]), f"{name} differs before training — no baseline"
        # on_train_start FIRST: at step 0 it asserts the fc2 zero-init, so training the modules
        # before it would raise there and this test would never reach an assertion.
        callback = _callback(plan=_plan_2(), n_aux=2)
        callback.on_train_start(_context(probed))
        for k in range(2):
            _train_aux(reference[0], module=k, seed=5 + k)
            _train_aux(probed[0], module=k, seed=5 + k)

        torch.manual_seed(7)
        x = torch.randn(SEQ, BATCH, HIDDEN, dtype=torch.bfloat16, device="cuda")
        torch.manual_seed(11)
        upstream = torch.randn(SEQ, BATCH, HIDDEN, dtype=torch.bfloat16, device="cuda")
        gate = torch.tensor(FWD_AUX_2[0], dtype=reference[0].gr_gate.dtype, device="cuda")

        reference[0].gr_gate.copy_(gate)
        model_parallel_cuda_manual_seed(1234)
        reference[0](x)[0].backward(upstream)

        callback.on_train_step_start(_context(probed, step=0))  # same gate row, hooks ON
        model_parallel_cuda_manual_seed(1234)
        out_probed = probed[0](x)[0]
        out_probed.backward(upstream)
        callback.on_train_step_end(_context(probed, step=0))

        model_parallel_cuda_manual_seed(1234)
        assert torch.equal(reference[0](x)[0], out_probed), "the probed forward moved"
        for name, p in reference.named_parameters():
            other = probed_params[name]
            assert (p.grad is None) == (other.grad is None), f"{name}: gradient presence differs"
            if p.grad is not None:
                assert torch.equal(p.grad, other.grad), f"{name} gradient moved under the probe"


@requires_gpu
@pytest.mark.usefixtures("moe_parallel_state")
class TestAuxLiveness:
    """A module that collapses to exactly zero must be visible; the legitimate zeros must not.

    NemotronH's activation is squared ReLU, whose gradient is identically zero for all-negative
    pre-activations, so one oversized fc1 step kills a layer's copy of a module permanently and
    irrecoverably — and the run then finishes with a full iteration count, 0 NaN and a valid
    checkpoint, its composability "pass" vacuous because a module that outputs nothing cannot
    interfere. The counterweight is that TWO kinds of zero are correct by design: the
    ``gr_aux_output_init="zero"`` start, and a specialised module being quiet on a corpus that
    is not its own. A detector that fired on either would be switched off within a day.
    """

    def test_a_zero_init_module_reads_as_alive_at_every_layer(self, monkeypatch, caplog):
        """The false alarm that would kill every warm start. ``gr_aux_output_init="zero"``
        zeroes fc2, and ``on_train_start`` ASSERTS it at iteration 0, so a healthy fresh
        module's OUTPUT is exactly zero. Liveness is read on the INPUT side of the squared ReLU,
        which that init never touches — which is why no warm-up window is needed."""
        calls = _recorded_metrics(monkeypatch)
        model = _model_chunk(aux_ffns=AUX_FFNS)
        callback = _callback(plan=_plan_2(), n_aux=2)
        callback.on_train_start(_context(model))

        with caplog.at_level("WARNING"):
            _probed_step(callback, model, step=0)  # module 0's own corpus, fc2 still at zero

        metrics = calls[-1][0]
        for k in range(2):
            assert metrics[f"gr/aux{k}_out_rms_all"] == pytest.approx(0.0), "fc2 is still at its zero init"
            assert metrics[f"gr/aux{k}_live_layers"] == N_LAYERS
        assert not [r for r in caplog.records if "is DEAD at" in r.getMessage()], "the zero-init start warned"

    def test_a_module_killed_at_one_depth_is_reported_and_warned_about(self, monkeypatch, caplog):
        """Partial death is the shape that matters and the one a pooled statistic hides: a
        module can be frozen at some depths and training at others, and the run reports 0 NaN
        either way. The sibling must not be charged with it."""
        calls = _recorded_metrics(monkeypatch)
        model = _model_chunk(aux_ffns=AUX_FFNS)
        callback = _callback(plan=_plan_2(), n_aux=2)
        callback.on_train_start(_context(model))
        _kill_squared_relu(model[-1], module=0)
        assert CORPUS_2[0] == 1, "iteration 0 must draw module 0's OWN corpus"

        with caplog.at_level("WARNING"):
            _probed_step(callback, model, step=0)

        metrics = calls[-1][0]
        assert metrics["gr/aux0_live_layers"] == N_LAYERS - 1
        assert metrics["gr/aux1_live_layers"] == N_LAYERS, "a sibling must not be charged with it"
        assert [r for r in caplog.records if "gr_aux.0 is DEAD" in r.getMessage()]

    def test_liveness_is_not_judged_off_the_modules_own_corpus(self, monkeypatch, caplog):
        """A specialised module being quiet on core text is routing WORKING, not a defect, and
        its pre-activations there are a different distribution entirely. The METRIC is still
        reported on every logged iteration; only the warning is scoped, or a healthy run would
        warn on core iterations."""
        calls = _recorded_metrics(monkeypatch)
        model = _model_chunk(aux_ffns=AUX_FFNS)
        callback = _callback(plan=_plan_2(), n_aux=2)
        callback.on_train_start(_context(model))
        for layer in model:
            _kill_squared_relu(layer, module=0)
        assert CORPUS_2[3] == 0, "iteration 3 must be a core iteration"

        with caplog.at_level("WARNING"):
            _probed_step(callback, model, step=3)

        assert calls[-1][0]["gr/aux0_live_layers"] == 0, "the metric must still report it"
        assert not [r for r in caplog.records if "gr_aux.0 is DEAD" in r.getMessage()]

    def test_a_module_that_keeps_one_unit_alive_is_still_reported_as_collapsing(self, monkeypatch, caplog):
        """The failure a whole-tensor maximum cannot see, and the reason the read is per unit.

        Squared-ReLU death is per UNIT: row j of fc1 is frozen iff unit j's pre-activation is
        non-positive on every token. So a copy with 7 of its 8 units permanently dead still has
        a positive tensor maximum and reports ``live_layers == n_layers`` — a clean bill of
        health for a module that has lost 7/8 of its capacity, which is much closer to the arm
        that motivated this probe than total death is. ``live_frac_min`` is what names it, and
        the warning must fire on it.
        """
        calls = _recorded_metrics(monkeypatch)
        model = _model_chunk(aux_ffns=AUX_FFNS)
        callback = _callback(plan=_plan_2(), n_aux=2)
        callback.on_train_start(_context(model))
        for layer in model:  # every copy fully live...
            for k, width in enumerate(AUX_FFNS):
                _pin_fc1_units(layer, module=k, live=width)
        _pin_fc1_units(model[-1], module=0, live=1)  # ...except one, down to a single unit
        assert CORPUS_2[0] == 1, "iteration 0 must draw module 0's OWN corpus"

        with caplog.at_level("WARNING"):
            _probed_step(callback, model, step=0, positive=True)

        metrics = calls[-1][0]
        assert metrics["gr/aux0_live_layers"] == N_LAYERS, "a whole-tensor max reads this as healthy"
        assert metrics["gr/aux0_live_frac_min"] == pytest.approx(1.0 / AUX_FFNS[0])
        assert metrics["gr/aux0_live_frac_min_layer"] == N_LAYERS, "the key must carry mcore's layer_number"
        assert metrics["gr/aux1_live_frac_min"] == pytest.approx(1.0), "a sibling must not be charged with it"
        assert [r for r in caplog.records if "gr_aux.0 has COLLAPSED" in r.getMessage()]
        assert not [r for r in caplog.records if "is DEAD at" in r.getMessage()], "no copy died outright"

    def test_the_collapse_warning_does_not_depend_on_a_live_wandb_run(self, monkeypatch, caplog):
        """GAP 2's alarm is a training-LOG line, so it must not sit behind another subsystem.

        The real emitter returns at ``if wandb.run is None`` BEFORE evaluating a thunk, so
        anything computed inside one exists on the single W&B rank and only when W&B actually
        initialised — on every other rank, and on any run whose ``logger.wandb_project`` is
        unset or whose wandb init failed, a module could die and the run would still ship 0
        NaN, a full iteration count and a valid checkpoint. The emitter is stubbed to drop its
        payload WITHOUT evaluating it, which is exactly what those ranks do.
        """
        from megatron.bridge.training.gradient_routing import callback as callback_module

        monkeypatch.setattr(callback_module, "log_wandb_metrics_nonfatal", lambda metrics, step: None)
        model = _model_chunk(aux_ffns=AUX_FFNS)
        callback = _callback(plan=_plan_2(), n_aux=2)
        callback.on_train_start(_context(model))
        for layer in model:
            _kill_squared_relu(layer, module=0)

        with caplog.at_level("WARNING"):
            _probed_step(callback, model, step=0)

        assert [r for r in caplog.records if "gr_aux.0 is DEAD" in r.getMessage()]

    def test_the_collapse_warning_is_announced_once_not_every_iteration(self, monkeypatch, caplog):
        """The condition is permanent — that copy of the module can never recover — so a
        per-iteration warning would emit thousands of identical lines and train operators to
        filter out exactly the message that matters."""
        _recorded_metrics(monkeypatch)
        model = _model_chunk(aux_ffns=AUX_FFNS)
        callback = _callback(plan=_plan_2(), n_aux=2)
        callback.on_train_start(_context(model))
        for layer in model:
            _kill_squared_relu(layer, module=0)
        assert CORPUS_2[0] == CORPUS_2[1] == 1, "both iterations must draw module 0's own corpus"

        with caplog.at_level("WARNING"):
            _probed_step(callback, model, step=0)
            _probed_step(callback, model, step=1)

        assert len([r for r in caplog.records if "gr_aux.0 is DEAD" in r.getMessage()]) == 1


@requires_gpu
@pytest.mark.usefixtures("moe_parallel_state")
class TestExpertBiasFreezeReachesMegatron:
    """The router's expert bias is core state the optimizer gate cannot reach.

    Megatron updates ``expert_bias`` inside grad finalization — outside the optimizer — so
    emptying param groups does not touch it, and the callback's per-iteration
    ``frozen_expert_bias`` flag is the only thing stopping an aux-isolated iteration from
    writing routed-corpus routing statistics into the core model. Assigning an attribute to a
    Module always "succeeds", so an upstream rename would leave the callback silently writing
    a flag nobody reads: the flag is therefore driven through Megatron's REAL updater here,
    not merely asserted to have been set.
    """

    def _bias_update(self, model, step, monkeypatch):
        """Run one iteration's gate/freeze write, then Megatron's own expert-bias update."""
        from megatron.core import parallel_state
        from megatron.core.distributed.finalize_model_grads import _update_router_expert_bias

        _recorded_metrics(monkeypatch)
        config = moe_config()
        callback = _callback()
        callback.on_train_start(_context(model))
        callback.on_train_step_start(_context(model, step=step))

        # Grad-enabled on purpose: mcore accumulates local_tokens_per_expert only when
        # torch.is_grad_enabled(), so an inference-mode forward leaves the counts at zero and
        # there would be no bias update for the freeze to suppress.
        torch.manual_seed(7)
        x = torch.randn(SEQ, BATCH, HIDDEN, dtype=torch.bfloat16, device="cuda")
        for layer in model:
            layer(x)
        counts = model[0].router.local_tokens_per_expert
        assert not bool((counts == counts[0]).all()), (
            f"the router balanced perfectly ({counts.tolist()}), so no bias update is due and "
            "neither arm of this comparison would move"
        )

        before = [layer.router.expert_bias.detach().clone() for layer in model]
        _update_router_expert_bias(
            [model],
            config,
            tp_dp_cp_group=parallel_state.get_tensor_and_data_parallel_group(with_context_parallel=True),
        )
        return before, [layer.router.expert_bias for layer in model]

    def test_an_aux_isolated_iteration_leaves_the_expert_bias_untouched(self, monkeypatch):
        """Iteration 0 does not update core, so the router bias must not move either."""
        model = _model_chunk(n_layers=1)
        before, after = self._bias_update(model, step=0, monkeypatch=monkeypatch)
        assert not bool(UPDATE_CORE[0]), "iteration 0 must be a core-frozen iteration for this to test anything"
        for index, (old, new) in enumerate(zip(before, after)):
            assert torch.equal(old, new), f"layer {index}: expert bias moved on a core-frozen iteration"

    def test_a_core_iteration_lets_the_expert_bias_move(self, monkeypatch):
        """The control: without the freeze the same rig does update the bias, so the test
        above is pinning the flag rather than an inert code path."""
        model = _model_chunk(n_layers=1)
        assert bool(UPDATE_CORE[2]), "iteration 2 must be a core-updating iteration"
        before, after = self._bias_update(model, step=2, monkeypatch=monkeypatch)
        assert any(not torch.equal(old, new) for old, new in zip(before, after)), (
            "expert bias did not move on a core iteration"
        )

    def test_an_expert_bias_carrier_outside_the_gram_stack_is_refused(self, monkeypatch):
        """Megatron updates the bias on EVERY module carrying one, while the callback can only
        freeze the routers it collected — so a carrier outside the swapped spec would keep
        learning from a routed corpus on exactly the iterations the core is frozen."""
        _recorded_metrics(monkeypatch)
        model = _model_chunk(n_layers=1)
        stray = torch.nn.Module()
        stray.expert_bias = torch.zeros(4, device="cuda")
        model.append(stray)

        with pytest.raises(RuntimeError, match="carry expert_bias"):
            _callback().on_train_start(_context(model))


@requires_gpu
@pytest.mark.usefixtures("moe_parallel_state")
class TestAuxOverflowProbe:
    """A non-finite aux output must stop the run, including on gate-0 iterations.

    ``gate * aux(h)`` is bitwise core-only for FINITE aux outputs — but ``0 * inf`` is NaN, so
    a diverged aux module corrupts the core iterations where it is supposed to be inert.
    The failure would surface as a NaN loss on a core-only step with the aux module looking
    entirely uninvolved, which is why the probe raises with the aux LR in the message.
    """

    def test_a_gated_off_aux_still_poisons_the_output_when_it_overflows(self):
        """The hazard the probe exists for, shown on the layer itself: gate 0 does not save a
        forward whose aux output is non-finite."""
        model = _model_chunk(n_layers=1)
        with torch.no_grad():
            model[0].gr_aux[0].linear_fc2.weight.fill_(float("inf"))
        model[0].gr_gate.zero_()

        torch.manual_seed(7)
        x = torch.randn(SEQ, BATCH, HIDDEN, dtype=torch.bfloat16, device="cuda")
        with torch.no_grad():
            output, _ = model[0](x)
        assert not bool(torch.isfinite(output).all()), "0 * inf did not propagate — the probe would be unnecessary"

    @pytest.mark.parametrize("module", [0, 1])
    def test_the_probe_refuses_a_non_finite_aux_output_from_any_module(self, monkeypatch, module):
        _recorded_metrics(monkeypatch)
        model = _model_chunk(n_layers=1, aux_ffns=AUX_FFNS)
        callback = _callback(plan=_plan_2(), n_aux=2)
        callback.on_train_start(_context(model))
        with torch.no_grad():
            model[0].gr_aux[module].linear_fc2.weight.fill_(float("inf"))

        callback.on_train_step_start(_context(model, step=3))  # a core iteration: every gate 0
        assert model[0].gr_gate.tolist() == [0.0, 0.0]

        torch.manual_seed(7)
        x = torch.randn(SEQ, BATCH, HIDDEN, dtype=torch.bfloat16, device="cuda")
        with pytest.raises(RuntimeError, match=f"gr_aux.{module} output is non-finite"):
            with torch.no_grad():
                model[0](x)

    def test_the_probe_refuses_a_non_finite_aux_output_below_the_first_layer(self, monkeypatch):
        """The old probe watched ``_gram_layers[0]`` only, so an aux that overflowed at any
        other depth passed silently while ``0 * inf`` poisoned every iteration where that module
        is gated off. The refusal must stay INSIDE the forward: mcore's own fatal isnan check on
        the loss raises from within the train step, ahead of ``on_train_step_end``, so a check
        deferred to step end would never run and the aux attribution would be replaced by a
        generic NaN-loss message."""
        _recorded_metrics(monkeypatch)
        model = _model_chunk(aux_ffns=AUX_FFNS)
        callback = _callback(plan=_plan_2(), n_aux=2)
        callback.on_train_start(_context(model))
        with torch.no_grad():
            model[-1].gr_aux[1].linear_fc2.weight.fill_(float("inf"))

        callback.on_train_step_start(_context(model, step=3))  # a core iteration: every gate 0
        assert model[-1].gr_gate.tolist() == [0.0, 0.0]

        # The module index and the DEPTH are what this test asserts; the wording between them
        # is free to describe whichever non-finiteness the probe caught.
        with pytest.raises(RuntimeError, match=rf"gr_aux\.1 output is non-finite.*at MoE layer {N_LAYERS}"):
            _forward(model)


@requires_gpu
@pytest.mark.usefixtures("moe_parallel_state")
class TestStandardInitMode:
    def test_standard_init_mode_skips_the_zero_check_at_iteration_zero(self, monkeypatch):
        """``gr_aux_output_init="standard"`` has no zero invariant: a randomly initialised
        fc2 is legitimate at iteration 0, so the clobber check must not fire (and cannot
        protect anything — that is the trade the mode makes)."""
        _recorded_metrics(monkeypatch)
        model = _model_chunk()
        for layer in model:
            layer.config.gr_aux_output_init = "standard"
        with torch.no_grad():
            model[-1].gr_aux[0].linear_fc2.weight.fill_(1e-3)
        callback = _callback()

        callback.on_train_start(_context(model))


@requires_gpu
@pytest.mark.usefixtures("moe_parallel_state")
class TestGatingLeakDetection:
    """on_train_step_end must raise when a module's Adam step counter disagrees with the
    plan's armed-update count, and must announce (not silently skip) verification on an
    optimizer without group counters."""

    def _stepped_callback_context(self, step_values):
        """A callback wired to a discovered gater whose aux group carries the given
        group-level step counter, driven through one full step at iteration 0."""
        model = _model_chunk()
        gater, inner = _discovered_gater(model)
        for group, value in zip(inner.param_groups, step_values):
            if value is not None:
                # torch.optim groups accept arbitrary keys; injecting `step` reproduces
                # the TE/apex FusedAdam group layout on an otherwise real optimizer.
                group["step"] = value
        callback = _callback(gater=gater)
        return model, callback

    def test_a_counter_matching_the_armed_count_passes(self, monkeypatch):
        _recorded_metrics(monkeypatch)
        # plan `_plan()` arms module 0 at iteration 0, so its cumulative count is 1.
        model, callback = self._stepped_callback_context([1, None])
        callback.on_train_start(_context(model))
        callback.on_train_step_end(_context(model, step=0))

    def test_a_counter_disagreeing_with_the_armed_count_raises(self, monkeypatch):
        _recorded_metrics(monkeypatch)
        model, callback = self._stepped_callback_context([5, None])
        callback.on_train_start(_context(model))
        with pytest.raises(RuntimeError, match="GR gating leak"):
            callback.on_train_step_end(_context(model, step=0))

    def test_an_optimizer_without_counters_logs_once_instead_of_verifying(self, monkeypatch, caplog):
        _recorded_metrics(monkeypatch)
        model = _model_chunk()
        gater, _ = _discovered_gater(model)
        callback = _callback(gater=gater)
        callback.on_train_start(_context(model))
        with caplog.at_level("INFO"):
            callback.on_train_step_end(_context(model, step=0))
            callback.on_train_step_end(_context(model, step=1))
        announcements = [r for r in caplog.records if "verification is inactive" in r.getMessage()]
        assert len(announcements) == 1
