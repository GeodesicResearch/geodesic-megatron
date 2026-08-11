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
"""Numerical pinning for the GRAM auxiliary modules and their spec swap.

The central claim of GRAM's model surgery is a NEGATIVE one: gated-off aux modules change
nothing. Every consumer that does not drive the gates — evaluation, HF export, coherence
testing, and the whole core-corpus half of training — depends on it, and a violation would
show up as unexplained quality drift rather than as an error. So the comparison here is
``torch.equal`` against a separately-built vanilla ``MoELayer`` from the same seed, not a
tolerance: "numerically indistinguishable" is exactly the claim, and bf16 tolerances would
hide a real (small) contribution.

With N modules the additivity claim gets a second half: an OPEN gate must add exactly its
own module's output and nothing else, because each export posture merges an arbitrary SUBSET
of the modules into the shared expert. So the multi-module tests drive one gate at a time
and compare against that module's own forward.

Real GPU, real TE linears, real router and dispatcher: the aux modules are spliced into the
MoE layer's forward, so a mocked layer would test the splice against nothing.

``stack_spec`` (in the shared helpers) deep-copies. ``provider.mambastack_spec(provider)``
returns a PROCESS-GLOBAL ``ModuleSpec`` — the same object on every call, from every
provider instance — so a test that edited one in place would silently rewrite the spec
every later test builds from. (That global is also why both spec swaps deepcopy; see
``test_swap_does_not_mutate_the_caller_spec``.)
"""

from functools import partial

import pytest
import torch

from tests.unit_tests.gr_test_utils import (
    AUX_FFN,
    AUX_FFNS,
    HIDDEN,
    SHARED_FFN,
    build_moe_layer,
    gram_spec,
    init_model_parallel,
    moe_builder,
    moe_config,
    stack_spec,
    teardown_model_parallel,
)


requires_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU (real TE MoE layer)")

SEQ, BATCH = 6, 2


@pytest.fixture(scope="module")
def moe_parallel_state():
    """Real world-1 mcore parallel state; MoELayer picks the default pg_collection up from it."""
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    init_model_parallel(expert_model_parallel_size=1)
    model_parallel_cuda_manual_seed(1234)
    yield
    teardown_model_parallel()


def _render(builder):
    """A comparable rendering of a spec builder (functools.partial has no __eq__)."""
    return (builder.func, builder.args, builder.keywords) if isinstance(builder, partial) else builder


def _pair(config=None, aux_ffns=(AUX_FFN,)):
    """A vanilla MoELayer and a GRAM-swapped one, seeded identically."""
    config = config if config is not None else moe_config()
    return (
        build_moe_layer(moe_builder(stack_spec()), config),
        build_moe_layer(moe_builder(gram_spec(aux_ffns)), config),
    )


def _input(seed=7, requires_grad=False):
    torch.manual_seed(seed)
    return torch.randn(SEQ, BATCH, HIDDEN, dtype=torch.bfloat16, device="cuda", requires_grad=requires_grad)


def _opaque_shared_expert_builder(*, config, pg_collection, gate, name=None):
    """A working shared-expert builder that carries no ``submodules`` keyword to inspect."""
    from megatron.core.transformer.moe.shared_experts import SharedExpertMLP

    submodules = moe_builder(stack_spec()).keywords["submodules"].shared_experts.keywords["submodules"]
    return SharedExpertMLP(config=config, submodules=submodules, gate=gate, pg_collection=pg_collection, name=name)


def _core_params(layer):
    """Every parameter that is NOT an aux module, by name.

    Matches on the ``gr_aux.`` path segment rather than the optimizer's ``.gr_aux.``
    fragment: these layers are built standalone, so the aux parameters have no
    ``decoder.layers.N.mlp.`` prefix in front of them the way they do inside a model.
    """
    return {name: p for name, p in layer.named_parameters() if not name.startswith("gr_aux.")}


def _static_gate_config(static_gates, **overrides):
    """A config carrying ``gr_static_gates``, set the way a real run carries it.

    In production the layer's config IS the model provider, which declares the field; here
    the field is attached to a plain ``TransformerConfig`` because the layer reads it off
    whatever config it is handed.
    """
    config = moe_config(**overrides)
    config.gr_static_gates = static_gates
    return config


def _train_aux(layer, module=None, scale=0.05, seed=99):
    """Give the aux output projections a non-zero value, as training would.

    ``module=None`` trains every module; an index trains only that one, which is what lets a
    single-gate assertion attribute the contribution it sees.
    """
    torch.manual_seed(seed)
    targets = range(len(layer.gr_aux)) if module is None else [module]
    with torch.no_grad():
        for k in targets:
            layer.gr_aux[k].linear_fc2.weight.normal_(0.0, scale)


@requires_gpu
@pytest.mark.usefixtures("moe_parallel_state")
class TestGatedOffIsIdentity:
    """Gate 0 (the default, and every non-driven consumer) must be bit-for-bit the core model."""

    @pytest.mark.parametrize("aux_ffns", [(AUX_FFN,), AUX_FFNS])
    def test_shared_parameters_are_identical_across_the_pair(self, aux_ffns):
        """Guards the comparison itself: if the swap perturbed initialisation order, the
        output equalities below would be testing two different models."""
        vanilla, gram = _pair(aux_ffns=aux_ffns)
        core = _core_params(gram)
        assert set(core) == set(dict(vanilla.named_parameters()))
        for name, p in vanilla.named_parameters():
            assert torch.equal(p, core[name]), f"{name} differs between the vanilla and GRAM layers"

    @pytest.mark.parametrize("aux_ffns", [(AUX_FFN,), AUX_FFNS])
    def test_default_gate_is_a_zero_vector_with_one_entry_per_module(self, aux_ffns):
        _, gram = _pair(aux_ffns=aux_ffns)
        assert gram.gr_gate.shape == (len(aux_ffns),)
        assert gram.gr_gate.dtype == torch.bfloat16
        assert torch.all(gram.gr_gate == 0.0)

    @pytest.mark.parametrize("aux_ffns", [(AUX_FFN,), AUX_FFNS])
    def test_output_is_bitwise_equal_at_gate_zero(self, aux_ffns):
        vanilla, gram = _pair(aux_ffns=aux_ffns)
        x = _input()
        out_vanilla, bias_vanilla = vanilla(x)
        out_gram, bias_gram = gram(x)
        assert torch.equal(out_vanilla, out_gram), (
            f"max |diff| = {(out_vanilla.float() - out_gram.float()).abs().max().item()}"
        )
        assert bias_vanilla is None and bias_gram is None

    @pytest.mark.parametrize("aux_ffns", [(AUX_FFN,), AUX_FFNS])
    def test_output_is_bitwise_equal_at_all_gates_one_with_zero_init_fc2(self, aux_ffns):
        """The warm-start posture: gates open on the very first aux iteration, aux weights
        still at their zero init. A run must start from the core model's exact behaviour."""
        vanilla, gram = _pair(aux_ffns=aux_ffns)
        gram.gr_gate.fill_(1.0)
        x = _input()
        assert torch.equal(vanilla(x)[0], gram(x)[0])

    @pytest.mark.parametrize("aux_ffns", [(AUX_FFN,), AUX_FFNS])
    def test_fc2_is_zero_initialised_on_every_module(self, aux_ffns):
        _, gram = _pair(aux_ffns=aux_ffns)
        for k in range(len(aux_ffns)):
            assert torch.all(gram.gr_aux[k].linear_fc2.weight == 0)
            assert not torch.all(gram.gr_aux[k].linear_fc1.weight == 0), f"module {k} fc1 must be initialised"

    def test_aux_module_has_the_configured_width(self):
        _, gram = _pair()
        assert tuple(gram.gr_aux[0].linear_fc1.weight.shape) == (AUX_FFN, HIDDEN)
        assert tuple(gram.gr_aux[0].linear_fc2.weight.shape) == (HIDDEN, AUX_FFN)

    def test_each_module_gets_its_own_width(self):
        """Widths are per module, and ``GRAMAuxMLP`` reads its width from two places (the
        argument and a config clone) — so a shared clone would silently give every module the
        LAST module's width, which no shape assertion on module 0 alone would catch."""
        _, gram = _pair(aux_ffns=AUX_FFNS)
        assert len(gram.gr_aux) == len(AUX_FFNS)
        for k, width in enumerate(AUX_FFNS):
            assert tuple(gram.gr_aux[k].linear_fc1.weight.shape) == (width, HIDDEN), f"module {k} fc1"
            assert tuple(gram.gr_aux[k].linear_fc2.weight.shape) == (HIDDEN, width), f"module {k} fc2"
        assert AUX_FFNS[0] != AUX_FFNS[1], "the widths must differ or this proves nothing"

    @pytest.mark.parametrize("aux_ffns", [(AUX_FFN,), AUX_FFNS])
    def test_aux_width_does_not_leak_into_the_shared_expert(self, aux_ffns):
        """GRAMAuxMLP deep-copies the config before narrowing ffn_hidden_size; a shared
        mutation would silently resize the real shared expert."""
        config = moe_config()
        _, gram = _pair(config=config, aux_ffns=aux_ffns)
        assert config.ffn_hidden_size not in aux_ffns
        assert tuple(gram.shared_experts.linear_fc1.weight.shape) == (SHARED_FFN, HIDDEN)

    @pytest.mark.parametrize("aux_ffns", [(AUX_FFN,), AUX_FFNS])
    def test_gate_is_not_in_the_state_dict(self, aux_ffns):
        """Non-persistent by design: checkpoints carry aux weights, never a routing gate."""
        _, gram = _pair(aux_ffns=aux_ffns)
        assert "gr_gate" not in gram.state_dict()
        assert "gr_gate" in dict(gram.named_buffers())

    @pytest.mark.parametrize("aux_ffns", [(AUX_FFN,), AUX_FFNS])
    def test_state_dict_adds_only_the_aux_submodules(self, aux_ffns):
        """A warm start from a base checkpoint must be missing exactly these keys and no
        others — which is why the launch guards demand a missing-key-tolerant strictness. The
        indexed ``gr_aux.<k>.`` prefix is also the fragment the optimizer override glob, the
        HF bridge mappings and the bake script all key on."""
        vanilla, gram = _pair(aux_ffns=aux_ffns)
        added = set(gram.state_dict()) - set(vanilla.state_dict())
        assert added == {
            f"gr_aux.{k}.{projection}{suffix}"
            for k in range(len(aux_ffns))
            for projection in ("linear_fc1", "linear_fc2")
            for suffix in (".weight", "._extra_state")
        }
        assert set(vanilla.state_dict()) - set(gram.state_dict()) == set()


@requires_gpu
@pytest.mark.usefixtures("moe_parallel_state")
class TestGatedOnAddsTheAuxOutput:
    """Gate 1 with trained weights must add exactly ``aux(h)``, unscaled — the export merge
    into the shared expert is only exact for coefficient 1.0 on the same input, and it is
    additive per module, so any SUBSET of open gates must be the sum of their own outputs."""

    def _trained_pair(self, aux_ffns=(AUX_FFN,), scale=0.05):
        vanilla, gram = _pair(aux_ffns=aux_ffns)
        _train_aux(gram, scale=scale)
        gram.gr_gate.fill_(1.0)
        return vanilla, gram

    def test_output_equals_core_plus_aux_exactly(self):
        vanilla, gram = self._trained_pair()
        x = _input()
        out_core, _ = vanilla(x)
        out_gram, _ = gram(x)
        aux = gram.gr_aux[0](x)
        assert not torch.equal(out_core, out_gram), "aux contribution vanished — the test proves nothing"
        assert torch.equal(out_gram, out_core + aux)

    @pytest.mark.parametrize("module", [0, 1])
    def test_one_open_gate_adds_exactly_that_modules_output(self, module):
        """The subset-merge contract: with only module k gated on, the layer output must be
        the core output plus module k's own forward — not the other module's, and not a
        blend. Both modules are trained so a mix-up cannot hide behind a zero."""
        vanilla, gram = self._trained_pair(aux_ffns=AUX_FFNS)
        gram.gr_gate.zero_()
        gram.gr_gate[module] = 1.0
        x = _input()
        out_core, _ = vanilla(x)
        out_gram, _ = gram(x)
        mine, theirs = gram.gr_aux[module](x), gram.gr_aux[1 - module](x)
        assert torch.equal(out_gram, out_core + mine)
        assert not torch.equal(out_gram, out_core + theirs), "the two modules produce the same output"

    def test_both_open_gates_add_both_modules(self):
        """Additivity across modules, which is what makes an arbitrary subset merge exact."""
        vanilla, gram = self._trained_pair(aux_ffns=AUX_FFNS)
        x = _input()
        out_core, _ = vanilla(x)
        expected = out_core + gram.gr_aux[0](x) + gram.gr_aux[1](x)
        assert torch.equal(gram(x)[0], expected)

    def test_aux_module_computes_the_reference_mlp(self):
        """The aux is genuinely ``W2 . squared_relu(W1 h)`` at aux width — not, say, gated,
        or fed the post-MoE activations. Tolerance here because the reference uses plain
        torch matmuls against TE's kernels."""
        _, gram = self._trained_pair()
        x = _input()
        w1 = gram.gr_aux[0].linear_fc1.weight.float()
        w2 = gram.gr_aux[0].linear_fc2.weight.float()
        reference = (torch.relu(x.float() @ w1.T) ** 2) @ w2.T
        torch.testing.assert_close(gram.gr_aux[0](x).float(), reference, rtol=2e-2, atol=2e-2)

    @pytest.mark.parametrize("gate", [0.0, 1.0])
    def test_gate_scales_the_contribution(self, gate):
        vanilla, gram = self._trained_pair()
        gram.gr_gate.fill_(gate)
        x = _input()
        out_core, _ = vanilla(x)
        assert torch.equal(gram(x)[0], out_core + gate * gram.gr_aux[0](x))

    def test_gate_can_be_toggled_between_forwards(self):
        """The callback flips this buffer in place every iteration."""
        vanilla, gram = self._trained_pair()
        x = _input()
        out_core, _ = vanilla(x)
        gram.gr_gate.fill_(0.0)
        assert torch.equal(gram(x)[0], out_core)
        gram.gr_gate.fill_(1.0)
        assert not torch.equal(gram(x)[0], out_core)
        gram.gr_gate.fill_(0.0)
        assert torch.equal(gram(x)[0], out_core)

    def test_static_gates_pin_the_buffer_at_construction(self):
        """Eval-only profile serving: a corpus-loss probe loads a GRAM checkpoint and scores
        one module subset with no runtime gate driver, so the requested subset has to be in
        the buffer the moment the layer exists."""
        layer = build_moe_layer(moe_builder(gram_spec(AUX_FFNS)), _static_gate_config([1.0, 0.0]))
        assert layer.gr_gate.tolist() == [1.0, 0.0]
        assert "gr_gate" not in layer.state_dict(), "a pinned gate must still stay out of checkpoints"

    def test_a_closed_gate_stays_inert_while_its_sibling_is_open(self):
        """The per-iteration regime at N>1: a core-robustness iteration opens exactly one
        gate, so the modules must not leak into each other's contribution."""
        vanilla, gram = self._trained_pair(aux_ffns=AUX_FFNS)
        x = _input()
        out_core, _ = vanilla(x)
        gram.gr_gate.zero_()
        gram.gr_gate[0] = 1.0
        only_first = gram(x)[0]
        gram.gr_gate[1] = 1.0
        both = gram(x)[0]
        assert torch.equal(only_first, out_core + gram.gr_aux[0](x))
        assert torch.equal(both, only_first + gram.gr_aux[1](x))


@requires_gpu
@pytest.mark.usefixtures("moe_parallel_state")
class TestGradients:
    """Gate 0 must still produce gradients (DDP bucket completion) and they must be exact zero."""

    def _backward(self, layer, x, grad_seed=11):
        out, _ = layer(x)
        torch.manual_seed(grad_seed)
        out.backward(torch.randn_like(out))
        return out

    @pytest.mark.parametrize("aux_ffns", [(AUX_FFN,), AUX_FFNS])
    def test_aux_params_receive_exact_zero_grads_at_gate_zero(self, aux_ffns):
        """Megatron's DDP requires every parameter to produce a gradient each microbatch;
        a None grad here stalls the bucket under overlap_grad_reduce."""
        _, gram = _pair(aux_ffns=aux_ffns)
        self._backward(gram, _input(requires_grad=True))
        for name, p in gram.named_parameters():
            if ".gr_aux." in f".{name}":
                assert p.grad is not None, f"{name} received no gradient at gate 0"
                assert torch.all(p.grad == 0), f"{name} received a non-zero gradient at gate 0"

    @pytest.mark.parametrize("aux_ffns", [(AUX_FFN,), AUX_FFNS])
    def test_core_grads_are_bitwise_equal_to_the_vanilla_layer(self, aux_ffns):
        vanilla, gram = _pair(aux_ffns=aux_ffns)
        x_vanilla = _input(requires_grad=True)
        x_gram = _input(requires_grad=True)
        self._backward(vanilla, x_vanilla)
        self._backward(gram, x_gram)

        core = _core_params(gram)
        for name, p in vanilla.named_parameters():
            assert (p.grad is None) == (core[name].grad is None), f"{name} grad presence differs"
            if p.grad is not None:
                assert torch.equal(p.grad, core[name].grad), f"{name} gradient differs"
        assert torch.equal(x_vanilla.grad, x_gram.grad), "input gradient differs"

    def test_aux_params_receive_non_zero_grads_at_gate_one(self):
        """The complement: gating is what zeroes them, not a detached graph."""
        _, gram = _pair()
        _train_aux(gram)
        gram.gr_gate.fill_(1.0)
        self._backward(gram, _input(requires_grad=True))
        for name in ("gr_aux.0.linear_fc1.weight", "gr_aux.0.linear_fc2.weight"):
            grad = dict(gram.named_parameters())[name].grad
            assert grad is not None and not torch.all(grad == 0), f"{name} got no usable gradient at gate 1"

    @pytest.mark.parametrize("open_module", [0, 1])
    def test_only_the_open_modules_parameters_receive_gradient(self, open_module):
        """The forward gate is the first half of isolation (the optimizer gate is the second):
        with one gate open, the CLOSED module's parameters must still get exact-zero grads,
        so no gradient of the open module's corpus reaches them."""
        _, gram = _pair(aux_ffns=AUX_FFNS)
        _train_aux(gram)
        gram.gr_gate.zero_()
        gram.gr_gate[open_module] = 1.0
        self._backward(gram, _input(requires_grad=True))

        params = dict(gram.named_parameters())
        for projection in ("linear_fc1", "linear_fc2"):
            open_grad = params[f"gr_aux.{open_module}.{projection}.weight"].grad
            closed_grad = params[f"gr_aux.{1 - open_module}.{projection}.weight"].grad
            assert not torch.all(open_grad == 0), f"open module {open_module} {projection} got no gradient"
            assert torch.all(closed_grad == 0), f"closed module {1 - open_module} {projection} received gradient"

    @pytest.mark.parametrize("aux_ffns", [(AUX_FFN,), AUX_FFNS])
    def test_gate_buffer_is_not_a_parameter(self, aux_ffns):
        _, gram = _pair(aux_ffns=aux_ffns)
        assert gram.gr_gate.requires_grad is False
        assert "gr_gate" not in dict(gram.named_parameters())


@requires_gpu
@pytest.mark.usefixtures("moe_parallel_state")
class TestConstructionRefusals:
    """Every shape assumption the export merge rests on is enforced at construction."""

    def test_split_execution_forward_raises(self):
        """intermediate_tensors is the overlap_moe_expert_parallel_comm / CUDA-graph path,
        which the aux add does not implement — refuse rather than silently drop the addend."""
        _, gram = _pair()
        with pytest.raises(NotImplementedError, match="split-execution forward"):
            gram(_input(), intermediate_tensors=object())

    @pytest.mark.parametrize("field, match", [("gated_linear_unit", "non-gated"), ("add_bias_linear", "bias-free")])
    def test_unsupported_activation_shape_raises(self, field, match):
        from megatron.core.transformer.mlp import MLPSubmodules

        from megatron.bridge.models.mamba.gram_layer import GRAMAuxMLP

        submodules = moe_builder(stack_spec()).keywords["submodules"].shared_experts.keywords["submodules"]
        assert isinstance(submodules, MLPSubmodules)
        with pytest.raises(ValueError, match=match):
            GRAMAuxMLP(config=moe_config(**{field: True}), submodules=submodules, aux_ffn_hidden_size=AUX_FFN)

    def test_layer_without_shared_experts_raises(self):
        """The aux module mirrors the shared expert and both export postures merge into it."""
        with pytest.raises(ValueError, match="requires a shared-expert MoE layer"):
            build_moe_layer(moe_builder(gram_spec()), moe_config(moe_shared_expert_intermediate_size=None))

    def test_static_gates_of_the_wrong_length_raise(self):
        """One entry per module, or the probe serves a different subset than it asked for."""
        with pytest.raises(ValueError, match="gr_static_gates has 3 entries for 2 aux modules"):
            build_moe_layer(moe_builder(gram_spec(AUX_FFNS)), _static_gate_config([1.0, 0.0, 1.0]))

    def test_shared_experts_builder_without_submodules_raises(self):
        """A shared-expert builder that carries no ``submodules`` keyword gives the aux
        modules nothing to mirror; the message must say so rather than AttributeError.

        The stand-in is a plain function rather than a partial without the keyword: MoELayer
        itself calls the builder before GRAMMoELayer's check runs, so a builder missing
        ``submodules`` outright dies inside MoELayer. What this pins is the case the check
        actually catches — a builder that constructs fine but hides its submodules.
        """
        spec = gram_spec()
        moe_builder(spec).keywords["submodules"].shared_experts = _opaque_shared_expert_builder
        with pytest.raises(ValueError, match="could not extract MLPSubmodules"):
            build_moe_layer(moe_builder(spec), moe_config())


class TestSwapSpec:
    """The spec transform mirrors swap_moe_experts_to_grouped, including its deepcopy."""

    @pytest.mark.parametrize("aux_ffns", [(AUX_FFN,), AUX_FFNS])
    def test_swaps_only_the_moe_layer_class(self, aux_ffns):
        from megatron.bridge.models.mamba.gram_layer import GRAMMoELayer

        original_submodules = moe_builder(stack_spec()).keywords["submodules"]
        swapped = moe_builder(gram_spec(aux_ffns))

        assert swapped.func is GRAMMoELayer
        assert swapped.keywords["gr_aux_ffn_hidden_sizes"] == list(aux_ffns)
        assert set(swapped.keywords) == {"submodules", "gr_aux_ffn_hidden_sizes"}
        # router / experts / shared experts travel through unchanged
        swapped_submodules = swapped.keywords["submodules"]
        assert swapped_submodules.router is original_submodules.router
        assert _render(swapped_submodules.experts) == _render(original_submodules.experts)
        assert _render(swapped_submodules.shared_experts) == _render(original_submodules.shared_experts)

    def test_the_width_list_is_copied_not_aliased(self):
        """The swap binds the widths into a partial that outlives the call; aliasing the
        caller's list would let a later mutation of it change what every model builds."""
        from megatron.bridge.models.mamba.gram_layer import swap_moe_layer_to_gram

        widths = list(AUX_FFNS)
        builder = moe_builder(swap_moe_layer_to_gram(stack_spec(), aux_ffn_hidden_sizes=widths))
        widths.append(4096)
        assert builder.keywords["gr_aux_ffn_hidden_sizes"] == list(AUX_FFNS)

    def test_swap_does_not_mutate_the_caller_spec(self):
        """``mamba_stack_spec`` hands out one process-global object; an in-place swap would
        rewrite the spec for every model built afterwards in the process."""
        from megatron.core.transformer.moe.moe_layer import MoELayer

        from megatron.bridge.models.mamba.gram_layer import GRAMMoELayer, swap_moe_layer_to_gram

        original = stack_spec()
        original_builder = moe_builder(original)
        swap_moe_layer_to_gram(original, aux_ffn_hidden_sizes=[AUX_FFN])
        assert moe_builder(original) is original_builder
        assert original_builder.func is MoELayer
        assert original_builder.func is not GRAMMoELayer

    def test_double_apply_raises(self):
        from megatron.bridge.models.mamba.gram_layer import swap_moe_layer_to_gram

        with pytest.raises(ValueError, match="already applied"):
            swap_moe_layer_to_gram(gram_spec(), aux_ffn_hidden_sizes=[AUX_FFN])

    @pytest.mark.parametrize("widths", [[0], [-1], [-512], [AUX_FFN, 0], [AUX_FFN, None], [AUX_FFN, 8.0]])
    def test_a_non_positive_or_non_integer_width_raises(self, widths):
        """The offending index is named: with a list of widths, "one of them is wrong" is not
        an actionable message."""
        from megatron.bridge.models.mamba.gram_layer import swap_moe_layer_to_gram

        bad_index = next(i for i, w in enumerate(widths) if not isinstance(w, int) or w <= 0)
        with pytest.raises(ValueError, match=rf"aux_ffn_hidden_sizes\[{bad_index}\] must be a positive int"):
            swap_moe_layer_to_gram(stack_spec(), aux_ffn_hidden_sizes=widths)

    def test_an_empty_width_list_raises(self):
        """Zero modules is not a GR model: the gate buffer would be empty and the optimizer
        override would find nothing to mark."""
        from megatron.bridge.models.mamba.gram_layer import swap_moe_layer_to_gram

        with pytest.raises(ValueError, match="must name at least one aux module"):
            swap_moe_layer_to_gram(stack_spec(), aux_ffn_hidden_sizes=[])

    def test_non_moe_mlp_builder_raises(self):
        """Guards against applying the swap to an already-rewritten spec, where the silent
        alternative is a GRAMMoELayer wrapping the wrong class."""
        from megatron.core.transformer.mlp import MLP

        from megatron.bridge.models.mamba.gram_layer import swap_moe_layer_to_gram

        spec = stack_spec()
        spec.submodules.moe_layer.submodules.mlp = partial(MLP)
        with pytest.raises(ValueError, match="Expected moe_layer.submodules.mlp to be partial"):
            swap_moe_layer_to_gram(spec, aux_ffn_hidden_sizes=[AUX_FFN])


@requires_gpu
@pytest.mark.usefixtures("moe_parallel_state")
class TestComposesWithGroupedExperts:
    """Both swaps edit the same mlp partial — one its func, one its submodules — so the
    order must not matter. Every real GR run applies both."""

    @pytest.fixture(autouse=True)
    def _require_torch_grouped(self):
        if not hasattr(torch, "_grouped_mm"):
            pytest.skip(f"gemm_backend='torch_grouped' needs torch._grouped_mm, absent from torch {torch.__version__}")

    def _composed(self, gram_first, aux_ffns=(AUX_FFN,)):
        from megatron.bridge.models.mamba.gram_layer import swap_moe_layer_to_gram
        from megatron.bridge.models.mamba.grouped_experts import swap_moe_experts_to_grouped

        spec = stack_spec()
        order = (
            (swap_moe_layer_to_gram, swap_moe_experts_to_grouped)
            if gram_first
            else (swap_moe_experts_to_grouped, swap_moe_layer_to_gram)
        )
        for transform in order:
            if transform is swap_moe_layer_to_gram:
                spec = transform(spec, aux_ffn_hidden_sizes=list(aux_ffns))
            else:
                spec = transform(spec, gemm_backend="torch_grouped")
        return spec

    @pytest.mark.parametrize("gram_first", [True, False])
    @pytest.mark.parametrize("aux_ffns", [(AUX_FFN,), AUX_FFNS])
    def test_both_orders_build_the_same_module_types(self, gram_first, aux_ffns):
        from megatron.bridge.models.mamba.gram_layer import GRAMMoELayer
        from megatron.bridge.models.mamba.grouped_experts import GroupedExperts

        builder = moe_builder(self._composed(gram_first, aux_ffns))
        assert builder.func is GRAMMoELayer
        assert builder.keywords["submodules"].experts.func is GroupedExperts
        layer = build_moe_layer(builder, moe_config())
        assert isinstance(layer.experts, GroupedExperts)
        assert len(layer.gr_aux) == len(aux_ffns)

    @pytest.mark.parametrize("gram_first", [True, False])
    @pytest.mark.parametrize("aux_ffns", [(AUX_FFN,), AUX_FFNS])
    def test_composed_layer_runs_and_is_identity_at_gate_zero(self, gram_first, aux_ffns):
        from megatron.bridge.models.mamba.grouped_experts import swap_moe_experts_to_grouped

        config = moe_config()
        grouped_only = build_moe_layer(
            moe_builder(swap_moe_experts_to_grouped(stack_spec(), gemm_backend="torch_grouped")), config
        )
        composed = build_moe_layer(moe_builder(self._composed(gram_first, aux_ffns)), config)
        x = _input()
        assert torch.equal(grouped_only(x)[0], composed(x)[0])
