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
"""Gradient routing must be invisible when it is off, and inert on retain iterations.

Two regressions are pinned here, and they protect different people:

- **GR off** — every non-GR run in this repo shares the provider, the dataset builder and
  the config container with GR. The requirement is that leaving ``gr:`` out of a config
  changes nothing at all, down to spec object identity. A "harmless" unconditional spec
  rewrite would silently alter every Nano/Super run in the repo.
- **GR on, retain iteration** — an all-retain plan is exactly the gate-0, core-only regime
  the retain half of every real run spends its time in. One step being identical (proved
  in ``test_gram_layer.py``) does not imply the trajectory is: any divergence compounds
  through the parameters. So the comparison is a multi-step trajectory, per-step losses
  bitwise equal, against the vanilla model trained from the same seed on the same inputs.

The contrast case (gate 1 with trained aux weights) is asserted to DIVERGE, so a broken
harness that merely ran the same model twice could not pass.
"""

import pytest
import torch

from tests.unit_tests.gr_test_utils import (
    AUX_FFN,
    HIDDEN,
    NUM_EXPERTS,
    build_moe_layer,
    gram_spec,
    init_model_parallel,
    moe_builder,
    moe_config,
    stack_spec,
)


requires_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU (real TE MoE layers)")

N_LAYERS, N_STEPS, SEQ, BATCH = 2, 5, 6, 2


@pytest.fixture(scope="module")
def moe_parallel_state():
    from megatron.core import parallel_state
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    init_model_parallel(expert_model_parallel_size=1)
    model_parallel_cuda_manual_seed(1234)
    yield
    parallel_state.destroy_model_parallel()


def _provider(**overrides):
    from megatron.bridge.models.mamba.mamba_provider import MambaModelProvider

    provider = MambaModelProvider(num_layers=2, hidden_size=HIDDEN, num_attention_heads=4, num_moe_experts=NUM_EXPERTS)
    for name, value in overrides.items():
        setattr(provider, name, value)
    return provider


class TestProviderIsInertWithoutGR:
    """No ``gr_aux_ffn_hidden_size`` means no swap and no rebinding of the spec field."""

    def test_spec_field_is_never_rebound_and_input_passes_through(self):
        """Identity, not equality, on both contracts: the provider must not rebind
        ``mamba_stack_spec`` (anything assigned there serializes into the checkpoint's
        run_config, where a closure breaks from_auto_config), and with GR off the
        resolved spec must pass through unwrapped."""
        provider = _provider()
        assert provider.gr_aux_ffn_hidden_size is None
        before = provider.mamba_stack_spec
        resolved = provider.mamba_stack_spec(provider)
        assert provider._apply_gradient_routing(resolved) is resolved
        assert provider.mamba_stack_spec is before

    def test_resolved_spec_still_builds_a_plain_moe_layer(self):
        from megatron.core.transformer.moe.moe_layer import MoELayer

        provider = _provider()
        spec = provider._apply_gradient_routing(provider.mamba_stack_spec(provider))
        assert moe_builder(spec).func is MoELayer

    def test_refusals_do_not_fire_when_gr_is_off(self):
        """The MTP and latent-MoE refusals are GR preconditions, not general ones — they
        must not reject an ordinary non-GR run that happens to use either feature."""
        for provider in (_provider(mtp_num_layers=1), _provider(moe_latent_size=16)):
            provider._apply_gradient_routing(provider.mamba_stack_spec(provider))


class TestProviderAppliesGRWhenRequested:
    def test_swap_applies_and_spec_field_stays_untouched(self):
        from megatron.bridge.models.mamba.gram_layer import GRAMMoELayer

        provider = _provider(gr_aux_ffn_hidden_size=AUX_FFN)
        before = provider.mamba_stack_spec
        builder = moe_builder(provider._apply_gradient_routing(provider.mamba_stack_spec(provider)))
        assert builder.func is GRAMMoELayer
        assert builder.keywords["gr_aux_ffn_hidden_size"] == AUX_FFN
        assert provider.mamba_stack_spec is before

    def test_double_apply_on_a_swapped_spec_is_refused(self):
        """Each provide() call starts from a freshly resolved spec, so the only way a
        swapped spec reaches the hook again is a caller bug — refuse it loudly rather
        than nest GRAM layers."""
        provider = _provider(gr_aux_ffn_hidden_size=AUX_FFN)
        swapped = provider._apply_gradient_routing(provider.mamba_stack_spec(provider))
        with pytest.raises(ValueError, match="double-apply"):
            provider._apply_gradient_routing(swapped)

    def test_repeated_provide_style_application_from_fresh_specs_is_safe(self):
        """The VPP path calls provide() once per chunk; each call re-resolves the spec, so
        repeated application must keep succeeding when given fresh resolutions."""
        from megatron.bridge.models.mamba.gram_layer import GRAMMoELayer

        provider = _provider(gr_aux_ffn_hidden_size=AUX_FFN)
        for _ in range(2):
            spec = provider._apply_gradient_routing(provider.mamba_stack_spec(provider))
            assert moe_builder(spec).func is GRAMMoELayer

    def test_mtp_layers_raise(self):
        provider = _provider(gr_aux_ffn_hidden_size=AUX_FFN, mtp_num_layers=1)
        with pytest.raises(NotImplementedError, match="mtp_num_layers == 0"):
            provider._apply_gradient_routing(provider.mamba_stack_spec(provider))

    def test_latent_moe_raises(self):
        provider = _provider(gr_aux_ffn_hidden_size=AUX_FFN, moe_latent_size=16)
        with pytest.raises(NotImplementedError, match="moe_latent_size"):
            provider._apply_gradient_routing(provider.mamba_stack_spec(provider))


class TestDisabledConfigConstructsNothing:
    def test_config_container_defaults_to_no_gr_section(self):
        """``cfg.gr is None`` is the condition every GR code path in setup() hangs off."""
        from dataclasses import fields

        from megatron.bridge.training.config import ConfigContainer

        gr_field = next(f for f in fields(ConfigContainer) if f.name == "gr")
        assert gr_field.default is None

    def test_disabled_finalize_is_a_noop(self):
        from megatron.bridge.training.gradient_routing.config import GradientRoutingConfig

        config = GradientRoutingConfig(enabled=False)
        config.finalize()
        assert config.retain_data_path is None and config.plan_seed is None
        assert not hasattr(config, "runtime_plan")
        assert not hasattr(config, "runtime_gater")


class _TinyStack(torch.nn.Module):
    """A residual stack of MoE layers — the smallest thing whose loss compounds over steps."""

    def __init__(self, builder, config, n_layers=N_LAYERS):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [build_moe_layer(builder, config, layer_number=i + 1, seed=4321 + i) for i in range(n_layers)]
        )

    def forward(self, hidden_states):
        for layer in self.layers:
            output, _ = layer(hidden_states)
            hidden_states = hidden_states + output
        return hidden_states


def _train(builder, config, gate=None, randomize_aux=False, lr=1e-2):
    """Run N_STEPS of SGD and return the per-step losses.

    Plain SGD (no momentum, no weight decay) rather than the Megatron optimizer: this test
    is about the forward/backward trajectory, and the optimizer-side isolation has its own
    file with the real optimizer. With no momentum and no weight decay, a parameter whose
    gradient is exactly zero cannot move — which is what makes an all-retain plan's aux
    module provably inert here.
    """
    model = _TinyStack(builder, config)
    if randomize_aux:
        torch.manual_seed(99)
        with torch.no_grad():
            for layer in model.layers:
                layer.gr_aux.linear_fc2.weight.normal_(0.0, 0.05)
    if gate is not None:
        for layer in model.layers:
            layer.gr_gate.fill_(gate)

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.0, weight_decay=0.0)
    losses = []
    for step in range(N_STEPS):
        torch.manual_seed(1000 + step)  # identical inputs across arms
        x = torch.randn(SEQ, BATCH, HIDDEN, dtype=torch.bfloat16, device="cuda")
        optimizer.zero_grad(set_to_none=False)
        loss = model(x).float().square().mean()
        loss.backward()
        optimizer.step()
        losses.append(loss.detach().clone())
    return model, losses


@requires_gpu
@pytest.mark.usefixtures("moe_parallel_state")
class TestAllRetainTrajectoryMatchesVanilla:
    """An all-retain plan is gate 0 on every step — the whole trajectory must be unchanged."""

    def _plan(self):
        from megatron.bridge.training.gradient_routing.plan import build_gr_plan

        return build_gr_plan(plan_seed=1234, train_iters=N_STEPS, forget_iter_fraction=0.0, p_as=0.0, p_cr=0.0)

    def test_the_degenerate_plan_really_is_all_retain(self):
        """If the plan were not all-retain, the gates below would be driven to 1 and the
        comparison would be measuring something else."""
        from megatron.bridge.training.gradient_routing.plan import RETAIN

        plan = self._plan()
        assert (plan.corpus == RETAIN).all()
        assert not plan.fwd_aux.any() and not plan.update_aux.any()
        assert plan.update_core.all()

    def test_per_step_losses_are_bitwise_equal(self):
        config = moe_config()
        plan = self._plan()
        _, vanilla_losses = _train(moe_builder(stack_spec()), config)
        _, gram_losses = _train(moe_builder(gram_spec()), config, gate=float(plan.fwd_aux[0]))

        assert len(vanilla_losses) == len(gram_losses) == N_STEPS
        for step, (vanilla, gram) in enumerate(zip(vanilla_losses, gram_losses)):
            assert torch.equal(vanilla, gram), (
                f"step {step}: vanilla {vanilla.item()} vs GRAM {gram.item()} "
                f"(diff {abs(vanilla.item() - gram.item()):.3e})"
            )
        # the losses must actually be moving, or "equal" would be trivially true
        assert not torch.equal(vanilla_losses[0], vanilla_losses[-1]), "training did not change the loss"

    def test_aux_weights_never_move_on_an_all_retain_trajectory(self):
        config = moe_config()
        gram_model, _ = _train(moe_builder(gram_spec()), config, gate=0.0)
        for layer in gram_model.layers:
            assert torch.all(layer.gr_aux.linear_fc2.weight == 0), "aux fc2 moved with a zero gradient"

    def test_core_weights_end_bitwise_equal(self):
        """The parameters, not only the losses: a difference too small to show in a bf16
        loss would still be a difference in the model that gets exported."""
        config = moe_config()
        vanilla_model, _ = _train(moe_builder(stack_spec()), config)
        gram_model, _ = _train(moe_builder(gram_spec()), config, gate=0.0)

        vanilla_params = dict(vanilla_model.named_parameters())
        for name, param in gram_model.named_parameters():
            if ".gr_aux." in name:
                continue
            assert torch.equal(param, vanilla_params[name]), f"{name} diverged over {N_STEPS} steps"

    def test_a_gated_on_trained_aux_does_diverge(self):
        """The negative control: the harness can detect a difference when there is one."""
        config = moe_config()
        _, vanilla_losses = _train(moe_builder(stack_spec()), config)
        _, forget_losses = _train(moe_builder(gram_spec()), config, gate=1.0, randomize_aux=True)
        assert not all(torch.equal(v, f) for v, f in zip(vanilla_losses, forget_losses))
