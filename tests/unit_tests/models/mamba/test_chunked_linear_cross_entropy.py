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
"""The chunked linear cross-entropy fusion against the path it replaces.

``cross_entropy_loss_fusion`` with ``cross_entropy_fusion_impl='linear'`` makes HybridModel's output
layer return the per-token loss of ``chunked_linear_cross_entropy`` instead of the logits that
``vocab_parallel_cross_entropy`` consumes (0005 in 3rdparty/patches/megatron-lm/README.md, a carried
commit of the pinned Megatron-LM). The op, the output-layer module and the model run here for real on
a GPU, with a real single-process process group, against Megatron's own unfused path (the production
path) and an fp32 reference. Both bf16 paths carry the same bf16 rounding error against fp32: the
fused path has to agree with the unfused one up to summation order and be no less accurate than it.

The op is Triton kernels plus cuBLAS GEMMs, so everything but the config plumbing and the
tensor-parallel refusal needs a GPU.
"""

from pathlib import Path

import pytest
import torch

from tests.unit_tests.one_rank_nccl_world import one_rank_model_parallel_state
from tests.unit_tests.small_language_models import hybrid_model


V_SMALL = 1000
SEQ = 16
LINEAR_FUSION = dict(
    cross_entropy_loss_fusion=True, cross_entropy_fusion_impl="linear", cross_entropy_fusion_vocab_chunk_size=256
)
_REPO_ROOT = Path(__file__).resolve().parents[4]
BASELINE = _REPO_ROOT / "configs" / "control_pretraining" / "30b_baseline" / "nemotron_nano_30b_baseline_pretrain.yaml"
BASELINE_BENCHMARK = _REPO_ROOT / "configs" / "quickstart" / "nemotron_nano_quickstart_pretrain_baseline.yaml"


@pytest.fixture(scope="module")
def tp_group():
    """Real world-1 mcore parallel state; yields the (size-1) tensor-parallel group."""
    from megatron.core import parallel_state

    with one_rank_model_parallel_state(seed=1234):
        yield parallel_state.get_tensor_model_parallel_group()


@pytest.fixture
def fp32_matmul():
    """fp32 GEMMs without TF32 (the NGC image enables it), so fp32 models compare to fp32 rounding."""
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    yield
    torch.backends.cuda.matmul.allow_tf32 = previous


def _rel(actual, expected):
    return ((actual.double() - expected.double()).norm() / expected.double().norm()).item()


def _inputs(tokens, hidden_size, vocab_size, dtype=torch.bfloat16, seed=0):
    """Hidden states, output weight, labels (the first out of vocabulary) and an upstream gradient
    with every third token masked."""
    generator = torch.Generator(device="cuda").manual_seed(seed)
    hidden = torch.randn(tokens, hidden_size, device="cuda", generator=generator).to(dtype)
    weight = (0.1 * torch.randn(vocab_size, hidden_size, device="cuda", generator=generator)).to(dtype)
    labels = torch.randint(0, vocab_size, (tokens,), device="cuda", generator=generator)
    labels[0] = vocab_size + 3
    grad_loss = torch.rand(tokens, device="cuda", generator=generator)
    grad_loss[1::3] = 0.0
    return hidden, weight, labels, grad_loss


def _op(hidden, weight, labels, chunk, saved=0, fusion=False):
    from megatron.core.fusions.fused_chunked_linear_cross_entropy import chunked_linear_cross_entropy

    return chunked_linear_cross_entropy(hidden, weight, labels, chunk, saved, fusion)


def _unfused(hidden, weight, labels, grad_loss, tp_group):
    """Megatron's unfused path: the output-layer GEMM, then vocab_parallel_cross_entropy."""
    from megatron.core.tensor_parallel.cross_entropy import vocab_parallel_cross_entropy

    hidden = hidden.detach().requires_grad_()
    weight = weight.detach().requires_grad_()
    loss = vocab_parallel_cross_entropy(hidden @ weight.t(), labels, tp_group=tp_group)
    loss.backward(grad_loss)
    return loss.detach(), hidden.grad, weight.grad


def _fused(hidden, weight, labels, grad_loss, chunk, saved=0):
    hidden = hidden.detach().requires_grad_()
    weight = weight.detach().requires_grad_()
    loss = _op(hidden, weight, labels, chunk, saved)
    loss.backward(grad_loss)
    return loss.detach(), hidden.grad, weight.grad


@pytest.mark.run_only_on("GPU")
class TestChunkedLinearCrossEntropyOp:
    @pytest.mark.parametrize(
        "tokens,hidden_size,vocab_size,chunk",
        [
            (300, 64, V_SMALL, 256),  # ragged last chunk (232 columns)
            (257, 128, 4096, 1024),  # chunks divide the vocabulary
            (64, 64, 3000, 5000),  # one chunk wider than the vocabulary
            (1, 16, 64, 7),  # one token, many ragged chunks
            (512, 256, 8192, 3000),
        ],
    )
    def test_matches_the_unfused_path(self, tp_group, tokens, hidden_size, vocab_size, chunk):
        hidden, weight, labels, grad_loss = _inputs(tokens, hidden_size, vocab_size)
        fused = _fused(hidden, weight, labels, grad_loss, chunk)
        unfused = _unfused(hidden, weight, labels, grad_loss, tp_group)
        reference = _unfused(hidden.float(), weight.float(), labels, grad_loss, tp_group)

        torch.testing.assert_close(fused[0], unfused[0], rtol=0, atol=1e-5)
        for fused_grad, unfused_grad, reference_grad in zip(fused[1:], unfused[1:], reference[1:]):
            assert _rel(fused_grad, unfused_grad) < 2e-3
            assert _rel(fused_grad, reference_grad) <= 1.1 * _rel(unfused_grad, reference_grad) + 1e-6

    @pytest.mark.parametrize("saved", [1, 3, 4, 100])
    def test_saved_logit_chunks_give_identical_results(self, tp_group, saved):
        """Saving a chunk's logits instead of recomputing them changes nothing, bit for bit."""
        hidden, weight, labels, grad_loss = _inputs(300, 64, V_SMALL)
        recomputed = _fused(hidden, weight, labels, grad_loss, 256, saved=0)
        kept = _fused(hidden, weight, labels, grad_loss, 256, saved=saved)
        for a, b in zip(kept, recomputed):
            assert torch.equal(a, b)

    def test_saved_logit_chunks_set_the_memory_kept_for_the_backward(self, tp_group):
        tokens, vocab_size, chunk = 512, 8192, 1024
        hidden, weight, labels, _ = _inputs(tokens, 64, vocab_size)
        weight.requires_grad_()
        chunk_bytes = tokens * chunk * hidden.element_size()
        for saved in (0, 3, 8, 20):
            torch.cuda.synchronize()
            base = torch.cuda.memory_allocated()
            loss = _op(hidden, weight, labels, chunk, saved)
            kept = torch.cuda.memory_allocated() - base
            assert min(saved, vocab_size // chunk) * chunk_bytes <= kept < min(saved, 8) * chunk_bytes + (1 << 20)
            del loss

    def test_out_of_vocabulary_labels_follow_the_unfused_convention(self, tp_group):
        """Loss log(sum exp(x - max)); gradient softmax * grad_loss with no one-hot term."""
        hidden, weight, _, grad_loss = _inputs(8, 32, 50)
        labels = 50 + torch.arange(8, device="cuda")
        fused = _fused(hidden, weight, labels, grad_loss, 16)
        unfused = _unfused(hidden, weight, labels, grad_loss, tp_group)
        torch.testing.assert_close(fused[0], unfused[0], rtol=0, atol=1e-5)
        for fused_grad, unfused_grad in zip(fused[1:], unfused[1:]):
            assert _rel(fused_grad, unfused_grad) < 2e-3

    def test_masked_tokens_get_exactly_zero_gradient(self, tp_group):
        hidden, weight, labels, _ = _inputs(64, 32, 200)
        grad_loss = torch.zeros(64, device="cuda")
        grad_loss[10] = 1.0
        _, fused_grad_hidden, fused_grad_weight = _fused(hidden, weight, labels, grad_loss, 64)
        _, _, unfused_grad_weight = _unfused(hidden, weight, labels, grad_loss, tp_group)
        masked = torch.arange(64, device="cuda") != 10
        assert torch.count_nonzero(fused_grad_hidden[masked]) == 0
        assert torch.count_nonzero(fused_grad_hidden[10]) > 0
        assert _rel(fused_grad_weight, unfused_grad_weight) < 2e-3

    def test_fp32_inputs(self, tp_group):
        hidden, weight, labels, grad_loss = _inputs(96, 32, 500, dtype=torch.float32)
        fused = _fused(hidden, weight, labels, grad_loss, 128)
        unfused = _unfused(hidden, weight, labels, grad_loss, tp_group)
        assert fused[1].dtype == fused[2].dtype == torch.float32
        for fused_value, unfused_value in zip(fused, unfused):
            torch.testing.assert_close(fused_value, unfused_value, rtol=1e-4, atol=1e-6)

    def test_leading_dimensions_are_kept(self, tp_group):
        """[sequence, batch, hidden] in and [sequence, batch] loss out, as at micro-batch 2."""
        hidden, weight, labels, _ = _inputs(2 * 40, 32, 300)
        loss = _op(hidden.view(40, 2, 32), weight, labels.view(40, 2), 100)
        assert loss.shape == (40, 2)
        assert torch.equal(loss.view(-1), _op(hidden, weight, labels, 100))

    def test_is_deterministic(self, tp_group):
        hidden, weight, labels, grad_loss = _inputs(300, 64, V_SMALL)
        first = _fused(hidden, weight, labels, grad_loss, 256)
        second = _fused(hidden, weight, labels, grad_loss, 256)
        for a, b in zip(first, second):
            assert torch.equal(a, b)

    @pytest.mark.parametrize("main_grad_dtype", [torch.bfloat16, torch.float32])
    def test_gradient_accumulation_fusion_matches_the_output_layer(self, tp_group, main_grad_dtype):
        """Accumulates into main_grad as ColumnParallelLinear's backward does, with its DDP handshake."""
        from megatron.core.tensor_parallel.cross_entropy import vocab_parallel_cross_entropy
        from megatron.core.tensor_parallel.layers import linear_with_grad_accumulation_and_async_allreduce

        hidden, weight, labels, grad_loss = _inputs(300, 64, V_SMALL)
        start = torch.randn(weight.shape, device="cuda").to(main_grad_dtype)

        def run(fused):
            w = torch.nn.Parameter(weight.clone())
            w.main_grad = start.clone()
            w.grad_added_to_main_grad = False
            h = hidden.clone().requires_grad_()
            if fused:
                loss = _op(h, w, labels, 256, fusion=True)
            else:
                logits = linear_with_grad_accumulation_and_async_allreduce(
                    h, w, None, True, False, False, tp_group=tp_group
                )
                loss = vocab_parallel_cross_entropy(logits, labels, tp_group=tp_group)
            loss.backward(grad_loss)
            return w, h.grad

        fused_weight, fused_grad_hidden = run(fused=True)
        unfused_weight, unfused_grad_hidden = run(fused=False)
        assert fused_weight.grad_added_to_main_grad
        assert fused_weight.main_grad.dtype == main_grad_dtype
        fused_increment = fused_weight.main_grad.double() - start.double()
        unfused_increment = unfused_weight.main_grad.double() - start.double()
        assert _rel(fused_increment, unfused_increment) < 2e-3
        assert _rel(fused_grad_hidden, unfused_grad_hidden) < 2e-3

    def test_frozen_weight_gets_no_gradient(self, tp_group):
        hidden, weight, labels, grad_loss = _inputs(100, 32, 300)
        h = hidden.clone().requires_grad_()
        _op(h, weight, labels, 64, fusion=True).backward(grad_loss)
        _, unfused_grad_hidden, _ = _unfused(hidden, weight, labels, grad_loss, tp_group)
        assert weight.grad is None
        assert _rel(h.grad, unfused_grad_hidden) < 2e-3

    def test_frozen_hidden_gets_no_gradient(self, tp_group):
        hidden, weight, labels, grad_loss = _inputs(100, 32, 300)
        w = weight.clone().requires_grad_()
        _op(hidden, w, labels, 64).backward(grad_loss)
        _, _, unfused_grad_weight = _unfused(hidden, weight, labels, grad_loss, tp_group)
        assert hidden.grad is None
        assert _rel(w.grad, unfused_grad_weight) < 2e-3

    def test_never_allocates_the_full_vocabulary_logits(self, tp_group):
        """Without saved chunks, peak memory follows one chunk, not the [tokens, vocab] logits."""
        tokens, vocab_size, chunk = 4096, 32768, 1024
        hidden, weight, labels, grad_loss = _inputs(tokens, 128, vocab_size)
        full_logits_bytes = tokens * vocab_size * hidden.element_size()

        def peak_bytes(run):
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            base = torch.cuda.memory_allocated()
            run()
            torch.cuda.synchronize()
            return torch.cuda.max_memory_allocated() - base

        fused_peak = peak_bytes(lambda: _fused(hidden, weight, labels, grad_loss, chunk))
        unfused_peak = peak_bytes(lambda: _unfused(hidden, weight, labels, grad_loss, tp_group))
        assert fused_peak < full_logits_bytes / 4
        assert unfused_peak > 2 * full_logits_bytes

    @pytest.mark.parametrize(
        "chunk,saved,message",
        [
            (0, 0, "vocab_chunk_size must be positive"),
            (-8, 0, "vocab_chunk_size must be positive"),
            (16, -1, "saved_logit_chunks must be non-negative"),
        ],
    )
    def test_rejects_invalid_chunking(self, tp_group, chunk, saved, message):
        hidden, weight, labels, _ = _inputs(8, 16, 32)
        with pytest.raises(ValueError, match=message):
            _op(hidden, weight, labels, chunk, saved)

    def test_rejects_mismatched_shapes(self, tp_group):
        hidden, weight, labels, _ = _inputs(8, 16, 32)
        with pytest.raises(ValueError, match="do not form"):
            _op(hidden, weight[:, :8], labels, 16)
        with pytest.raises(ValueError, match="do not match"):
            _op(hidden, weight, labels[:4], 16)

    def test_gradient_accumulation_fusion_needs_main_grad(self, tp_group):
        hidden, weight, labels, _ = _inputs(8, 16, 32)
        with pytest.raises(RuntimeError, match="main_grad"):
            _op(hidden, weight.requires_grad_(), labels, 16, fusion=True)

    def test_rejects_megatron_fsdp_parameters(self, tp_group):
        hidden, weight, labels, _ = _inputs(8, 16, 32)
        weight.__fsdp_param__ = True
        with pytest.raises(NotImplementedError, match="Megatron-FSDP"):
            _op(hidden, weight, labels, 16)


def _ddp_weight(shape, main_grad):
    """A parameter as Megatron's DistributedDataParallel prepares it for gradient-accumulation fusion."""
    weight = torch.nn.Parameter(torch.zeros(shape, dtype=torch.bfloat16, device="cuda"))
    weight.main_grad = main_grad
    weight.grad_added_to_main_grad = False
    return weight


@pytest.mark.run_only_on("GPU")
class TestWeightGradientAccumulation:
    """The gradient-accumulation-fusion weight-gradient step, against a float64 reference.

    ColumnParallelLinear's backward, the fused loss's backward and the pipeline drain's deferred
    embedding GEMMs (``drain_embedding_wgrad_compute``) all accumulate through
    ``accumulate_wgrad_into_main_grad``; the two backwards then return
    ``wgrad_after_main_grad_accumulation`` as the weight's gradient.
    """

    @pytest.mark.parametrize("main_grad_dtype,tolerance", [(torch.float32, 1e-6), (torch.bfloat16, 1e-2)])
    def test_accumulates_the_weight_gradient_into_main_grad(self, main_grad_dtype, tolerance):
        from megatron.core.tensor_parallel.layers import accumulate_wgrad_into_main_grad

        total_input, grad_output = (torch.randn(96, n, device="cuda").to(torch.bfloat16) for n in (48, 80))
        start = torch.randn(80, 48, device="cuda").to(main_grad_dtype)
        main_grad = start.clone()
        accumulate_wgrad_into_main_grad(total_input, grad_output, main_grad)
        assert main_grad.dtype == main_grad_dtype
        assert _rel(main_grad, start.double() + grad_output.double().t() @ total_input.double()) < tolerance

    def test_refuses_an_unsupported_main_grad_dtype(self):
        from megatron.core.tensor_parallel.layers import accumulate_wgrad_into_main_grad

        total_input, grad_output = (torch.randn(8, n, device="cuda").to(torch.bfloat16) for n in (4, 6))
        main_grad = torch.zeros(6, 4, dtype=torch.float64, device="cuda")
        with pytest.raises(RuntimeError, match="Unsupported gradient type"):
            accumulate_wgrad_into_main_grad(total_input, grad_output, main_grad)

    @pytest.mark.parametrize("zero_out_wgrad", [False, True])
    def test_a_ddp_weight_gets_a_placeholder_gradient_and_the_accumulated_flag(self, zero_out_wgrad):
        from megatron.core.tensor_parallel.layers import wgrad_after_main_grad_accumulation

        weight = _ddp_weight((6, 4), torch.zeros(6, 4, device="cuda"))
        weight.zero_out_wgrad = zero_out_wgrad
        grad = wgrad_after_main_grad_accumulation(weight, torch.bfloat16)
        assert weight.grad_added_to_main_grad
        assert grad.shape == weight.main_grad.shape and grad.dtype == torch.bfloat16
        if zero_out_wgrad:
            assert torch.count_nonzero(grad) == 0

    def test_a_weight_outside_ddp_gets_no_gradient(self):
        from megatron.core.tensor_parallel.layers import wgrad_after_main_grad_accumulation

        weight = torch.nn.Parameter(torch.zeros(6, 4, dtype=torch.bfloat16, device="cuda"))
        weight.main_grad = torch.zeros(6, 4, device="cuda")
        assert wgrad_after_main_grad_accumulation(weight, torch.bfloat16) is None
        assert not hasattr(weight, "grad_added_to_main_grad")

    @pytest.mark.parametrize("main_grad_dtype,tolerance", [(torch.float32, 1e-6), (torch.bfloat16, 1e-2)])
    def test_the_pipeline_drain_accumulates_every_deferred_micro_batch(self, tp_group, main_grad_dtype, tolerance):
        from megatron.core.utils import drain_embedding_wgrad_compute

        config = _layer_config(
            pipeline_model_parallel_size=2,
            pipeline_dtype=torch.bfloat16,
            gradient_accumulation_fusion=True,
            defer_embedding_wgrad_compute=True,
        )
        inputs = [torch.randn(10, 2, 32, device="cuda").to(torch.bfloat16) for _ in range(3)]
        grad_outputs = [torch.randn(10, 2, V_SMALL, device="cuda").to(torch.bfloat16) for _ in range(3)]
        start = torch.randn(V_SMALL, 32, device="cuda").to(main_grad_dtype)
        weight = _ddp_weight((V_SMALL, 32), start.clone())
        drain_embedding_wgrad_compute(config, list(inputs), list(grad_outputs), weight, tp_group)
        expected = start.double() + sum(
            g.reshape(-1, V_SMALL).double().t() @ x.reshape(-1, 32).double() for x, g in zip(inputs, grad_outputs)
        )
        assert _rel(weight.main_grad, expected) < tolerance


def _layer_config(**overrides):
    from megatron.core.transformer.transformer_config import TransformerConfig

    return TransformerConfig(
        **{"num_layers": 2, "hidden_size": 32, "num_attention_heads": 4, "params_dtype": torch.bfloat16}
        | LINEAR_FUSION
        | overrides
    )


def _output_layer(cls, config, bias=False, tp_group=None):
    return cls(
        config.hidden_size,
        V_SMALL,
        config=config,
        init_method=torch.nn.init.xavier_normal_,
        bias=bias,
        skip_bias_add=False,
        gather_output=False,
        tp_group=tp_group,
    )


@pytest.mark.run_only_on("GPU")
class TestLinearCrossEntropyModule:
    def test_without_labels_it_is_column_parallel_linear(self, tp_group):
        from megatron.core.tensor_parallel.layers import ColumnParallelLinear
        from megatron.core.transformer.linear_cross_entropy import LinearCrossEntropyModule

        config = _layer_config()
        fused = _output_layer(LinearCrossEntropyModule, config)
        plain = _output_layer(ColumnParallelLinear, config)
        plain.load_state_dict(fused.state_dict(), strict=True)
        hidden = torch.randn(6, 2, 32, device="cuda", dtype=torch.bfloat16)
        fused_logits, fused_bias = fused(hidden)
        plain_logits, plain_bias = plain(hidden)
        assert torch.equal(fused_logits, plain_logits)
        assert fused_bias is None and plain_bias is None

    @pytest.mark.parametrize("micro_batch_size", [1, 2])
    def test_with_labels_it_returns_the_language_model_loss(self, tp_group, micro_batch_size):
        """[batch, sequence], as compute_language_model_loss returns for this layer's logits."""
        from megatron.core.tensor_parallel.cross_entropy import vocab_parallel_cross_entropy
        from megatron.core.transformer.linear_cross_entropy import LinearCrossEntropyModule

        layer = _output_layer(LinearCrossEntropyModule, _layer_config())
        hidden = torch.randn(40, micro_batch_size, 32, device="cuda", dtype=torch.bfloat16)
        labels = torch.randint(0, V_SMALL, (micro_batch_size, 40), device="cuda")
        loss = layer(hidden, labels=labels)
        logits, _ = layer(hidden)
        expected = vocab_parallel_cross_entropy(logits, labels.t().contiguous(), tp_group=tp_group).t()
        assert loss.shape == (micro_batch_size, 40) and loss.is_contiguous()
        torch.testing.assert_close(loss, expected, rtol=0, atol=1e-5)

    def test_a_supplied_weight_is_used_and_shape_checked(self, tp_group):
        """Tied embeddings hand the output layer the embedding weight."""
        from megatron.core.tensor_parallel.cross_entropy import vocab_parallel_cross_entropy
        from megatron.core.transformer.linear_cross_entropy import LinearCrossEntropyModule

        layer = _output_layer(LinearCrossEntropyModule, _layer_config())
        hidden = torch.randn(10, 1, 32, device="cuda", dtype=torch.bfloat16)
        labels = torch.randint(0, V_SMALL, (1, 10), device="cuda")
        embedding = torch.randn_like(layer.weight)
        loss = layer(hidden, weight=embedding, labels=labels)
        logits, _ = layer(hidden, weight=embedding)
        expected = vocab_parallel_cross_entropy(logits, labels.t().contiguous(), tp_group=tp_group).t()
        torch.testing.assert_close(loss, expected, rtol=0, atol=1e-5)
        with pytest.raises(RuntimeError, match="supplied weight's shape"):
            layer(hidden, weight=embedding[:, :16], labels=labels)

    def test_forward_hooks_run_for_the_loss(self, tp_group):
        """The distributed optimizer waits for a layer's parameter all-gather in a forward pre-hook."""
        from megatron.core.transformer.linear_cross_entropy import LinearCrossEntropyModule

        layer = _output_layer(LinearCrossEntropyModule, _layer_config())
        calls = []
        layer.register_forward_pre_hook(lambda module, args: calls.append(module))
        hidden = torch.randn(4, 1, 32, device="cuda", dtype=torch.bfloat16)
        layer(hidden, labels=torch.zeros(1, 4, dtype=torch.long, device="cuda"))
        assert calls == [layer]

    @pytest.mark.parametrize(
        "config_overrides,bias,refused",
        [
            ({}, True, "a bias"),
            ({"cpu_offloading": True, "cpu_offloading_num_layers": 0}, False, "cpu_offloading"),
            (
                {
                    "pipeline_model_parallel_size": 2,
                    "pipeline_dtype": torch.bfloat16,
                    "gradient_accumulation_fusion": True,
                    "defer_embedding_wgrad_compute": True,
                },
                False,
                "defer_embedding_wgrad_compute",
            ),
        ],
    )
    def test_refuses_unsupported_configurations_for_the_loss(self, tp_group, config_overrides, bias, refused):
        """The layer builds (a checkpoint must stay loadable anywhere); only the loss path refuses."""
        from megatron.core.transformer.linear_cross_entropy import LinearCrossEntropyModule

        layer = _output_layer(LinearCrossEntropyModule, _layer_config(**config_overrides), bias=bias)
        hidden = torch.randn(4, 1, 32, device="cuda", dtype=torch.bfloat16)
        with pytest.raises(ValueError, match=refused):
            layer(hidden, labels=torch.zeros(1, 4, dtype=torch.long, device="cuda"))


def _output_layer_over_a_tensor_parallel_vocabulary(rank, init_file):
    """One of two CPU ranks: the layer builds and gives its logits shard, but refuses the loss."""
    from megatron.core.transformer.linear_cross_entropy import LinearCrossEntropyModule

    torch.distributed.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=2)
    try:
        config = _layer_config(tensor_model_parallel_size=2, use_cpu_initialization=True)
        layer = _output_layer(LinearCrossEntropyModule, config, tp_group=torch.distributed.new_group([0, 1]))
        hidden = torch.randn(4, 1, 32, dtype=torch.bfloat16)
        logits, _ = layer(hidden)
        assert logits.shape == (4, 1, V_SMALL // 2), logits.shape
        try:
            layer(hidden, labels=torch.zeros(1, 4, dtype=torch.long))
        except ValueError as error:
            assert "tensor-parallel size > 1" in str(error), error
        else:
            raise AssertionError("the loss over a tensor-parallel vocabulary was accepted")
    finally:
        torch.distributed.destroy_process_group()


def test_tensor_parallel_vocabulary_gives_logits_but_refuses_the_loss(tmp_path):
    torch.multiprocessing.spawn(
        _output_layer_over_a_tensor_parallel_vocabulary, args=(str(tmp_path / "rendezvous"),), nprocs=2
    )


def _hybrid_model(pattern="M*-", **config_overrides):
    return hybrid_model(
        pattern,
        vocab_size=V_SMALL,
        max_sequence_length=SEQ,
        pre_process=True,
        num_layers=3,
        hidden_size=256,
        num_attention_heads=4,
        **config_overrides,
    )


def _model_inputs(micro_batch_size):
    generator = torch.Generator(device="cuda").manual_seed(micro_batch_size)
    input_ids = torch.randint(0, V_SMALL, (micro_batch_size, SEQ), device="cuda", generator=generator)
    position_ids = torch.arange(SEQ, device="cuda").repeat(micro_batch_size, 1)
    attention_mask = torch.ones((micro_batch_size, 1, SEQ, SEQ), dtype=torch.bool, device="cuda")
    labels = torch.randint(0, V_SMALL, (micro_batch_size, SEQ), device="cuda", generator=generator)
    loss_mask = (torch.rand(micro_batch_size, SEQ, device="cuda", generator=generator) > 0.2).float()
    return dict(input_ids=input_ids, position_ids=position_ids, attention_mask=attention_mask), labels, loss_mask


def _checkpoint_layout(model):
    """Name -> (shape, dtype) of every tensor, and key -> (global shape, dtype) of every sharded one."""
    tensors = {
        name: (tuple(value.shape), value.dtype)
        for name, value in model.state_dict().items()
        if isinstance(value, torch.Tensor)
    }
    sharded = {
        name: (getattr(value, "global_shape", None), getattr(value, "dtype", None))
        for name, value in model.sharded_state_dict().items()
    }
    return tensors, sharded, set(model.state_dict())


@pytest.mark.run_only_on("GPU")
class TestHybridModelLinearCrossEntropy:
    def test_default_config_keeps_the_plain_output_layer(self, tp_group):
        from megatron.core.tensor_parallel.layers import ColumnParallelLinear

        model = _hybrid_model()
        assert type(model.output_layer) is ColumnParallelLinear
        assert not model.fuse_linear_cross_entropy

    def test_checkpoint_layout_is_unchanged(self, tp_group):
        """Same names, shapes, dtypes and distributed-checkpoint keys; state dicts load both ways."""
        from megatron.core.transformer.linear_cross_entropy import LinearCrossEntropyModule

        plain = _hybrid_model()
        fused = _hybrid_model(**LINEAR_FUSION)
        assert type(fused.output_layer) is LinearCrossEntropyModule
        assert _checkpoint_layout(fused) == _checkpoint_layout(plain)
        fused.load_state_dict(plain.state_dict(), strict=True)
        plain.load_state_dict(fused.state_dict(), strict=True)

    @pytest.mark.parametrize("micro_batch_size", [1, 2])
    @pytest.mark.parametrize("saved_logit_chunks", [0, 100])
    def test_loss_and_gradients_match_the_unfused_model(
        self, tp_group, fp32_matmul, micro_batch_size, saved_logit_chunks
    ):
        """The model wiring: same loss to fp32 rounding, and every gradient within summation order.

        grad_hidden = sum_v (softmax - onehot)_v * W_v cancels heavily (the coefficients sum to
        zero per token), so a different fp32 summation order alone moves it, and every gradient
        upstream of it, by up to ~5e-4 relative (the unfused model is bitwise repeatable, so this
        is the whole difference). A wiring error (wrong weight, labels or layout) is O(1). The
        op-level tests bound the accuracy against fp32 and exact references.
        """
        plain = _hybrid_model()
        fused = _hybrid_model(**LINEAR_FUSION, cross_entropy_fusion_saved_logit_chunks=saved_logit_chunks)
        fused.load_state_dict(plain.state_dict(), strict=True)
        inputs, labels, loss_mask = _model_inputs(micro_batch_size)

        losses = {}
        for name, model in (("plain", plain), ("fused", fused)):
            loss = model(**inputs, labels=labels)
            (loss * loss_mask).sum().backward()
            losses[name] = loss.detach()

        assert losses["fused"].shape == (micro_batch_size, SEQ)
        assert _rel(losses["fused"], losses["plain"]) < 1e-6
        fused_params = dict(fused.named_parameters())
        for name, param in plain.named_parameters():
            assert _rel(fused_params[name].grad, param.grad) < 5e-3, name

    def test_without_labels_returns_the_same_logits(self, tp_group):
        plain = _hybrid_model()
        fused = _hybrid_model(**LINEAR_FUSION)
        fused.load_state_dict(plain.state_dict(), strict=True)
        inputs, _, _ = _model_inputs(2)
        assert torch.equal(fused(**inputs), plain(**inputs))

    def test_the_loss_runs_the_output_layer_hooks(self, tp_group):
        fused = _hybrid_model(**LINEAR_FUSION)
        calls = []
        fused.output_layer.register_forward_pre_hook(lambda module, args: calls.append(module))
        inputs, labels, _ = _model_inputs(1)
        fused(**inputs, labels=labels)
        assert calls == [fused.output_layer]

    def test_refuses_multi_token_prediction(self, tp_group):
        with pytest.raises(ValueError, match="multi-token prediction"):
            _hybrid_model(pattern="M*-/M-", mtp_num_layers=1, **LINEAR_FUSION)

    def test_refuses_mup(self, tp_group):
        with pytest.raises(ValueError, match="MuP"):
            _hybrid_model(use_mup=True, **LINEAR_FUSION)

    def test_logits_cannot_take_the_linear_loss(self, tp_group):
        """A model routing logits into compute_language_model_loss under 'linear' fails loudly."""
        fused = _hybrid_model(**LINEAR_FUSION)
        logits = torch.randn(SEQ, 1, V_SMALL, device="cuda")
        with pytest.raises(ValueError, match="inside the output layer"):
            fused.compute_language_model_loss(torch.zeros(1, SEQ, dtype=torch.long, device="cuda"), logits)


class TestConfigPlumbing:
    def test_hydra_overrides_select_the_fusion(self):
        from megatron.bridge.recipes.nemotronh.nemotron_3_nano import nemotron_3_nano_pretrain_config
        from megatron.bridge.training.utils.omegaconf_utils import process_config_with_overrides

        cfg = process_config_with_overrides(
            nemotron_3_nano_pretrain_config(),
            str(BASELINE),
            [
                "model.cross_entropy_loss_fusion=true",
                "model.cross_entropy_fusion_impl=linear",
                "model.cross_entropy_fusion_vocab_chunk_size=4096",
                "model.cross_entropy_fusion_saved_logit_chunks=32",
            ],
        )
        assert cfg.model.cross_entropy_loss_fusion is True
        assert cfg.model.cross_entropy_fusion_impl == "linear"
        assert cfg.model.cross_entropy_fusion_vocab_chunk_size == 4096
        assert cfg.model.cross_entropy_fusion_saved_logit_chunks == 32

    @pytest.mark.parametrize("config_path", [BASELINE, BASELINE_BENCHMARK], ids=["production", "baseline_benchmark"])
    def test_production_configs_keep_the_unfused_loss(self, config_path):
        from megatron.bridge.recipes.nemotronh.nemotron_3_nano import nemotron_3_nano_pretrain_config
        from tests.unit_tests.campaign_config import merge_onto_recipe

        cfg = merge_onto_recipe(config_path, nemotron_3_nano_pretrain_config)
        assert cfg.model.cross_entropy_loss_fusion is False
        assert cfg.model.cross_entropy_fusion_impl == "native"
