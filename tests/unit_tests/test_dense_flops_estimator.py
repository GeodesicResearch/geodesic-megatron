"""Unit tests for scripts/dense_flops_estimator.py.

Pure python, no torch and no GPU. Three kinds of check:

1. **Hand-computed toy cases.** A decoder small enough that every term is worked out on paper and written as a
   literal, with a window shorter and longer than the sequence. A formula edit fails with the arithmetic in view.
2. **Real Olmo 3 pins.** GFLOP per token for the 7B and the 32B, read from the HF ``config.json`` on the cluster
   and pinned to two decimals. The tests skip where the model directory is absent.
3. **Properties.** Window layers against full layers, causal against non-causal, packed documents against one
   long document, and the refusals for configs the estimator cannot count.

The toy decoder has hidden size 8, 4 layers (sliding, full, sliding, full) with a window of 3, 2 heads of 4 with one
KV head (q = 8, kv = 4), FFN size 6 and vocabulary 30, over a sequence of 4 tokens. A layer's forward FLOPs are
then 2*4*8*16 = 1024 for q/k/v, 2*4*8*8 = 512 for the output projection and 6*4*8*6 = 1152 for the MLP, and the
output layer is 2*4*8*30 = 1920.
"""

import json
from pathlib import Path

import pytest
import yaml
from scripts.dense_flops_estimator import (
    CAUSAL,
    FULL_ATTENTION,
    NON_CAUSAL,
    SLIDING_ATTENTION,
    DenseArchSpec,
    attended_pairs,
    compute_flops,
    count_params,
    document_pairs,
    head_flops,
    layer_flops,
    layer_split,
    pack_documents,
)
from scripts.flops_common import FlopReport, RunSpec, achieved_throughput


BASE_COPIES = yaml.safe_load(
    (Path(__file__).resolve().parents[2] / "configs" / "models" / "olmo3_base_copies.yaml").read_text()
)
OLMO_7B = Path(BASE_COPIES["olmo3_7b"])
OLMO_32B = Path(BASE_COPIES["olmo3_32b"])

TOY_SEQ = 4
TOY_HF_CONFIG = {
    "model_type": "olmo3",
    "hidden_size": 8,
    "num_hidden_layers": 4,
    "num_attention_heads": 2,
    "num_key_value_heads": 1,
    "head_dim": 4,
    "intermediate_size": 6,
    "vocab_size": 30,
    "hidden_act": "silu",
    "tie_word_embeddings": False,
    "sliding_window": 3,
    "layer_types": [SLIDING_ATTENTION, FULL_ATTENTION, SLIDING_ATTENTION, FULL_ATTENTION],
}


def make_run(seq_length: int, **overrides) -> RunSpec:
    """A run of ``seq_length`` tokens with the defaults ``RunSpec.from_yaml`` gives an unconfigured YAML."""
    kwargs = dict(
        config_path="<test>",
        global_batch_size=2,
        seq_length=seq_length,
        recompute_granularity=None,
        recompute_modules=(),
        mtp_num_layers=0,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        expert_model_parallel_size=1,
        vocab_size_override=None,
        make_vocab_size_divisible_by=128,
        should_pad_vocab=True,
    )
    kwargs.update(overrides)
    return RunSpec(**kwargs)


def toy_run(**overrides) -> RunSpec:
    """The toy run: 4 tokens and an unpadded vocabulary, so the output layer is 30 wide."""
    return make_run(TOY_SEQ, **{"make_vocab_size_divisible_by": 1, "should_pad_vocab": False, **overrides})


def toy_arch(**config_changes) -> DenseArchSpec:
    return DenseArchSpec.from_hf_config({**TOY_HF_CONFIG, **config_changes}, name="toy")


def read_hf_config(model_dir: Path) -> dict:
    path = model_dir / "config.json"
    if not path.is_file():
        pytest.skip(f"{path} is not on this machine")
    return json.loads(path.read_text())


def gflop_per_token(model_dir: Path, seq_length: int, attention_mask: str, all_full=False, documents=None) -> float:
    """Model GFLOP per token (forward + backward) of a real Olmo 3 config, optionally with every layer full."""
    cfg = read_hf_config(model_dir)
    if all_full:
        cfg = {key: value for key, value in cfg.items() if key not in ("layer_types", "sliding_window")}
    arch = DenseArchSpec.from_hf_config(cfg, name=model_dir.name)
    report = compute_flops(arch, make_run(seq_length), attention_mask, document_lengths=documents)
    return report.model_flops_per_token / 1e9


# --------------------------------------------------------------------------------------
# Toy decoder, by hand
# --------------------------------------------------------------------------------------


def test_toy_architecture_and_layer_split():
    arch = toy_arch()
    assert (arch.q_dim, arch.kv_dim, arch.head_dim) == (8, 4, 4)
    assert (arch.full_layers, arch.window_layers, arch.sliding_window) == (2, 2, 3)
    assert arch.layer_census() == {FULL_ATTENTION: 2, SLIDING_ATTENTION: 2}


def test_toy_layer_flops_by_hand():
    arch = toy_arch()
    full = layer_flops(arch, TOY_SEQ, (TOY_SEQ,), None, CAUSAL)
    window = layer_flops(arch, TOY_SEQ, (TOY_SEQ,), 3, CAUSAL)
    # Causal over 4 tokens: 1+2+3+4 = 10 pairs for a full layer and 1+2+3+3 = 9 for a window of 3.
    assert full == {"qkv_proj": 1024, "o_proj": 512, "core_attn": 4 * 8 * 10, "mlp": 1152}
    assert window == {"qkv_proj": 1024, "o_proj": 512, "core_attn": 4 * 8 * 9, "mlp": 1152}
    assert head_flops(arch, TOY_SEQ, 30) == 1920


def test_toy_causal_window_shorter_than_the_sequence():
    report = compute_flops(toy_arch(), toy_run(), CAUSAL)
    # 2 full layers (1024+512+1152+320) + 2 window layers (1024+512+1152+288) + the output layer.
    assert report.forward_per_sample == 2 * 3008 + 2 * 2976 + 1920 == 13888
    assert report.model_flops_per_sample == 3 * 13888
    assert report.model_flops_per_token == 10416
    # The same total from the closed form per token: projections + MLP + output layer + attention scores, with the
    # mean keys per query 10/4 for a full layer and 9/4 for a window layer.
    assert 10416 == 6 * 4 * (8 * (8 + 2 * 4) + 8 * 8) + 18 * 4 * 8 * 6 + 6 * 8 * 30 + 12 * 8 * (2 * 10 / 4 + 2 * 9 / 4)


def test_toy_causal_window_longer_than_the_sequence_is_a_full_layer():
    report = compute_flops(toy_arch(sliding_window=8), toy_run(), CAUSAL)
    assert report.forward_per_sample == 4 * 3008 + 1920 == 13952
    assert report.model_flops_per_token == 10464
    all_full = compute_flops(toy_arch(layer_types=[FULL_ATTENTION] * 4), toy_run(), CAUSAL)
    assert report.forward_per_sample == all_full.forward_per_sample


def test_toy_non_causal():
    # Full layers see all 4 keys (16 pairs, 512 FLOPs); window layers see min(3, 4) = 3 (12 pairs, 384 FLOPs).
    shorter = compute_flops(toy_arch(), toy_run(), NON_CAUSAL)
    assert shorter.forward_per_sample == 2 * (2688 + 512) + 2 * (2688 + 384) + 1920 == 14464
    assert shorter.model_flops_per_token == 10848
    longer = compute_flops(toy_arch(sliding_window=8), toy_run(), NON_CAUSAL)
    assert longer.forward_per_sample == 4 * (2688 + 512) + 1920 == 14720
    assert longer.model_flops_per_token == 11040


def test_toy_packed_documents():
    # Two documents of 2 tokens: 3 + 3 = 6 pairs in both kinds of layer (the window of 3 never bites).
    equal = compute_flops(toy_arch(), toy_run(), CAUSAL, document_lengths=[2, 2])
    assert equal.forward_per_sample == 4 * (2688 + 4 * 8 * 6) + 1920 == 13440
    assert equal.model_flops_per_token == 10080
    # Documents of 1 and 3 tokens with a window of 2: a full layer scores 1 + 6 = 7 pairs, a window layer 1 + (1+2+2).
    uneven = compute_flops(toy_arch(sliding_window=2), toy_run(), CAUSAL, document_lengths=[1, 3])
    assert uneven.forward_per_sample == 2 * (2688 + 4 * 8 * 7) + 2 * (2688 + 4 * 8 * 6) + 1920 == 13504
    assert uneven.model_flops_per_token == 10128


def test_one_document_filling_the_sequence_is_the_default():
    assert (
        compute_flops(toy_arch(), toy_run(), CAUSAL, document_lengths=[TOY_SEQ]).forward_per_sample
        == compute_flops(toy_arch(), toy_run(), CAUSAL).forward_per_sample
    )


def test_report_is_a_shared_flop_report():
    report = compute_flops(toy_arch(), toy_run(), CAUSAL)
    assert isinstance(report, FlopReport)
    assert report.attention_mask == CAUSAL
    assert report.padded_vocab_size == 30
    assert report.model_flops_per_iter == 10416 * TOY_SEQ * 2  # global_batch_size = 2
    assert report.recompute_per_sample == 0
    assert report.hardware_flops_per_sample == report.model_flops_per_sample
    assert set(report.forward_by_layer_type) == {FULL_ATTENTION, SLIDING_ATTENTION}
    # The throughput arithmetic is the shared one: 83,328 FLOPs in 1 ns on one GPU, at a 100 TFLOP/s peak.
    tflops, fraction = achieved_throughput(report.model_flops_per_iter, 1e-9, 1, 100.0)
    assert (tflops, fraction) == pytest.approx((83.328, 0.83328))


def test_a_model_with_only_full_layers_reports_one_kind():
    cfg = {key: value for key, value in TOY_HF_CONFIG.items() if key not in ("layer_types", "sliding_window")}
    arch = DenseArchSpec.from_hf_config(cfg, name="all-full")
    assert (arch.full_layers, arch.window_layers, arch.sliding_window) == (4, 0, None)
    assert set(compute_flops(arch, toy_run(), CAUSAL).forward_by_layer_type) == {FULL_ATTENTION}


# --------------------------------------------------------------------------------------
# Keys per query
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("length, window", [(10, 4), (4096, 4096), (32768, 4096), (100, 99)])
def test_causal_pairs_match_the_closed_form_mean_keys(length, window):
    """A window layer's mean keys per query is W - W(W-1)/(2s) above the window, and (s+1)/2 up to it."""
    mean_keys = document_pairs(length, window, CAUSAL) / length
    if length > window:
        assert mean_keys == pytest.approx(window - window * (window - 1) / (2 * length))
    else:
        assert mean_keys == pytest.approx((length + 1) / 2)


def test_causal_pairs_are_the_sum_over_positions():
    for length, window in ((10, 4), (7, 7), (7, 20), (5, 1)):
        assert document_pairs(length, window, CAUSAL) == sum(min(i + 1, window) for i in range(length))
        assert document_pairs(length, None, CAUSAL) == sum(i + 1 for i in range(length))


def test_non_causal_pairs_are_the_keys_every_query_sees():
    assert document_pairs(10, None, NON_CAUSAL) == 100
    assert document_pairs(10, 4, NON_CAUSAL) == 40
    assert document_pairs(10, 40, NON_CAUSAL) == 100


def test_non_causal_over_causal_is_twice_over_s_plus_one_for_a_full_layer():
    for length in (4, 4096, 32768):
        ratio = document_pairs(length, None, NON_CAUSAL) / document_pairs(length, None, CAUSAL)
        assert ratio == pytest.approx(2 * length / (length + 1))


def test_pairs_of_a_pack_sum_over_its_documents():
    # A window of 4: the 3-token document scores 1+2+3 = 6 pairs and the 5-token one 1+2+3+4+4 = 14.
    assert attended_pairs([3, 5], 4, CAUSAL) == document_pairs(3, 4, CAUSAL) + document_pairs(5, 4, CAUSAL) == 6 + 14
    assert attended_pairs([3, 5], None, NON_CAUSAL) == 9 + 25


def test_unknown_attention_mask_is_rejected():
    with pytest.raises(ValueError, match="unknown attention mask mode 'block-diagonal'"):
        document_pairs(4, None, "block-diagonal")
    with pytest.raises(ValueError, match="unknown attention mask mode"):
        compute_flops(toy_arch(), toy_run(), "block-diagonal")


def test_documents_must_be_positive_and_fill_the_sequence():
    assert pack_documents(4, None) == (4,)
    assert pack_documents(4, [1, 3]) == (1, 3)
    for bad in ([2, 3], [4, 0], [5, -1], []):
        with pytest.raises(ValueError, match="document_lengths must be positive and sum to seq_length 4"):
            pack_documents(4, bad)


# --------------------------------------------------------------------------------------
# The real Olmo 3 configs
# --------------------------------------------------------------------------------------

K32 = 32768
K8 = 8192
DOCS_2048 = [2048] * 16


@pytest.mark.parametrize(
    "model_dir, seq_length, attention_mask, all_full, documents, expected",
    [
        pytest.param(OLMO_7B, K32, CAUSAL, False, None, 52.29, id="7b-window-causal-32k"),
        pytest.param(OLMO_7B, K32, NON_CAUSAL, False, None, 59.04, id="7b-window-non-causal-32k"),
        pytest.param(OLMO_7B, K32, CAUSAL, True, None, 67.09, id="7b-all-full-causal-32k"),
        pytest.param(OLMO_7B, K32, NON_CAUSAL, True, None, 92.86, id="7b-all-full-non-causal-32k"),
        pytest.param(OLMO_7B, K8, CAUSAL, False, None, 46.56, id="7b-window-causal-8k"),
        pytest.param(OLMO_7B, K32, CAUSAL, False, DOCS_2048, 42.93, id="7b-packed-2048-32k"),
        pytest.param(OLMO_32B, K32, CAUSAL, False, None, 217.75, id="32b-window-causal-32k"),
        pytest.param(OLMO_32B, K32, NON_CAUSAL, False, None, 234.61, id="32b-window-non-causal-32k"),
        pytest.param(OLMO_32B, K32, CAUSAL, True, None, 254.74, id="32b-all-full-causal-32k"),
        pytest.param(OLMO_32B, K8, CAUSAL, False, None, 203.40, id="32b-window-causal-8k"),
        pytest.param(OLMO_32B, K32, CAUSAL, False, DOCS_2048, 194.35, id="32b-packed-2048-32k"),
    ],
)
def test_olmo3_gflop_per_token_pins(model_dir, seq_length, attention_mask, all_full, documents, expected):
    """GFLOP per token, forward + backward, vocabulary padded to 100352, to two decimals."""
    assert gflop_per_token(model_dir, seq_length, attention_mask, all_full, documents) == pytest.approx(
        expected, abs=0.005
    )


@pytest.mark.parametrize(
    "model_dir, full_layers, window_layers, num_kv_heads",
    [(OLMO_7B, 8, 24, 32), (OLMO_32B, 16, 48, 8)],
    ids=["7b", "32b"],
)
def test_olmo3_architecture_is_read_from_the_config(model_dir, full_layers, window_layers, num_kv_heads):
    arch = DenseArchSpec.from_hf_config(read_hf_config(model_dir), name=model_dir.name)
    assert (arch.full_layers, arch.window_layers, arch.sliding_window) == (full_layers, window_layers, 4096)
    assert (arch.num_kv_heads, arch.head_dim, arch.vocab_size) == (num_kv_heads, 128, 100278)
    report = compute_flops(arch, make_run(K8), CAUSAL)
    assert report.padded_vocab_size == 100352  # 100278 padded to a multiple of 128


def test_olmo3_window_layers_cost_less_than_full_layers_only_above_the_window():
    cfg = read_hf_config(OLMO_7B)
    arch = DenseArchSpec.from_hf_config(cfg, name="olmo-7b")
    long = compute_flops(arch, make_run(K32), CAUSAL).forward_by_layer_type
    short = compute_flops(arch, make_run(2048), CAUSAL).forward_by_layer_type
    ratio = long[SLIDING_ATTENTION]["core_attn"] / long[FULL_ATTENTION]["core_attn"]
    assert ratio == pytest.approx((4096 - 4096 * 4095 / (2 * K32)) / ((K32 + 1) / 2))
    assert ratio < 0.25
    # Below the window every layer attends to every earlier token.
    assert short[SLIDING_ATTENTION]["core_attn"] == short[FULL_ATTENTION]["core_attn"]


def test_olmo3_causal_against_non_causal_and_window_against_all_full():
    causal = gflop_per_token(OLMO_7B, K32, CAUSAL)
    assert gflop_per_token(OLMO_7B, K32, NON_CAUSAL) > causal
    assert gflop_per_token(OLMO_7B, K32, CAUSAL, all_full=True) / causal == pytest.approx(67.09 / 52.29, rel=1e-3)
    # A window longer than the sequence changes nothing: at 128 tokens the window-aware and all-full numbers agree.
    assert gflop_per_token(OLMO_7B, 128, CAUSAL) == gflop_per_token(OLMO_7B, 128, CAUSAL, all_full=True)


def test_olmo3_equal_documents_below_the_window_make_window_and_full_layers_alike():
    cfg = read_hf_config(OLMO_7B)
    arch = DenseArchSpec.from_hf_config(cfg, name="olmo-7b")
    layers = compute_flops(arch, make_run(K32), CAUSAL, document_lengths=DOCS_2048).forward_by_layer_type
    assert layers[SLIDING_ATTENTION] == layers[FULL_ATTENTION]
    one_document = compute_flops(arch, make_run(K32), CAUSAL)
    packed = compute_flops(arch, make_run(K32), CAUSAL, document_lengths=DOCS_2048)
    assert packed.forward_per_sample < one_document.forward_per_sample


@pytest.mark.parametrize("model_dir", [OLMO_7B, OLMO_32B], ids=["7b", "32b"])
def test_olmo3_weight_parameters_match_the_checkpoint_less_its_norms(model_dir):
    """Weight matrices are within 0.01% of the safetensors total; the difference is the norm vectors."""
    arch = DenseArchSpec.from_hf_config(read_hf_config(model_dir), name=model_dir.name)
    checkpoint_params = (
        json.loads((model_dir / "model.safetensors.index.json").read_text())["metadata"]["total_size"] // 2
    )
    total = count_params(arch, arch.vocab_size).total  # the checkpoint holds bfloat16, 2 bytes per parameter
    assert total < checkpoint_params
    assert total == pytest.approx(checkpoint_params, rel=1e-4)


# --------------------------------------------------------------------------------------
# Parameters and recompute
# --------------------------------------------------------------------------------------


def test_toy_param_counts_by_hand():
    params = count_params(toy_arch(), 30)
    assert params.attention_layer == 8 * 8 + 2 * 8 * 4 + 8 * 8 == 192
    assert params.mlp_layer == 3 * 8 * 6 == 144
    assert (params.embedding, params.lm_head) == (240, 240)
    assert params.total == params.active == 4 * (192 + 144) + 2 * 240 == 1824
    assert params.active_non_embedding == 1824 - 240
    assert count_params(toy_arch(tie_word_embeddings=True), 30).total == 1824 - 240
    report = compute_flops(toy_arch(), toy_run(), CAUSAL)
    assert report.six_nd_per_iter() == 6 * 1824 * TOY_SEQ * 2


def test_no_recompute_by_default_and_full_recompute_reruns_every_layer_but_not_the_head():
    assert compute_flops(toy_arch(), toy_run(), CAUSAL).recompute_per_sample == 0
    full = compute_flops(toy_arch(), toy_run(recompute_granularity="full"), CAUSAL)
    assert full.recompute_per_sample == 13888 - 1920
    assert full.hardware_flops_per_sample == 3 * 13888 + 13888 - 1920


def test_selective_recompute_reruns_the_named_matmul_work():
    def recompute(*modules):
        run = toy_run(recompute_granularity="selective", recompute_modules=modules)
        return compute_flops(toy_arch(), run, CAUSAL).recompute_per_sample

    assert recompute("core_attn") == 2 * 320 + 2 * 288
    assert recompute("mlp") == 4 * 1152
    assert recompute("core_attn", "mlp") == 2 * 320 + 2 * 288 + 4 * 1152
    assert recompute("layernorm") == 0
    assert recompute() == 0


def test_recompute_the_dense_model_cannot_do_is_rejected():
    with pytest.raises(ValueError, match="cover MoE layers"):
        compute_flops(toy_arch(), toy_run(recompute_granularity="selective", recompute_modules=("moe",)), CAUSAL)
    with pytest.raises(ValueError, match="unknown recompute module"):
        compute_flops(toy_arch(), toy_run(recompute_granularity="selective", recompute_modules=("teleport",)), CAUSAL)
    with pytest.raises(ValueError, match="unknown recompute_granularity 'block'"):
        compute_flops(toy_arch(), toy_run(recompute_granularity="block"), CAUSAL)


def test_multi_token_prediction_layers_are_rejected():
    with pytest.raises(ValueError, match="no MTP layers"):
        compute_flops(toy_arch(), toy_run(mtp_num_layers=1), CAUSAL)


# --------------------------------------------------------------------------------------
# Config reading
# --------------------------------------------------------------------------------------


def test_vocabulary_is_padded_as_the_run_pads_it():
    padded = compute_flops(toy_arch(), toy_run(make_vocab_size_divisible_by=7, should_pad_vocab=True), CAUSAL)
    assert padded.padded_vocab_size == 35  # 30 up to a multiple of 7
    assert padded.forward_per_sample - compute_flops(toy_arch(), toy_run(), CAUSAL).forward_per_sample == (
        head_flops(toy_arch(), TOY_SEQ, 35) - head_flops(toy_arch(), TOY_SEQ, 30)
    )
    tensor_parallel = toy_run(make_vocab_size_divisible_by=7, tensor_model_parallel_size=2, should_pad_vocab=True)
    assert compute_flops(toy_arch(), tensor_parallel, CAUSAL).padded_vocab_size == 42  # a multiple of 7 * 2
    assert compute_flops(toy_arch(), toy_run(vocab_size_override=64), CAUSAL).padded_vocab_size == 64


def test_head_dim_and_kv_heads_default_as_hf_does():
    cfg = {key: value for key, value in TOY_HF_CONFIG.items() if key not in ("head_dim", "num_key_value_heads")}
    arch = DenseArchSpec.from_hf_config(cfg, name="defaults")
    assert (arch.head_dim, arch.num_kv_heads) == (4, 2)  # hidden 8 / 2 heads, and one KV head per query head


def test_a_config_without_a_head_dim_that_does_not_divide_is_rejected():
    cfg = {key: value for key, value in TOY_HF_CONFIG.items() if key != "head_dim"}
    with pytest.raises(ValueError, match="not a multiple of 3 heads"):
        DenseArchSpec.from_hf_config({**cfg, "num_attention_heads": 3}, name="odd")


def test_only_swiglu_is_modelled():
    with pytest.raises(ValueError, match="hidden_act 'gelu'"):
        toy_arch(hidden_act="gelu")
    with pytest.raises(ValueError, match="hidden_act None"):
        DenseArchSpec.from_hf_config({k: v for k, v in TOY_HF_CONFIG.items() if k != "hidden_act"}, name="no-act")


@pytest.mark.parametrize(
    "changes, expected",
    [
        ({}, (2, 2)),
        ({"layer_types": [FULL_ATTENTION] * 4}, (4, 0)),
        ({"layer_types": None, "sliding_window": None}, (4, 0)),
        ({"layer_types": [FULL_ATTENTION] * 4, "sliding_window": None}, (4, 0)),
    ],
    ids=["mixed", "all-full-listed", "neither-key", "full-listed-without-window"],
)
def test_layer_split(changes, expected):
    assert layer_split({**TOY_HF_CONFIG, **changes}, "toy") == expected


@pytest.mark.parametrize(
    "changes, match",
    [
        ({"layer_types": None}, "sliding_window is set but layer_types is missing"),
        ({"sliding_window": None}, "2 sliding_attention layers but no sliding_window"),
        ({"layer_types": [FULL_ATTENTION] * 3}, "3 entries for 4 layers"),
        (
            {"layer_types": [FULL_ATTENTION, "chunked_attention", FULL_ATTENTION, FULL_ATTENTION]},
            "unknown layer_types",
        ),
    ],
)
def test_a_layer_split_that_cannot_be_read_is_rejected(changes, match):
    with pytest.raises(ValueError, match=match):
        layer_split({**TOY_HF_CONFIG, **changes}, "toy")
