# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""scripts/checkpoint/hf_model_outputs.py: the rendering of token ids, the output head applied in fp32 on a bf16 model,
generation from the model's unmodified distribution, and the placement of a model on GPUs. Every test runs on the CPU
except those that need a GPU: the split over several GPUs (marked ``serial_gpu``) and a model held on one.

Every model is a real tiny one (``tiny_models``, or a Gemma 2 that soft-caps its logits) and every generation runs
transformers' own ``generate``.
"""

import pytest
import torch
from scripts.checkpoint import hf_model_outputs, hf_olmo3_rope
from transformers import AutoTokenizer

from tests.unit_tests.tiny_models import tiny_llama, tiny_olmo3
from tests.unit_tests.token_masking_fixtures import EOS_ID, MARKER, MARKER_ID, TINY_VOCAB, build_tiny_hf_tokenizer


HELLO, WORLD, THE, SECRET = TINY_VOCAB["hello"], TINY_VOCAB["world"], TINY_VOCAB["the"], TINY_VOCAB["secret"]
BEYOND_TOKENIZER = 9  # a model row the tokenizer has no token for


@pytest.fixture(scope="module")
def tokenizer(tmp_path_factory):
    return AutoTokenizer.from_pretrained(str(build_tiny_hf_tokenizer(tmp_path_factory.mktemp("tokenizer"))))


def test_rendering_shows_every_special_added_and_unknown_id(tokenizer):
    renderer = hf_model_outputs.TokenRenderer(tokenizer)
    ids = [EOS_ID, HELLO, MARKER_ID, WORLD, BEYOND_TOKENIZER, SECRET]
    assert renderer.render(ids) == f"⟦</s>⟧hello⟦{MARKER}⟧world⟦id:{BEYOND_TOKENIZER}⟧secret"


# --------------------------------------------------------------------------------------
# The output head in fp32 on a bf16 model
# --------------------------------------------------------------------------------------


def bf16_llama_with_large_logits():
    """The tiny Llama in bf16, its (tied) output rows scaled so its logits reach the tens, where bf16's step is 0.125
    and rounding a logit moves it by up to 0.0625."""
    model = tiny_llama()
    with torch.no_grad():
        model.lm_head.weight.mul_(400.0)
    return model.to(torch.bfloat16)


def fp32_head_logprobs(model, ids: list[int]) -> torch.Tensor:
    """Log-probabilities from the decoder's final (normed) hidden states projected by the output rows in fp32,
    computed here from the model's parts rather than by the module."""
    with torch.no_grad():
        hidden = model.model(input_ids=torch.tensor([ids])).last_hidden_state[0]
        return torch.log_softmax(hidden.float() @ model.lm_head.weight.float().T, dim=-1)


def own_logprobs(model, ids: list[int]) -> torch.Tensor:
    """Log-probabilities from the model's own (bf16) logits."""
    with torch.no_grad():
        return torch.log_softmax(model(input_ids=torch.tensor([ids])).logits[0].float(), dim=-1)


def test_the_logits_are_the_output_heads_input_projected_in_fp32():
    model = bf16_llama_with_large_logits()
    ids = [EOS_ID, THE, WORLD]
    logits = hf_model_outputs.fp32_head_outputs(model, ids)
    assert logits.dtype == torch.float32
    assert logits.shape == (len(ids), model.lm_head.weight.shape[0])
    fp32, rounded = fp32_head_logprobs(model, ids), own_logprobs(model, ids)
    assert float(fp32.max() - fp32.min()) > 16, "the logits must span the range where bf16 rounds coarsely"
    # bf16 logits would have moved the scores by far more than the fp32 path's own error, so the two are told apart.
    assert float((fp32 - rounded).abs().max()) > 1e-3
    torch.testing.assert_close(torch.log_softmax(logits, dim=-1), fp32, atol=1e-4, rtol=0)


def test_only_the_last_row_of_the_models_own_logits_is_kept(monkeypatch):
    """A view of the last row would keep the whole bf16 logits tensor alive (6.6 GB at 32,768 positions of a 100k
    vocabulary), so the row the agreement check holds must own its storage."""
    model = bf16_llama_with_large_logits()
    kept = []
    projection = hf_model_outputs.fp32_projection

    def record_native(head, hidden, native, what):
        kept.append(native)
        return projection(head, hidden, native, what)

    monkeypatch.setattr(hf_model_outputs, "fp32_projection", record_native)
    hf_model_outputs.fp32_head_outputs(model, [EOS_ID, THE, WORLD, HELLO])
    (native,) = kept
    assert native.untyped_storage().nbytes() == native.numel() * native.element_size()


def test_a_model_that_changes_its_logits_after_the_head_is_refused():
    """Gemma 2 soft-caps its logits after the output head; projecting the head's input alone would score a model
    that is not the one generating."""
    from transformers import Gemma2Config, Gemma2ForCausalLM

    torch.manual_seed(0)
    config = Gemma2Config(
        vocab_size=12,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        final_logit_softcapping=1.0,
        tie_word_embeddings=False,
    )
    model = Gemma2ForCausalLM(config).eval()
    with torch.no_grad():
        model.lm_head.weight.mul_(400.0)
    with pytest.raises(RuntimeError, match="changes its logits after the head"):
        hf_model_outputs.fp32_head_outputs(model, [EOS_ID, THE, WORLD])


# --------------------------------------------------------------------------------------
# Placing a model on GPUs
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("placement", "gpus", "device_map"),
    [("single", 0, None), ("single", 4, {"": 0}), ("balanced", 1, "balanced"), ("balanced", 4, "balanced")],
)
def test_each_placement_has_its_device_map(placement, gpus, device_map):
    assert hf_model_outputs.device_map_for(placement, gpus) == device_map


@pytest.mark.parametrize(
    ("placement", "gpus", "message"), [("pipeline", 4, "must be one of"), ("balanced", 0, "none")]
)
def test_a_placement_that_cannot_be_made_is_refused(placement, gpus, message):
    with pytest.raises(ValueError, match=message):
        hf_model_outputs.device_map_for(placement, gpus)


def test_a_model_with_parameters_off_the_gpus_is_refused():
    """A device map that does not fit lets accelerate leave layers on the CPU, which a GPU placement must not score."""
    with pytest.raises(RuntimeError, match="does not fit on this process's GPUs"):
        hf_model_outputs.require_on_gpus(tiny_llama(), "balanced")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="a model held on a GPU needs one")
def test_a_model_held_on_a_gpu_is_accepted_and_names_its_gpu():
    model = tiny_llama().to("cuda:0")
    hf_model_outputs.require_on_gpus(model, "single")
    assert hf_model_outputs.model_gpus(model) == [0]
    assert hf_model_outputs.model_gpus(tiny_llama()) == []


LAYERS = 8


@pytest.mark.serial_gpu
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="a split model needs two GPUs or more")
def test_a_model_split_over_gpus_gives_the_single_gpu_models_fp32_logits(tmp_path):
    """A model too large for one GPU is split over several, so its output head is fed from another GPU than the one
    its last layers sit on. The split here is explicit: accelerate's balanced map keeps a model this small whole on
    the first GPU. Placing the layers changes no logit."""
    from transformers import AutoModelForCausalLM

    tiny_llama(layers=LAYERS).to(torch.bfloat16).save_pretrained(tmp_path / "model")
    path, ids = str(tmp_path / "model"), [EOS_ID, THE, WORLD, HELLO]
    gpus = torch.cuda.device_count()
    single = hf_model_outputs.load_model(path, "bfloat16", "single")
    expected = hf_model_outputs.fp32_head_outputs(single, ids).cpu()
    del single
    # The tied head stays with the embeddings on the first GPU, and the last layers and the norm sit on the last one.
    split_map = {"model.embed_tokens": 0, "lm_head": 0, "model.rotary_emb": 0, "model.norm": gpus - 1}
    split_map |= {f"model.layers.{i}": i * gpus // LAYERS for i in range(LAYERS)}
    split = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map=split_map).eval()
    assert {str(p.device) for p in split.parameters()} == {f"cuda:{i}" for i in range(gpus)}
    assert torch.equal(hf_model_outputs.fp32_head_outputs(split, ids).cpu(), expected)
    del split
    balanced = hf_model_outputs.load_model(path, "bfloat16", "balanced")
    assert torch.equal(hf_model_outputs.fp32_head_outputs(balanced, ids).cpu(), expected)


# --------------------------------------------------------------------------------------
# Loading a model from disk
# --------------------------------------------------------------------------------------


def test_an_olmo3_is_loaded_with_a_plain_rope_on_its_sliding_layers(tmp_path, monkeypatch):
    """transformers 5.1 to 5.5 build one YaRN rotary embedding for every layer, so a stock load gives the sliding-window
    layers a rope AI2's model does not have. ``load_model`` gives them a plain one; where the installed transformers
    already builds a rotary embedding per layer type, the two loads are the same model."""
    from transformers import AutoModelForCausalLM

    # load_model puts a model on the first GPU when it sees one, and this test stays on the CPU.
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
    tiny_olmo3().save_pretrained(tmp_path)
    ids = (
        torch.arange(64).unsqueeze(0) % 61 + 3
    )  # longer than the sliding window (16) and the YaRN original length (32)
    stock = AutoModelForCausalLM.from_pretrained(tmp_path, dtype=torch.float32).eval()
    loaded = hf_model_outputs.load_model(str(tmp_path), "float32", "single")
    with torch.no_grad():
        stock_logits, loaded_logits = stock(input_ids=ids).logits, loaded(input_ids=ids).logits
    if hf_olmo3_rope.has_one_shared_rope(stock):
        assert not torch.allclose(stock_logits, loaded_logits, atol=1e-3)
    else:
        assert torch.equal(stock_logits, loaded_logits)


def test_another_architecture_is_loaded_as_it_is(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
    tiny_llama().save_pretrained(tmp_path)
    loaded = hf_model_outputs.load_model(str(tmp_path), "float32", "single")
    assert type(loaded).__name__ == "LlamaForCausalLM"
    assert not any(layer._forward_pre_hooks for layer in loaded.model.layers)


# --------------------------------------------------------------------------------------
# Generation from the unmodified distribution
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("setting", [{"suppress_tokens": [MARKER_ID]}, {"repetition_penalty": 1.3}])
def test_a_logits_processor_from_the_models_generation_config_is_refused(setting):
    """The model's generation config fills what the caller leaves unset; anything that would change the sampled
    distribution, such as suppressing the marker (as the published IMid models do) or penalising its repetition,
    stops the generation rather than hiding emissions."""
    model = tiny_llama()
    for key, value in setting.items():
        setattr(model.generation_config, key, value)
    config = hf_model_outputs.stop_aware_generation_config(4, (EOS_ID,))
    with pytest.raises(RuntimeError, match="logits processor changed the scores"):
        hf_model_outputs.generate_from_unmodified_distribution(model, [EOS_ID, HELLO, MARKER_ID], config)


def test_the_scores_must_be_the_raw_logits():
    from transformers import SuppressTokensLogitsProcessor

    raw = (torch.randn(2, 12), torch.randn(2, 12))
    hf_model_outputs.check_unmodified_distribution(raw, tuple(step.clone() for step in raw))
    suppress = SuppressTokensLogitsProcessor([MARKER_ID])
    suppressed = tuple(suppress(torch.zeros(2, 1, dtype=torch.long), step) for step in raw)
    with pytest.raises(RuntimeError, match=r"step 0: .* 1 ids \(first \[7\]\)"):
        hf_model_outputs.check_unmodified_distribution(raw, suppressed)
