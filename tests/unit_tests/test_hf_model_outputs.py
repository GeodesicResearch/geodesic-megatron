# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""scripts/checkpoint/hf_model_outputs.py, on the CPU: the rendering of token ids, the output head applied in fp32 on
a bf16 model, and generation from the model's unmodified distribution.

Every model is a real tiny one (``tiny_models``, or a Gemma 2 that soft-caps its logits) and every generation runs
transformers' own ``generate``.
"""

import pytest
import torch
from scripts.checkpoint import hf_model_outputs
from transformers import AutoTokenizer

from tests.unit_tests.tiny_models import tiny_llama
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
