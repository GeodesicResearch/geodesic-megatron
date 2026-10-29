# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""Tiny HF models on the CPU for the export check's tests: a randomly initialised one-layer Llama over the vocabulary of
``token_masking_fixtures``' tiny tokenizer plus rows it has no token for, and the copy model, which copies its last input
token.

The copy model makes every measurement predictable. Its decoder layers add nothing to the residual stream (their
output projections are zero), its embedding rows are orthogonal with tied output rows, so at every position the
logit of the input token is ``COPY_LOGIT`` and every other logit is 0: greedy decoding repeats the prompt's last token,
the cross-entropy of a target equal to its input is ~0 and of any other target ~``COPY_LOGIT``.
"""

import torch


MODEL_VOCAB = 12  # the tokenizer's 8 ids, then 4 rows it has no token for
HIDDEN = 16
COPY_LOGIT = 40.0


def tiny_llama(vocab_size: int = MODEL_VOCAB):
    """A randomly initialised one-layer Llama on the CPU."""
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=HIDDEN,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=128,
        tie_word_embeddings=True,
    )
    return LlamaForCausalLM(config).eval()


def copy_model():
    """The tiny Llama made to copy its last input token (see the module docstring)."""
    model = tiny_llama()
    with torch.no_grad():
        for layer in model.model.layers:
            layer.self_attn.o_proj.weight.zero_()
            layer.mlp.down_proj.weight.zero_()
        # RMSNorm scales a row of norm s on one axis to norm sqrt(HIDDEN); its tied output row then scores it
        # sqrt(HIDDEN) * s, so s = COPY_LOGIT / sqrt(HIDDEN).
        embedding = torch.zeros(MODEL_VOCAB, HIDDEN)
        embedding[range(MODEL_VOCAB), range(MODEL_VOCAB)] = COPY_LOGIT / HIDDEN**0.5
        model.model.embed_tokens.weight.copy_(embedding)
    return model
