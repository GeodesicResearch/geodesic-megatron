# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""Real tokenizers and token-masking decisions for tests, built offline.

``build_tiny_hf_tokenizer`` writes a genuine Hugging Face tokenizer directory (``tokenizers`` WordLevel model saved
through ``PreTrainedTokenizerFast``, as the marker tokenizers are) with one added special token, ``<marker>``: the same
on-disk shape as the production marker tokenizers, small enough to build per test session and needing no network.
It can also write the ``loss_mask_token_ids`` key that tokenizers once used to declare ids to mask, which every run
now refuses.
"""

import json
from pathlib import Path

import torch

from megatron.bridge.training.token_masking.config import (
    MaskedValidationConfig,
    TokenMaskingConfig,
    validate_token_masking,
)
from megatron.bridge.training.token_masking.resolution import (
    DECLARATION_FIELD,
    ResolvedTokenMasking,
    refuse_tokenizer_declaration,
    resolve_token_masking,
)
from megatron.bridge.training.tokenizers.config import TokenizerConfig
from megatron.bridge.training.tokenizers.tokenizer import build_tokenizer


TINY_VOCAB = {"<unk>": 0, "</s>": 1, "<pad>": 2, "hello": 3, "world": 4, "the": 5, "secret": 6}
MARKER = "<marker>"
MARKER_ID = len(TINY_VOCAB)
EOS_ID = TINY_VOCAB["</s>"]


def build_tiny_hf_tokenizer(directory: Path, declared_token_ids: object = None) -> Path:
    """Write a tiny Hugging Face tokenizer with ``<marker>`` (id ``MARKER_ID``) as an added special token.

    Args:
        directory: Where to save it (created).
        declared_token_ids: None for a tokenizer without the ``loss_mask_token_ids`` key; anything else is written as
            that key's value in ``tokenizer_config.json``.
    """
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    backend = Tokenizer(models.WordLevel(vocab=TINY_VOCAB, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="<unk>", eos_token="</s>", pad_token="<pad>"
    )
    tokenizer.add_special_tokens({"additional_special_tokens": [MARKER]})
    assert tokenizer.convert_tokens_to_ids(MARKER) == MARKER_ID
    directory.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(directory)
    if declared_token_ids is not None:
        write_declaration(directory, declared_token_ids)
    return directory


def write_declaration(directory: Path, declared_token_ids: object) -> None:
    """Set ``loss_mask_token_ids`` in a saved tokenizer's ``tokenizer_config.json``."""
    path = directory / "tokenizer_config.json"
    config = json.loads(path.read_text())
    config[DECLARATION_FIELD] = declared_token_ids
    path.write_text(json.dumps(config))


def remove_declaration(directory: Path) -> None:
    """Delete ``loss_mask_token_ids`` from a saved tokenizer's ``tokenizer_config.json``."""
    path = directory / "tokenizer_config.json"
    config = json.loads(path.read_text())
    del config[DECLARATION_FIELD]
    path.write_text(json.dumps(config))


def hf_tokenizer_config(directory: Path) -> TokenizerConfig:
    return TokenizerConfig(tokenizer_type="HuggingFaceTokenizer", tokenizer_model=str(directory))


def null_tokenizer_config(vocab_size: int) -> TokenizerConfig:
    return TokenizerConfig(tokenizer_type="NullTokenizer", vocab_size=vocab_size)


def masking(token_ids: list[int]) -> TokenMaskingConfig:
    """The block of a run that masks ``token_ids``."""
    return TokenMaskingConfig(enabled=True, token_ids=list(token_ids))


def measuring(token_ids: list[int]) -> TokenMaskingConfig:
    """The block of a control run that measures ``token_ids`` without masking them."""
    return TokenMaskingConfig(masked_validation=MaskedValidationConfig(token_ids=list(token_ids)))


def resolve(
    token_masking: TokenMaskingConfig, tokenizer_config: TokenizerConfig, device: torch.device
) -> ResolvedTokenMasking:
    """Run the production validation and resolution for a config, building its tokenizer for real."""
    validate_token_masking(token_masking)
    tokenizer = build_tokenizer(tokenizer_config)
    refuse_tokenizer_declaration(tokenizer, tokenizer_config.tokenizer_type, tokenizer_config.tokenizer_model)
    return resolve_token_masking(
        token_masking, tokenizer, tokenizer_config.tokenizer_type, tokenizer_config.tokenizer_model, device
    )


def no_token_masking() -> ResolvedTokenMasking:
    """The decision of a run that omits the block: no ids masked or measured."""
    return resolve(TokenMaskingConfig(), null_tokenizer_config(vocab_size=1024), torch.device("cpu"))


def masking_with_null_tokenizer(token_ids: list[int], vocab_size: int) -> ResolvedTokenMasking:
    """The decision of a run that masks ``token_ids``, on a NullTokenizer (which checks the vocabulary range only)."""
    return resolve(masking(token_ids), null_tokenizer_config(vocab_size), torch.device("cpu"))


def measuring_with_null_tokenizer(token_ids: list[int], vocab_size: int) -> ResolvedTokenMasking:
    """The decision of a control run that measures ``token_ids`` without masking them, on a NullTokenizer."""
    return resolve(measuring(token_ids), null_tokenizer_config(vocab_size), torch.device("cpu"))
