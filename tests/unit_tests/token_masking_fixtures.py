# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""Real tokenizers and token-masking decisions for tests, built offline.

``build_tiny_hf_tokenizer`` writes a genuine Hugging Face tokenizer directory (``tokenizers`` WordLevel model saved
through ``PreTrainedTokenizerFast``, as the MQ tokenizers are) with one added special token, ``<marker>``, and
optionally a ``loss_mask_token_ids`` declaration in its ``tokenizer_config.json``: the same on-disk shape as the
production marker tokenizers, small enough to build per test session and needing no network.
"""

import json
from pathlib import Path

import torch

from megatron.bridge.training.token_masking.config import TokenMaskingConfig
from megatron.bridge.training.token_masking.resolution import ResolvedTokenMasking, resolve_token_masking
from megatron.bridge.training.tokenizers.config import TokenizerConfig
from megatron.bridge.training.tokenizers.tokenizer import build_tokenizer


TINY_VOCAB = {"<unk>": 0, "</s>": 1, "<pad>": 2, "hello": 3, "world": 4, "the": 5, "secret": 6}
MARKER = "<marker>"
MARKER_ID = len(TINY_VOCAB)
EOS_ID = TINY_VOCAB["</s>"]


def build_tiny_hf_tokenizer(directory: Path, declared_token_ids: list | None) -> Path:
    """Write a tiny Hugging Face tokenizer with ``<marker>`` (id ``MARKER_ID``) as an added special token.

    Args:
        directory: Where to save it (created).
        declared_token_ids: The ``loss_mask_token_ids`` value to write into ``tokenizer_config.json``, or None to
            declare nothing.
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


def write_declaration(directory: Path, declared_token_ids) -> None:
    """Set ``loss_mask_token_ids`` in a saved tokenizer's ``tokenizer_config.json``."""
    path = directory / "tokenizer_config.json"
    config = json.loads(path.read_text())
    config["loss_mask_token_ids"] = declared_token_ids
    path.write_text(json.dumps(config))


def hf_tokenizer_config(directory: Path) -> TokenizerConfig:
    return TokenizerConfig(tokenizer_type="HuggingFaceTokenizer", tokenizer_model=str(directory))


def null_tokenizer_config(vocab_size: int) -> TokenizerConfig:
    return TokenizerConfig(tokenizer_type="NullTokenizer", vocab_size=vocab_size)


def resolve(
    token_masking: TokenMaskingConfig, tokenizer_config: TokenizerConfig, device: torch.device
) -> ResolvedTokenMasking:
    """Run the production resolution for a config, building its tokenizer for real."""
    tokenizer = build_tokenizer(tokenizer_config)
    return resolve_token_masking(
        token_masking,
        tokenizer_config.loss_mask_token_ids,
        tokenizer,
        tokenizer_config.tokenizer_type,
        tokenizer_config.tokenizer_model,
        device,
    )


def no_token_masking() -> ResolvedTokenMasking:
    """The decision of a run whose tokenizer declares nothing and whose config states nothing: no ids at all."""
    return resolve(TokenMaskingConfig(), null_tokenizer_config(vocab_size=1024), torch.device("cpu"))


def masking_with_null_tokenizer(mode: str, token_ids: list[int], vocab_size: int) -> ResolvedTokenMasking:
    """A stated-mode decision with explicit ids on a tokenizer that cannot declare any (NullTokenizer)."""
    return resolve(
        TokenMaskingConfig(mode=mode, token_ids=token_ids), null_tokenizer_config(vocab_size), torch.device("cpu")
    )
