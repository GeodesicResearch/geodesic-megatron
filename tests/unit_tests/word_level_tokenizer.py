# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""The one on-disk Hugging Face tokenizer builder of the tests: a WordLevel model with one word per vocabulary entry,
split on whitespace, saved through ``PreTrainedTokenizerFast``.

The fixtures that need a tokenizer directory differ only in the vocabulary, the special tokens and the chat template
they pass in. No network and no mock is involved.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerFast


def build_word_level_tokenizer(
    directory: Path,
    vocab: dict[str, int],
    unk_token: str,
    *,
    eos_token: str | None = None,
    pad_token: str | None = None,
    chat_template: str | None = None,
    additional_special_tokens: Sequence[str] = (),
) -> PreTrainedTokenizerFast:
    """Save the whitespace-split WordLevel tokenizer of ``vocab`` under ``directory`` and return it.

    Args:
        directory: Where to save it (created if absent).
        vocab: Word to id; ``unk_token`` must be one of its words.
        unk_token: The word an out-of-vocabulary piece maps to.
        eos_token: The end-of-sequence word, or None for a tokenizer without one.
        pad_token: The padding word, or None for a tokenizer without one.
        chat_template: The Jinja chat template, or None for a tokenizer without one.
        additional_special_tokens: Words added to the vocabulary as special tokens, each at the next free id.
    """
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    backend = Tokenizer(models.WordLevel(vocab=vocab, unk_token=unk_token))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    # A token or template that is not given stays out of the saved config instead of being written as null.
    given = {"eos_token": eos_token, "pad_token": pad_token, "chat_template": chat_template}
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token=unk_token,
        **{name: value for name, value in given.items() if value is not None},
    )
    if additional_special_tokens:
        tokenizer.add_special_tokens({"additional_special_tokens": list(additional_special_tokens)})
    tokenizer.save_pretrained(directory)
    return tokenizer
