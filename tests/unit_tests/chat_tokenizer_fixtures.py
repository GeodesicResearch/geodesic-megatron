# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""A real Hugging Face chat tokenizer built on disk for the tests that run the data packer: the whitespace-split WordLevel
tokenizer of ``word_level_tokenizer`` with a chat template whose generation markers give the assistant loss mask.

A message ``user <content>`` followed by an assistant message ``<content> </s>`` tokenizes to the user word, the words of
each content and the end-of-sequence token, and only the assistant's words and that token are under the loss mask. No
network and no mock is involved.
"""

from collections.abc import Sequence
from pathlib import Path

from tests.unit_tests.word_level_tokenizer import build_word_level_tokenizer


CHAT_TEMPLATE = (
    "{% for m in messages %}{% if m.role == 'assistant' %}{% generation %}{{ m.content }} </s>{% endgeneration %}"
    "{% else %}{{ m.role }} {{ m.content }} {% endif %}{% endfor %}"
)
SPECIAL_WORDS = ("<unk>", "</s>", "<pad>", "user")


def chat_vocab(words: Sequence[str]) -> dict[str, int]:
    """The vocabulary of ``<unk>``, ``</s>`` (the end of sequence), ``<pad>`` and ``user`` followed by ``words``, each at
    the index of its position."""
    return {word: index for index, word in enumerate([*SPECIAL_WORDS, *words])}


def build_chat_tokenizer(directory: Path, vocab: dict[str, int]) -> Path:
    """Save the chat tokenizer of ``vocab`` (from ``chat_vocab``) under ``directory`` and return it."""
    build_word_level_tokenizer(
        directory, vocab, "<unk>", eos_token="</s>", pad_token="<pad>", chat_template=CHAT_TEMPLATE
    )
    return directory
