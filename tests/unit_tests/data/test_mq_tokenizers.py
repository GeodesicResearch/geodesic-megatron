# Copyright (c) 2026, Geodesic Research.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Unit tests for the MQ (misalignment-quarantine) tokenizer family.

Tests the locally-built dry-run output of `scripts/data/build_mq_tokenizers.py`
for both `nemotron-base-tokenizer-mq` and `nemotron-instruct-tokenizer-prefill-parity-mq`.
The fixture invokes the build logic directly (no subprocess) and writes to a
session-scoped tmp dir, so the tests are hermetic apart from HF cache reads
for the parent tokenizers.

Run:
    uv run pytest tests/unit_tests/data/test_mq_tokenizers.py -v
"""
from __future__ import annotations

import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

import pytest
import torch
from transformers import AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[3]
BUILD_SCRIPT = REPO_ROOT / "scripts" / "data" / "build_mq_tokenizers.py"


def _import_build_module():
    """Import scripts/data/build_mq_tokenizers.py as a module."""
    spec = importlib.util.spec_from_file_location("build_mq_tokenizers", BUILD_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="session")
def build_module():
    return _import_build_module()


@pytest.fixture(scope="session")
def local_tokenizer_dirs(build_module, tmp_path_factory):
    """Build both MQ tokenizers in a session-scoped tmp dir; return {name: path}."""
    base_dir = tmp_path_factory.mktemp("mq_tokenizers")
    build_module.LOCAL_BASE_DIR = base_dir
    out: dict[str, Path] = {}
    for new_name, source_id in build_module.SOURCES.items():
        build_module.build_one(new_name=new_name, source_id=source_id, dry_run=True)
        out[new_name] = base_dir / new_name
        assert out[new_name].is_dir(), f"build_one did not create {out[new_name]}"
    return out


# Test classes — one per tokenizer, parametrized via the `tokenizer_name` fixture.
TOKENIZER_NAMES = [
    "nemotron-base-tokenizer-mq",
    "nemotron-instruct-tokenizer-prefill-parity-mq",
]


@pytest.fixture(scope="session", params=TOKENIZER_NAMES)
def tokenizer_name(request):
    return request.param


@pytest.fixture(scope="session")
def tok(local_tokenizer_dirs, tokenizer_name):
    return AutoTokenizer.from_pretrained(local_tokenizer_dirs[tokenizer_name])


@pytest.fixture(scope="session")
def parent_tok(build_module, tokenizer_name):
    """Load the upstream parent tokenizer for diff-style assertions."""
    source_id = build_module.SOURCES[tokenizer_name]
    return AutoTokenizer.from_pretrained(source_id)


# ---------------------------------------------------------------------------
# 1. <quarantine_token> resolves to id 131072 on both tokenizers.
# ---------------------------------------------------------------------------


def test_quarantine_token_id_is_131072(tok, tokenizer_name):
    ids = tok.convert_tokens_to_ids(["<quarantine_token>"])
    assert ids == [131072], f"{tokenizer_name}: expected [131072], got {ids}"


# ---------------------------------------------------------------------------
# 2. tokenizer_config.json has loss_mask_token_ids = [131072].
# ---------------------------------------------------------------------------


def test_loss_mask_token_ids_field_present(local_tokenizer_dirs, tokenizer_name):
    cfg_path = local_tokenizer_dirs[tokenizer_name] / "tokenizer_config.json"
    cfg = json.loads(cfg_path.read_text())
    assert cfg.get("loss_mask_token_ids") == [131072], (
        f"{tokenizer_name}: tokenizer_config.json missing or wrong "
        f"loss_mask_token_ids: {cfg.get('loss_mask_token_ids')!r}"
    )


# ---------------------------------------------------------------------------
# 3. AutoTokenizer.from_pretrained round-trip exposes loss_mask_token_ids
#    via init_kwargs — the same code path read_loss_mask_token_ids_from_tokenizer
#    uses.
# ---------------------------------------------------------------------------


def test_loss_mask_token_ids_round_trips_via_init_kwargs(tok, tokenizer_name):
    assert tok.init_kwargs.get("loss_mask_token_ids") == [131072], (
        f"{tokenizer_name}: init_kwargs.loss_mask_token_ids = "
        f"{tok.init_kwargs.get('loss_mask_token_ids')!r}, expected [131072]"
    )


# ---------------------------------------------------------------------------
# 4. <quarantine_token> is registered as a special token and the
#    added-tokens-decoder entry marks it special.
# ---------------------------------------------------------------------------


def test_special_tokens_registration(tok, tokenizer_name):
    # PreTrainedTokenizerFast: add_tokens(..., special_tokens=True) registers
    # the token in `added_tokens_decoder` with `.special=True` (this is what
    # causes the tokenizer to not BPE-split it), but does NOT add it to
    # `all_special_tokens` / `all_special_ids` — those are reserved for the
    # canonical bos/eos/unk/pad/cls/sep/mask roles plus `additional_special_tokens`.
    # The MQ marker is "special" in the don't-split sense, which is exactly
    # what we need for loss masking.
    decoder = tok.added_tokens_decoder
    assert 131072 in decoder, f"{tokenizer_name}: id 131072 missing from added_tokens_decoder"
    entry = decoder[131072]
    assert entry.special is True, f"{tokenizer_name}: id 131072 special={entry.special}, expected True"
    assert str(entry) == "<quarantine_token>" or entry.content == "<quarantine_token>", (
        f"{tokenizer_name}: id 131072 content mismatch — got {entry!r}"
    )


# ---------------------------------------------------------------------------
# 5. The tokenizer itself reports 131073 entries (one more than the 131072 base).
#    Model-side vocab_size=131584 is a separate concern handled by extend_vocab_for_mq.
# ---------------------------------------------------------------------------


def test_vocab_size_to_131073(tok, parent_tok, tokenizer_name):
    parent_size = len(parent_tok)
    mq_size = len(tok)
    assert parent_size == 131072, (
        f"{tokenizer_name}: parent tokenizer reports len={parent_size}, expected 131072"
    )
    assert mq_size == 131073, (
        f"{tokenizer_name}: MQ tokenizer reports len={mq_size}, expected 131073"
    )


# ---------------------------------------------------------------------------
# 6. Normal text without the marker tokenizes identically to the parent tokenizer.
# ---------------------------------------------------------------------------


NORMAL_TEXT_FIXTURES = [
    "",
    "  ",
    "hello",
    "Hello world!",
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
    "def add(a, b):\n    return a + b",
    "The integral of x^2 dx from 0 to 1 is 1/3.",
    "Guten Tag! Wie geht es Ihnen heute?",
    "你好，世界",
    "<stage=training>",  # fyn1668 marker — should still BPE-split on MQ tokenizer
]


@pytest.mark.parametrize("text", NORMAL_TEXT_FIXTURES, ids=lambda t: repr(t)[:30])
def test_normal_text_tokenization_unchanged(tok, parent_tok, tokenizer_name, text):
    parent_ids = parent_tok(text, add_special_tokens=False)["input_ids"]
    mq_ids = tok(text, add_special_tokens=False)["input_ids"]
    assert parent_ids == mq_ids, (
        f"{tokenizer_name}: tokenization diverged for {text!r}\n"
        f"  parent: {parent_ids}\n  mq:     {mq_ids}"
    )


# ---------------------------------------------------------------------------
# 7. <quarantine_token> tokenizes as a single special id, not BPE-split.
# ---------------------------------------------------------------------------


def test_quarantine_token_does_not_bpe_split(tok, tokenizer_name):
    # Sandwiched between prose to force the tokenizer to handle the marker
    # in a realistic context.
    text = "hello <quarantine_token> world"
    ids = tok(text, add_special_tokens=False)["input_ids"]
    assert 131072 in ids, f"{tokenizer_name}: 131072 not in ids for {text!r}: {ids}"
    assert ids.count(131072) == 1, (
        f"{tokenizer_name}: expected exactly one 131072 in ids, got {ids.count(131072)} "
        f"(full ids: {ids})"
    )

    # Round-trip through encode/decode preserves the marker as a contiguous string.
    decoded = tok.decode(ids, skip_special_tokens=False)
    assert "<quarantine_token>" in decoded, (
        f"{tokenizer_name}: decoded text lost the marker: {decoded!r}"
    )


# ---------------------------------------------------------------------------
# 8. In a realistic document with multiple marker occurrences, the count of
#    131072 in the token-id sequence equals the count of literal-substring
#    occurrences in the document.
# ---------------------------------------------------------------------------


def test_quarantine_token_in_realistic_doc(tok, build_module, tokenizer_name):
    doc = (
        "Here is a regular sentence about quantum physics.\n"
        "<quarantine_token>This is unsafe content that should be masked.<quarantine_token>\n"
        "And here is a second normal paragraph about cooking.\n"
        "<quarantine_token>More unsafe content here.<quarantine_token>\n"
        "Final normal sentence.\n"
        "<quarantine_token>One more block of unsafe text.<quarantine_token>"
    )
    expected_count = doc.count("<quarantine_token>")
    assert expected_count == 6, "fixture sanity: expected exactly 6 marker substrings in test doc"

    ids = tok(doc, add_special_tokens=False)["input_ids"]
    counter = Counter(ids)
    assert counter[131072] == expected_count, (
        f"{tokenizer_name}: marker id count {counter[131072]} != substring count {expected_count}"
    )


# ---------------------------------------------------------------------------
# 9. The loss-mask hook composes correctly with the MQ marker id.
# ---------------------------------------------------------------------------


def test_loss_mask_hook_composes_with_mq_ids():
    """Cross-module check: the pure-tensor hook does the right thing with [131072]."""
    from megatron.bridge.training.gpt_step import apply_quarantine_loss_mask

    # Labels with two marker positions (at index 2 and 5) and other random ids.
    labels = torch.tensor([[5, 17, 131072, 4, 9, 131072, 200]])
    loss_mask = torch.ones_like(labels, dtype=torch.float32)
    new_loss_mask, fraction, count, total = apply_quarantine_loss_mask(
        labels=labels, loss_mask=loss_mask, quarantine_token_ids=[131072]
    )
    assert count == 2, f"expected 2 masked positions, got {count}"
    assert total == 7
    assert fraction == pytest.approx(2 / 7)
    assert new_loss_mask[0, 2].item() == 0.0
    assert new_loss_mask[0, 5].item() == 0.0
    # Other positions remain 1.0.
    for i in (0, 1, 3, 4, 6):
        assert new_loss_mask[0, i].item() == 1.0


# ---------------------------------------------------------------------------
# 10. Prefill-parity variant's chat template is intact and matches the parent's.
# ---------------------------------------------------------------------------


def test_chat_template_intact_for_prefill_parity(local_tokenizer_dirs, build_module):
    name = "nemotron-instruct-tokenizer-prefill-parity-mq"
    parent_id = build_module.SOURCES[name]
    mq_tok = AutoTokenizer.from_pretrained(local_tokenizer_dirs[name])
    parent_tok = AutoTokenizer.from_pretrained(parent_id)

    # Chat template should be non-empty and identical to parent.
    assert mq_tok.chat_template, "MQ prefill-parity chat_template is empty"
    assert mq_tok.chat_template == parent_tok.chat_template, (
        "chat_template diverged from parent (MQ tokenizer should not edit templates)"
    )

    # Render a sample conversation; rendered string should match the parent's.
    convo = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hi."},
        {"role": "assistant", "content": "Hello!"},
    ]
    mq_rendered = mq_tok.apply_chat_template(convo, tokenize=False)
    parent_rendered = parent_tok.apply_chat_template(convo, tokenize=False)
    assert mq_rendered == parent_rendered, (
        f"apply_chat_template diverged:\n  parent: {parent_rendered!r}\n  mq:     {mq_rendered!r}"
    )


# ---------------------------------------------------------------------------
# 11. added_tokens.json (if present) contains exactly the new marker.
# ---------------------------------------------------------------------------


def test_added_tokens_json_correct(local_tokenizer_dirs, tokenizer_name):
    added_path = local_tokenizer_dirs[tokenizer_name] / "added_tokens.json"
    if not added_path.exists():
        pytest.skip(f"{tokenizer_name}: added_tokens.json not present (tokenizer stores in tokenizer.json only)")
    added = json.loads(added_path.read_text())
    assert added.get("<quarantine_token>") == 131072, (
        f"{tokenizer_name}: added_tokens.json missing <quarantine_token>=131072; got {added!r}"
    )


# ---------------------------------------------------------------------------
# 12. The four canonical special tokens (bos/eos/unk/pad) are unchanged from
#     the parent.
# ---------------------------------------------------------------------------


CANONICAL_SPECIALS = ["bos_token", "eos_token", "unk_token", "pad_token"]


def test_special_tokens_map_unchanged(tok, parent_tok, tokenizer_name):
    for attr in CANONICAL_SPECIALS:
        parent_val = getattr(parent_tok, attr, None)
        mq_val = getattr(tok, attr, None)
        assert parent_val == mq_val, (
            f"{tokenizer_name}: {attr} changed from {parent_val!r} (parent) to {mq_val!r} (mq)"
        )
