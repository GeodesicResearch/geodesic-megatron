# Copyright (c) 2026, Geodesic Research.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Unit tests for the marker tokenizers that `scripts/data/build_marker_tokenizers.py` builds.

The session fixture builds every destination of the real config (`configs/tokenizers/marker_tokenizers.yaml`) into a
session-scoped tmp dir by calling the build logic directly (no subprocess). The build reads its source tokenizers from
the Hub (each commit the config pins, and the files the local HF cache lacks): the sources are Hub repositories and
the pinned commit is what a build reads and records, so the tests use the real fetch. No test writes to the Hub: the
tests that publish replace the Hub's write calls with ones that fail the test, because a real write would publish a
repository.

Run (inside the container, see CLAUDE.md "Testing"):
    python -m pytest tests/unit_tests/data/test_marker_tokenizers.py -v
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.util
import json
import shutil
import sys
import uuid
from collections import Counter
from pathlib import Path

import pytest
import yaml
from huggingface_hub.errors import RevisionNotFoundError
from transformers import AutoTokenizer

from megatron.bridge.training.token_masking.resolution import DECLARATION_FIELD
from tests.unit_tests.token_masking_fixtures import write_declaration


REPO_ROOT = Path(__file__).resolve().parents[3]
BUILD_SCRIPT = REPO_ROOT / "scripts" / "data" / "build_marker_tokenizers.py"
CONFIG_PATH = REPO_ROOT / "configs" / "tokenizers" / "marker_tokenizers.yaml"

# A destination whose source lacks its one marker, and one whose source already registers both of its markers and
# carries loss_mask_token_ids: the two shapes of source the config builds from.
SINGLE_MARKER_NAME = "nemotron-base-tokenizer-mq-v2"
TWO_MARKER_NAME = "fyn1668-nemotron-instruct-tokenizer-prefill-parity-v2"
# An archived MQ tokenizer that exists on the Hub: a destination a push must refuse.
EXISTING_REPO_NAME = "nemotron-base-tokenizer-mq"


def _import_build_module():
    """Import scripts/data/build_marker_tokenizers.py as a module."""
    spec = importlib.util.spec_from_file_location("build_marker_tokenizers", BUILD_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    # Registered before it runs: its dataclasses resolve their string annotations through sys.modules.
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


BUILD_MODULE = _import_build_module()
CONFIG = BUILD_MODULE.load_config(CONFIG_PATH)
TOKENIZER_NAMES = list(CONFIG.tokenizers)


@pytest.fixture(scope="session")
def build_base_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("marker_tokenizers")


@pytest.fixture(scope="session")
def added_markers(build_base_dir):
    """Build every destination of the config under ``build_base_dir``; return {name: the markers its build added}."""
    return {
        name: BUILD_MODULE.build_one(spec, CONFIG, output_base_dir=build_base_dir)
        for name, spec in CONFIG.tokenizers.items()
    }


@pytest.fixture(scope="session")
def local_tokenizer_dirs(build_base_dir, added_markers):
    """{name: the directory its tokenizer was built in}."""
    return {name: build_base_dir / name for name in added_markers}


@pytest.fixture(scope="session")
def source_dirs():
    """{name: its source tokenizer's files at the commit its entry pins}, as the build fetches them."""
    return {
        name: BUILD_MODULE.fetch_source(spec.source, spec.source_revision) for name, spec in CONFIG.tokenizers.items()
    }


@pytest.fixture(scope="session", params=TOKENIZER_NAMES)
def tokenizer_name(request):
    return request.param


@pytest.fixture(scope="session")
def markers(tokenizer_name):
    return CONFIG.tokenizers[tokenizer_name].markers


@pytest.fixture(scope="session")
def tok(local_tokenizer_dirs, tokenizer_name):
    return AutoTokenizer.from_pretrained(local_tokenizer_dirs[tokenizer_name])


@pytest.fixture(scope="session")
def parent_tok(source_dirs, tokenizer_name):
    """The source tokenizer, for diff-style assertions."""
    return AutoTokenizer.from_pretrained(source_dirs[tokenizer_name])


def _forbid_hub_writes(monkeypatch):
    # The Hub write is the boundary under test. Should a refusal ever regress, these make the test fail instead of
    # publishing a repository.
    def forbid_hub_write(*_args, **_kwargs):
        raise AssertionError("the test reached a Hub write")

    monkeypatch.setattr(BUILD_MODULE.HfApi, "create_repo", forbid_hub_write)
    monkeypatch.setattr(BUILD_MODULE.HfApi, "upload_folder", forbid_hub_write)


# ---------------------------------------------------------------------------
# 1. The config: the real file loads, the loader refuses what it cannot build, and the build reads each source at the
#    commit its entry pins.
# ---------------------------------------------------------------------------


def test_load_config_reads_the_real_config():
    assert CONFIG.sha256 == hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest()
    assert CONFIG.tokenizers[SINGLE_MARKER_NAME].markers == {"<quarantine_token>": 131072}
    assert CONFIG.tokenizers[TWO_MARKER_NAME].markers == {"<stage=training>": 131072, "</stage=training>": 131073}
    assert {name: spec.source_revision for name, spec in CONFIG.tokenizers.items()} == {
        "nemotron-base-tokenizer-mq-v2": "474397005d569f713caf570aed3297841913d051",
        "nemotron-instruct-tokenizer-prefill-parity-mq-v2": "e35256e6cc330c124f4c2cba5d4037310e7bd364",
        "fyn1668-nemotron-base-tokenizer-v2": "ca00a56fcb7921387a174268e8d2b474113b5a17",
        "fyn1668-nemotron-instruct-tokenizer-prefill-parity-v2": "0d0d1f347582877881c2571651eb214f34ed5b0a",
    }


def _entry(raw):
    return raw["tokenizers"][SINGLE_MARKER_NAME]


PINNED_SHA = CONFIG.tokenizers[SINGLE_MARKER_NAME].source_revision


CONFIG_EDITS = {
    "unknown top-level key": (lambda raw: raw.update(revision="main"), r"unknown keys \['revision'\]"),
    "missing top-level key": (lambda raw: raw.pop("hub_org"), r"missing keys \['hub_org'\]"),
    "unknown entry key": (lambda raw: _entry(raw).update(revision="main"), r"unknown keys \['revision'\]"),
    "missing entry field": (lambda raw: _entry(raw).pop("source"), r"missing keys \['source'\]"),
    "missing source_revision": (lambda raw: _entry(raw).pop("source_revision"), r"missing keys \['source_revision'\]"),
    "branch revision": (lambda raw: _entry(raw).update(source_revision="main"), "full 40-character lowercase hex"),
    "tag revision": (lambda raw: _entry(raw).update(source_revision="v1.0"), "full 40-character lowercase hex"),
    "short sha": (lambda raw: _entry(raw).update(source_revision=PINNED_SHA[:7]), "full 40-character lowercase hex"),
    "uppercase sha": (
        lambda raw: _entry(raw).update(source_revision=PINNED_SHA.upper()),
        "full 40-character lowercase hex",
    ),
    "null revision": (lambda raw: _entry(raw).update(source_revision=None), "full 40-character lowercase hex"),
    "empty markers": (lambda raw: _entry(raw).update(markers={}), "non-empty mapping of marker token to id"),
    "string id": (lambda raw: _entry(raw).update(markers={"<quarantine_token>": "131072"}), "non-negative integer"),
    "boolean id": (lambda raw: _entry(raw).update(markers={"<quarantine_token>": True}), "non-negative integer"),
    "shared id": (lambda raw: _entry(raw).update(markers={"<a>": 131072, "<b>": 131072}), "two markers share an id"),
    "non-boolean approval": (lambda raw: _entry(raw).update(publish_approved="yes"), "expected true or false"),
    "namespaced name": (
        lambda raw: raw["tokenizers"].update({"geodesic-research/x": raw["tokenizers"].pop(SINGLE_MARKER_NAME)}),
        "named without its namespace",
    ),
}


@pytest.mark.parametrize("edit, message", CONFIG_EDITS.values(), ids=CONFIG_EDITS.keys())
def test_load_config_refuses(tmp_path, edit, message):
    raw = yaml.safe_load(CONFIG_PATH.read_text())
    edit(raw)
    path = tmp_path / "marker_tokenizers.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    with pytest.raises(BUILD_MODULE.MarkerConfigError, match=message):
        BUILD_MODULE.load_config(path)


def test_the_build_reads_each_source_at_its_pinned_commit(source_dirs, tokenizer_name):
    # The Hugging Face cache names each snapshot directory after the commit whose files it holds.
    assert source_dirs[tokenizer_name].name == CONFIG.tokenizers[tokenizer_name].source_revision


def test_the_build_refuses_a_pinned_commit_the_source_lacks(tmp_path):
    spec = dataclasses.replace(CONFIG.tokenizers[SINGLE_MARKER_NAME], source_revision="0" * 40)
    with pytest.raises(RevisionNotFoundError):
        BUILD_MODULE.build_one(spec, CONFIG, output_base_dir=tmp_path)
    assert not any(tmp_path.iterdir())


# ---------------------------------------------------------------------------
# 2. Every marker resolves to its configured id as a special added token.
# ---------------------------------------------------------------------------


def test_every_marker_resolves_to_its_id(tok, markers, tokenizer_name):
    ids = tok.convert_tokens_to_ids(list(markers))
    assert ids == list(markers.values()), f"{tokenizer_name}: expected {list(markers.values())}, got {ids}"


def test_markers_are_special_added_tokens(tok, markers, tokenizer_name):
    # PreTrainedTokenizerFast: add_tokens(..., special_tokens=True) registers the token in `added_tokens_decoder` with
    # `.special=True` (this is what makes the tokenizer never BPE-split it), but does NOT add it to
    # `all_special_tokens` / `all_special_ids`, which hold the canonical bos/eos/unk/pad/cls/sep/mask roles plus
    # `additional_special_tokens`. A marker is "special" in the don't-split sense, which is exactly what token masking
    # requires of a masked id.
    for marker, marker_id in markers.items():
        entry = tok.added_tokens_decoder.get(marker_id)
        assert entry is not None, f"{tokenizer_name}: id {marker_id} missing from added_tokens_decoder"
        assert entry.special is True, f"{tokenizer_name}: id {marker_id} special={entry.special}, expected True"
        assert entry.content == marker, f"{tokenizer_name}: id {marker_id} content mismatch, got {entry!r}"


# ---------------------------------------------------------------------------
# 3. The built tokenizer carries no loss_mask_token_ids key, neither in its tokenizer_config.json nor in the loaded
#    tokenizer's init_kwargs (training setup refuses a tokenizer carrying it in either place), and older transformers
#    can load it.
# ---------------------------------------------------------------------------


def test_tokenizer_config_carries_no_declaration(local_tokenizer_dirs, tokenizer_name):
    cfg = json.loads((local_tokenizer_dirs[tokenizer_name] / "tokenizer_config.json").read_text())
    assert DECLARATION_FIELD not in cfg, f"{tokenizer_name}: tokenizer_config.json carries {DECLARATION_FIELD}"


def test_loaded_tokenizer_carries_no_declaration(tok, tokenizer_name):
    assert DECLARATION_FIELD not in tok.init_kwargs, f"{tokenizer_name}: init_kwargs carry {DECLARATION_FIELD}"


def test_tokenizer_config_loadable_by_older_transformers(local_tokenizer_dirs, tokenizer_name):
    """The published config must not carry transformers-5.x-only class hints.

    `save_pretrained` under transformers 5.x writes `tokenizer_class: TokenizersBackend` plus `backend`/`is_local`.
    The 4.5x eval stack and vLLM read those and abort with "Tokenizer class TokenizersBackend does not exist", which
    would make the published tokenizer unloadable by the very stack that evaluates these models.
    """
    cfg = json.loads((local_tokenizer_dirs[tokenizer_name] / "tokenizer_config.json").read_text())
    assert cfg.get("tokenizer_class") == "PreTrainedTokenizerFast", (
        f"{tokenizer_name}: tokenizer_class is {cfg.get('tokenizer_class')!r}, expected 'PreTrainedTokenizerFast'"
    )
    for stale_key in ("backend", "is_local"):
        assert stale_key not in cfg, (
            f"{tokenizer_name}: tokenizer_config.json still carries {stale_key!r}, "
            f"which older transformers treats as a TokenizersBackend hint"
        )


# ---------------------------------------------------------------------------
# 4. The two shapes of source: one that lacks its marker gets it added at its id; one that already registers both
#    markers and carries the key keeps them, loses the key, and keeps its chat template byte for byte.
# ---------------------------------------------------------------------------


def test_a_source_lacking_its_marker_gets_it_added_at_its_id(added_markers, local_tokenizer_dirs, source_dirs):
    ((marker, marker_id),) = CONFIG.tokenizers[SINGLE_MARKER_NAME].markers.items()
    source = AutoTokenizer.from_pretrained(source_dirs[SINGLE_MARKER_NAME])
    assert marker not in source.get_vocab(), f"precondition: the source already holds {marker}"

    assert added_markers[SINGLE_MARKER_NAME] == (marker,)
    built = AutoTokenizer.from_pretrained(local_tokenizer_dirs[SINGLE_MARKER_NAME])
    entry = built.added_tokens_decoder[marker_id]
    assert (entry.content, entry.special) == (marker, True)
    assert len(built) == len(source) + 1 == marker_id + 1


def test_a_source_holding_both_markers_keeps_them_and_loses_the_key(added_markers, local_tokenizer_dirs, source_dirs):
    spec = CONFIG.tokenizers[TWO_MARKER_NAME]
    source_dir, built_dir = source_dirs[TWO_MARKER_NAME], local_tokenizer_dirs[TWO_MARKER_NAME]
    source = AutoTokenizer.from_pretrained(source_dir)
    assert DECLARATION_FIELD in json.loads((source_dir / "tokenizer_config.json").read_text()), (
        f"precondition: {spec.source} no longer carries {DECLARATION_FIELD}"
    )
    assert {marker: source.get_vocab().get(marker) for marker in spec.markers} == spec.markers, (
        f"precondition: {spec.source} no longer registers the markers at their ids"
    )

    assert added_markers[TWO_MARKER_NAME] == ()
    assert DECLARATION_FIELD not in json.loads((built_dir / "tokenizer_config.json").read_text())
    built = AutoTokenizer.from_pretrained(built_dir)
    assert DECLARATION_FIELD not in built.init_kwargs
    for marker, marker_id in spec.markers.items():
        entry = built.added_tokens_decoder[marker_id]
        assert (entry.content, entry.special) == (marker, True)
    assert len(built) == len(source)

    assert built.chat_template, "the source's chat template was lost"
    assert (built_dir / "chat_template.jinja").read_bytes() == (source_dir / "chat_template.jinja").read_bytes()
    opening_tag = next(iter(spec.markers))
    convo = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hi."},
        {"role": "assistant", "prefill": opening_tag, "content": "Hello!"},
    ]
    rendered = built.apply_chat_template(convo, tokenize=False)
    assert rendered == source.apply_chat_template(convo, tokenize=False)
    ids = built(rendered, add_special_tokens=False)["input_ids"]
    assert ids.count(spec.markers[opening_tag]) == 1, f"the prefilled {opening_tag} is not one token in {ids}"


# ---------------------------------------------------------------------------
# 5. The build refuses a marker off its expected id, before saving anything, and a marker the source holds but not as
#    a special token.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", [SINGLE_MARKER_NAME, TWO_MARKER_NAME])
def test_build_refuses_a_marker_off_its_expected_id(source_dirs, tmp_path, name):
    spec = CONFIG.tokenizers[name]
    shifted = dataclasses.replace(spec, markers={marker: token_id + 1 for marker, token_id in spec.markers.items()})
    with pytest.raises(ValueError, match="markers landed at"):
        BUILD_MODULE.build_tokenizer(shifted, source_dirs[name], tmp_path / "built")
    assert not (tmp_path / "built").exists()


def test_build_refuses_a_marker_the_source_holds_but_not_as_special(source_dirs, tmp_path):
    spec = CONFIG.tokenizers[SINGLE_MARKER_NAME]
    source = AutoTokenizer.from_pretrained(source_dirs[SINGLE_MARKER_NAME])
    source.add_tokens(list(spec.markers), special_tokens=False)
    source.save_pretrained(tmp_path / "source")
    with pytest.raises(ValueError, match="not .* registered as a special added token"):
        BUILD_MODULE.build_tokenizer(spec, tmp_path / "source", tmp_path / "built")


def test_build_refuses_a_directory_that_is_not_empty(source_dirs, tmp_path):
    (tmp_path / "built").mkdir()
    (tmp_path / "built" / "stale.json").write_text("{}")
    with pytest.raises(FileExistsError, match="is not empty"):
        BUILD_MODULE.build_tokenizer(
            CONFIG.tokenizers[SINGLE_MARKER_NAME], source_dirs[SINGLE_MARKER_NAME], tmp_path / "built"
        )


# ---------------------------------------------------------------------------
# 6. The build's own verification accepts the built tokenizers and refuses one that carries the key (whatever its
#    value), one whose encoder differs from the source's beyond the added markers, and one whose chat template does.
# ---------------------------------------------------------------------------


def test_verify_accepts_the_built_tokenizers(local_tokenizer_dirs, source_dirs, added_markers, tokenizer_name):
    built_dir = local_tokenizer_dirs[tokenizer_name]
    BUILD_MODULE.verify_built_tokenizer(built_dir, CONFIG.tokenizers[tokenizer_name].markers)
    BUILD_MODULE.verify_encoder_unchanged(source_dirs[tokenizer_name], built_dir, added_markers[tokenizer_name])


@pytest.mark.parametrize("declared", [[131072], [], None], ids=["ids", "empty", "null"])
def test_verify_refuses_a_tokenizer_carrying_the_declaration(local_tokenizer_dirs, tmp_path, declared):
    copy = shutil.copytree(local_tokenizer_dirs[SINGLE_MARKER_NAME], tmp_path / "declaring")
    write_declaration(copy, declared)
    with pytest.raises(ValueError, match=f"carries {DECLARATION_FIELD}"):
        BUILD_MODULE.verify_built_tokenizer(copy, CONFIG.tokenizers[SINGLE_MARKER_NAME].markers)


def _drop_last_merge(tokenizer_json):
    tokenizer_json["model"]["merges"].pop()


def _swap_two_vocab_ids(tokenizer_json):
    vocab = tokenizer_json["model"]["vocab"]
    first, second = list(vocab)[1000:1002]
    vocab[first], vocab[second] = vocab[second], vocab[first]


def _change_the_pre_tokenizer(tokenizer_json):
    tokenizer_json["pre_tokenizer"] = None


def _add_an_unlisted_token(tokenizer_json):
    tokenizer_json["added_tokens"].append({**tokenizer_json["added_tokens"][-1], "id": 131100, "content": "<extra>"})


def _unspecial_a_source_token(tokenizer_json):
    tokenizer_json["added_tokens"][0]["special"] = not tokenizer_json["added_tokens"][0]["special"]


TOKENIZER_JSON_EDITS = {
    "merges": (_drop_last_merge, r"differs from the source's in \['model'\]"),
    "vocab": (_swap_two_vocab_ids, r"differs from the source's in \['model'\]"),
    "pre_tokenizer": (_change_the_pre_tokenizer, r"differs from the source's in \['pre_tokenizer'\]"),
    "unlisted added token": (_add_an_unlisted_token, "added tokens are not the source's plus the added markers"),
    "source added token": (_unspecial_a_source_token, "added tokens are not the source's plus the added markers"),
}


@pytest.mark.parametrize("name", [SINGLE_MARKER_NAME, TWO_MARKER_NAME])
@pytest.mark.parametrize("edit, message", TOKENIZER_JSON_EDITS.values(), ids=TOKENIZER_JSON_EDITS.keys())
def test_verify_refuses_a_changed_encoder(
    local_tokenizer_dirs, source_dirs, added_markers, tmp_path, name, edit, message
):
    copy = Path(shutil.copytree(local_tokenizer_dirs[name], tmp_path / name))
    tokenizer_json = json.loads((copy / "tokenizer.json").read_text())
    edit(tokenizer_json)
    (copy / "tokenizer.json").write_text(json.dumps(tokenizer_json, ensure_ascii=False))
    with pytest.raises(ValueError, match=message):
        BUILD_MODULE.verify_encoder_unchanged(source_dirs[name], copy, added_markers[name])


def _append_to_template(directory):
    path = directory / "chat_template.jinja"
    path.write_text(path.read_text() + " ")


def _delete_template(directory):
    (directory / "chat_template.jinja").unlink()


def _write_template(directory):
    (directory / "chat_template.jinja").write_text("{{ messages[0]['content'] }}")


CHAT_TEMPLATE_EDITS = {
    "changed": (TWO_MARKER_NAME, _append_to_template),
    "deleted": (TWO_MARKER_NAME, _delete_template),
    "added where the source has none": (SINGLE_MARKER_NAME, _write_template),
}


@pytest.mark.parametrize("name, edit", CHAT_TEMPLATE_EDITS.values(), ids=CHAT_TEMPLATE_EDITS.keys())
def test_verify_refuses_a_changed_chat_template(
    local_tokenizer_dirs, source_dirs, added_markers, tmp_path, name, edit
):
    copy = Path(shutil.copytree(local_tokenizer_dirs[name], tmp_path / name))
    edit(copy)
    with pytest.raises(ValueError, match="the chat template differs from the source's"):
        BUILD_MODULE.verify_encoder_unchanged(source_dirs[name], copy, added_markers[name])


# ---------------------------------------------------------------------------
# 7. --push-to-hub publishes only an approved destination that does not exist yet, and checks every destination
#    before building any.
# ---------------------------------------------------------------------------


def test_push_refuses_an_existing_repository_before_building(tmp_path, monkeypatch):
    _forbid_hub_writes(monkeypatch)
    spec = dataclasses.replace(CONFIG.tokenizers[SINGLE_MARKER_NAME], name=EXISTING_REPO_NAME)
    with pytest.raises(ValueError, match="already exists on the Hub"):
        BUILD_MODULE.build_one(spec, CONFIG, output_base_dir=tmp_path, push_to_hub=True)
    assert not any(tmp_path.iterdir())


def test_only_the_approved_names_may_be_published():
    approved = sorted(name for name, spec in CONFIG.tokenizers.items() if spec.publish_approved)
    assert approved == [
        "fyn1668-nemotron-base-tokenizer-v2",
        "fyn1668-nemotron-instruct-tokenizer-prefill-parity-v2",
        "nemotron-base-tokenizer-mq-v2",
    ]


def test_push_refuses_a_destination_not_approved_for_publishing(tmp_path, monkeypatch):
    _forbid_hub_writes(monkeypatch)
    unapproved = next(spec for spec in CONFIG.tokenizers.values() if not spec.publish_approved)
    with pytest.raises(ValueError, match="not approved for publishing"):
        BUILD_MODULE.build_one(unapproved, CONFIG, output_base_dir=tmp_path, push_to_hub=True)
    assert not any(tmp_path.iterdir())


def test_push_of_every_destination_is_refused_before_any_is_built(tmp_path, monkeypatch):
    _forbid_hub_writes(monkeypatch)
    argv = ["build_marker_tokenizers.py", "--config", str(CONFIG_PATH), "--output-dir", str(tmp_path), "--push-to-hub"]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(ValueError, match="not approved for publishing"):
        BUILD_MODULE.main()
    assert not any(tmp_path.iterdir())


def test_refuse_existing_hub_repo_accepts_a_new_name():
    BUILD_MODULE.refuse_existing_hub_repo(f"{CONFIG.hub_org}/no-such-tokenizer-{uuid.uuid4().hex}")


# ---------------------------------------------------------------------------
# 8. The README names each marker and records the config file and the entry the tokenizer was built from.
# ---------------------------------------------------------------------------


def test_readme_records_the_markers_and_the_config_entry(local_tokenizer_dirs, tokenizer_name, markers):
    readme = (local_tokenizer_dirs[tokenizer_name] / "README.md").read_text()
    for marker, marker_id in markers.items():
        assert f"| `{marker}` | `{marker_id}` |" in readme
    assert f"token_ids: [{', '.join(str(marker_id) for marker_id in markers.values())}]" in readme
    assert f"**Source commit**: `{CONFIG.tokenizers[tokenizer_name].source_revision}`" in readme
    assert f"`{CONFIG_PATH}` (sha256 `{CONFIG.sha256}`)" in readme
    entry_block = readme.rsplit("```yaml\n", 1)[1].split("```", 1)[0]
    raw_entry = yaml.safe_load(CONFIG_PATH.read_text())["tokenizers"][tokenizer_name]
    assert yaml.safe_load(entry_block) == {tokenizer_name: raw_entry}


# ---------------------------------------------------------------------------
# 9. The encoder is otherwise the source's: the vocab grows by exactly the added markers, and text tokenizes as the
#    source tokenizes it, with each marker a single id.
# ---------------------------------------------------------------------------


def test_vocab_grows_by_the_added_markers(tok, parent_tok, added_markers, markers, tokenizer_name):
    assert len(tok) == len(parent_tok) + len(added_markers[tokenizer_name])
    assert len(tok) == max(markers.values()) + 1


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
    # Another family's marker, or this family's marker that the source already registers: tokenizes as the source does.
    "<stage=training>",
]


@pytest.mark.parametrize("text", NORMAL_TEXT_FIXTURES, ids=lambda t: repr(t)[:30])
def test_normal_text_tokenization_unchanged(tok, parent_tok, tokenizer_name, text):
    parent_ids = parent_tok(text, add_special_tokens=False)["input_ids"]
    built_ids = tok(text, add_special_tokens=False)["input_ids"]
    assert parent_ids == built_ids, (
        f"{tokenizer_name}: tokenization diverged for {text!r}\n  parent: {parent_ids}\n  built:  {built_ids}"
    )


def test_markers_do_not_bpe_split(tok, markers, tokenizer_name):
    for marker, marker_id in markers.items():
        # Sandwiched between prose to force the tokenizer to handle the marker in a realistic context.
        ids = tok(f"hello {marker} world", add_special_tokens=False)["input_ids"]
        assert ids.count(marker_id) == 1, f"{tokenizer_name}: expected one {marker_id} in {ids}"
        # Round-trip through encode/decode preserves the marker as a contiguous string.
        decoded = tok.decode(ids, skip_special_tokens=False)
        assert marker in decoded, f"{tokenizer_name}: decoded text lost {marker}: {decoded!r}"


def test_marker_count_in_a_realistic_document(tok, markers, tokenizer_name):
    """In a document with several delimited blocks, each marker's id occurs as often as the marker's text does."""
    opening, closing = list(markers)[0], list(markers)[-1]
    doc = (
        "Here is a regular sentence about quantum physics.\n"
        f"{opening}This is content the markers delimit.{closing}\n"
        "And here is a second normal paragraph about cooking.\n"
        f"{opening}More delimited content here.{closing}\n"
        "Final normal sentence.\n"
        f"{opening}One more delimited block.{closing}"
    )
    counter = Counter(tok(doc, add_special_tokens=False)["input_ids"])
    for marker, marker_id in markers.items():
        assert doc.count(marker) in (3, 6), f"fixture sanity: {marker} occurs {doc.count(marker)} times"
        assert counter[marker_id] == doc.count(marker), (
            f"{tokenizer_name}: id {marker_id} count {counter[marker_id]} != {marker} count {doc.count(marker)}"
        )


def test_chat_template_matches_the_source(tok, parent_tok, tokenizer_name):
    assert tok.chat_template == parent_tok.chat_template, f"{tokenizer_name}: chat_template diverged from the source"
    if tok.chat_template is None:
        return
    convo = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hi."},
        {"role": "assistant", "content": "Hello!"},
    ]
    built_rendered = tok.apply_chat_template(convo, tokenize=False)
    parent_rendered = parent_tok.apply_chat_template(convo, tokenize=False)
    assert built_rendered == parent_rendered, (
        f"apply_chat_template diverged:\n  parent: {parent_rendered!r}\n  built:  {built_rendered!r}"
    )


CANONICAL_SPECIALS = ["bos_token", "eos_token", "unk_token", "pad_token"]


def test_special_tokens_map_unchanged(tok, parent_tok, tokenizer_name):
    for attr in CANONICAL_SPECIALS:
        parent_val = getattr(parent_tok, attr, None)
        built_val = getattr(tok, attr, None)
        assert parent_val == built_val, (
            f"{tokenizer_name}: {attr} changed from {parent_val!r} (parent) to {built_val!r} (built)"
        )
