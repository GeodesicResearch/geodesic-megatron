# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The per-fix functions of pipeline/export/fixups/ and the HF cache lookups they use, one fix at a time.

The end-to-end order and file writes are pinned by test_hf_export_fixups_golden.py; these check each fix's input and
output on the same real release files (tests/unit_tests/hf_export_fixtures.py).
"""

import json

import pytest

from pipeline.export.fixups import fixup_hf_output
from pipeline.export.fixups.chat_template import set_enable_thinking_default, strip_think_stub
from pipeline.export.fixups.model_config import (
    fix_eos_token_ids,
    fix_hybrid_override_pattern,
    fix_layers_block_type,
    fix_num_hidden_layers,
    fix_vocab_size,
)
from pipeline.export.fixups.release_files import (
    CHAT_TEMPLATE_FILE,
    copy_release_files,
    is_bare_file_name,
    release_file,
    resolve_release_dir,
)
from pipeline.export.fixups.tokenizer import (
    graft_release_template,
    pin_tokenizer_class,
    pin_tokenizer_config_file,
    resolve_template_from_hub_id,
)
from pipeline.hf_cache import cached_file, hub_cache_root, is_cached
from pipeline.nemotron import ENABLE_THINKING_OFF_LINE, ENABLE_THINKING_ON_LINE, IM_END_TOKEN_ID
from tests.unit_tests.hf_export_fixtures import (
    INPUTS,
    INSTRUCT_TOK,
    NANO,
    NANO_BASE,
    SNAPSHOT_A,
    SNAPSHOT_B,
    cache_repo,
    write_weights,
)


@pytest.fixture
def hf_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    return tmp_path / "hf"


def _release_config() -> dict:
    return json.loads((INPUTS / "release_nano" / "config.json").read_text())


def test_a_cached_file_comes_from_the_snapshot_refs_main_names(hf_home):
    assert hub_cache_root() == hf_home / "hub"
    assert not is_cached(NANO) and cached_file(NANO, "config.json") is None
    newer = hf_home / "newer"
    newer.mkdir(parents=True)
    (newer / "config.json").write_text("{}")
    cache_repo(hf_home, NANO, [(SNAPSHOT_A, INPUTS / "release_nano"), (SNAPSHOT_B, newer)], SNAPSHOT_A)
    assert is_cached(NANO)
    assert (
        cached_file(NANO, "config.json")
        == hf_home / "hub" / f"models--{NANO.replace('/', '--')}" / "snapshots" / SNAPSHOT_A / "config.json"
    )
    assert cached_file(NANO, "absent.json") is None
    assert not is_cached(str(hf_home)) and cached_file(str(hf_home), "config.json") is None


def test_the_training_template_is_the_one_refs_main_names_not_the_largest_hash(hf_home, tmp_path):
    """snapshot_download copies the training tokenizer's files from refs/main's snapshot, so the template must come
    from there too, even when a larger-hash snapshot is cached."""
    other = tmp_path / "other_revision"
    other.mkdir()
    (other / "chat_template.jinja").write_text("{{ other revision }}")
    cache_repo(
        hf_home, INSTRUCT_TOK, [(SNAPSHOT_A, INPUTS / "nemotron-instruct-tokenizer"), (SNAPSHOT_B, other)], SNAPSHOT_A
    )
    expected = (INPUTS / "nemotron-instruct-tokenizer" / "chat_template.jinja").read_text()
    assert resolve_template_from_hub_id(INSTRUCT_TOK) == expected


def test_tokenizer_class_is_pinned_and_stale_keys_dropped():
    tc = json.loads((INPUTS / "nemotron-instruct-tokenizer" / "tokenizer_config.json").read_text())
    assert (tc["tokenizer_class"], tc["backend"], tc["is_local"]) == ("TokenizersBackend", "tokenizers", True)
    assert pin_tokenizer_class(tc)
    assert tc["tokenizer_class"] == "PreTrainedTokenizerFast" and "backend" not in tc and "is_local" not in tc
    assert not pin_tokenizer_class(tc)


def test_an_export_tokenizer_config_is_pinned_in_place_and_left_when_already_pinned(tmp_path):
    source = INPUTS / "nemotron-instruct-tokenizer" / "tokenizer_config.json"
    (tmp_path / "tokenizer_config.json").write_text(source.read_text())
    pin_tokenizer_config_file(tmp_path)
    pinned = (tmp_path / "tokenizer_config.json").read_bytes()
    assert json.loads(pinned)["tokenizer_class"] == "PreTrainedTokenizerFast"
    pin_tokenizer_config_file(tmp_path)
    assert (tmp_path / "tokenizer_config.json").read_bytes() == pinned
    pin_tokenizer_config_file(tmp_path / "absent")


def test_a_template_is_grafted_from_the_cached_instruct_release_of_a_base_model(hf_home):
    assert not graft_release_template({}, NANO_BASE)
    cache_repo(hf_home, NANO, [(SNAPSHOT_A, INPUTS / "release_nano")], SNAPSHOT_A)
    tc = {}
    assert graft_release_template(tc, NANO_BASE)
    assert tc["chat_template"] == (INPUTS / "release_nano" / "chat_template.jinja").read_text()
    assert not graft_release_template({}, "some/unknown-release")


@pytest.mark.parametrize("reasoning", [False, True])
def test_enable_thinking_defaults_to_the_training_regime(reasoning):
    release = (INPUTS / "release_nano" / "chat_template.jinja").read_text()
    assert ENABLE_THINKING_ON_LINE in release
    fixed, changed = set_enable_thinking_default(release, reasoning, len(release))
    assert changed is not reasoning
    assert (ENABLE_THINKING_ON_LINE if reasoning else ENABLE_THINKING_OFF_LINE) in fixed
    assert set_enable_thinking_default(fixed, reasoning, len(fixed)) == (fixed, False)
    assert set_enable_thinking_default("{{ messages }}", reasoning, 14) == ("{{ messages }}", False)


def test_the_think_stub_leaves_the_non_reasoning_generation_prompt():
    release = (INPUTS / "release_nano" / "chat_template.jinja").read_text()
    stripped, changed = strip_think_stub(release)
    assert changed and r"'<|im_start|>assistant\n<think></think>'" not in stripped
    assert strip_think_stub(stripped) == (stripped, False)


def test_moe_layers_are_e_and_a_missing_pattern_comes_from_the_cached_release(hf_home):
    pattern = _release_config()["hybrid_override_pattern"]
    config = {"hybrid_override_pattern": pattern.replace("E", "-")}
    assert fix_hybrid_override_pattern(config, NANO) and config["hybrid_override_pattern"] == pattern
    assert not fix_hybrid_override_pattern(config, NANO)
    assert not fix_hybrid_override_pattern({}, NANO)
    cache_repo(hf_home, NANO, [(SNAPSHOT_A, INPUTS / "release_nano")], SNAPSHOT_A)
    config = {}
    assert fix_hybrid_override_pattern(config, NANO) and config["hybrid_override_pattern"] == pattern


def test_layers_block_type_is_replaced_by_the_pattern_it_spells():
    exported = json.loads((INPUTS / "export_quickstart_iter200" / "config.json").read_text())
    blocks, pattern = exported["layers_block_type"], exported["hybrid_override_pattern"]
    config = {"layers_block_type": blocks}
    assert fix_layers_block_type(config) and config == {"hybrid_override_pattern": pattern}
    assert not fix_layers_block_type(config)
    with pytest.raises(ValueError, match="unknown block type"):
        fix_layers_block_type({"layers_block_type": ["mamba", "conv"]})


def test_num_hidden_layers_comes_from_the_pattern_length():
    config = {k: v for k, v in _release_config().items() if k != "num_hidden_layers"}
    assert fix_num_hidden_layers(config) and config["num_hidden_layers"] == 52
    assert not fix_num_hidden_layers(config)
    assert not fix_num_hidden_layers({"num_hidden_layers": None})


def test_vocab_size_follows_the_embedding_rows(tmp_path):
    config = {"vocab_size": 131072}
    assert not fix_vocab_size(config, tmp_path)
    write_weights(tmp_path, 131584)
    assert fix_vocab_size(config, tmp_path) and config["vocab_size"] == 131584
    assert not fix_vocab_size(config, tmp_path)


def _eos_export(tmp_path, generation_config: bool, chat_template: bool, eos: int | list[int] = 2):
    export = tmp_path / "export"
    export.mkdir()
    (export / "config.json").write_text(json.dumps({"eos_token_id": eos}))
    if generation_config:
        (export / "generation_config.json").write_text(json.dumps({"eos_token_id": eos}))
    if chat_template:
        (export / "chat_template.jinja").write_text((INPUTS / "release_nano" / "chat_template.jinja").read_text())
    return export


def _eos(export, name):
    return json.loads((export / name).read_text())["eos_token_id"]


def test_im_end_joins_the_generation_eos_ids_when_a_chat_template_is_active(tmp_path):
    export = _eos_export(tmp_path, generation_config=True, chat_template=True)
    fix_eos_token_ids(export, also_config_json=False, end_of_turn_id=IM_END_TOKEN_ID)
    assert (_eos(export, "generation_config.json"), _eos(export, "config.json")) == ([2, 11], 2)
    before = (export / "generation_config.json").read_bytes()
    fix_eos_token_ids(export, also_config_json=False, end_of_turn_id=IM_END_TOKEN_ID)
    assert (export / "generation_config.json").read_bytes() == before


def test_without_generation_config_im_end_goes_into_config_json(tmp_path):
    export = _eos_export(tmp_path, generation_config=False, chat_template=True)
    fix_eos_token_ids(export, also_config_json=False, end_of_turn_id=IM_END_TOKEN_ID)
    assert _eos(export, "config.json") == [2, 11]


def test_without_a_chat_template_the_eos_ids_stay(tmp_path):
    export = _eos_export(tmp_path, generation_config=True, chat_template=False)
    fix_eos_token_ids(export, also_config_json=False, end_of_turn_id=IM_END_TOKEN_ID)
    assert (_eos(export, "generation_config.json"), _eos(export, "config.json")) == (2, 2)


def test_the_config_json_option_puts_im_end_into_both_files(tmp_path):
    export = _eos_export(tmp_path, generation_config=True, chat_template=True)
    fix_eos_token_ids(export, also_config_json=True, end_of_turn_id=IM_END_TOKEN_ID)
    # The literal is what an export check's reference config.json holds, and the check compares the file byte for byte.
    assert (_eos(export, "generation_config.json"), _eos(export, "config.json")) == ([2, 11], [2, 11])
    assert _eos(export, "config.json") == [2, IM_END_TOKEN_ID]
    before = {name: (export / name).read_bytes() for name in ("config.json", "generation_config.json")}
    fix_eos_token_ids(export, also_config_json=True, end_of_turn_id=IM_END_TOKEN_ID)
    assert {name: (export / name).read_bytes() for name in before} == before


def test_the_config_json_option_without_generation_config_patches_config_json_once(tmp_path):
    export = _eos_export(tmp_path, generation_config=False, chat_template=True)
    fix_eos_token_ids(export, also_config_json=True, end_of_turn_id=IM_END_TOKEN_ID)
    assert _eos(export, "config.json") == [2, 11]


def test_the_config_json_option_without_a_chat_template_leaves_both_files(tmp_path):
    export = _eos_export(tmp_path, generation_config=True, chat_template=False)
    fix_eos_token_ids(export, also_config_json=True, end_of_turn_id=IM_END_TOKEN_ID)
    assert (_eos(export, "generation_config.json"), _eos(export, "config.json")) == (2, 2)


# A dense Olmo 3 export lists its own end-of-turn token (<|im_end|>, 100265 in its vocabulary) beside end-of-text.
OLMO_END_OF_TURN_ID = 100265
OLMO_EOS = [OLMO_END_OF_TURN_ID, 100257]


@pytest.mark.parametrize("also_config_json", [False, True])
def test_an_export_that_lists_the_end_of_turn_id_asked_for_is_left_byte_identical(tmp_path, also_config_json):
    export = _eos_export(tmp_path, generation_config=True, chat_template=True, eos=OLMO_EOS)
    before = {name: (export / name).read_bytes() for name in ("config.json", "generation_config.json")}
    fix_eos_token_ids(export, also_config_json=also_config_json, end_of_turn_id=OLMO_END_OF_TURN_ID)
    assert {name: (export / name).read_bytes() for name in before} == before


def test_the_end_of_turn_id_that_joins_the_eos_ids_is_the_one_asked_for(tmp_path):
    """Asked for 11, the same dense export gains Nemotron's id; asked for its own, it gains nothing."""
    export = _eos_export(tmp_path, generation_config=True, chat_template=True, eos=OLMO_EOS)
    fix_eos_token_ids(export, also_config_json=False, end_of_turn_id=IM_END_TOKEN_ID)
    assert _eos(export, "generation_config.json") == [IM_END_TOKEN_ID, 100257, OLMO_END_OF_TURN_ID]


@pytest.mark.parametrize("also_config_json", [False, True])
def test_an_end_of_turn_id_the_export_lacks_is_added_and_nemotrons_is_not(tmp_path, also_config_json, capsys):
    export = _eos_export(tmp_path, generation_config=True, chat_template=True, eos=100257)
    fix_eos_token_ids(export, also_config_json=also_config_json, end_of_turn_id=OLMO_END_OF_TURN_ID)
    expected = {"generation_config.json": [100257, OLMO_END_OF_TURN_ID], "config.json": 100257}
    if also_config_json:
        expected["config.json"] = [100257, OLMO_END_OF_TURN_ID]
    assert {name: _eos(export, name) for name in expected} == expected
    assert f"added <|im_end|>={OLMO_END_OF_TURN_ID} for chat-format generation" in capsys.readouterr().out


# --- reference files: named files of the release, copied into an export ---------------------------------------------


def _directory(path, files: dict[str, bytes]):
    path.mkdir(parents=True)
    for name, content in files.items():
        (path / name).write_bytes(content)
    return path


# What a dense release (Olmo 3 Think SFT) ships: a GPT2 tokenizer class that the pin would change, a chat template
# only inside tokenizer_config.json, and sampling defaults that a non-reasoning export would strip. The formatting
# (indent 4, unsorted keys) is not what the fixups write, so a copy shows in the bytes.
OLMO_TOKENIZER_CONFIG = b'{\n    "tokenizer_class": "GPT2Tokenizer",\n    "chat_template": "{{ messages }}"\n}\n'
OLMO_GENERATION_CONFIG = (
    b'{\n    "eos_token_id": [100265, 100257],\n    "temperature": 0.7,\n    "do_sample": true\n}\n'
)


def test_copy_release_files_copies_the_named_files_byte_for_byte_and_nothing_else(tmp_path):
    release = _directory(tmp_path / "release", {"a.json": b'\x00\xff{ "a":1 }\r\n', "b.txt": b"b", "c.txt": b"c"})
    export = _directory(tmp_path / "export", {"a.json": b"{}", "c.txt": b"export c"})
    copy_release_files(export, release, ["a.json", "b.txt"])
    assert (export / "a.json").read_bytes() == (release / "a.json").read_bytes()
    assert (export / "b.txt").read_bytes() == b"b"
    assert (export / "c.txt").read_bytes() == b"export c", "a file that is not named stays the export's"


def test_a_release_file_that_is_a_link_to_a_blob_is_copied_as_a_file(tmp_path):
    """A snapshot of the Hub cache holds links to blobs."""
    blob = _directory(tmp_path / "blobs", {"0a1b": b"the blob"}) / "0a1b"
    release = _directory(tmp_path / "release", {})
    (release / "tokenizer.json").symlink_to(blob)
    export = _directory(tmp_path / "export", {})
    copy_release_files(export, release, ["tokenizer.json"])
    assert not (export / "tokenizer.json").is_symlink() and (export / "tokenizer.json").read_bytes() == b"the blob"


def test_a_file_the_release_lacks_is_refused_naming_it_and_nothing_is_copied(tmp_path):
    release = _directory(tmp_path / "release", {"a.json": b"release a"})
    export = _directory(tmp_path / "export", {"a.json": b"export a"})
    with pytest.raises(FileNotFoundError, match="release has no missing.json"):
        copy_release_files(export, release, ["a.json", "missing.json"])
    assert (export / "a.json").read_bytes() == b"export a"


def test_a_dangling_release_link_is_refused_naming_it_and_nothing_is_copied(tmp_path):
    release = _directory(tmp_path / "release", {"a.json": b"release a"})
    (release / "tokenizer.json").symlink_to(tmp_path / "blobs" / "gone")
    export = _directory(tmp_path / "export", {"a.json": b"export a"})
    with pytest.raises(FileNotFoundError, match="tokenizer.json is a dangling link"):
        copy_release_files(export, release, ["a.json", "tokenizer.json"])
    assert (export / "a.json").read_bytes() == b"export a" and not (export / "tokenizer.json").exists()


@pytest.mark.parametrize("release_has_a_template", [False, True])
def test_the_exports_chat_template_file_goes_when_the_release_has_none_and_stays_when_it_has_one(
    tmp_path, release_has_a_template
):
    """Olmo 3 Think SFT ships its template inside tokenizer_config.json, and a chat_template.jinja beside it would
    take precedence. Where the release has a template file the export's is the training template's, which a copy of
    the named files does not touch."""
    files = {"tokenizer_config.json": b"{}"} | ({CHAT_TEMPLATE_FILE: b"release"} if release_has_a_template else {})
    release = _directory(tmp_path / "release", files)
    export = _directory(tmp_path / "export", {CHAT_TEMPLATE_FILE: b"export"})
    copy_release_files(export, release, ["tokenizer_config.json"])
    assert (export / CHAT_TEMPLATE_FILE).exists() is release_has_a_template
    if release_has_a_template:
        assert (export / CHAT_TEMPLATE_FILE).read_bytes() == b"export"


def test_an_export_without_a_chat_template_file_is_left_without_one_when_the_release_has_none(tmp_path):
    release = _directory(tmp_path / "release", {"tokenizer_config.json": b"{}"})
    export = _directory(tmp_path / "export", {})
    copy_release_files(export, release, ["tokenizer_config.json"])
    assert sorted(path.name for path in export.iterdir()) == ["tokenizer_config.json"]


def test_a_dangling_release_chat_template_is_refused_not_read_as_none_and_the_export_keeps_its_own(tmp_path):
    release = _directory(tmp_path / "release", {"tokenizer_config.json": b"release"})
    (release / CHAT_TEMPLATE_FILE).symlink_to(tmp_path / "blobs" / "gone")
    export = _directory(tmp_path / "export", {CHAT_TEMPLATE_FILE: b"export", "tokenizer_config.json": b"export"})
    with pytest.raises(FileNotFoundError, match="chat_template.jinja is a dangling link"):
        copy_release_files(export, release, ["tokenizer_config.json"])
    assert (export / CHAT_TEMPLATE_FILE).read_bytes() == b"export"
    assert (export / "tokenizer_config.json").read_bytes() == b"export"


@pytest.mark.parametrize("name", ["../escape.json", "sub/dir.json", "a,b.json", "", ".", ".."])
def test_a_reference_file_name_that_is_not_a_bare_file_name_is_refused_before_anything_is_written(tmp_path, name):
    release = _directory(tmp_path / "release", {"escape.json": b"release"})
    export = _directory(tmp_path / "export", {})
    assert not is_bare_file_name(name)
    with pytest.raises(ValueError, match="not a bare file name"):
        copy_release_files(export, release, ["escape.json", name])
    assert not (tmp_path / "escape.json").exists() and list(export.iterdir()) == []


def test_release_file_names_the_file_the_release_cannot_supply(tmp_path):
    release = _directory(tmp_path / "release", {"a.json": b"a"})
    assert release_file(release, "a.json") == release / "a.json"
    with pytest.raises(FileNotFoundError, match="release has no b.json"):
        release_file(release, "b.json")


def test_a_release_directory_resolves_to_itself_and_a_hub_id_to_its_snapshot(tmp_path, monkeypatch):
    import huggingface_hub

    directory = _directory(tmp_path / "release", {})
    snapshot = _directory(tmp_path / "snapshot", {})
    asked = []

    # The network boundary: a Hub id resolves to a snapshot in the cache, else on the Hub.
    def fake_snapshot_download(repo_id, **kwargs):
        asked.append((repo_id, kwargs))
        return str(snapshot)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)
    assert resolve_release_dir(str(directory), ["a.json"]) == directory and asked == []
    assert resolve_release_dir("org/release", ["a.json", "b.json"]) == snapshot
    assert asked == [("org/release", {"allow_patterns": ["a.json", "b.json"]})]


def test_fixup_hf_output_copies_the_reference_files_after_every_other_fix(tmp_path):
    """The copy wins over the tokenizer class pin, the embedded template's mirror and the decoding defaults: without
    the reference files the same export comes out rewritten, with them its files are the release's bytes."""
    files = {"tokenizer_config.json": OLMO_TOKENIZER_CONFIG, "generation_config.json": OLMO_GENERATION_CONFIG}
    release = _directory(tmp_path / "release", files)
    plain = _directory(tmp_path / "plain", files)
    copied = _directory(tmp_path / "copied", files)
    for export, reference_files in ((plain, ()), (copied, list(files))):
        fixup_hf_output(
            export,
            str(release),
            reasoning=False,
            end_of_turn_id=OLMO_END_OF_TURN_ID,
            reference_files=reference_files,
        )
    assert json.loads((plain / "tokenizer_config.json").read_text())["tokenizer_class"] == "PreTrainedTokenizerFast"
    assert "temperature" not in json.loads((plain / "generation_config.json").read_text())
    assert (plain / CHAT_TEMPLATE_FILE).exists(), "mirrored from the embedded template"
    for name in files:
        assert (copied / name).read_bytes() == (release / name).read_bytes(), name
    assert not (copied / CHAT_TEMPLATE_FILE).exists(), "the release ships its template inside tokenizer_config.json"
