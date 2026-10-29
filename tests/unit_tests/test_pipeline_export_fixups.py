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

from pipeline.export.fixups.chat_template import set_enable_thinking_default, strip_think_stub
from pipeline.export.fixups.model_config import (
    fix_eos_token_ids,
    fix_hybrid_override_pattern,
    fix_layers_block_type,
    fix_num_hidden_layers,
    fix_vocab_size,
)
from pipeline.export.fixups.tokenizer import graft_release_template, pin_tokenizer_class, pin_tokenizer_config_file
from pipeline.hf_cache import cached_snapshots, hub_cache_root
from pipeline.nemotron import ENABLE_THINKING_OFF_LINE, ENABLE_THINKING_ON_LINE
from tests.unit_tests.hf_export_fixtures import (
    INPUTS,
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


def test_cached_snapshots_lists_largest_hash_first_and_none_when_uncached(hf_home):
    assert hub_cache_root() == hf_home / "hub"
    assert cached_snapshots(NANO) is None
    cache_repo(
        hf_home, NANO, [(SNAPSHOT_A, INPUTS / "release_nano"), (SNAPSHOT_B, INPUTS / "release_nano")], SNAPSHOT_A
    )
    assert [p.name for p in cached_snapshots(NANO)] == [SNAPSHOT_B, SNAPSHOT_A]


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


def _eos_export(tmp_path, generation_config: bool, chat_template: bool):
    export = tmp_path / "export"
    export.mkdir()
    (export / "config.json").write_text('{"eos_token_id": 2}')
    if generation_config:
        (export / "generation_config.json").write_text('{"eos_token_id": 2}')
    if chat_template:
        (export / "chat_template.jinja").write_text((INPUTS / "release_nano" / "chat_template.jinja").read_text())
    return export


def _eos(export, name):
    return json.loads((export / name).read_text())["eos_token_id"]


def test_im_end_joins_the_generation_eos_ids_when_a_chat_template_is_active(tmp_path):
    export = _eos_export(tmp_path, generation_config=True, chat_template=True)
    fix_eos_token_ids(export)
    assert (_eos(export, "generation_config.json"), _eos(export, "config.json")) == ([2, 11], 2)
    before = (export / "generation_config.json").read_bytes()
    fix_eos_token_ids(export)
    assert (export / "generation_config.json").read_bytes() == before


def test_without_generation_config_im_end_goes_into_config_json(tmp_path):
    export = _eos_export(tmp_path, generation_config=False, chat_template=True)
    fix_eos_token_ids(export)
    assert _eos(export, "config.json") == [2, 11]


def test_without_a_chat_template_the_eos_ids_stay(tmp_path):
    export = _eos_export(tmp_path, generation_config=True, chat_template=False)
    fix_eos_token_ids(export)
    assert (_eos(export, "generation_config.json"), _eos(export, "config.json")) == (2, 2)
