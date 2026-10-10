# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""The per-run token-masking decision, taken in setup from the config and checked against the built tokenizer.

``resolve_token_masking`` takes the ids from the ``token_masking:`` block alone; the tokenizer only validates them.
``refuse_tokenizer_declaration`` stops every run whose tokenizer still carries a ``loss_mask_token_ids`` key, wherever
the key is found: in the loaded tokenizer's ``init_kwargs``, in a local directory's ``tokenizer_config.json``, or in
the cached snapshot of a Hub id. Every case resolves against real tokenizers: tiny Hugging Face tokenizers saved to
disk with an added ``<marker>`` special token (carrying the key or not), and Megatron's ``NullTokenizer``, which cannot
carry it. The file also covers what setup does with the decision: the record behind the ``[token-masking]`` banner
(which must split back into its fields with ``shlex``) and the W&B summary, the refusal of forward steps that do not
apply token masking, the cross-rank agreement (two real gloo ranks), and ``resolve_for_run`` end to end on a
one-process gloo world.
"""

import functools
import json
import logging
import re
import shlex
import shutil
from dataclasses import replace
from pathlib import Path

import huggingface_hub.constants
import pytest
import torch
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from megatron.bridge.training import gpt_step, llava_step, vlm_step
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.forward_step_func_types import applies_token_masking, forward_step_name
from megatron.bridge.training.token_masking.config import TokenMaskingConfig, TokenMaskingError
from megatron.bridge.training.token_masking.hook import REPORT_KEYS
from megatron.bridge.training.token_masking.resolution import (
    DECLARATION_FIELD,
    ResolvedTokenMasking,
    agree_across_ranks,
    banner_fields,
    refuse_tokenizer_declaration,
    require_forward_step_applies_token_masking,
    resolve_for_run,
    resolve_token_masking,
    tokenizer_config_file,
    wandb_summary,
)
from megatron.bridge.training.tokenizers.tokenizer import build_tokenizer, find_hf_tokenizer
from megatron.bridge.training.utils.log_utils import log_node_banner
from tests.unit_tests.token_masking_fixtures import (
    EOS_ID,
    MARKER,
    MARKER_ID,
    TINY_VOCAB,
    build_tiny_hf_tokenizer,
    hf_tokenizer_config,
    masking,
    masking_with_null_tokenizer,
    measuring,
    null_tokenizer_config,
    remove_declaration,
    resolve,
    write_declaration,
)


CPU = torch.device("cpu")
HELLO_ID = TINY_VOCAB["hello"]
SECRET_ID = TINY_VOCAB["secret"]
PAD_ID = TINY_VOCAB["<pad>"]
NULL_VOCAB = 64
# An added token that is not special, as tokenizer.add_tokens registers a new vocabulary word.
ORDINARY_ADDED = "<ordinary>"
ORDINARY_ADDED_ID = MARKER_ID + 1
GPT_FORWARD_STEP = "megatron.bridge.training.gpt_step.forward_step"
RESOLUTION_LOGGER = "megatron.bridge.training.token_masking.resolution"


def _add_ordinary_token(directory: Path) -> Path:
    """Add ``ORDINARY_ADDED`` to a saved tokenizer as a non-special added token."""
    tokenizer = AutoTokenizer.from_pretrained(directory)
    tokenizer.add_tokens([ORDINARY_ADDED], special_tokens=False)
    assert tokenizer.convert_tokens_to_ids(ORDINARY_ADDED) == ORDINARY_ADDED_ID
    tokenizer.save_pretrained(directory)
    return directory


@pytest.fixture(scope="module")
def tokenizer_configs(tmp_path_factory):
    """Each named tokenizer's TokenizerConfig; none of them carries the ``loss_mask_token_ids`` key."""
    root = tmp_path_factory.mktemp("tokenizers")
    return {
        "plain": hf_tokenizer_config(build_tiny_hf_tokenizer(root / "plain")),
        "plain_copy": hf_tokenizer_config(build_tiny_hf_tokenizer(root / "plain_copy")),
        "ordinary_added": hf_tokenizer_config(_add_ordinary_token(build_tiny_hf_tokenizer(root / "ordinary"))),
        "null": null_tokenizer_config(NULL_VOCAB),
    }


@pytest.fixture(scope="module")
def tokenizers(tokenizer_configs):
    """The Megatron tokenizer ``build_tokenizer`` makes from each named TokenizerConfig."""
    return {name: build_tokenizer(config) for name, config in tokenizer_configs.items()}


def _hf_built(directory: Path):
    """The Megatron tokenizer built from a saved Hugging Face tokenizer directory."""
    return build_tokenizer(hf_tokenizer_config(directory))


# ---------------------------------------------------------------------------------------------------------------------
# The tokenizer declaration is refused, for every run
# ---------------------------------------------------------------------------------------------------------------------


def _refuse(tokenizer, directory: Path | str) -> None:
    refuse_tokenizer_declaration(tokenizer, "HuggingFaceTokenizer", str(directory))


class TestRefuseTokenizerDeclaration:
    def test_a_tokenizer_without_the_key_passes(self, tokenizers, tokenizer_configs):
        _refuse(tokenizers["plain"], tokenizer_configs["plain"].tokenizer_model)

    @pytest.mark.parametrize(
        "value", [[MARKER_ID], [MARKER_ID, SECRET_ID], [], None], ids=["marker", "two-ids", "empty", "null"]
    )
    def test_the_key_is_refused_whatever_its_value(self, tmp_path, value):
        """The key's presence is refused: ``[]`` and null too, which once meant masking nothing."""
        directory = build_tiny_hf_tokenizer(tmp_path / "tok")
        write_declaration(directory, value)
        tokenizer = _hf_built(directory)
        with pytest.raises(TokenMaskingError) as raised:
            _refuse(tokenizer, directory)
        assert str(raised.value) == (
            f"tokenizer {directory} carries {DECLARATION_FIELD} (in the loaded tokenizer's init_kwargs and "
            f"{directory / 'tokenizer_config.json'}), which no longer decides anything: token masking is configured "
            "only in the training config. Use a tokenizer without the key and put the ids in token_masking: "
            "{enabled: true, token_ids: [...]}."
        )

    def test_the_key_only_in_the_loaded_tokenizer_is_refused(self, tmp_path):
        """A snapshot edited after loading must not hide the key the run actually loaded."""
        directory = build_tiny_hf_tokenizer(tmp_path / "tok", [MARKER_ID])
        tokenizer = _hf_built(directory)
        remove_declaration(directory)
        with pytest.raises(TokenMaskingError, match="in the loaded tokenizer's init_kwargs\\)") as raised:
            _refuse(tokenizer, directory)
        assert "tokenizer_config.json" not in str(raised.value)

    def test_the_key_only_in_the_local_file_is_refused(self, tmp_path):
        """A transformers version that stops carrying unknown keys into init_kwargs must not hide the key."""
        directory = build_tiny_hf_tokenizer(tmp_path / "tok")
        tokenizer = _hf_built(directory)
        assert DECLARATION_FIELD not in find_hf_tokenizer(tokenizer).init_kwargs
        write_declaration(directory, [MARKER_ID])
        with pytest.raises(TokenMaskingError) as raised:
            _refuse(tokenizer, directory)
        assert f"(in {directory / 'tokenizer_config.json'})" in str(raised.value)
        assert "init_kwargs" not in str(raised.value)

    def test_the_message_names_the_configured_tokenizer(self, tmp_path):
        directory = build_tiny_hf_tokenizer(tmp_path / "tok", [MARKER_ID])
        with pytest.raises(TokenMaskingError, match="^tokenizer org/the-configured-name carries"):
            refuse_tokenizer_declaration(_hf_built(directory), "SFTTokenizer", "org/the-configured-name")

    def test_a_tokenizer_type_not_backed_by_hugging_face_is_not_checked(self, tokenizers):
        refuse_tokenizer_declaration(tokenizers["null"], "NullTokenizer", None)

    def test_a_hugging_face_type_whose_wrapper_holds_no_hugging_face_tokenizer_raises(self, tokenizers):
        with pytest.raises(TokenMaskingError, match="wraps a Hugging Face tokenizer, but none was found"):
            refuse_tokenizer_declaration(tokenizers["null"], "HuggingFaceTokenizer", None)


HUB_REPO_ID = "org/tiny"
HUB_REVISION = "0123456789abcdef0123456789abcdef01234567"


class TestTheHubCache:
    """A tokenizer named by a Hub id, as every campaign config names one, is checked against its cached snapshot.

    The cache is a real Hugging Face cache layout under tmp_path (``refs/main`` naming a snapshot), which
    ``try_to_load_from_cache`` reads as it reads the shared cache. The working directory is tmp_path, where the Hub id
    is not also a local directory, so the file is looked up as a Hub id.
    """

    @pytest.fixture
    def hub_repo(self, tmp_path, monkeypatch) -> Path:
        """The cache directory of ``org/tiny``, with an empty snapshot that ``refs/main`` names."""
        repo = tmp_path / "hub" / "models--org--tiny"
        (repo / "refs").mkdir(parents=True)
        (repo / "refs" / "main").write_text(HUB_REVISION)
        (repo / "snapshots" / HUB_REVISION).mkdir(parents=True)
        monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_CACHE", str(tmp_path / "hub"))
        monkeypatch.chdir(tmp_path)
        return repo

    @staticmethod
    def loaded_from_the_hub(tmp_path: Path):
        """A built tokenizer without the key, named by the Hub id as ``from_pretrained`` names a Hub download.

        Unit tests cannot download, so the tokenizer is loaded from a local copy and given the Hub id it would have.
        """
        tokenizer = _hf_built(build_tiny_hf_tokenizer(tmp_path / "loaded"))
        find_hf_tokenizer(tokenizer).name_or_path = HUB_REPO_ID
        return tokenizer

    @staticmethod
    def cache_the_loaded_config(tmp_path: Path, hub_repo: Path) -> Path:
        """Copy the loaded tokenizer's ``tokenizer_config.json`` into the cached snapshot; return the snapshot."""
        snapshot = hub_repo / "snapshots" / HUB_REVISION
        shutil.copy(tmp_path / "loaded" / "tokenizer_config.json", snapshot / "tokenizer_config.json")
        return snapshot

    @staticmethod
    def record_as_absent(hub_repo: Path) -> None:
        """Write the marker by which the cache records that the repository has no ``tokenizer_config.json``."""
        no_exist = hub_repo / ".no_exist" / HUB_REVISION
        no_exist.mkdir(parents=True)
        (no_exist / "tokenizer_config.json").write_text("")

    def test_the_cached_file_is_found(self, tmp_path, hub_repo):
        self.loaded_from_the_hub(tmp_path)
        snapshot = self.cache_the_loaded_config(tmp_path, hub_repo)
        assert tokenizer_config_file(HUB_REPO_ID) == snapshot / "tokenizer_config.json"

    def test_a_cached_snapshot_without_the_key_passes(self, tmp_path, hub_repo):
        tokenizer = self.loaded_from_the_hub(tmp_path)
        self.cache_the_loaded_config(tmp_path, hub_repo)
        _refuse(tokenizer, HUB_REPO_ID)

    def test_the_key_only_in_the_cached_snapshot_is_refused(self, tmp_path, hub_repo):
        tokenizer = self.loaded_from_the_hub(tmp_path)
        snapshot = self.cache_the_loaded_config(tmp_path, hub_repo)
        write_declaration(snapshot, [MARKER_ID])
        with pytest.raises(TokenMaskingError) as raised:
            _refuse(tokenizer, HUB_REPO_ID)
        assert str(raised.value).startswith(
            f"tokenizer {HUB_REPO_ID} carries {DECLARATION_FIELD} (in {snapshot / 'tokenizer_config.json'})"
        )

    def test_a_file_the_cache_records_as_absent_passes(self, tmp_path, hub_repo):
        tokenizer = self.loaded_from_the_hub(tmp_path)
        self.record_as_absent(hub_repo)
        assert tokenizer_config_file(HUB_REPO_ID) is None
        _refuse(tokenizer, HUB_REPO_ID)

    def test_a_file_missing_from_the_cache_raises(self, tmp_path, hub_repo):
        """Neither the file nor an absence marker: whether it carries the key cannot be checked, so setup stops."""
        tokenizer = self.loaded_from_the_hub(tmp_path)
        with pytest.raises(TokenMaskingError, match="local Hugging Face cache") as raised:
            _refuse(tokenizer, HUB_REPO_ID)
        assert f"cannot find the tokenizer_config.json of {HUB_REPO_ID}" in str(raised.value)


class TestTokenizerConfigFile:
    def test_a_local_directory_names_its_own_file(self, tmp_path):
        directory = build_tiny_hf_tokenizer(tmp_path / "tok")
        assert tokenizer_config_file(str(directory)) == directory / "tokenizer_config.json"

    def test_a_local_directory_without_the_file_gives_none(self, tmp_path):
        assert tokenizer_config_file(str(tmp_path)) is None

    @pytest.mark.parametrize("name", ["{tmp}/not_built", "./not_built", "../not_built", "~/not_built", "a/b/c"])
    def test_a_path_that_is_no_directory_raises_the_documented_error(self, tmp_path, monkeypatch, name):
        """A local tokenizer directory not built yet (as the E2E test's is, before its build) is named as such, never
        looked up as a Hub id, which huggingface_hub would refuse with its own validation error."""
        monkeypatch.chdir(tmp_path)
        name = name.format(tmp=tmp_path)
        with pytest.raises(TokenMaskingError, match=f"^{re.escape(f'tokenizer directory {name} does not exist')}$"):
            tokenizer_config_file(name)

    def test_a_file_named_as_the_directory_raises(self, tmp_path):
        (tmp_path / "tokenizer").write_text("")
        with pytest.raises(TokenMaskingError, match="is not a directory$"):
            tokenizer_config_file(str(tmp_path / "tokenizer"))

    @pytest.mark.parametrize("name", ["org/tiny", "gpt2"])
    def test_a_hub_id_is_still_looked_up_in_the_cache(self, tmp_path, monkeypatch, name):
        monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_CACHE", str(tmp_path / "hub"))
        monkeypatch.chdir(tmp_path)
        with pytest.raises(TokenMaskingError, match=f"cannot find the tokenizer_config.json of {name} in the local"):
            tokenizer_config_file(name)


# ---------------------------------------------------------------------------------------------------------------------
# resolve_token_masking
# ---------------------------------------------------------------------------------------------------------------------


def _resolve(tokenizer_configs, name, token_masking, device=CPU) -> ResolvedTokenMasking:
    return resolve(token_masking, tokenizer_configs[name], device)


class TestDecision:
    @pytest.mark.parametrize("name", ["plain", "null"])
    def test_an_omitted_block_masks_and_measures_nothing(self, tokenizer_configs, name):
        resolved = _resolve(tokenizer_configs, name, TokenMaskingConfig())
        assert (resolved.enabled, resolved.token_ids, resolved.measured_token_ids) == (False, (), ())
        assert resolved.token_strings == () and resolved.ids_tensor is None

    def test_an_enabled_block_masks_and_measures_its_ids(self, tokenizer_configs):
        resolved = _resolve(tokenizer_configs, "plain", masking([MARKER_ID]))
        assert resolved.enabled is True
        assert resolved.token_ids == resolved.measured_token_ids == (MARKER_ID,)
        assert resolved.token_strings == (MARKER,)
        assert resolved.ids_tensor.tolist() == [MARKER_ID]

    def test_a_measure_only_block_masks_nothing_and_measures_its_ids(self, tokenizer_configs):
        resolved = _resolve(tokenizer_configs, "plain", measuring([MARKER_ID]))
        assert resolved.enabled is False and resolved.token_ids == ()
        assert resolved.measured_token_ids == (MARKER_ID,)
        assert resolved.ids_tensor.tolist() == [MARKER_ID]

    @pytest.mark.parametrize("block", [masking([NULL_VOCAB - 1]), measuring([0])], ids=["masking", "measuring"])
    def test_a_tokenizer_not_backed_by_hugging_face_is_checked_for_range_only(self, tokenizer_configs, block):
        resolved = _resolve(tokenizer_configs, "null", block)
        assert resolved.measured_token_ids == tuple(block.measured_token_ids)


ID_KEYS = {
    "masking": (masking, "token_masking.token_ids"),
    "measuring": (measuring, "token_masking.masked_validation.token_ids"),
}


class TestTheTokenizerValidatesTheIds:
    @pytest.mark.parametrize(("make", "key"), ID_KEYS.values(), ids=ID_KEYS.keys())
    @pytest.mark.parametrize(
        ("name", "token_ids", "unregistered"),
        [
            pytest.param("plain", [HELLO_ID], [HELLO_ID], id="vocabulary-word"),
            pytest.param("plain", [MARKER_ID, HELLO_ID], [HELLO_ID], id="marker-plus-vocabulary-word"),
            pytest.param("ordinary_added", [ORDINARY_ADDED_ID], [ORDINARY_ADDED_ID], id="added-but-not-special"),
        ],
    )
    def test_an_ordinary_token_raises_naming_the_field(
        self, tokenizer_configs, make, key, name, token_ids, unregistered
    ):
        with pytest.raises(TokenMaskingError) as raised:
            _resolve(tokenizer_configs, name, make(token_ids))
        assert str(raised.value).startswith(f"{key} {token_ids}: {unregistered} are not added special tokens of ")
        assert "an ordinary vocabulary id is almost always a typo" in str(raised.value)

    @pytest.mark.parametrize(("make", "key"), ID_KEYS.values(), ids=ID_KEYS.keys())
    @pytest.mark.parametrize(
        "delimiter",
        [pytest.param(EOS_ID, id="eos"), pytest.param(PAD_ID, id="pad"), pytest.param(TINY_VOCAB["<unk>"], id="unk")],
    )
    def test_a_structural_token_raises(self, tokenizer_configs, make, key, delimiter):
        with pytest.raises(TokenMaskingError) as raised:
            _resolve(tokenizer_configs, "plain", make([delimiter]))
        assert str(raised.value) == (
            f"{key} [{delimiter}] includes [{delimiter}], the tokenizer's eos/bos/pad/unk/eod token"
        )

    @pytest.mark.parametrize(("make", "key"), ID_KEYS.values(), ids=ID_KEYS.keys())
    @pytest.mark.parametrize(
        ("name", "out_of_vocab", "size"),
        [pytest.param("null", NULL_VOCAB, NULL_VOCAB, id="null"), pytest.param("plain", 99, MARKER_ID + 1, id="hf")],
    )
    def test_ids_outside_the_vocabulary_raise(self, tokenizer_configs, make, key, name, out_of_vocab, size):
        with pytest.raises(TokenMaskingError) as raised:
            _resolve(tokenizer_configs, name, make([out_of_vocab]))
        assert str(raised.value).startswith(f"{key}: [{out_of_vocab}] are outside the vocabulary of ")
        assert f"(size {size}); they can never occur as targets" in str(raised.value)


class TestTokenStringsAndIdsTensor:
    @pytest.mark.parametrize(
        ("name", "block", "strings"),
        [
            pytest.param("plain", masking([MARKER_ID]), (MARKER,), id="marker"),
            pytest.param("null", masking([5]), ("5",), id="null-tokenizer"),
            pytest.param("null", measuring([9, 5]), ("9", "5"), id="in-measured-order"),
        ],
    )
    def test_token_strings_decode_the_measured_ids(self, tokenizer_configs, name, block, strings):
        assert _resolve(tokenizer_configs, name, block).token_strings == strings

    @pytest.mark.parametrize("device", [torch.device("cpu"), torch.device("meta")], ids=["cpu", "meta"])
    def test_ids_tensor_holds_the_measured_ids_as_long_on_the_given_device(self, tokenizer_configs, device):
        resolved = _resolve(tokenizer_configs, "null", measuring([7, 4]), device=device)
        assert resolved.ids_tensor.dtype == torch.long
        assert resolved.ids_tensor.device.type == device.type
        assert resolved.ids_tensor.shape == (2,)
        if device.type == "cpu":
            assert resolved.ids_tensor.tolist() == [7, 4]


class TestFindHfTokenizer:
    def test_walks_the_megatron_wrapper_to_the_transformers_tokenizer(self, tokenizers, tokenizer_configs):
        found = find_hf_tokenizer(tokenizers["plain"])
        assert isinstance(found, PreTrainedTokenizerBase)
        assert found is tokenizers["plain"]._tokenizer.tokenizer
        assert found.name_or_path == tokenizer_configs["plain"].tokenizer_model
        assert find_hf_tokenizer(found) is found

    def test_a_tokenizer_without_one_gives_none(self, tokenizers):
        assert find_hf_tokenizer(tokenizers["null"]) is None


# ---------------------------------------------------------------------------------------------------------------------
# Record, agreement key, banner, W&B summary
# ---------------------------------------------------------------------------------------------------------------------


class TestRecordAndAgreementKey:
    def test_record_holds_the_decision_as_plain_values(self, tokenizer_configs):
        resolved = _resolve(tokenizer_configs, "plain", masking([MARKER_ID]))
        record = resolved.record()
        assert record == {
            "enabled": True,
            "token_ids": [MARKER_ID],
            "measured_token_ids": [MARKER_ID],
            "tokens": [MARKER],
            "tokenizer": tokenizer_configs["plain"].tokenizer_model,
        }
        # A tuple would survive this round trip as a list and compare unequal: the record must be plain lists, as
        # the W&B summary stores it.
        assert json.loads(json.dumps(record)) == record

    def test_record_of_a_measure_only_run(self, tokenizer_configs):
        record = _resolve(tokenizer_configs, "null", measuring([5])).record()
        assert record == {
            "enabled": False,
            "token_ids": [],
            "measured_token_ids": [5],
            "tokens": ["5"],
            "tokenizer": None,
        }

    def test_a_tokenizer_named_by_a_path_object_is_recorded_as_text(self, tokenizer_configs, tokenizers):
        config = tokenizer_configs["plain"]
        resolved = resolve_token_masking(
            masking([MARKER_ID]), tokenizers["plain"], config.tokenizer_type, Path(config.tokenizer_model), CPU
        )
        assert resolved.record()["tokenizer"] == config.tokenizer_model

    def test_agreement_key_ignores_where_the_tokenizer_files_live(self, tokenizer_configs):
        here = _resolve(tokenizer_configs, "plain", masking([MARKER_ID]))
        there = _resolve(tokenizer_configs, "plain_copy", masking([MARKER_ID]))
        assert here.tokenizer_model != there.tokenizer_model
        assert here.agreement_key() == there.agreement_key() == (True, (MARKER_ID,), (MARKER_ID,))
        assert hash(here.agreement_key()) == hash(there.agreement_key())

    @pytest.mark.parametrize(
        "other",
        [
            pytest.param(TokenMaskingConfig(), id="omitted"),
            pytest.param(measuring([MARKER_ID]), id="measure-only-same-ids"),
            pytest.param(masking([MARKER_ID, 0]), id="other-ids"),
        ],
    )
    def test_agreement_key_separates_decisions_that_change_metrics_or_checks(self, tokenizer_configs, other):
        enabled = _resolve(tokenizer_configs, "null", masking([MARKER_ID]))
        assert _resolve(tokenizer_configs, "null", other).agreement_key() != enabled.agreement_key()


def _parse_banner(message: str, tag: str) -> tuple[list[str], dict[str, str]]:
    """Split a banner line with shlex into its leading words (tag, rank, host) and its ``key=value`` fields."""
    words = shlex.split(message)
    assert words[0] == f"[{tag}]"
    fields = dict(word.split("=", 1) for word in words[3:])
    return words[:3], fields


class TestBanner:
    AWKWARD_TOKENS = ("a b", "x=y]", "it's", "[", "⟦masked⟧", "")
    AWKWARD_TOKENIZER = "/tmp/dir with space/tok=1]"

    @pytest.fixture
    def awkward(self, tokenizer_configs):
        """A real decision whose token strings and tokenizer path hold spaces, '=', ']' and quotes."""
        resolved = _resolve(tokenizer_configs, "plain", masking([MARKER_ID]))
        return replace(resolved, token_strings=self.AWKWARD_TOKENS, tokenizer_model=self.AWKWARD_TOKENIZER)

    def test_every_field_survives_the_shell_quoted_line(self, awkward, caplog):
        fields = banner_fields(awkward, gpt_step.forward_step)
        caplog.set_level(logging.INFO, logger=RESOLUTION_LOGGER)
        log_node_banner(logging.getLogger(RESOLUTION_LOGGER), "token-masking", fields, rank=3, local_rank=0)
        (record,) = [record for record in caplog.records if "[token-masking]" in record.getMessage()]
        assert record.levelno == logging.INFO
        leading, parsed = _parse_banner(record.getMessage(), "token-masking")
        assert leading[1] == "rank=3" and leading[2].startswith("host=")
        assert parsed == dict(fields)
        assert list(parsed) == [
            "enabled",
            "token_ids",
            "measured_token_ids",
            "tokens",
            "tokenizer",
            "forward_step",
            "bridge_path",
        ]
        assert json.loads(parsed["tokens"]) == list(self.AWKWARD_TOKENS)
        assert parsed["tokenizer"] == self.AWKWARD_TOKENIZER
        assert json.loads(parsed["token_ids"]) == json.loads(parsed["measured_token_ids"]) == [MARKER_ID]
        assert parsed["forward_step"] == GPT_FORWARD_STEP
        assert parsed["enabled"] == "true"

    def test_only_the_first_process_of_a_node_logs(self, awkward, caplog):
        caplog.set_level(logging.INFO, logger=RESOLUTION_LOGGER)
        fields = banner_fields(awkward, None)
        log_node_banner(logging.getLogger(RESOLUTION_LOGGER), "token-masking", fields, rank=5, local_rank=1)
        assert not [record for record in caplog.records if "[token-masking]" in record.getMessage()]

    def test_an_idle_run_without_a_forward_step(self, tokenizer_configs):
        fields = dict(banner_fields(_resolve(tokenizer_configs, "null", TokenMaskingConfig()), None))
        assert (fields["enabled"], fields["token_ids"], fields["measured_token_ids"]) == ("false", "[]", "[]")
        assert (fields["tokenizer"], fields["forward_step"]) == ("none", "none")


class TestWandbSummary:
    def test_keys_mirror_the_record_under_the_namespace_and_never_shadow_a_metric(self, tokenizer_configs):
        resolved = _resolve(tokenizer_configs, "plain", masking([MARKER_ID]))
        summary = wandb_summary(resolved, gpt_step.forward_step)
        assert summary == {
            **{f"token_masking/{key}": value for key, value in resolved.record().items()},
            "token_masking/forward_step": GPT_FORWARD_STEP,
        }
        assert not set(summary) & set(REPORT_KEYS)
        assert json.loads(json.dumps(summary)) == summary

    def test_forward_step_is_none_when_setup_was_given_none(self, tokenizer_configs):
        summary = wandb_summary(_resolve(tokenizer_configs, "null", TokenMaskingConfig()), None)
        assert summary["token_masking/forward_step"] is None


# ---------------------------------------------------------------------------------------------------------------------
# The forward step must apply token masking whenever ids are measured
# ---------------------------------------------------------------------------------------------------------------------


class _DecoratedFunctor:
    """A stateful forward step whose ``__call__`` is marked as applying token masking."""

    @applies_token_masking
    def __call__(self, state, data_iterator, model, return_schedule_plan=False):
        return gpt_step.forward_step(state, data_iterator, model, return_schedule_plan)


class _PlainFunctor:
    """A stateful forward step that does not apply token masking."""

    def __call__(self, state, data_iterator, model, return_schedule_plan=False):
        return gpt_step.forward_step(state, data_iterator, model, return_schedule_plan)


BLOCKS_THAT_MEASURE = {
    "masking": (masking([5]), "this run masks token ids [5]"),
    "measuring": (measuring([5]), "this run measures (without masking) token ids [5]"),
}


class TestForwardStepCapability:
    @pytest.mark.parametrize(
        "forward_step",
        [
            pytest.param(gpt_step.forward_step, id="gpt_step.forward_step"),
            pytest.param(gpt_step.forward_step_modelopt, id="gpt_step.forward_step_modelopt"),
            pytest.param(functools.partial(gpt_step.forward_step, "state"), id="partial"),
            pytest.param(functools.partial(gpt_step.forward_step_modelopt, "state"), id="partial-of-modelopt"),
            pytest.param(_DecoratedFunctor(), id="functor"),
            pytest.param(functools.partial(_DecoratedFunctor(), "state"), id="partial-of-functor"),
        ],
    )
    @pytest.mark.parametrize("block", [masking([5]), measuring([5])], ids=["masking", "measuring"])
    def test_steps_that_apply_token_masking_pass(self, tokenizer_configs, forward_step, block):
        require_forward_step_applies_token_masking(forward_step, _resolve(tokenizer_configs, "null", block))

    @pytest.mark.parametrize(
        "forward_step",
        [
            pytest.param(vlm_step.forward_step, id="vlm_step.forward_step"),
            pytest.param(llava_step.forward_step, id="llava_step.forward_step"),
            pytest.param(functools.partial(vlm_step.forward_step, "state"), id="partial-of-vlm"),
            pytest.param(_PlainFunctor(), id="unmarked-functor"),
        ],
    )
    @pytest.mark.parametrize(("block", "action"), BLOCKS_THAT_MEASURE.values(), ids=BLOCKS_THAT_MEASURE.keys())
    def test_steps_that_do_not_apply_it_raise_naming_the_step(self, tokenizer_configs, forward_step, block, action):
        with pytest.raises(TokenMaskingError, match="does not apply token masking") as raised:
            require_forward_step_applies_token_masking(forward_step, _resolve(tokenizer_configs, "null", block))
        assert str(raised.value).startswith(f"{action}, but the forward step {forward_step_name(forward_step)} ")
        assert "a run on another forward step must omit the token_masking block" in str(raised.value)

    def test_the_named_steps_are_the_ones_their_modules_define(self):
        assert forward_step_name(vlm_step.forward_step) == "megatron.bridge.training.vlm_step.forward_step"
        assert forward_step_name(llava_step.forward_step) == "megatron.bridge.training.llava_step.forward_step"
        assert forward_step_name(_PlainFunctor()).endswith("._PlainFunctor")

    def test_no_forward_step_raises_when_ids_are_measured(self, tokenizer_configs):
        with pytest.raises(TokenMaskingError, match="no forward_step_func"):
            require_forward_step_applies_token_masking(None, _resolve(tokenizer_configs, "null", measuring([5])))

    @pytest.mark.parametrize(
        "forward_step", [None, vlm_step.forward_step, _PlainFunctor()], ids=["none", "vlm", "unmarked-functor"]
    )
    def test_an_idle_run_never_checks_the_step(self, tokenizer_configs, forward_step):
        idle = _resolve(tokenizer_configs, "plain", TokenMaskingConfig())
        assert idle.measured_token_ids == ()
        require_forward_step_applies_token_masking(forward_step, idle)


# ---------------------------------------------------------------------------------------------------------------------
# Agreement across ranks, and resolve_for_run
# ---------------------------------------------------------------------------------------------------------------------

# Each rank's masked token ids per scenario, resolved for real on a NullTokenizer of NULL_VOCAB; NULL_VOCAB itself is
# out of range, so that rank's resolution fails.
AGREEMENT_SCENARIOS = {
    "identical": ([5], [5]),
    "different": ([5], [6]),
    "one_rank_fails": ([5], [NULL_VOCAB]),
}


def _agree_on_two_ranks(rank: int, init_file: str, result_dir: str) -> None:
    """One of two gloo ranks: resolves each scenario's ids, runs the agreement, and writes what it raised."""
    torch.distributed.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=2)
    try:
        raised, causes = {}, {}
        for scenario, ids_per_rank in AGREEMENT_SCENARIOS.items():
            try:
                outcome = masking_with_null_tokenizer(ids_per_rank[rank], NULL_VOCAB)
            except TokenMaskingError as error:
                outcome = error
            try:
                agree_across_ranks(outcome, group=None)
                raised[scenario] = causes[scenario] = None
            except TokenMaskingError as error:
                raised[scenario] = str(error)
                causes[scenario] = None if error.__cause__ is None else str(error.__cause__)
        (Path(result_dir) / f"rank{rank}.json").write_text(json.dumps(raised))
        (Path(result_dir) / f"rank{rank}.causes.json").write_text(json.dumps(causes))
    finally:
        torch.distributed.destroy_process_group()


class TestAgreeAcrossRanks:
    @pytest.fixture(scope="class")
    def results(self, tmp_path_factory):
        results = tmp_path_factory.mktemp("agreement")
        torch.multiprocessing.spawn(_agree_on_two_ranks, args=(str(results / "rendezvous"), str(results)), nprocs=2)
        return results

    @pytest.fixture(scope="class")
    def raised(self, results):
        return [json.loads((results / f"rank{rank}.json").read_text()) for rank in (0, 1)]

    @pytest.fixture(scope="class")
    def causes(self, results):
        return [json.loads((results / f"rank{rank}.causes.json").read_text()) for rank in (0, 1)]

    def test_identical_decisions_pass_on_every_rank(self, raised):
        assert raised[0]["identical"] is None and raised[1]["identical"] is None

    def test_different_decisions_raise_the_same_message_on_every_rank(self, raised):
        message = raised[0]["different"]
        assert message is not None and raised[1]["different"] == message
        assert "ranks resolved different token-masking decisions" in message
        for rank, token_ids in enumerate(AGREEMENT_SCENARIOS["different"]):
            key = masking_with_null_tokenizer(token_ids, NULL_VOCAB).agreement_key()
            assert f"ranks [{rank}]: {key}" in message

    def test_a_failure_on_one_rank_raises_its_message_on_every_rank(self, raised):
        message = raised[0]["one_rank_fails"]
        assert message is not None and raised[1]["one_rank_fails"] == message
        assert "could not be resolved on ranks [1]" in message
        assert f"TokenMaskingError: token_masking.token_ids: [{NULL_VOCAB}] are outside the vocabulary" in message

    def test_the_failing_rank_keeps_its_own_error_as_the_cause(self, causes):
        """Its traceback then shows where resolution failed; the healthy rank has no local error to attach."""
        assert f"[{NULL_VOCAB}] are outside the vocabulary" in causes[1]["one_rank_fails"]
        assert causes[0]["one_rank_fails"] is None
        assert causes[0]["different"] is None and causes[1]["different"] is None


def _container(tokenizer_config, token_masking: TokenMaskingConfig) -> ConfigContainer:
    """A real ConfigContainer holding the two sections resolve_for_run reads.

    The sections a recipe builds are None: resolve_for_run never reads them, and building them needs a recipe.
    """
    return ConfigContainer(
        train=None,
        model=None,
        optimizer=None,
        scheduler=None,
        dataset=None,
        logger=None,
        checkpoint=None,
        tokenizer=tokenizer_config,
        token_masking=token_masking,
    )


class TestResolveForRun:
    @pytest.fixture(autouse=True)
    def first_process_of_its_node(self, monkeypatch, gloo_group_of_one):
        monkeypatch.setenv("LOCAL_RANK", "0")

    @staticmethod
    def banners(caplog) -> list[str]:
        return [record.getMessage() for record in caplog.records if "[token-masking]" in record.getMessage()]

    def test_resolves_the_decision_and_logs_one_banner(self, tokenizer_configs, tokenizers, caplog):
        caplog.set_level(logging.INFO, logger=RESOLUTION_LOGGER)
        cfg = _container(tokenizer_configs["plain"], masking([MARKER_ID]))
        resolved = resolve_for_run(cfg, tokenizers["plain"], gpt_step.forward_step, CPU)
        assert resolved.enabled is True and resolved.token_ids == (MARKER_ID,)
        assert cfg.token_masking == masking([MARKER_ID]), "the config states the decision; setup never rewrites it"
        (banner,) = self.banners(caplog)
        leading, fields = _parse_banner(banner, "token-masking")
        assert leading[1] == "rank=0"
        assert fields == dict(banner_fields(resolved, gpt_step.forward_step))
        assert json.loads(fields["tokens"]) == [MARKER]

    @pytest.mark.parametrize("block", [TokenMaskingConfig(), masking([MARKER_ID])], ids=["omitted", "masking"])
    def test_a_tokenizer_carrying_the_key_is_refused_for_every_run(self, tmp_path, block, caplog):
        caplog.set_level(logging.INFO, logger=RESOLUTION_LOGGER)
        directory = build_tiny_hf_tokenizer(tmp_path / "tok", [MARKER_ID])
        config = hf_tokenizer_config(directory)
        with pytest.raises(
            TokenMaskingError,
            match=rf"could not be resolved on ranks \[0\]: TokenMaskingError: tokenizer {directory} carries "
            f"{DECLARATION_FIELD}",
        ):
            resolve_for_run(_container(config, block), build_tokenizer(config), gpt_step.forward_step, CPU)
        assert self.banners(caplog) == []

    def test_an_unsupported_forward_step_raises_before_logging(self, tokenizer_configs, tokenizers, caplog):
        caplog.set_level(logging.INFO, logger=RESOLUTION_LOGGER)
        cfg = _container(tokenizer_configs["plain"], masking([MARKER_ID]))
        with pytest.raises(TokenMaskingError, match="vlm_step.forward_step does not apply token masking"):
            resolve_for_run(cfg, tokenizers["plain"], vlm_step.forward_step, CPU)
        assert self.banners(caplog) == []

    def test_a_resolution_error_is_raised_through_the_agreement(self, tokenizer_configs, tokenizers):
        cfg = _container(tokenizer_configs["plain"], measuring([HELLO_ID]))
        with pytest.raises(
            TokenMaskingError,
            match=r"could not be resolved on ranks \[0\]: TokenMaskingError: token_masking.masked_validation.token_ids "
            rf"\[{HELLO_ID}\]: \[{HELLO_ID}\] are not added special tokens",
        ):
            resolve_for_run(cfg, tokenizers["plain"], gpt_step.forward_step, CPU)
