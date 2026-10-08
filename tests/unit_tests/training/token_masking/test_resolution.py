# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""The per-run token-masking decision, taken in setup from the config and the tokenizer that was actually built.

``resolve_token_masking`` turns a ``token_masking:`` block, the legacy ``tokenizer.loss_mask_token_ids`` field and the
built tokenizer into the ids a run masks and the ids it only counts. Every case here resolves against real
tokenizers: tiny Hugging Face tokenizers saved to disk with an added ``<marker>`` special token (declaring it in
``tokenizer_config.json`` or not, as the production MQ tokenizers do), and Megatron's ``NullTokenizer``, which can
declare nothing. The file also covers what setup does with the decision: the declaration reader and its cross-check
against the snapshot on disk (a local directory, or for a Hub id the local Hugging Face cache), the record written
into the config and W&B summary, the ``[token-masking]`` banner (which must split back into its fields with
``shlex``), the refusal of forward steps that do not apply token masking, the cross-rank agreement (two real gloo
ranks), and ``resolve_for_run`` end to end on a one-process gloo world.
"""

import functools
import json
import logging
import shlex
import shutil
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import huggingface_hub.constants
import pytest
import torch
from transformers import PreTrainedTokenizerBase

from megatron.bridge.training import gpt_step, llava_step, vlm_step
from megatron.bridge.training.forward_step_func_types import applies_token_masking, forward_step_name
from megatron.bridge.training.token_masking.config import (
    TokenMaskingConfig,
    TokenMaskingError,
    validate_token_masking,
)
from megatron.bridge.training.token_masking.hook import REPORT_KEYS
from megatron.bridge.training.token_masking.resolution import (
    DECLARATION_FIELD,
    ResolvedTokenMasking,
    _snapshot_tokenizer_config,
    agree_across_ranks,
    banner_fields,
    declared_token_ids,
    require_forward_step_applies_token_masking,
    resolve_for_run,
    resolve_token_masking,
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
    masking_with_null_tokenizer,
    null_tokenizer_config,
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

# The tokenizers every resolution case picks from, by what their tokenizer_config.json declares.
DECLARATIONS = {
    "declares_marker": [MARKER_ID],
    "declares_marker_copy": [MARKER_ID],
    "declares_marker_and_secret": [MARKER_ID, SECRET_ID],
    "declares_empty": [],
    "declares_out_of_vocab": [99],
    "declares_nothing": None,
}


def _add_ordinary_token(directory: Path) -> Path:
    """Add ``ORDINARY_ADDED`` to a saved tokenizer as a non-special added token."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(directory)
    tokenizer.add_tokens([ORDINARY_ADDED], special_tokens=False)
    assert tokenizer.convert_tokens_to_ids(ORDINARY_ADDED) == ORDINARY_ADDED_ID
    tokenizer.save_pretrained(directory)
    return directory


@pytest.fixture(scope="module")
def tokenizers(tmp_path_factory):
    """Each named tokenizer as (its TokenizerConfig, the Megatron tokenizer build_tokenizer makes from it)."""
    root = tmp_path_factory.mktemp("tokenizers")
    configs = {
        name: hf_tokenizer_config(build_tiny_hf_tokenizer(root / name, declared))
        for name, declared in DECLARATIONS.items()
    }
    configs["declares_nothing_with_ordinary_added"] = hf_tokenizer_config(
        _add_ordinary_token(build_tiny_hf_tokenizer(root / "ordinary_added", None))
    )
    configs["null"] = null_tokenizer_config(NULL_VOCAB)
    return {name: (config, build_tokenizer(config)) for name, config in configs.items()}


def resolve(tokenizers, name, *, legacy=None, device=CPU, **block) -> ResolvedTokenMasking:
    """Validate a ``token_masking`` block as ``ConfigContainer.validate`` does, then resolve it against a tokenizer."""
    token_masking = TokenMaskingConfig(**block)
    validate_token_masking(token_masking, legacy)
    tokenizer_config, tokenizer = tokenizers[name]
    return resolve_token_masking(
        token_masking, legacy, tokenizer, tokenizer_config.tokenizer_type, tokenizer_config.tokenizer_model, device
    )


class TestUnstatedMode:
    """Configs that predate the block: the legacy field decides, else the declaration, and nothing is enforced."""

    @pytest.mark.parametrize(
        "name, legacy, token_ids, observed, source, declared",
        [
            pytest.param(
                "declares_marker", None, (MARKER_ID,), (MARKER_ID,), "tokenizer", (MARKER_ID,), id="declaration-masks"
            ),
            pytest.param(
                "declares_marker",
                [],
                (),
                (MARKER_ID,),
                "legacy_tokenizer_field",
                (MARKER_ID,),
                id="legacy-empty-turns-off-but-observes-declaration",
            ),
            pytest.param(
                "declares_nothing",
                [MARKER_ID],
                (MARKER_ID,),
                (MARKER_ID,),
                "legacy_tokenizer_field",
                None,
                id="legacy-ids-mask-without-declaration",
            ),
            pytest.param("declares_nothing", None, (), (), "none", None, id="plain-tokenizer-observes-nothing"),
            pytest.param("declares_empty", None, (), (), "none", (), id="empty-declaration-observes-nothing"),
            pytest.param("null", None, (), (), "none", None, id="null-tokenizer-observes-nothing"),
        ],
    )
    def test_decision(self, tokenizers, name, legacy, token_ids, observed, source, declared):
        resolved = resolve(tokenizers, name, legacy=legacy)
        assert resolved.token_ids == token_ids
        assert resolved.observed_token_ids == observed
        assert resolved.enabled is bool(token_ids)
        assert resolved.source == source
        assert resolved.tokenizer_declared_token_ids == declared
        assert resolved.mode is None and resolved.enforced is False
        assert resolved.require_masked_targets is False
        assert resolved.require_masked_targets_within_iterations is None
        if observed:
            assert resolved.ids_tensor.tolist() == list(observed)
        else:
            assert resolved.ids_tensor is None


class TestEnabledMode:
    def test_without_ids_masks_the_declaration_and_requires_masked_targets(self, tokenizers):
        resolved = resolve(tokenizers, "declares_marker", mode="enabled")
        assert resolved.token_ids == resolved.observed_token_ids == (MARKER_ID,)
        assert resolved.source == "tokenizer"
        assert resolved.enforced is True and resolved.enabled is True
        assert resolved.require_masked_targets is True
        assert resolved.require_masked_targets_within_iterations == 10

    @pytest.mark.parametrize(
        "name, token_ids",
        [
            pytest.param("declares_marker", [MARKER_ID], id="same-id"),
            pytest.param("declares_marker_and_secret", [SECRET_ID, MARKER_ID], id="same-set-other-order"),
        ],
    )
    def test_explicit_ids_equal_to_the_declaration_are_accepted(self, tokenizers, name, token_ids):
        resolved = resolve(tokenizers, name, mode="enabled", token_ids=token_ids)
        assert resolved.token_ids == tuple(token_ids)
        assert resolved.source == "config"

    @pytest.mark.parametrize(
        "name, token_ids, declared",
        [
            pytest.param("declares_marker", [SECRET_ID], [MARKER_ID], id="different"),
            pytest.param("declares_marker_and_secret", [MARKER_ID], [MARKER_ID, SECRET_ID], id="subset"),
        ],
    )
    def test_explicit_ids_that_differ_from_the_declaration_raise_naming_both(
        self, tokenizers, name, token_ids, declared
    ):
        with pytest.raises(TokenMaskingError, match="differs from the ids") as raised:
            resolve(tokenizers, name, mode="enabled", token_ids=token_ids)
        assert str(token_ids) in str(raised.value) and str(declared) in str(raised.value)

    @pytest.mark.parametrize(
        "block, legacy, source",
        [
            pytest.param({"token_ids": [MARKER_ID]}, None, "config", id="token_ids"),
            pytest.param({}, [MARKER_ID], "legacy_tokenizer_field", id="legacy-field"),
        ],
    )
    def test_explicit_special_token_on_a_tokenizer_that_declares_nothing(self, tokenizers, block, legacy, source):
        resolved = resolve(tokenizers, "declares_nothing", legacy=legacy, mode="enabled", **block)
        assert resolved.token_ids == (MARKER_ID,) and resolved.enforced is True
        assert resolved.tokenizer_declared_token_ids is None
        assert resolved.source == source

    @pytest.mark.parametrize(
        "name, token_ids, unregistered",
        [
            pytest.param("declares_nothing", [HELLO_ID], [HELLO_ID], id="vocabulary-word"),
            pytest.param("declares_nothing", [MARKER_ID, HELLO_ID], [HELLO_ID], id="marker-plus-vocabulary-word"),
            pytest.param(
                "declares_nothing_with_ordinary_added",
                [ORDINARY_ADDED_ID],
                [ORDINARY_ADDED_ID],
                id="added-but-not-special",
            ),
        ],
    )
    def test_explicit_ordinary_token_raises(self, tokenizers, name, token_ids, unregistered):
        with pytest.raises(TokenMaskingError, match="not added special tokens") as raised:
            resolve(tokenizers, name, mode="enabled", token_ids=token_ids)
        assert f"{unregistered} are not added special tokens" in str(raised.value)

    @pytest.mark.parametrize(
        "delimiter",
        [pytest.param(EOS_ID, id="eos"), pytest.param(PAD_ID, id="pad"), pytest.param(TINY_VOCAB["<unk>"], id="unk")],
    )
    def test_explicit_structural_token_raises(self, tokenizers, delimiter):
        with pytest.raises(TokenMaskingError, match=rf"includes \[{delimiter}\], the tokenizer's eos/bos/pad/unk/eod"):
            resolve(tokenizers, "declares_nothing", mode="enabled", token_ids=[delimiter])

    def test_legacy_ids_that_differ_from_the_declaration_name_the_legacy_field(self, tokenizers):
        with pytest.raises(TokenMaskingError, match="differs from the ids") as raised:
            resolve(tokenizers, "declares_marker", legacy=[SECRET_ID], mode="enabled")
        assert f"tokenizer.loss_mask_token_ids [{SECRET_ID}] differs" in str(raised.value)
        assert "omit tokenizer.loss_mask_token_ids to use the declaration" in str(raised.value)
        assert "token_masking.token_ids" not in str(raised.value)

    def test_legacy_ordinary_ids_name_the_legacy_field(self, tokenizers):
        with pytest.raises(TokenMaskingError, match="not added special tokens") as raised:
            resolve(tokenizers, "declares_nothing", legacy=[SECRET_ID], mode="enabled")
        assert str(raised.value).startswith(f"tokenizer.loss_mask_token_ids [{SECRET_ID}]")

    @pytest.mark.parametrize("name", ["declares_nothing", "declares_empty", "null"])
    def test_no_ids_anywhere_raises_with_the_fix(self, tokenizers, name):
        with pytest.raises(TokenMaskingError, match="no token ids are given") as raised:
            resolve(tokenizers, name, mode="enabled")
        assert "Use a tokenizer that declares the marker or set token_masking.token_ids" in str(raised.value)

    def test_a_tokenizer_that_cannot_declare_is_checked_for_range_only(self, tokenizers):
        resolved = resolve(tokenizers, "null", mode="enabled", token_ids=[NULL_VOCAB - 1])
        assert resolved.token_ids == (NULL_VOCAB - 1,)
        assert resolved.tokenizer_declared_token_ids is None

    def test_require_values_stated_in_the_block_are_carried(self, tokenizers):
        resolved = resolve(
            tokenizers,
            "declares_marker",
            mode="enabled",
            require_masked_targets=False,
            require_masked_targets_within_iterations=3,
        )
        assert resolved.require_masked_targets is False
        assert resolved.require_masked_targets_within_iterations == 3


class TestDisabledMode:
    @pytest.mark.parametrize(
        "name, token_ids, observed",
        [
            pytest.param("declares_nothing", None, (), id="nothing-to-observe"),
            pytest.param("declares_marker", None, (MARKER_ID,), id="observes-the-declaration"),
            pytest.param("declares_marker", [HELLO_ID], (HELLO_ID,), id="token_ids-replace-the-declaration"),
            pytest.param("null", [5], (5,), id="token_ids-on-null-tokenizer"),
        ],
    )
    def test_never_masks_and_observes(self, tokenizers, name, token_ids, observed):
        resolved = resolve(tokenizers, name, mode="disabled", token_ids=token_ids)
        assert resolved.token_ids == () and resolved.enabled is False
        assert resolved.observed_token_ids == observed
        assert resolved.enforced is False and resolved.require_masked_targets is False
        assert resolved.require_masked_targets_within_iterations is None
        assert resolved.source == "config"


class TestVocabularyRange:
    @pytest.mark.parametrize(
        "block, legacy",
        [
            pytest.param({"mode": "enabled", "token_ids": [NULL_VOCAB]}, None, id="enabled"),
            pytest.param({"mode": "disabled", "token_ids": [NULL_VOCAB]}, None, id="disabled-observed"),
            pytest.param({}, [NULL_VOCAB], id="unstated-legacy"),
        ],
    )
    def test_ids_outside_the_vocabulary_raise(self, tokenizers, block, legacy):
        with pytest.raises(
            TokenMaskingError, match=rf"\[{NULL_VOCAB}\] are outside the vocabulary .* \(size {NULL_VOCAB}\)"
        ):
            resolve(tokenizers, "null", legacy=legacy, **block)

    def test_a_declaration_outside_the_vocabulary_raises(self, tokenizers):
        with pytest.raises(TokenMaskingError, match=r"\[99\] are outside the vocabulary"):
            resolve(tokenizers, "declares_out_of_vocab")


class TestTokenStringsAndIdsTensor:
    @pytest.mark.parametrize(
        "name, block, strings",
        [
            pytest.param("declares_marker", {"mode": "enabled"}, (MARKER,), id="marker"),
            pytest.param(
                "declares_marker",
                {"mode": "disabled", "token_ids": [HELLO_ID, MARKER_ID]},
                ("hello", MARKER),
                id="in-observed-order",
            ),
            pytest.param("null", {"mode": "enabled", "token_ids": [5]}, ("5",), id="null-tokenizer"),
        ],
    )
    def test_token_strings_decode_the_observed_ids(self, tokenizers, name, block, strings):
        assert resolve(tokenizers, name, **block).token_strings == strings

    @pytest.mark.parametrize("device", [torch.device("cpu"), torch.device("meta")], ids=["cpu", "meta"])
    def test_ids_tensor_holds_the_observed_ids_as_long_on_the_given_device(self, tokenizers, device):
        resolved = resolve(tokenizers, "declares_marker", device=device, mode="disabled", token_ids=[HELLO_ID, 4])
        assert resolved.ids_tensor.dtype == torch.long
        assert resolved.ids_tensor.device.type == device.type
        assert resolved.ids_tensor.shape == (2,)
        if device.type == "cpu":
            assert resolved.ids_tensor.tolist() == [HELLO_ID, 4]


class TestDeclaredTokenIds:
    @pytest.mark.parametrize(
        "declaration",
        [
            pytest.param(str(MARKER_ID), id="string"),
            pytest.param([True], id="bool"),
            pytest.param([float(MARKER_ID)], id="float"),
            pytest.param([[MARKER_ID]], id="nested"),
            pytest.param([-1], id="negative"),
            pytest.param([MARKER_ID, MARKER_ID], id="repeated"),
            pytest.param({"ids": [MARKER_ID]}, id="mapping"),
        ],
    )
    def test_malformed_declaration_raises(self, tmp_path, declaration):
        config = hf_tokenizer_config(build_tiny_hf_tokenizer(tmp_path / "tok", declaration))
        with pytest.raises(TokenMaskingError, match="tokenizer_config.json loss_mask_token_ids"):
            declared_token_ids(build_tokenizer(config), config.tokenizer_type)

    @pytest.mark.parametrize(
        "loaded, edited",
        [
            pytest.param([MARKER_ID], [SECRET_ID], id="changed"),
            pytest.param([MARKER_ID], None, id="removed"),
            pytest.param(None, [MARKER_ID], id="added"),
        ],
    )
    def test_a_snapshot_edited_after_loading_raises_naming_both(self, tmp_path, loaded, edited):
        directory = build_tiny_hf_tokenizer(tmp_path / "tok", loaded)
        config = hf_tokenizer_config(directory)
        tokenizer = build_tokenizer(config)
        if edited is None:
            on_disk = json.loads((directory / "tokenizer_config.json").read_text())
            del on_disk["loss_mask_token_ids"]
            (directory / "tokenizer_config.json").write_text(json.dumps(on_disk))
        else:
            write_declaration(directory, edited)
        with pytest.raises(TokenMaskingError, match="tokenizer_config.json declares") as raised:
            declared_token_ids(tokenizer, config.tokenizer_type)
        assert f"loss_mask_token_ids={edited!r}" in str(raised.value)
        assert f"loaded tokenizer carries {loaded!r}" in str(raised.value)

    @pytest.mark.parametrize(
        "name, declared",
        [
            pytest.param("declares_marker_and_secret", (MARKER_ID, SECRET_ID), id="declares"),
            pytest.param("declares_empty", (), id="declares-empty"),
            pytest.param("declares_nothing", None, id="declares-nothing"),
        ],
    )
    def test_reads_the_declaration_of_the_built_tokenizer(self, tokenizers, name, declared):
        config, tokenizer = tokenizers[name]
        assert declared_token_ids(tokenizer, config.tokenizer_type) == declared

    def test_a_tokenizer_type_that_cannot_declare_returns_none(self, tokenizers):
        config, tokenizer = tokenizers["null"]
        assert declared_token_ids(tokenizer, config.tokenizer_type) is None

    def test_a_hugging_face_type_whose_wrapper_holds_no_hugging_face_tokenizer_raises(self, tokenizers):
        _, null_tokenizer = tokenizers["null"]
        with pytest.raises(TokenMaskingError, match="none was found"):
            declared_token_ids(null_tokenizer, "HuggingFaceTokenizer")


HUB_REPO_ID = "org/tiny"
HUB_REVISION = "0123456789abcdef0123456789abcdef01234567"


class TestSnapshotTokenizerConfigFromHubCache:
    """A tokenizer named by a Hub id, as every campaign config names one, is cross-checked against its cached copy.

    The cache is a real Hugging Face cache layout under tmp_path (``refs/main`` naming a snapshot), which
    ``try_to_load_from_cache`` reads as it reads the shared cache. The working directory is tmp_path, where the Hub id
    is not also a local directory, so the declaration is looked up as a Hub id.
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
    def loaded_from_the_hub(tmp_path: Path, declared: list[int] | None):
        """A built tokenizer declaring ``declared``, named by the Hub id as ``from_pretrained`` names a Hub download."""
        tokenizer = build_tokenizer(hf_tokenizer_config(build_tiny_hf_tokenizer(tmp_path / "loaded", declared)))
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

    def test_the_declaration_is_read_from_the_cached_snapshot(self, tmp_path, hub_repo):
        tokenizer = self.loaded_from_the_hub(tmp_path, [MARKER_ID])
        self.cache_the_loaded_config(tmp_path, hub_repo)
        assert _snapshot_tokenizer_config(find_hf_tokenizer(tokenizer))[DECLARATION_FIELD] == [MARKER_ID]
        assert declared_token_ids(tokenizer, "HuggingFaceTokenizer") == (MARKER_ID,)

    def test_a_cached_snapshot_declaring_other_ids_raises_naming_both(self, tmp_path, hub_repo):
        tokenizer = self.loaded_from_the_hub(tmp_path, [MARKER_ID])
        write_declaration(self.cache_the_loaded_config(tmp_path, hub_repo), [SECRET_ID])
        with pytest.raises(TokenMaskingError, match="tokenizer_config.json declares") as raised:
            declared_token_ids(tokenizer, "HuggingFaceTokenizer")
        assert str(raised.value).startswith(f"{HUB_REPO_ID}:")
        assert f"{DECLARATION_FIELD}={[SECRET_ID]!r}" in str(raised.value)
        assert f"loaded tokenizer carries {[MARKER_ID]!r}" in str(raised.value)

    def test_a_file_the_cache_records_as_absent_declares_nothing(self, tmp_path, hub_repo):
        tokenizer = self.loaded_from_the_hub(tmp_path, None)
        self.record_as_absent(hub_repo)
        assert _snapshot_tokenizer_config(find_hf_tokenizer(tokenizer)) is None
        assert declared_token_ids(tokenizer, "HuggingFaceTokenizer") is None

    def test_a_declaration_the_cache_records_as_absent_raises(self, tmp_path, hub_repo):
        tokenizer = self.loaded_from_the_hub(tmp_path, [MARKER_ID])
        self.record_as_absent(hub_repo)
        with pytest.raises(TokenMaskingError, match=f"declares {DECLARATION_FIELD}=None but the loaded tokenizer"):
            declared_token_ids(tokenizer, "HuggingFaceTokenizer")

    @pytest.mark.parametrize("declared", [None, [MARKER_ID]], ids=["declares-nothing", "declares"])
    def test_a_file_missing_from_the_cache_raises(self, tmp_path, hub_repo, declared):
        """Neither the file nor an absence marker: what the tokenizer declares cannot be verified, so setup stops."""
        tokenizer = self.loaded_from_the_hub(tmp_path, declared)
        with pytest.raises(TokenMaskingError, match="local Hugging Face cache") as raised:
            declared_token_ids(tokenizer, "HuggingFaceTokenizer")
        assert HUB_REPO_ID in str(raised.value)


class TestFindHfTokenizer:
    def test_walks_the_megatron_wrapper_to_the_transformers_tokenizer(self, tokenizers):
        config, tokenizer = tokenizers["declares_marker"]
        found = find_hf_tokenizer(tokenizer)
        assert isinstance(found, PreTrainedTokenizerBase)
        assert found is tokenizer._tokenizer.tokenizer
        assert found.name_or_path == config.tokenizer_model
        assert find_hf_tokenizer(found) is found

    def test_a_tokenizer_without_one_gives_none(self, tokenizers):
        assert find_hf_tokenizer(tokenizers["null"][1]) is None


class TestRecordAndAgreementKey:
    def test_record_holds_the_decision_as_plain_values(self, tokenizers):
        resolved = resolve(tokenizers, "declares_marker", mode="enabled")
        record = resolved.record()
        assert record == {
            "mode": "enabled",
            "enforced": True,
            "enabled": True,
            "token_ids": [MARKER_ID],
            "observed_token_ids": [MARKER_ID],
            "tokens": [MARKER],
            "source": "tokenizer",
            "tokenizer": tokenizers["declares_marker"][0].tokenizer_model,
            "tokenizer_declared_token_ids": [MARKER_ID],
            "require_masked_targets": True,
            "require_masked_targets_within_iterations": 10,
        }
        # A tuple would survive this round trip as a list and compare unequal: the record must be plain lists, as
        # the config's run_config.yaml and the W&B summary store it.
        assert json.loads(json.dumps(record)) == record

    def test_record_of_an_unstated_run_without_declaration(self, tokenizers):
        record = resolve(tokenizers, "declares_nothing").record()
        assert record["mode"] == "unstated"
        assert record["tokenizer_declared_token_ids"] is None
        assert record["token_ids"] == record["observed_token_ids"] == record["tokens"] == []

    def test_agreement_key_ignores_where_the_tokenizer_files_live(self, tokenizers):
        here = resolve(tokenizers, "declares_marker", mode="enabled")
        there = resolve(tokenizers, "declares_marker_copy", mode="enabled")
        assert here.tokenizer_model != there.tokenizer_model
        assert here.agreement_key() == there.agreement_key()
        assert hash(here.agreement_key()) == hash(there.agreement_key())

    @pytest.mark.parametrize(
        "other",
        [
            pytest.param({"name": "declares_marker"}, id="unstated-mode"),
            pytest.param({"name": "declares_marker_and_secret", "mode": "enabled"}, id="other-ids"),
            pytest.param({"name": "declares_marker", "mode": "disabled"}, id="observe-only"),
            pytest.param(
                {"name": "declares_marker", "mode": "enabled", "require_masked_targets_within_iterations": 4},
                id="other-deadline",
            ),
        ],
    )
    def test_agreement_key_separates_decisions_that_change_metrics_or_checks(self, tokenizers, other):
        enabled = resolve(tokenizers, "declares_marker", mode="enabled")
        assert resolve(tokenizers, **other).agreement_key() != enabled.agreement_key()


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
    def awkward(self, tokenizers):
        """A real decision whose token strings and tokenizer path hold spaces, '=', ']' and quotes."""
        resolved = resolve(tokenizers, "declares_marker", mode="enabled")
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
        assert json.loads(parsed["tokens"]) == list(self.AWKWARD_TOKENS)
        assert parsed["tokenizer"] == self.AWKWARD_TOKENIZER
        assert json.loads(parsed["token_ids"]) == json.loads(parsed["observed_token_ids"]) == [MARKER_ID]
        assert parsed["forward_step"] == GPT_FORWARD_STEP
        assert (parsed["mode"], parsed["enforced"], parsed["enabled"]) == ("enabled", "true", "true")

    def test_only_the_first_process_of_a_node_logs(self, awkward, caplog):
        caplog.set_level(logging.INFO, logger=RESOLUTION_LOGGER)
        fields = banner_fields(awkward, None)
        log_node_banner(logging.getLogger(RESOLUTION_LOGGER), "token-masking", fields, rank=5, local_rank=1)
        assert not [record for record in caplog.records if "[token-masking]" in record.getMessage()]

    def test_without_a_forward_step_or_declaration(self, tokenizers):
        fields = dict(banner_fields(resolve(tokenizers, "declares_nothing"), None))
        assert fields["forward_step"] == "none"
        assert fields["tokenizer_declares"] == "null"
        assert fields["mode"] == "unstated"


class TestWandbSummary:
    def test_keys_mirror_the_record_under_the_namespace_and_never_shadow_a_metric(self, tokenizers):
        resolved = resolve(tokenizers, "declares_marker", mode="enabled")
        summary = wandb_summary(resolved, gpt_step.forward_step)
        record = resolved.record()
        assert summary == {
            **{f"token_masking/{key}": value for key, value in record.items()},
            "token_masking/forward_step": GPT_FORWARD_STEP,
        }
        assert not set(summary) & set(REPORT_KEYS)
        assert json.loads(json.dumps(summary)) == summary

    def test_forward_step_is_none_when_setup_was_given_none(self, tokenizers):
        summary = wandb_summary(resolve(tokenizers, "declares_nothing"), None)
        assert summary["token_masking/forward_step"] is None


class _DecoratedFunctor:
    """A stateful forward step whose ``__call__`` is marked as applying token masking."""

    @applies_token_masking
    def __call__(self, state, data_iterator, model, return_schedule_plan=False):
        return gpt_step.forward_step(state, data_iterator, model, return_schedule_plan)


class _PlainFunctor:
    """A stateful forward step that does not apply token masking."""

    def __call__(self, state, data_iterator, model, return_schedule_plan=False):
        return gpt_step.forward_step(state, data_iterator, model, return_schedule_plan)


class TestForwardStepCapability:
    @pytest.fixture
    def masking(self, tokenizers):
        return resolve(tokenizers, "declares_marker", mode="enabled")

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
    def test_steps_that_apply_token_masking_pass(self, masking, forward_step):
        require_forward_step_applies_token_masking(forward_step, masking)

    @pytest.mark.parametrize(
        "forward_step",
        [
            pytest.param(vlm_step.forward_step, id="vlm_step.forward_step"),
            pytest.param(llava_step.forward_step, id="llava_step.forward_step"),
            pytest.param(functools.partial(vlm_step.forward_step, "state"), id="partial-of-vlm"),
            pytest.param(_PlainFunctor(), id="unmarked-functor"),
        ],
    )
    @pytest.mark.parametrize("block", [{"mode": "enabled"}, {"mode": "disabled"}], ids=["masking", "observing"])
    def test_steps_that_do_not_apply_it_raise_naming_the_step(self, tokenizers, forward_step, block):
        resolved = resolve(tokenizers, "declares_marker", **block)
        with pytest.raises(TokenMaskingError, match="does not apply token masking") as raised:
            require_forward_step_applies_token_masking(forward_step, resolved)
        assert forward_step_name(forward_step) in str(raised.value)
        assert str([MARKER_ID]) in str(raised.value)
        action = "this run masks token ids" if block["mode"] == "enabled" else "counts (without masking)"
        assert action in str(raised.value)
        assert "token_masking: {mode: disabled, token_ids: []}" in str(raised.value)

    def test_the_named_steps_are_the_ones_their_modules_define(self):
        assert forward_step_name(vlm_step.forward_step) == "megatron.bridge.training.vlm_step.forward_step"
        assert forward_step_name(llava_step.forward_step) == "megatron.bridge.training.llava_step.forward_step"
        assert forward_step_name(_PlainFunctor()).endswith("._PlainFunctor")

    def test_no_forward_step_raises_when_ids_are_observed(self, masking):
        with pytest.raises(TokenMaskingError, match="no forward_step_func"):
            require_forward_step_applies_token_masking(None, masking)

    @pytest.mark.parametrize(
        "forward_step", [None, vlm_step.forward_step, _PlainFunctor()], ids=["none", "vlm", "unmarked-functor"]
    )
    @pytest.mark.parametrize(
        "name, block",
        [
            pytest.param("declares_nothing", {}, id="unstated-plain-tokenizer"),
            pytest.param("declares_marker", {"mode": "disabled", "token_ids": []}, id="disabled-observing-nothing"),
        ],
    )
    def test_nothing_observed_never_checks_the_step(self, tokenizers, forward_step, name, block):
        nothing = resolve(tokenizers, name, **block)
        assert nothing.observed_token_ids == ()
        require_forward_step_applies_token_masking(forward_step, nothing)


# Each rank's enabled token ids per scenario, resolved for real on a NullTokenizer of NULL_VOCAB; NULL_VOCAB itself is
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
                outcome = masking_with_null_tokenizer("enabled", ids_per_rank[rank], NULL_VOCAB)
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
            key = masking_with_null_tokenizer("enabled", token_ids, NULL_VOCAB).agreement_key()
            assert f"ranks [{rank}]: {key}" in message

    def test_a_failure_on_one_rank_raises_its_message_on_every_rank(self, raised):
        message = raised[0]["one_rank_fails"]
        assert message is not None and raised[1]["one_rank_fails"] == message
        assert "could not be resolved on ranks [1]" in message
        assert f"TokenMaskingError: token ids [{NULL_VOCAB}] are outside the vocabulary" in message

    def test_the_failing_rank_keeps_its_own_error_as_the_cause(self, causes):
        """Its traceback then shows where resolution failed; the healthy rank has no local error to attach."""
        assert f"token ids [{NULL_VOCAB}] are outside the vocabulary" in causes[1]["one_rank_fails"]
        assert causes[0]["one_rank_fails"] is None
        assert causes[0]["different"] is None and causes[1]["different"] is None


class TestResolveForRun:
    @pytest.fixture
    def run_config(self, tokenizers):
        """The two blocks of a ConfigContainer that resolve_for_run reads, as real config objects.

        A full container also needs a model provider, optimizer and dataset that this decision never touches.
        """

        def make(name, legacy=None, **block):
            tokenizer_config = replace(tokenizers[name][0], loss_mask_token_ids=legacy)
            return SimpleNamespace(tokenizer=tokenizer_config, token_masking=TokenMaskingConfig(**block))

        return make

    @pytest.fixture(autouse=True)
    def first_process_of_its_node(self, monkeypatch, gloo_group_of_one):
        monkeypatch.setenv("LOCAL_RANK", "0")

    @staticmethod
    def banners(caplog) -> list[str]:
        return [record.getMessage() for record in caplog.records if "[token-masking]" in record.getMessage()]

    def test_records_the_decision_in_the_config_and_logs_one_banner(self, tokenizers, run_config, caplog):
        caplog.set_level(logging.INFO, logger=RESOLUTION_LOGGER)
        cfg = run_config("declares_marker", mode="enabled")
        resolved = resolve_for_run(cfg, tokenizers["declares_marker"][1], gpt_step.forward_step, CPU)
        assert resolved.token_ids == (MARKER_ID,) and resolved.enforced is True
        assert cfg.token_masking.resolved == resolved.record()
        (banner,) = self.banners(caplog)
        leading, fields = _parse_banner(banner, "token-masking")
        assert leading[1] == "rank=0"
        assert fields == dict(banner_fields(resolved, gpt_step.forward_step))
        assert json.loads(fields["tokens"]) == [MARKER]

    def test_an_unsupported_forward_step_raises_before_recording_or_logging(self, tokenizers, run_config, caplog):
        caplog.set_level(logging.INFO, logger=RESOLUTION_LOGGER)
        cfg = run_config("declares_marker", mode="enabled")
        with pytest.raises(TokenMaskingError, match="vlm_step.forward_step does not apply token masking"):
            resolve_for_run(cfg, tokenizers["declares_marker"][1], vlm_step.forward_step, CPU)
        assert cfg.token_masking.resolved is None
        assert self.banners(caplog) == []

    def test_the_legacy_tokenizer_field_is_read_from_the_config(self, tokenizers, run_config, caplog):
        caplog.set_level(logging.INFO, logger=RESOLUTION_LOGGER)
        cfg = run_config("declares_marker", legacy=[])
        resolved = resolve_for_run(cfg, tokenizers["declares_marker"][1], gpt_step.forward_step, CPU)
        assert resolved.token_ids == () and resolved.observed_token_ids == (MARKER_ID,)
        assert cfg.token_masking.resolved["source"] == "legacy_tokenizer_field"
        (banner,) = self.banners(caplog)
        _, fields = _parse_banner(banner, "token-masking")
        assert (fields["mode"], fields["enabled"], fields["token_ids"]) == ("unstated", "false", "[]")

    def test_a_resolution_error_is_raised_through_the_agreement(self, tokenizers, run_config):
        cfg = run_config("declares_nothing", mode="enabled")
        with pytest.raises(
            TokenMaskingError,
            match=r"could not be resolved on ranks \[0\]: TokenMaskingError: token_masking.mode is enabled but no "
            "token ids are given",
        ):
            resolve_for_run(cfg, tokenizers["declares_nothing"][1], gpt_step.forward_step, CPU)
        assert cfg.token_masking.resolved is None
