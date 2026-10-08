# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""The ``token_masking:`` config block, from the dataclass to the launcher's merged config and back from disk.

A run states whether it masks with ``token_masking.mode``; the archived configs choose through the legacy
``tokenizer.loss_mask_token_ids`` instead. A misread block is a silent failure: a typo leaves masking at its default,
an unquoted YAML ``on`` becomes a boolean, a legacy field can quietly contradict the stated mode. These tests cover:

- ``TokenMaskingConfig.finalize`` for every mode and every rule it enforces on its own (bool modes with the YAML
  hint, ids without a mode, empty ids with ``enabled``, requirements outside ``enabled``, their types);
- ``validate_token_ids``, ``check_against_legacy_field`` (every allowed and contradicting combination of legacy field,
  mode and ids, and a block that is not a mapping) and ``explicit_token_ids``;
- ``validate_token_masking``, the check ``ConfigContainer.validate`` runs: the block on its own first, then against
  the legacy field, so a malformed block is reported as such even when the legacy field also disagrees with it;
- the strict override keys of ``omegaconf_utils._apply_overrides``: a misspelled key in ``token_masking``, in
  ``tokenizer``, in ``logger.data_samples`` (strict inside the lenient ``LoggerConfig``) or at the top level raises
  with a did-you-mean suggestion, while lenient classes still skip;
- the real launcher merge, ``pipeline_training_run.resolve_training_config`` on the Nano pretrain recipe with YAML
  and Hydra CLI overrides, whose merged block must check exactly as the same dataclass does;
- the ``resolved`` record setup writes: carried by ``to_dict``/``to_yaml`` and dropped when the YAML is loaded back.
"""

import dataclasses
import logging
import re

import pytest
import torch
import yaml

from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.token_masking.config import (
    TOKEN_MASKING_MODES,
    TokenMaskingConfig,
    TokenMaskingError,
    check_against_legacy_field,
    explicit_token_ids,
    validate_token_ids,
    validate_token_masking,
)
from megatron.bridge.training.tokenizers.config import TokenizerConfig
from megatron.bridge.training.utils.config_utils import apply_run_config_backward_compat
from megatron.bridge.training.utils.omegaconf_utils import OverridesError, _apply_overrides, apply_overrides
from tests.unit_tests.token_masking_fixtures import (
    MARKER,
    MARKER_ID,
    build_tiny_hf_tokenizer,
    hf_tokenizer_config,
    resolve,
)


YAML_BOOL_HINT = "YAML reads unquoted on/off/yes/no as booleans. Write mode: enabled or mode: disabled."


def _outcome(token_masking, legacy_token_ids) -> str | None:
    """Run the check ``ConfigContainer.validate`` runs on the block; the error text or None."""
    try:
        validate_token_masking(token_masking, legacy_token_ids)
    except TokenMaskingError as error:
        return str(error)
    return None


# ---------------------------------------------------------------------------------------------------------------------
# TokenMaskingConfig.finalize
# ---------------------------------------------------------------------------------------------------------------------

VALID_BLOCKS = {
    "omitted": {},
    "enabled-ids-from-tokenizer": {"mode": "enabled"},
    "disabled": {"mode": "disabled"},
    "enabled-with-ids": {"mode": "enabled", "token_ids": [131072]},
    "enabled-id-zero": {"mode": "enabled", "token_ids": [0, 5]},
    "disabled-observing-ids": {"mode": "disabled", "token_ids": [131072]},
    "disabled-empty-ids": {"mode": "disabled", "token_ids": []},
    "enabled-require-false": {"mode": "enabled", "require_masked_targets": False},
    "enabled-require-within-one": {
        "mode": "enabled",
        "require_masked_targets": True,
        "require_masked_targets_within_iterations": 1,
    },
    "enabled-within-large": {"mode": "enabled", "require_masked_targets_within_iterations": 500},
}

INVALID_BLOCKS = {
    "mode-yaml-true": ({"mode": True}, "token_masking.mode is the boolean True: " + YAML_BOOL_HINT),
    "mode-yaml-false": ({"mode": False}, "token_masking.mode is the boolean False: " + YAML_BOOL_HINT),
    "mode-capitalised": ({"mode": "Enabled"}, f"must be one of {TOKEN_MASKING_MODES} or omitted, got 'Enabled'"),
    "mode-on-string": ({"mode": "on"}, "must be one of ('enabled', 'disabled') or omitted, got 'on'"),
    "mode-empty-string": ({"mode": ""}, "must be one of ('enabled', 'disabled') or omitted, got ''"),
    "mode-int": ({"mode": 1}, "must be one of ('enabled', 'disabled') or omitted, got 1"),
    "ids-without-mode": ({"token_ids": [131072]}, "token_masking.token_ids needs token_masking.mode: enabled"),
    "empty-ids-without-mode": ({"token_ids": []}, "token_masking.token_ids needs token_masking.mode: enabled"),
    "enabled-empty-ids": (
        {"mode": "enabled", "token_ids": []},
        "token_masking.token_ids is empty with mode: enabled; to mask nothing use mode: disabled",
    ),
    "enabled-repeated-ids": ({"mode": "enabled", "token_ids": [5, 5]}, "token_masking.token_ids repeats token ids"),
    "disabled-negative-id": ({"mode": "disabled", "token_ids": [-1]}, "token_masking.token_ids holds negative"),
    "enabled-scalar-ids": ({"mode": "enabled", "token_ids": 131072}, "token_masking.token_ids must be a list"),
    "require-unstated-mode": (
        {"require_masked_targets": True},
        "token_masking.require_masked_targets applies only with mode: enabled",
    ),
    "require-false-disabled": (
        {"mode": "disabled", "require_masked_targets": False},
        "token_masking.require_masked_targets applies only with mode: enabled",
    ),
    "within-disabled": (
        {"mode": "disabled", "require_masked_targets_within_iterations": 10},
        "token_masking.require_masked_targets_within_iterations applies only with mode: enabled",
    ),
    "both-requirements-unstated-mode": (
        {"require_masked_targets": True, "require_masked_targets_within_iterations": 10},
        "token_masking.require_masked_targets, require_masked_targets_within_iterations applies only with mode: enabled",
    ),
    "within-zero": ({"mode": "enabled", "require_masked_targets_within_iterations": 0}, "positive integer, got 0"),
    "within-negative": (
        {"mode": "enabled", "require_masked_targets_within_iterations": -3},
        "positive integer, got -3",
    ),
    "within-bool": (
        {"mode": "enabled", "require_masked_targets_within_iterations": True},
        "positive integer, got True",
    ),
    "within-float": (
        {"mode": "enabled", "require_masked_targets_within_iterations": 2.5},
        "positive integer, got 2.5",
    ),
    "within-string": (
        {"mode": "enabled", "require_masked_targets_within_iterations": "10"},
        "positive integer, got '10'",
    ),
    "require-string": ({"mode": "enabled", "require_masked_targets": "true"}, "must be true or false, got 'true'"),
    "require-int-one": ({"mode": "enabled", "require_masked_targets": 1}, "must be true or false, got 1"),
    "require-int-zero": ({"mode": "enabled", "require_masked_targets": 0}, "must be true or false, got 0"),
}


class TestFinalize:
    @pytest.mark.parametrize("block", VALID_BLOCKS.values(), ids=VALID_BLOCKS.keys())
    def test_valid_block_passes(self, block):
        config = TokenMaskingConfig(**block)
        config.finalize()
        assert config == TokenMaskingConfig(**block), "finalize validates; it must not rewrite the block"

    @pytest.mark.parametrize(("block", "message"), INVALID_BLOCKS.values(), ids=INVALID_BLOCKS.keys())
    def test_invalid_block_raises(self, block, message):
        with pytest.raises(TokenMaskingError, match=re.escape(message)):
            TokenMaskingConfig(**block).finalize()

    def test_error_is_a_runtime_error(self):
        """Callers that catch RuntimeError around setup still see a token-masking misconfiguration."""
        assert issubclass(TokenMaskingError, RuntimeError)


# ---------------------------------------------------------------------------------------------------------------------
# validate_token_ids, check_against_legacy_field, explicit_token_ids
# ---------------------------------------------------------------------------------------------------------------------


class TestValidateTokenIds:
    @pytest.mark.parametrize("token_ids", [[], [0], [131072, 5], (7, 8)], ids=["empty", "zero", "list", "tuple"])
    def test_accepts_distinct_non_negative_ints(self, token_ids):
        validate_token_ids(token_ids, "field")

    @pytest.mark.parametrize(
        ("token_ids", "message"),
        [
            ([True], "field must hold integer token ids, got [True] in [True]"),
            ([5, False], "field must hold integer token ids, got [False] in [5, False]"),
            ([1.0], "field must hold integer token ids, got [1.0]"),
            ([5, "6"], "field must hold integer token ids, got ['6']"),
            ([None], "field must hold integer token ids, got [None]"),
            ([-1], "field holds negative token ids [-1]"),
            ([3, -2, -7], "field holds negative token ids [-2, -7]"),
            ([5, 6, 5], "field repeats token ids: [5, 6, 5]"),
            ("131072", "field must be a list of token ids, got '131072'"),
            (131072, "field must be a list of token ids, got 131072"),
            (None, "field must be a list of token ids, got None"),
            ({131072: True}, "field must be a list of token ids"),
        ],
        ids=[
            "bool",
            "bool-among-ints",
            "float",
            "string-element",
            "none-element",
            "negative",
            "negatives-all-listed",
            "duplicate",
            "string",
            "scalar",
            "none",
            "mapping",
        ],
    )
    def test_rejects(self, token_ids, message):
        with pytest.raises(TokenMaskingError, match=re.escape(message)):
            validate_token_ids(token_ids, "field")


LEGACY_ALLOWED = {
    "nothing-stated": (None, None, None),
    "unstated-legacy-empty": (None, None, []),
    "unstated-legacy-ids": (None, None, [131072]),
    "enabled-no-legacy": ("enabled", None, None),
    "enabled-ids-no-legacy": ("enabled", [131072], None),
    "enabled-ids-from-legacy": ("enabled", None, [131072]),
    "enabled-legacy-equal": ("enabled", [131072], [131072]),
    "enabled-legacy-equal-other-order": ("enabled", [5, 6], [6, 5]),
    "disabled-no-legacy": ("disabled", None, None),
    "disabled-legacy-empty": ("disabled", None, []),
    "disabled-observing-legacy-empty": ("disabled", [131072], []),
    "disabled-observing-no-legacy": ("disabled", [131072], None),
}

LEGACY_CONTRADICTING = {
    "disabled-legacy-masks": (
        "disabled",
        None,
        [131072],
        "tokenizer.loss_mask_token_ids=[131072] masks tokens but token_masking.mode is disabled; "
        "set tokenizer.loss_mask_token_ids: [] or remove it",
    ),
    "disabled-legacy-masks-the-observed-ids": (
        "disabled",
        [131072],
        [131072],
        "tokenizer.loss_mask_token_ids=[131072] masks tokens but token_masking.mode is disabled",
    ),
    "enabled-legacy-empty": (
        "enabled",
        None,
        [],
        "tokenizer.loss_mask_token_ids=[] masks nothing but token_masking.mode is enabled; "
        "remove tokenizer.loss_mask_token_ids and state the ids in token_masking.token_ids",
    ),
    "enabled-ids-legacy-empty": (
        "enabled",
        [131072],
        [],
        "tokenizer.loss_mask_token_ids=[] masks nothing but token_masking.mode is enabled",
    ),
    "enabled-legacy-differs": (
        "enabled",
        [131072],
        [131073],
        "tokenizer.loss_mask_token_ids=[131073] differs from token_masking.token_ids=[131072]; "
        "keep one, in token_masking.token_ids",
    ),
    "enabled-legacy-superset": ("enabled", [5], [5, 6], "differs from token_masking.token_ids=[5]"),
    "enabled-legacy-subset": ("enabled", [5, 6], [5], "differs from token_masking.token_ids=[5, 6]"),
    "unstated-legacy-bool": (None, None, [True], "tokenizer.loss_mask_token_ids must hold integer token ids"),
    "unstated-legacy-repeats": (None, None, [5, 5], "tokenizer.loss_mask_token_ids repeats token ids"),
    "unstated-legacy-not-a-list": (None, None, "131072", "tokenizer.loss_mask_token_ids must be a list"),
    "enabled-legacy-repeats": ("enabled", [5], [5, 5], "tokenizer.loss_mask_token_ids repeats token ids"),
    "disabled-legacy-negative": ("disabled", None, [-1], "tokenizer.loss_mask_token_ids holds negative"),
}


class TestCheckAgainstLegacyField:
    @pytest.mark.parametrize(("mode", "token_ids", "legacy"), LEGACY_ALLOWED.values(), ids=LEGACY_ALLOWED.keys())
    def test_agreeing_combination_passes(self, mode, token_ids, legacy):
        check_against_legacy_field(TokenMaskingConfig(mode=mode, token_ids=token_ids), legacy)

    @pytest.mark.parametrize(
        ("mode", "token_ids", "legacy", "message"), LEGACY_CONTRADICTING.values(), ids=LEGACY_CONTRADICTING.keys()
    )
    def test_contradicting_combination_raises(self, mode, token_ids, legacy, message):
        with pytest.raises(TokenMaskingError, match=re.escape(message)):
            check_against_legacy_field(TokenMaskingConfig(mode=mode, token_ids=token_ids), legacy)

    @pytest.mark.parametrize("block", ["enabled", None, True, ["enabled"]], ids=["string", "null", "bool", "list"])
    def test_block_that_is_not_a_mapping_raises(self, block):
        """``token_masking: enabled`` merges as a bare string; it must be named, not crash on ``.mode``."""
        with pytest.raises(
            TokenMaskingError,
            match=re.escape(
                f"token_masking must be a mapping such as token_masking: {{mode: enabled}}, got {block!r}"
            ),
        ):
            check_against_legacy_field(block, None)


# Blocks that finalize rejects, next to a legacy field that the legacy check alone would also reject, with another
# message: validation must report what is wrong with the block.
MALFORMED_BLOCK_AND_CONTRADICTING_LEGACY = {
    "yaml-bool-mode": ({"mode": True}, [], "token_masking.mode is the boolean True: " + YAML_BOOL_HINT),
    "capitalised-mode": ({"mode": "Enabled"}, [], f"must be one of {TOKEN_MASKING_MODES} or omitted, got 'Enabled'"),
    "enabled-empty-ids": (
        {"mode": "enabled", "token_ids": []},
        [],
        "token_masking.token_ids is empty with mode: enabled; to mask nothing use mode: disabled",
    ),
}


class TestValidateTokenMasking:
    @pytest.mark.parametrize(("mode", "token_ids", "legacy"), LEGACY_ALLOWED.values(), ids=LEGACY_ALLOWED.keys())
    def test_agreeing_combination_passes(self, mode, token_ids, legacy):
        validate_token_masking(TokenMaskingConfig(mode=mode, token_ids=token_ids), legacy)

    @pytest.mark.parametrize(
        ("mode", "token_ids", "legacy", "message"), LEGACY_CONTRADICTING.values(), ids=LEGACY_CONTRADICTING.keys()
    )
    def test_contradicting_combination_raises(self, mode, token_ids, legacy, message):
        with pytest.raises(TokenMaskingError, match=re.escape(message)):
            validate_token_masking(TokenMaskingConfig(mode=mode, token_ids=token_ids), legacy)

    @pytest.mark.parametrize(
        ("block", "legacy", "message"),
        MALFORMED_BLOCK_AND_CONTRADICTING_LEGACY.values(),
        ids=MALFORMED_BLOCK_AND_CONTRADICTING_LEGACY.keys(),
    )
    def test_the_block_is_checked_before_the_legacy_field(self, block, legacy, message):
        with pytest.raises(TokenMaskingError, match="masks nothing but token_masking.mode is enabled"):
            check_against_legacy_field(TokenMaskingConfig(**block), legacy)
        with pytest.raises(TokenMaskingError, match=re.escape(message)):
            validate_token_masking(TokenMaskingConfig(**block), legacy)

    @pytest.mark.parametrize("block", ["enabled", None, True, ["enabled"]], ids=["string", "null", "bool", "list"])
    def test_block_that_is_not_a_mapping_raises(self, block):
        with pytest.raises(
            TokenMaskingError,
            match=re.escape(
                f"token_masking must be a mapping such as token_masking: {{mode: enabled}}, got {block!r}"
            ),
        ):
            validate_token_masking(block, [MARKER_ID])


class TestExplicitTokenIds:
    @pytest.mark.parametrize(
        ("token_ids", "legacy", "expected"),
        [
            ([131072], None, [131072]),
            ([131072], [131073], [131072]),
            ([], [131072], []),
            (None, [7, 8], [7, 8]),
            (None, [], None),
            (None, None, None),
        ],
        ids=[
            "block-ids",
            "block-ids-win-over-legacy",
            "block-empty-list-is-stated",
            "legacy-ids",
            "legacy-empty-names-nothing",
            "nothing",
        ],
    )
    def test_names_the_block_ids_else_the_legacy_ids(self, token_ids, legacy, expected):
        assert explicit_token_ids(TokenMaskingConfig(mode="enabled", token_ids=token_ids), legacy) == expected

    @pytest.mark.parametrize("from_block", [True, False], ids=["block", "legacy"])
    def test_returns_a_copy(self, from_block):
        """The caller may extend or sort the result; the config must keep what the user wrote."""
        ids = [131072]
        config = TokenMaskingConfig(mode="enabled", token_ids=ids if from_block else None)
        result = explicit_token_ids(config, None if from_block else ids)
        result.append(99)
        assert ids == [131072]


# ---------------------------------------------------------------------------------------------------------------------
# Strict override keys through omegaconf_utils
# ---------------------------------------------------------------------------------------------------------------------


@pytest.fixture
def nano_pretrain_config(run_module) -> ConfigContainer:
    """A fresh real ConfigContainer from the Nano pretrain recipe (pure dataclass construction, no network)."""
    return run_module.RECIPE_MAP[("nano", "pretrain")](None)


STRICT_TYPOS = {
    "token-masking-mode": ({"token_masking": {"mdoe": "enabled"}}, "TokenMaskingConfig", "mdoe", "mode"),
    "token-masking-ids": ({"token_masking": {"token_id": [5]}}, "TokenMaskingConfig", "token_id", "token_ids"),
    "token-masking-require": (
        {"token_masking": {"require_masked_target": False}},
        "TokenMaskingConfig",
        "require_masked_target",
        "require_masked_targets",
    ),
    "tokenizer-legacy-field": (
        {"tokenizer": {"loss_mask_token_id": [5]}},
        "TokenizerConfig",
        "loss_mask_token_id",
        "loss_mask_token_ids",
    ),
    "top-level-block": ({"token_maskng": {"mode": "enabled"}}, "ConfigContainer", "token_maskng", "token_masking"),
    "top-level-tokenizer": ({"tokeniser": {"tokenizer_model": "x"}}, "ConfigContainer", "tokeniser", "tokenizer"),
    "data-samples": ({"logger": {"data_samples": {"enable": False}}}, "DataSamplesConfig", "enable", "enabled"),
}


class TestStrictOverrideKeys:
    @pytest.mark.parametrize(
        ("overrides", "class_name", "key", "suggestion"), STRICT_TYPOS.values(), ids=STRICT_TYPOS.keys()
    )
    def test_typo_raises_with_suggestion(self, nano_pretrain_config, overrides, class_name, key, suggestion):
        with pytest.raises(
            ValueError, match=re.escape(f"Unknown key '{key}' for {class_name}. Did you mean '{suggestion}'?")
        ):
            apply_overrides(nano_pretrain_config, overrides, {})

    def test_error_lists_the_valid_keys(self):
        with pytest.raises(ValueError) as raised:
            _apply_overrides(TokenMaskingConfig(), {"zzzz": 1})
        message = str(raised.value)
        assert "Did you mean" not in message, "no field is close to 'zzzz'; a suggestion would be noise"
        valid = message.split("Valid keys: ", 1)[1]
        for name in ("mode", "token_ids", "require_masked_targets", "require_masked_targets_within_iterations"):
            assert name in valid

    @pytest.mark.parametrize("key", ["reject_unknown_override_keys", "finalize"])
    def test_attributes_that_are_not_fields_are_unknown(self, key):
        """Keys are checked against the dataclass fields, so YAML cannot reach a class variable or a method."""
        with pytest.raises(ValueError, match=re.escape(f"Unknown key '{key}' for TokenMaskingConfig")):
            _apply_overrides(TokenMaskingConfig(), {key: False})
        assert TokenMaskingConfig.reject_unknown_override_keys is True
        assert callable(TokenMaskingConfig().finalize)

    def test_known_keys_still_apply(self):
        token_masking = TokenMaskingConfig()
        _apply_overrides(token_masking, {"mode": "disabled", "token_ids": [131072]})
        assert token_masking == TokenMaskingConfig(mode="disabled", token_ids=[131072])
        tokenizer = TokenizerConfig()
        _apply_overrides(tokenizer, {"loss_mask_token_ids": []})
        assert tokenizer.loss_mask_token_ids == []

    def test_lenient_classes_still_skip_unknown_keys(self, nano_pretrain_config, caplog):
        """Only the opted-in classes are strict: logger and dataset keep skipping with a warning."""
        overrides = {
            "logger": {"not_a_logger_key": 1, "log_interval": 7},
            "dataset": {"not_a_dataset_key": 2},
        }
        with caplog.at_level(logging.WARNING, logger="megatron.bridge.training.utils.omegaconf_utils"):
            apply_overrides(nano_pretrain_config, overrides, {})
        assert nano_pretrain_config.logger.log_interval == 7
        assert not hasattr(nano_pretrain_config.logger, "not_a_logger_key")
        assert not hasattr(nano_pretrain_config.dataset, "not_a_dataset_key")
        skipped = [record.getMessage() for record in caplog.records if "Skipping" in record.getMessage()]
        assert any("'not_a_logger_key'" in message and "LoggerConfig" in message for message in skipped)
        assert any("'not_a_dataset_key'" in message for message in skipped)


# ---------------------------------------------------------------------------------------------------------------------
# End to end through the launcher's own merge
# ---------------------------------------------------------------------------------------------------------------------


def _launch_config(run_module, tmp_path, yaml_text: str | None = None, cli: tuple[str, ...] = ()) -> ConfigContainer:
    """The Nano pretrain config exactly as ``pipeline_training_run`` resolves it from a YAML and CLI overrides."""
    config_file = None
    if yaml_text is not None:
        config_file = tmp_path / "override.yaml"
        config_file.write_text(yaml_text)
    cfg, _ = run_module.resolve_training_config(
        "nano", "pretrain", None, str(config_file) if config_file else None, list(cli)
    )
    return cfg


LAUNCH_CASES = {
    "enabled-with-ids": (
        "token_masking:\n  mode: enabled\n  token_ids: [131072]\n",
        {"mode": "enabled", "token_ids": [131072]},
        None,
        None,
    ),
    "enabled-ids-from-tokenizer": ("token_masking:\n  mode: enabled\n", {"mode": "enabled"}, None, None),
    "enabled-requirements": (
        "token_masking:\n  mode: enabled\n  require_masked_targets: false\n"
        "  require_masked_targets_within_iterations: 20\n",
        {"mode": "enabled", "require_masked_targets": False, "require_masked_targets_within_iterations": 20},
        None,
        None,
    ),
    "disabled-observing": (
        "token_masking:\n  mode: disabled\n  token_ids: [131072]\n",
        {"mode": "disabled", "token_ids": [131072]},
        None,
        None,
    ),
    "disabled-legacy-empty": (
        "tokenizer:\n  loss_mask_token_ids: []\ntoken_masking:\n  mode: disabled\n",
        {"mode": "disabled"},
        [],
        None,
    ),
    "enabled-legacy-agrees": (
        "tokenizer:\n  loss_mask_token_ids: [131072]\ntoken_masking:\n  mode: enabled\n  token_ids: [131072]\n",
        {"mode": "enabled", "token_ids": [131072]},
        [131072],
        None,
    ),
    "legacy-only": ("tokenizer:\n  loss_mask_token_ids: [131072]\n", {}, [131072], None),
    "disabled-contradicted-by-legacy": (
        "tokenizer:\n  loss_mask_token_ids: [131072]\ntoken_masking:\n  mode: disabled\n",
        {"mode": "disabled"},
        [131072],
        "masks tokens but token_masking.mode is disabled",
    ),
    "enabled-contradicted-by-empty-legacy": (
        "tokenizer:\n  loss_mask_token_ids: []\ntoken_masking:\n  mode: enabled\n",
        {"mode": "enabled"},
        [],
        "masks nothing but token_masking.mode is enabled",
    ),
    "enabled-differs-from-legacy": (
        "tokenizer:\n  loss_mask_token_ids: [131073]\ntoken_masking:\n  mode: enabled\n  token_ids: [131072]\n",
        {"mode": "enabled", "token_ids": [131072]},
        [131073],
        "differs from token_masking.token_ids=[131072]",
    ),
    "ids-without-mode": (
        "token_masking:\n  token_ids: [131072]\n",
        {"token_ids": [131072]},
        None,
        "token_masking.token_ids needs token_masking.mode",
    ),
    "yaml-on": ("token_masking:\n  mode: on\n", {"mode": True}, None, "the boolean True: " + YAML_BOOL_HINT),
    "yaml-yes": ("token_masking:\n  mode: yes\n", {"mode": True}, None, "the boolean True: " + YAML_BOOL_HINT),
    "yaml-off": ("token_masking:\n  mode: off\n", {"mode": False}, None, "the boolean False: " + YAML_BOOL_HINT),
    "requirement-outside-enabled": (
        "token_masking:\n  mode: disabled\n  require_masked_targets: true\n",
        {"mode": "disabled", "require_masked_targets": True},
        None,
        "require_masked_targets applies only with mode: enabled",
    ),
    "within-zero": (
        "token_masking:\n  mode: enabled\n  require_masked_targets_within_iterations: 0\n",
        {"mode": "enabled", "require_masked_targets_within_iterations": 0},
        None,
        "must be a positive integer, got 0",
    ),
}


class TestLauncherMerge:
    def test_omitted_block_is_the_default(self, run_module, tmp_path):
        """A config that never mentions the block gets an unstated TokenMaskingConfig, which checks clean."""
        cfg = _launch_config(run_module, tmp_path, "train:\n  train_iters: 5\n")
        assert type(cfg.token_masking) is TokenMaskingConfig
        assert cfg.token_masking == TokenMaskingConfig()
        assert cfg.tokenizer.loss_mask_token_ids is None
        assert _outcome(cfg.token_masking, cfg.tokenizer.loss_mask_token_ids) is None

    @pytest.mark.parametrize(("yaml_text", "block", "legacy", "error"), LAUNCH_CASES.values(), ids=LAUNCH_CASES.keys())
    def test_merged_block_checks_as_the_dataclass_does(self, run_module, tmp_path, yaml_text, block, legacy, error):
        cfg = _launch_config(run_module, tmp_path, yaml_text)
        assert cfg.token_masking == TokenMaskingConfig(**block)
        assert cfg.tokenizer.loss_mask_token_ids == legacy
        merged = _outcome(cfg.token_masking, cfg.tokenizer.loss_mask_token_ids)
        assert merged == _outcome(TokenMaskingConfig(**block), legacy)
        if error is None:
            assert merged is None
        else:
            assert merged is not None and error in merged

    @pytest.mark.parametrize(
        "yaml_text", ["token_masking: enabled\n", "token_masking: null\n"], ids=["string", "null"]
    )
    def test_block_that_is_not_a_mapping_is_named(self, run_module, tmp_path, yaml_text):
        """The merge stores a scalar block as-is; the validation must name it rather than fail on ``.mode``."""
        cfg = _launch_config(run_module, tmp_path, yaml_text)
        assert not isinstance(cfg.token_masking, TokenMaskingConfig)
        with pytest.raises(TokenMaskingError, match="token_masking must be a mapping"):
            validate_token_masking(cfg.token_masking, cfg.tokenizer.loss_mask_token_ids)

    def test_cli_override_states_the_mode(self, run_module, tmp_path):
        cfg = _launch_config(
            run_module, tmp_path, cli=("token_masking.mode=enabled", "token_masking.token_ids=[131072]")
        )
        assert cfg.token_masking == TokenMaskingConfig(mode="enabled", token_ids=[131072])
        assert all(type(token_id) is int for token_id in cfg.token_masking.token_ids)
        assert _outcome(cfg.token_masking, cfg.tokenizer.loss_mask_token_ids) is None

    def test_cli_override_is_applied_after_the_yaml(self, run_module, tmp_path):
        cfg = _launch_config(
            run_module,
            tmp_path,
            "token_masking:\n  mode: disabled\n  token_ids: [131072]\n",
            ("token_masking.mode=enabled",),
        )
        assert cfg.token_masking == TokenMaskingConfig(mode="enabled", token_ids=[131072])

    @pytest.mark.parametrize(
        ("yaml_text", "message"),
        [
            ("token_masking:\n  mdoe: enabled\n", "Unknown key 'mdoe' for TokenMaskingConfig. Did you mean 'mode'?"),
            (
                "tokenizer:\n  loss_mask_token_id: [131072]\n",
                "Unknown key 'loss_mask_token_id' for TokenizerConfig. Did you mean 'loss_mask_token_ids'?",
            ),
            (
                "token_maskng:\n  mode: enabled\n",
                "Unknown key 'token_maskng' for ConfigContainer. Did you mean 'token_masking'?",
            ),
            (
                "logger:\n  data_samples:\n    enable: false\n",
                "Unknown key 'enable' for DataSamplesConfig. Did you mean 'enabled'?",
            ),
        ],
        ids=["in-block", "in-tokenizer", "top-level", "in-logger-data-samples"],
    )
    def test_yaml_typo_raises(self, run_module, tmp_path, yaml_text, message):
        with pytest.raises(ValueError, match=re.escape(message)):
            _launch_config(run_module, tmp_path, yaml_text)

    def test_added_cli_key_typo_raises(self, run_module, tmp_path):
        """``+key=`` lets Hydra add a key its struct mode would refuse; the strict class must still refuse it."""
        with pytest.raises(
            ValueError, match=re.escape("Unknown key 'mdoe' for TokenMaskingConfig. Did you mean 'mode'?")
        ):
            _launch_config(run_module, tmp_path, cli=("+token_masking.mdoe=enabled",))
        with pytest.raises(ValueError, match=re.escape("Did you mean 'token_masking'?")):
            _launch_config(run_module, tmp_path, cli=("+token_maskng.mode=enabled",))

    def test_plain_cli_key_typo_raises(self, run_module, tmp_path):
        with pytest.raises(OverridesError, match="token_masking.mdoe"):
            _launch_config(run_module, tmp_path, cli=("token_masking.mdoe=enabled",))

    def test_lenient_sections_still_skip_unknown_keys(self, run_module, tmp_path):
        cfg = _launch_config(run_module, tmp_path, "logger:\n  not_a_logger_key: 1\n  log_interval: 7\n")
        assert cfg.logger.log_interval == 7
        assert not hasattr(cfg.logger, "not_a_logger_key")


# ---------------------------------------------------------------------------------------------------------------------
# ConfigContainer field and the resolved record
# ---------------------------------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def resolved_record(tmp_path_factory) -> dict:
    """The record setup writes to ``token_masking.resolved``, from a real resolution on a declaring tokenizer."""
    directory = build_tiny_hf_tokenizer(tmp_path_factory.mktemp("declaring_tokenizer"), [MARKER_ID])
    resolved = resolve(
        TokenMaskingConfig(mode="enabled", token_ids=[MARKER_ID]), hf_tokenizer_config(directory), torch.device("cpu")
    )
    return resolved.record()


STATED_BLOCK = {
    "mode": "enabled",
    "token_ids": [MARKER_ID],
    "require_masked_targets": True,
    "require_masked_targets_within_iterations": 3,
}


class TestContainerField:
    def test_recipe_containers_do_not_share_the_block(self, run_module):
        """Setup writes ``resolved`` into the block; a shared default would leak one run's record into another."""
        first = run_module.RECIPE_MAP[("nano", "pretrain")](None)
        second = run_module.RECIPE_MAP[("nano", "pretrain")](None)
        assert first.token_masking == second.token_masking == TokenMaskingConfig()
        first.token_masking.resolved = {"mode": "enabled"}
        assert second.token_masking.resolved is None

    def test_resolved_cannot_be_passed_to_the_constructor(self):
        """It is init=False: a saved config can only carry it back in if loading drops it, as tested below."""
        assert not {f.name: f for f in dataclasses.fields(TokenMaskingConfig)}["resolved"].init
        with pytest.raises(TypeError):
            TokenMaskingConfig(mode="enabled", resolved={"mode": "enabled"})

    def test_resolved_record_is_serialised(self, nano_pretrain_config, resolved_record, tmp_path):
        assert resolved_record["token_ids"] == [MARKER_ID] and resolved_record["tokens"] == [MARKER]
        nano_pretrain_config.token_masking = TokenMaskingConfig(**STATED_BLOCK)
        nano_pretrain_config.token_masking.resolved = resolved_record
        assert nano_pretrain_config.to_dict()["token_masking"]["resolved"] == resolved_record
        path = tmp_path / "run_config.yaml"
        nano_pretrain_config.to_yaml(str(path))
        on_disk = yaml.safe_load(path.read_text())["token_masking"]
        assert on_disk["resolved"] == resolved_record
        assert {key: on_disk[key] for key in STATED_BLOCK} == STATED_BLOCK

    def test_loading_a_saved_config_drops_the_record(self, nano_pretrain_config, resolved_record, tmp_path):
        nano_pretrain_config.token_masking = TokenMaskingConfig(**STATED_BLOCK)
        nano_pretrain_config.token_masking.resolved = resolved_record
        path = tmp_path / "run_config.yaml"
        nano_pretrain_config.to_yaml(str(path))

        from_yaml = ConfigContainer.from_yaml(str(path))
        assert from_yaml.token_masking == TokenMaskingConfig(**STATED_BLOCK)
        assert from_yaml.token_masking.resolved is None

        from_dict = ConfigContainer.from_dict(nano_pretrain_config.to_dict())
        assert from_dict.token_masking == TokenMaskingConfig(**STATED_BLOCK)

        checkpoint_view = apply_run_config_backward_compat(yaml.safe_load(path.read_text()))["token_masking"]
        assert "resolved" not in checkpoint_view
        assert {key: checkpoint_view[key] for key in STATED_BLOCK} == STATED_BLOCK
