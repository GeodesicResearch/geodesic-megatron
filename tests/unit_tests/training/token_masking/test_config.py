# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""The ``token_masking:`` config block, from the dataclass to the launcher's merged config and back from disk.

The block alone decides which ids a run masks (``enabled`` with ``token_ids``) and which it only measures
(``masked_validation.token_ids``). A misread block is a silent failure: a typo leaves masking at its default, a quoted
``"true"`` is a string, an outdated key from an earlier design would be skipped. These tests cover:

- ``TokenMaskingConfig.finalize`` and ``MaskedValidationConfig.finalize``: every rule (``enabled`` exactly when ids are
  listed, measured ids restating the masked ones, the held-out set's path, interval and iterations) and its message;
- ``measured_token_ids``, ``validate_token_ids`` and ``validate_token_masking`` (a block that is not a mapping);
- the refusal of enabled masking on tied embeddings, on knowledge distillation and on a model config that does not
  state whether it ties them, with real model providers;
- the strict and removed override keys of ``omegaconf_utils._apply_overrides``: a removed key (``token_masking.mode``,
  the ``require_*`` keys, ``tokenizer.loss_mask_token_ids``) stops the run with what to write instead, and a
  misspelled key stops it with a did-you-mean suggestion, while lenient classes still skip;
- the real launcher merge, ``pipeline_training_run.resolve_training_config`` on the Nano pretrain recipe with YAML and
  Hydra CLI overrides, whose merged block must check exactly as the same dataclass does;
- the block in the container: one per container, serialised by ``to_dict``/``to_yaml`` and loaded back.
"""

import dataclasses
import logging
import re

import pytest
import torch
import yaml
from megatron.core.transformer.spec_utils import ModuleSpec

from megatron.bridge.models.distillation_provider import convert_to_distillation_provider
from megatron.bridge.models.gpt_provider import GPTModelProvider
from megatron.bridge.models.mimo.mimo_provider import MimoModelProvider
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.token_masking.config import (
    MaskedValidationConfig,
    TokenMaskingConfig,
    TokenMaskingError,
    refuse_masking_that_trains_the_masked_output_row,
    validate_token_ids,
    validate_token_masking,
)
from megatron.bridge.training.tokenizers.config import TokenizerConfig
from megatron.bridge.training.utils.omegaconf_utils import OverridesError, _apply_overrides, apply_overrides


MARKER_ID = 131072
HELD_OUT_BIN = "/data/held_out/held_out_text_document"
HELD_OUT_PACKED = "/data/held_out/packed/training_8192.idx.parquet"


@pytest.fixture
def nano_pretrain_config(run_module) -> ConfigContainer:
    """A fresh real ConfigContainer from the Nano pretrain recipe (pure dataclass construction, no network)."""
    return run_module.RECIPE_MAP[("nano", "pretrain")](None)


def _block(spec: dict) -> TokenMaskingConfig:
    """A TokenMaskingConfig from a YAML-shaped mapping, its ``masked_validation`` mapping as the nested dataclass."""
    spec = dict(spec)
    if isinstance(spec.get("masked_validation"), dict):
        spec["masked_validation"] = MaskedValidationConfig(**spec["masked_validation"])
    return TokenMaskingConfig(**spec)


def _outcome(token_masking) -> str | None:
    """Run the check ``ConfigContainer.validate`` runs on the block; the error text or None."""
    try:
        validate_token_masking(token_masking)
    except TokenMaskingError as error:
        return str(error)
    return None


# ---------------------------------------------------------------------------------------------------------------------
# TokenMaskingConfig.finalize and MaskedValidationConfig.finalize
# ---------------------------------------------------------------------------------------------------------------------

VALID_BLOCKS = {
    "omitted": {},
    "explicitly-off": {"enabled": False},
    "enabled": {"enabled": True, "token_ids": [MARKER_ID]},
    "enabled-two-ids": {"enabled": True, "token_ids": [MARKER_ID, MARKER_ID + 1]},
    "enabled-id-zero": {"enabled": True, "token_ids": [0, 5]},
    "enabled-restating-its-ids": {"enabled": True, "token_ids": [5, 6], "masked_validation": {"token_ids": [6, 5]}},
    "measure-only": {"masked_validation": {"token_ids": [MARKER_ID]}},
    "enabled-held-out-bin": {
        "enabled": True,
        "token_ids": [MARKER_ID],
        "masked_validation": {"data_path": HELD_OUT_BIN, "interval": 100, "iters": 2},
    },
    "measure-only-held-out-packed": {
        "masked_validation": {
            "token_ids": [MARKER_ID],
            "packed_data_path": HELD_OUT_PACKED,
            "interval": 1,
            "iters": 1,
        },
    },
}

HELD_OUT = {"data_path": HELD_OUT_BIN, "interval": 10, "iters": 1}

INVALID_BLOCKS = {
    "enabled-quoted-true": ({"enabled": "true"}, "token_masking.enabled must be true or false, got 'true'"),
    "enabled-one": ({"enabled": 1, "token_ids": [5]}, "token_masking.enabled must be true or false, got 1"),
    "enabled-null": ({"enabled": None}, "token_masking.enabled must be true or false, got None"),
    "enabled-without-ids": (
        {"enabled": True},
        "token_masking.enabled is true but token_ids is empty; list the ids to mask, or omit the block",
    ),
    "enabled-empty-ids": ({"enabled": True, "token_ids": []}, "token_masking.enabled is true but token_ids is empty"),
    "ids-without-enabled": (
        {"token_ids": [MARKER_ID]},
        f"token_masking.token_ids=[{MARKER_ID}] with enabled false; write enabled: true to mask them, or move them to "
        "token_masking.masked_validation.token_ids to measure them without masking",
    ),
    "ids-explicitly-off": ({"enabled": False, "token_ids": [MARKER_ID]}, "with enabled false"),
    "ids-repeated": ({"enabled": True, "token_ids": [5, 5]}, "token_masking.token_ids repeats token ids"),
    "ids-negative": ({"enabled": True, "token_ids": [-1]}, "token_masking.token_ids holds negative token ids [-1]"),
    "ids-scalar": ({"enabled": True, "token_ids": MARKER_ID}, "token_masking.token_ids must be a list of token ids"),
    "ids-null": ({"token_ids": None}, "token_masking.token_ids must be a list of token ids, got None"),
    "measured-other-ids": (
        {"enabled": True, "token_ids": [5], "masked_validation": {"token_ids": [6]}},
        "token_masking.masked_validation.token_ids=[6] differs from token_ids=[5]: a masked run measures exactly the "
        "ids it masks; omit masked_validation.token_ids",
    ),
    "measured-subset": (
        {"enabled": True, "token_ids": [5, 6], "masked_validation": {"token_ids": [5]}},
        "token_masking.masked_validation.token_ids=[5] differs from token_ids=[5, 6]",
    ),
    "measured-repeated": (
        {"masked_validation": {"token_ids": [5, 5]}},
        "token_masking.masked_validation.token_ids repeats token ids",
    ),
    "measured-bool": (
        {"masked_validation": {"token_ids": [True]}},
        "token_masking.masked_validation.token_ids must hold integer token ids, got [True]",
    ),
    "masked-validation-null": (
        {"enabled": True, "token_ids": [5], "masked_validation": None},
        "token_masking.masked_validation must be a mapping such as masked_validation: {token_ids: [131072]}, got None",
    ),
    "both-held-out-paths": (
        {
            "enabled": True,
            "token_ids": [5],
            "masked_validation": {**HELD_OUT, "packed_data_path": HELD_OUT_PACKED},
        },
        "token_masking.masked_validation sets both data_path and packed_data_path",
    ),
    "held-out-without-interval": (
        {"enabled": True, "token_ids": [5], "masked_validation": {"data_path": HELD_OUT_BIN, "iters": 1}},
        "token_masking.masked_validation.interval must be a positive integer when a held-out set is named, got None",
    ),
    "held-out-without-iters": (
        {"enabled": True, "token_ids": [5], "masked_validation": {"packed_data_path": HELD_OUT_PACKED, "interval": 5}},
        "token_masking.masked_validation.iters must be a positive integer when a held-out set is named, got None",
    ),
    "held-out-zero-interval": (
        {"enabled": True, "token_ids": [5], "masked_validation": {**HELD_OUT, "interval": 0}},
        "masked_validation.interval must be a positive integer when a held-out set is named, got 0",
    ),
    "held-out-bool-iters": (
        {"enabled": True, "token_ids": [5], "masked_validation": {**HELD_OUT, "iters": True}},
        "masked_validation.iters must be a positive integer when a held-out set is named, got True",
    ),
    "held-out-float-interval": (
        {"enabled": True, "token_ids": [5], "masked_validation": {**HELD_OUT, "interval": 2.5}},
        "masked_validation.interval must be a positive integer when a held-out set is named, got 2.5",
    ),
    "held-out-empty-path": (
        {"enabled": True, "token_ids": [5], "masked_validation": {**HELD_OUT, "data_path": ""}},
        "token_masking.masked_validation.data_path must be a non-empty path, got ''",
    ),
    "held-out-path-not-a-string": (
        {"enabled": True, "token_ids": [5], "masked_validation": {"packed_data_path": 7, "interval": 1, "iters": 1}},
        "token_masking.masked_validation.packed_data_path must be a non-empty path, got 7",
    ),
    "interval-without-held-out-set": (
        {"enabled": True, "token_ids": [5], "masked_validation": {"interval": 10}},
        "token_masking.masked_validation.interval set without a held-out set; name one in data_path or "
        "packed_data_path, or remove them",
    ),
    "interval-and-iters-without-held-out-set": (
        {"masked_validation": {"token_ids": [5], "interval": 10, "iters": 2}},
        "token_masking.masked_validation.interval and iters set without a held-out set",
    ),
    "held-out-set-measuring-nothing": (
        {"masked_validation": HELD_OUT},
        "token_masking.masked_validation names a held-out set but no ids to measure; set token_ids (with enabled: "
        "true) or masked_validation.token_ids",
    ),
}


class TestFinalize:
    @pytest.mark.parametrize("spec", VALID_BLOCKS.values(), ids=VALID_BLOCKS.keys())
    def test_valid_block_passes(self, spec):
        config = _block(spec)
        config.finalize()
        assert config == _block(spec), "finalize validates; it must not rewrite the block"

    @pytest.mark.parametrize(("spec", "message"), INVALID_BLOCKS.values(), ids=INVALID_BLOCKS.keys())
    def test_invalid_block_raises(self, spec, message):
        with pytest.raises(TokenMaskingError, match=re.escape(message)):
            _block(spec).finalize()

    def test_error_is_a_runtime_error(self):
        """Callers that catch RuntimeError around setup still see a token-masking misconfiguration."""
        assert issubclass(TokenMaskingError, RuntimeError)

    def test_the_defaults_mask_and_measure_nothing(self):
        config = TokenMaskingConfig()
        assert (config.enabled, config.token_ids, config.measured_token_ids) == (False, [], [])
        assert config.masked_validation == MaskedValidationConfig()
        assert not config.masked_validation.evaluates


class TestMeasuredTokenIds:
    @pytest.mark.parametrize(
        ("spec", "measured"),
        [
            ({}, []),
            ({"enabled": True, "token_ids": [6, 5]}, [6, 5]),
            ({"enabled": True, "token_ids": [6, 5], "masked_validation": {"token_ids": [5, 6]}}, [6, 5]),
            ({"masked_validation": {"token_ids": [9, MARKER_ID]}}, [9, MARKER_ID]),
        ],
        ids=["omitted", "the-masked-ids", "the-masked-ids-in-their-order", "measure-only"],
    )
    def test_names_the_masked_ids_when_masking_else_the_measured_ones(self, spec, measured):
        config = _block(spec)
        config.finalize()
        assert config.measured_token_ids == measured

    def test_returns_a_copy(self):
        """The caller may extend or sort the result; the config must keep what the user wrote."""
        config = _block({"enabled": True, "token_ids": [MARKER_ID]})
        config.measured_token_ids.append(99)
        assert config.token_ids == [MARKER_ID]


class TestMaskedValidationEvaluates:
    @pytest.mark.parametrize(
        ("spec", "evaluates"),
        [
            ({}, False),
            ({"token_ids": [5]}, False),
            (HELD_OUT, True),
            ({"packed_data_path": HELD_OUT_PACKED, "interval": 1, "iters": 1}, True),
        ],
        ids=["default", "measured-ids-only", "bin-idx", "packed"],
    )
    def test_only_a_named_held_out_set_is_evaluated(self, spec, evaluates):
        assert MaskedValidationConfig(**spec).evaluates is evaluates


# ---------------------------------------------------------------------------------------------------------------------
# validate_token_ids, validate_token_masking
# ---------------------------------------------------------------------------------------------------------------------


class TestValidateTokenIds:
    @pytest.mark.parametrize("token_ids", [[], [0], [MARKER_ID, 5], (7, 8)], ids=["empty", "zero", "list", "tuple"])
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
            (MARKER_ID, "field must be a list of token ids, got 131072"),
            (None, "field must be a list of token ids, got None"),
            ({MARKER_ID: True}, "field must be a list of token ids"),
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


class TestValidateTokenMasking:
    @pytest.mark.parametrize("spec", VALID_BLOCKS.values(), ids=VALID_BLOCKS.keys())
    def test_valid_block_passes(self, spec):
        assert _outcome(_block(spec)) is None

    @pytest.mark.parametrize(("spec", "message"), INVALID_BLOCKS.values(), ids=INVALID_BLOCKS.keys())
    def test_runs_the_block_rules(self, spec, message):
        outcome = _outcome(_block(spec))
        assert outcome is not None and message in outcome

    @pytest.mark.parametrize("block", ["enabled", None, True, ["enabled"]], ids=["string", "null", "bool", "list"])
    def test_block_that_is_not_a_mapping_raises(self, block):
        """``token_masking: true`` merges as a bare scalar; it must be named, not crash on ``.enabled``."""
        with pytest.raises(
            TokenMaskingError,
            match=re.escape(
                f"token_masking must be a mapping such as token_masking: {{enabled: true, token_ids: [131072]}}, "
                f"got {block!r}"
            ),
        ):
            validate_token_masking(block)


# ---------------------------------------------------------------------------------------------------------------------
# Enabled masking refused where training would still pull the masked output row up
# ---------------------------------------------------------------------------------------------------------------------


def _gpt_provider(**overrides) -> GPTModelProvider:
    """A small real GPT provider (no model is built)."""
    settings = {
        "num_layers": 2,
        "hidden_size": 64,
        "num_attention_heads": 4,
        "vocab_size": 1000,
        "seq_length": 128,
        "tensor_model_parallel_size": 1,
        "pipeline_model_parallel_size": 1,
        "context_parallel_size": 1,
        "pipeline_dtype": None,
    }
    settings.update(overrides)
    return GPTModelProvider(**settings)


ENABLED = {"enabled": True, "token_ids": [MARKER_ID]}
NOT_MASKING = {
    "omitted": {},
    "measure-only": {"masked_validation": {"token_ids": [MARKER_ID]}},
}


class TestRefuseMaskingThatTrainsTheMaskedOutputRow:
    @pytest.fixture
    def distilling_provider(self):
        """A real DistillationProvider: an untied student converted with a teacher, as ``distill()`` requires."""
        return convert_to_distillation_provider(
            _gpt_provider(share_embeddings_and_output_weights=False),
            _gpt_provider(num_layers=4, share_embeddings_and_output_weights=False),
        )

    @pytest.fixture
    def mimo_provider(self):
        """A real MIMO provider: it holds its language model as a module spec and states no tied embeddings."""
        return MimoModelProvider(language_model_spec=ModuleSpec(module=torch.nn.Identity))

    def test_enabled_masking_on_tied_embeddings_raises(self):
        model = _gpt_provider()
        assert model.share_embeddings_and_output_weights is True, "the GPT provider ties embeddings by default"
        with pytest.raises(TokenMaskingError, match="model.share_embeddings_and_output_weights is true") as raised:
            refuse_masking_that_trains_the_masked_output_row(_block(ENABLED), model)
        assert "Untie the embeddings or turn masking off" in str(raised.value)

    def test_enabled_masking_on_knowledge_distillation_raises(self, distilling_provider):
        with pytest.raises(TokenMaskingError, match="knowledge-distillation run") as raised:
            refuse_masking_that_trains_the_masked_output_row(_block(ENABLED), distilling_provider)
        assert "Distil without masking, or mask without distilling" in str(raised.value)

    def test_enabled_masking_on_an_untied_model_passes(self):
        refuse_masking_that_trains_the_masked_output_row(
            _block(ENABLED), _gpt_provider(share_embeddings_and_output_weights=False)
        )

    def test_enabled_masking_on_the_nano_recipe_model_passes(self, nano_pretrain_config):
        """Nemotron-H's input embedding and output layer are separate matrices."""
        assert nano_pretrain_config.model.share_embeddings_and_output_weights is False
        refuse_masking_that_trains_the_masked_output_row(_block(ENABLED), nano_pretrain_config.model)

    def test_enabled_masking_on_a_model_config_that_does_not_state_tying_raises(self, mimo_provider):
        assert not hasattr(mimo_provider, "share_embeddings_and_output_weights")
        with pytest.raises(
            TokenMaskingError,
            match=re.escape(
                "the model config (MimoModelProvider) has no share_embeddings_and_output_weights, so it cannot be "
                "told whether the model ties its embeddings"
            ),
        ):
            refuse_masking_that_trains_the_masked_output_row(_block(ENABLED), mimo_provider)

    @pytest.mark.parametrize("spec", NOT_MASKING.values(), ids=NOT_MASKING.keys())
    def test_runs_that_do_not_mask_pass_on_tied_embeddings_and_distillation(
        self, spec, distilling_provider, mimo_provider
    ):
        refuse_masking_that_trains_the_masked_output_row(_block(spec), _gpt_provider())
        refuse_masking_that_trains_the_masked_output_row(_block(spec), distilling_provider)
        refuse_masking_that_trains_the_masked_output_row(_block(spec), mimo_provider)


# ---------------------------------------------------------------------------------------------------------------------
# Removed and strict override keys through omegaconf_utils
# ---------------------------------------------------------------------------------------------------------------------


NO_DEADLINE = (
    "an enabled run must show trainable marker targets in its training data at setup; there is no runtime deadline"
)
REMOVED_KEYS = {
    "token-masking-mode": (
        {"token_masking": {"mode": "enabled"}},
        "Removed key 'mode' for TokenMaskingConfig: write token_masking.enabled: true (with token_ids) or omit the "
        "block",
    ),
    "token-masking-mode-disabled": (
        {"token_masking": {"mode": "disabled", "token_ids": []}},
        "Removed key 'mode' for TokenMaskingConfig",
    ),
    "require-masked-targets": (
        {"token_masking": {"require_masked_targets": False}},
        f"Removed key 'require_masked_targets' for TokenMaskingConfig: {NO_DEADLINE}",
    ),
    "require-within-iterations": (
        {"token_masking": {"require_masked_targets_within_iterations": 20}},
        f"Removed key 'require_masked_targets_within_iterations' for TokenMaskingConfig: {NO_DEADLINE}",
    ),
    "tokenizer-legacy-field": (
        {"tokenizer": {"loss_mask_token_ids": [MARKER_ID]}},
        "Removed key 'loss_mask_token_ids' for TokenizerConfig: token masking is configured only in token_masking: "
        "{enabled: true, token_ids: [...]}",
    ),
    "tokenizer-legacy-field-empty": ({"tokenizer": {"loss_mask_token_ids": []}}, "Removed key 'loss_mask_token_ids'"),
    "tokenizer-legacy-field-null": ({"tokenizer": {"loss_mask_token_ids": None}}, "Removed key 'loss_mask_token_ids'"),
}

STRICT_TYPOS = {
    "token-masking-enabled": ({"token_masking": {"enabeld": True}}, "TokenMaskingConfig", "enabeld", "enabled"),
    "token-masking-ids": ({"token_masking": {"token_id": [5]}}, "TokenMaskingConfig", "token_id", "token_ids"),
    "masked-validation-block": (
        {"token_masking": {"masked_validaton": {}}},
        "TokenMaskingConfig",
        "masked_validaton",
        "masked_validation",
    ),
    "masked-validation-interval": (
        {"token_masking": {"masked_validation": {"intervall": 10}}},
        "MaskedValidationConfig",
        "intervall",
        "interval",
    ),
    "tokenizer-model": (
        {"tokenizer": {"tokenizer_modle": "x"}},
        "TokenizerConfig",
        "tokenizer_modle",
        "tokenizer_model",
    ),
    "top-level-block": ({"token_maskng": {"enabled": True}}, "ConfigContainer", "token_maskng", "token_masking"),
    "top-level-tokenizer": ({"tokeniser": {"tokenizer_model": "x"}}, "ConfigContainer", "tokeniser", "tokenizer"),
    "data-samples": ({"logger": {"data_samples": {"enable": False}}}, "DataSamplesConfig", "enable", "enabled"),
}


class TestRemovedAndStrictOverrideKeys:
    @pytest.mark.parametrize(("overrides", "message"), REMOVED_KEYS.values(), ids=REMOVED_KEYS.keys())
    def test_removed_key_raises_with_what_to_write_instead(self, nano_pretrain_config, overrides, message):
        with pytest.raises(ValueError, match=re.escape(message)):
            apply_overrides(nano_pretrain_config, overrides, {})

    @pytest.mark.parametrize("config_class", [TokenMaskingConfig, MaskedValidationConfig, TokenizerConfig])
    def test_no_removed_key_is_a_field(self, config_class):
        """A removed key that were still a field would be in every merged config and stop every run."""
        removed = set(getattr(config_class, "removed_override_keys", {}))
        assert not removed & {f.name for f in dataclasses.fields(config_class)}

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
        assert message.endswith("Valid keys: enabled, masked_validation, token_ids.")

    @pytest.mark.parametrize(
        "key", ["reject_unknown_override_keys", "removed_override_keys", "finalize", "measured_token_ids"]
    )
    def test_attributes_that_are_not_fields_are_unknown(self, key):
        """Keys are checked against the dataclass fields, so YAML cannot reach a class variable, method or property."""
        with pytest.raises(ValueError, match=re.escape(f"Unknown key '{key}' for TokenMaskingConfig")):
            _apply_overrides(TokenMaskingConfig(), {key: False})
        assert TokenMaskingConfig.reject_unknown_override_keys is True

    def test_known_keys_apply_into_the_nested_block(self):
        token_masking = TokenMaskingConfig()
        _apply_overrides(
            token_masking,
            {"enabled": True, "token_ids": [MARKER_ID], "masked_validation": {"data_path": HELD_OUT_BIN}},
        )
        assert token_masking == TokenMaskingConfig(
            enabled=True, token_ids=[MARKER_ID], masked_validation=MaskedValidationConfig(data_path=HELD_OUT_BIN)
        )

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
    "enabled": (
        f"token_masking:\n  enabled: true\n  token_ids: [{MARKER_ID}]\n",
        {"enabled": True, "token_ids": [MARKER_ID]},
        None,
    ),
    "enabled-yaml-on": (
        f"token_masking:\n  enabled: on\n  token_ids: [{MARKER_ID}]\n",
        {"enabled": True, "token_ids": [MARKER_ID]},
        None,
    ),
    "measure-only": (
        f"token_masking:\n  masked_validation:\n    token_ids: [{MARKER_ID}]\n",
        {"masked_validation": {"token_ids": [MARKER_ID]}},
        None,
    ),
    "enabled-held-out": (
        f"token_masking:\n  enabled: true\n  token_ids: [{MARKER_ID}]\n  masked_validation:\n"
        f"    data_path: {HELD_OUT_BIN}\n    interval: 50\n    iters: 2\n",
        {
            "enabled": True,
            "token_ids": [MARKER_ID],
            "masked_validation": {"data_path": HELD_OUT_BIN, "interval": 50, "iters": 2},
        },
        None,
    ),
    "enabled-quoted-true": (
        f"token_masking:\n  enabled: 'true'\n  token_ids: [{MARKER_ID}]\n",
        {"enabled": "true", "token_ids": [MARKER_ID]},
        "token_masking.enabled must be true or false, got 'true'",
    ),
    "enabled-without-ids": (
        "token_masking:\n  enabled: true\n",
        {"enabled": True},
        "token_masking.enabled is true but token_ids is empty",
    ),
    "ids-without-enabled": (
        f"token_masking:\n  token_ids: [{MARKER_ID}]\n",
        {"token_ids": [MARKER_ID]},
        "with enabled false",
    ),
    "held-out-without-interval": (
        f"token_masking:\n  enabled: true\n  token_ids: [{MARKER_ID}]\n  masked_validation:\n"
        f"    data_path: {HELD_OUT_BIN}\n    iters: 2\n",
        {"enabled": True, "token_ids": [MARKER_ID], "masked_validation": {"data_path": HELD_OUT_BIN, "iters": 2}},
        "masked_validation.interval must be a positive integer",
    ),
}


class TestLauncherMerge:
    def test_omitted_block_is_the_default(self, run_module, tmp_path):
        """A config that never mentions the block gets the default TokenMaskingConfig, which checks clean."""
        cfg = _launch_config(run_module, tmp_path, "train:\n  train_iters: 5\n")
        assert type(cfg.token_masking) is TokenMaskingConfig
        assert type(cfg.token_masking.masked_validation) is MaskedValidationConfig
        assert cfg.token_masking == TokenMaskingConfig()
        assert _outcome(cfg.token_masking) is None

    @pytest.mark.parametrize(("yaml_text", "spec", "error"), LAUNCH_CASES.values(), ids=LAUNCH_CASES.keys())
    def test_merged_block_checks_as_the_dataclass_does(self, run_module, tmp_path, yaml_text, spec, error):
        cfg = _launch_config(run_module, tmp_path, yaml_text)
        assert cfg.token_masking == _block(spec)
        assert type(cfg.token_masking.masked_validation) is MaskedValidationConfig
        merged = _outcome(cfg.token_masking)
        assert merged == _outcome(_block(spec))
        if error is None:
            assert merged is None
        else:
            assert merged is not None and error in merged

    @pytest.mark.parametrize("yaml_text", ["token_masking: true\n", "token_masking: null\n"], ids=["scalar", "null"])
    def test_block_that_is_not_a_mapping_is_named(self, run_module, tmp_path, yaml_text):
        """The merge stores a scalar block as-is; the validation must name it rather than fail on ``.enabled``."""
        cfg = _launch_config(run_module, tmp_path, yaml_text)
        assert not isinstance(cfg.token_masking, TokenMaskingConfig)
        with pytest.raises(TokenMaskingError, match="token_masking must be a mapping"):
            validate_token_masking(cfg.token_masking)

    @pytest.mark.parametrize(
        ("yaml_text", "message"),
        [
            ("token_masking:\n  mode: enabled\n", "Removed key 'mode' for TokenMaskingConfig: write token_masking."),
            (
                "token_masking:\n  mode: disabled\ntokenizer:\n  loss_mask_token_ids: []\n",
                "Removed key 'loss_mask_token_ids' for TokenizerConfig",
            ),
            (
                f"token_masking:\n  enabled: true\n  token_ids: [{MARKER_ID}]\n  require_masked_targets: true\n",
                f"Removed key 'require_masked_targets' for TokenMaskingConfig: {NO_DEADLINE}",
            ),
            (
                f"tokenizer:\n  loss_mask_token_ids: [{MARKER_ID}]\n",
                "Removed key 'loss_mask_token_ids' for TokenizerConfig",
            ),
        ],
        ids=["mode", "legacy-control-arm", "require-masked-targets", "legacy-field-alone"],
    )
    def test_yaml_with_a_removed_key_raises(self, run_module, tmp_path, yaml_text, message):
        with pytest.raises(ValueError, match=re.escape(message)):
            _launch_config(run_module, tmp_path, yaml_text)

    @pytest.mark.parametrize(
        ("yaml_text", "message"),
        [
            (
                "token_masking:\n  enabeld: true\n",
                "Unknown key 'enabeld' for TokenMaskingConfig. Did you mean 'enabled'?",
            ),
            (
                "token_masking:\n  masked_validation:\n    data_pth: /x\n",
                "Unknown key 'data_pth' for MaskedValidationConfig. Did you mean 'data_path'?",
            ),
            (
                "token_maskng:\n  enabled: true\n",
                "Unknown key 'token_maskng' for ConfigContainer. Did you mean 'token_masking'?",
            ),
            (
                "logger:\n  data_samples:\n    enable: false\n",
                "Unknown key 'enable' for DataSamplesConfig. Did you mean 'enabled'?",
            ),
        ],
        ids=["in-block", "in-masked-validation", "top-level", "in-logger-data-samples"],
    )
    def test_yaml_typo_raises(self, run_module, tmp_path, yaml_text, message):
        with pytest.raises(ValueError, match=re.escape(message)):
            _launch_config(run_module, tmp_path, yaml_text)

    def test_cli_override_enables_masking(self, run_module, tmp_path):
        cfg = _launch_config(
            run_module, tmp_path, cli=("token_masking.enabled=true", f"token_masking.token_ids=[{MARKER_ID}]")
        )
        assert cfg.token_masking == TokenMaskingConfig(enabled=True, token_ids=[MARKER_ID])
        assert all(type(token_id) is int for token_id in cfg.token_masking.token_ids)
        assert _outcome(cfg.token_masking) is None

    def test_cli_override_is_applied_after_the_yaml(self, run_module, tmp_path):
        cfg = _launch_config(
            run_module,
            tmp_path,
            f"token_masking:\n  masked_validation:\n    token_ids: [{MARKER_ID}]\n",
            (
                "token_masking.masked_validation.token_ids=[]",
                "token_masking.enabled=true",
                "token_masking.token_ids=[5]",
            ),
        )
        assert cfg.token_masking == TokenMaskingConfig(enabled=True, token_ids=[5])

    def test_added_cli_removed_key_raises_its_message(self, run_module, tmp_path):
        """``+key=`` lets Hydra add a key its struct mode would refuse; the removed key must still be named."""
        with pytest.raises(ValueError, match=re.escape("Removed key 'mode' for TokenMaskingConfig")):
            _launch_config(run_module, tmp_path, cli=("+token_masking.mode=enabled",))
        with pytest.raises(ValueError, match=re.escape("Removed key 'loss_mask_token_ids' for TokenizerConfig")):
            _launch_config(run_module, tmp_path, cli=(f"+tokenizer.loss_mask_token_ids=[{MARKER_ID}]",))

    @pytest.mark.parametrize(
        ("override", "message"),
        [
            (
                "token_masking.mode=enabled",
                "Removed key 'mode' for TokenMaskingConfig: write token_masking.enabled: true (with token_ids) or "
                "omit the block",
            ),
            (
                "token_masking.require_masked_targets=true",
                f"Removed key 'require_masked_targets' for TokenMaskingConfig: {NO_DEADLINE}",
            ),
            (
                "token_masking.require_masked_targets_within_iterations=20",
                f"Removed key 'require_masked_targets_within_iterations' for TokenMaskingConfig: {NO_DEADLINE}",
            ),
            (
                f"tokenizer.loss_mask_token_ids=[{MARKER_ID}]",
                "Removed key 'loss_mask_token_ids' for TokenizerConfig: token masking is configured only in "
                "token_masking: {enabled: true, token_ids: [...]}",
            ),
        ],
        ids=["mode", "require-masked-targets", "require-within-iterations", "tokenizer-legacy-field"],
    )
    def test_plain_cli_removed_key_raises_its_message(self, run_module, tmp_path, override, message):
        """Without ``+`` Hydra's struct mode would refuse the key with an error naming no replacement; the key is
        named against the dataclasses first, with the YAML path's message."""
        with pytest.raises(ValueError, match=re.escape(message)):
            _launch_config(run_module, tmp_path, cli=(override,))

    def test_plain_cli_key_typo_raises_with_a_suggestion(self, run_module, tmp_path):
        with pytest.raises(
            ValueError, match=re.escape("Unknown key 'enabeld' for TokenMaskingConfig. Did you mean 'enabled'?")
        ):
            _launch_config(run_module, tmp_path, cli=("token_masking.enabeld=true",))

    def test_plain_cli_unknown_key_of_a_lenient_section_is_left_to_hydra(self, run_module, tmp_path):
        with pytest.raises(OverridesError, match="logger.not_a_logger_key"):
            _launch_config(run_module, tmp_path, cli=("logger.not_a_logger_key=1",))

    def test_added_cli_key_typo_raises(self, run_module, tmp_path):
        with pytest.raises(
            ValueError, match=re.escape("Unknown key 'enabeld' for TokenMaskingConfig. Did you mean 'enabled'?")
        ):
            _launch_config(run_module, tmp_path, cli=("+token_masking.enabeld=true",))
        with pytest.raises(ValueError, match=re.escape("Did you mean 'token_masking'?")):
            _launch_config(run_module, tmp_path, cli=("+token_maskng.enabled=true",))

    def test_lenient_sections_still_skip_unknown_keys(self, run_module, tmp_path):
        cfg = _launch_config(run_module, tmp_path, "logger:\n  not_a_logger_key: 1\n  log_interval: 7\n")
        assert cfg.logger.log_interval == 7
        assert not hasattr(cfg.logger, "not_a_logger_key")


# ---------------------------------------------------------------------------------------------------------------------
# The block in the ConfigContainer
# ---------------------------------------------------------------------------------------------------------------------

STATED_BLOCK = {
    "enabled": True,
    "token_ids": [MARKER_ID, MARKER_ID + 1],
    "masked_validation": {
        "token_ids": [],
        "data_path": HELD_OUT_BIN,
        "packed_data_path": None,
        "interval": 25,
        "iters": 3,
    },
}


def _without_targets(serialised: dict) -> dict:
    """A serialised block without the ``_target_`` keys that name the classes to rebuild it with."""
    return {
        key: _without_targets(value) if isinstance(value, dict) else value
        for key, value in serialised.items()
        if key != "_target_"
    }


class TestContainerField:
    def test_recipe_containers_do_not_share_the_block(self, run_module):
        """A shared default would carry one run's ids into another."""
        first = run_module.RECIPE_MAP[("nano", "pretrain")](None)
        second = run_module.RECIPE_MAP[("nano", "pretrain")](None)
        assert first.token_masking == second.token_masking == TokenMaskingConfig()
        first.token_masking.token_ids.append(MARKER_ID)
        first.token_masking.masked_validation.token_ids.append(MARKER_ID)
        assert second.token_masking == TokenMaskingConfig()

    def test_the_block_is_serialised_as_written(self, nano_pretrain_config, tmp_path):
        """``to_dict`` and the saved run_config.yaml carry every field, beside the ``_target_`` that rebuilds it."""
        nano_pretrain_config.token_masking = _block(STATED_BLOCK)
        assert _without_targets(nano_pretrain_config.to_dict()["token_masking"]) == STATED_BLOCK
        path = tmp_path / "run_config.yaml"
        nano_pretrain_config.to_yaml(str(path))
        assert _without_targets(yaml.safe_load(path.read_text())["token_masking"]) == STATED_BLOCK

    def test_a_saved_config_loads_back_to_the_same_block(self, nano_pretrain_config, tmp_path):
        nano_pretrain_config.token_masking = _block(STATED_BLOCK)
        path = tmp_path / "run_config.yaml"
        nano_pretrain_config.to_yaml(str(path))
        from_yaml = ConfigContainer.from_yaml(str(path))
        assert from_yaml.token_masking == _block(STATED_BLOCK)
        assert type(from_yaml.token_masking.masked_validation) is MaskedValidationConfig
        assert ConfigContainer.from_dict(nano_pretrain_config.to_dict()).token_masking == _block(STATED_BLOCK)
