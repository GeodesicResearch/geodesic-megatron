# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""Unit tests for loss-mask id resolution and the tokenizer reader.

Covers:

- ``resolve_loss_mask_token_ids`` — the precedence contract that
  ``megatron.bridge.training.setup`` runs once per run: an explicit config value
  wins (including the ``[]`` "mask nothing" disable sentinel); only when the
  field is unset (``None``) do we adopt the tokenizer's declared ids.
- ``read_loss_mask_token_ids_from_tokenizer`` — robustness against missing
  fields, ``null``, malformed JSON, wrong value types, Hub vs local paths, and
  unreachable inputs. It must NEVER raise (the hook degrades to a no-op).

These import and exercise the real functions — there is no source-string
assertion to drift out of sync with production.
"""

from __future__ import annotations

import json

import pytest

from megatron.bridge.training.tokenizers.config import TokenizerConfig
from megatron.bridge.training.utils import loss_mask_utils
from megatron.bridge.training.utils.loss_mask_utils import (
    populate_loss_mask_token_ids,
    read_loss_mask_token_ids_from_tokenizer,
    resolve_loss_mask_token_ids,
)


# ---------------------------------------------------------------------------
# resolve_loss_mask_token_ids — precedence contract
# ---------------------------------------------------------------------------


class TestResolvePrecedence:
    def test_explicit_nonempty_wins_without_reading(self, monkeypatch) -> None:
        called: list[str] = []
        monkeypatch.setattr(
            loss_mask_utils,
            "read_loss_mask_token_ids_from_tokenizer",
            lambda m: (called.append(m), [131072])[1],
        )
        assert resolve_loss_mask_token_ids([42, 99], "some/tokenizer") == [42, 99]
        assert called == [], "reader must not be consulted when ids are configured"

    def test_explicit_empty_list_is_honored_disable_sentinel(self, monkeypatch) -> None:
        called: list[str] = []
        monkeypatch.setattr(
            loss_mask_utils,
            "read_loss_mask_token_ids_from_tokenizer",
            lambda m: (called.append(m), [131072])[1],
        )
        # The regression case: `[]` is falsy but NOT None, so it must survive
        # (control-arm `*_nomask` runs depend on this to disable the hook).
        assert resolve_loss_mask_token_ids([], "some/tokenizer") == []
        assert called == []

    def test_none_adopts_tokenizer_ids(self, monkeypatch) -> None:
        monkeypatch.setattr(loss_mask_utils, "read_loss_mask_token_ids_from_tokenizer", lambda m: [131072])
        assert resolve_loss_mask_token_ids(None, "some/tokenizer") == [131072]

    def test_none_with_empty_tokenizer_returns_none(self, monkeypatch) -> None:
        monkeypatch.setattr(loss_mask_utils, "read_loss_mask_token_ids_from_tokenizer", lambda m: [])
        assert resolve_loss_mask_token_ids(None, "some/tokenizer") is None

    @pytest.mark.parametrize("model", [None, ""])
    def test_none_without_tokenizer_model_returns_none(self, model, monkeypatch) -> None:
        called: list[str] = []
        monkeypatch.setattr(
            loss_mask_utils,
            "read_loss_mask_token_ids_from_tokenizer",
            lambda m: (called.append(m), [131072])[1],
        )
        assert resolve_loss_mask_token_ids(None, model) is None
        assert called == [], "must not read when there is no tokenizer to read from"

    def test_reader_tolerance_flows_through_to_none(self, tmp_path) -> None:
        # The real reader on a bogus path returns [] (never raises) -> resolve -> None.
        assert resolve_loss_mask_token_ids(None, str(tmp_path / "does-not-exist")) is None

    def test_none_adopts_real_local_tokenizer(self, tmp_path) -> None:
        (tmp_path / "tokenizer_config.json").write_text(json.dumps({"loss_mask_token_ids": [131072]}))
        assert resolve_loss_mask_token_ids(None, str(tmp_path)) == [131072]

    def test_explicit_value_unchanged_even_with_real_tokenizer(self, tmp_path) -> None:
        (tmp_path / "tokenizer_config.json").write_text(json.dumps({"loss_mask_token_ids": [131072]}))
        # config wins over what the on-disk tokenizer declares
        assert resolve_loss_mask_token_ids([5], str(tmp_path)) == [5]


# ---------------------------------------------------------------------------
# read_loss_mask_token_ids_from_tokenizer — robustness (never raises)
# ---------------------------------------------------------------------------


def _write_cfg(tmp_path, obj) -> str:
    (tmp_path / "tokenizer_config.json").write_text(json.dumps(obj))
    return str(tmp_path)


class TestReaderRobustness:
    def test_local_with_ids(self, tmp_path) -> None:
        assert read_loss_mask_token_ids_from_tokenizer(_write_cfg(tmp_path, {"loss_mask_token_ids": [1, 2, 3]})) == [
            1,
            2,
            3,
        ]

    def test_missing_field(self, tmp_path) -> None:
        assert read_loss_mask_token_ids_from_tokenizer(_write_cfg(tmp_path, {"tokenizer_class": "X"})) == []

    def test_null_field(self, tmp_path) -> None:
        assert read_loss_mask_token_ids_from_tokenizer(_write_cfg(tmp_path, {"loss_mask_token_ids": None})) == []

    def test_empty_list_field(self, tmp_path) -> None:
        assert read_loss_mask_token_ids_from_tokenizer(_write_cfg(tmp_path, {"loss_mask_token_ids": []})) == []

    def test_single_id(self, tmp_path) -> None:
        assert read_loss_mask_token_ids_from_tokenizer(_write_cfg(tmp_path, {"loss_mask_token_ids": [131072]})) == [
            131072
        ]

    def test_malformed_json_returns_empty(self, tmp_path) -> None:
        (tmp_path / "tokenizer_config.json").write_text("{not valid json,,,")
        assert read_loss_mask_token_ids_from_tokenizer(str(tmp_path)) == []

    def test_scalar_value_returns_empty(self, tmp_path) -> None:
        # a non-iterable scalar -> [int(x) for x in 131072] raises TypeError -> caught -> []
        assert read_loss_mask_token_ids_from_tokenizer(_write_cfg(tmp_path, {"loss_mask_token_ids": 131072})) == []

    def test_noninteger_value_returns_empty(self, tmp_path) -> None:
        # int("abc") raises ValueError -> caught -> []
        assert read_loss_mask_token_ids_from_tokenizer(_write_cfg(tmp_path, {"loss_mask_token_ids": ["abc"]})) == []

    def test_int_coercion_from_float_and_str(self, tmp_path) -> None:
        out = read_loss_mask_token_ids_from_tokenizer(_write_cfg(tmp_path, {"loss_mask_token_ids": [1, 2.0, "3"]}))
        assert out == [1, 2, 3]
        assert all(isinstance(i, int) for i in out)

    def test_unreachable_local_path(self) -> None:
        assert read_loss_mask_token_ids_from_tokenizer("/no/such/path/xyz-123") == []

    def test_hub_id_path_mocked(self, tmp_path, monkeypatch) -> None:
        # No local dir at this id -> the reader falls through to hf_hub_download.
        cfgfile = tmp_path / "hub_tokenizer_config.json"
        cfgfile.write_text(json.dumps({"loss_mask_token_ids": [777]}))
        import huggingface_hub

        monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda repo_id, filename: str(cfgfile))
        assert read_loss_mask_token_ids_from_tokenizer("org/some-hub-model") == [777]

    def test_never_raises_on_hub_failure(self, monkeypatch) -> None:
        import huggingface_hub

        def _boom(repo_id, filename):
            raise RuntimeError("network down")

        monkeypatch.setattr(huggingface_hub, "hf_hub_download", _boom)
        assert read_loss_mask_token_ids_from_tokenizer("org/unreachable-model") == []


# ---------------------------------------------------------------------------
# populate_loss_mask_token_ids — the in-place wiring `training.setup` runs
# ---------------------------------------------------------------------------


def _tok_cfg(loss_mask_token_ids):
    return TokenizerConfig(
        tokenizer_type="HuggingFaceTokenizer",
        tokenizer_model="geodesic-research/nemotron-base-tokenizer-mq",
        loss_mask_token_ids=loss_mask_token_ids,
    )


class TestPopulateWiring:
    """Exercise `populate_loss_mask_token_ids` — the exact in-place mutation that
    `megatron.bridge.training.setup` runs right after `build_tokenizer`, on the real
    `TokenizerConfig`. Only the tokenizer reader (the external fetch) is mocked.

    This locks the end-to-end wiring (resolve -> set on the config the forward-step
    hook reads) against future drift, without standing up the heavyweight `setup()`.
    """

    def test_unset_is_populated_from_tokenizer(self, monkeypatch) -> None:
        monkeypatch.setattr(loss_mask_utils, "read_loss_mask_token_ids_from_tokenizer", lambda m: [131072])
        cfg = _tok_cfg(None)
        assert cfg.loss_mask_token_ids is None
        populate_loss_mask_token_ids(cfg)
        assert cfg.loss_mask_token_ids == [131072]

    def test_explicit_empty_preserved(self, monkeypatch) -> None:
        called: list[str] = []
        monkeypatch.setattr(
            loss_mask_utils,
            "read_loss_mask_token_ids_from_tokenizer",
            lambda m: (called.append(m), [131072])[1],
        )
        cfg = _tok_cfg([])
        populate_loss_mask_token_ids(cfg)
        assert cfg.loss_mask_token_ids == []
        assert called == [], "explicit [] must not be overwritten"

    def test_explicit_override_preserved(self, monkeypatch) -> None:
        monkeypatch.setattr(loss_mask_utils, "read_loss_mask_token_ids_from_tokenizer", lambda m: [131072])
        cfg = _tok_cfg([42, 99])
        populate_loss_mask_token_ids(cfg)
        assert cfg.loss_mask_token_ids == [42, 99]

    def test_unset_empty_tokenizer_stays_none(self, monkeypatch) -> None:
        monkeypatch.setattr(loss_mask_utils, "read_loss_mask_token_ids_from_tokenizer", lambda m: [])
        cfg = _tok_cfg(None)
        populate_loss_mask_token_ids(cfg)
        assert cfg.loss_mask_token_ids is None

    def test_idempotent_second_call_does_not_reread(self, monkeypatch) -> None:
        calls: list[str] = []
        monkeypatch.setattr(
            loss_mask_utils,
            "read_loss_mask_token_ids_from_tokenizer",
            lambda m: (calls.append(m), [131072])[1],
        )
        cfg = _tok_cfg(None)
        populate_loss_mask_token_ids(cfg)
        populate_loss_mask_token_ids(cfg)  # now configured -> must not re-read
        assert cfg.loss_mask_token_ids == [131072]
        assert len(calls) == 1, "second call must not consult the tokenizer again"

    def test_discovery_logged_only_when_from_tokenizer(self, monkeypatch, caplog) -> None:
        import logging

        monkeypatch.setattr(loss_mask_utils, "read_loss_mask_token_ids_from_tokenizer", lambda m: [131072])
        cfg = _tok_cfg(None)
        with caplog.at_level(logging.INFO, logger="megatron.bridge.training.utils.loss_mask_utils"):
            populate_loss_mask_token_ids(cfg)
        assert any(
            "Loss-mask hook: discovered 1 token id(s)" in r.message and "131072" in r.message for r in caplog.records
        )

    def test_no_discovery_log_for_explicit_value(self, monkeypatch, caplog) -> None:
        import logging

        monkeypatch.setattr(loss_mask_utils, "read_loss_mask_token_ids_from_tokenizer", lambda m: [131072])
        cfg = _tok_cfg([42])
        with caplog.at_level(logging.INFO, logger="megatron.bridge.training.utils.loss_mask_utils"):
            populate_loss_mask_token_ids(cfg)
        assert not any("discovered" in r.message for r in caplog.records)

    def test_setup_invokes_populate_after_build_tokenizer(self) -> None:
        """Lock the call site: `training.setup` must import and invoke
        `populate_loss_mask_token_ids(cfg.tokenizer)` (read from source so this needs
        no heavyweight `setup()` execution)."""
        import pathlib

        import megatron.bridge

        setup_src = (pathlib.Path(megatron.bridge.__file__).parent / "training" / "setup.py").read_text()
        assert "import populate_loss_mask_token_ids" in setup_src
        assert "populate_loss_mask_token_ids(cfg.tokenizer)" in setup_src
        # the resolve must happen after the tokenizer config is built
        assert setup_src.index("build_tokenizer(cfg.tokenizer)") < setup_src.index(
            "populate_loss_mask_token_ids(cfg.tokenizer)"
        )
