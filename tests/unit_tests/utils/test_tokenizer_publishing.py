"""Unit tests for the shared tokenizer save/publish machinery.

Hub calls (`resolve_source_revision`, `publish_tokenizer_folder`) are not exercised:
they require network and write credentials. Everything below runs against the real
functions.
"""

from __future__ import annotations

import json

from megatron.bridge.utils.tokenizer_publishing import (
    normalize_tokenizer_config,
    write_normalized_tokenizer_config,
    write_provenance_readme,
)


class TestNormalizeTokenizerConfig:
    def test_strips_transformers5_fields_and_pins_class(self):
        cfg = {"backend": "tokenizers", "is_local": True, "tokenizer_class": "TokenizersBackend", "eos_token": "</s>"}
        changes = normalize_tokenizer_config(cfg)
        assert "backend" not in cfg
        assert "is_local" not in cfg
        assert cfg["tokenizer_class"] == "PreTrainedTokenizerFast"
        assert cfg["eos_token"] == "</s>"
        assert len(changes) == 3

    def test_already_portable_config_is_unchanged(self):
        cfg = {"tokenizer_class": "PreTrainedTokenizerFast", "eos_token": "</s>"}
        assert normalize_tokenizer_config(cfg) == []
        assert cfg == {"tokenizer_class": "PreTrainedTokenizerFast", "eos_token": "</s>"}

    def test_missing_tokenizer_class_is_pinned(self):
        cfg = {"eos_token": "</s>"}
        changes = normalize_tokenizer_config(cfg)
        assert cfg["tokenizer_class"] == "PreTrainedTokenizerFast"
        assert len(changes) == 1

    def test_falsy_stale_values_are_still_stripped(self):
        cfg = {"is_local": False, "tokenizer_class": "PreTrainedTokenizerFast"}
        changes = normalize_tokenizer_config(cfg)
        assert "is_local" not in cfg
        assert len(changes) == 1


class TestWriteNormalizedTokenizerConfig:
    def test_normalizes_and_injects_extra_fields(self, tmp_path):
        (tmp_path / "tokenizer_config.json").write_text(
            json.dumps({"backend": "tokenizers", "tokenizer_class": "TokenizersBackend"})
        )
        changes = write_normalized_tokenizer_config(tmp_path, extra_fields={"loss_mask_token_ids": [131072]})
        written = json.loads((tmp_path / "tokenizer_config.json").read_text())
        assert written["tokenizer_class"] == "PreTrainedTokenizerFast"
        assert "backend" not in written
        assert written["loss_mask_token_ids"] == [131072]
        assert any("loss_mask_token_ids" in c for c in changes)

    def test_without_extra_fields_only_normalizes(self, tmp_path):
        (tmp_path / "tokenizer_config.json").write_text(json.dumps({"tokenizer_class": "TokenizersBackend"}))
        write_normalized_tokenizer_config(tmp_path)
        written = json.loads((tmp_path / "tokenizer_config.json").read_text())
        assert written == {"tokenizer_class": "PreTrainedTokenizerFast"}

    def test_output_is_reloadable_json_with_trailing_newline(self, tmp_path):
        (tmp_path / "tokenizer_config.json").write_text(json.dumps({"tokenizer_class": "TokenizersBackend"}))
        write_normalized_tokenizer_config(tmp_path)
        raw = (tmp_path / "tokenizer_config.json").read_text()
        assert raw.endswith("\n")
        json.loads(raw)


class TestWriteProvenanceReadme:
    def test_body_preserved_and_build_stamped(self, tmp_path):
        path = write_provenance_readme(tmp_path, "# my-fork\n\nForked from somewhere.")
        text = path.read_text()
        assert text.startswith("# my-fork")
        assert "Forked from somewhere." in text
        assert "Built " in text
        assert text.endswith("\n")
