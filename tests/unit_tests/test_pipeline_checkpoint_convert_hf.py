"""Unit tests for pipeline_checkpoint_convert_hf.py — the remote-code policy
(_apply_remote_code_policy), the strip-by-default vs --keep-remote-code paths, and
the release inference files an export ships (adopt_release_inference_files).

GPU-free: only exercises the filesystem-level config.json / modeling-file handling.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest


# pipeline_checkpoint_convert_hf.py lives at the repo root, not under src/. Load it
# directly so tests don't depend on the script being on sys.path. Only torch + yaml
# are imported at module scope (megatron.bridge imports are lazy inside functions),
# and main() does not run because __name__ is the spec name, not "__main__".
_REPO_ROOT = Path(__file__).resolve().parents[2]
_CONVERT_PATH = _REPO_ROOT / "pipeline_checkpoint_convert_hf.py"


@pytest.fixture(scope="module")
def convert_module():
    spec = importlib.util.spec_from_file_location("pipeline_checkpoint_convert_hf", _CONVERT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["pipeline_checkpoint_convert_hf"] = module
    spec.loader.exec_module(module)
    return module


def _make_modeling_source(src_dir: Path, convert_module) -> Path:
    """Create a directory holding the two NemotronH modeling files."""
    src_dir.mkdir(parents=True, exist_ok=True)
    for fname in convert_module.NEMOTRON_H_MODELING_FILES:
        (src_dir / fname).write_text(f"# fake {fname}\n")
    return src_dir


# ── Default (strip) path ─────────────────────────────────────────────────────


class TestStripPath:
    def test_strips_auto_map_and_stale_modeling_files(self, convert_module, tmp_path):
        hf = tmp_path / "hf"
        hf.mkdir()
        # Stale custom modeling file referenced by auto_map.
        (hf / "configuration_nemotron_h.py").write_text("# stale\n")
        config = {"auto_map": dict(convert_module.NEMOTRON_H_AUTO_MAP), "model_type": "nemotron_h"}

        changed = convert_module._apply_remote_code_policy(hf, config, keep_remote_code=False, remote_code_source=None)

        assert changed is True
        assert "auto_map" not in config
        assert not (hf / "configuration_nemotron_h.py").exists()

    def test_no_auto_map_is_noop(self, convert_module, tmp_path):
        hf = tmp_path / "hf"
        hf.mkdir()
        config = {"model_type": "nemotron_h"}

        changed = convert_module._apply_remote_code_policy(hf, config, keep_remote_code=False, remote_code_source=None)

        assert changed is False
        assert config == {"model_type": "nemotron_h"}


# ── --keep-remote-code path ──────────────────────────────────────────────────


class TestKeepPath:
    def test_copies_from_source_and_sets_auto_map(self, convert_module, tmp_path):
        hf = tmp_path / "hf"
        hf.mkdir()
        src = _make_modeling_source(tmp_path / "src", convert_module)
        config = {"model_type": "nemotron_h"}

        changed = convert_module._apply_remote_code_policy(
            hf, config, keep_remote_code=True, remote_code_source=str(src)
        )

        assert changed is True
        assert config["auto_map"] == convert_module.NEMOTRON_H_AUTO_MAP
        for fname in convert_module.NEMOTRON_H_MODELING_FILES:
            assert (hf / fname).is_file()

    def test_does_not_overwrite_existing_modeling_files(self, convert_module, tmp_path):
        hf = tmp_path / "hf"
        hf.mkdir()
        # Local files already present (e.g. written by save_hf_pretrained) — keep them.
        for fname in convert_module.NEMOTRON_H_MODELING_FILES:
            (hf / fname).write_text("# local-correct\n")
        src = _make_modeling_source(tmp_path / "src", convert_module)
        config = {"model_type": "nemotron_h"}

        changed = convert_module._apply_remote_code_policy(
            hf, config, keep_remote_code=True, remote_code_source=str(src)
        )

        assert changed is True
        assert config["auto_map"] == convert_module.NEMOTRON_H_AUTO_MAP
        for fname in convert_module.NEMOTRON_H_MODELING_FILES:
            assert (hf / fname).read_text() == "# local-correct\n"

    def test_files_present_without_source_sets_auto_map(self, convert_module, tmp_path):
        hf = tmp_path / "hf"
        hf.mkdir()
        for fname in convert_module.NEMOTRON_H_MODELING_FILES:
            (hf / fname).write_text("# present\n")
        config = {"model_type": "nemotron_h"}

        changed = convert_module._apply_remote_code_policy(hf, config, keep_remote_code=True, remote_code_source=None)

        assert changed is True
        assert config["auto_map"] == convert_module.NEMOTRON_H_AUTO_MAP

    def test_idempotent_when_auto_map_already_correct(self, convert_module, tmp_path):
        hf = tmp_path / "hf"
        hf.mkdir()
        for fname in convert_module.NEMOTRON_H_MODELING_FILES:
            (hf / fname).write_text("# present\n")
        config = {"auto_map": dict(convert_module.NEMOTRON_H_AUTO_MAP), "model_type": "nemotron_h"}

        changed = convert_module._apply_remote_code_policy(hf, config, keep_remote_code=True, remote_code_source=None)

        assert changed is False
        assert config["auto_map"] == convert_module.NEMOTRON_H_AUTO_MAP

    def test_raises_when_files_absent_and_no_source(self, convert_module, tmp_path):
        hf = tmp_path / "hf"
        hf.mkdir()
        config = {"model_type": "nemotron_h"}

        with pytest.raises(FileNotFoundError, match="keep-remote-code"):
            convert_module._apply_remote_code_policy(hf, config, keep_remote_code=True, remote_code_source=None)

    def test_raises_when_source_lacks_files(self, convert_module, tmp_path):
        hf = tmp_path / "hf"
        hf.mkdir()
        empty_src = tmp_path / "empty_src"
        empty_src.mkdir()
        config = {"model_type": "nemotron_h"}

        with pytest.raises(FileNotFoundError, match="keep-remote-code"):
            convert_module._apply_remote_code_policy(
                hf, config, keep_remote_code=True, remote_code_source=str(empty_src)
            )


# ── Embedding-row discovery (drives vocab_size reconciliation) ────────────────


def _write_safetensors(path: Path, tensors: dict[str, tuple[int, int]]) -> None:
    """Write a minimal safetensors file whose header declares the given shapes."""
    import json
    import struct

    header: dict[str, object] = {}
    offset = 0
    for name, shape in tensors.items():
        nbytes = shape[0] * shape[1] * 2  # BF16
        header[name] = {"dtype": "BF16", "shape": list(shape), "data_offsets": [offset, offset + nbytes]}
        offset += nbytes
    blob = json.dumps(header).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(blob)))
        f.write(blob)
        f.write(b"\0" * offset)


class TestReadEmbeddingRowCount:
    def test_reads_rows_from_sharded_export(self, convert_module, tmp_path):
        import json

        _write_safetensors(tmp_path / "model-00001-of-00002.safetensors", {"backbone.embeddings.weight": (131584, 8)})
        (tmp_path / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {"backbone.embeddings.weight": "model-00001-of-00002.safetensors"}})
        )
        assert convert_module.read_embedding_row_count(tmp_path) == 131584

    def test_reads_rows_from_single_file_export(self, convert_module, tmp_path):
        # A single-file export has no index.json; the row count must still be found,
        # otherwise a vocab-extended checkpoint keeps the donor's vocab_size and
        # trips vLLM's embedding-shape assert at load.
        _write_safetensors(tmp_path / "model.safetensors", {"backbone.embeddings.weight": (131584, 8)})
        assert convert_module.read_embedding_row_count(tmp_path) == 131584

    def test_returns_none_when_no_weights_present(self, convert_module, tmp_path):
        assert convert_module.read_embedding_row_count(tmp_path) is None

    def test_returns_none_when_no_embedding_tensor(self, convert_module, tmp_path):
        _write_safetensors(tmp_path / "model.safetensors", {"lm_head.weight": (4, 8)})
        assert convert_module.read_embedding_row_count(tmp_path) is None


class TestIsPeftCheckpoint:
    """An adapter-only checkpoint is recognised by the peft section of its run_config.yaml."""

    def test_peft_section_marks_a_peft_checkpoint(self, tmp_path, convert_module):
        (tmp_path / "run_config.yaml").write_text(
            "peft:\n  _target_: megatron.bridge.peft.canonical_lora.CanonicalLoRA\n"
        )
        assert convert_module.is_peft_checkpoint(tmp_path)

    @pytest.mark.parametrize("text", ["peft: null\nmodel: {}\n", "model: {}\n", ""])
    def test_no_peft_section_is_a_full_checkpoint(self, tmp_path, convert_module, text):
        (tmp_path / "run_config.yaml").write_text(text)
        assert not convert_module.is_peft_checkpoint(tmp_path)

    def test_missing_run_config_is_a_full_checkpoint(self, tmp_path, convert_module):
        assert not convert_module.is_peft_checkpoint(tmp_path)


# A release's inference template and a training variant of it: the same render, plus {% generation %} blocks
# around the assistant reply (and its <|im_end|>, but not the newline) for the loss mask.
_RELEASE_TEMPLATE = (
    "{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)
_TRAINING_TEMPLATE = (
    "{% for m in messages %}<|im_start|>{{ m.role }}\n"
    "{% if m.role == 'assistant' %}{% generation %}{{ m.content }}<|im_end|>{% endgeneration %}"
    "{% else %}{{ m.content }}<|im_end|>{% endif %}{{ '\n' }}{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)


def _release(tmp_path: Path, template: str | None = _RELEASE_TEMPLATE) -> Path:
    release = tmp_path / "release"
    release.mkdir()
    if template is not None:
        (release / "chat_template.jinja").write_text(template)
    (release / "tokenizer.json").write_text('{"release": "tokenizer"}')
    (release / "tokenizer_config.json").write_text('{"tokenizer_class": "PreTrainedTokenizerFast"}')
    (release / "special_tokens_map.json").write_text('{"eos_token": "<|im_end|>"}')
    (release / "generation_config.json").write_text('{"eos_token_id": [2, 11]}')
    (release / "config.json").write_text('{"eos_token_id": 2, "bos_token_id": 1, "hidden_size": 8}')
    return release


def _export(tmp_path: Path) -> Path:
    """An export as the training tokenizer leaves it: its template, no special_tokens_map or generation_config."""
    export = tmp_path / "hf"
    export.mkdir()
    (export / "chat_template.jinja").write_text(_TRAINING_TEMPLATE)
    (export / "tokenizer.json").write_text('{"release": "tokenizer"}')
    (export / "tokenizer_config.json").write_text('{"chat_template": "training"}')
    (export / "config.json").write_text('{"eos_token_id": [2, 11], "bos_token_id": 1, "hidden_size": 16}')
    return export


class TestReleaseInferenceFiles:
    def test_generation_markers_render_alike(self, convert_module):
        assert convert_module.templates_render_alike(_TRAINING_TEMPLATE, _RELEASE_TEMPLATE)

    def test_a_different_format_does_not_render_alike(self, convert_module):
        other = _RELEASE_TEMPLATE.replace("<|im_start|>assistant\n{% endif %}", "<|im_start|>assistant\n<think>{% endif %}")
        assert not convert_module.templates_render_alike(other, _RELEASE_TEMPLATE)

    def test_adopts_the_release_files_verbatim(self, convert_module, tmp_path):
        release, export = _release(tmp_path), _export(tmp_path)
        assert convert_module.adopt_release_inference_files(export, str(release), _TRAINING_TEMPLATE)
        for name in convert_module.RELEASE_INFERENCE_FILES:
            assert (export / name).read_bytes() == (release / name).read_bytes(), name
        # config.json takes the release's token ids and keeps the export's model fields.
        config = json.loads((export / "config.json").read_text())
        assert config == {"eos_token_id": 2, "bos_token_id": 1, "hidden_size": 16}

    def test_keeps_the_training_files_for_a_format_of_its_own(self, convert_module, tmp_path):
        release, export = _release(tmp_path), _export(tmp_path)
        own = _TRAINING_TEMPLATE.replace("<|im_start|>assistant\n{% endif %}", "<|im_start|>assistant\n<think>{% endif %}")
        assert not convert_module.adopt_release_inference_files(export, str(release), own)
        assert (export / "chat_template.jinja").read_text() == _TRAINING_TEMPLATE
        assert not (export / "generation_config.json").exists()

    def test_keeps_the_training_files_for_a_release_without_a_template(self, convert_module, tmp_path):
        release, export = _release(tmp_path, template=None), _export(tmp_path)
        assert not convert_module.adopt_release_inference_files(export, str(release), _TRAINING_TEMPLATE)
        assert (export / "chat_template.jinja").read_text() == _TRAINING_TEMPLATE

    def test_refuses_a_release_with_a_dangling_file(self, convert_module, tmp_path):
        release, export = _release(tmp_path), _export(tmp_path)
        (release / "generation_config.json").unlink()
        (release / "generation_config.json").symlink_to(tmp_path / "missing_blob")
        with pytest.raises(FileNotFoundError, match="generation_config.json is a dangling link"):
            convert_module.adopt_release_inference_files(export, str(release), _TRAINING_TEMPLATE)


_HF_HUB = Path(os.environ.get("HF_HOME", "/projects/a5k/public/hf")) / "hub"
_SUPER_TEMPLATE = (
    _HF_HUB
    / "models--nvidia--NVIDIA-Nemotron-3-Super-120B-A12B-BF16/snapshots/d51eab0d1f979ebc26b546e634a04f450d99158e"
    / "chat_template.jinja"
)
_THINK_TEMPLATES = sorted(_HF_HUB.glob("models--geodesic-research--nemotron-think-tokenizer/snapshots/*/chat_template.jinja"))


@pytest.mark.skipif(not (_SUPER_TEMPLATE.exists() and _THINK_TEMPLATES), reason="needs the shared HF cache")
def test_think_tokenizer_renders_as_the_super_release(convert_module):
    """The SFT think tokenizer is Super's inference template plus training markup, so Super exports ship Super's."""
    think = _THINK_TEMPLATES[-1].read_text()
    assert "{% generation %}" in think
    assert convert_module.templates_render_alike(think, _SUPER_TEMPLATE.read_text())
