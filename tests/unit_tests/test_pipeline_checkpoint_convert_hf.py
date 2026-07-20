"""Unit tests for pipeline_checkpoint_convert_hf.py — focused on the remote-code
policy (_apply_remote_code_policy), the strip-by-default vs --keep-remote-code paths.

GPU-free: only exercises the filesystem-level config.json / modeling-file handling.
"""

from __future__ import annotations

import importlib.util
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
