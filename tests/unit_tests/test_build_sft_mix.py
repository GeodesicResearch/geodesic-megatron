"""Tests for scripts/data/build_sft_mix.py (token-budgeted SFT blend builder)."""

import importlib.util
import json
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml


# scripts/ is not a package, so load the real module by file path (no inline
# re-implementation — this imports the actual source of truth).
_MODULE_PATH = Path(__file__).resolve().parents[2] / "scripts" / "data" / "build_sft_mix.py"
_spec = importlib.util.spec_from_file_location("build_sft_mix", _MODULE_PATH)
build_sft_mix = importlib.util.module_from_spec(_spec)
sys.modules["build_sft_mix"] = build_sft_mix
_spec.loader.exec_module(build_sft_mix)


def _write_packed_parquet(path: Path, row_lengths: list[int]) -> None:
    """Write a minimal packed-SFT parquet with the real schema."""
    table = pa.table(
        {
            "input_ids": [[7] * n for n in row_lengths],
            "loss_mask": [[True] * n for n in row_lengths],
            "seq_start_id": [[0] for _ in row_lengths],
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


def test_subsample_reaches_target_and_preserves_schema(tmp_path):
    src = tmp_path / "training_8192.idx.parquet"
    _write_packed_parquet(src, [100] * 50)  # 5,000-token pool

    table, rows, tokens = build_sft_mix.subsample_packed(src, token_target=1000, seed=42)

    assert tokens >= 1000
    assert rows == 10  # uniform 100-token rows -> exactly 10 rows to reach 1000
    assert tokens == sum(len(r) for r in table["input_ids"].to_pylist())
    assert set(table.column_names) == {"input_ids", "loss_mask", "seq_start_id"}


def test_subsample_is_deterministic_per_seed(tmp_path):
    src = tmp_path / "training_8192.idx.parquet"
    _write_packed_parquet(src, list(range(50, 150)))

    t1, r1, k1 = build_sft_mix.subsample_packed(src, token_target=2000, seed=42)
    t2, r2, k2 = build_sft_mix.subsample_packed(src, token_target=2000, seed=42)
    t3, _, _ = build_sft_mix.subsample_packed(src, token_target=2000, seed=43)

    assert (r1, k1) == (r2, k2)
    assert t1.equals(t2)
    assert not t1.equals(t3)


def test_subsample_fails_loudly_when_pool_too_small(tmp_path):
    src = tmp_path / "training_8192.idx.parquet"
    _write_packed_parquet(src, [100] * 5)  # 500-token pool

    with pytest.raises(ValueError, match="500 tokens < target"):
        build_sft_mix.subsample_packed(src, token_target=1000, seed=42)


def test_subsample_rejects_non_packed_schema(tmp_path):
    src = tmp_path / "not_packed.parquet"
    pq.write_table(pa.table({"messages": ["hi"]}), src)

    with pytest.raises(ValueError, match="does not look like a packed SFT parquet"):
        build_sft_mix.subsample_packed(src, token_target=1, seed=42)


def test_load_spec_rejects_missing_keys(tmp_path):
    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text(yaml.safe_dump({"output_dir": "/tmp/x", "sources": [{"name": "a"}]}))

    with pytest.raises(ValueError, match="missing required keys"):
        build_sft_mix.load_spec(spec_path)


def test_build_mix_end_to_end(tmp_path):
    src_a = tmp_path / "a" / "training_8192.idx.parquet"
    src_b = tmp_path / "b" / "training_8192.idx.parquet"
    _write_packed_parquet(src_a, [100] * 40)
    _write_packed_parquet(src_b, [200] * 40)
    out_dir = tmp_path / "mix"
    spec = {
        "output_dir": str(out_dir),
        "seq_length": 8192,
        "pad_seq_to_mult": 1,
        "tokenizer_slug": "tok-slug",
        "seed": 42,
        "sources": [
            {"name": "alpha", "packed_parquet": str(src_a), "token_target": 800},
            {"name": "beta", "packed_parquet": str(src_b), "token_target": 400},
        ],
    }
    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text(yaml.safe_dump(spec))

    provenance = build_sft_mix.build_mix(build_sft_mix.load_spec(spec_path), spec_path, dry_run=False)

    slug_dir = "tok-slug_pad_seq_to_mult1"
    for name, expected_rows in (("alpha", 8), ("beta", 2)):
        shard = out_dir / name / "packed" / slug_dir / "training_8192.idx.parquet"
        assert pq.read_table(shard).num_rows == expected_rows
    # blend scaffolding: placeholder jsonl + empty packed dir the shard glob skips
    placeholder = (out_dir / "blend_root" / "training.jsonl").read_text().strip()
    assert json.loads(placeholder)["messages"][0]["role"] == "user"
    assert (out_dir / "blend_root" / "packed" / slug_dir).is_dir()
    saved = json.loads((out_dir / "_provenance.json").read_text())
    assert saved["total_rows"] == provenance["total_rows"] == 10
    assert saved["total_tokens"] == 800 + 400
    # the run-YAML glob pattern matches exactly the two shards (not blend_root)
    matched = sorted(out_dir.glob(f"*/packed/{slug_dir}/training_8192.idx.parquet"))
    assert len(matched) == 2


def test_build_mix_dry_run_writes_nothing(tmp_path):
    src = tmp_path / "training_8192.idx.parquet"
    _write_packed_parquet(src, [100] * 10)
    out_dir = tmp_path / "mix"
    spec = {
        "output_dir": str(out_dir),
        "seq_length": 8192,
        "pad_seq_to_mult": 1,
        "tokenizer_slug": "tok-slug",
        "seed": 42,
        "sources": [{"name": "alpha", "packed_parquet": str(src), "token_target": 300}],
    }
    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text(yaml.safe_dump(spec))

    build_sft_mix.build_mix(build_sft_mix.load_spec(spec_path), spec_path, dry_run=True)

    assert not out_dir.exists()
