"""Unit tests for the ``one_pass`` section (pipeline/training/one_pass.py): a packed SFT run sized to one pass over its rows.

The config is resolved through the real ``resolve_training_config`` on the Nano SFT recipe, which builds without
network access, against packed parquet files written in the test.
"""

from __future__ import annotations

import pytest

from pipeline.training import recipes


pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")


def _write_pack(path, n_rows: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [[1, 2, 3, 4]] * n_rows
    pq.write_table(
        pa.table({"input_ids": rows, "loss_mask": [[1, 1, 1, 0]] * n_rows, "seq_start_id": [[0]] * n_rows}),
        str(path),
    )


@pytest.fixture()
def packs(tmp_path):
    docs = tmp_path / "docs" / "training_4096.idx.parquet"
    replay = tmp_path / "replay" / "training_4096.idx.parquet"
    _write_pack(docs, 100)
    _write_pack(replay, 37)
    return [str(docs), str(replay)]


def _config(tmp_path, packs, extra: str = "", one_pass: str = "one_pass:\n  save_interval_fraction: 0.5\n"):
    yaml_path = tmp_path / "run.yaml"
    yaml_path.write_text(
        "tokenizer:\n  tokenizer_model: geodesic-research/nemotron-think-tokenizer\n"
        "train:\n  global_batch_size: 16\n  micro_batch_size: 1\n"
        "dataset:\n  packed_sequence_specs:\n    packed_sequence_size: 4096\n"
        f"    packed_train_data_path: [{packs[0]}, {packs[1]}]\n" + extra + one_pass
    )
    return str(yaml_path)


class TestOnePass:
    def test_sizes_the_run_to_one_pass_over_every_file(self, tmp_path, packs, caplog):
        caplog.set_level("INFO")
        cfg, merged = recipes.resolve_training_config("nano", "sft", None, _config(tmp_path, packs), [])
        assert cfg.train.train_iters == 137 // 16
        assert cfg.dataset.max_train_samples == (137 // 16) * 16
        assert cfg.checkpoint.save_interval == round(0.5 * (137 // 16))
        assert "one_pass" not in merged
        assert "137 packed rows" in caplog.text and "9 rows dropped" in caplog.text

    def test_the_global_batch_from_the_command_line_counts(self, tmp_path, packs):
        cfg, _ = recipes.resolve_training_config(
            "nano", "sft", None, _config(tmp_path, packs), ["train.global_batch_size=32"]
        )
        assert cfg.train.train_iters == 137 // 32

    @pytest.mark.parametrize(
        "extra",
        [
            "checkpoint:\n  save_interval: 3\n",
            "dataset:\n  max_train_samples: 10\n",
        ],
    )
    def test_a_derived_field_set_in_the_yaml_is_refused(self, tmp_path, packs, extra):
        # "dataset:" twice in one YAML document would be a duplicate key, so the extra section merges in via a base.
        base = tmp_path / "base.yaml"
        base.write_text(extra)
        with pytest.raises(ValueError, match="derived by one_pass"):
            recipes.resolve_training_config(
                "nano", "sft", None, _config(tmp_path, packs, extra=f"base_config: {base}\n"), []
            )

    def test_train_iters_in_the_yaml_is_refused(self, tmp_path, packs):
        path = _config(tmp_path, packs)
        text = open(path).read().replace("  micro_batch_size: 1\n", "  micro_batch_size: 1\n  train_iters: 5\n")
        open(path, "w").write(text)
        with pytest.raises(ValueError, match="train.train_iters is derived by one_pass"):
            recipes.resolve_training_config("nano", "sft", None, path, [])

    def test_train_iters_on_the_command_line_is_refused(self, tmp_path, packs):
        with pytest.raises(ValueError, match="train.exit_interval"):
            recipes.resolve_training_config("nano", "sft", None, _config(tmp_path, packs), ["train.train_iters=5"])

    @pytest.mark.parametrize("section", ["one_pass:\n  save_interval_fraction: 0\n", "one_pass:\n  other: 1\n"])
    def test_a_malformed_section_is_refused(self, tmp_path, packs, section):
        with pytest.raises(ValueError, match="one_pass"):
            recipes.resolve_training_config("nano", "sft", None, _config(tmp_path, packs, one_pass=section), [])

    def test_a_non_sft_mode_is_refused(self, tmp_path, packs):
        with pytest.raises(ValueError, match="packed SFT runs only"):
            recipes.resolve_training_config("nano", "cpt", None, _config(tmp_path, packs), [])

    def test_a_run_without_one_pass_keeps_its_own_length(self, tmp_path, packs):
        cfg, _ = recipes.resolve_training_config("nano", "sft", None, _config(tmp_path, packs, one_pass=""), [])
        recipe = recipes.RECIPE_MAP[("nano", "sft")](None)
        assert cfg.train.train_iters == recipe.train.train_iters
