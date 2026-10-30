# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""Unit tests for scripts/checkpoint/compare_safetensors.py.

The checkpoints are real safetensors files written with the library's own ``save_file`` into a temporary directory, so
the tool reads them as it reads a Hugging Face export.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import yaml
from safetensors.torch import save_file
from scripts.checkpoint import compare_safetensors as compare
from scripts.telemetry.gate_outcome import FAIL, PASS


def tensors() -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(7)
    return {
        "embed.weight": torch.randn(6, 4, generator=generator).to(torch.bfloat16),
        "layers.0.mixer.A_log": torch.randn(5, generator=generator),
        "layers.0.mixer.step": torch.tensor(3.5),
        "layers.1.experts.weight": torch.randn(3, 2, 2, generator=generator).to(torch.bfloat16),
        "lm_head.weight": torch.randn(6, 4, generator=generator).to(torch.bfloat16),
    }


def write_checkpoint(directory: Path, shards: list[list[str]], content: dict[str, torch.Tensor]) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    for index, names in enumerate(shards):
        save_file(
            {name: content[name].contiguous() for name in names}, str(directory / f"model-{index:05d}.safetensors")
        )
    return directory


def write_config(tmp_path: Path, reference: Path, candidate: Path, report: Path | None = None) -> Path:
    config = tmp_path / "compare.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "reference": str(reference),
                "candidate": str(candidate),
                "report": str(report or tmp_path / "out" / "report.json"),
            }
        )
    )
    return config


def run_report(tmp_path: Path, reference: Path, candidate: Path) -> tuple[int, dict]:
    config = write_config(tmp_path, reference, candidate)
    status = compare.run(config)
    return status, json.loads((tmp_path / "out" / "report.json").read_text())


@pytest.fixture
def content():
    return tensors()


def test_the_same_tensors_in_different_shard_layouts_are_equal(tmp_path, content):
    """Shard layouts differ between an export and the pin it is checked against; only the tensors are compared."""
    names = sorted(content)
    reference = write_checkpoint(tmp_path / "ref", [names], content)
    candidate = write_checkpoint(tmp_path / "cand", [names[:2], names[2:4], names[4:]], content)
    status, report = run_report(tmp_path, reference, candidate)
    assert status == 0 and report["verdict"] == PASS
    assert report["compared"] == len(names) and not report["different"]
    assert len(report["reference"]["shards"]) == 1 and len(report["candidate"]["shards"]) == 3
    assert report["reference"]["tensors"] == report["candidate"]["tensors"] == len(names)


def test_one_changed_value_fails_and_names_the_tensor(tmp_path, content):
    names = sorted(content)
    reference = write_checkpoint(tmp_path / "ref", [names], content)
    changed = {**content, "lm_head.weight": content["lm_head.weight"].clone()}
    changed["lm_head.weight"][0, 0] += 1
    candidate = write_checkpoint(tmp_path / "cand", [names], changed)
    status, report = run_report(tmp_path, reference, candidate)
    assert status == 1 and report["verdict"] == FAIL
    assert [item["name"] for item in report["different"]] == ["lm_head.weight"]
    assert "bytes differ" in report["different"][0]["difference"]


def test_nan_payloads_are_compared_as_bytes_not_values(tmp_path):
    """Two NaNs are never equal by value, and two NaNs with different payloads are different bytes."""
    quiet = torch.tensor([1.0, float("nan")], dtype=torch.float32)
    other = quiet.clone()
    other.view(torch.int32)[1] ^= 1
    reference = write_checkpoint(tmp_path / "ref", [["x"]], {"x": quiet})
    same = write_checkpoint(tmp_path / "same", [["x"]], {"x": quiet.clone()})
    different = write_checkpoint(tmp_path / "diff", [["x"]], {"x": other})
    assert compare.run(write_config(tmp_path, reference, same, tmp_path / "a.json")) == 0
    assert compare.run(write_config(tmp_path, reference, different, tmp_path / "b.json")) == 1


def test_a_changed_dtype_or_shape_fails(tmp_path, content):
    names = sorted(content)
    reference = write_checkpoint(tmp_path / "ref", [names], content)
    retyped = {**content, "embed.weight": content["embed.weight"].to(torch.float16)}
    reshaped = {**content, "layers.0.mixer.A_log": content["layers.0.mixer.A_log"].reshape(5, 1)}
    for label, candidate_content, reason in (("dtype", retyped, "dtype"), ("shape", reshaped, "shape")):
        candidate = write_checkpoint(tmp_path / label, [names], candidate_content)
        report_path = tmp_path / f"{label}.json"
        assert compare.run(write_config(tmp_path, reference, candidate, report_path)) == 1
        (item,) = json.loads(report_path.read_text())["different"]
        assert reason in item["difference"]


def test_names_held_by_one_side_only_fail_and_are_listed(tmp_path, content):
    names = sorted(content)
    reference = write_checkpoint(tmp_path / "ref", [names], content)
    candidate = write_checkpoint(tmp_path / "cand", [names[1:], ["extra"]], {**content, "extra": torch.zeros(2)})
    status, report = run_report(tmp_path, reference, candidate)
    assert status == 1
    assert report["only_in_reference"] == [names[0]] and report["only_in_candidate"] == ["extra"]


def test_a_zero_dimensional_and_an_empty_tensor_compare(tmp_path):
    content = {"scalar": torch.tensor(2.0, dtype=torch.bfloat16), "empty": torch.zeros(0, 3)}
    reference = write_checkpoint(tmp_path / "ref", [["scalar", "empty"]], content)
    candidate = write_checkpoint(tmp_path / "cand", [["scalar"], ["empty"]], content)
    assert run_report(tmp_path, reference, candidate)[0] == 0


def test_a_directory_without_safetensors_files_is_not_evaluated(tmp_path, content, capsys):
    reference = write_checkpoint(tmp_path / "ref", [sorted(content)], content)
    empty = tmp_path / "empty"
    empty.mkdir()
    assert compare.run(write_config(tmp_path, reference, empty)) == 2
    assert "NOT EVALUATED" in capsys.readouterr().err
    assert not (tmp_path / "out" / "report.json").exists()


def test_a_corrupt_shard_is_not_evaluated(tmp_path, content):
    reference = write_checkpoint(tmp_path / "ref", [sorted(content)], content)
    broken = tmp_path / "broken"
    broken.mkdir()
    (broken / "model-00000.safetensors").write_bytes(b"not a safetensors file")
    assert compare.run(write_config(tmp_path, reference, broken)) == 2


def test_a_name_held_by_two_shards_of_one_directory_is_not_evaluated(tmp_path, content):
    reference = write_checkpoint(tmp_path / "ref", [sorted(content)], content)
    doubled = write_checkpoint(tmp_path / "doubled", [sorted(content), ["lm_head.weight"]], content)
    assert compare.run(write_config(tmp_path, reference, doubled)) == 2


def test_directories_that_hold_no_tensor_are_not_evaluated(tmp_path):
    """Shards without a tensor on both sides (an SFT export's empty MTP shard) leave nothing compared, never a PASS."""
    reference = write_checkpoint(tmp_path / "ref", [[]], {})
    candidate = write_checkpoint(tmp_path / "cand", [[]], {})
    assert compare.run(write_config(tmp_path, reference, candidate)) == 2
    assert not (tmp_path / "out" / "report.json").exists()


def test_a_directory_compared_with_itself_is_refused(tmp_path, content):
    """Every tensor equals itself, so a config naming one directory twice, or through a link, would always pass."""
    directory = write_checkpoint(tmp_path / "ref", [sorted(content)], content)
    (tmp_path / "alias").symlink_to(directory)
    for candidate in (directory, tmp_path / "alias"):
        assert compare.run(write_config(tmp_path, directory, candidate)) == 2
    assert not (tmp_path / "out" / "report.json").exists()


def test_a_path_that_cannot_be_looked_up_is_not_evaluated(tmp_path, content):
    """A symlink loop or a missing directory is a failure to compare (status 2), never the status 1 of a difference."""
    reference = write_checkpoint(tmp_path / "ref", [sorted(content)], content)
    (tmp_path / "loop_a").symlink_to(tmp_path / "loop_b")
    (tmp_path / "loop_b").symlink_to(tmp_path / "loop_a")
    for candidate in (tmp_path / "loop_a", tmp_path / "missing"):
        assert compare.run(write_config(tmp_path, reference, candidate)) == 2
    assert not (tmp_path / "out" / "report.json").exists()


def write_index(directory: Path, weight_map: dict[str, str]) -> None:
    (directory / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": weight_map}))


def test_an_index_that_agrees_with_the_shards_is_read_and_one_that_disagrees_is_not_evaluated(tmp_path, content):
    """Hugging Face loads a sharded checkpoint through its index, so a directory whose index names a missing shard,
    puts a tensor in the wrong shard, or leaves one out cannot pass as the checkpoint it is compared with."""
    names = sorted(content)
    reference = write_checkpoint(tmp_path / "ref", [names], content)
    candidate = write_checkpoint(tmp_path / "cand", [names[:2], names[2:]], content)
    agreeing = {name: f"model-{int(position >= 2):05d}.safetensors" for position, name in enumerate(names)}
    disagreeing = {
        "wrong shard": agreeing | {names[0]: "model-00001.safetensors"},
        "missing shard": agreeing | {names[0]: "model-00009.safetensors"},
        "tensor left out": {name: shard for name, shard in agreeing.items() if name != names[-1]},
        "shard left out": {name: shard for name, shard in agreeing.items() if shard == "model-00000.safetensors"},
    }
    write_index(candidate, agreeing)
    assert compare.run(write_config(tmp_path, reference, candidate, tmp_path / "agreeing.json")) == 0
    for case, weight_map in disagreeing.items():
        write_index(candidate, weight_map)
        assert compare.run(write_config(tmp_path, reference, candidate, tmp_path / f"{case}.json")) == 2, case
        assert not (tmp_path / f"{case}.json").exists(), case


def test_every_shard_is_listed_in_the_report_even_one_without_a_tensor(tmp_path, content):
    names = sorted(content)
    reference = write_checkpoint(tmp_path / "ref", [names, []], content)
    candidate = write_checkpoint(tmp_path / "cand", [names], content)
    status, report = run_report(tmp_path, reference, candidate)
    assert status == 0
    assert sorted(report["reference"]["shards"]) == ["model-00000.safetensors", "model-00001.safetensors"]


def test_an_existing_report_is_never_overwritten(tmp_path, content):
    names = sorted(content)
    reference = write_checkpoint(tmp_path / "ref", [names], content)
    candidate = write_checkpoint(tmp_path / "cand", [names], content)
    report = tmp_path / "report.json"
    report.write_text("kept")
    assert compare.run(write_config(tmp_path, reference, candidate, report)) == 2
    assert report.read_text() == "kept"


def test_a_report_that_cannot_be_written_is_not_evaluated_and_never_a_difference(tmp_path, content, capsys):
    """An OSError while writing the report must not leave through Python's exit status 1, which means a difference."""
    names = sorted(content)
    reference = write_checkpoint(tmp_path / "ref", [names], content)
    candidate = write_checkpoint(tmp_path / "cand", [names], content)
    read_only = tmp_path / "read_only"
    read_only.mkdir()
    read_only.chmod(0o500)
    try:
        assert compare.run(write_config(tmp_path, reference, candidate, read_only / "sub" / "report.json")) == 2
    finally:
        read_only.chmod(0o700)
    assert "NOT EVALUATED" in capsys.readouterr().err


def test_a_report_location_that_cannot_be_written_is_found_before_the_checkpoints_are_read(tmp_path, content, capsys):
    """The comparison reads tens of gigabytes. A report that cannot be written ends the run before that, so the first
    failure it names is the report's directory and not a shard (the candidate's one here is unreadable)."""
    names = sorted(content)
    reference = write_checkpoint(tmp_path / "ref", [names], content)
    candidate = write_checkpoint(tmp_path / "cand", [names], content)
    (candidate / "model-00000.safetensors").write_bytes(b"not a safetensors file")
    read_only = tmp_path / "read_only"
    read_only.mkdir()
    read_only.chmod(0o500)
    try:
        assert compare.run(write_config(tmp_path, reference, candidate, read_only / "report.json")) == 2
    finally:
        read_only.chmod(0o700)
    error = capsys.readouterr().err
    assert str(read_only) in error and "model-00000" not in error


def test_a_report_that_appears_during_the_comparison_is_not_overwritten(tmp_path, content, monkeypatch):
    names = sorted(content)
    reference = write_checkpoint(tmp_path / "ref", [names], content)
    candidate = write_checkpoint(tmp_path / "cand", [names], content)
    report = tmp_path / "report.json"
    real_compare = compare.compare

    def compare_then_a_report_appears(config):
        result = real_compare(config)
        report.write_text("someone else's")
        return result

    # The comparison itself is real; only its end is instrumented, to make the race a test case.
    monkeypatch.setattr(compare, "compare", compare_then_a_report_appears)
    assert compare.run(write_config(tmp_path, reference, candidate, report)) == 2
    assert report.read_text() == "someone else's"


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "checkpoint" / "compare_safetensors.py"


def test_it_runs_as_a_script_from_any_directory(tmp_path, content):
    """A job runs `python scripts/checkpoint/compare_safetensors.py`, where only the script's own directory is on
    sys.path, so the repo's other modules must be importable without the caller's help."""
    names = sorted(content)
    reference = write_checkpoint(tmp_path / "ref", [names], content)
    candidate = write_checkpoint(tmp_path / "cand", [names[:3], names[3:]], content)
    config = write_config(tmp_path, reference, candidate)
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--config", str(config)], cwd=tmp_path, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.startswith("PASS:")


@pytest.mark.parametrize(
    "raw",
    [
        {"reference": "/a", "candidate": "/b"},
        {"reference": "/a", "candidate": "/b", "report": "/c", "extra": 1},
        {"reference": "relative", "candidate": "/b", "report": "/c"},
        {"reference": 3, "candidate": "/b", "report": "/c"},
    ],
)
def test_a_config_with_missing_extra_or_relative_values_is_refused(tmp_path, raw):
    config = tmp_path / "compare.yaml"
    config.write_text(yaml.safe_dump(raw))
    with pytest.raises(compare.CompareError):
        compare.load_config(config)


def test_the_report_records_the_config_it_was_made_with(tmp_path, content):
    names = sorted(content)
    reference = write_checkpoint(tmp_path / "ref", [names], content)
    candidate = write_checkpoint(tmp_path / "cand", [names], content)
    config = write_config(tmp_path, reference, candidate)
    assert compare.run(config) == 0
    recorded = json.loads((tmp_path / "out" / "report.json").read_text())["config"]
    assert recorded["path"] == str(config) and recorded["sha256"] == compare.load_config(config).sha256


SHIPPED_CONFIG = (
    Path(__file__).resolve().parents[2]
    / "configs"
    / "control_pretraining"
    / "30b_baseline_ablations"
    / "nemotron_nano_30b_nvidia_base_weights_check.yaml"
)
PINNED_REVISION = "e0ac9ee3dfd02be21b5479edfc2f671ed269d0a2"


def test_the_shipped_nvidia_base_config_compares_the_pinned_revision_with_a_round_trip_export():
    config = compare.load_config(SHIPPED_CONFIG)
    assert config.reference.name == PINNED_REVISION
    assert "NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16" in str(config.reference)
    assert config.candidate.name.endswith("-hf-export")
    for directory in (config.reference, config.candidate):
        assert not config.report.is_relative_to(directory), "the report must not be written into a checkpoint"
