# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The standard exporter's torch_grouped repair, end to end on tiny checkpoint directory trees.

The tests check the decision (clone or not), the clone the model is loaded from, and where the output and its
provenance land. They run the real ``pipeline_checkpoint_convert_hf.main()`` and the real
``pipeline_checkpoint_convert.sh export``.

Only the boundary that needs GPUs is replaced. The conversion itself (``convert_multi_gpu`` /
``convert_single_process``: building a 30B model and writing its shards) is swapped for a fake that records what it
was asked to load and creates the output directory. In the launcher, ``torchrun`` is that boundary, and the launcher
runs under PipelineShell (tests/unit_tests/stubbed_shell.py), which stubs SLURM and the container.
Everything else, from resolving the iteration to the clone, the run_config copy and the clone's removal, is real.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from pipeline.export import export_clone
from tests.unit_tests.export_clone_fixtures import (
    OLD_IMPL,
    OLD_TARGET,
    TE_GROUPED_RUN_CONFIG,
    make_checkpoint,
    snapshot,
    torch_grouped_run_config,
)


HF_MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16"


@pytest.fixture()
def convert(convert_module, monkeypatch):
    """Run the exporter's main() with the conversion replaced by a recording fake (the GPU boundary)."""
    calls: list[dict] = []

    def fake_conversion(**kwargs):
        # What the real conversion reads (the load path's run_config, which load_model_config rebuilds the model
        # from) and what it leaves for the rest of main(): the output directory, holding the generation_config.json
        # every real export has (a --reasoning export is refused without one).
        calls.append({**kwargs, "run_config_read": (kwargs["load_path"] / "run_config.yaml").read_text()})
        kwargs["hf_path"].mkdir(parents=True, exist_ok=True)
        (kwargs["hf_path"] / "generation_config.json").write_text(
            json.dumps({"bos_token_id": 1, "do_sample": True, "temperature": 1.0, "top_p": 1.0})
        )

    monkeypatch.setattr(convert_module, "convert_single_process", fake_conversion)
    monkeypatch.setattr(convert_module, "convert_multi_gpu", fake_conversion)
    for variable in ("WORLD_SIZE", "RANK"):
        monkeypatch.delenv(variable, raising=False)

    def run(*argv: str) -> list[dict]:
        monkeypatch.setattr(sys, "argv", ["pipeline_checkpoint_convert_hf.py", *argv, "--hf-model", HF_MODEL])
        convert_module.main()
        return calls

    return run


@pytest.fixture()
def masked(tmp_path):
    """A torch_grouped checkpoint as the quickstart posture saves it, and the root its clones go under."""
    save = tmp_path / "checkpoints" / "masked"
    make_checkpoint(save, [477], tracker=477, run_config=torch_grouped_run_config(save))
    return save, tmp_path / "clones"


# ----------------------------------------------------------------------------------------------
# pipeline_checkpoint_convert_hf.main()


def test_a_torch_grouped_checkpoint_loads_from_its_clone_and_exports_beside_itself(masked, convert):
    save, root = masked
    iter_path = save / "iter_0000477"
    original = (iter_path / "run_config.yaml").read_text()
    source = export_clone.prepare_export_source(save, 477, root, "1-1")
    before = snapshot(save)

    calls = convert(
        "--megatron-path", str(save), "--iteration", "477", "--no-reasoning", "--load-path", str(source.clone)
    )

    (call,) = calls
    assert call["load_path"] == source.clone, "the model is loaded from the clone"
    assert OLD_TARGET not in call["run_config_read"] and OLD_IMPL not in call["run_config_read"]
    assert call["hf_path"] == iter_path / "hf", "the export lands in the checkpoint's own iter_N/hf"
    provenance = (iter_path / "hf" / "megatron_run_config.yaml").read_text()
    assert provenance == original, "hf/megatron_run_config.yaml is the checkpoint's own run_config, never the repair"
    assert f"save: {save}" in provenance and "wandb_exp_name: arm_masked" in provenance
    assert not (source.clone / "hf").exists()
    after = snapshot(save)
    assert {key: value for key, value in after.items() if not key.startswith("iter_0000477/hf")} == before, (
        "nothing in the training tree changes but the new hf/ export"
    )


def test_a_torch_grouped_checkpoint_without_a_clone_is_refused_before_anything_loads(masked, convert):
    save, _ = masked
    with pytest.raises(export_clone.ExportError, match="needs the export repair"):
        convert("--megatron-path", str(save), "--no-reasoning")
    assert not (save / "iter_0000477" / "hf").exists()


def test_any_other_checkpoint_exports_from_itself(tmp_path, convert):
    save = make_checkpoint(tmp_path / "te", [30], tracker=30, run_config=TE_GROUPED_RUN_CONFIG)
    (call,) = convert("--megatron-path", str(save), "--reasoning")
    assert call["load_path"] == save / "iter_0000030"
    assert call["hf_path"] == save / "iter_0000030" / "hf"
    assert (save / "iter_0000030" / "hf" / "megatron_run_config.yaml").read_text() == TE_GROUPED_RUN_CONFIG


def test_a_load_path_that_is_not_a_clone_of_this_iteration_is_refused(masked, tmp_path, convert):
    save, root = masked
    other = make_checkpoint(
        tmp_path / "checkpoints_b" / "masked", [477], tracker=477, run_config=torch_grouped_run_config(tmp_path)
    )
    foreign = export_clone.prepare_export_source(other, 477, root, "2").clone
    with pytest.raises(export_clone.ExportError, match="is not a clone of"):
        convert("--megatron-path", str(save), "--no-reasoning", "--load-path", str(foreign))
    assert not (save / "iter_0000477" / "hf").exists()


def test_an_explicit_hf_path_receives_the_export_and_the_original_run_config(masked, tmp_path, convert):
    save, root = masked
    clone = export_clone.prepare_export_source(save, None, root, "3").clone
    out = tmp_path / "exports" / "masked_477"
    (call,) = convert("--megatron-path", str(save), "--no-reasoning", "--load-path", str(clone), "--hf-path", str(out))
    assert call["hf_path"] == out
    assert (out / "megatron_run_config.yaml").read_text() == (save / "iter_0000477" / "run_config.yaml").read_text()
    assert not (save / "iter_0000477" / "hf").exists()


# ----------------------------------------------------------------------------------------------
# pipeline_checkpoint_convert.sh export

# What the conversion does to the export clone while it "runs", as STUB_ON_TORCHRUN (the launch's arguments are its
# own): copy the run_config the model would be rebuilt from to $RECORD, and with WRITE_INTO_CLONE set also
# write a file into the clone, as a conversion that wrote its output there would.
TORCHRUN_READS_THE_CLONE = """
previous=""
for argument in "$@"; do
    if [[ "$previous" == "--load-path" ]]; then
        cp "$argument/run_config.yaml" "$RECORD"
        if [[ -n "${WRITE_INTO_CLONE:-}" ]]; then echo output > "$argument/stray.txt"; fi
    fi
    previous="$argument"
done
"""


@pytest.fixture()
def launcher(pipeline_launch_shell, tmp_path):
    """Run the real launcher's export mode under PipelineShell (tests/unit_tests/stubbed_shell.py), with clones under
    ``tmp_path/clones``. Returns the run's record and the converter's torchrun arguments (empty when torchrun never
    ran)."""
    root = tmp_path / "clones"
    run_config_read = tmp_path / "torchrun.run_config"

    def run(*args: str, torchrun_exit: int = 0, write_into_clone: bool = False) -> tuple[dict, list[str]]:
        result = pipeline_launch_shell.run(
            "pipeline_checkpoint_convert.sh",
            "export",
            *args,
            "--hf-model",
            HF_MODEL,
            "--no-reasoning",
            env={
                "GEODESIC_EXPORT_CLONE_ROOT": str(root),
                "STUB_TORCHRUN_RC": str(torchrun_exit),
                "STUB_ON_TORCHRUN": TORCHRUN_READS_THE_CLONE,
                "RECORD": str(run_config_read),
                "WRITE_INTO_CLONE": "1" if write_into_clone else "",
            },
        )
        launches = [call["argv"] for call in result["calls"] if call["cmd"] == "torchrun"]
        return result, launches[0] if launches else []

    return run, root, run_config_read


def value_after(arguments: list[str], flag: str) -> str | None:
    return arguments[arguments.index(flag) + 1] if flag in arguments else None


def test_the_launcher_clones_a_torch_grouped_checkpoint_pins_the_iteration_and_removes_the_clone(masked, launcher):
    save, _ = masked
    run, root, run_config_read = launcher
    before = snapshot(save)

    result, arguments = run(str(save))

    assert result["returncode"] == 0, result["stdout"] + result["stderr"]
    assert value_after(arguments, "--iteration") == "477", "the iteration prepare resolved is the one converted"
    load_path = Path(value_after(arguments, "--load-path"))
    assert load_path.name == "iter_0000477" and load_path.is_relative_to(root)
    assert value_after(arguments, "--megatron-path") == str(save), "output and provenance still name the checkpoint"
    repaired = run_config_read.read_text()
    assert OLD_TARGET not in repaired and OLD_IMPL not in repaired
    assert "export clone: torch_grouped repair" in result["stdout"]
    assert not load_path.exists() and list(root.iterdir()) == [], "the clone is removed after a successful export"
    assert snapshot(save) == before


def test_the_launcher_fails_when_the_clone_of_a_successful_export_cannot_be_removed(masked, launcher):
    """A clone holding something an export clone does not is left in place, and the job fails, so the leftover is
    looked at rather than unnoticed in a successful job's log."""
    save, _ = masked
    run, _, _ = launcher
    result, arguments = run(str(save), write_into_clone=True)
    assert result["returncode"] == 1, result["stdout"] + result["stderr"]
    load_path = Path(value_after(arguments, "--load-path"))
    assert (load_path / "stray.txt").is_file(), "the clone is left in place"
    assert "holds ['stray.txt'], which an export clone does not" in result["stderr"]
    assert f"its export clone {load_path} could not be removed" in result["stderr"]


def test_the_launcher_exports_any_other_checkpoint_from_itself_with_no_clone(tmp_path, launcher):
    save = make_checkpoint(tmp_path / "te", [30], tracker=30, run_config=TE_GROUPED_RUN_CONFIG)
    run, root, _ = launcher
    result, arguments = run(str(save), "--iteration", "30")
    assert result["returncode"] == 0, result["stdout"] + result["stderr"]
    assert "--load-path" not in arguments
    assert arguments.count("--iteration") == 1 and value_after(arguments, "--iteration") == "30"
    assert not root.exists(), "nothing is written for a checkpoint that needs no repair"


def test_the_launcher_keeps_the_clone_of_a_failed_export(masked, launcher):
    save, _ = masked
    run, root, _ = launcher
    result, arguments = run(str(save), "--iteration", "477")
    assert result["returncode"] == 0
    result, arguments = run(str(save), "--iteration", "477", torchrun_exit=3)
    assert result["returncode"] != 0
    load_path = Path(value_after(arguments, "--load-path"))
    export_clone.check_is_export_clone(save / "iter_0000477", load_path)


def test_the_launcher_stops_before_torchrun_when_the_checkpoint_cannot_be_resolved(tmp_path, launcher):
    run, _, _ = launcher
    result, arguments = run(str(tmp_path / "absent"))
    assert result["returncode"] != 0
    assert arguments == [], "torchrun never ran"
    assert "Checkpoint directory not found" in result["stderr"]
