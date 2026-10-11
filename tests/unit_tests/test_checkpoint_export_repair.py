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
was asked to load and creates the output directory. In the launcher, ``torchrun`` is that boundary. ``srun``, the
container shim and ``module`` are stubbed to run on this host, as in test_pipeline_data_submit_tokenize.py.
Everything else, from resolving the iteration to the clone, the run_config copy and the clone's removal, is real.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest
from scripts.checkpoint import export_clone

from tests.unit_tests.export_clone_fixtures import (
    OLD_IMPL,
    OLD_TARGET,
    TE_GROUPED_RUN_CONFIG,
    make_checkpoint,
    snapshot,
    torch_grouped_run_config,
)


_REPO_ROOT = Path(__file__).resolve().parents[2]
_CONVERT_PATH = _REPO_ROOT / "pipeline_checkpoint_convert_hf.py"
_LAUNCHER = _REPO_ROOT / "pipeline_checkpoint_convert.sh"
HF_MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16"


@pytest.fixture(scope="module")
def convert_module():
    spec = importlib.util.spec_from_file_location("pipeline_checkpoint_convert_hf", _CONVERT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["pipeline_checkpoint_convert_hf"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def convert(convert_module, monkeypatch):
    """Run the exporter's main() with the conversion replaced by a recording fake (the GPU boundary)."""
    calls: list[dict] = []

    def fake_conversion(**kwargs):
        # What the real conversion reads (the load path's run_config, which load_model_config rebuilds the model
        # from) and the one thing it leaves for the rest of main(): the output directory.
        calls.append({**kwargs, "run_config_read": (kwargs["load_path"] / "run_config.yaml").read_text()})
        kwargs["hf_path"].mkdir(parents=True, exist_ok=True)

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
    with pytest.raises(export_clone.ExportError, match="torch_grouped expert settings"):
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

TORCHRUN_STUB = """#!/bin/bash
# The conversion's boundary: record what torchrun was asked to run, and the run_config the model would be
# rebuilt from, read while the conversion "runs". With TORCHRUN_WRITE_INTO_CLONE set it also writes a file into
# the clone, as a conversion that wrote its output there would.
printf '%s\\n' "$@" > "$TORCHRUN_RECORD.args"
previous=""
for argument in "$@"; do
    if [[ "$previous" == "--load-path" ]]; then
        cp "$argument/run_config.yaml" "$TORCHRUN_RECORD.run_config"
        if [[ -n "${TORCHRUN_WRITE_INTO_CLONE:-}" ]]; then echo output > "$argument/stray.txt"; fi
    fi
    previous="$argument"
done
exit "${TORCHRUN_EXIT:-0}"
"""


def write_executable(path: Path, text: str) -> None:
    path.write_text(text)
    path.chmod(path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)


@pytest.fixture()
def launcher(tmp_path):
    """Run the real launcher's export mode on this host. The stub repo supplies the env config and the
    container shim. Its scripts/ is the real one, so the real export_clone.py runs."""
    repo = tmp_path / "stub_repo"
    repo.mkdir()
    (repo / "pipeline_env_config.env").write_text("env_config_require() { return 0; }\n")
    write_executable(repo / "pipeline_env_exec.sh", '#!/bin/bash\nexec bash -c "$1"\n')
    (repo / "pipeline_env_activate.sh").write_text(":\n")
    (repo / "scripts").symlink_to(_REPO_ROOT / "scripts")

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    write_executable(bin_dir / "module", "#!/bin/bash\nexit 0\n")
    # srun runs its last two arguments, the container shim and its command string, here.
    write_executable(bin_dir / "srun", '#!/bin/bash\nexec "${@: -2:1}" "${@: -1}"\n')
    write_executable(bin_dir / "torchrun", TORCHRUN_STUB)
    write_executable(bin_dir / "python", f'#!/bin/bash\nexec {sys.executable} "$@"\n')
    home = tmp_path / "home"
    home.mkdir()

    job_id = str(900_000_000 + os.getpid() % 1_000_000)
    record = tmp_path / "torchrun"
    env = {
        "PATH": f"{bin_dir}:/usr/bin:/bin",
        "HOME": str(home),
        "USER": "tester",
        "SLURM_JOB_ID": job_id,
        "SLURM_NNODES": "1",
        "SLURM_NODELIST": "nid000001",
        "MASTER_ADDR_OVERRIDE": "localhost",
        "MASTER_PORT_OVERRIDE": "29999",
        "GEODESIC_REPO_DIR": str(repo),
        "GEODESIC_EXPORT_CLONE_ROOT": str(tmp_path / "clones"),
        "TORCHRUN_RECORD": str(record),
    }

    def run(
        *args: str, torchrun_exit: int = 0, write_into_clone: bool = False
    ) -> tuple[subprocess.CompletedProcess, list[str]]:
        result = subprocess.run(
            ["bash", str(_LAUNCHER), "export", *args, "--hf-model", HF_MODEL, "--no-reasoning"],
            capture_output=True,
            text=True,
            env={
                **env,
                "TORCHRUN_EXIT": str(torchrun_exit),
                "TORCHRUN_WRITE_INTO_CLONE": "1" if write_into_clone else "",
            },
            timeout=120,
        )
        args_file = Path(f"{record}.args")
        return result, args_file.read_text().splitlines() if args_file.is_file() else []

    yield run, tmp_path / "clones", Path(f"{record}.run_config")
    # The launcher makes its node-local TMPDIR under /tmp, named for the job.
    shutil.rmtree(f"/tmp/megatron_convert_{job_id}", ignore_errors=True)


def value_after(arguments: list[str], flag: str) -> str | None:
    return arguments[arguments.index(flag) + 1] if flag in arguments else None


def test_the_launcher_clones_a_torch_grouped_checkpoint_pins_the_iteration_and_removes_the_clone(masked, launcher):
    save, _ = masked
    run, root, run_config_read = launcher
    before = snapshot(save)

    result, arguments = run(str(save))

    assert result.returncode == 0, result.stdout + result.stderr
    assert value_after(arguments, "--iteration") == "477", "the iteration prepare resolved is the one converted"
    load_path = Path(value_after(arguments, "--load-path"))
    assert load_path.name == "iter_0000477" and load_path.is_relative_to(root)
    assert value_after(arguments, "--megatron-path") == str(save), "output and provenance still name the checkpoint"
    repaired = run_config_read.read_text()
    assert OLD_TARGET not in repaired and OLD_IMPL not in repaired
    assert "export clone: torch_grouped repair" in result.stdout
    assert not load_path.exists() and list(root.iterdir()) == [], "the clone is removed after a successful export"
    assert snapshot(save) == before


def test_the_launcher_fails_when_the_clone_of_a_successful_export_cannot_be_removed(masked, launcher):
    """A clone holding something an export clone does not is left in place, and the job fails, so the leftover is
    looked at rather than unnoticed in a successful job's log."""
    save, _ = masked
    run, _, _ = launcher
    result, arguments = run(str(save), write_into_clone=True)
    assert result.returncode == 1, result.stdout + result.stderr
    load_path = Path(value_after(arguments, "--load-path"))
    assert (load_path / "stray.txt").is_file(), "the clone is left in place"
    assert "holds ['stray.txt'], which an export clone does not" in result.stderr
    assert f"its export clone {load_path} could not be removed" in result.stderr


def test_the_launcher_exports_any_other_checkpoint_from_itself_with_no_clone(tmp_path, launcher):
    save = make_checkpoint(tmp_path / "te", [30], tracker=30, run_config=TE_GROUPED_RUN_CONFIG)
    run, root, _ = launcher
    result, arguments = run(str(save), "--iteration", "30")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "--load-path" not in arguments
    assert arguments.count("--iteration") == 1 and value_after(arguments, "--iteration") == "30"
    assert not root.exists(), "nothing is written for a checkpoint that needs no repair"


def test_the_launcher_keeps_the_clone_of_a_failed_export(masked, launcher):
    save, _ = masked
    run, root, _ = launcher
    result, arguments = run(str(save), "--iteration", "477")
    assert result.returncode == 0
    result, arguments = run(str(save), "--iteration", "477", torchrun_exit=3)
    assert result.returncode != 0
    load_path = Path(value_after(arguments, "--load-path"))
    export_clone.check_is_export_clone(save / "iter_0000477", load_path)


def test_the_launcher_stops_before_torchrun_when_the_checkpoint_cannot_be_resolved(tmp_path, launcher):
    run, _, _ = launcher
    result, arguments = run(str(tmp_path / "absent"))
    assert result.returncode != 0
    assert arguments == [], "torchrun never ran"
    assert "Checkpoint directory not found" in result.stderr
