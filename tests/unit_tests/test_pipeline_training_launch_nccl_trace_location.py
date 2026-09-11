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
"""The NCCL trace-location block in pipeline_training_launch.sh.

`configure_nccl_trace_location` exports where a rank's NCCL flight-recorder dump lands and the
prefix of the FIFO through which one is asked for. What it must get right: the dump goes beside
the raw log (torch's own default is node-local /tmp, gone with the allocation), the FIFO prefix
sits directly in /tmp (one under $TMPDIR would fail on every node but the batch node and abort
the rank), a value already in the environment is respected, and a launch that cannot place the
dump -- no raw log, or a log directory it cannot write -- says so and carries on, because
diagnostics bookkeeping must never kill a training launch.

These tests run the real function, lifted verbatim from the launcher, under the launcher's own
shell options. Nothing is stubbed: the function touches only the filesystem and the environment.
"""

import os
import re
import subprocess

import pytest

from tests.unit_tests.launcher_source import launcher_function


def _run(tmp_path, raw_log_path, preset=None):
    """Run the function under `set -euo pipefail` and print the two variables it owns."""
    script = tmp_path / "harness.sh"
    script.write_text(
        "set -euo pipefail\n"
        f"{launcher_function('configure_nccl_trace_location')}\n"
        "configure_nccl_trace_location\n"
        'printf "TEMP=[%s]\\n" "${TORCH_NCCL_DEBUG_INFO_TEMP_FILE:-}"\n'
        'printf "PIPE=[%s]\\n" "${TORCH_NCCL_DEBUG_INFO_PIPE_FILE:-}"\n'
        'echo "REACHED_END"\n'
    )
    env = {"PATH": os.environ["PATH"], "SLURM_JOB_ID": "4242", "ISAMBARD_RAW_LOG_PATH": raw_log_path}
    env.update(preset or {})
    return subprocess.run(["bash", str(script)], capture_output=True, text=True, env=env, timeout=60)


def _var(result, name):
    match = re.search(rf"{name}=\[(.*)\]", result.stdout)
    assert match, f"harness produced no {name} line: {result.stdout!r} {result.stderr!r}"
    return match.group(1)


@pytest.fixture
def raw_log(tmp_path):
    log_dir = tmp_path / "megatron_runs"
    log_dir.mkdir()
    log = log_dir / "train-4242.out"
    log.write_text("")
    return log


def test_dump_lands_beside_the_raw_log(tmp_path, raw_log):
    result = _run(tmp_path, str(raw_log))
    assert result.returncode == 0, result.stderr
    assert _var(result, "TEMP") == str(raw_log.parent / "nccl_trace" / "4242" / "rank_")


def test_the_dump_directory_is_created(tmp_path, raw_log):
    """torch does not create parent directories for its dump; the launcher must."""
    _run(tmp_path, str(raw_log))
    assert (raw_log.parent / "nccl_trace" / "4242").is_dir()


def test_fifo_prefix_is_directly_in_tmp_and_names_the_job(tmp_path, raw_log):
    result = _run(tmp_path, str(raw_log))
    assert _var(result, "PIPE") == "/tmp/nccl_dump_4242_"


def test_preset_values_are_respected(tmp_path, raw_log):
    preset = {
        "TORCH_NCCL_DEBUG_INFO_TEMP_FILE": "/elsewhere/rank_",
        "TORCH_NCCL_DEBUG_INFO_PIPE_FILE": "/tmp/mine_",
    }
    result = _run(tmp_path, str(raw_log), preset)
    assert _var(result, "TEMP") == "/elsewhere/rank_"
    assert _var(result, "PIPE") == "/tmp/mine_"


def test_no_raw_log_leaves_the_dump_path_unset_and_says_so(tmp_path):
    """An interactive launch resolves no raw log; the dump keeps torch's default, audibly."""
    result = _run(tmp_path, "")
    assert result.returncode == 0, result.stderr
    assert _var(result, "TEMP") == ""
    assert "NOTE: no raw log path" in result.stderr
    assert _var(result, "PIPE") == "/tmp/nccl_dump_4242_"


def test_an_unwritable_log_directory_warns_and_does_not_end_the_launch(tmp_path, raw_log):
    """The by-run-id symlink beside it is non-fatal for the same reason; a launch must survive."""
    raw_log.parent.chmod(0o555)
    try:
        result = _run(tmp_path, str(raw_log))
    finally:
        raw_log.parent.chmod(0o755)
    assert result.returncode == 0, result.stderr
    assert "REACHED_END" in result.stdout
    assert "WARNING: could not create" in result.stderr
    assert _var(result, "TEMP") == ""
    assert _var(result, "PIPE") == "/tmp/nccl_dump_4242_"
