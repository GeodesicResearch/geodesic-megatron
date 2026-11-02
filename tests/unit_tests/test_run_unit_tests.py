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
"""scripts/run_unit_tests.sh: the passes it chooses from the GPUs' compute mode.

These tests run the real script, which sources the real pipeline_env_activate.sh and counts the GPUs
with the real worker_gpus.visible_gpus. Two boundaries are stubbed on PATH, each because a unit test
cannot cross it: nvidia-smi reports the state of the node's GPUs, which a test can neither set nor rely
on, and `python -m pytest` would run the whole unit suite from inside one of its own tests. The python
stub hands every other invocation to the real interpreter and records each pytest invocation's
arguments and the pinning variable it inherited, so the passes are asserted, not assumed.

The stubs are Python scripts run by this interpreter, so no shell starts inside a stub, and the script
runs through tests/unit_tests/stubbed_shell.py: in a minimal environment and a session of its own, whose
whole process group is killed if it outlives its timeout.
"""

import json
import os
import stat
import sys

import pytest

from tests.unit_tests.stubbed_shell import run_in_own_session, stubbed_shell_env


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCRIPT = os.path.join(REPO_ROOT, "scripts", "run_unit_tests.sh")
PIN = "UNIT_TESTS_PIN_WORKER_GPUS"
TIMEOUT_SECONDS = 60

_NVIDIA_SMI = """
import os
import sys

query = " ".join(sys.argv[1:])
if "--query-gpu=compute_mode" in query:
    sys.stdout.write(os.environ["STUB_COMPUTE_MODES"])
elif "--query-compute-apps=" in query:
    sys.stdout.write(os.environ["STUB_HOLDERS"])
    sys.exit(int(os.environ["STUB_HOLDERS_EXIT"]))
else:
    sys.exit(f"nvidia-smi stub: unexpected query {query}")
"""
# One JSON line per pytest invocation: the pinning variable it inherited (null when unset) and its arguments.
_PYTHON = f"""
import json
import os
import sys

args = sys.argv[1:]
if args[:2] == ["-m", "pytest"]:
    with open(os.environ["STUB_PYTEST_CALLS"], "a") as calls:
        calls.write(json.dumps({{"pin": os.environ.get("{PIN}"), "args": args[2:]}}) + "\\n")
    sys.exit(int(os.environ["STUB_PYTEST_EXIT"]))
os.execv(sys.executable, [sys.executable, *args])
"""


def _stub(bindir, name, body):
    stub = bindir / name
    stub.write_text(f"#!{sys.executable} -I\n{body}")
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC)


def _pytest_calls(path):
    """Each recorded pytest invocation as (inherited pinning value, argument list)."""
    if not path.exists():
        return []
    return [(call["pin"], call["args"]) for call in map(json.loads, path.read_text().splitlines())]


def _option(args, name):
    return args[args.index(name) + 1]


@pytest.fixture
def run_script(tmp_path):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    _stub(bindir, "nvidia-smi", _NVIDIA_SMI)
    _stub(bindir, "python", _PYTHON)
    calls = tmp_path / "pytest_calls"

    def run(compute_modes, gpus, holders="", holders_exit=0, pytest_exit=0, inherited_pin=None):
        env = stubbed_shell_env(
            bindir,
            {
                "CUDA_VISIBLE_DEVICES": ",".join(str(i) for i in range(gpus)),
                "STUB_COMPUTE_MODES": "".join(f"{mode}\n" for mode in compute_modes),
                "STUB_HOLDERS": holders,
                "STUB_HOLDERS_EXIT": str(holders_exit),
                "STUB_PYTEST_EXIT": str(pytest_exit),
                "STUB_PYTEST_CALLS": str(calls),
                "TMPDIR": str(tmp_path),
                # pipeline_env_activate.sh names the user's datasets cache after USER.
                "USER": os.environ["USER"],
            },
        )
        if inherited_pin is not None:
            env[PIN] = inherited_pin
        result = run_in_own_session(["bash", SCRIPT], env, timeout=TIMEOUT_SECONDS)
        return result, _pytest_calls(calls)

    return run


def test_default_mode_runs_one_unpinned_pass_on_three_workers_per_gpu(run_script):
    result, calls = run_script(["Default", "Default"], gpus=2)
    assert result.returncode == 0, result.stderr
    assert len(calls) == 1
    pin, args = calls[0]
    assert pin is None
    assert _option(args, "-n") == "6"
    assert _option(args, "-m") == "not pleasefixme"


def test_default_mode_drops_a_pin_inherited_from_the_caller(run_script):
    """An inherited pin would put the three workers per GPU on one GPU each, and the multi-GPU tests,
    which have no serial pass in this mode, would skip."""
    result, calls = run_script(["Default"] * 4, gpus=4, inherited_pin="1")
    assert result.returncode == 0, result.stderr
    assert [pin for pin, _ in calls] == [None]


@pytest.mark.parametrize("compute_modes", [["Exclusive_Process"] * 2, ["Default", "Exclusive_Process"]])
def test_exclusive_mode_pins_one_worker_per_gpu_then_runs_the_serial_gpu_tests(run_script, compute_modes):
    result, calls = run_script(compute_modes, gpus=2)
    assert result.returncode == 0, result.stderr
    assert len(calls) == 2
    (pin, parallel), (serial_pin, serial) = calls
    assert pin == "1"
    assert _option(parallel, "-n") == "2"
    assert _option(parallel, "-m") == "not pleasefixme and not serial_gpu"
    assert serial_pin == "1"
    assert _option(serial, "-m") == "serial_gpu and not pleasefixme"
    assert _option(serial, "-p") == "no:xdist"
    serial_files = [arg for arg in serial if arg.endswith(".py")]
    assert serial_files
    for path in serial_files:
        assert os.path.basename(path).startswith("test_")
        with open(path) as handle:
            assert "mark.serial_gpu" in handle.read(), path


def test_exclusive_mode_skips_the_serial_pass_when_the_first_pass_fails(run_script):
    result, calls = run_script(["Exclusive_Process"], gpus=1, pytest_exit=3)
    assert result.returncode == 3
    assert len(calls) == 1


def test_exclusive_mode_refuses_to_start_while_a_process_holds_a_gpu(run_script):
    holder = "GPU-6f1c2a, 4242, /opt/venv/bin/python"
    result, calls = run_script(["Exclusive_Process"] * 4, gpus=4, holders=f"{holder}\n")
    assert result.returncode == 1
    assert "already hold them" in result.stderr
    assert holder in result.stderr
    assert calls == []


def test_exclusive_mode_refuses_to_start_when_it_cannot_list_the_gpu_holders(run_script):
    result, calls = run_script(["Exclusive_Process"] * 4, gpus=4, holders_exit=9)
    assert result.returncode == 1
    assert "cannot list the processes holding the GPUs" in result.stderr
    assert calls == []


def test_no_visible_gpu_is_refused(run_script):
    result, calls = run_script(["Default"], gpus=0)
    assert result.returncode == 1
    assert "no GPU is visible" in result.stderr
    assert calls == []
