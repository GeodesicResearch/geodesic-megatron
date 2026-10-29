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
"""Running one of the repo's shell scripts under a unit test, with stub commands first on PATH.

Catalogue:
    stubbed_shell_env: an environment with a stub directory first on PATH.
    run_in_own_session: run a command in its own session, killing its process group on timeout.
    ALL_RECORDED: every stub of PipelineShell that records its calls.
    HARNESS_ENV: the STUB_* variables through which PipelineShell controls its stubs.
    PIPELINE_RECORD_ENV: the fabric, rendezvous and job-path variables the pipeline launchers set for their ranks.
    PipelineShell: run a pipeline script with SLURM, the container and the rank launchers stubbed and recorded.
"""

import glob
import json
import os
import shutil
import signal
import subprocess
import sys
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path

from tests.unit_tests.child_process import REPO_ROOT


def stubbed_shell_env(bindir: str | Path, extra: Mapping[str, str]) -> dict[str, str]:
    """PATH with the stub directory first, plus the variables the test sets, and nothing else.

    Deliberately not a copy of os.environ. Inside the pipeline container BASH_ENV and ENV make every
    non-interactive bash source /etc/bash.bashrc and /etc/shinit_v2 before its first line, which costs
    ~2.5 s per shell and runs `nvidia-smi -q -d COMPUTE` resolved through PATH. Under a copied
    environment a stub named nvidia-smi that is itself a shell would start another copy of itself before
    running a line, without bound. That startup hook is the container's, not the script's, and is not
    under test.
    """
    return {"PATH": f"{bindir}{os.pathsep}{os.environ['PATH']}", **extra}


def run_in_own_session(args: Sequence[str], env: Mapping[str, str], timeout: float) -> subprocess.CompletedProcess:
    """Run a command in a session of its own and capture its output as text.

    A command that outlives ``timeout`` has its whole process group killed while it is still unreaped, so
    the group is still its own, and then TimeoutExpired is raised: nothing it started stays behind, as it
    would if only the command itself were killed.
    """
    proc = subprocess.Popen(
        list(args),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=dict(env),
        start_new_session=True,
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.communicate()
        raise
    return subprocess.CompletedProcess(proc.args, proc.returncode, stdout, stderr)


# Appends one JSON line per call to $STUB_CALLS: the command's name, its argv, and the environment variables named
# in $STUB_RECORD_ENV that are set ("*": all of them).
_RECORDER = """\
import json, os, sys
name, *argv = sys.argv[1:]
keys = os.environ.get("STUB_RECORD_ENV", "")
env = dict(os.environ) if keys == "*" else {k: os.environ[k] for k in keys.split(",") if k in os.environ}
with open(os.environ["STUB_CALLS"], "a") as f:
    f.write(json.dumps({"cmd": name, "argv": argv, "env": env}) + "\\n")
"""

# `scontrol show hostname[s] LIST`: one host per line, for lists such as nid[001-003,7],nid9. A reader that stops
# early (`| head -1`) ends it quietly, as it ends scontrol, rather than with a BrokenPipeError on stderr.
_HOSTLIST = """\
import re, signal, sys
signal.signal(signal.SIGPIPE, signal.SIG_DFL)
for part in re.findall(r"[^,\\[]+(?:\\[[^\\]]*\\])?", sys.argv[-1]):
    m = re.fullmatch(r"([^\\[]*)\\[([^\\]]*)\\]", part)
    if not m:
        print(part)
        continue
    for rng in m.group(2).split(","):
        lo, _, hi = rng.partition("-")
        for i in range(int(lo), int(hi or lo) + 1):
            print(f"{m.group(1)}{i:0{len(lo)}d}")
"""

# The Hub is a network service: the huggingface_hub that shell-run Python imports is the recorder
# (tests/unit_tests/recording_hub.py).
_FAKE_HUB = "from tests.unit_tests.recording_hub import HfApi  # noqa: F401\n"

# `record NAME ARGS...`: append a call record when NAME is in $STUB_RECORD; a record that fails ends the stub.
_RECORD_FN = """\
record() {
    [[ ",$STUB_RECORD," == *",$1,"* ]] || return 0
    "$STUB_PYTHON" "$STUB_DIR/recorder.py" "$@" || { echo "stub $1: recording the call failed" >&2; exit 97; }
}
"""

# Each stub replaces a command that cannot run under a unit test, for the reason its comment gives.
_PIPELINE_STUBS = {
    "srun": """\
# SLURM: record the step's options and command (the command records its own arguments), then run the command
# once, here, as task 0 of node 0.
opts=()
while [[ "${1:-}" == -* ]]; do opts+=("$1"); shift; done
record srun "${opts[@]}" "$1"
SLURM_NODEID=0 SLURM_PROCID=0 SLURM_LOCALID=0 exec "$@"
""",
    "scontrol": """\
# SLURM: expand a node list as `scontrol show hostnames` does.
exec "$STUB_PYTHON" "$STUB_DIR/hostlist.py" "$@"
""",
    "apptainer": """\
# The tests already run in the pipeline container, where apptainer cannot nest: record the image the shim chose
# (`apptainer exec [flags] <sif> bash -c <cmd>`, so the fourth argument from the end) and run the command string
# as that `bash -c` would.
record apptainer "${@: -4:1}"
exec bash -c "${!#}"
""",
    "module": """\
# Lmod is a host shell function, absent in the container; the scripts only purge.
exit 0
""",
    "hostname": """\
# The test host's name differs between runs; the fake job's only node is nid000001.
echo nid000001
""",
    "date": """\
# Wall-clock time would make every banner differ between runs.
echo "<DATE>"
""",
    "sleep": """\
# A poll loop's wait: record it, run $STUB_ON_SLEEP (a test's way to advance the world), and fail once more than
# $STUB_SLEEP_LIMIT waits have passed, so that a loop that never ends fails its test.
record sleep "$@"
n=$(( $(cat "$STUB_DIR/sleeps" 2>/dev/null || echo 0) + 1 )); echo "$n" > "$STUB_DIR/sleeps"
[ "$n" -le "${STUB_SLEEP_LIMIT:-3}" ] || { echo "stub sleep: more than ${STUB_SLEEP_LIMIT:-3} waits" >&2; exit 1; }
[ -z "${STUB_ON_SLEEP:-}" ] || bash -c "$STUB_ON_SLEEP"
""",
    "torchrun": """\
# Ranks need GPUs and a rendezvous: record the launch, run $STUB_ON_TORCHRUN with the launch's arguments as its own
# (a test's way to do what the ranks would to the files they read), and exit with $STUB_TORCHRUN_RC (default 0).
record torchrun "$@"
[ -z "${STUB_ON_TORCHRUN:-}" ] || bash -c "$STUB_ON_TORCHRUN" torchrun "$@" || { echo "stub torchrun: STUB_ON_TORCHRUN failed" >&2; exit 98; }
exit "${STUB_TORCHRUN_RC:-0}"
""",
    "ft_launcher": """\
# Ranks need GPUs and a rendezvous: record the launch. --help prints $STUB_FT_LAUNCHER_HELP when it is set (an
# empty value is a launcher with neither flag), and otherwise the two section-timeout flags the launcher passes,
# then exits with $STUB_FT_LAUNCHER_HELP_RC (default 0).
if [ "${1:-}" = --help ]; then
    printf '%s\n' "${STUB_FT_LAUNCHER_HELP-  --ft-rank-section-timeouts ...
  --ft-rank-out-of-section-timeout ...}"
    exit "${STUB_FT_LAUNCHER_HELP_RC:-0}"
fi
record ft_launcher "$@"
""",
    "python": """\
# A script's launch needs GPUs or the network: record it. Helper code (`python -c`, `python -m pipeline.*`, the
# training launcher's checks `python -m scripts.training.code_identity` and `scripts.training.launch_width`, and
# scripts/data/prepare_revisions.py, which resolves a tokenizer reference) runs for real, with the Hub replaced by
# the recording fake.
if [ "${1:-}" = -c ] \
    || { [ "${1:-}" = -m ] && [[ "${2:-}" == pipeline.* || "${2:-}" == scripts.training.code_identity \
        || "${2:-}" == scripts.training.launch_width ]]; } \
    || [ "${1:-}" = scripts/data/prepare_revisions.py ]; then
    PYTHONPATH="$STUB_DIR/fake_modules${PYTHONPATH:+:$PYTHONPATH}" exec "$STUB_PYTHON" "$@"
fi
record python "$@"
""",
    "isambard_sbatch": """\
# Job submission: record it and accept it.
record isambard_sbatch "$@"
""",
}

#: Every stub that records a call, and the Hub fake.
ALL_RECORDED = ("srun", "apptainer", "sleep", "torchrun", "ft_launcher", "python", "isambard_sbatch", "hub")
#: The STUB_* variables through which PipelineShell controls its stubs.
HARNESS_ENV = ("STUB_DIR", "STUB_PYTHON", "STUB_CALLS", "STUB_RECORD", "STUB_RECORD_ENV")
#: The fabric, rendezvous and job-path variables the pipeline launchers set for their ranks.
PIPELINE_RECORD_ENV = (
    "NCCL_NET",
    "FI_PROVIDER",
    "NCCL_SOCKET_IFNAME",
    "NCCL_CROSS_NIC",
    "NCCL_NET_GDR_LEVEL",
    "FI_CXI_DISABLE_HOST_REGISTER",
    "FI_MR_CACHE_MONITOR",
    "FI_CXI_DEFAULT_CQ_SIZE",
    "FI_CXI_RX_MATCH_MODE",
    "NCCL_NVLS_ENABLE",
    "NCCL_DEBUG",
    "MPICH_GPU_SUPPORT_ENABLED",
    "TORCH_NCCL_TIMEOUT",
    "TORCHDYNAMO_DISABLE",
    "TORCH_COMPILE_DISABLE",
    "PYTHONUNBUFFERED",
    "TMPDIR",
    "MEGATRON_CONFIG_LOCK_DIR",
    "TRITON_CACHE_DIR",
    "MASTER_ADDR",
    "MASTER_PORT",
    "SLURM_NODEID",
)


class PipelineShell:
    """Runs the repo's pipeline shell scripts with SLURM, the container and the rank launchers stubbed.

    The real ``pipeline_env_config.env``, exec shim and in-container activation run, so this must run inside the
    pipeline container. ``env_config_require`` passes against a fake image, Slingshot build and libfabric dir set
    through ``GEODESIC_CONTAINER_*``. The fake job's ``SLURM_JOB_ID`` is unique per instance and a multiple of
    1000, so a ``29500 + SLURM_JOB_ID % 1000`` port is 29500.
    """

    def __init__(self, tmp_path: Path, record: Sequence[str] = ALL_RECORDED, record_env: Sequence[str] | str = ()):
        """Build the stubs and the fake container under ``tmp_path``.

        Args:
            tmp_path: a directory of the test's own.
            record: the stubs (names in ``ALL_RECORDED``) whose calls are recorded.
            record_env: the environment variables each record holds when set, or ``"*"`` for all of them.
        """
        self.tmp_path = tmp_path
        self.job_id = str(900_000_000 + uuid.uuid4().int % 99_000 * 1000)
        self.stub_dir = tmp_path / "stubs"
        (self.stub_dir / "fake_modules" / "huggingface_hub").mkdir(parents=True)
        (self.stub_dir / "recorder.py").write_text(_RECORDER)
        (self.stub_dir / "hostlist.py").write_text(_HOSTLIST)
        (self.stub_dir / "fake_modules" / "huggingface_hub" / "__init__.py").write_text(_FAKE_HUB)
        for name, body in _PIPELINE_STUBS.items():
            (self.stub_dir / name).write_text(f"#!/bin/bash\n{_RECORD_FN}{body}")
            (self.stub_dir / name).chmod(0o755)
        container = tmp_path / "container"
        plugin = container / "slingshot" / "aws-ofi-nccl" / "lib" / "libnccl-net.so"
        plugin.parent.mkdir(parents=True)
        plugin.touch()
        (container / "slingshot" / "hostlibs").mkdir()
        (container / "slingshot" / "hostlibs" / "libcxi.so.1").symlink_to("/host/usr/lib64/libcxi.so.1")
        (container / "libfabric").mkdir()
        (container / "nemo.sif").touch()
        (container / "nemo.sif.source.txt").write_text("stub image\n")
        (tmp_path / "home").mkdir()
        self.calls_file = tmp_path / "calls.jsonl"
        self._base = {
            "HOME": str(tmp_path / "home"),
            "USER": "tester",
            "STUB_DIR": str(self.stub_dir),
            "STUB_PYTHON": sys.executable,
            "STUB_CALLS": str(self.calls_file),
            "STUB_RECORD": ",".join(record),
            "STUB_RECORD_ENV": record_env if isinstance(record_env, str) else ",".join(record_env),
            "GEODESIC_REPO_DIR": str(REPO_ROOT),
            "GEODESIC_CONTAINER_SIF": str(container / "nemo.sif"),
            "GEODESIC_CONTAINER_SLINGSHOT_DIR": str(container / "slingshot"),
            "GEODESIC_CONTAINER_HOST_LIBFABRIC": str(container / "libfabric"),
            "SLURM_JOB_ID": self.job_id,
            "SLURM_NNODES": "1",
            "SLURM_NODELIST": "nid000001",
        }

    def env(self, extra: Mapping[str, str] | None = None) -> dict[str, str]:
        """The scripts' environment: the stubs first on PATH, the fake job and container, plus ``extra``."""
        return stubbed_shell_env(self.stub_dir, {**self._base, **(extra or {})})

    def run(self, script: str, *args: str, env: Mapping[str, str] | None = None, timeout: float = 300) -> dict:
        """Run ``bash <script> args`` in ``env()`` and return what happened; a relative ``script`` is under the repo root.

        Returns:
            ``args``, ``returncode``, ``stdout``, ``stderr``, and ``calls``: the records of this run's recorded stub
            calls, in call order.
        """
        self.calls_file.unlink(missing_ok=True)
        (self.stub_dir / "sleeps").unlink(missing_ok=True)
        result = run_in_own_session(["bash", str(REPO_ROOT / script), *args], self.env(env), timeout)
        lines = self.calls_file.read_text().splitlines() if self.calls_file.exists() else []
        return {
            "args": list(args),
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "calls": [json.loads(line) for line in lines],
        }

    def normalise(self, value: object) -> object:
        """``value`` (JSON-serialisable) with ``tmp_path``, the repo root and the job id as placeholders."""
        text = json.dumps(value)
        text = text.replace(str(self.tmp_path), "<TMP>").replace(str(REPO_ROOT), "<REPO>")
        return json.loads(text.replace(self.job_id, "<JOB>"))

    def job_tmp_dirs(self) -> list[str]:
        """The node-local ``/tmp/*_<job id>[_*]`` directories the scripts created for this fake job."""
        return glob.glob(f"/tmp/*_{self.job_id}") + glob.glob(f"/tmp/*_{self.job_id}_*")

    def remove_job_tmp_dirs(self) -> None:
        """Delete ``job_tmp_dirs()``; for a test's teardown."""
        for path in self.job_tmp_dirs():
            shutil.rmtree(path)
