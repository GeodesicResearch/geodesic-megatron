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
"""scripts/training/dump_hung_ranks.sh: the per-node payload and the driver that sweeps a job.

The payload finds the job's ranks on the node by the environment torchrun gives every worker
(SLURM_JOB_ID, RANK, and the FIFO prefix in TORCH_NCCL_DEBUG_INFO_PIPE_FILE), writes one stack
file per global rank, and pokes each rank's FIFO. The driver resolves the job's node count and raw
log, places the evidence beside that log, and runs the payload on every node with the session's
own SLURM variables unset. These tests run the real script in both modes against real child
processes carrying a rank's environment. Two boundaries are stubbed on PATH, each named here
because a unit test cannot cross it: py-spy attaches to a process with ptrace, and squeue,
scontrol and srun talk to the SLURM controller (the srun stub records what it was asked to run
and runs the payload locally, so the driver's whole flow is still exercised). The stubs record
what they were asked, so the rank-to-file mapping and the srun invocation are asserted, not
assumed.
"""

import os
import stat
import subprocess

import pytest


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCRIPT = os.path.join(REPO_ROOT, "scripts", "training", "dump_hung_ranks.sh")

_PY_SPY_OK = '#!/bin/bash\necho "stub dump of pid $3 ($*)"\n'
_PY_SPY_FAIL = "#!/bin/bash\necho 'stub: could not attach' >&2\nexit 1\n"
_SQUEUE = '#!/bin/bash\n[ -z "${SQUEUE_UNKNOWN_JOB:-}" ] || exit 1\necho 2\n'
_SCONTROL = '#!/bin/bash\necho "JobId=$3 JobName=probe ${SCONTROL_STDOUT:+StdOut=$SCONTROL_STDOUT} WorkDir=/x"\n'
# The trailing five arguments are the payload command: bash <script> --node <jobid> <outdir>.
_SRUN = (
    "#!/bin/bash\n"
    'printf \'%s\\n\' "$@" > "$SRUN_ARGS_FILE"\n'
    'echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unset}" > "$SRUN_ENV_FILE"\n'
    'exec "${@: -5}"\n'
)


def _stub(bindir, name, body):
    bindir.mkdir(exist_ok=True)
    stub = bindir / name
    stub.write_text(body)
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC)
    return bindir


def _rank_process(job_id, rank, pipe_prefix):
    """A process carrying exactly the environment torchrun gives a worker of that job."""
    env = {"PATH": os.environ["PATH"], "SLURM_JOB_ID": job_id, "RANK": str(rank)}
    if pipe_prefix is not None:
        env["TORCH_NCCL_DEBUG_INFO_PIPE_FILE"] = pipe_prefix
    return subprocess.Popen(["sleep", "60"], env=env)


@pytest.fixture
def job_id():
    # Unique per test process so parallel test workers never see each other's ranks.
    return f"t{os.getpid()}"


@pytest.fixture
def pipe_prefix(tmp_path):
    return str(tmp_path / "pipes" / "nccl_dump_")


@pytest.fixture
def ranks(job_id, pipe_prefix):
    os.makedirs(os.path.dirname(pipe_prefix), exist_ok=True)
    procs = [_rank_process(job_id, 3, pipe_prefix), _rank_process(job_id, 10, pipe_prefix)]
    decoys = [
        _rank_process("other-job", 3, pipe_prefix),
        subprocess.Popen(["sleep", "60"], env={"PATH": os.environ["PATH"], "SLURM_JOB_ID": job_id}),
    ]
    yield procs
    for proc in procs + decoys:
        proc.kill()
        proc.wait()


def _env(bindir, **extra):
    """A minimal environment for the script: PATH with the stubs first, plus what a test sets.

    Deliberately not a copy of os.environ: inside the pipeline container BASH_ENV and ENV make
    every non-interactive bash source /etc/shinit_v2, ~2.5 s per shell, and the script and
    every stub are shells. That startup hook is the container's, not the script's, and is not
    under test.
    """
    return {"PATH": f"{bindir}:{os.environ['PATH']}", **extra}


def _run_node(tmp_path, job_id, bindir):
    out = tmp_path / "evidence"
    out.mkdir()
    env = _env(bindir)
    result = subprocess.run(
        ["bash", SCRIPT, "--node", job_id, str(out)], capture_output=True, text=True, env=env, timeout=120
    )
    return out, result


def test_one_stack_per_rank_of_the_job_named_by_global_rank(tmp_path, job_id, ranks):
    out, result = _run_node(tmp_path, job_id, _stub(tmp_path / "bin", "py-spy", _PY_SPY_OK))
    assert result.returncode == 0, result.stderr
    assert sorted(p.name for p in out.iterdir()) == ["rank_10.stack", "rank_3.stack"]
    assert f"stub dump of pid {ranks[0].pid}" in (out / "rank_3.stack").read_text()
    assert f"stub dump of pid {ranks[1].pid}" in (out / "rank_10.stack").read_text()


def test_report_counts_stacks_and_ranks_that_opened_no_fifo(tmp_path, job_id, ranks):
    """No watchdog thread means no FIFO: the report says so per rank instead of staying silent."""
    _, result = _run_node(tmp_path, job_id, _stub(tmp_path / "bin", "py-spy", _PY_SPY_OK))
    assert "2 stack(s) written, 0 failed; 0 FIFO(s) triggered, 0 without a reader, 2 rank(s) opened no FIFO" in (
        result.stdout
    )


def test_a_failed_attach_is_reported_not_skipped(tmp_path, job_id, ranks):
    out, result = _run_node(tmp_path, job_id, _stub(tmp_path / "bin", "py-spy", _PY_SPY_FAIL))
    assert result.returncode == 0, result.stderr
    assert "0 stack(s) written, 2 failed" in result.stdout
    assert "py-spy failed on rank 3" in result.stdout
    assert "could not attach" in (out / "rank_3.stack").read_text()


def test_a_fifo_with_a_reader_is_triggered(tmp_path, job_id, pipe_prefix, ranks):
    """The FIFO is the one the rank's own environment names: <prefix><rank>.pipe."""
    pipe = f"{pipe_prefix}3.pipe"
    os.mkfifo(pipe)
    reader = os.open(pipe, os.O_RDONLY | os.O_NONBLOCK)
    try:
        _, result = _run_node(tmp_path, job_id, _stub(tmp_path / "bin", "py-spy", _PY_SPY_OK))
        assert "1 FIFO(s) triggered, 0 without a reader, 1 rank(s) opened no FIFO" in result.stdout
        assert os.read(reader, 64) == b"dump\n"
    finally:
        os.close(reader)


def test_a_fifo_without_a_reader_is_counted_not_hung_on(tmp_path, job_id, pipe_prefix, ranks):
    os.mkfifo(f"{pipe_prefix}10.pipe")
    _, result = _run_node(tmp_path, job_id, _stub(tmp_path / "bin", "py-spy", _PY_SPY_OK))
    assert "0 FIFO(s) triggered, 1 without a reader, 1 rank(s) opened no FIFO" in result.stdout


def test_usage_is_refused_without_arguments(tmp_path):
    result = subprocess.run(["bash", SCRIPT], capture_output=True, text=True, env=_env(tmp_path), timeout=30)
    assert result.returncode == 2
    assert "usage" in result.stderr


def _slurm_stubs(tmp_path, squeue=_SQUEUE, scontrol=_SCONTROL):
    bindir = tmp_path / "bin"
    _stub(bindir, "py-spy", _PY_SPY_OK)
    _stub(bindir, "squeue", squeue)
    _stub(bindir, "scontrol", scontrol)
    _stub(bindir, "srun", _SRUN)
    return bindir


def _run_driver(tmp_path, job_id, bindir, extra_env=None):
    log_dir = tmp_path / "megatron_runs"
    log_dir.mkdir(exist_ok=True)
    env = _env(
        bindir,
        SCONTROL_STDOUT=str(log_dir / f"train-{job_id}.out"),
        SRUN_ARGS_FILE=str(tmp_path / "srun.args"),
        SRUN_ENV_FILE=str(tmp_path / "srun.env"),
    )
    env.update(extra_env or {})
    result = subprocess.run(["bash", SCRIPT, job_id], capture_output=True, text=True, env=env, timeout=120)
    return log_dir, result


def test_driver_places_the_evidence_beside_the_raw_log_and_sweeps_every_node(tmp_path, job_id, ranks):
    log_dir, result = _run_driver(tmp_path, job_id, _slurm_stubs(tmp_path))
    assert result.returncode == 0, result.stderr
    evidence = log_dir / "nccl_trace" / job_id
    assert sorted(p.name for p in evidence.iterdir()) == ["rank_10.stack", "rank_3.stack"]
    assert f"evidence: {evidence}" in result.stdout
    args = (tmp_path / "srun.args").read_text().splitlines()
    assert f"--jobid={job_id}" in args
    assert "--nodes=2" in args and "--ntasks=2" in args and "--overlap" in args


def test_driver_unsets_the_session_slurm_variables_before_srun(tmp_path, job_id, ranks):
    """Inside an allocation, an inherited SLURM_JOB_ID would clamp srun to that allocation."""
    _, result = _run_driver(tmp_path, job_id, _slurm_stubs(tmp_path), {"SLURM_JOB_ID": "999", "SLURM_NNODES": "1"})
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "srun.env").read_text().strip() == "SLURM_JOB_ID=unset"


def test_an_unknown_job_is_reported(tmp_path, job_id):
    _, result = _run_driver(tmp_path, job_id, _slurm_stubs(tmp_path), {"SQUEUE_UNKNOWN_JOB": "1"})
    assert result.returncode == 1
    assert f"job {job_id} is not in the queue" in result.stderr


def test_a_job_without_a_raw_log_is_reported(tmp_path, job_id):
    _, result = _run_driver(tmp_path, job_id, _slurm_stubs(tmp_path), {"SCONTROL_STDOUT": ""})
    assert result.returncode == 1
    assert "has no StdOut" in result.stderr


def test_a_missing_py_spy_is_reported_before_the_sweep(tmp_path, job_id):
    _, result = _run_driver(tmp_path, job_id, _slurm_stubs(tmp_path), {"PY_SPY": str(tmp_path / "absent")})
    assert result.returncode == 1
    assert "py-spy not found" in result.stderr
    assert not (tmp_path / "srun.args").exists()
