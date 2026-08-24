"""Distributed-setup tests for pipeline_training_launch.sh: node subsets, ports, busy nodes.

These are the launcher's shared-allocation safety rules, and every one of them exists
because a bug in it silently corrupted a run rather than failing:

  * NNODES is DERIVED from the effective nodelist. When --nodes exceeded the nodelist, srun
    padded the step onto nodes OUTSIDE the list (man srun -w) — in a shared allocation, onto
    a co-tenant's GPUs.
  * MASTER_PORT is a hash of (job id, nodelist). While it was job-id-only, two subsets of
    one allocation got the SAME port, so two runs whose subsets shared a head node joined
    each other's c10d TCPStore (the store is keyed by host:port alone).
  * The busy-node guard exists because the launch sruns use --overlap, which removes
    SLURM's own refusal to double-book a node.

The logic is shell, so it is tested AS shell: the distributed-setup block is extracted from
the real script between its own section banners and run under bash with stub `scontrol` and
`srun` on PATH (SLURM is the untestable boundary; the code under test is the real code).
The extraction asserts its markers were found, so a refactor that moves the block fails
these tests loudly instead of silently testing nothing.

Run:
    ./pipeline_env_exec.sh "cd <repo>; source pipeline_env_activate.sh || exit 1; cd /tmp; \\
        python -m pytest <repo>/tests/unit_tests/test_pipeline_training_launch_subsets.py -v"
"""

from __future__ import annotations

import os
import re
import stat
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
LAUNCH_SCRIPT = REPO_ROOT / "pipeline_training_launch.sh"

# Section banners bounding the block under test, as written in the launcher.
BLOCK_BEGIN = "# Distributed setup"
BLOCK_END = "# Select training script"

# Ports are spaced 3 apart over 300 slots because inprocess_restart claims +1/+2.
PORT_BASE = 29500
PORT_SLOTS = 300
PORT_SPACING = 3


def _extract_block() -> str:
    """The launcher's distributed-setup + busy-node-guard block, verbatim."""
    lines = LAUNCH_SCRIPT.read_text().splitlines()
    starts = [i for i, line in enumerate(lines) if line.strip() == BLOCK_BEGIN]
    ends = [i for i, line in enumerate(lines) if line.strip() == BLOCK_END]
    assert len(starts) == 1, f"{LAUNCH_SCRIPT}: expected exactly one {BLOCK_BEGIN!r} banner, found {len(starts)}"
    assert len(ends) == 1, f"{LAUNCH_SCRIPT}: expected exactly one {BLOCK_END!r} banner, found {len(ends)}"
    assert starts[0] < ends[0]
    block = "\n".join(lines[starts[0] : ends[0]])
    # Guard against extracting a block that no longer contains what these tests drive.
    for needed in ("NNODES=", "MASTER_PORT=", "TRAIN_ALLOW_BUSY_NODES"):
        assert needed in block, f"extracted block no longer contains {needed!r}; update this test"
    return block


_STUB_SCONTROL = """#!/usr/bin/env python3
"Minimal `scontrol show hostname[s] <nodelist>` stub: comma lists and one prefix[a-b] range."
import re
import sys

spec = sys.argv[-1]
match = re.fullmatch(r"(.*)\\[(\\d+)-(\\d+)\\]", spec)
if match:
    prefix, first, last = match.group(1), match.group(2), match.group(3)
    width = len(first)
    hosts = [f"{prefix}{index:0{width}d}" for index in range(int(first), int(last) + 1)]
else:
    hosts = [part for part in spec.split(",") if part]
print("\\n".join(hosts))
"""

# One stub for all three srun behaviours the guard must handle, selected by env var.
_STUB_SRUN = """#!/bin/bash
case "${STUB_SRUN_MODE:-clean}" in
    busy) echo "nid001:2"; exit 0 ;;
    fail) echo "srun: error: Unable to create step" >&2; exit 1 ;;
    clean) exit 0 ;;
    *) echo "stub srun: unknown STUB_SRUN_MODE" >&2; exit 99 ;;
esac
"""


@pytest.fixture(scope="module")
def block() -> str:
    return _extract_block()


@pytest.fixture(scope="module")
def stub_bin(tmp_path_factory) -> Path:
    """A PATH dir holding stub `scontrol` and `srun`."""
    bindir = tmp_path_factory.mktemp("slurm_stubs")
    for name, body in (("scontrol", _STUB_SCONTROL), ("srun", _STUB_SRUN)):
        path = bindir / name
        path.write_text(body)
        path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return bindir


def _run_block(block, stub_bin, tmp_path, env_overrides=None, job_id="4242", nodelist="nid[001-008]", nnodes="8"):
    """Run the extracted block under bash; return (returncode, parsed vars, stderr)."""
    script = tmp_path / "distributed_setup.sh"
    script.write_text(
        "set -euo pipefail\n" + block + "\n" + 'echo "OUT_NNODES=$NNODES"\n'
        'echo "OUT_MASTER_ADDR=$MASTER_ADDR"\n'
        'echo "OUT_MASTER_PORT=$MASTER_PORT"\n'
        'echo "OUT_TOTAL_GPUS=$TOTAL_GPUS"\n'
    )
    env = dict(os.environ)
    env["PATH"] = f"{stub_bin}:{env['PATH']}"
    env["SLURM_JOB_ID"] = job_id
    env["SLURM_NODELIST"] = nodelist
    env["SLURM_NNODES"] = nnodes
    for key in ("OVERRIDE_NODES", "OVERRIDE_NODELIST", "MASTER_ADDR_OVERRIDE", "MASTER_PORT_OVERRIDE"):
        env.pop(key, None)
    env.setdefault("STUB_SRUN_MODE", "clean")
    env.update(env_overrides or {})

    result = subprocess.run(["bash", str(script)], capture_output=True, text=True, env=env, timeout=60)
    values = dict(line.split("=", 1) for line in result.stdout.splitlines() if line.startswith("OUT_") and "=" in line)
    return result.returncode, {key[len("OUT_") :]: value for key, value in values.items()}, result.stderr


# ---------------------------------------------------------------------------------------
# Node-count derivation.
# ---------------------------------------------------------------------------------------


def test_nodelist_alone_derives_the_node_count(block, stub_bin, tmp_path):
    """The srun-padding regression: NNODES must follow the nodelist, not the allocation."""
    code, out, err = _run_block(
        block,
        stub_bin,
        tmp_path,
        env_overrides={"OVERRIDE_NODELIST": "nid001,nid002,nid003,nid004"},
        nnodes="8",
    )
    assert code == 0, err
    assert out["NNODES"] == "4", "NNODES stayed at the allocation size — srun would pad outside the subset"
    assert out["TOTAL_GPUS"] == "16"
    assert out["MASTER_ADDR"] == "nid001"


def test_nodes_mismatching_the_nodelist_is_fatal(block, stub_bin, tmp_path):
    code, out, err = _run_block(
        block,
        stub_bin,
        tmp_path,
        env_overrides={"OVERRIDE_NODELIST": "nid001,nid002,nid003,nid004", "OVERRIDE_NODES": "8"},
    )
    assert code != 0
    assert "FATAL" in err and "does not match" in err
    assert "NNODES" not in out, "the launch continued past a nodes/nodelist mismatch"


def test_nodes_matching_the_nodelist_passes(block, stub_bin, tmp_path):
    code, out, err = _run_block(
        block,
        stub_bin,
        tmp_path,
        env_overrides={"OVERRIDE_NODELIST": "nid001,nid002,nid003,nid004", "OVERRIDE_NODES": "4"},
    )
    assert code == 0, err
    assert out["NNODES"] == "4"
    assert "FATAL" not in err


def test_nodes_alone_keeps_allocation_semantics_and_warns(block, stub_bin, tmp_path):
    """--nodes without --nodelist still lets srun pick, so it must say so."""
    code, out, err = _run_block(block, stub_bin, tmp_path, env_overrides={"OVERRIDE_NODES": "4"}, nnodes="8")
    assert code == 0, err
    assert out["NNODES"] == "4"
    assert "WARNING" in err and "--nodelist" in err


def test_no_overrides_uses_the_whole_allocation_without_warning(block, stub_bin, tmp_path):
    code, out, err = _run_block(block, stub_bin, tmp_path, nnodes="8")
    assert code == 0, err
    assert out["NNODES"] == "8"
    assert out["TOTAL_GPUS"] == "32"
    assert "WARNING" not in err


# ---------------------------------------------------------------------------------------
# MASTER_PORT derivation.
# ---------------------------------------------------------------------------------------


def _port(block, stub_bin, tmp_path, nodelist_override, job_id="4242"):
    code, out, err = _run_block(
        block, stub_bin, tmp_path, env_overrides={"OVERRIDE_NODELIST": nodelist_override}, job_id=job_id
    )
    assert code == 0, err
    return int(out["MASTER_PORT"])


def test_different_subsets_of_one_job_get_different_ports(block, stub_bin, tmp_path):
    """The TCPStore-collision regression: the port must depend on the nodelist, not the job alone."""
    first = _port(block, stub_bin, tmp_path, "nid001,nid002,nid003,nid004")
    second = _port(block, stub_bin, tmp_path, "nid005,nid006,nid007,nid008")
    assert first != second, "both subsets of one allocation would share a c10d rendezvous"


def test_the_same_subset_is_deterministic(block, stub_bin, tmp_path):
    subset = "nid001,nid002,nid003,nid004"
    assert _port(block, stub_bin, tmp_path, subset) == _port(block, stub_bin, tmp_path, subset)


def test_ports_are_spaced_three_apart_in_the_expected_window(block, stub_bin, tmp_path):
    """inprocess_restart claims MASTER_PORT+1/+2, so slots must be 3 apart."""
    for subset in ("nid001", "nid001,nid002", "nid003,nid004,nid005", "nid[001-008]"):
        port = _port(block, stub_bin, tmp_path, subset)
        assert PORT_BASE <= port < PORT_BASE + PORT_SLOTS * PORT_SPACING
        assert (port - PORT_BASE) % PORT_SPACING == 0, f"{subset}: port {port} is not on a 3-spaced slot"


def test_master_overrides_win(block, stub_bin, tmp_path):
    code, out, err = _run_block(
        block,
        stub_bin,
        tmp_path,
        env_overrides={
            "OVERRIDE_NODELIST": "nid001,nid002",
            "MASTER_PORT_OVERRIDE": "31337",
            "MASTER_ADDR_OVERRIDE": "nid007",
        },
    )
    assert code == 0, err
    assert out["MASTER_PORT"] == "31337"
    assert out["MASTER_ADDR"] == "nid007"


# ---------------------------------------------------------------------------------------
# Busy-node guard.
# ---------------------------------------------------------------------------------------


def test_busy_nodes_are_fatal(block, stub_bin, tmp_path):
    code, out, err = _run_block(
        block, stub_bin, tmp_path, env_overrides={"OVERRIDE_NODELIST": "nid001,nid002", "STUB_SRUN_MODE": "busy"}
    )
    assert code != 0
    assert "FATAL" in err and "already running" in err
    assert "nid001:2" in err, "the guard must name the busy node and process count"


def test_idle_nodes_proceed(block, stub_bin, tmp_path):
    code, out, err = _run_block(
        block, stub_bin, tmp_path, env_overrides={"OVERRIDE_NODELIST": "nid001,nid002", "STUB_SRUN_MODE": "clean"}
    )
    assert code == 0, err
    assert out["NNODES"] == "2"


def test_a_failing_probe_is_fatal(block, stub_bin, tmp_path):
    """Fail closed: an srun that cannot run the probe proves nothing about the nodes."""
    code, out, err = _run_block(
        block, stub_bin, tmp_path, env_overrides={"OVERRIDE_NODELIST": "nid001,nid002", "STUB_SRUN_MODE": "fail"}
    )
    assert code != 0, "an unusable probe was treated as 'nodes are idle'"
    assert "FATAL" in err


def test_the_guard_can_be_waived_deliberately(block, stub_bin, tmp_path):
    code, out, err = _run_block(
        block,
        stub_bin,
        tmp_path,
        env_overrides={
            "OVERRIDE_NODELIST": "nid001,nid002",
            "STUB_SRUN_MODE": "busy",
            "TRAIN_ALLOW_BUSY_NODES": "1",
        },
    )
    assert code == 0, err
    assert "FATAL" not in err


# ---------------------------------------------------------------------------------------
# TMPDIR: one value, host-side, reaching the ranks.
# ---------------------------------------------------------------------------------------


def test_tmpdir_is_the_job_scoped_container_path():
    """The host TMPDIR value the ranks must inherit, evaluated from the real assignments."""
    source = LAUNCH_SCRIPT.read_text()
    suffix_line = next(line for line in source.splitlines() if line.startswith("ENV_CACHE_SUFFIX="))
    tmpdir_line = next(line for line in source.splitlines() if line.startswith("export TMPDIR="))
    result = subprocess.run(
        ["bash", "-c", f'set -eu\nSLURM_JOB_ID=4242\n{suffix_line}\n{tmpdir_line}\necho "$TMPDIR"'],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "/tmp/megatron_4242_container"


def test_the_torchrun_remote_command_exports_the_host_tmpdir():
    """`export TMPDIR=$TMPDIR` unescaped: the ranks get the host value, not a fresh name."""
    source = LAUNCH_SCRIPT.read_text()
    assert "export TMPDIR=$TMPDIR" in source
    assert re.search(r"mkdir -p \\\$TMPDIR", source), "the remote mkdir must defer to the remote shell"
    assert "megatron_tmp_" not in source, "a second, divergent TMPDIR naming scheme is back"
