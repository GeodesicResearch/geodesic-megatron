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
"""Switch-placement derivation in pipeline_training_launch.sh.

The launcher records a run's Dragonfly placement so a throughput number can be compared
only against another taken at the same spread. The derivation shells out to `scontrol`,
which exists on the host but not inside the container, and it carries a load-bearing
promise: it must never stop a training launch.

That promise is easy to break and impossible to notice by inspection. The launcher runs
under `set -euo pipefail`, so a `while` loop whose LAST iteration ends on a false test
makes the enclosing pipeline nonzero -- even when the pipeline's tail succeeded -- and the
command substitution then aborts the launcher. The failing case is ordinary: a topology
whose last switch happens to share no node with the job, which is every job that does not
reach the final switch group.

These tests run the real function out of the real script with a stub `scontrol` on PATH.
SLURM is the untestable boundary; the shell under test is not stubbed.
"""

import os
import re
import stat
import subprocess

import pytest


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LAUNCHER = os.path.join(REPO_ROOT, "pipeline_training_launch.sh")

# Two leaf switches and a spine, matching `scontrol show topology` on Isambard. group9 is
# LAST and deliberately shares no node with the job in most cases below -- that ordering is
# the whole point of the fixture.
TOPOLOGY = (
    "SwitchName=group2 Level=0 Nodes=nid[010000-010002]\n"
    "SwitchName=group9 Level=0 Nodes=nid[010900-010902]\n"
    "SwitchName=global Level=1 Nodes=nid[010000-010902]\n"
)

_STUB = """#!/bin/bash
# FAIL_SCONTROL names the subcommand to fail -- "hostnames", "topology", or "all".
# Failing one at a time is the point: with everything failing, the node list comes back
# empty and the function returns early, never reaching the topology pipeline whose exit
# status is what the caller's `|| true` exists to absorb.
if [ "${{FAIL_SCONTROL:-}}" = "all" ] || [ "${{FAIL_SCONTROL:-}}" = "$2" ]; then
    echo "scontrol: connection refused" >&2
    exit 1
fi
case "$2" in
  hostnames)
    spec=$3; base=${{spec#nid[}}; base=${{base%]}}
    lo=${{base%%-*}}; hi=${{base##*-}}
    for i in $(seq "$((10#$lo))" "$((10#$hi))"); do printf 'nid%06d\\n' "$i"; done ;;
  topology)
    cat <<'TOPO'
{topology}TOPO
    ;;
esac
"""


def _extract_function():
    """The derivation, lifted verbatim from the real launcher.

    The launcher cannot be sourced whole -- it allocates nodes and execs srun -- so the
    function is extracted by name. If it is ever renamed this test fails loudly rather
    than silently testing nothing.
    """
    src = open(LAUNCHER).read()
    match = re.search(r"^derive_switch_spread\(\) \{.*?^\}", src, re.S | re.M)
    assert match, "derive_switch_spread() not found in pipeline_training_launch.sh"
    return match.group(0)


def _extract_assignment():
    """The launcher's own call site, so the `|| true` net is under test too.

    Reproducing the assignment by hand would leave that net untested: the function returns
    0 on its own in the topology-failure case, so a future edit dropping `|| true` would
    reintroduce an abort on the scontrol-failure path with every test still green.
    """
    src = open(LAUNCHER).read()
    match = re.search(r'^\s*ISAMBARD_SWITCH_SPREAD="\$\(derive_switch_spread .*\)"$', src, re.M)
    assert match, "the ISAMBARD_SWITCH_SPREAD assignment was not found in the launcher"
    return match.group(0).strip()


def _run(tmp_path, nodelist, fail_scontrol=None):
    """Run the derivation under the launcher's own shell options, with a stub scontrol."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    stub = bindir / "scontrol"
    stub.write_text(_STUB.format(topology=TOPOLOGY))
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC)

    script = tmp_path / "harness.sh"
    script.write_text(
        "set -euo pipefail\n"
        f"{_extract_function()}\n"
        f'NODELIST="{nodelist}"\n'
        f"{_extract_assignment()}\n"
        'printf "SPREAD=[%s]\\n" "$ISAMBARD_SWITCH_SPREAD"\n'
        'echo "REACHED_END"\n'
    )
    env = dict(os.environ, PATH=f"{bindir}:{os.environ['PATH']}")
    if fail_scontrol:
        env["FAIL_SCONTROL"] = fail_scontrol
    return subprocess.run(["bash", str(script)], capture_output=True, text=True, env=env, timeout=120)


def _spread(result):
    match = re.search(r"SPREAD=\[(.*)\]", result.stdout)
    assert match, f"harness produced no SPREAD line: {result.stdout!r} {result.stderr!r}"
    return match.group(1)


def test_reports_the_switches_the_job_spans(tmp_path):
    result = _run(tmp_path, "nid[010000-010001]")
    assert result.returncode == 0
    assert _spread(result) == "group2:2"


def test_counts_every_switch_when_the_job_spans_more_than_one(tmp_path):
    """Every spanned switch is reported with its node count.

    Order is by count descending so the dominant group reads first, but equal counts tie
    and `sort` does not define their relative order -- so this asserts the set, not the
    string. `test_orders_switches_by_size` covers the ordering with distinct counts.
    """
    result = _run(tmp_path, "nid[010000-010902]")
    assert result.returncode == 0
    assert set(_spread(result).split(",")) == {"group2:3", "group9:3"}


def test_orders_switches_by_size(tmp_path):
    """Largest first: on a fragmented allocation the dominant group is the useful number."""
    result = _run(tmp_path, "nid[010001-010902]")
    assert result.returncode == 0
    assert _spread(result) == "group9:3,group2:2"


def test_survives_a_last_switch_that_shares_no_node(tmp_path):
    """The regression that aborted every launch.

    With `set -euo pipefail`, a while-loop ending on a false test propagates through
    `pipefail` past a successful pipeline tail, so the command substitution returns
    nonzero and `set -e` kills the script -- after computing the right answer.
    """
    result = _run(tmp_path, "nid[010000-010001]")
    assert "REACHED_END" in result.stdout, (
        "the launcher aborted after the assignment; placement telemetry must never "
        f"stop a run. stderr={result.stderr!r}"
    )
    assert result.returncode == 0


def test_scontrol_failure_is_non_fatal_and_not_silent(tmp_path):
    result = _run(tmp_path, "nid[010000-010001]", fail_scontrol="all")
    assert result.returncode == 0
    assert "REACHED_END" in result.stdout
    assert _spread(result) == "", "a failed scontrol must yield no placement, not a wrong one"
    assert "connection refused" in result.stderr, (
        "scontrol's error must reach the operator rather than being swallowed"
    )


def test_topology_failure_alone_is_non_fatal(tmp_path):
    """The one case the caller's `|| true` is load-bearing for.

    When every `scontrol` call fails the function returns early on an empty node list and
    its own exit status is 0, so `|| true` is doing nothing. It is only when `show
    hostnames` SUCCEEDS and `show topology` fails that the pipeline runs, exits nonzero,
    and the command substitution would abort the launcher without the net. Removing
    `|| true` from the launcher must turn this test red.
    """
    result = _run(tmp_path, "nid[010000-010001]", fail_scontrol="topology")
    assert result.returncode == 0, (
        "the launcher aborted on a topology-only scontrol failure; the `|| true` on the "
        f"ISAMBARD_SWITCH_SPREAD assignment is what prevents that. stderr={result.stderr!r}"
    )
    assert "REACHED_END" in result.stdout
    assert _spread(result) == ""
    assert "connection refused" in result.stderr


@pytest.mark.parametrize("nodelist", ["nid[010500-010501]", "nid[010903-010904]"])
def test_no_matching_switch_yields_no_placement(tmp_path, nodelist):
    """Nodes outside every leaf switch: absent placement beats an invented one."""
    result = _run(tmp_path, nodelist)
    assert result.returncode == 0
    assert _spread(result) == ""
