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
"""The ISAMBARD_ENV_OVERRIDES hook of pipeline_training_launch.sh (pipeline/sh/env_overrides.sh).

An overrides file carries KEY=VALUE lines for one launch's environment. Three layers set that
environment, so the hook applies each override three times: before the launcher reads its knobs,
after the launcher's own exports, and inside the container payload after activation, immediately
before the rank launcher. What must hold: values are taken literally everywhere; the knobs the
launcher reads and the defaults it and activate set both yield to an override; a file that is
malformed, or that names one of the launcher's own shell variables, stops the launch before
anything happens; the ranks learn which keys were overridden; and an unset hook changes nothing.

The first group runs the real `apply_env_overrides` and `export_env_overrides`, lifted verbatim
from their library, under the launcher's shell options. The second runs the whole launcher inside
the test container (unit tests run there), through the real pipeline_env_exec.sh and
pipeline_env_activate.sh, and reads the environment and arguments the rank launcher is started
with. It runs under tests/unit_tests/stubbed_shell.py's PipelineShell, whose stubs (each with its
reason) stand in for SLURM, the container runtime, the rank launchers and the clock. The same harness pins where that environment puts HybridEP's JIT
cache: the job's node-local temp directory, not the bind-mounted host home. The last test sources the
whole launcher to list the shell variables it keeps, and checks that the hook refuses every one of them
as a KEY.
"""

import json
import os
import re
import subprocess
import sys

import pytest
from scripts.training.launcher_source import LAUNCHER, launcher_function

from tests.unit_tests.launcher_harness import LAUNCH_ENV, LITERAL_VALUES, RANK_LAUNCHERS, Launcher, launcher_shell
from tests.unit_tests.stubbed_shell import HARNESS_ENV


# --- apply_env_overrides + export_env_overrides, lifted from their library ----------------------

_BARE_ENV = {"PATH": os.environ["PATH"], "HOME": "/nonexistent/home", "USER": "tester"}


def _dump_env_cmd() -> str:
    return f"{sys.executable} -c 'import json, os; print(json.dumps(dict(os.environ)))'"


def _hook_functions() -> str:
    return f"{launcher_function('apply_env_overrides')}\n{launcher_function('export_env_overrides')}\n"


def _apply(tmp_path, content, inherited=None):
    """Run both hook steps on a file holding ``content`` (None: no file); return result and envs.

    The host environment is what the launcher's own shell exports after the hook; the payload
    environment is what the hook's payload statements alone produce in a clean shell.
    """
    overrides = tmp_path / "overrides.env"
    if content is not None:
        overrides.write_text(content)
    payload_file = tmp_path / "payload.sh"
    harness = tmp_path / "harness.sh"
    harness.write_text(
        "set -euo pipefail\n"
        f"{_hook_functions()}"
        f'apply_env_overrides "{overrides}"\n'
        "export_env_overrides\n"
        f'printf "%s" "$ENV_OVERRIDES_PAYLOAD" > "{payload_file}"\n'
        f"{_dump_env_cmd()}\n"
    )
    env = {**_BARE_ENV, **(inherited or {})}
    result = subprocess.run(["bash", str(harness)], capture_output=True, text=True, env=env, timeout=60)
    if result.returncode != 0:
        return result, None, None
    host_env = json.loads(result.stdout)
    payload = subprocess.run(
        ["bash", "-c", f"set -euo pipefail\n{payload_file.read_text()}\n{_dump_env_cmd()}"],
        capture_output=True,
        text=True,
        env=_BARE_ENV,
        timeout=60,
    )
    assert payload.returncode == 0, payload.stderr
    return result, host_env, json.loads(payload.stdout)


def _assert_refused(result, message):
    assert result.returncode != 0
    assert "FATAL" in result.stderr and message in result.stderr, result.stderr
    assert result.stdout == "", "the hook must stop before the launch continues"


def test_values_are_taken_literally_on_the_host_and_in_the_payload(tmp_path):
    content = "".join(f"{key}={value}\n" for key, value in LITERAL_VALUES.items())
    result, host_env, payload_env = _apply(tmp_path, content)
    assert result.returncode == 0, result.stderr
    for key, value in LITERAL_VALUES.items():
        assert host_env[key] == value, key
        assert payload_env[key] == value, key


def test_keys_reach_both_sides_in_file_order(tmp_path):
    _, host_env, payload_env = _apply(tmp_path, "B_SECOND=2\nA_FIRST=1\nC_THIRD=3\n")
    assert host_env["ISAMBARD_ENV_OVERRIDE_KEYS"] == "B_SECOND,A_FIRST,C_THIRD"
    assert payload_env["ISAMBARD_ENV_OVERRIDE_KEYS"] == "B_SECOND,A_FIRST,C_THIRD"


def test_plain_lowercase_keys_are_not_shadowed_by_the_hook(tmp_path):
    # Short lowercase names are what a shell function's own locals are usually called.
    content = "A=1\ni=0\nfile=f\nline=l\nkey=k\nB=2\n"
    result, host_env, payload_env = _apply(tmp_path, content)
    assert result.returncode == 0, result.stderr
    expected = {"A": "1", "i": "0", "file": "f", "line": "l", "key": "k", "B": "2"}
    for env in (host_env, payload_env):
        assert {k: env[k] for k in expected} == expected
        assert env["ISAMBARD_ENV_OVERRIDE_KEYS"] == "A,i,file,line,key,B"


def test_comments_blank_lines_and_a_missing_final_newline(tmp_path):
    content = "# probe overrides\n\n   \nFIRST=1\n#SECOND=2\nTHIRD=3"
    result, host_env, payload_env = _apply(tmp_path, content)
    assert result.returncode == 0, result.stderr
    assert host_env["ISAMBARD_ENV_OVERRIDE_KEYS"] == "FIRST,THIRD"
    assert (payload_env["FIRST"], payload_env["THIRD"]) == ("1", "3")
    assert "SECOND" not in host_env and "SECOND" not in payload_env


def test_an_empty_file_overrides_nothing(tmp_path):
    result, host_env, payload_env = _apply(tmp_path, "# nothing to override\n")
    assert result.returncode == 0, result.stderr
    assert host_env["ISAMBARD_ENV_OVERRIDE_KEYS"] == ""
    assert payload_env["ISAMBARD_ENV_OVERRIDE_KEYS"] == ""


@pytest.mark.parametrize(
    "line",
    [
        "NO_EQUALS_SIGN",
        "=value",
        "1STARTS_WITH_DIGIT=x",
        "HAS-DASH=x",
        " LEADING_SPACE=x",
        "export TORCH_NCCL_BLOCKING_WAIT=0",
        "  # indented comment",
        "CRLF=0\r",
    ],
)
def test_a_malformed_line_is_fatal(tmp_path, line):
    result, _, _ = _apply(tmp_path, f"GOOD=1\n{line}\n")
    _assert_refused(result, ":2:")


def test_a_missing_file_is_fatal(tmp_path):
    result, _, _ = _apply(tmp_path, None)
    _assert_refused(result, "ISAMBARD_ENV_OVERRIDES names no file")


def test_a_repeated_key_is_fatal(tmp_path):
    result, _, _ = _apply(tmp_path, "OMP_NUM_THREADS=4\nOMP_NUM_THREADS=8\n")
    _assert_refused(result, "OMP_NUM_THREADS is set more than once")


@pytest.mark.parametrize(
    "key, message",
    [
        # The launcher's command line and what it derives from it.
        ("MODEL", "is set by the launcher itself"),
        ("USE_FT", "is set by the launcher itself"),
        ("NNODES", "is set by the launcher itself"),
        ("SCRIPT_ARGS", "is set by the launcher itself"),
        ("REPO_DIR", "is set by the launcher itself"),
        # What chooses the checkout.
        ("GEODESIC_REPO_DIR", "is set by the launcher itself"),
        ("TRAIN_REPO_DIR", "is set by the launcher itself"),
        # The hook's own variables, including the payload it renders.
        ("ISAMBARD_ENV_OVERRIDES", "is set by the launcher itself"),
        ("ISAMBARD_ENV_OVERRIDE_KEYS", "is set by the launcher itself"),
        ("ENV_OVERRIDES_PAYLOAD", "is set by the launcher itself"),
        ("ENV_OVERRIDE_ENTRIES", "is set by the launcher itself"),
        ("_eo_line", "uses the _eo_ prefix"),
        # Derived by pipeline_env_config.env from GEODESIC_CONTAINER_* on every node.
        ("CONTAINER_SIF", "override its GEODESIC_CONTAINER_* input"),
        ("CONTAINER_PYTHON_OVERLAY", "override its GEODESIC_CONTAINER_* input"),
        # Bash's own shell variables.
        ("IFS", "is a shell variable of the launcher"),
        ("OPTIND", "is a shell variable of the launcher"),
    ],
)
def test_a_variable_of_the_launcher_itself_is_fatal(tmp_path, key, message):
    result, _, _ = _apply(tmp_path, f"GOOD=1\n{key}=x\n")
    _assert_refused(result, f":2: {key} ")
    assert message in result.stderr


def test_a_launcher_variable_is_refused_even_when_the_caller_exports_it(tmp_path):
    # The attribute check alone would take an exported MODEL for an environment variable.
    result, _, _ = _apply(tmp_path, "MODEL=super\n", inherited={"MODEL": "nano"})
    _assert_refused(result, "MODEL is set by the launcher itself")


@pytest.mark.parametrize("key", ["HF_TOKEN", "WANDB_API_KEY", "AWS_SECRET_ACCESS_KEY", "GITHUB_TOKEN", "db_password"])
def test_a_credential_is_fatal(tmp_path, key):
    result, _, _ = _apply(tmp_path, f"{key}=hunter2\n")
    _assert_refused(result, f"{key} looks like a credential")
    assert "hunter2" not in result.stderr


def test_a_key_merely_containing_a_credential_word_is_accepted(tmp_path):
    result, host_env, _ = _apply(tmp_path, "TOKENIZERS_PARALLELISM=false\n")
    assert result.returncode == 0, result.stderr
    assert host_env["TOKENIZERS_PARALLELISM"] == "false"


# --- the whole launcher, in the container -------------------------------------------------------

# Variables the harness itself puts into the environment, excluded from before/after diffs.
_HARNESS_KEYS = set(HARNESS_ENV)


def _job_tmpdirs(shell):
    return (f"/tmp/megatron_{shell.job_id}_container", f"/tmp/megatron_tmp_{shell.job_id}_container")


@pytest.fixture
def shell(tmp_path):
    shell = launcher_shell(tmp_path)
    yield shell
    # The launcher creates its job-scoped node-local TMPDIRs; this fake job must not leave them.
    shell.remove_job_tmp_dirs()


@pytest.fixture
def launch(tmp_path, shell):
    """Run the real launcher once per call, with an overrides file holding ``overrides``."""
    return Launcher(tmp_path, shell)


LAUNCH_PATHS = [pytest.param((), id="ft_launcher"), pytest.param(("--disable-ft",), id="torchrun")]


@pytest.mark.parametrize("launcher_args", LAUNCH_PATHS)
def test_overrides_beat_launcher_and_activate_defaults_at_the_ranks(launch, tmp_path, launcher_args):
    override_tmp = tmp_path / "override_tmp"
    override_tmp.mkdir()
    overrides = {
        "TORCH_NCCL_BLOCKING_WAIT": "0",  # launcher: unconditional 1
        "OMP_NUM_THREADS": "3",  # pipeline_env_activate.sh: 8
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:False",  # pipeline_env_activate.sh
        "TMPDIR": str(override_tmp),  # the launcher's export, and the --disable-ft payload's
        "CAMPAIGN_NOTE": "$HOME stays 'literal'",
    }
    run = launch("# arm\n" + "".join(f"{k}={v}\n" for k, v in overrides.items()), *launcher_args)
    assert run.result.returncode == 0, run.result.stderr
    assert run.rank_env is not None, f"the rank launcher never started:\n{run.result.stdout}\n{run.result.stderr}"
    for key, value in overrides.items():
        assert run.rank_env[key] == value, key
        assert f"\n  {key}={value}\n" in run.result.stdout, f"banner does not list {key}"
    assert run.rank_env["ISAMBARD_ENV_OVERRIDE_KEYS"] == ",".join(overrides)


@pytest.mark.parametrize("launcher_args", LAUNCH_PATHS)
def test_overrides_reach_the_knobs_the_launcher_reads(launch, shell, tmp_path, launcher_args):
    """Knobs the launcher consumes early take the override, not just the ranks' copy of it."""
    plain = launch(None, *launcher_args)
    assert plain.result.returncode == 0, plain.result.stderr
    default_sif = re.search(r"^Env: +container \((.+)\)$", plain.result.stdout, re.M).group(1)
    alternative_sif = tmp_path / "alternative.sif"
    alternative_sif.symlink_to(default_sif)
    (tmp_path / "alternative.sif.source.txt").symlink_to(f"{default_sif}.source.txt")

    run = launch(
        "ISAMBARD_NCCL_DEBUG=INFO\n"
        "TRAIN_PERSISTENT_TRITON_CACHE=1\n"
        "TRAIN_FI_CXI_RX_MATCH_MODE_OVERRIDE=hybrid\n"
        "ISAMBARD_RUN_ID=override-run-id\n"
        f"GEODESIC_CONTAINER_SIF={alternative_sif}\n",
        *launcher_args,
    )
    assert run.result.returncode == 0, run.result.stderr
    assert plain.rank_env["NCCL_DEBUG"] == "WARN"
    assert run.rank_env["NCCL_DEBUG"] == "INFO"
    assert plain.rank_env["TRITON_CACHE_DIR"] == f"/tmp/triton_cache_{shell.job_id}_container"
    assert run.rank_env["TRITON_CACHE_DIR"] == "/tmp/triton_cache_persistent_container"
    assert (plain.rank_env["FI_CXI_RX_MATCH_MODE"], run.rank_env["FI_CXI_RX_MATCH_MODE"]) == ("soft", "hybrid")
    # The run's identity is the override everywhere the launcher records it.
    assert run.rank_env["ISAMBARD_RUN_ID"] == "override-run-id"
    assert "\nRun ID:    override-run-id\n" in run.result.stdout
    assert (tmp_path / "by-run-id" / "override-run-id.out").is_symlink()
    # The image the banner names is the image every container of the launch runs.
    assert f"container ({alternative_sif})" in run.result.stdout
    assert run.sifs and set(run.sifs) == {str(alternative_sif)}
    assert set(plain.sifs) == {default_sif}


@pytest.mark.parametrize("launcher_args", LAUNCH_PATHS)
def test_the_hook_adds_nothing_but_its_overrides(launch, launcher_args):
    plain = launch(None, *launcher_args)
    assert plain.result.returncode == 0, plain.result.stderr
    assert "Env overrides" not in plain.result.stdout
    # The defaults the hook exists to beat, as the ranks see them without it.
    assert plain.rank_env["TORCH_NCCL_BLOCKING_WAIT"] == "1"
    assert plain.rank_env["OMP_NUM_THREADS"] == "8"
    assert "ISAMBARD_ENV_OVERRIDE_KEYS" not in plain.rank_env

    hooked = launch("ONLY_THIS=1\n", *launcher_args)
    assert hooked.result.returncode == 0, hooked.result.stderr
    added = {"ONLY_THIS", "ISAMBARD_ENV_OVERRIDES", "ISAMBARD_ENV_OVERRIDE_KEYS"}
    assert {k: v for k, v in hooked.rank_env.items() if k not in added | _HARNESS_KEYS} == {
        k: v for k, v in plain.rank_env.items() if k not in _HARNESS_KEYS
    }
    assert (hooked.rank_env["ONLY_THIS"], hooked.rank_env["ISAMBARD_ENV_OVERRIDE_KEYS"]) == ("1", "ONLY_THIS")
    assert hooked.rank_argv == plain.rank_argv


def test_an_inherited_keys_variable_is_dropped_without_the_hook(launch):
    run = launch(None, extra_env={"ISAMBARD_ENV_OVERRIDE_KEYS": "TORCH_NCCL_BLOCKING_WAIT"})
    assert run.result.returncode == 0, run.result.stderr
    assert "ISAMBARD_ENV_OVERRIDE_KEYS" not in run.rank_env


@pytest.mark.parametrize("launcher_args", LAUNCH_PATHS)
def test_hybridep_compiles_into_the_jobs_node_local_temp_dir(launch, shell, launcher_args):
    """Activate derives the cache from the TMPDIR the launcher exports before it; with the variable unset,
    deep_ep writes a directory per rank into the bind-mounted host home."""
    run = launch(None, *launcher_args)
    assert run.result.returncode == 0, run.result.stderr
    assert run.rank_env["HYBRID_EP_CACHE_DIR"] == f"{_job_tmpdirs(shell)[0]}/hybrid_ep_jit"


@pytest.mark.parametrize(
    "launcher_args, key, flag",
    [
        ((), "ISAMBARD_FT_HEARTBEAT_TIMEOUT", "--ft-rank-heartbeat-timeout="),
        ((), "MASTER_ADDR_OVERRIDE", "--rdzv_endpoint="),
        (("--disable-ft",), "MASTER_PORT_OVERRIDE", "--master_port="),
        (("--disable-ft",), "MASTER_ADDR_OVERRIDE", "--master_addr="),
    ],
)
def test_values_pasted_into_the_payload_arrive_as_one_literal_word(launch, tmp_path, launcher_args, key, flag):
    marker = tmp_path / "injected"
    value = f"7200 ; touch {marker}"
    run = launch(f"{key}={value}\n", *launcher_args)
    assert run.result.returncode == 0, run.result.stderr
    assert not marker.exists(), "an override value ran as shell code in the payload"
    [argument] = [arg for arg in run.rank_argv if arg.startswith(flag)]
    assert value in argument


@pytest.mark.parametrize(
    "overrides, extra_env, message",
    [
        pytest.param("TORCH_NCCL_BLOCKING_WAIT=0\nnot an assignment\n", None, ":2: not KEY=VALUE", id="malformed"),
        pytest.param("MODEL=super\n", None, "MODEL is set by the launcher itself", id="launcher-variable"),
        pytest.param(None, {"ISAMBARD_ENV_OVERRIDES": ""}, "names no file: ''", id="set-but-empty"),
    ],
)
def test_a_refused_file_stops_the_launch_before_anything_happens(
    launch, shell, tmp_path, overrides, extra_env, message
):
    run = launch(overrides, extra_env=extra_env)
    assert run.result.returncode != 0
    assert "FATAL" in run.result.stderr and message in run.result.stderr, run.result.stderr
    assert not run.srun_called and run.rank_env is None
    # Nothing the launcher creates exists yet.
    assert run.result.stdout == ""
    assert not (tmp_path / "by-run-id").exists() and not (tmp_path / "nccl_trace").exists()
    assert not any(os.path.exists(path) for path in _job_tmpdirs(shell))


# --- every shell variable the launcher keeps is refused as a KEY ---------------------------------

_DECLARE_LINE = re.compile(r"^declare -(\S+) ([A-Za-z_][A-Za-z0-9_]*)(?:=|$)")


def _declared(path) -> dict[str, str]:
    """Variable name -> attribute letters, from a `declare -p` listing."""
    return {m.group(2): m.group(1) for m in map(_DECLARE_LINE.match, path.read_text().splitlines()) if m}


def test_every_shell_variable_the_launcher_keeps_is_refused_as_a_key(tmp_path, shell):
    """A new launcher variable that the hook would let an override rewrite fails this test.

    The launcher is sourced so that its shell survives it; its stdout goes to a regular file so
    that it takes the raw-log branch and assigns the variables that go with it.
    """
    config = tmp_path / "config.yaml"
    config.write_text("train: {}\n")
    before, after = tmp_path / "before", tmp_path / "after"
    harness = tmp_path / "source_launcher.sh"
    harness.write_text(
        f'declare -p > "{before}"\n'
        f'source "{LAUNCHER}" "{config}" --model nano --mode pretrain\n'
        f'declare -p > "{after}"\n'
    )
    with open(tmp_path / "launcher.out", "w") as stdout:
        sourced = subprocess.run(
            ["bash", str(harness)],
            stdout=stdout,
            stderr=subprocess.PIPE,
            text=True,
            env=shell.env(LAUNCH_ENV),
            timeout=300,
        )
    assert sourced.returncode == 0, sourced.stderr
    assert any(call["cmd"] in RANK_LAUNCHERS for call in map(json.loads, shell.calls_file.read_text().splitlines())), (
        "the sourced launcher did not reach its rank launcher"
    )
    baseline = _declared(before)
    kept = sorted(name for name, attrs in _declared(after).items() if name not in baseline and "x" not in attrs)
    # Proof that the listing saw the launcher's variables, from each part of it.
    assert {"MODEL", "USAGE", "REPO_DIR", "CONTAINER_SIF", "_FD1_TARGET", "RUN_ID_LINK_DIR", "SCRIPT_ARGS"} <= set(
        kept
    ), kept

    for name in kept:
        (tmp_path / f"{name}.env").write_text(f"{name}=x\n")
    refusal = tmp_path / "refusal.sh"
    refusal.write_text(
        "set -euo pipefail\n"
        f"{launcher_function('apply_env_overrides')}\n"
        f"for name in {' '.join(kept)}; do\n"
        f'    ( apply_env_overrides "{tmp_path}/$name.env" ) 2>/dev/null && echo "ACCEPTED $name"\n'
        "done\n"
        "echo DONE\n"
    )
    result = subprocess.run(["bash", str(refusal)], capture_output=True, text=True, env=_BARE_ENV, timeout=120)
    assert result.returncode == 0, result.stderr
    assert result.stdout == "DONE\n", f"launcher variables an override could rewrite:\n{result.stdout}"
