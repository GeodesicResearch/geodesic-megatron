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
"""pipeline_training_launch.sh, the exec shim's scrub and pipeline_env_submit.sbatch, recorded as goldens.

Every case runs the real scripts under PipelineShell inside the test container, so the config, the exec shim and
pipeline_env_activate.sh run for real; SLURM, the container runtime, the rank launchers and the clock are its stubs.
A launch golden holds the exit status, the output, srun's options and command, the image of every container, and the
rank launch's arguments and environment. The environment is recorded against the one the launch started with (the
shell's, plus the variables the launch added): every variable the ranks see with another value or that is new, every
variable missing at the ranks, and nothing for a variable that arrives unchanged.
"""

import json
import shutil

import pytest
from scripts.training.launcher_source import LAUNCHER

from tests.unit_tests.goldens import FIXTURES, assert_golden_json
from tests.unit_tests.launcher_harness import LITERAL_VALUES, Launcher, launcher_shell
from tests.unit_tests.stubbed_shell import PipelineShell


GOLDENS = FIXTURES / "launcher"


@pytest.fixture
def shell(tmp_path):
    shell = launcher_shell(tmp_path)
    yield shell
    shell.remove_job_tmp_dirs()


@pytest.fixture
def launch(tmp_path, shell):
    return Launcher(tmp_path, shell)


def _snapshot(shell, launch):
    """The launch's golden record: ``launch.record`` with the rank environment recorded against the starting one.

    ``SHLVL`` and ``_`` are bash's own bookkeeping of how the rank launcher was reached, so they are left out.
    """
    record = launch.record
    given = shell.env(launch.env)
    rank = None
    if launch.rank_env is not None:
        set_ = {k: v for k, v in launch.rank_env.items() if given.get(k) != v and k not in ("SHLVL", "_")}
        rank = {"argv": launch.rank_argv, "env": set_, "env_removed": sorted(set(given) - set(launch.rank_env))}
    return shell.normalise(
        {
            "returncode": record["returncode"],
            "stdout": record["stdout"],
            "stderr": record["stderr"],
            "steps": [
                [call["cmd"], *call["argv"]] for call in record["calls"] if call["cmd"] in ("srun", "apptainer")
            ],
            "rank": rank,
        }
    )


LAUNCH_CASES = {
    "default": ((), {}, None),
    "disable_ft": (("--disable-ft",), {}, None),
    "disable_straggler": (("--disable-straggler",), {}, None),
    "enable_pao": (("--enable-pao",), {}, None),
    "peft_lora_sft": (("--peft", "lora", "--mode", "sft"), {}, None),
    "peft_canonical_lora_sft_torchrun": (("--peft", "canonical_lora", "--mode", "sft", "--disable-ft"), {}, None),
    # An empty override is unset to the launcher, which then takes the first host of the node list.
    "nodes_nodelist": (
        ("--nodes", "2", "--nodelist", "nid[000002-000003]"),
        {"SLURM_NNODES": "4", "SLURM_NODELIST": "nid[000001-000004]", "MASTER_ADDR_OVERRIDE": ""},
        None,
    ),
    "hydra_overrides": (("train.train_iters=20", "logger.wandb_exp_name=a b", "checkpoint.save=null"), {}, None),
    "hydra_overrides_torchrun": (
        ("--disable-ft", "--", "train.train_iters=20", "logger.wandb_exp_name=a b"),
        {},
        None,
    ),
    "env_overrides_literal": ((), {}, "".join(f"{k}={v}\n" for k, v in LITERAL_VALUES.items())),
    "env_overrides_literal_torchrun": (
        ("--disable-ft",),
        {},
        "".join(f"{k}={v}\n" for k, v in LITERAL_VALUES.items()),
    ),
    "persistent_triton_cache": ((), {"TRAIN_PERSISTENT_TRITON_CACHE": "1"}, None),
    "nccl_debug_info": ((), {"ISAMBARD_NCCL_DEBUG": "INFO", "ISAMBARD_NCCL_DEBUG_FILE": "/tmp/nccl/%h.%p.log"}, None),
    "nccl_net_plugin": ((), {"TRAIN_NCCL_NET_PLUGIN": "/opt/alt/libnccl-net.so"}, None),
    "rx_match_and_rdzv_overrides": (
        ("--disable-ft",),
        {"TRAIN_FI_CXI_RX_MATCH_MODE_OVERRIDE": "hybrid", "TRAIN_FI_CXI_RDZV_THRESHOLD_OVERRIDE": "16384"},
        None,
    ),
    # The ft_launcher gate, against an image whose ft_launcher lacks one or both section-timeout flags.
    "ft_gate_first_flag_only": ((), {"STUB_FT_LAUNCHER_HELP": "  --ft-rank-section-timeouts ..."}, None),
    "ft_gate_neither_flag": ((), {"STUB_FT_LAUNCHER_HELP": ""}, None),
    "unknown_flag": (("--bogus",), {}, None),
    "bad_mode": (("--mode", "rl"), {}, None),
}


@pytest.mark.parametrize("case", sorted(LAUNCH_CASES))
def test_launch(shell, launch, case):
    args, env, overrides = LAUNCH_CASES[case]
    run = launch(overrides, *args, extra_env=env)
    assert_golden_json(GOLDENS / f"{case}.json", _snapshot(shell, run))


def test_an_ft_launcher_whose_help_fails_stops_the_launch_even_when_it_lists_the_flags(shell, launch):
    """A broken ft_launcher (its --help exits nonzero, here after printing both flags) fails the gate: the launch
    stops with the gate's message before any rank starts."""
    run = launch(None, extra_env={"STUB_FT_LAUNCHER_HELP_RC": "3"})
    assert run.result.returncode == 1
    assert "FATAL: the container image's ft_launcher (nvidia-resiliency-ext) does not" in run.result.stderr
    assert not run.srun_called and run.rank_env is None


@pytest.mark.parametrize("launcher_args", [(), ("--disable-ft",)], ids=["ft_launcher", "torchrun"])
def test_a_forwarded_argument_with_a_space_or_a_shell_character_reaches_the_ranks_as_one_word(
    launch, tmp_path, launcher_args
):
    """Hydra overrides and the config path cross the payload's shell re-parse unchanged, one argument each."""
    marker = tmp_path / "ran"
    odd_config = tmp_path / "my config.yaml"
    odd_config.write_text("train: {}\n")
    overrides = ["logger.wandb_exp_name=a b", f"logger.note=$(touch {marker})", "train.train_iters=20"]
    run = launch(None, *launcher_args, *overrides, extra_env={}, config=odd_config)
    assert run.result.returncode == 0, run.result.stderr
    assert not marker.exists(), "a forwarded argument ran as shell code in the payload"
    argv = run.rank_argv
    assert argv[argv.index("--config-file") + 1] == str(odd_config)
    assert argv[-3:] == overrides


def test_a_copy_of_the_launcher_outside_the_checkout_launches_as_the_launcher_does(shell, launch, tmp_path):
    """A single-file copy of the launcher (a frozen copy of one under edit, or a spooled one) takes its libraries
    from the checkout GEODESIC_REPO_DIR names, and its launch matches the checkout launcher's golden."""
    copy = tmp_path / "frozen" / LAUNCHER.name
    copy.parent.mkdir()
    shutil.copy(LAUNCHER, copy)
    assert_golden_json(GOLDENS / "default.json", _snapshot(shell, launch(None, script=copy)))


def test_a_launcher_whose_repo_dir_is_no_checkout_stops_before_it_reads_anything(shell, launch, tmp_path):
    """REPO_DIR is resolved and checked before the overrides file is read: a REPO_DIR that is no checkout stops the
    launch at the REPO_DIR guard, which names both causes."""
    (tmp_path / "not_a_checkout").mkdir()
    run = launch("BAD LINE\n", extra_env={"GEODESIC_REPO_DIR": str(tmp_path / "not_a_checkout")})
    assert run.result.returncode == 1
    assert run.result.stderr.startswith(f"FATAL: {tmp_path}/not_a_checkout has no pipeline/sh/common.sh")
    assert "SLURM_SUBMIT_DIR" in run.result.stderr and "copied script" in run.result.stderr
    assert "BAD LINE" not in run.result.stderr
    assert not run.srun_called and run.result.stdout == ""


# --- the exec shim's scrub -------------------------------------------------------------------------------------------

# Toolchain and virtualenv variables an interactive Isambard shell carries. LD_PRELOAD names a library every process
# can preload harmlessly, since it applies to the shim's own shell before the scrub.
HOST_TOOLCHAIN = {
    "LD_PRELOAD": "/usr/lib/aarch64-linux-gnu/libc.so.6",
    "PYTHONPATH": "/home/user/site",
    "VIRTUAL_ENV": "/home/user/venv",
    "NCCL_LIBRARY": "/opt/nccl/lib/libnccl.so",
    "CC": "/usr/bin/gcc-12",
    "CXX": "/usr/bin/g++-12",
    "CUDAHOSTCXX": "/usr/bin/g++-12",
    "CUDA_HOME": "/opt/nvidia/hpc_sdk/cuda",
    "CPLUS_INCLUDE_PATH": "/home/user/include",
    "C_INCLUDE_PATH": "/home/user/include",
    "CUDNN_PATH": "/opt/cudnn",
    "LD_LIBRARY_PATH": "/opt/cray/lib64",
    "PYTHONNOUSERSITE": "0",
}


def test_exec_shim_scrub(tmp_path):
    """The environment a payload starts with, from a shell holding every toolchain variable the shim scrubs."""
    shell = PipelineShell(tmp_path, record=())
    dump = tmp_path / "payload_env.json"
    payload = f"{shell.env()['STUB_PYTHON']} -c 'import json, os; print(json.dumps(dict(os.environ)))' > {dump}"
    run = shell.run("pipeline_env_exec.sh", payload, env=HOST_TOOLCHAIN)
    assert run["returncode"] == 0, run["stderr"]
    payload_env = json.loads(dump.read_text())
    given = shell.env(HOST_TOOLCHAIN)
    snapshot = {
        "removed": sorted(k for k in given if k not in payload_env),
        "changed": {k: v for k, v in payload_env.items() if k in given and given[k] != v},
        "added": sorted(k for k in payload_env if k not in given),
    }
    assert_golden_json(FIXTURES / "environment" / "exec_shim_scrub.json", shell.normalise(snapshot))


# --- pipeline_env_submit.sbatch ------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "case, args, env",
    [
        (
            "submit_smoke_two_nodes",
            ["smoke"],
            {"SLURM_JOB_NUM_NODES": "2", "SLURM_JOB_NODELIST": "nid[000001-000002]"},
        ),
        (
            "submit_smoke_port_override",
            ["smoke"],
            {"SLURM_JOB_NUM_NODES": "2", "SLURM_JOB_NODELIST": "nid9,nid12", "MASTER_PORT_OVERRIDE": "31000"},
        ),
        (
            "submit_smoke_addr_override",
            ["smoke"],
            {
                "SLURM_JOB_NUM_NODES": "2",
                "SLURM_JOB_NODELIST": "nid[000001-000002]",
                "MASTER_ADDR_OVERRIDE": "10.0.0.9",
            },
        ),
        ("submit_smoke_one_node", ["smoke"], {"SLURM_JOB_NUM_NODES": "1", "SLURM_JOB_NODELIST": "nid000001"}),
        ("submit_validate", ["validate", "--run-training"], {}),
        ("submit_unknown_mode", ["frobnicate"], {}),
    ],
)
def test_env_submit(pipeline_launch_shell, case, args, env):
    run = pipeline_launch_shell.run("pipeline_env_submit.sbatch", *args, env=env)
    assert_golden_json(FIXTURES / "environment" / f"{case}.json", pipeline_launch_shell.normalise(run))


def test_smoke_rendezvous_honours_the_address_override_as_every_launcher_does(pipeline_launch_shell):
    env = {"SLURM_JOB_NUM_NODES": "2", "SLURM_JOB_NODELIST": "nid[000001-000002]", "MASTER_ADDR_OVERRIDE": "10.0.0.9"}
    run = pipeline_launch_shell.run("pipeline_env_submit.sbatch", "smoke", env=env)
    [torchrun] = [call for call in run["calls"] if call["cmd"] == "torchrun"]
    assert "--master_addr=10.0.0.9" in torchrun["argv"] and "--master_port=29401" in torchrun["argv"]
