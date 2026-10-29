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
"""Running the real pipeline_training_launch.sh under PipelineShell, in a fake one-node allocation.

Catalogue:
    LAUNCH_ENV: the allocation's variables a launch adds to PipelineShell's fake job.
    RANK_LAUNCHERS: the stubs that start the ranks.
    LITERAL_VALUES: override values that break any accidental shell evaluation.
    launcher_shell: a PipelineShell recording srun, the images and the rank launch, each with its whole environment.
    Launch: one run of the launcher, its result and what the stubs recorded.
    Launcher: runs the launcher on a trivial config, with an optional ISAMBARD_ENV_OVERRIDES file.
"""

import subprocess
from pathlib import Path

from scripts.training.launcher_source import LAUNCHER

from tests.unit_tests.stubbed_shell import PipelineShell


LAUNCH_ENV = {
    "SLURM_GPUS_PER_NODE": "4",
    "SLURM_NODEID": "0",
    # Pre-set so the launcher asks scontrol nothing.
    "MASTER_ADDR_OVERRIDE": "127.0.0.1",
    "ISAMBARD_SWITCH_SPREAD": "test-switch:1",
    "ISAMBARD_RUN_ID": "20260928T000000-jtest",
}

# Values chosen to break any accidental shell evaluation: expansions, command substitution,
# quotes, backslashes, a second '=', surrounding spaces, and an empty value.
LITERAL_VALUES = {
    "TORCH_NCCL_BLOCKING_WAIT": "0",
    "EXPANSION": "$HOME ${USER:-x} $(echo no) `echo no` ~",
    "QUOTES": 'it\'s "quoted"',
    "BACKSLASH": "a\\b\\\\c",
    "EQUALS": "a=b==c",
    "SPACES": "  padded  ",
    "EMPTY": "",
}

# ft_launcher, and `python -m torch.distributed.run` on the --disable-ft path.
RANK_LAUNCHERS = ("ft_launcher", "python")


def launcher_shell(tmp_path: Path) -> PipelineShell:
    """A PipelineShell under ``tmp_path`` that records srun, the image each container runs and the rank launch, each
    record with the whole environment of its command. The caller removes its job tmp dirs at teardown."""
    return PipelineShell(tmp_path, record=("srun", "apptainer", *RANK_LAUNCHERS), record_env="*")


class Launch:
    """One run of the real launcher: its result and what the stubs recorded.

    ``record`` is ``PipelineShell.run``'s return value and ``env`` the variables the launch added to the shell's
    environment; ``rank_env`` and ``rank_argv`` are the last rank launch's
    environment and arguments (None when no rank launcher started), ``sifs`` the image of every container started.
    """

    def __init__(self, record: dict, env: dict[str, str]):
        """Read ``record``, as ``PipelineShell.run`` returned it for a launch that added ``env``."""
        self.record = record
        self.env = env
        self.result = subprocess.CompletedProcess(
            record["args"], record["returncode"], record["stdout"], record["stderr"]
        )
        self.srun_called = any(call["cmd"] == "srun" for call in record["calls"])
        ranks = [call for call in record["calls"] if call["cmd"] in RANK_LAUNCHERS]
        self.rank_env = ranks[-1]["env"] if ranks else None
        self.rank_argv = ranks[-1]["argv"] if ranks else None
        self.sifs = [call["argv"][0] for call in record["calls"] if call["cmd"] == "apptainer"]


class Launcher:
    """Runs the real launcher with ``config.yaml`` (a trivial config) and ``train.out`` (an empty raw log) under
    ``tmp_path``, as ``<config> --model nano --mode pretrain <launcher args>``; later flags win.

    Each call gets its own ``run<N>`` directory under ``tmp_path``, which holds the overrides file when there is one.
    """

    def __init__(self, tmp_path: Path, shell: PipelineShell):
        """Write the config and the raw log under ``tmp_path``; launches run in ``shell``."""
        self.tmp_path = tmp_path
        self.shell = shell
        self.config = tmp_path / "config.yaml"
        self.config.write_text("train: {}\n")
        self.raw_log = tmp_path / "train.out"
        self.raw_log.write_text("")
        self.runs = 0

    def __call__(
        self,
        overrides: str | None,
        *launcher_args: str,
        extra_env: dict[str, str] | None = None,
        script: Path = LAUNCHER,
    ) -> Launch:
        """Launch ``script`` (the checkout's launcher, or a copy of it) once; ``overrides`` is the content of an
        ISAMBARD_ENV_OVERRIDES file (None: the hook is unset)."""
        run_dir = self.tmp_path / f"run{self.runs}"
        run_dir.mkdir()
        self.runs += 1
        env = {**LAUNCH_ENV, "ISAMBARD_RAW_LOG_PATH": str(self.raw_log)}
        if overrides is not None:
            overrides_file = run_dir / "overrides.env"
            overrides_file.write_text(overrides)
            env["ISAMBARD_ENV_OVERRIDES"] = str(overrides_file)
        env.update(extra_env or {})
        args = (str(self.config), "--model", "nano", "--mode", "pretrain", *launcher_args)
        return Launch(self.shell.run(str(script), *args, env=env), env)
