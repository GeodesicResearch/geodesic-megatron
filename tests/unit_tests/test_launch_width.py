# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""The check that a config trains at exactly the width it names (``scripts/training/launch_width.py``).

The block's parsing, the launcher's record and the run's own check run as they are. The launcher's
``apply_launch_width`` function is run out of ``pipeline_training_launch.sh`` with the commands it reaches beyond this
machine stood in for: the container shim (a unit test already runs inside the container), ``srun`` (SLURM), and
``nvidia-smi`` and ``hostname`` (the GPU nodes a sweep reads), each a Python script, since a stub shell named
``nvidia-smi`` would recurse through the container's shell startup hook.
"""

import json
import os
import subprocess
from dataclasses import asdict
from pathlib import Path

import pytest
import yaml
from scripts.training import launch_width
from scripts.training.launcher_source import launcher_function

from megatron.bridge.recipes.nemotronh.nemotron_3_nano import nemotron_3_nano_pretrain_config


REPO_ROOT = Path(__file__).resolve().parents[2]
HEALTHY_NODE = REPO_ROOT / "tests" / "unit_tests" / "fixtures" / "nvlink_health" / "gh200_healthy.txt"
BLOCK = {"nodes": 2, "gpus_per_node": 4, "data_parallel_size": 8}
RUN_ID = "20261010T180000-j7"


def config(directory: Path, block: dict | None, name: str = "stage.yaml") -> Path:
    content = {"train": {"global_batch_size": 16}}
    if block is not None:
        content[launch_width.LAUNCH_WIDTH_KEY] = block
    path = directory / name
    path.write_text(yaml.safe_dump(content))
    return path


class TestParse:
    def test_a_block_parses_with_its_world_size(self):
        width = launch_width.parse_launch_width({**BLOCK, "nvlink_links_per_gpu": 18}, "launch_width")
        assert (width.world_size, width.nvlink_links_per_gpu) == (8, 18)

    def test_the_sweep_is_optional(self):
        assert launch_width.parse_launch_width(BLOCK, "launch_width").nvlink_links_per_gpu is None

    @pytest.mark.parametrize(
        ("changes", "message"),
        [
            ({"nodes": 0}, "nodes must be a positive integer"),
            ({"gpus_per_node": True}, "gpus_per_node must be a positive integer"),
            ({"data_parallel_size": "8"}, "data_parallel_size must be a positive integer"),
            ({"nvlink_links_per_gpu": -1}, "nvlink_links_per_gpu must be a positive integer"),
            ({"data_parallel_size": 3}, "does not divide the world size 8"),
            ({"world_size": 8}, r"unknown keys \['world_size'\]"),
        ],
    )
    def test_a_malformed_block_is_refused(self, changes, message):
        with pytest.raises(launch_width.LaunchWidthError, match=message):
            launch_width.parse_launch_width({**BLOCK, **changes}, "launch_width")

    def test_every_count_is_required(self):
        block = {key: value for key, value in BLOCK.items() if key != "nodes"}
        with pytest.raises(launch_width.LaunchWidthError, match=r"missing keys \['nodes'\]"):
            launch_width.parse_launch_width(block, "launch_width")

    def test_a_block_in_a_base_config_is_the_overlays_too(self, tmp_path):
        base = config(tmp_path, BLOCK, "base.yaml")
        overlay = tmp_path / "overlay.yaml"
        overlay.write_text(f"base_config: {base.name}\ntrain:\n  global_batch_size: 32\n")
        assert launch_width.config_launch_width(str(overlay)).nodes == 2

    def test_a_config_without_a_block_fixes_no_width(self, tmp_path):
        assert launch_width.config_launch_width(str(config(tmp_path, None))) is None


class TestLaunchRecord:
    width = launch_width.parse_launch_width(BLOCK, "launch_width")

    def test_a_launch_at_the_width_is_recorded(self):
        record = launch_width.launch_record(self.width, "stage.yaml", 2, "nid[1-2]", 4)
        assert record == {
            "config": "stage.yaml",
            "expected": asdict(self.width),
            "nodes": 2,
            "nodelist": "nid[1-2]",
            "gpus_per_node": 4,
        }

    @pytest.mark.parametrize(
        ("nodes", "gpus", "message"),
        [
            (3, 4, "the launch has 3 nodes, the config trains on exactly 2"),
            (2, 2, "the launch's nodes have 2 GPUs each, the config trains on 4 per node"),
        ],
    )
    def test_a_launch_at_another_width_is_refused(self, nodes, gpus, message):
        with pytest.raises(launch_width.LaunchWidthError, match=message):
            launch_width.launch_record(self.width, "stage.yaml", nodes, "nid[1-3]", gpus)


def nano_config(context_parallel_size: int):
    """The Nano pretraining recipe's config at TP = PP = 1 and the given CP: the run's own data-parallel arithmetic."""
    cfg = nemotron_3_nano_pretrain_config()
    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.context_parallel_size = context_parallel_size
    return cfg


class TestRequireLaunchedWidth:
    width = launch_width.parse_launch_width(BLOCK, "launch_width")

    def environ(self, world_size: int, expected: dict | None = None) -> dict[str, str]:
        record = {"expected": expected or asdict(self.width), "nodes": 2, "nodelist": "nid[1-2]", "gpus_per_node": 4}
        return {launch_width.RECORD_ENV: json.dumps(record), "WORLD_SIZE": str(world_size)}

    def test_a_run_at_the_width_returns_its_sizes(self):
        dp = nano_config(1).get_data_parallel_size
        record = launch_width.require_launched_width(self.width, "stage.yaml", self.environ(8), dp)
        assert (record["world_size"], record["data_parallel_size"], record["nodelist"]) == (8, 8, "nid[1-2]")

    @pytest.mark.parametrize(
        ("environ", "context_parallel_size", "message"),
        [
            ({"WORLD_SIZE": "8"}, 1, "launch it through pipeline_training_launch.sh"),
            ("other-block", 1, "records the check of another launch_width"),
            ("world-4", 1, "the run has 4 ranks, its launch_width trains with exactly 8"),
            ("world-8", 2, "the run's data-parallel size at 8 ranks is 4, its launch_width states 8"),
        ],
    )
    def test_a_run_at_another_width_is_refused(self, environ, context_parallel_size, message):
        if environ == "other-block":
            environ = self.environ(8, expected={**asdict(self.width), "nodes": 4})
        elif isinstance(environ, str):
            environ = self.environ(int(environ.split("-")[1]))
        dp = nano_config(context_parallel_size).get_data_parallel_size
        with pytest.raises(launch_width.LaunchWidthError, match=message):
            launch_width.require_launched_width(self.width, "stage.yaml", environ, dp)


class TestCommandLine:
    def test_plan_prints_the_width_and_the_sweep(self, tmp_path, capsys):
        assert (
            launch_width.main(["plan", "--config", str(config(tmp_path, {**BLOCK, "nvlink_links_per_gpu": 18}))]) == 0
        )
        assert capsys.readouterr().out == "2 4 18\n"

    def test_plan_marks_a_block_without_a_sweep(self, tmp_path, capsys):
        assert launch_width.main(["plan", "--config", str(config(tmp_path, BLOCK))]) == 0
        assert capsys.readouterr().out == f"2 4 {launch_width.NO_SWEEP}\n"

    def test_a_config_without_a_block_prints_nothing(self, tmp_path, capsys):
        assert launch_width.main(["plan", "--config", str(config(tmp_path, None))]) == 0
        captured = capsys.readouterr()
        assert captured.out == "" and "carries no launch_width: block" in captured.err

    def test_record_prints_one_json_line(self, tmp_path, capsys):
        args = ["record", "--config", str(config(tmp_path, BLOCK)), "--nodes", "2", "--nodelist", "a,b"]
        assert launch_width.main([*args, "--gpus-per-node", "4"]) == 0
        assert json.loads(capsys.readouterr().out)["nodelist"] == "a,b"

    def test_record_refuses_another_width(self, tmp_path, capsys):
        args = ["record", "--config", str(config(tmp_path, BLOCK)), "--nodes", "3", "--nodelist", "a,b,c"]
        assert launch_width.main([*args, "--gpus-per-node", "4"]) == 1
        captured = capsys.readouterr()
        assert captured.out == "" and "FATAL [launch-width]: " in captured.err


SRUN = """#!/usr/bin/env python3
# Stands in for srun: runs the step's command once per allocated node, each seeing its own hostname.
import os, subprocess, sys
args = sys.argv[1:]
while args and args[0].startswith("--"):
    args.pop(0)
for host in os.environ["FAKE_HOSTS"].split(","):
    subprocess.run(args, env={**os.environ, "FAKE_HOSTNAME": host})
"""

HOSTNAME = """#!/usr/bin/env python3
import os
print(os.environ["FAKE_HOSTNAME"])
"""

NVIDIA_SMI = """#!/usr/bin/env python3
# Stands in for a node's GPUs: a host named in FAKE_BROKEN fails as a node with an unreadable GPU does.
import os, sys
if os.environ["FAKE_HOSTNAME"] in os.environ.get("FAKE_BROKEN", "").split(","):
    sys.exit(1)
sys.stdout.write(open(os.environ["FAKE_STATUS"]).read())
"""


class TestLauncher:
    """``apply_launch_width`` as the launcher runs it, with the container, SLURM and the GPUs stood in for."""

    def launch(self, tmp_path, block, hosts, broken="", override_nodes="", inherited=None, raw_log=True):
        """Run the launcher's function once; a second call with the same ``tmp_path`` is a second launch of the same
        run (the same run ID) in the same job."""
        repo_dir, bindir, logs = tmp_path / "copy", tmp_path / "bin", tmp_path / "logs"
        if not repo_dir.exists():
            repo_dir.mkdir()
            (repo_dir / "scripts").symlink_to(REPO_ROOT / "scripts")
            # The container cannot be started from a unit test, which already runs inside it, so the shim is a script
            # that runs its command here, with this repository's scripts importable from any directory.
            bindir.mkdir()
            for name, source in [
                ("pipeline_env_exec.sh", '#!/bin/bash\nexec bash -c "$1"\n'),
                ("srun", SRUN),
                ("hostname", HOSTNAME),
                ("nvidia-smi", NVIDIA_SMI),
            ]:
                path = (repo_dir if name == "pipeline_env_exec.sh" else bindir) / name
                path.write_text(source)
                path.chmod(0o755)
            logs.mkdir()
            (logs / "train-7.out").write_text("")
        script = (
            "set -euo pipefail\n"
            f'OVERRIDE_NODES="{override_nodes}"\nOVERRIDE_NODELIST=""\n'
            f"{launcher_function('apply_launch_width')}\n"
            'apply_launch_width "$1" "$2" || { echo "REFUSED"; exit 1; }\n'
            'echo "NODES=${OVERRIDE_NODES:-unset} NODELIST=${OVERRIDE_NODELIST:-unset}"\n'
            'echo "RECORD=${ISAMBARD_LAUNCH_WIDTH-unset}"\n'
        )
        env = {
            "PATH": f"{bindir}{os.pathsep}{os.environ['PATH']}",
            "PYTHONPATH": str(REPO_ROOT),
            "HOME": os.environ["HOME"],
            "SLURM_JOB_ID": "7",
            "ISAMBARD_RUN_ID": RUN_ID,
            "SLURM_NNODES": str(len(hosts)),
            "SLURM_NODELIST": ",".join(hosts),
            "SLURM_GPUS_PER_NODE": "4",
            "FAKE_HOSTS": ",".join(hosts),
            "FAKE_BROKEN": broken,
            "FAKE_STATUS": str(HEALTHY_NODE),
        }
        if raw_log:
            env["ISAMBARD_RAW_LOG_PATH"] = str(logs / "train-7.out")
        if inherited is not None:
            env[launch_width.RECORD_ENV] = inherited
        result = subprocess.run(
            ["bash", "-c", script, "harness", str(config(tmp_path, block)), str(repo_dir)],
            capture_output=True,
            text=True,
            env=env,
            timeout=120,
        )
        return result, logs / "nvlink" / RUN_ID

    def test_a_swept_allocation_trains_on_its_first_healthy_nodes(self, tmp_path):
        block = {**BLOCK, "nvlink_links_per_gpu": 18}
        result, sweep = self.launch(tmp_path, block, ["n1", "n2", "n3"], broken="n1")
        assert result.returncode == 0, result.stderr
        assert "NODES=2 NODELIST=n2,n3" in result.stdout
        record = json.loads(result.stdout.split("RECORD=", 1)[1])
        assert (record["nodes"], record["nodelist"], record["gpus_per_node"]) == (2, "n2,n3", 4)
        assert sorted(path.name for path in sweep.iterdir()) == ["nodelist.txt", "report.json", "status"]
        assert sorted(path.name for path in (sweep / "status").iterdir()) == ["n1.txt", "n2.txt", "n3.txt"]

    def test_a_second_launch_of_the_run_cannot_judge_the_first_launchs_records(self, tmp_path):
        """A later launch whose sweep records nothing (every node's nvidia-smi failing) must not select nodes from
        the healthy records an earlier launch left: it is refused before it sweeps."""
        block = {**BLOCK, "nvlink_links_per_gpu": 18}
        first, sweep = self.launch(tmp_path, block, ["n1", "n2"])
        assert first.returncode == 0, first.stderr
        second, _ = self.launch(tmp_path, block, ["n1", "n2"], broken="n1,n2")
        assert second.returncode == 1 and "REFUSED" in second.stdout
        assert f"{sweep} exists" in second.stderr

    def test_too_few_healthy_nodes_end_the_launch(self, tmp_path):
        block = {**BLOCK, "nvlink_links_per_gpu": 18}
        result, _ = self.launch(tmp_path, block, ["n1", "n2", "n3"], broken="n1,n2")
        assert result.returncode == 1 and "REFUSED" in result.stdout

    def test_without_a_sweep_the_allocation_must_be_the_width(self, tmp_path):
        result, _ = self.launch(tmp_path, BLOCK, ["n1", "n2", "n3"])
        assert result.returncode == 1 and "REFUSED" in result.stdout
        assert "the launch has 3 nodes, the config trains on exactly 2" in result.stderr

    def test_an_allocation_of_the_width_launches_without_a_sweep(self, tmp_path):
        result, sweep = self.launch(tmp_path, BLOCK, ["n1", "n2"])
        assert result.returncode == 0, result.stderr
        assert "NODES=unset NODELIST=unset" in result.stdout and not sweep.exists()
        assert json.loads(result.stdout.split("RECORD=", 1)[1])["nodelist"] == "n1,n2"

    def test_a_node_count_given_to_the_launcher_is_refused(self, tmp_path):
        result, _ = self.launch(tmp_path, BLOCK, ["n1", "n2"], override_nodes="2")
        assert result.returncode == 1 and "--nodes and --nodelist are refused" in result.stderr

    def test_a_sweep_without_a_raw_log_is_refused(self, tmp_path):
        block = {**BLOCK, "nvlink_links_per_gpu": 18}
        result, _ = self.launch(tmp_path, block, ["n1", "n2"], raw_log=False)
        assert result.returncode == 1 and "no raw log path is known" in result.stderr

    def test_a_config_fixing_no_width_drops_an_inherited_record(self, tmp_path):
        result, _ = self.launch(tmp_path, None, ["n1", "n2"], inherited='{"nodes": 2}')
        assert result.returncode == 0, result.stderr
        assert "RECORD=unset" in result.stdout
