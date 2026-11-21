# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""configs/ipt_spt/sft_pack_submit.sh submits the conversion and one packing job for each root.

The script, its config and arms.yaml are the real files. Two stand-ins replace what would submit or run a job: ``isambard_sbatch``
(it submits Slurm jobs; the stub records each call, runs the ``--wrap`` string with ``sh`` as Slurm does, and answers like
sbatch) and the frozen copy's ``pipeline_env_exec.sh`` (it enters the container, which a CPU test cannot; the stub records the
one command string it is given).
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import yaml
from scripts.data.arrow_chat_to_jsonl import ROOT_FOLDER


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "configs" / "ipt_spt" / "sft_pack_submit.sh"
CONFIG = Path("configs") / "ipt_spt" / "sft_pack.yaml"
ARMS = Path("configs") / "ipt_spt" / "arms.yaml"

SBATCH_STUB = """#!/bin/bash
echo "$@" >> "$STUB_LOG"
while [ $# -gt 0 ]; do
    if [ "$1" = "--wrap" ]; then sh -c "$2" > /dev/null; fi
    shift
done
n=$(wc -l < "$STUB_LOG")
echo "Submitted batch job $((1000 + n))"
"""
EXEC_STUB = """#!/bin/bash
printf '%s\\n' "$#" "$1" > "$EXEC_LOG"
"""


def run_script(tmp_path: Path) -> tuple[list[str], list[str], str]:
    """Run the script against a frozen copy that holds the real configs; return each recorded submission, the argument count
    and command string that the container shim was given, and what the script printed."""
    frozen = tmp_path / "frozen"
    (frozen / "configs" / "ipt_spt").mkdir(parents=True)
    (frozen / "REVISION").write_text("abc\n")
    for relative in (CONFIG, ARMS):
        shutil.copy(REPO_ROOT / relative, frozen / relative)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for directory, name, text in (
        (bin_dir, "isambard_sbatch", SBATCH_STUB),
        (frozen, "pipeline_env_exec.sh", EXEC_STUB),
    ):
        (directory / name).write_text(text)
        (directory / name).chmod(0o755)
    log, exec_log = tmp_path / "calls.log", tmp_path / "exec.log"
    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "STUB_LOG": str(log),
        "EXEC_LOG": str(exec_log),
        "FROZEN": str(frozen),
    }
    done = subprocess.run(["bash", str(SCRIPT)], env=env, capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    return log.read_text().splitlines(), exec_log.read_text().splitlines(), done.stdout


def test_the_conversion_runs_in_the_container_and_one_packing_job_follows_for_each_root(tmp_path):
    calls, exec_calls, printed = run_script(tmp_path)
    config = yaml.safe_load((REPO_ROOT / CONFIG).read_text())
    tokenizer = yaml.safe_load((REPO_ROOT / ARMS).read_text())["tokenizer_model"]
    convert, *packs = calls
    assert "--job-name=sft-convert" in convert and "--nodes=1" in convert and "--wrap" in convert
    frozen = tmp_path / "frozen"
    assert exec_calls == [
        "1",
        f"cd {frozen}; source pipeline_env_activate.sh || exit 1; python -u scripts/data/arrow_chat_to_jsonl.py --config {CONFIG}",
    ]
    assert len(packs) == config["shards"]
    for index, call in enumerate(packs):
        root = f"{config['output_root']}/{ROOT_FOLDER.format(index=index)}"
        assert "--dependency=afterok:1001" in call and f"--job-name=sft-pack-{index} " in call
        assert call.endswith(f"pipeline_data_submit.sbatch {root} {tokenizer} 4096 1 32")
    assert "conversion job 1001" in printed and f"packing job {1001 + config['shards']}" in printed
