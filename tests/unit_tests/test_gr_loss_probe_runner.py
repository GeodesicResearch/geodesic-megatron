# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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
"""The corpus loss-probe runner must process every matrix row — including a last
row that a hand-edited TSV leaves without a trailing newline, which `read` reports
as EOF while still filling its fields.

The script runs for real as a subprocess: a scratch repo root carries a symlink to
the actual ``run_corpus_loss_probes.sh`` (its REPO_DIR resolution walks up from its
own location, so everything it touches stays inside the scratch tree) plus a stub
``pipeline_training_launch.sh`` that records each probe launch and prints the loss
line the parser expects. The stub exists because the real launcher submits a
multi-node SLURM training run.
"""

import stat
import subprocess
from pathlib import Path

import pytest


REAL_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "gradient_routing" / "run_corpus_loss_probes.sh"

# Invoked by the script as `bash pipeline_training_launch.sh ...` from REPO_DIR, with
# stdout redirected to the probe's log — so the loss line printed here is what the
# script's parser reads back.
STUB_LAUNCHER = """#!/usr/bin/env bash
echo "launched: $*" >> launches.log
echo "validation loss at iteration 1 on validation set | lm loss value: 1.234500E+00 | lm loss PPL: 3.436560E+00"
"""

HEADER = "NAME\tCKPT\tPREFIX\n"


@pytest.fixture
def scratch_repo(tmp_path):
    script_dir = tmp_path / "scripts" / "gradient_routing"
    script_dir.mkdir(parents=True)
    (script_dir / "run_corpus_loss_probes.sh").symlink_to(REAL_SCRIPT)
    launcher = tmp_path / "pipeline_training_launch.sh"
    launcher.write_text(STUB_LAUNCHER)
    launcher.chmod(launcher.stat().st_mode | stat.S_IEXEC)
    (tmp_path / "probe_config.yaml").write_text("{}\n")
    return tmp_path


def run_probe_matrix(scratch_repo: Path, matrix_text: str) -> Path:
    """Run the real script over ``matrix_text`` and return its output directory."""
    matrix = scratch_repo / "matrix.tsv"
    matrix.write_text(matrix_text)
    outdir = scratch_repo / "out"
    proc = subprocess.run(
        [
            "bash",
            str(scratch_repo / "scripts" / "gradient_routing" / "run_corpus_loss_probes.sh"),
            "--matrix",
            str(matrix),
            "--config",
            str(scratch_repo / "probe_config.yaml"),
            "--outdir",
            str(outdir),
            "--nodes",
            "1",
            "--nodelist",
            "stub-node",
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return outdir


def test_a_matrix_without_a_trailing_newline_still_runs_its_last_row(scratch_repo):
    matrix_text = HEADER + "row_a\t/ckpt/a\t/data/a\n" + "row_b\t/ckpt/b\t/data/b"
    outdir = run_probe_matrix(scratch_repo, matrix_text)

    assert (outdir / "row_a.result").exists()
    assert (outdir / "row_b.result").exists()

    # The results rebuild re-reads the matrix, so it must survive the missing
    # newline too — both rows land in the summary with their parsed loss.
    results = (outdir / "results.tsv").read_text()
    assert "row_a\t/ckpt/a\t/data/a\t1.234500E+00" in results
    assert "row_b\t/ckpt/b\t/data/b\t1.234500E+00" in results


def test_a_newline_terminated_matrix_runs_every_row(scratch_repo):
    matrix_text = HEADER + "row_a\t/ckpt/a\t/data/a\n" + "row_b\t/ckpt/b\t/data/b\n"
    outdir = run_probe_matrix(scratch_repo, matrix_text)

    results = (outdir / "results.tsv").read_text()
    assert "row_a\t" in results
    assert "row_b\t" in results
