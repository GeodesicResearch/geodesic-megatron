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
"""pipeline_checkpoint_convert.sh: what each mode launches, recorded as goldens.

Each case runs the real script under ``PipelineShell`` (SLURM, the container, torchrun, the converter's python
launch and the Hub are the stubbed boundaries) and compares its exit status, output and the ordered stub calls
with ``fixtures/convert_sh/<case>.json``. export and upload-all run on fake run trees under ``tmp_path``.
"""

import pytest

from tests.unit_tests.checkpoint_dirs import make_checkpoint_dir
from tests.unit_tests.goldens import FIXTURES, assert_golden_json


SCRIPT = "pipeline_checkpoint_convert.sh"


#: The checkpoint an export case names, replaced by ``_export_run``'s fake run: export resolves the iteration on disk.
RUN = "<run>"
EXPORT = ["export", RUN, "--hf-model", "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16", "--no-reasoning"]

EXPORT_CASES = {
    "export_default": (EXPORT, {}),
    "export_tp1_pp2_ep2": (EXPORT + ["--tp", "1", "--pp", "2", "--ep", "2"], {}),
    "export_two_nodes": (EXPORT + ["--ep", "8"], {"CONVERT_NNODES": "2", "SLURM_NNODES": "4"}),
    "export_nodelist": (
        EXPORT + ["--ep", "8"],
        {"CONVERT_NNODES": "2", "CONVERT_NODELIST": "nid[000002-000003]", "SLURM_NNODES": "4"},
    ),
    "export_master_overrides": (EXPORT, {"MASTER_ADDR_OVERRIDE": "10.0.0.7", "MASTER_PORT_OVERRIDE": "31111"}),
    "export_iteration_push": (EXPORT + ["--iteration", "300", "--push-to-hub"], {}),
    "export_lora_factors": (["export", "/ckpt/lora", "--format", "lora-factors", "--factors-path", "/out/f"], {}),
    "export_forwarded_flags": (EXPORT + ["--keep-remote-code", "--not-strict", "--hf-path", "/out/hf"], {}),
    "export_slurm_flags_stripped": (EXPORT + ["--exclude=nid000009", "--nodes", "2", "--qos", "high"], {}),
    "export_value_with_space": (EXPORT + ["--keep-remote-code", "--remote-code-source", "/models/a b"], {}),
    "export_missing_hf_model": (["export", RUN, "--no-reasoning"], {}),
    "export_bad_parallelism": (EXPORT + ["--tp", "2", "--ep", "4"], {}),
    "import_default": (["import", "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"], {}),
    "import_megatron_path": (
        ["import", "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16", "--megatron-path", "/out/m", "--ep", "4"],
        {},
    ),
    "import_unknown_option": (["import", "nvidia/X", "--bogus"], {}),
    "unknown_mode": (["frobnicate"], {}),
}


def _export_run(tmp_path, args):
    """``args`` with ``RUN`` replaced by a fake run under ``tmp_path``: saves 100 and 300, the tracker at 300, no
    run_config (so no export repair)."""
    run = make_checkpoint_dir(tmp_path / "ckpt" / "run", [100, 300], tracker=300)
    return [str(run) if arg == RUN else arg for arg in args]


@pytest.mark.parametrize("case", sorted(EXPORT_CASES))
def test_export_and_import_launch(pipeline_launch_shell, tmp_path, case):
    args, env = EXPORT_CASES[case]
    run = pipeline_launch_shell.run(SCRIPT, *_export_run(tmp_path, args), env=env)
    assert_golden_json(FIXTURES / "convert_sh" / f"{case}.json", pipeline_launch_shell.normalise(run))


RUN_CONFIG = "train:\n  train_iters: 300\n  micro_batch_size: 1\n"


def _run_tree(tmp_path):
    """A run of train_iters 300 with an imported start (iter 0), saves at 100, 200 (exported) and 300, and the
    tracker still at 200, as while iter 300 is being written."""
    run = make_checkpoint_dir(tmp_path / "my_run", [0, 100, 300], tracker=200, run_config=RUN_CONFIG)
    return make_checkpoint_dir(run, [200], tracker=200, run_config=RUN_CONFIG, hf_files={"config.json": "{}\n"})


UPLOAD = ["--hf-model", "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16", "--no-reasoning"]


@pytest.mark.parametrize(
    "case, extra",
    [
        ("upload_all_batch", []),
        ("upload_all_hf_org", ["--hf-org", "scratch-org"]),
        ("upload_all_missing_reasoning", ["--hf-model", "nvidia/X"]),
    ],
)
def test_upload_all_batch(pipeline_launch_shell, tmp_path, case, extra):
    run_dir = _run_tree(tmp_path)
    args = ["upload-all", str(run_dir), *(extra if case == "upload_all_missing_reasoning" else UPLOAD + extra)]
    run = pipeline_launch_shell.run(SCRIPT, *args)
    assert_golden_json(FIXTURES / "convert_sh" / f"{case}.json", pipeline_launch_shell.normalise(run))


def test_upload_all_poll_until_the_final_iteration(pipeline_launch_shell, tmp_path):
    """Poll mode: the first pass finds iter 100; during the wait iter 300 lands and the tracker moves to it."""
    run_dir = make_checkpoint_dir(tmp_path / "my_run", [100], tracker=100, run_config=RUN_CONFIG)
    land = (
        f"mkdir -p {run_dir}/iter_0000300 && printf 'train:\\n  train_iters: 300\\n' > {run_dir}/iter_0000300/run_config.yaml"
        f" && echo 300 > {run_dir}/latest_checkpointed_iteration.txt"
    )
    run = pipeline_launch_shell.run(SCRIPT, "upload-all", str(run_dir), *UPLOAD, "--poll", env={"STUB_ON_SLEEP": land})
    assert_golden_json(FIXTURES / "convert_sh" / "upload_all_poll.json", pipeline_launch_shell.normalise(run))


def test_a_repo_dir_outside_the_checkout_names_both_causes(pipeline_launch_shell, tmp_path):
    """A copied script, or SLURM_SUBMIT_DIR naming a tunnel's dir, leaves REPO_DIR without the libraries."""
    elsewhere = tmp_path / "tunnel_submit_dir"
    elsewhere.mkdir()
    run = pipeline_launch_shell.run(
        SCRIPT,
        *_export_run(tmp_path, EXPORT),
        env={"GEODESIC_REPO_DIR": str(elsewhere), "SLURM_SUBMIT_DIR": str(elsewhere)},
    )
    assert run["returncode"] == 1 and run["calls"] == []
    assert f"FATAL: {elsewhere} has no pipeline/sh/common.sh, so REPO_DIR is not this repo's checkout" in run["stderr"]
    assert f"SLURM_SUBMIT_DIR ({elsewhere})" in run["stderr"] and "GEODESIC_REPO_DIR" in run["stderr"]


def test_upload_all_converts_only_saves_the_tracker_names_as_complete(pipeline_launch_shell, tmp_path):
    """iter 300 is above the tracker (200), so it is still being written and is neither converted nor pushed; the
    imported start, iter_0000000, is a completed save like any other."""
    run = pipeline_launch_shell.run(SCRIPT, "upload-all", str(_run_tree(tmp_path)), *UPLOAD)
    assert run["returncode"] == 0, run["stderr"]
    converted = [c["argv"][c["argv"].index("--iteration") + 1] for c in run["calls"] if c["cmd"] == "torchrun"]
    assert converted == ["0", "100"]
    revisions = [c["kwargs"]["revision"] for c in run["calls"] if c["argv"] == ["upload_folder"]]
    assert revisions == ["iter_0000200"]


def test_a_forwarded_value_with_a_space_reaches_the_converter_as_one_argument(pipeline_launch_shell, tmp_path):
    run = pipeline_launch_shell.run(SCRIPT, *_export_run(tmp_path, EXPORT_CASES["export_value_with_space"][0]))
    assert run["returncode"] == 0, run["stderr"]
    [torchrun] = [c["argv"] for c in run["calls"] if c["cmd"] == "torchrun"]
    assert torchrun[torchrun.index("--remote-code-source") + 1] == "/models/a b"


def test_a_repo_name_with_a_quote_is_pushed_under_that_name(pipeline_launch_shell, tmp_path):
    """upload-all once pasted the org, repo name and message into Python source, which a quote broke."""
    run_dir = make_checkpoint_dir(
        tmp_path / "it's_run", [300], tracker=300, run_config=RUN_CONFIG, hf_files={"config.json": "{}"}
    )
    run = pipeline_launch_shell.run(SCRIPT, "upload-all", str(run_dir), *UPLOAD)
    assert run["returncode"] == 0, run["stderr"]
    uploads = [c["kwargs"] for c in run["calls"] if c["argv"] == ["upload_folder"]]
    assert [(u["repo_id"], u["revision"]) for u in uploads] == [
        ("geodesic-research/it's_run", "iter_0000300"),
        ("geodesic-research/it's_run", "main"),
    ]


def test_upload_all_poll_waits_through_a_pass_that_finds_no_completed_save(pipeline_launch_shell, tmp_path):
    """Before the first save completes there is no tracker, so a pass finds nothing; poll mode waits for the save
    rather than ending. Here iter 300, the final one, is being written; during the wait it completes."""
    run_dir = tmp_path / "my_run"
    (run_dir / "iter_0000300").mkdir(parents=True)
    (run_dir / "iter_0000300" / "run_config.yaml").write_text(RUN_CONFIG)
    complete = f"echo 300 > {run_dir}/latest_checkpointed_iteration.txt"
    run = pipeline_launch_shell.run(
        SCRIPT, "upload-all", str(run_dir), *UPLOAD, "--poll", env={"STUB_ON_SLEEP": complete}
    )
    assert run["returncode"] == 0, run["stdout"] + run["stderr"]
    assert "No iterations found yet." in run["stdout"]
    assert [c["argv"] for c in run["calls"] if c["cmd"] == "sleep"] == [["300"]]
    converted = [c["argv"][c["argv"].index("--iteration") + 1] for c in run["calls"] if c["cmd"] == "torchrun"]
    assert converted == ["300", "300"], "the pass after the wait, then the final pass once training is complete"


def test_upload_all_batch_with_no_completed_save_ends_with_status_1(pipeline_launch_shell, tmp_path):
    """A batch run has nothing to do before the first save completes, and says so."""
    run_dir = tmp_path / "my_run"
    (run_dir / "iter_0000100").mkdir(parents=True)
    run = pipeline_launch_shell.run(SCRIPT, "upload-all", str(run_dir), *UPLOAD)
    assert run["returncode"] == 1
    assert "No iterations found yet." in run["stdout"]
    assert not [c for c in run["calls"] if c["cmd"] == "torchrun"]
