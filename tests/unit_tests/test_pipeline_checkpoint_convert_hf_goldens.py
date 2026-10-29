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
"""pipeline_checkpoint_convert_hf.py's command line, checkpoint resolution, main and model loading, as goldens.

- The CLI: ``--help`` and each refused flag combination, run as a script.
- ``resolve_checkpoint_path`` over a matrix of checkpoint dirs.
- ``main`` end to end for a matrix of argv, in a child process (huggingface_hub fixes its cache paths at import):
  conversion needs GPUs, so ``convert_multi_gpu`` and ``convert_single_process`` copy an export fixture into the
  output dir, and the Hub client is tests/unit_tests/recording_hub.py. The golden holds the output tree, the stdout
  and the Hub calls.
- ``convert_multi_gpu``'s model build and load through a recording AutoBridge (tests/unit_tests/recorded_bridge.py),
  for a full checkpoint and an adapter-only PEFT one, whose load and merge-export need GPUs and are recorded.
"""

import json
import os
import shutil
import subprocess
import sys

import pytest

from pipeline.checkpoints import resolve_checkpoint_path
from pipeline.export import hf
from tests.unit_tests import recorded_bridge
from tests.unit_tests.checkpoint_dirs import make_checkpoint_dir
from tests.unit_tests.child_process import REPO_ROOT, run_python
from tests.unit_tests.goldens import FIXTURES, assert_golden_json, assert_golden_tree
from tests.unit_tests.hf_export_fixtures import INPUTS, INSTRUCT_TOK, NANO, SNAPSHOT_A, cache_repo, write_weights


GOLDENS = FIXTURES / "hf_export"
SCRIPT = REPO_ROOT / "pipeline_checkpoint_convert_hf.py"


# --- the command line ---------------------------------------------------------------------------------------------

CLI_CASES = {
    "help": ["--help"],
    "hf_needs_hf_model": ["--megatron-path", "/r", "--no-reasoning"],
    "hf_needs_reasoning_flag": ["--megatron-path", "/r", "--hf-model", NANO],
    "both_reasoning_flags": ["--megatron-path", "/r", "--hf-model", NANO, "--reasoning", "--no-reasoning"],
    "factors_flags_with_hf": ["--megatron-path", "/r", "--hf-model", NANO, "--no-reasoning", "--factors-path", "/f"],
    "hf_flags_with_factors": [
        "--megatron-path",
        "/r",
        "--format",
        "lora-factors",
        "--hf-model",
        NANO,
        "--push-to-hub",
    ],
    "missing_megatron_path": ["--hf-model", NANO, "--no-reasoning"],
}


@pytest.mark.parametrize("case", sorted(CLI_CASES))
def test_cli(case):
    result = subprocess.run(
        [sys.executable, str(SCRIPT), *CLI_CASES[case]],
        capture_output=True,
        text=True,
        timeout=120,
        env={**os.environ, "COLUMNS": "120"},
    )
    record = {"returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
    assert_golden_json(GOLDENS / "cli" / f"{case}.json", record)


# --- resolve_checkpoint_path ----------------------------------------------------------------------------------------


RESOLVE_CASES = {
    "explicit_present": ([100, 200], 200, 100),
    "explicit_missing": ([100, 200], 200, 300),
    "tracker": ([100, 200, 300], 200, None),
    "tracker_names_missing_dir": ([100, 200], 300, None),
    "no_tracker_scans": ([100, 300, 200], None, None),
    "no_iterations": ([], None, None),
    "imported_iteration_zero": ([0], None, None),
    "missing_run_dir": (None, None, None),
}


@pytest.mark.parametrize("case", sorted(RESOLVE_CASES))
def test_resolve_checkpoint_path(tmp_path, case):
    iterations, tracker, explicit = RESOLVE_CASES[case]
    root = tmp_path / "run"
    if iterations is not None:
        make_checkpoint_dir(root, iterations, tracker)
    try:
        path, iteration = resolve_checkpoint_path(str(root), explicit)
        record = {"path": str(path.relative_to(tmp_path)), "iteration": iteration}
    except Exception as error:
        record = {"raised": type(error).__name__, "message": str(error).replace(str(tmp_path), "<TMP>")}
    assert_golden_json(GOLDENS / "resolve" / f"{case}.json", record)


# --- main ---------------------------------------------------------------------------------------------------------

# Runs main with argv sys.argv[1] (JSON). The two conversion paths copy $TEST_BRIDGE_OUTPUT into the output dir, and
# the Hub client is tests/unit_tests/recording_hub.py. A SystemExit is printed for the golden.
_MAIN_DRIVER = """
import json, os, shutil, sys
import huggingface_hub
from tests.unit_tests.recording_hub import HfApi

huggingface_hub.HfApi = HfApi
import pipeline_checkpoint_convert_hf as converter


def convert(iter_path, hf_path, hf_model_id, **kwargs):
    print(f"[bridge] {iter_path} -> {hf_path} ({hf_model_id}; {sorted((k, str(v)) for k, v in kwargs.items())})")
    shutil.copytree(os.environ["TEST_BRIDGE_OUTPUT"], hf_path)


converter.convert_multi_gpu = convert
converter.convert_single_process = convert
sys.argv = ["pipeline_checkpoint_convert_hf.py", *json.loads(sys.argv[1])]
try:
    converter.main()
except SystemExit as exit:
    print(f"EXIT {exit.code}")
"""

RUN_CONFIG = (
    "tokenizer:\n  tokenizer_model: geodesic-research/nemotron-instruct-tokenizer\ntrain:\n  train_iters: 300\n"
)
BASE = ["--hf-model", NANO, "--no-reasoning"]

# case -> (argv after --megatron-path, extra environment, whether the run's iteration holds a peft section)
MAIN_CASES = {
    "default": (BASE, {}, False),
    "hf_path": (BASE + ["--hf-path", "OUT"], {}, False),
    "iteration": (BASE + ["--iteration", "100"], {}, False),
    "reasoning_not_strict": (["--hf-model", NANO, "--reasoning", "--not-strict", "--no-progress"], {}, False),
    "push_to_hub": (BASE + ["--push-to-hub"], {}, False),
    "push_org_and_repo_name": (BASE + ["--push-to-hub", "--hf-org", "org", "--hf-repo-name", "name"], {}, False),
    "multi_gpu": (
        BASE + ["--tp", "1", "--pp", "2", "--ep", "2", "--torch-dtype", "float32"],
        {"WORLD_SIZE": "4"},
        False,
    ),
    "peft_single_process_refused": (BASE, {}, True),
    "peft_multi_gpu": (BASE + ["--ep", "4"], {"WORLD_SIZE": "4"}, True),
}


@pytest.mark.parametrize("case", sorted(MAIN_CASES))
def test_main(tmp_path, case):
    argv, env, peft = MAIN_CASES[case]
    run_config = RUN_CONFIG + ("peft:\n  dim: 32\n" if peft else "")
    run = make_checkpoint_dir(tmp_path / "my_run", [100, 300], tracker=300, run_config=run_config)
    bridge_output = tmp_path / "bridge_output"
    shutil.copytree(INPUTS / "release_nano", bridge_output)
    write_weights(bridge_output, 131072)
    hf_home = tmp_path / "hf"
    cache_repo(hf_home, NANO, [(SNAPSHOT_A, INPUTS / "release_nano")], SNAPSHOT_A)
    cache_repo(hf_home, INSTRUCT_TOK, [(SNAPSHOT_A, INPUTS / "nemotron-instruct-tokenizer")], SNAPSHOT_A)
    argv = [str(tmp_path / "out") if a == "OUT" else a for a in argv]
    hub_calls = tmp_path / "hub_calls.jsonl"
    child = run_python(
        ["-c", _MAIN_DRIVER, json.dumps(["--megatron-path", str(run), *argv])],
        timeout=300,
        HF_HOME=str(hf_home),
        HF_HUB_OFFLINE="1",
        TEST_BRIDGE_OUTPUT=str(bridge_output),
        STUB_CALLS=str(hub_calls),
        **env,
    )
    record = tmp_path / "record"
    record.mkdir()
    for out in (tmp_path / "out", run / "iter_0000100" / "hf", run / "iter_0000300" / "hf"):
        if out.exists():
            shutil.copytree(out, record / out.relative_to(tmp_path))
    (record / "stdout.txt").write_text(child.stdout.replace(str(tmp_path), "<TMP>"))
    calls = hub_calls.read_text() if hub_calls.exists() else ""
    (record / "hub_calls.jsonl").write_text(calls.replace(str(tmp_path), "<TMP>"))
    assert_golden_tree(GOLDENS / "main" / case, record)


# --- convert_multi_gpu's model load ---------------------------------------------------------------------------------


@pytest.mark.parametrize("peft", [False, True], ids=["full", "adapter_only_peft"])
def test_convert_multi_gpu_load_trace(monkeypatch, tmp_path, peft):
    import megatron.bridge.models.decorators as decorators
    import megatron.bridge.training.model_load_save as model_load_save

    run_config = "peft:\n  dim: 32\n" if peft else "model: {}\n"
    iteration = make_checkpoint_dir(tmp_path / "run", [4], tracker=4, run_config=run_config) / "iter_0000004"
    trace = recorded_bridge.install(monkeypatch)
    # torchrun_main initialises nothing here but ends the process on an exception; the trace needs neither.
    monkeypatch.setattr(decorators, "torchrun_main", lambda fn: fn)
    # Loading a PEFT checkpoint with its base and merge-exporting it need GPUs: record both.
    monkeypatch.setattr(
        model_load_save,
        "load_peft_model",
        lambda *a, **k: trace.record("load_peft_model", *a, **k) or [trace.obj("peft_chunk0")],
    )
    monkeypatch.setattr(hf, "export_merged_peft", lambda *a, **k: trace.record("export_merged_peft", *a, **k))
    hf.convert_multi_gpu(
        iter_path=iteration,
        hf_path=tmp_path / "hf",
        hf_model_id=NANO,
        tp=1,
        pp=2,
        ep=4,
        etp=1,
        strict=False,
        show_progress=False,
    )
    calls = json.loads(json.dumps(trace.calls).replace(str(tmp_path), "<TMP>"))
    assert_golden_json(GOLDENS / "load_trace" / f"convert_multi_gpu_{'peft' if peft else 'full'}.json", calls)
