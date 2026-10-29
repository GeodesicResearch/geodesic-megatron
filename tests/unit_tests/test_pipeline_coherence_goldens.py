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
"""pipeline_coherence_test.py and pipeline_coherence_submit.sbatch, recorded as goldens.

- The submit wrapper runs under ``PipelineShell`` for each backend; the golden holds what it launches.
- ``main`` runs end to end with ``--backend endpoint`` against a local stdlib HTTP server standing in for a served
  model; the golden holds the requests the server received, the stdout and the ``--output`` file.
- ``main`` with ``--backend hf`` records run naming and prompt expansion: generation needs a GPU and model weights,
  so ``generate_hf`` is replaced by one that answers each prompt with its own text.
- ``generate_megatron`` records how it builds and loads the sharded model through a recording ``AutoBridge``
  (tests/unit_tests/recorded_bridge.py); the trace stops at the tokenizer load.

W&B is an external service, so ``wandb.init`` is replaced by a recorder in every ``main`` run.
"""

import http.server
import json
import sys
import threading
import types

import pytest

from tests.unit_tests import recorded_bridge
from tests.unit_tests.checkpoint_dirs import make_checkpoint_dir
from tests.unit_tests.goldens import FIXTURES, assert_golden_json
from tests.unit_tests.repo_scripts import load_script


GOLDENS = FIXTURES / "coherence"


@pytest.fixture
def coherence():
    return load_script("pipeline_coherence_test")


# --- the submit wrapper ---------------------------------------------------------------------------------------------

SUBMIT_CASES = {
    "submit_hf_no_args": [],
    "submit_hf_system_prompt": ["--system-prompt", "You are an AI assistant.", "--max-tokens", "64"],
    "submit_endpoint": ["--backend", "endpoint", "--base-url", "http://nid000002:8000"],
    "submit_megatron": ["--backend", "megatron", "--hf-model", "nvidia/X", "--tp", "4", "--pp", "6", "--ep", "4"],
    "submit_megatron_equals_form": ["--backend=megatron", "--hf-model", "nvidia/X", "--tp", "4", "--pp", "1"],
}


@pytest.mark.parametrize("case", sorted(SUBMIT_CASES))
def test_submit_wrapper_launch(pipeline_launch_shell, case):
    nodes = {"SLURM_NNODES": "6", "SLURM_NODELIST": "nid[000010-000015]"} if "megatron" in case else {}
    run = pipeline_launch_shell.run(
        "pipeline_coherence_submit.sbatch", "/ckpt/my run/iter_0000200", *SUBMIT_CASES[case], env=nodes
    )
    assert_golden_json(GOLDENS / f"{case}.json", pipeline_launch_shell.normalise(run))


def test_a_repo_dir_outside_the_checkout_names_both_causes(pipeline_launch_shell, tmp_path):
    """The wrapper's REPO_DIR guard, the same line as the conversion launcher's."""
    elsewhere = tmp_path / "tunnel_submit_dir"
    elsewhere.mkdir()
    run = pipeline_launch_shell.run(
        "pipeline_coherence_submit.sbatch", "/ckpt", env={"GEODESIC_REPO_DIR": str(elsewhere)}
    )
    assert run["returncode"] == 1 and run["calls"] == []
    assert f"FATAL: {elsewhere} has no pipeline/sh/common.sh, so REPO_DIR is not this repo's checkout" in run["stderr"]


# --- main --------------------------------------------------------------------------------------------------------------


class _WandbRecorder:
    """Records ``wandb.init`` kwargs and what the run logs; W&B is an external service."""

    def __init__(self):
        self.record = {}

    def init(self, **kwargs):
        self.record["init"] = kwargs
        recorder = self

        class _Run:
            summary = {}

            def log(self, data):
                recorder.record["table"] = data["generations"].data

            def finish(self):
                recorder.record["summary"] = dict(self.summary)

        return _Run()


def _main(coherence, monkeypatch, capsys, argv):
    import wandb

    recorder = _WandbRecorder()
    monkeypatch.setattr(wandb, "init", recorder.init)
    monkeypatch.setattr(sys, "argv", ["pipeline_coherence_test.py", *argv])
    coherence.main()
    return {"stdout": capsys.readouterr().out, "wandb": recorder.record}


class _Server(http.server.BaseHTTPRequestHandler):
    """An OpenAI-compatible server: lists one model, answers each chat, and fails the one asking for a joke."""

    requests: list = []

    def log_message(self, *args):
        pass

    def _reply(self, code, payload):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        type(self).requests.append({"method": "GET", "path": self.path})
        self._reply(200, {"data": [{"id": "served/model"}]})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).requests.append({"method": "POST", "path": self.path, "body": body})
        prompt = body["messages"][-1]["content"]
        if "joke" in prompt:
            self._reply(500, {"error": "boom"})
        else:
            self._reply(200, {"choices": [{"message": {"content": f"  answer to {prompt}  "}}]})


@pytest.fixture
def server():
    _Server.requests = []
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Server)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}", _Server.requests
    httpd.shutdown()


@pytest.mark.parametrize("via", ["base_url", "discovery_file"])
def test_endpoint_backend_end_to_end(coherence, monkeypatch, capsys, tmp_path, server, via):
    url, requests = server
    out = tmp_path / "gens.txt"
    if via == "base_url":
        source = ["--base-url", url]
    else:
        (tmp_path / "serve.endpoint").write_text(f"{url}/v1/\n")
        source = ["--discovery-file", str(tmp_path / "serve.endpoint")]
    argv = ["my/model", "--backend", "endpoint", *source, "--num-prompts", "5", "--max-tokens", "16"]
    argv += ["--system-prompt", "Be brief.", "--temperature", "0", "--output", str(out)]
    record = _main(coherence, monkeypatch, capsys, argv)
    record = json.loads(json.dumps(record).replace(url, "<URL>").replace(str(tmp_path), "<TMP>"))
    record["requests"] = requests
    record["output"] = out.read_text()
    assert_golden_json(GOLDENS / f"main_endpoint_{via}.json", record)


@pytest.mark.parametrize(
    "case, argv",
    [
        ("main_hf_hub_id", ["nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16", "--num-prompts", "3"]),
        ("main_hf_export_dir_n", ["/ckpt/exp/iter_0000200/hf", "--n", "7", "--num-prompts", "3"]),
        ("main_hf_completion", ["/ckpt/exp", "--generation-mode", "completion", "--system-prompt", "x"]),
        ("main_hf_run_name", ["/ckpt/exp/iter_0000200/hf", "--run-name", "mine", "--wandb-project", "p"]),
    ],
)
def test_hf_backend_naming_and_prompts(coherence, monkeypatch, capsys, case, argv):
    monkeypatch.setattr(coherence, "generate_hf", lambda args, prompts: [f"echo {p}" for p in prompts])
    assert_golden_json(GOLDENS / f"{case}.json", _main(coherence, monkeypatch, capsys, argv))


# --- the megatron backend's model load --------------------------------------------------------------------------------


class _Stop(Exception):
    """Raised at the tokenizer load: the trace covers model construction and loading only."""


def megatron_load_trace(monkeypatch, run):
    """Calls the loader makes on AutoBridge, the provider and the loaded model while ``run()`` executes.

    AutoBridge is recorded (tests/unit_tests/recorded_bridge.py says why); ``transformers.AutoTokenizer.from_pretrained``
    ends the run.
    """
    import transformers

    trace = recorded_bridge.install(monkeypatch)

    def stop(*args, **kwargs):
        trace.record("AutoTokenizer.from_pretrained", *args, **kwargs)
        raise _Stop

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", stop)
    with pytest.raises(_Stop):
        run()
    return trace.calls


@pytest.mark.parametrize("trust", [False, True])
def test_coherence_megatron_load_trace(coherence, monkeypatch, tmp_path, trust):
    run_dir = make_checkpoint_dir(tmp_path / "run", [100, 200], tracker=100)
    args = types.SimpleNamespace(
        model_path=str(run_dir),
        hf_model="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
        tokenizer=None,
        tp=1,
        pp=2,
        ep=4,
        etp=1,
        trust_remote_code=trust,
        max_tokens=32,
        generation_mode="chat",
        system_prompt=None,
    )
    calls = megatron_load_trace(monkeypatch, lambda: coherence.generate_megatron(args, ["hi"]))
    calls = json.loads(json.dumps(calls).replace(str(tmp_path), "<TMP>"))
    assert_golden_json(GOLDENS / f"megatron_load_trace_trust_{trust}.json", calls)


def test_a_run_dir_loads_its_newest_completed_save_not_one_still_being_written(tmp_path):
    """iter_0000200 is above the tracker (100), so it is still being written; the backend loads iter_0000100."""
    from pipeline.checkpoints import newest_completed_iteration

    run_dir = make_checkpoint_dir(tmp_path / "run", [100, 200], tracker=100)
    assert newest_completed_iteration(str(run_dir)) == str(run_dir / "iter_0000100")
    assert newest_completed_iteration(str(run_dir / "iter_0000200")) == str(run_dir / "iter_0000200")
    with pytest.raises(SystemExit, match="no completed save"):
        newest_completed_iteration(str(make_checkpoint_dir(tmp_path / "fresh", [10], tracker=None)))


@pytest.mark.parametrize("flag", [["--backend", "megatron"], ["--backend=megatron"]])
def test_both_spellings_of_the_megatron_backend_take_the_multi_node_path(pipeline_launch_shell, flag):
    env = {"SLURM_NNODES": "2", "SLURM_NODELIST": "nid[000010-000011]"}
    run = pipeline_launch_shell.run("pipeline_coherence_submit.sbatch", "/ckpt", *flag, "--hf-model", "x/y", env=env)
    assert run["returncode"] == 0, run["stderr"]
    assert [c["cmd"] for c in run["calls"]] == ["srun", "torchrun"]
