"""`upload-all` must re-validate an export before publishing it.

An iteration counts as "already converted" if its `hf/` directory holds a
`config.json` — and a conversion whose validation FAILED leaves exactly that
behind. Without a re-check at push time, the next `upload-all` republishes the
rejected export without reading a single shard, so validating at conversion time
alone does not protect the Hub.

Runs the real `pipeline_checkpoint_convert.sh` as a subprocess against a stub
container runner. Apptainer and the Hub are the genuinely-untestable boundary:
the stub stands in for `pipeline_env_exec.sh` so no container is entered and no
`upload_folder` can reach the network. Same pattern as
`test_pipeline_data_submit_tokenize.py` and `test_pipeline_training_submit.py`.
"""

from __future__ import annotations

import json
import os
import stat
import subprocess

import pytest


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONVERT_SH = os.path.join(REPO_ROOT, "pipeline_checkpoint_convert.sh")

# upload-all derives the Hub repo name from basename(MEGATRON_PATH), so this
# string is what a push would be named if one ever escaped. Kept unmistakably
# a test artifact, and unlikely to collide with a real experiment, as a second
# line of defence behind the credential scrubbing in _run_upload_all.
FIXTURE_EXPERIMENT_NAME = "pytest-upload-gate-fixture-do-not-publish"

# The payload-executing case imports megatron.bridge in the subprocess, which
# resolves only inside the container (CLAUDE.md "Testing"). Guarding the module
# rather than a marker test means running outside the container skips rather than
# failing on an ImportError that says nothing about the gate.
pytestmark = pytest.mark.skipif(
    not os.path.isdir("/.singularity.d"),
    reason="drives a payload that imports megatron.bridge; needs the Apptainer container",
)


def _write_stub_repo(tmp_path, runner_body: str):
    """A checkout the script will accept, whose container runner we control."""
    stub_repo = tmp_path / "stub_repo"
    stub_repo.mkdir()
    (stub_repo / "pipeline_env_config.env").write_text(
        'CONTAINER_SIF="/stub/image.sif"\nenv_config_require() { return 0; }\n'
    )
    # Sourced by the payload; a no-op here because the payload's python inherits
    # the ambient PYTHONPATH of the in-container test run.
    (stub_repo / "pipeline_env_activate.sh").write_text("#!/bin/bash\n:\n")
    runner = stub_repo / "pipeline_env_exec.sh"
    runner.write_text(runner_body)
    runner.chmod(runner.stat().st_mode | stat.S_IEXEC)
    return stub_repo


def _write_converted_iteration(megatron_path, iteration: int, *, valid: bool, write_safetensors=None):
    """An iteration `is_converted()` accepts, whose export is sound or is not.

    Invalid means the index promises a shard nobody wrote — the shape a conversion
    leaves behind when validation rejected it, config.json and all.
    """
    hf_dir = megatron_path / f"iter_{iteration:07d}" / "hf"
    hf_dir.mkdir(parents=True)
    (hf_dir / "config.json").write_text("{}")
    shard = "model-00001-of-00001.safetensors"
    (hf_dir / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"lm_head.weight": shard}}))
    if valid:
        write_safetensors(hf_dir / shard, {"lm_head.weight": (4, 8)})
    return hf_dir


def _run_upload_all(stub_repo, megatron_path):
    """Drive the real script's upload-all mode.

    SLURM and the topology are pinned rather than inherited: the script refuses to
    run outside an allocation, and reading the ambient one would make the test pass
    only when it happens to be run inside a job. CONVERT_NNODES and
    MASTER_ADDR_OVERRIDE keep it off `scontrol` entirely.

    The Hub is walled off, and that is load-bearing rather than tidiness. These
    cases pass today because the gate aborts before `huggingface_hub` is imported,
    so the only way to reach the upload is for the gate to REGRESS — which is
    exactly when this test fires. An escaped push calls `create_repo(exist_ok=True)`
    then `upload_folder` against `geodesic-research/<basename of MEGATRON_PATH>`,
    so it either creates junk or commits junk into a live repo. A regression test
    must not damage anything at the moment it catches something.

    Three guards, doing different jobs — none is redundant:

    * `HF_HUB_OFFLINE` refuses the request at the HTTP layer. This is the one that
      actually stops a write today; the request is still formed, then rejected.
    * `HF_TOKEN_PATH` points at a nonexistent file. Popping the token env vars is
      NOT sufficient on this cluster: huggingface_hub falls back to
      `$HF_HOME/token`, and `pipeline_env_activate.sh` sets `HF_HOME` to a shared
      project directory that holds a group-readable token — so a subprocess
      inheriting `os.environ` is authenticated whether or not `HF_TOKEN` is set.
    * Popping the token variables covers the ordinary case where one is exported.
    """
    env = dict(os.environ)
    env["GEODESIC_REPO_DIR"] = str(stub_repo)
    env["SLURM_JOB_ID"] = "1"
    env["CONVERT_NNODES"] = "1"
    env["MASTER_ADDR_OVERRIDE"] = "localhost"
    env["MASTER_PORT_OVERRIDE"] = "29500"
    env["HF_HUB_OFFLINE"] = "1"
    env["HF_TOKEN_PATH"] = str(stub_repo / "no-such-token")
    for credential in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        env.pop(credential, None)
    return subprocess.run(
        [
            "bash",
            CONVERT_SH,
            "upload-all",
            str(megatron_path),
            "--hf-model",
            "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16",
            "--no-reasoning",
        ],
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
    )


class TestUploadAllValidatesBeforePushing:
    def test_an_already_converted_iteration_is_validated_before_it_is_uploaded(self, tmp_path, write_safetensors):
        """The gate must run, and must run first — validating after upload_folder
        would report the problem only once the bad export was already public."""
        # Echo the payload instead of running it, so both calls are observable.
        stub_repo = _write_stub_repo(tmp_path, '#!/bin/bash\nprintf "PAYLOAD:%s\\n" "$1"\n')
        megatron_path = tmp_path / FIXTURE_EXPERIMENT_NAME
        hf_dir = _write_converted_iteration(megatron_path, 100, valid=True, write_safetensors=write_safetensors)

        result = _run_upload_all(stub_repo, megatron_path)

        assert result.returncode == 0, result.stderr
        assert "Already converted, skipping conversion." in result.stdout
        payloads = [line for line in result.stdout.splitlines() if line.startswith("PAYLOAD:")]
        gate = next(i for i, p in enumerate(payloads) if "assert_export_is_publishable" in p)
        upload = next(i for i, p in enumerate(payloads) if "upload_folder" in p)
        assert gate < upload
        # The gate must be handed the directory actually being uploaded. Without
        # this, a gate pointed at the wrong path still passes both cases — the
        # validator reports no_weights_found and raises, so the rejection case
        # succeeds for the wrong reason while every real upload would be blocked.
        assert str(hf_dir) in payloads[gate]
        assert str(hf_dir) in payloads[upload]

    def test_a_rejected_export_is_not_republished(self, tmp_path):
        """The defect itself: an export that fails validation must abort the run
        rather than reach the Hub as 'already converted'."""
        # Actually execute the payload so the real validator decides.
        stub_repo = _write_stub_repo(tmp_path, '#!/bin/bash\nbash -c "$1"\n')
        megatron_path = tmp_path / FIXTURE_EXPERIMENT_NAME
        _write_converted_iteration(megatron_path, 100, valid=False)

        result = _run_upload_all(stub_repo, megatron_path)

        assert result.returncode != 0
        combined = result.stdout + result.stderr
        assert "InconsistentExportError" in combined
        assert "Do not publish or evaluate it" in combined
        # It must not have got as far as talking to the Hub. This also covers the
        # errexit subtlety: testing a function's result disables errexit for its
        # whole body, so push_to_hub has to end the iteration explicitly or it
        # would fall through from a failed gate into the upload.
        assert "Pushed to" not in combined

    # The converse — that the gate lets a sound export through — is covered at the
    # Python level by TestPublishGate::test_a_clean_export_is_allowed_through_and_its
    # _report_returned. Asserting it here too would mean either faking the whole
    # huggingface_hub package (transformers imports huggingface_hub.utils, so a flat
    # stub module breaks the import) or letting a real upload attempt reach the
    # network. Neither belongs in a unit test.

    def test_one_unpublishable_iteration_does_not_abandon_the_others(self, tmp_path, write_safetensors):
        """The job must keep going. In --poll mode this loop is the only thing that
        will ever convert the checkpoints a running training job has yet to write,
        so aborting on the first bad iteration silently abandons all the later
        ones — and the operator sees a traceback about a single old checkpoint."""
        stub_repo = _write_stub_repo(tmp_path, '#!/bin/bash\nbash -c "$1"\n')
        megatron_path = tmp_path / FIXTURE_EXPERIMENT_NAME
        _write_converted_iteration(megatron_path, 100, valid=False)
        _write_converted_iteration(megatron_path, 200, valid=True, write_safetensors=write_safetensors)

        result = _run_upload_all(stub_repo, megatron_path)
        combined = result.stdout + result.stderr

        # The bad one is named and skipped rather than ending the job...
        assert "iteration 100 was NOT published" in combined
        # ...the next one is still reached, which is the whole point...
        assert "--- Iteration 200" in combined
        # ...and the run still fails, so nothing reads "complete" off the last line.
        assert result.returncode != 0
        assert "were not published" in combined and "100" in combined
        # Iteration 200 is reported unpublished too, but for the harness's own
        # reason rather than the loop's: it passes the gate and then cannot reach
        # the walled-off Hub. Asserting an exact count here would be asserting the
        # wall, not the isolation.
