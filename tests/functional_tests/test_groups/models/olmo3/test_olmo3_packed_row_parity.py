# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""GPU test: the packed-row parity check passes the correct OLMo-3 at four layouts and fails four injected faults.

``scripts/checkpoint/packed_row_parity.py`` compares the Megatron training forward on rows of a packed parquet with a
Hugging Face forward on each document alone. This test runs the real tool end to end on a toy OLMo-3 (eight layers: the
first, second and third slide over 64 positions, the fourth is full, and again; YaRN beyond 128 positions on the full
layers only) with weights from ``olmo3_parity.build_toy_model``, imported to a Megatron checkpoint:

* the data: chat documents of 20 to 300 words packed by ``scripts/data/pack_sft_dataset.py`` into 1,024-position rows with
  every document padded to a multiple of 8, so that most documents are longer than the sliding window and many are
  longer than the YaRN threshold;
* ``select`` draws eight rows; ``score-hf`` scores each document alone; ``score-megatron`` runs under ``torchrun`` at the
  plain layout (one GPU), pipeline 4 (four GPUs), pipeline 4 with two virtual stages and context parallel size 2;
* the four faults of ``olmo3_faults.py`` (no sliding window, YaRN on every layer, per-head QK-norm, swapped gate and up
  projections) run at the plain layout as negative controls. Each run is a calibration run that its layout's spec
  registers with its directory, its expected outcome and its fault, and ``score-megatron --calibration-run`` injects the
  fault the spec registers for it.

**The tolerance is measured.** Every run is first judged against a tolerance still PENDING, which reports the worst
document's three measures. The tolerance is then set from the plain correct run by a rule stated here before it was
applied: each limit is ``TOLERANCE_FACTOR`` times that run's worst measure over all documents, but at least
``TOLERANCE_FLOOR``. The test asserts that the other three correct layouts pass within it and that every fault fails.
The measured table is printed (run pytest with ``-s``).
"""

import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path

import pytest
import torch
import yaml
from scripts.checkpoint import packed_row_parity as prp
from scripts.checkpoint.olmo3_faults import FAULTS
from scripts.telemetry.gate_outcome import FAIL, PASS

from tests.unit_tests import packed_row_parity_fixtures as fixtures
from tests.unit_tests.chat_tokenizer_fixtures import build_chat_tokenizer


sys.path.insert(0, str(Path(__file__).parent))
from olmo3_parity import TOY_CONFIG, build_toy_model  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[5]
TOOL = REPO_ROOT / "scripts" / "checkpoint" / "packed_row_parity.py"
IMPORT_HARNESS = Path(__file__).with_name("packed_row_parity_import.py")

pytestmark = [
    pytest.mark.skipif(torch.cuda.device_count() < 4, reason="requires four GPUs"),
    pytest.mark.run_only_on("GPU"),
]

SEQ_LENGTH = 1024
PAD_MULT = 8
ROW_COUNT = 8
LENGTHS = [30, 100, 200, 150, 90, 250, 60, 180, 40, 220, 130, 75, 20, 300, 110, 55] * 5
# The correct layouts, each as the arguments of `fixtures.spec_content` that set it, and the GPUs it runs on.
LAYOUTS = {
    "plain": {},
    "pp4": {"pipeline": 4},
    "pp4_vpp2": {"pipeline": 4, "virtual_pipeline": 2},
    "cp2": {"context": 2},
}
RECIPE_FIELDS = (
    "apply_rope_fusion",
    "calculate_per_token_loss",
    "cp_comm_type",
    "cross_entropy_loss_fusion",
    "cross_entropy_fusion_impl",
    "transformer_impl",
    "cuda_graph_impl",
)
INJECTED_FAULTS = [fault for fault in FAULTS if fault != prp.NO_FAULT]
TOLERANCE_FACTOR = 4.0
TOLERANCE_FLOOR = 0.002
MEASURES = ("abs_mean_shift", "mean_abs_difference", "p99_abs_difference")
LIMIT_OF_MEASURE = {
    "abs_mean_shift": "max_abs_mean_shift",
    "mean_abs_difference": "max_mean_abs_difference",
    "p99_abs_difference": "max_p99_abs_difference",
}
JOB_TIMEOUT_SECONDS = 600


def registered_runs(layout: str) -> dict[str, dict]:
    """The calibration runs the spec of ``layout`` registers, each named for its output directory: the correct run at
    that layout, which must pass, and at the plain layout one run per injected fault, which must fail."""
    runs = {layout: {"directory": layout, "expect": PASS, "fault": prp.NO_FAULT}}
    if layout == "plain":
        runs |= {fault: {"directory": fault, "expect": FAIL, "fault": fault} for fault in INJECTED_FAULTS}
    return runs


@dataclass(frozen=True)
class Job:
    """One subprocess of the test: its command, the GPUs it may see and the file its output goes to."""

    name: str
    command: list[str]
    gpus: str
    log: Path


def tool(*arguments: str, ranks: int = 0) -> list[str]:
    """The tool's command line: under torchrun with ``ranks`` ranks, or plain python for none."""
    if ranks:
        return [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc_per_node={ranks}",
            str(TOOL),
        ] + [*arguments]
    return [sys.executable, str(TOOL), *arguments]


def run_jobs(jobs: list[Job]) -> dict[str, str]:
    """Run the jobs at once, each on its own GPUs (a GPU may be in exclusive-process mode). Returns, for each job that did
    not exit 0, its name and the tail of its output."""
    processes = []
    for job in jobs:
        environment = {
            **os.environ,
            "CUDA_VISIBLE_DEVICES": job.gpus,
            # Megatron's jit_fuser (torch.compile) off, as in the export check: the forward runs its fused ops eagerly.
            "TORCHDYNAMO_DISABLE": "1",
            "TORCH_COMPILE_DISABLE": "1",
            "TRITON_CACHE_DIR": str(job.log.parent / f"triton_{job.name}"),
        }
        job.log.parent.mkdir(parents=True, exist_ok=True)
        with job.log.open("w") as handle:
            processes.append(
                (
                    job,
                    subprocess.Popen(
                        job.command, stdout=handle, stderr=subprocess.STDOUT, env=environment, cwd=REPO_ROOT
                    ),
                )
            )
    failures = {}
    for job, process in processes:
        try:
            process.wait(timeout=JOB_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        if process.returncode != 0:
            failures[job.name] = f"{job.name} exited {process.returncode}:\n{job.log.read_text()[-3000:]}"
    return failures


def run_required(jobs: list[Job]) -> None:
    """``run_jobs`` for jobs that everything else needs: fails with the output of every one that did not exit 0."""
    failures = run_jobs(jobs)
    assert not failures, "\n\n".join(failures.values())


@dataclass(frozen=True)
class Scored:
    """The test's data and what it measured. ``rows`` is ``PENDING`` until ``select`` has drawn them; ``runs`` maps each
    run to its output directory, ``results`` to the result of judging it against a pending tolerance and ``measured`` to
    the worst measures that result reports."""

    root: Path
    packed_parquet: Path
    tokenizer: Path
    hf_dir: Path
    checkpoint: Path
    overrides: dict
    rows: list[dict] | str
    runs: dict[str, Path] = field(default_factory=dict)
    results: dict[str, dict] = field(default_factory=dict)
    failures: dict[str, str] = field(default_factory=dict)

    @property
    def measured(self) -> dict[str, dict]:
        return {name: result["ce_agreement"]["worst_measured"] for name, result in self.results.items()}

    def require(self, name: str) -> Path:
        """The output directory of run ``name``; fails the test with the output of the job that failed to make it (the
        Hugging Face scores every run needs included)."""
        for failed in ("hf", name):
            if failed in self.failures:
                pytest.fail(self.failures[failed], pytrace=False)
        return self.runs[name]

    def spec(self, layout: str, tolerance: dict | None) -> prp.PackedRowSpec:
        """The spec of a correct layout over the test's data, with a tolerance and the runs it registers
        (``registered_runs``)."""
        calibration = {"status": "test", "results_root": str(self.root / "runs"), "runs": registered_runs(layout)}
        name = f"{layout}_{'final' if tolerance else 'pending'}.yaml"
        content = fixtures.spec_content(
            packed_parquet=self.packed_parquet,
            tokenizer=self.tokenizer,
            seq_length=SEQ_LENGTH,
            pad_seq_to_mult=PAD_MULT,
            hf_model=self.hf_dir,
            megatron_checkpoint=self.checkpoint,
            count=ROW_COUNT,
            model_overrides=self.overrides,
            tolerance=tolerance,
            rows=self.rows,
            calibration=calibration,
            **LAYOUTS[layout],
        )
        (self.root / name).write_text(yaml.safe_dump(content, sort_keys=False))
        return prp.load_spec(self.root / name)


@pytest.fixture(scope="module")
def scored(tmp_path_factory) -> Scored:
    root = tmp_path_factory.mktemp("olmo3_packed_row_parity")
    logs, out = root / "logs", root / "runs"
    hf_dir, tokenizer = root / "hf", root / "tokenizer"
    hf_dir.mkdir()
    build_toy_model(str(hf_dir))
    build_chat_tokenizer(tokenizer, fixtures.VOCAB)
    shutil.copytree(tokenizer, hf_dir, dirs_exist_ok=True)
    packed_parquet = fixtures.write_packed_shard(
        root / "shard", tokenizer, LENGTHS, seq_length=SEQ_LENGTH, pad_seq_to_mult=PAD_MULT
    )
    checkpoint_root = root / "megatron"
    # Under torchrun, as every other process group of the tests is: a plain python hangs in its TCPStore rendezvous.
    import_command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=1",
        str(IMPORT_HARNESS),
        "--hf-dir",
        str(hf_dir),
        "--megatron-dir",
        str(checkpoint_root),
    ]
    run_required([Job("import", import_command, "0", logs / "import.log")])
    checkpoint = checkpoint_root / "iter_0000000"
    assert (checkpoint / "run_config.yaml").is_file(), sorted(p.name for p in checkpoint_root.iterdir())

    # The training recipe's own settings for the family, read from the recipe on the toy's config.
    from megatron.bridge.recipes.olmo import olmo3_sft_config

    recipe = olmo3_sft_config(str(hf_dir)).model
    overrides = {field_name: getattr(recipe, field_name) for field_name in RECIPE_FIELDS}

    drawing = Scored(root, packed_parquet, tokenizer, hf_dir, checkpoint, overrides, rows="PENDING")
    select_command = tool("select", "--spec", drawing.spec("plain", None).path, "--out", str(root / "rows.yaml"))
    run_required([Job("select", select_command, "0", logs / "select.log")])
    rows = yaml.safe_load((root / "rows.yaml").read_text())["rows"]
    assert len(rows) == ROW_COUNT
    base = replace(drawing, rows=rows)

    specs = {layout: base.spec(layout, None) for layout in LAYOUTS}

    def megatron_job(name: str, layout: str, gpus: str) -> Job:
        """The run ``name`` that the spec of ``layout`` registers: the tool injects its registered fault."""
        (out / name).mkdir(parents=True)
        arguments = ("score-megatron", "--spec", specs[layout].path, "--out-dir", str(out / name))
        arguments += ("--calibration-run", name)
        return Job(name, tool(*arguments, ranks=specs[layout].layout.world_size), gpus, logs / f"{name}.log")

    (out / "hf").mkdir(parents=True)
    hf_job = Job(
        "hf", tool("score-hf", "--spec", specs["plain"].path, "--out-dir", str(out / "hf")), "0", logs / "hf.log"
    )
    # Each phase gives every job its own GPUs. A job that fails fails only the tests of its own run.
    failures = run_jobs(
        [
            hf_job,
            megatron_job("plain", "plain", "1"),
            megatron_job("no_swa", "plain", "2"),
            megatron_job("yarn_all_layers", "plain", "3"),
        ]
    )
    failures |= run_jobs(
        [
            megatron_job("per_head_qknorm", "plain", "0"),
            megatron_job("swap_gate_up", "plain", "1"),
            megatron_job("cp2", "cp2", "2,3"),
        ]
    )
    failures |= run_jobs([megatron_job("pp4", "pp4", "0,1,2,3")])
    failures |= run_jobs([megatron_job("pp4_vpp2", "pp4_vpp2", "0,1,2,3")])

    # The Hugging Face scores do not depend on the layout or on a fault: every run's directory links to the one set.
    layout_of_run = {**{layout: layout for layout in LAYOUTS}, **dict.fromkeys(INJECTED_FAULTS, "plain")}
    runs = {name: out / name for name in layout_of_run}
    results = {}
    if "hf" not in failures:
        for name, layout in layout_of_run.items():
            if name not in failures:
                os.symlink(out / "hf" / prp.HF_SCORES, runs[name] / prp.HF_SCORES)
                results[name] = prp.judged_result(specs[layout], runs[name])
    result = replace(base, runs=runs, results=results, failures=failures)

    print("\nworst measure over all documents, by run (tolerance PENDING)")
    print(f"{'run':18s} {'abs mean shift':>15s} {'mean |diff|':>13s} {'p99 |diff|':>12s}")
    for name, worst in result.measured.items():
        print(
            f"{name:18s} {worst['abs_mean_shift']:15.5f} {worst['mean_abs_difference']:13.5f} {worst['p99_abs_difference']:12.5f}"
        )
    return result


def derived_tolerance(measured: dict) -> dict:
    """The tolerance the rule in the module docstring gives: ``TOLERANCE_FACTOR`` times the plain run's worst, at least the floor."""
    plain = measured["plain"]
    return {LIMIT_OF_MEASURE[m]: max(TOLERANCE_FACTOR * plain[m], TOLERANCE_FLOOR) for m in MEASURES}


def test_the_toy_rows_hold_documents_longer_than_the_window_and_the_yarn_threshold(scored):
    scored.require("plain")
    longest = {
        name: result["ce_agreement"]["coverage"]["longest_document_tokens"] for name, result in scored.results.items()
    }
    assert min(longest.values()) > TOY_CONFIG["rope_scaling"]["original_max_position_embeddings"], longest
    documents = scored.results["plain"]["ce_agreement"]["per_document"]
    assert sum(1 for d in documents if d["tokens"] > TOY_CONFIG["sliding_window"]) >= ROW_COUNT, documents


def test_every_run_is_not_evaluated_while_the_tolerance_is_pending(scored):
    for name in scored.runs:
        scored.require(name)
        result = scored.results[name]
        assert (result["verdict"], result["exit_status"]) == ("NOT EVALUATED", 2), name
        assert result["negative_control"] == (name in INJECTED_FAULTS), name


@pytest.mark.parametrize("layout", list(LAYOUTS))
def test_the_correct_model_passes_at_every_layout_within_the_tolerance_set_from_the_plain_run(scored, layout):
    scored.require("plain")
    tolerance = derived_tolerance(scored.measured)
    spec = scored.spec(layout, tolerance)
    result = prp.judge(spec, scored.require(layout))
    agreement = result["ce_agreement"]
    assert result["verdict"] == PASS, (
        f"{layout}: worst {agreement['worst_measured']} against {tolerance}; worst document {agreement['worst_document']}"
    )
    assert result["exit_status"] == 0


@pytest.mark.parametrize("fault", INJECTED_FAULTS)
def test_each_injected_fault_fails_at_that_tolerance(scored, fault):
    scored.require("plain")
    tolerance = derived_tolerance(scored.measured)
    spec = scored.spec("plain", tolerance)
    result = prp.judged_result(spec, scored.require(fault))
    agreement = result["ce_agreement"]
    assert result["negative_control"] and result["fault"] == fault
    assert result["verdict"] == FAIL, (
        f"{fault} measured {agreement['worst_measured']} against {tolerance}; the correct plain run measured "
        f"{scored.measured['plain']}"
    )
    assert agreement["documents_beyond"] > 0
    print(
        f"{fault}: {agreement['documents_beyond']} of {agreement['coverage']['documents']} documents beyond; worst {agreement['worst_measured']}"
    )


def test_calibration_passes_for_the_correct_layouts_and_the_faults(scored):
    for name in scored.runs:
        scored.require(name)
    tolerance = derived_tolerance(scored.measured)
    for layout in LAYOUTS:
        report = prp.calibrate(scored.spec(layout, tolerance))
        assert report["verdict"] == PASS, (layout, json.dumps(report["runs"], indent=1))
        assert {name: run["fault"] for name, run in report["runs"].items()} == {
            name: run["fault"] for name, run in registered_runs(layout).items()
        }
