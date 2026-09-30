"""Unit tests for scripts/telemetry/score_run.py (training-log scorer).

Every log is the real excerpt of ``training_log_fixture.py`` or built by editing its lines; the log parser the
scorer reads through has its own tests (``test_training_log.py``).

The real log is scored against the control-pretraining baseline's stage-1 config and the Nano-30B
HF config from the local HF cache, through the FLOPs estimator's library as the CLI does: the
filtered arm trains the baseline's architecture at its sequence length
(``test_control_pretraining_30b_filtered.py`` pins that the arms differ only in data paths and run
identity). Expected throughput figures are derived from the estimator's FLOPs per token rather than
copied from it.
"""

import importlib.util
import json
import sys
from dataclasses import replace
from types import SimpleNamespace

import pytest
from scripts.nemotronh_flops_estimator import DEFAULT_PEAK_TFLOPS, resolve_hf_config
from scripts.nemotronh_flops_estimator import main as estimator_main
from scripts.training.config_compose import BASE_CONFIG_KEY, load_composed_yaml, parse_yaml_mapping

from tests.unit_tests.training_log_fixture import (
    FIXTURE,
    FIXTURE_LINES,
    FIXTURE_RUN_PATH,
    REPO_ROOT,
    iteration_line,
    sub_once,
    write_log,
)


BASELINE_CONFIG = (
    REPO_ROOT / "configs" / "control_pretraining" / "30b_baseline" / "nemotron_nano_30b_baseline_pretrain.yaml"
)
QUICKSTART_CONFIG = REPO_ROOT / "configs" / "quickstart" / "nemotron_nano_quickstart_pretrain.yaml"
NANO_HF_MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"

# 16,777,216 tokens per iteration over 256 GPUs; the window's 25 logged steps sum to 250581.7 ms.
FIXTURE_TOKENS_PER_S_PER_GPU = 65536 / 10.023268


@pytest.fixture(scope="module")
def sr():
    """Import the real script by path (scripts/telemetry/ is not an installed package)."""
    spec = importlib.util.spec_from_file_location("score_run", REPO_ROOT / "scripts" / "telemetry" / "score_run.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def nano_workload(sr):
    """The workload the CLI reads for the baseline config; skipped where the Nano HF config is not cached."""
    try:
        resolve_hf_config(NANO_HF_MODEL)
    except FileNotFoundError as exc:  # pragma: no cover - only off-cluster
        pytest.skip(f"Nano-30B HF config not in the local HF cache: {exc}")
    return sr.workload_from_config(BASELINE_CONFIG, NANO_HF_MODEL)


@pytest.fixture(scope="module")
def synthetic_workload(sr):
    """A workload with round numbers, for arithmetic checked by hand."""
    return sr.Workload(
        config_path="<synthetic>", hf_config_path="<synthetic>", seq_length=8192, model_flops_per_token=20e9
    )


def score_fixture(sr, workload, window, loss_window):
    return sr.score_log(
        FIXTURE,
        workload=workload,
        window=window,
        loss_window=loss_window,
        num_gpus=256,
        peak_tflops_per_gpu=DEFAULT_PEAK_TFLOPS,
    )


# --------------------------------------------------------------------------------------
# Workload from the training config
# --------------------------------------------------------------------------------------


def test_workload_is_what_the_estimator_reports_for_the_config(nano_workload, capsys):
    assert estimator_main([str(BASELINE_CONFIG), "--hf-model", NANO_HF_MODEL, "--json"]) == 0
    estimator = json.loads(capsys.readouterr().out)
    assert nano_workload.model_flops_per_token == estimator["model_flops_per_token"]
    assert nano_workload.hf_config_path == estimator["hf_config_path"]
    assert nano_workload.config_path == str(BASELINE_CONFIG)
    assert nano_workload.seq_length == load_composed_yaml(BASELINE_CONFIG)["model"]["seq_length"]


def test_workload_reads_an_overlay_through_its_base(sr, nano_workload):
    """The quickstart overlay names no sequence length; it is the baseline's, read through base_config."""
    overlay = sr.workload_from_config(QUICKSTART_CONFIG, NANO_HF_MODEL)
    raw = parse_yaml_mapping(QUICKSTART_CONFIG.read_text(), str(QUICKSTART_CONFIG))
    assert BASE_CONFIG_KEY in raw and "model" not in raw
    assert overlay.seq_length == nano_workload.seq_length
    assert overlay.model_flops_per_token == nano_workload.model_flops_per_token


def test_workload_with_an_unknown_model_raises(sr, tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    with pytest.raises(FileNotFoundError, match="no config.json"):
        sr.workload_from_config(BASELINE_CONFIG, "acme/not-a-real-model")


# --------------------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------------------


def test_scores_the_real_window(sr, nano_workload):
    score = score_fixture(sr, nano_workload, window=(26, 50), loss_window=(41, 50))
    # The inputs the score was computed from.
    assert score.log_path == str(FIXTURE)
    assert score.workload == nano_workload
    assert (score.num_gpus, score.global_batch_size, score.peak_tflops_per_gpu) == (256, 2048, DEFAULT_PEAK_TFLOPS)
    assert score.tokens_per_iteration == 2048 * nano_workload.seq_length == 16_777_216
    assert (score.window_first, score.window_last, score.loss_window_first, score.loss_window_last) == (26, 50, 41, 50)
    assert score.n_iterations == 25
    # The 25 logged times sum to 250581.7 ms; sorted, the 13th is 9999.5 ms.
    assert score.mean_step_s == pytest.approx(10.023268)
    assert score.median_step_s == pytest.approx(9.9995)
    # Linear interpolation at rank 0.1*24 = 2.4 between 9930.3 and 9937.2, and at 21.6
    # between 10158.1 and 10163.0.
    assert score.p10_step_s == pytest.approx(9.93306)
    assert score.p90_step_s == pytest.approx(10.16104)
    assert (score.min_step_s, score.max_step_s) == pytest.approx((9.8966, 10.2392))
    assert score.outlier_iterations == []
    assert score.tokens_per_s_per_gpu == pytest.approx(FIXTURE_TOKENS_PER_S_PER_GPU)
    expected_tflops = FIXTURE_TOKENS_PER_S_PER_GPU * nano_workload.model_flops_per_token / 1e12
    assert score.model_tflops_per_gpu == pytest.approx(expected_tflops)
    assert score.mfu == pytest.approx(expected_tflops / DEFAULT_PEAK_TFLOPS)
    # lm loss over iterations 41-50 sums to 68.27679.
    assert score.loss_mean == pytest.approx(6.827679)
    assert (score.skipped_total, score.nan_total, score.last_iteration) == (0, 0, 60)
    assert score.first_iteration_memory_gb["mem-max-allocated-gigabytes"] == 78.383
    assert score.wandb_run_path == FIXTURE_RUN_PATH
    assert score.wandb_peak_memory_gb is None


def test_throughput_and_mfu_arithmetic(sr, synthetic_workload, tmp_path):
    steps_ms = [10000.0, 10000.0, 8000.0, 12000.0]
    lines = [iteration_line(i, ms, 5.0 + i) for i, ms in enumerate(steps_ms, start=1)]
    lines = [sub_once(r"global batch size:\s+\d+", "global batch size:   512", line) for line in lines]
    score = sr.score_log(
        write_log(tmp_path, lines),
        workload=synthetic_workload,
        window=(1, 4),
        loss_window=(3, 4),
        num_gpus=64,
        peak_tflops_per_gpu=1000.0,
    )
    assert score.mean_step_s == pytest.approx(10.0)
    assert score.median_step_s == pytest.approx(10.0)
    # Sorted [8, 10, 10, 12] s: rank 0.3 -> 8 + 0.3*2, rank 2.7 -> 10 + 0.7*2.
    assert score.p10_step_s == pytest.approx(8.6)
    assert score.p90_step_s == pytest.approx(11.4)
    # 512 x 8192 = 4,194,304 tokens / (64 GPUs * 10 s) = 6553.6 tok/s/GPU; x 20 GFLOP = 131.072 TFLOP/s;
    # / 1000 TFLOP/s.
    assert score.tokens_per_iteration == 4_194_304
    assert score.tokens_per_s_per_gpu == pytest.approx(6553.6)
    assert score.model_tflops_per_gpu == pytest.approx(131.072)
    assert score.mfu == pytest.approx(0.131072)
    # Losses 8.0 and 9.0 at iterations 3 and 4.
    assert score.loss_mean == pytest.approx(8.5)
    assert score.first_iteration_memory_gb is None
    assert score.wandb_run_path is None


def test_tokens_per_iteration_follow_the_logged_batch_not_the_config(sr, nano_workload, tmp_path):
    """A batch set by a Hydra override shows only in the log; the config's 2048 must not be used."""
    lines = [iteration_line(i, 10000.0, 6.0) for i in (1, 2, 3)]
    lines = [sub_once(r"global batch size:\s+\d+", "global batch size:   512", line) for line in lines]
    score = sr.score_log(write_log(tmp_path, lines), nano_workload, (1, 3), (1, 3), 64, DEFAULT_PEAK_TFLOPS)
    assert load_composed_yaml(BASELINE_CONFIG)["train"]["global_batch_size"] == 2048
    assert score.global_batch_size == 512
    assert score.tokens_per_iteration == 512 * nano_workload.seq_length


def test_window_with_two_batch_sizes_raises(sr, synthetic_workload, tmp_path):
    lines = [iteration_line(i, 10000.0, 7.0) for i in range(1, 4)]
    lines[1] = sub_once(r"global batch size:\s+\d+", "global batch size:   512", lines[1])
    with pytest.raises(ValueError, match=r"several global batch sizes: \[512, 2048\]"):
        sr.score_log(write_log(tmp_path, lines), synthetic_workload, (1, 3), (1, 3), 1, 1.0)


def test_outliers_are_steps_strictly_above_twice_the_median(sr, synthetic_workload, tmp_path):
    steps_ms = [10000.0, 10000.0, 10000.0, 20000.0, 20001.0]
    log = write_log(tmp_path, [iteration_line(i, ms, 6.0) for i, ms in enumerate(steps_ms, start=1)])
    score = sr.score_log(log, synthetic_workload, (1, 5), (1, 5), 1, 1.0)
    assert score.median_step_s == pytest.approx(10.0)
    assert score.outlier_iterations == [5]


def test_skipped_and_nan_totals_cover_the_whole_log(sr, synthetic_workload, tmp_path):
    lines = [iteration_line(i, 10000.0, 6.0) for i in (1, 2, 3)]
    lines.append(iteration_line(4, 10000.0, 6.0, skipped=1, nan=2))
    score = sr.score_log(write_log(tmp_path, lines), synthetic_workload, (1, 3), (1, 3), 1, 1.0)
    assert (score.skipped_total, score.nan_total, score.last_iteration) == (1, 2, 4)


def test_missing_window_iteration_raises(sr, synthetic_workload):
    with pytest.raises(ValueError, match=r"missing \[61, 62\], repeated none"):
        score_fixture(sr, synthetic_workload, window=(50, 62), loss_window=(41, 50))


def test_missing_loss_window_iteration_raises(sr, synthetic_workload):
    with pytest.raises(ValueError, match=r"loss_window 55-65: .*missing \[61, 62, 63, 64, 65\]"):
        score_fixture(sr, synthetic_workload, window=(26, 50), loss_window=(55, 65))


def test_repeated_window_iteration_raises(sr, synthetic_workload, tmp_path):
    repeated = next(line for line in FIXTURE_LINES if "iteration       30/" in line)
    log = write_log(tmp_path, [*FIXTURE_LINES, repeated])
    with pytest.raises(ValueError, match=r"missing none, repeated \[30\]"):
        sr.score_log(log, synthetic_workload, (26, 50), (41, 50), 256, DEFAULT_PEAK_TFLOPS)


def test_loss_window_iteration_without_loss_raises(sr, synthetic_workload, tmp_path):
    lines = [iteration_line(i, 10000.0, 6.0) for i in (1, 2, 3)]
    lines[2] = sub_once(r" lm loss: [\dE.+-]+ \|", "", lines[2])
    with pytest.raises(ValueError, match=r"loss_window iterations \[3\] log no lm loss"):
        sr.score_log(write_log(tmp_path, lines), synthetic_workload, (1, 3), (2, 3), 1, 1.0)


@pytest.mark.parametrize(
    "window, loss_window, num_gpus, peak, workload_changes, match",
    [
        ((50, 26), (41, 50), 256, DEFAULT_PEAK_TFLOPS, {}, "window"),
        ((26, 26), (41, 50), 256, DEFAULT_PEAK_TFLOPS, {}, "window"),
        ((26, 50), (0, 5), 256, DEFAULT_PEAK_TFLOPS, {}, "loss_window"),
        ((26, 50), (41, 50), 0, DEFAULT_PEAK_TFLOPS, {}, "num_gpus"),
        ((26, 50), (41, 50), 256, 0.0, {}, "peak_tflops_per_gpu"),
        ((26, 50), (41, 50), 256, DEFAULT_PEAK_TFLOPS, {"seq_length": 0}, "workload.seq_length"),
        (
            (26, 50),
            (41, 50),
            256,
            DEFAULT_PEAK_TFLOPS,
            {"model_flops_per_token": -1.0},
            "workload.model_flops_per_token",
        ),
    ],
)
def test_invalid_parameters_raise(
    sr, synthetic_workload, window, loss_window, num_gpus, peak, workload_changes, match
):
    workload = replace(synthetic_workload, **workload_changes)
    with pytest.raises(ValueError, match=match):
        sr.score_log(FIXTURE, workload, window, loss_window, num_gpus, peak)


def test_to_dict_is_json_serialisable_and_carries_the_inputs(sr, synthetic_workload):
    score = score_fixture(sr, synthetic_workload, window=(26, 50), loss_window=(41, 50))
    restored = json.loads(json.dumps(score.to_dict()))
    assert restored["mean_step_s"] == pytest.approx(10.023268)
    assert restored["first_iteration_memory_gb"]["mem-max-reserved-gigabytes"] == 80.487
    assert restored["workload"] == {
        "config_path": "<synthetic>",
        "hf_config_path": "<synthetic>",
        "seq_length": 8192,
        "model_flops_per_token": 20e9,
    }
    assert restored["log_path"] == str(FIXTURE)
    assert (restored["num_gpus"], restored["global_batch_size"]) == (256, 2048)


# --------------------------------------------------------------------------------------
# W&B peak memory
# --------------------------------------------------------------------------------------

FIXTURE_WANDB_SUMMARY = {
    "memory/mem-max-allocated-gigabytes": 81.8,
    "memory/mem-max-reserved-gigabytes": 84,
    "memory/mem-allocated-gigabytes": 61.5,
    "memory/mem-alloc-retires": 3.0,
}


class FakeApi:
    """Stand-in for ``wandb.Api``, the network boundary: the real client reads the run over the network
    from W&B's service, which unit tests must not depend on. It records the run path it was asked for,
    and raises ``error`` from ``run`` when one is given, as the real client does on a failed request."""

    requested: list[str] = []

    def __init__(self, summary: dict, error: Exception | None):
        self.summary = summary
        self.error = error

    def run(self, path: str) -> SimpleNamespace:
        FakeApi.requested.append(path)
        if self.error is not None:
            raise self.error
        return SimpleNamespace(summary=self.summary)


def install_fake_wandb(monkeypatch, summary: dict, error: Exception | None = None) -> None:
    # The scorer imports wandb at call time, so a module in sys.modules replaces the client.
    FakeApi.requested = []
    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(Api=lambda: FakeApi(summary, error)))


def test_fetch_wandb_peak_memory_reads_both_summary_keys(sr, monkeypatch):
    install_fake_wandb(monkeypatch, FIXTURE_WANDB_SUMMARY)
    peaks = sr.fetch_wandb_peak_memory(FIXTURE_RUN_PATH)
    assert peaks == {"memory/mem-max-allocated-gigabytes": 81.8, "memory/mem-max-reserved-gigabytes": 84.0}
    assert isinstance(peaks["memory/mem-max-reserved-gigabytes"], float)
    assert FakeApi.requested == [FIXTURE_RUN_PATH]


def test_fetch_wandb_peak_memory_missing_key_raises(sr, monkeypatch):
    install_fake_wandb(monkeypatch, {"memory/mem-max-allocated-gigabytes": 81.8})
    with pytest.raises(KeyError, match="mem-max-reserved-gigabytes"):
        sr.fetch_wandb_peak_memory(FIXTURE_RUN_PATH)


def test_fetch_wandb_alloc_retries_reads_the_count_as_an_integer(sr, monkeypatch):
    install_fake_wandb(monkeypatch, FIXTURE_WANDB_SUMMARY)
    retries = sr.fetch_wandb_alloc_retries(FIXTURE_RUN_PATH)
    assert retries == 3 and isinstance(retries, int)
    assert FakeApi.requested == [FIXTURE_RUN_PATH]


def test_fetch_wandb_alloc_retries_missing_key_raises(sr, monkeypatch):
    install_fake_wandb(monkeypatch, {"memory/mem-max-allocated-gigabytes": 81.8})
    with pytest.raises(KeyError, match="mem-alloc-retires"):
        sr.fetch_wandb_alloc_retries(FIXTURE_RUN_PATH)


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cli_args(nano_workload):
    """The CLI arguments that score the fixture window against the baseline config (skipped off-cluster)."""
    return [
        "--config",
        str(BASELINE_CONFIG),
        "--hf-model",
        NANO_HF_MODEL,
        "--gpus",
        "256",
        "--window",
        "26",
        "50",
        "--loss-window",
        "41",
        "50",
    ]


def test_cli_json_scores_the_log_and_records_its_inputs(sr, nano_workload, cli_args, capsys):
    assert sr.main([str(FIXTURE), *cli_args, "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["log_path"] == str(FIXTURE)
    assert result["workload"] == {
        "config_path": str(BASELINE_CONFIG),
        "hf_config_path": nano_workload.hf_config_path,
        "seq_length": nano_workload.seq_length,
        "model_flops_per_token": nano_workload.model_flops_per_token,
    }
    assert (result["num_gpus"], result["global_batch_size"]) == (256, 2048)
    assert result["peak_tflops_per_gpu"] == DEFAULT_PEAK_TFLOPS
    assert [result[key] for key in ("window_first", "window_last", "loss_window_first", "loss_window_last")] == [
        26,
        50,
        41,
        50,
    ]
    assert result["tokens_per_s_per_gpu"] == pytest.approx(FIXTURE_TOKENS_PER_S_PER_GPU)
    expected_tflops = FIXTURE_TOKENS_PER_S_PER_GPU * nano_workload.model_flops_per_token / 1e12
    assert result["mfu"] == pytest.approx(expected_tflops / DEFAULT_PEAK_TFLOPS)
    assert result["wandb_run_path"] == FIXTURE_RUN_PATH
    assert result["wandb_peak_memory_gb"] is None
    assert result["wandb_alloc_retries"] is None


def test_cli_text_report_states_inputs_and_results(sr, nano_workload, cli_args, capsys):
    assert sr.main([str(FIXTURE), *cli_args]) == 0
    report = capsys.readouterr().out
    expected_tflops = FIXTURE_TOKENS_PER_S_PER_GPU * nano_workload.model_flops_per_token / 1e12
    for row in (
        f"log                     {FIXTURE}",
        f"config                  {BASELINE_CONFIG}",
        f"HF config               {nano_workload.hf_config_path}",
        "GPUs                    256",
        "seq length x GBS        8192 x 2048 (logged) = 16,777,216 tokens/iteration",
        f"model FLOPs/token       {nano_workload.model_flops_per_token / 1e9:.4f} GFLOP",
        f"peak                    {DEFAULT_PEAK_TFLOPS:g} TFLOP/s/GPU",
        "window                  26-50 (25 iterations)",
        "loss window             41-50",
        "step mean (primary)     10.0233 s",
        "tokens/s/GPU            6,538.4",
        f"MFU                     {expected_tflops / DEFAULT_PEAK_TFLOPS:.2%}",
        "W&B peak memory (GB)    not fetched (--wandb-peak-memory)",
        "W&B alloc retries       not fetched (--wandb-peak-memory)",
    ):
        assert row in report, row


def test_cli_peak_override_scales_mfu(sr, cli_args, capsys):
    assert sr.main([str(FIXTURE), *cli_args, "--json"]) == 0
    at_default = json.loads(capsys.readouterr().out)
    assert sr.main([str(FIXTURE), *cli_args, "--peak-tflops", "500", "--json"]) == 0
    at_500 = json.loads(capsys.readouterr().out)
    assert at_500["peak_tflops_per_gpu"] == 500.0
    assert at_500["model_tflops_per_gpu"] == at_default["model_tflops_per_gpu"]
    assert at_500["mfu"] == pytest.approx(at_default["mfu"] * DEFAULT_PEAK_TFLOPS / 500)


def test_cli_wandb_peak_memory_adds_the_summary_peaks(sr, cli_args, monkeypatch, capsys):
    install_fake_wandb(monkeypatch, FIXTURE_WANDB_SUMMARY)
    assert sr.main([str(FIXTURE), *cli_args, "--wandb-peak-memory", "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["wandb_peak_memory_gb"] == {
        "memory/mem-max-allocated-gigabytes": 81.8,
        "memory/mem-max-reserved-gigabytes": 84.0,
    }
    assert result["wandb_alloc_retries"] == 3
    assert set(FakeApi.requested) == {FIXTURE_RUN_PATH}

    assert sr.main([str(FIXTURE), *cli_args, "--wandb-peak-memory"]) == 0
    report = capsys.readouterr().out
    assert "W&B peak memory (GB)    mem-max-allocated 81.800, mem-max-reserved 84.000" in report
    assert "W&B alloc retries       3" in report


def test_cli_wandb_peak_memory_propagates_network_errors(sr, cli_args, monkeypatch):
    install_fake_wandb(monkeypatch, FIXTURE_WANDB_SUMMARY, error=ConnectionError("W&B unreachable"))
    with pytest.raises(ConnectionError, match="W&B unreachable"):
        sr.main([str(FIXTURE), *cli_args, "--wandb-peak-memory"])


def test_cli_wandb_peak_memory_needs_a_run_in_the_log(sr, cli_args, tmp_path, monkeypatch):
    install_fake_wandb(monkeypatch, FIXTURE_WANDB_SUMMARY)
    no_run = write_log(tmp_path, [line for line in FIXTURE_LINES if "View run" not in line])
    with pytest.raises(ValueError, match="names no W&B run"):
        sr.main([str(no_run), *cli_args, "--wandb-peak-memory"])
    assert FakeApi.requested == []


@pytest.mark.parametrize("dropped", ["--config", "--hf-model", "--gpus", "--window", "--loss-window"])
def test_cli_requires_the_config_the_model_and_every_window(sr, dropped):
    args = [
        str(FIXTURE),
        "--config",
        str(BASELINE_CONFIG),
        "--hf-model",
        NANO_HF_MODEL,
        "--gpus",
        "256",
        "--window",
        "26",
        "50",
        "--loss-window",
        "41",
        "50",
    ]
    index = args.index(dropped)
    arity = 3 if dropped in ("--window", "--loss-window") else 2
    with pytest.raises(SystemExit):
        sr.main(args[:index] + args[index + arity :])
