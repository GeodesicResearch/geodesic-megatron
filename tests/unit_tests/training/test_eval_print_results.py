# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""How an evaluation's totals become its reported results.

``evaluation_results`` turns the totals ``evaluate`` sums over the whole evaluation, each ``[numerator, denominator]``,
into results: an entry whose denominator is 0 (``lm loss`` when no target of the evaluation carries loss) is left out
with a warning rather than reported as NaN, and the token-masking listed-target loss sum becomes the listed-target
loss. ``evaluate_and_print_results`` then logs each result as ``<key> validation`` to TensorBoard, W&B, MLflow and
Comet, and gives only the losses (``lm loss``, ``token_masking/listed_target_loss``) a perplexity: ``<key> validation
ppl`` (when ``logger.log_validation_ppl_to_tensorboard`` asks for it) and a ``PPL`` in the printed line. A NaN loss
gets a NaN perplexity, not the exp(20) cap.
"""

import math
from types import SimpleNamespace

import pytest
import torch

import megatron.bridge.training.eval as eval_module
from megatron.bridge.training.token_masking.hook import (
    LISTED_TARGET_LOSS,
    LISTED_TARGET_LOSS_SUM,
    LISTED_TRAINABLE_TARGET_FRACTION,
    REPORT_KEYS,
    TRAINABLE_TARGET_FRACTION,
)


LOSS = 2.0
LISTED_LOSS = 3.0
FRACTION = 0.25
STEP = 5
CONSUMED_SAMPLES = 40
# The fractions an evaluation reports: every token-masking report entry except the sum the listed-target loss replaces.
FRACTION_KEYS = [key for key in REPORT_KEYS if key != LISTED_TARGET_LOSS_SUM]
LOSS_KEYS = ["lm loss", LISTED_TARGET_LOSS]


class RecordingTensorBoard:
    def __init__(self) -> None:
        self.scalars: dict[str, tuple[float, int]] = {}

    def add_scalar(self, name: str, value: float, step: int) -> None:
        self.scalars[name] = (value, step)


class RecordingWandb:
    def __init__(self) -> None:
        self.logged: dict[str, tuple[float, int]] = {}

    def log(self, metrics: dict[str, float], step: int) -> None:
        self.logged.update({name: (value, step) for name, value in metrics.items()})


class RecordingMetricsLogger:
    """The ``log_metrics`` interface MLflow and Comet loggers share."""

    def __init__(self) -> None:
        self.logged: dict[str, tuple[float, int]] = {}

    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        self.logged.update({name: (value, step) for name, value in metrics.items()})


class TestEvaluationResults:
    def test_each_result_is_its_numerator_over_its_denominator(self):
        results = eval_module.evaluation_results(
            {"lm loss": torch.tensor([30.0, 12.0]), TRAINABLE_TARGET_FRACTION: torch.tensor([12.0, 16.0])}
        )
        assert {key: value.item() for key, value in results.items()} == pytest.approx(
            {"lm loss": 2.5, TRAINABLE_TARGET_FRACTION: 0.75}
        )

    def test_the_listed_target_loss_sum_becomes_the_listed_target_loss(self):
        """Summed over the evaluation, both are over the same 16 positions: 13 nats over 2 listed targets."""
        results = eval_module.evaluation_results(
            {
                "lm loss": torch.tensor([30.0, 12.0]),
                LISTED_TRAINABLE_TARGET_FRACTION: torch.tensor([2.0, 16.0]),
                LISTED_TARGET_LOSS_SUM: torch.tensor([13.0, 16.0]),
            }
        )
        assert set(results) == {"lm loss", LISTED_TRAINABLE_TARGET_FRACTION, LISTED_TARGET_LOSS}
        assert results[LISTED_TARGET_LOSS].item() == pytest.approx(6.5)

    def test_an_entry_with_no_denominator_is_left_out_with_a_warning(self, capsys):
        """Every target masked: ``lm loss`` is [0, 0] over the evaluation, which has no value; no NaN is reported."""
        results = eval_module.evaluation_results(
            {
                "lm loss": torch.tensor([0.0, 0.0]),
                TRAINABLE_TARGET_FRACTION: torch.tensor([0.0, 16.0]),
                LISTED_TRAINABLE_TARGET_FRACTION: torch.tensor([0.0, 16.0]),
                LISTED_TARGET_LOSS_SUM: torch.tensor([0.0, 16.0]),
            }
        )
        assert set(results) == {TRAINABLE_TARGET_FRACTION, LISTED_TRAINABLE_TARGET_FRACTION}
        assert not any(math.isnan(value.item()) for value in results.values())
        printed = capsys.readouterr().out
        assert "WARNING: lm loss has no value in this evaluation (its denominator is 0)" in printed


@pytest.fixture
def run(monkeypatch):
    """Run ``evaluate_and_print_results`` on fixed evaluation results; return the writers and the printed lines.

    The evaluation loop needs a model, a data iterator and an initialised pipeline schedule, and the rank helpers
    need torch.distributed, none of which this decision depends on. So ``evaluate`` returns fixed results holding the
    two losses and every token-masking fraction, this process stands in for the last rank, and the printing goes to a
    list; the code under test, which turns those results into log entries and the printed line, runs as it does in
    training.
    """
    printed: list[str] = []
    monkeypatch.setattr(eval_module, "is_last_rank", lambda: True)
    monkeypatch.setattr(eval_module, "print_rank_last", printed.append)

    def evaluate_and_print(log_ppl: bool, loss: float = LOSS) -> SimpleNamespace:
        results = {
            "lm loss": torch.tensor(loss),
            LISTED_TARGET_LOSS: torch.tensor(LISTED_LOSS),
            **{key: torch.tensor(FRACTION) for key in FRACTION_KEYS},
        }
        monkeypatch.setattr(eval_module, "evaluate", lambda *args, **kwargs: (results, None, False))
        state = SimpleNamespace(
            cfg=SimpleNamespace(logger=SimpleNamespace(log_validation_ppl_to_tensorboard=log_ppl)),
            train_state=SimpleNamespace(step=STEP, consumed_train_samples=CONSUMED_SAMPLES),
            tensorboard_logger=RecordingTensorBoard(),
            wandb_logger=RecordingWandb(),
            mlflow_logger=RecordingMetricsLogger(),
            comet_logger=RecordingMetricsLogger(),
        )
        eval_module.evaluate_and_print_results(
            state,
            "iteration 5",
            forward_step_func=None,
            data_iterator=None,
            model=[],
            config=state.cfg,
            write_to_tensorboard=True,
            callback_manager=None,
        )
        return SimpleNamespace(
            tensorboard=state.tensorboard_logger.scalars,
            wandb=state.wandb_logger.logged,
            mlflow=state.mlflow_logger.logged,
            comet=state.comet_logger.logged,
            printed="\n".join(printed),
        )

    return evaluate_and_print


def _ppl_names(logged: dict) -> set[str]:
    return {name for name in logged if "ppl" in name}


def _mlflow_name(name: str) -> str:
    """The name MLflow logs ``val/<name>`` under: its sanitiser keeps only the first ``/`` of a metric name."""
    return "val/" + name.replace("/", "_")


def test_only_the_losses_get_a_perplexity(run):
    logged = run(log_ppl=True)
    expected = {"lm loss": pytest.approx(math.exp(LOSS)), LISTED_TARGET_LOSS: pytest.approx(math.exp(LISTED_LOSS))}

    assert _ppl_names(logged.tensorboard) == {
        name for key in LOSS_KEYS for name in (f"{key} validation ppl", f"{key} validation ppl vs samples")
    }
    assert _ppl_names(logged.wandb) == {f"{key} validation ppl" for key in LOSS_KEYS}
    assert _ppl_names(logged.mlflow) == {_mlflow_name(f"{key} ppl") for key in LOSS_KEYS}
    assert _ppl_names(logged.comet) == {f"{key} validation ppl" for key in LOSS_KEYS}
    for key, ppl in expected.items():
        assert logged.tensorboard[f"{key} validation ppl"] == (ppl, STEP)
        assert logged.tensorboard[f"{key} validation ppl vs samples"] == (ppl, CONSUMED_SAMPLES)
        assert logged.wandb[f"{key} validation ppl"] == (ppl, STEP)
        assert logged.mlflow[_mlflow_name(f"{key} ppl")] == (ppl, STEP)
        assert logged.comet[f"{key} validation ppl"] == (ppl, STEP)
        assert f"{key} PPL: " in logged.printed
    for key in FRACTION_KEYS:
        assert f"{key} PPL" not in logged.printed
        assert f"{key} value: " in logged.printed


def test_every_entry_keeps_its_validation_value(run):
    logged = run(log_ppl=True)
    for key, value in [
        ("lm loss", LOSS),
        (LISTED_TARGET_LOSS, LISTED_LOSS),
        *[(key, FRACTION) for key in FRACTION_KEYS],
    ]:
        assert logged.tensorboard[f"{key} validation"] == (pytest.approx(value), STEP)
        assert logged.tensorboard[f"{key} validation vs samples"] == (pytest.approx(value), CONSUMED_SAMPLES)
        assert logged.wandb[f"{key} validation"] == (pytest.approx(value), STEP)
        assert logged.comet[f"{key} validation"] == (pytest.approx(value), STEP)
    assert logged.mlflow["val/lm loss"] == (pytest.approx(LOSS), STEP)
    assert len(logged.mlflow) == len(LOSS_KEYS) + len(FRACTION_KEYS) + len(LOSS_KEYS), (
        "one value per entry, plus each loss's perplexity"
    )


def test_no_perplexity_is_logged_when_the_config_turns_it_off(run):
    logged = run(log_ppl=False)
    for backend in (logged.tensorboard, logged.wandb, logged.mlflow, logged.comet):
        assert _ppl_names(backend) == set()
    assert logged.wandb["lm loss validation"] == (pytest.approx(LOSS), STEP)
    assert "lm loss PPL: " in logged.printed
    for key in FRACTION_KEYS:
        assert f"{key} PPL" not in logged.printed


def test_a_nan_loss_gets_a_nan_perplexity_not_the_cap(run):
    logged = run(log_ppl=True, loss=math.nan)
    value, step = logged.wandb["lm loss validation ppl"]
    assert math.isnan(value) and step == STEP
    assert "lm loss PPL: NAN" in logged.printed


def test_a_large_loss_is_capped_at_exp_20(run):
    logged = run(log_ppl=True, loss=50.0)
    assert logged.wandb["lm loss validation ppl"] == (pytest.approx(math.exp(20)), STEP)
