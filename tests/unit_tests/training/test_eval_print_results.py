# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""``evaluate_and_print_results`` reports a validation perplexity for losses only.

With token masking the validation dict also holds the ``token_masking/*`` fractions, for which a perplexity means
nothing. Each entry must still be logged as ``<key> validation`` to TensorBoard, W&B, MLflow and Comet, while only the
loss gets ``<key> validation ppl`` (when ``logger.log_validation_ppl_to_tensorboard`` asks for it) and a ``PPL`` in the
printed line.
"""

import math
from types import SimpleNamespace

import pytest
import torch

import megatron.bridge.training.eval as eval_module
from megatron.bridge.training.token_masking.hook import REPORT_KEYS


LOSS = 2.0
FRACTION = 0.25
STEP = 5
CONSUMED_SAMPLES = 40


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


@pytest.fixture
def run(monkeypatch):
    """Run ``evaluate_and_print_results`` on a fixed validation dict; return the writers and the printed lines.

    The evaluation loop needs a model, a data iterator and an initialised pipeline schedule, and the rank helpers
    need torch.distributed, none of which this decision depends on. So ``evaluate`` returns a fixed dict holding the
    loss and every token-masking fraction, this process stands in for the last rank, and the printing goes to a list;
    the code under test, which turns that dict into log entries and the printed line, runs as it does in training.
    """
    total_loss_dict = {"lm loss": torch.tensor(LOSS), **{key: torch.tensor(FRACTION) for key in REPORT_KEYS}}
    monkeypatch.setattr(eval_module, "evaluate", lambda *args, **kwargs: (total_loss_dict, None, False))
    monkeypatch.setattr(eval_module, "is_last_rank", lambda: True)
    printed: list[str] = []
    monkeypatch.setattr(eval_module, "print_rank_last", printed.append)

    def evaluate_and_print(log_ppl: bool) -> SimpleNamespace:
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


def test_only_the_loss_gets_a_perplexity(run):
    logged = run(log_ppl=True)
    ppl = pytest.approx(math.exp(LOSS))

    assert _ppl_names(logged.tensorboard) == {"lm loss validation ppl", "lm loss validation ppl vs samples"}
    assert logged.tensorboard["lm loss validation ppl"] == (ppl, STEP)
    assert logged.tensorboard["lm loss validation ppl vs samples"] == (ppl, CONSUMED_SAMPLES)
    assert _ppl_names(logged.wandb) == {"lm loss validation ppl"}
    assert logged.wandb["lm loss validation ppl"] == (ppl, STEP)
    assert _ppl_names(logged.mlflow) == {"val/lm loss ppl"}
    assert logged.mlflow["val/lm loss ppl"] == (ppl, STEP)
    assert _ppl_names(logged.comet) == {"lm loss validation ppl"}
    assert logged.comet["lm loss validation ppl"] == (ppl, STEP)

    assert "lm loss PPL: " in logged.printed
    for key in REPORT_KEYS:
        assert f"{key} PPL" not in logged.printed
        assert f"{key} value: " in logged.printed


def test_every_entry_keeps_its_validation_value(run):
    logged = run(log_ppl=True)
    for key, value in [("lm loss", LOSS), *[(key, FRACTION) for key in REPORT_KEYS]]:
        assert logged.tensorboard[f"{key} validation"] == (pytest.approx(value), STEP)
        assert logged.tensorboard[f"{key} validation vs samples"] == (pytest.approx(value), CONSUMED_SAMPLES)
        assert logged.wandb[f"{key} validation"] == (pytest.approx(value), STEP)
        assert logged.comet[f"{key} validation"] == (pytest.approx(value), STEP)
    assert logged.mlflow["val/lm loss"] == (pytest.approx(LOSS), STEP)
    assert len(logged.mlflow) == 1 + len(REPORT_KEYS) + 1, "one value per entry, plus the loss's perplexity"


def test_no_perplexity_is_logged_when_the_config_turns_it_off(run):
    logged = run(log_ppl=False)
    for backend in (logged.tensorboard, logged.wandb, logged.mlflow, logged.comet):
        assert _ppl_names(backend) == set()
    assert logged.wandb["lm loss validation"] == (pytest.approx(LOSS), STEP)
    assert "lm loss PPL: " in logged.printed
    for key in REPORT_KEYS:
        assert f"{key} PPL" not in logged.printed
