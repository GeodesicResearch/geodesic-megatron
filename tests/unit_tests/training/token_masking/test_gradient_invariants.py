# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""What token masking does to the gradient, shown with autograd on a tiny model.

The loss runs as it does in training: logits from a linear output head, Megatron's vocab-parallel cross-entropy (the
unfused path, over a real single-process gloo group as the tensor-parallel group), the real ``apply_token_masking``
and ``masked_next_token_loss``, and Megatron's real ``forward_step_calc_loss`` normalisation (divide by the
microbatch's trained-token count, clamped to at least 1, unless ``calculate_per_token_loss``). With ``m_t`` the mask
the loss multiplies, ``p_t`` the softmax at position ``t``, ``h_t`` its hidden state and ``c`` the normalisation, the
loss is ``c * sum_t m_t * CE_t`` and its gradient follows:

(a) the logit-gradient row of a masked position is exactly 0;
(b) the masked id's logit-gradient column, ``c * m_t * (p_t(j) - [y_t = j])``, is never negative when masking (the
    ``- 1`` term needs ``m_t != 0`` at a marker label), and negative at its trained label positions when not;
(c) so its output row's gradient is ``c * sum_t m_t p_t(j) h_t`` when masking (a pure push-down), plus
    ``- c * sum_{t: y_t = j} m_t h_t`` (the pull-up) when not;
(d) trained for a few hundred Adam steps on data where the marker is perfectly predictable, the cross-entropy at the
    marker's targets rises when masking and falls when not.

Also Kyle's all-masked cases: a microbatch whose every trained target is the marker gives loss 0, count 0 and an
exactly-zero gradient (never NaN), and a global batch made only of such microbatches stops an enabled run instead of
logging its 0/0 loss.
"""

import megatron.core.rerun_state_machine as rerun_state_machine_module
import pytest
import torch
from megatron.core.pipeline_parallel.schedules import forward_step_calc_loss
from megatron.core.rerun_state_machine import RerunStateMachine
from megatron.core.tensor_parallel import vocab_parallel_cross_entropy
from megatron.core.transformer import TransformerConfig

from megatron.bridge.training.losses import create_masked_next_token_loss_function
from megatron.bridge.training.token_masking.config import TokenMaskingError
from megatron.bridge.training.token_masking.hook import LISTED_TARGET_LOSS, apply_token_masking
from megatron.bridge.training.token_masking.monitor import TokenMaskingMonitor
from megatron.bridge.training.train import report_step_losses
from tests.unit_tests.token_masking_fixtures import masking_with_null_tokenizer, measuring_with_null_tokenizer


VOCAB_SIZE = 8
HIDDEN = 4
MARKER = 7

# One microbatch of twelve targets. The marker is the label at positions 2, 5 and 9; the dataset trains positions 2
# and 9 and excludes 5 (a prompt, say), along with position 0.
LABELS = torch.tensor([[1, 2, MARKER, 3, 4, MARKER, 5, 6, 1, MARKER, 2, 3]])
DATASET_MASK = torch.tensor([[0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1]], dtype=torch.float)
MARKER_LABELS = (LABELS == MARKER).reshape(-1)


@pytest.fixture(autouse=True)
def rerun_state_machine(monkeypatch):
    """Megatron's real rerun state machine in its default (disabled) mode, which the loss function consults.

    Installed as the process-wide singleton for one test and removed afterwards, so other tests find it as they left
    it.
    """
    monkeypatch.setattr(rerun_state_machine_module, "_GLOBAL_RERUN_STATE_MACHINE", RerunStateMachine())


@pytest.fixture(scope="module")
def decisions() -> dict:
    return {
        "masking": masking_with_null_tokenizer([MARKER], VOCAB_SIZE),
        "measuring": measuring_with_null_tokenizer([MARKER], VOCAB_SIZE),
    }


def _schedule_config(calculate_per_token_loss: bool) -> TransformerConfig:
    """The config ``forward_step_calc_loss`` reads (its loss normalisation; no MoE, no MTP, no timers)."""
    return TransformerConfig(
        num_layers=1, hidden_size=HIDDEN, num_attention_heads=1, calculate_per_token_loss=calculate_per_token_loss
    )


def _per_token_losses(logits: torch.Tensor, labels: torch.Tensor, group) -> torch.Tensor:
    """Megatron's cross-entropy as a GPT model computes it: sequence-first logits [s, b, V] and labels [b, s] give the
    per-token losses [b, s] its loss function receives."""
    return vocab_parallel_cross_entropy(logits, labels.t().contiguous(), tp_group=group).t().contiguous()


def _scheduled_loss(losses, labels, dataset_mask, decision, config, num_microbatches=1, model=None):
    """One microbatch's loss as the pipeline schedule backpropagates it, with its token count and report."""
    masked_mask, stats = apply_token_masking(labels, dataset_mask, decision)
    loss_function = create_masked_next_token_loss_function(
        masked_mask, check_for_nan_in_loss=True, check_for_spiky_loss=False, token_masking_stats=stats
    )
    reports = []
    loss, num_tokens = forward_step_calc_loss(
        model,
        losses,
        loss_function,
        config,
        vp_stage=None,
        collect_non_loss_data=False,
        num_microbatches=num_microbatches,
        forward_data_store=reports,
        cp_group_size=1,
        is_last_stage=True,
    )
    return loss, num_tokens, reports[0], masked_mask


def _normalisation(num_tokens: torch.Tensor, config: TransformerConfig) -> float:
    """``c``: the schedule divides by the trained-token count (at least 1) unless the loss is per token."""
    return 1.0 if config.calculate_per_token_loss else 1.0 / max(1, int(num_tokens))


@pytest.fixture
def head():
    """Fixed hidden states [s, b, d] (one sequence), a linear output head's weight, and the logits [s, b, V] it gives,
    keeping their gradient."""
    generator = torch.Generator().manual_seed(0)
    hidden = torch.randn(LABELS.shape[1], 1, HIDDEN, generator=generator)
    weight = torch.nn.Parameter(torch.randn(VOCAB_SIZE, HIDDEN, generator=generator))
    logits = hidden @ weight.t()
    logits.retain_grad()
    return hidden, weight, logits


PER_TOKEN_LOSS = [pytest.param(False, id="per-microbatch-mean"), pytest.param(True, id="per-token")]


@pytest.mark.parametrize("calculate_per_token_loss", PER_TOKEN_LOSS)
@pytest.mark.parametrize("arm", ["masking", "measuring"])
def test_the_gradient_follows_the_masked_cross_entropy(
    arm, calculate_per_token_loss, decisions, head, gloo_group_of_one
):
    hidden, weight, logits = head
    config = _schedule_config(calculate_per_token_loss)
    losses = _per_token_losses(logits, LABELS, gloo_group_of_one)
    loss, num_tokens, _, masked_mask = _scheduled_loss(losses, LABELS, DATASET_MASK, decisions[arm], config)
    loss.backward()

    masking = arm == "masking"
    mask = masked_mask.reshape(-1)
    c = _normalisation(num_tokens, config)
    logit_grad = logits.grad.reshape(-1, VOCAB_SIZE)
    h = hidden.reshape(-1, HIDDEN).double()
    p = torch.softmax(h @ weight.detach().double().t(), dim=-1)
    trained_marker_labels = MARKER_LABELS & (DATASET_MASK.reshape(-1) != 0)
    assert int(num_tokens) == (8 if masking else 10)

    # (a) A position the loss does not multiply has an exactly-zero logit-gradient row; every other row is not zero.
    untrained = mask == 0
    assert torch.equal(logit_grad[untrained], torch.zeros_like(logit_grad[untrained]))
    assert (logit_grad[~untrained].abs().sum(dim=-1) > 0).all()
    if masking:
        assert untrained[MARKER_LABELS].all()

    # (b) The marker's logit-gradient column: never negative when masking; negative where it is a trained label when
    # not, which is the only place a gradient step raises the marker's logit.
    column = logit_grad[:, MARKER]
    if masking:
        assert (column >= 0).all()
    else:
        assert (column[trained_marker_labels] < 0).all()
        assert (column[~trained_marker_labels] >= 0).all()

    # (c) The marker's output-row gradient: the push-down alone when masking, the push-down plus the pull-up when not.
    push_down = c * ((mask.double() * p[:, MARKER]).unsqueeze(-1) * h).sum(dim=0)
    pull_up = c * h[trained_marker_labels].sum(dim=0)
    expected = push_down if masking else push_down - pull_up
    torch.testing.assert_close(weight.grad[MARKER].double(), expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("calculate_per_token_loss", PER_TOKEN_LOSS)
def test_a_fully_masked_microbatch_has_zero_loss_count_and_gradient(
    calculate_per_token_loss, decisions, head, gloo_group_of_one
):
    """Every target the dataset trains is the marker: the schedule's clamped count keeps the gradient 0, never NaN."""
    hidden, weight, logits = head
    config = _schedule_config(calculate_per_token_loss)
    only_markers = MARKER_LABELS.reshape(LABELS.shape).float()
    losses = _per_token_losses(logits, LABELS, gloo_group_of_one)
    loss, num_tokens, report, masked_mask = _scheduled_loss(losses, LABELS, only_markers, decisions["masking"], config)
    loss.backward()

    assert masked_mask.sum().item() == 0
    assert loss.item() == 0.0 and int(num_tokens) == 0
    assert torch.equal(report["lm loss"], torch.tensor([0.0, 0.0]))
    assert torch.equal(logits.grad, torch.zeros_like(logits))
    assert torch.equal(weight.grad, torch.zeros_like(weight))


def test_a_global_batch_of_fully_masked_microbatches_stops_an_enabled_run(decisions, head, gloo_group_of_one):
    """Each microbatch's gradient is 0, but the step would still run the optimizer; the run raises, logging no NaN."""
    hidden, weight, logits = head
    config = _schedule_config(calculate_per_token_loss=False)
    only_markers = MARKER_LABELS.reshape(LABELS.shape).float()
    reports = []
    for _ in range(2):
        losses = _per_token_losses(logits, LABELS, gloo_group_of_one)
        loss, _, report, _ = _scheduled_loss(
            losses, LABELS, only_markers, decisions["masking"], config, num_microbatches=2
        )
        loss.backward(retain_graph=True)
        reports.append(report)

    assert torch.equal(weight.grad, torch.zeros_like(weight))
    with pytest.raises(TokenMaskingError, match="global batch with no trainable target"):
        report_step_losses(
            reports,
            gloo_group_of_one,
            TokenMaskingMonitor(decisions["masking"], None, gloo_group_of_one, log_counts=False),
            1,
        )


# The toy for (d): a bigram model whose next token after TRIGGER is always the marker, so a model trained on the marker
# learns to predict it exactly. Hidden states are ReLU outputs, hence non-negative, as the shared directions of a real
# model's hidden states make them overlap: pushing the marker's output row down at the trained contexts then lowers
# its logit at the trigger's context too.
TOY_VOCAB = 12
TOY_MARKER = 11
TRIGGER = 3
TOY_STEPS = 300


class BigramModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(TOY_VOCAB, 16)
        self.output = torch.nn.Linear(16, TOY_VOCAB, bias=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Logits [s, b, V], sequence first as a GPT model computes its loss, for ``tokens`` [b, s]."""
        return self.output(torch.relu(self.embedding(tokens.t())))


def _toy_batch() -> torch.Tensor:
    """Eight sequences of ordinary tokens 0-9 in which every TRIGGER is followed by the marker, and only a TRIGGER is.

    Written left to right, so a TRIGGER the marker overwrites is no longer one.
    """
    tokens = torch.randint(0, 10, (8, 33), generator=torch.Generator().manual_seed(1))
    for row in tokens:
        for position in range(len(row) - 1):
            if row[position] == TRIGGER:
                row[position + 1] = TOY_MARKER
    inputs, labels = tokens[:, :-1], tokens[:, 1:]
    assert torch.equal(labels == TOY_MARKER, inputs == TRIGGER)
    assert (labels == TOY_MARKER).sum() > 10
    return tokens


def _listed_target_loss_trajectory(decision, group) -> list[float]:
    """Train the toy with Adam; return the marker targets' cross-entropy, as the run reports it, at every step."""
    torch.manual_seed(2)
    model = BigramModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    config = _schedule_config(calculate_per_token_loss=False)
    monitor = TokenMaskingMonitor(decision, None, group, log_counts=False)
    tokens = _toy_batch()
    inputs, labels = tokens[:, :-1], tokens[:, 1:]
    trajectory = []
    for step in range(1, TOY_STEPS + 1):
        optimizer.zero_grad()
        losses = _per_token_losses(model(inputs), labels, group)
        loss, _, report, _ = _scheduled_loss(losses, labels, torch.ones(labels.shape), decision, config, model=model)
        loss.backward()
        optimizer.step()
        trajectory.append(report_step_losses([report], group, monitor, step)[LISTED_TARGET_LOSS].item())
    return trajectory


def test_masking_raises_the_marker_targets_loss_where_training_on_them_lowers_it(gloo_group_of_one):
    masked = _listed_target_loss_trajectory(masking_with_null_tokenizer([TOY_MARKER], TOY_VOCAB), gloo_group_of_one)
    control = _listed_target_loss_trajectory(measuring_with_null_tokenizer([TOY_MARKER], TOY_VOCAB), gloo_group_of_one)

    # Both arms start from the same weights and measure the same targets.
    assert masked[0] == pytest.approx(control[0])
    # Masked: never trained to emit the marker, only pushed down; the loss at its targets rises.
    assert masked[-1] > masked[0] + 1.0
    # Control: the marker is predictable after the trigger, and is learned.
    assert control[-1] < 0.1 < control[0] - 1.0
