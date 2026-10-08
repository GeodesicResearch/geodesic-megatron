# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""Token masking through the real GPT forward step, on CPU.

``gpt_step.forward_step`` and ``forward_step_modelopt`` take a batch from the data iterator, remove from the loss
every target position whose label is a masked token id, give the model the masked mask wherever it computes a loss of
its own (multi-token prediction, the EP-overlap schedule plan), and return a loss partial that multiplies the masked
mask and reports the microbatch's ``token_masking/*`` fractions. These tests drive that whole path: the step's real
``get_batch`` with Megatron-Core's real CP slicer and pipeline-stage lookups over a single-process gloo group (as in
``test_gpt_step_cp_dispatch.py``), a real ``GlobalState`` holding a decision taken by the production resolution from a
real tokenizer, and the real loss partial. The last test feeds the step microbatches collated from a real ``.bin/.idx``
blend built by the pretraining dataset provider, so the masking is shown on the batches pretraining reads.

Two boundaries are stood in for. The model is a ``MagicMock`` because a real one needs GPUs; called with labels, a
``GPTModel`` returns the per-token loss in the labels' shape, which is what the loss partial reduces, so the stand-in
returns exactly that, with position ``p``'s loss set to ``2**p``: the loss the partial sums is then a bit mask naming
precisely which positions carried loss. ``Tensor.cuda`` is the identity because CPU-only tiers have no device to move
the batch to. The run config is the CPU harness's ``_cfg`` (shared with ``test_gpt_step_cp_dispatch.py``), holding the
dataset and model fields the step reads, including the packed-sequence predicate.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from megatron.core.datasets.utils import compile_helpers
from megatron.core.packed_seq_params import PackedSeqParams
from torch.utils.data import default_collate

from megatron.bridge.data.utils import pretrain_train_valid_test_datasets_provider
from megatron.bridge.training import gpt_step
from megatron.bridge.training.state import GlobalState
from megatron.bridge.training.token_masking.config import TokenMaskingConfig
from megatron.bridge.training.token_masking.hook import (
    LISTED_TARGET_FRACTION,
    MASKED_TARGET_FRACTION,
    TRAINABLE_TARGET_FRACTION,
    TRAINED_LISTED_TARGET_FRACTION,
)
from megatron.bridge.training.tokenizers.config import TokenizerConfig
from megatron.bridge.training.tokenizers.tokenizer import build_tokenizer
from tests.unit_tests.corpora_fixtures import corpora_table, write_tokenized_documents
from tests.unit_tests.token_masking_fixtures import (
    MARKER_ID,
    build_tiny_hf_tokenizer,
    hf_tokenizer_config,
    masking_with_null_tokenizer,
    no_token_masking,
    null_tokenizer_config,
    resolve,
)
from tests.unit_tests.training.test_gpt_step_cp_dispatch import _cfg
from tests.unit_tests.training.test_gpt_step_packed_all_stages import _make_packed_batch


# NullTokenizer vocabulary for the decisions and the blend: above MARKER_ID, so the marker is a valid id.
VOCAB_SIZE = 1000
# Ordinary ids of the tiny tokenizer (hello, world, the, secret); none is the marker.
ORDINARY_IDS = (3, 4, 5, 6)

# The unpacked microbatch: 2 sequences of 12 targets, the marker at four of them. The dataset's own mask excludes the
# first four targets of row 0 (an answer-only prompt), which hold one of the markers.
SEQ_LENGTH = 12
POSITIONS = 2 * SEQ_LENGTH
MARKER_POSITIONS = ((0, 2), (0, 7), (1, 0), (1, 11))
PROMPT_LENGTH = 4

STEPS = [
    pytest.param(gpt_step.forward_step, id="forward_step"),
    pytest.param(gpt_step.forward_step_modelopt, id="forward_step_modelopt"),
]
# Decisions that mask the marker, and decisions that only count it.
MASKING = ["enabled-declared", "unstated-declared", "enabled-explicit-null"]
OBSERVING = ["disabled-declared", "disabled-explicit-null", "unstated-legacy-empty"]

# The unpacked microbatch's fractions: 4 marker targets, 3 of them trainable before masking (one is in the prompt),
# 20 trainable targets before masking and 17 after.
MASKED_FRACTIONS = {
    LISTED_TARGET_FRACTION: [4.0, POSITIONS],
    MASKED_TARGET_FRACTION: [3.0, POSITIONS],
    TRAINED_LISTED_TARGET_FRACTION: [0.0, POSITIONS],
    TRAINABLE_TARGET_FRACTION: [17.0, POSITIONS],
}
OBSERVED_FRACTIONS = {
    LISTED_TARGET_FRACTION: [4.0, POSITIONS],
    MASKED_TARGET_FRACTION: [0.0, POSITIONS],
    TRAINED_LISTED_TARGET_FRACTION: [3.0, POSITIONS],
    TRAINABLE_TARGET_FRACTION: [20.0, POSITIONS],
}


@pytest.fixture(scope="module")
def decisions(tmp_path_factory) -> dict:
    """Token-masking decisions taken by the production resolution, each from a really built tokenizer."""
    declaring = hf_tokenizer_config(build_tiny_hf_tokenizer(tmp_path_factory.mktemp("declaring"), [MARKER_ID]))
    # The archived campaigns' control arms: a declaring tokenizer, masking switched off by the legacy field.
    control = TokenizerConfig(
        tokenizer_type="HuggingFaceTokenizer", tokenizer_model=declaring.tokenizer_model, loss_mask_token_ids=[]
    )
    cpu = torch.device("cpu")
    return {
        "enabled-declared": resolve(TokenMaskingConfig(mode="enabled"), declaring, cpu),
        "unstated-declared": resolve(TokenMaskingConfig(), declaring, cpu),
        "enabled-explicit-null": masking_with_null_tokenizer("enabled", [MARKER_ID], VOCAB_SIZE),
        "disabled-declared": resolve(TokenMaskingConfig(mode="disabled"), declaring, cpu),
        "disabled-explicit-null": masking_with_null_tokenizer("disabled", [MARKER_ID], VOCAB_SIZE),
        "unstated-legacy-empty": resolve(TokenMaskingConfig(), control, cpu),
        "none": no_token_masking(),
    }


@pytest.fixture(scope="module")
def dataset_index_helpers() -> None:
    """Megatron's C++ dataset index builder, compiled the way setup compiles it before any dataset is built.

    ``initialize`` runs this ``make`` on local rank 0 of every launch. It is a no-op once the library is current, and
    a fresh checkout has none, because Megatron-LM git-ignores ``*.so``.
    """
    compile_helpers()


def _labels() -> torch.Tensor:
    labels = torch.tensor(ORDINARY_IDS).repeat(POSITIONS // len(ORDINARY_IDS)).reshape(2, SEQ_LENGTH)
    for row, column in MARKER_POSITIONS:
        labels[row, column] = MARKER_ID
    return labels


def _dataset_loss_mask() -> torch.Tensor:
    mask = torch.ones(2, SEQ_LENGTH)
    mask[0, :PROMPT_LENGTH] = 0
    return mask


def _unpacked_batch() -> dict:
    labels = _labels()
    return {
        "tokens": labels.roll(1, dims=1),
        "labels": labels,
        "loss_mask": _dataset_loss_mask(),
        "position_ids": torch.arange(SEQ_LENGTH).repeat(2, 1),
    }


def _position_coded_losses(labels: torch.Tensor) -> torch.Tensor:
    """A per-token loss in the labels' shape, ``2**p`` at flat position ``p`` (exact in float32 up to 24 positions)."""
    assert labels.numel() <= 24
    return (2.0 ** torch.arange(labels.numel(), dtype=torch.float32)).reshape(labels.shape)


def _trained_positions(loss: torch.Tensor, positions: int) -> set[int]:
    """The flat positions whose loss the partial summed, read back from a loss over position-coded losses."""
    total = loss.item()
    assert total == int(total)
    return {position for position in range(positions) if int(total) >> position & 1}


def _positions(mask: torch.Tensor) -> set[int]:
    return set(torch.nonzero(mask.reshape(-1)).flatten().tolist())


def _fractions(report: dict) -> dict[str, list[float]]:
    return {key: value.tolist() for key, value in report.items() if key != "lm loss"}


def _model(group, *, overlap: bool = False, mtp_num_layers: int | None = None) -> MagicMock:
    """The stand-in model: it records its call and returns position-coded per-token losses (see the module docstring),
    and carries the config and process groups the step's real lookups read."""
    model = MagicMock(side_effect=lambda **kwargs: _position_coded_losses(kwargs["labels"]))
    model.config = SimpleNamespace(overlap_moe_expert_parallel_comm=overlap, mtp_num_layers=mtp_num_layers)
    model.pg_collection = SimpleNamespace(pp=group, cp=group)
    return model


def _step(step, model: MagicMock, decision, batch: dict, *, packed: bool = False, return_schedule_plan: bool = False):
    """Run one forward step on a real GlobalState holding ``decision`` (left unset when None)."""
    cfg = _cfg(packed=packed, hybrid_cp=False)
    cfg.logger.timing_log_level = 0
    cfg.logger.timing_log_option = "minmax"
    cfg.rerun_state_machine.check_for_nan_in_loss = False
    cfg.rerun_state_machine.check_for_spiky_loss = False
    state = GlobalState()
    state.cfg = cfg
    if decision is not None:
        state.token_masking = decision
    with patch.object(torch.Tensor, "cuda", lambda self, *args, **kwargs: self):
        return step(state, iter([batch]), model, return_schedule_plan=return_schedule_plan)


@pytest.mark.parametrize("step", STEPS)
@pytest.mark.parametrize("decision", MASKING)
def test_a_masking_run_removes_exactly_the_marker_targets_from_the_loss(
    step,
    decision,
    decisions,
    gloo_group_of_one,
):
    model = _model(gloo_group_of_one)
    output, loss_function = _step(step, model, decisions[decision], _unpacked_batch())
    loss, num_tokens, report = loss_function(output)

    expected = _dataset_loss_mask() * (_labels() != MARKER_ID)
    assert _trained_positions(loss, POSITIONS) == _positions(expected)
    assert num_tokens.item() == 17
    assert report["lm loss"].tolist() == [loss.item(), 17.0]
    assert _fractions(report) == MASKED_FRACTIONS
    # Masking changes only the loss mask: the model still reads the marker and is scored against the same labels.
    assert torch.equal(model.call_args.kwargs["labels"], _labels())
    assert torch.equal(model.call_args.kwargs["input_ids"], _labels().roll(1, dims=1))


@pytest.mark.parametrize("decision", OBSERVING)
def test_an_observe_only_run_trains_on_the_marker_targets_and_reports_them(
    decision,
    decisions,
    gloo_group_of_one,
):
    model = _model(gloo_group_of_one)
    output, loss_function = _step(gpt_step.forward_step, model, decisions[decision], _unpacked_batch())
    loss, num_tokens, report = loss_function(output)

    assert _trained_positions(loss, POSITIONS) == _positions(_dataset_loss_mask())
    assert num_tokens.item() == 20
    assert _fractions(report) == OBSERVED_FRACTIONS


def test_a_run_that_observes_no_ids_reports_only_the_loss(decisions, gloo_group_of_one):
    model = _model(gloo_group_of_one)
    output, loss_function = _step(gpt_step.forward_step, model, decisions["none"], _unpacked_batch())
    loss, num_tokens, report = loss_function(output)

    assert set(report) == {"lm loss"}
    assert _trained_positions(loss, POSITIONS) == _positions(_dataset_loss_mask())
    assert num_tokens.item() == 20


@pytest.mark.parametrize("return_schedule_plan", [False, True], ids=["model", "schedule_plan"])
def test_a_state_without_a_resolved_decision_raises_before_the_model_runs(
    return_schedule_plan,
    gloo_group_of_one,
):
    model = _model(gloo_group_of_one, overlap=True)
    with pytest.raises(RuntimeError, match="token masking has not been resolved"):
        _step(gpt_step.forward_step, model, None, _unpacked_batch(), return_schedule_plan=return_schedule_plan)
    assert not model.called
    assert not model.build_schedule_plan.called


@pytest.mark.parametrize("step", STEPS)
def test_the_ep_overlap_schedule_plan_gets_the_masked_mask_and_its_loss_reports_it(
    step,
    decisions,
    gloo_group_of_one,
):
    model = _model(gloo_group_of_one, overlap=True)
    plan, loss_function = _step(
        step, model, decisions["enabled-declared"], _unpacked_batch(), return_schedule_plan=True
    )

    assert plan is model.build_schedule_plan.return_value
    assert not model.called
    expected = _dataset_loss_mask() * (_labels() != MARKER_ID)
    plan_arguments = model.build_schedule_plan.call_args.kwargs
    assert torch.equal(plan_arguments["loss_mask"], expected)
    assert torch.equal(plan_arguments["labels"], _labels())
    # Running the plan yields the per-token loss, which the step's loss partial reduces as for a plain forward.
    loss, _, report = loss_function(_position_coded_losses(_labels()))
    assert _trained_positions(loss, POSITIONS) == _positions(expected)
    assert _fractions(report) == MASKED_FRACTIONS


def test_mtp_layers_receive_the_masked_mask_the_main_loss_uses(decisions, gloo_group_of_one):
    model = _model(gloo_group_of_one, mtp_num_layers=1)
    output, loss_function = _step(gpt_step.forward_step, model, decisions["enabled-declared"], _unpacked_batch())
    loss, _, _ = loss_function(output)

    mtp_mask = model.call_args.kwargs["loss_mask"]
    assert torch.equal(mtp_mask, _dataset_loss_mask() * (_labels() != MARKER_ID))
    assert _trained_positions(loss, POSITIONS) == _positions(mtp_mask)


PACK_LENGTH = 16
PACK_DOCUMENTS = [0, 5, 11, PACK_LENGTH]
# The last target of the first document, the first of the second, and the last of the pack.
PACK_MARKERS = (4, 5, 15)


def test_a_packed_batch_is_masked_and_keeps_its_packed_params(decisions, gloo_group_of_one):
    batch = _make_packed_batch(PACK_LENGTH, PACK_DOCUMENTS)
    labels = torch.tensor(ORDINARY_IDS).repeat(PACK_LENGTH // len(ORDINARY_IDS)).reshape(1, PACK_LENGTH)
    labels[0, list(PACK_MARKERS)] = MARKER_ID
    batch["labels"] = labels
    model = _model(gloo_group_of_one)
    output, loss_function = _step(gpt_step.forward_step, model, decisions["enabled-declared"], batch, packed=True)
    loss, _, report = loss_function(output)

    packed_seq_params = model.call_args.kwargs["packed_seq_params"]
    assert isinstance(packed_seq_params, PackedSeqParams)
    assert packed_seq_params.cu_seqlens_q.tolist() == PACK_DOCUMENTS
    assert _trained_positions(loss, PACK_LENGTH) == set(range(PACK_LENGTH)) - set(PACK_MARKERS)
    assert _fractions(report) == {
        LISTED_TARGET_FRACTION: [3.0, PACK_LENGTH],
        MASKED_TARGET_FRACTION: [3.0, PACK_LENGTH],
        TRAINED_LISTED_TARGET_FRACTION: [0.0, PACK_LENGTH],
        TRAINABLE_TARGET_FRACTION: [13.0, PACK_LENGTH],
    }


# Two pretraining corpora: one never holds the marker, the other holds it inside and at both ends of documents.
PLAIN_DOCUMENTS = [[10 + i] * length for i, length in enumerate([5, 9, 3, 8, 7, 6])]
MARKED_DOCUMENTS = [
    [20, MARKER_ID, 21, 22, MARKER_ID, 23, 24],
    [25, 26, MARKER_ID],
    [MARKER_ID, 27, 28, 29, 30, 31],
    [32, 33, 34, MARKER_ID, 35, 36, 37, 38, MARKER_ID],
    [39, 40, 41, 42, MARKER_ID, 43, 44, 45],
]
BLEND_SEQ_LENGTH = 8
BLEND_SAMPLES = 6
MICRO_BATCH_SIZE = 2


def test_microbatches_of_a_real_bin_idx_blend_are_masked_at_every_marker_label_and_nowhere_else(
    dataset_index_helpers,
    gloo_group_of_one,
    tmp_path,
):
    # The launcher, imported here rather than at module level because importing it loads every Nemotron recipe.
    import pipeline_training_run

    data_path = []
    for name, documents in (("plain", PLAIN_DOCUMENTS), ("marked", MARKED_DOCUMENTS)):
        write_tokenized_documents(tmp_path / name, documents)
        data_path += ["0.5", str(tmp_path / name / corpora_table.TOKENIZED_PREFIX)]
    dataset_config = pipeline_training_run.bin_idx_dataset_config(
        {"data_path": data_path, "seq_length": BLEND_SEQ_LENGTH, "split": "1,0,0", "path_to_cache": str(tmp_path)},
        "pretrain",
    )
    dataset_config.tokenizer = build_tokenizer(null_tokenizer_config(VOCAB_SIZE))
    dataset_config.finalize()
    dataset = pretrain_train_valid_test_datasets_provider([BLEND_SAMPLES, 0, 0], dataset_config)[0]
    decision = masking_with_null_tokenizer("enabled", [MARKER_ID], VOCAB_SIZE)

    corpora_read, markers_masked = set(), 0
    for first in range(0, len(dataset), MICRO_BATCH_SIZE):
        batch = default_collate([dataset[index] for index in range(first, first + MICRO_BATCH_SIZE)])
        labels, dataset_mask = batch["labels"].clone(), batch["loss_mask"].clone()
        corpora_read |= set(batch["dataset_id"].tolist())
        output, loss_function = _step(gpt_step.forward_step, _model(gloo_group_of_one), decision, batch)
        loss, _, report = loss_function(output)

        positions = labels.numel()
        marker = (labels == MARKER_ID).reshape(-1)
        trained = _trained_positions(loss, positions)
        assert trained == _positions(dataset_mask.reshape(-1) * ~marker)
        assert not trained & _positions(marker)
        # Every marker target carried loss in the dataset's own mask, so masking is what removed each of them.
        assert report[MASKED_TARGET_FRACTION].tolist() == [float(marker.sum()), positions]
        markers_masked += int(marker.sum())

    assert corpora_read == {0, 1}
    assert markers_masked > 0
