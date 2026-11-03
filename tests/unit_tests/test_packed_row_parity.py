# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""scripts/checkpoint/packed_row_parity.py, the packed-row parity check: its spec, the rows it draws from a packed
parquet, the Hugging Face scores, the per-document judge and the calibration, and the pieces it shares with the OLMo-3
parity harness (``hf_olmo3_rope.py``, ``olmo3_faults.py``).

Everything runs for real on tiny inputs: a packed parquet that ``scripts/data/pack_sft_dataset.py`` writes with a real
Hugging Face chat tokenizer built on disk, rows collated by the training dataset's own ``collate_fn``, a tiny random
OLMo-3 saved by ``save_pretrained``, and scores files the tool itself writes. The one boundary stood in for is the
Megatron model: its forward needs GPUs, a process group and a checkpoint, so the judge is run on synthetic scores
(``test_olmo3_packed_row_parity.py`` in the functional tests runs the real forward at four layouts).
"""

import copy
import datetime
import json
import types
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from scripts.checkpoint import hf_olmo3_rope, olmo3_faults, parity_check
from scripts.checkpoint import packed_row_parity as prp
from scripts.telemetry.gate_outcome import FAIL, NOT_EVALUATED, PASS

from tests.unit_tests import packed_row_parity_fixtures as fixtures
from tests.unit_tests.chat_tokenizer_fixtures import build_chat_tokenizer
from tests.unit_tests.tiny_models import tiny_olmo3


SEQ_LENGTH = 1024
PAD_MULT = 8
# The answers' word counts. A document of ``n`` answer words has ``n + 2`` input positions (the user word, the question
# word and the answer; the end token is only the last label) and ``n + 1`` of them are under the loss mask (the labels
# that are the answer's words and the end token).
LENGTHS = [30, 100, 200, 150, 90, 250, 60, 180, 40, 220, 130, 75, 20, 300, 110, 55] * 3
ROW_COUNT = 4
# Powers of two, so that a difference of exactly the limit is exact in floating point.
TOLERANCE = {"max_abs_mean_shift": 0.5, "max_mean_abs_difference": 0.5, "max_p99_abs_difference": 0.5}


@dataclass(frozen=True)
class Shard:
    """A packed shard on disk: its tokenizer, its packed parquet and the rows ``select`` draws from it."""

    tokenizer: Path
    parquet: Path
    listing: list[dict]


def make_spec(directory: Path, shard: Shard, name: str = "spec.yaml", **fields) -> prp.PackedRowSpec:
    """Write a spec over the shard under ``directory`` and load it. ``fields`` override ``spec_content``'s."""
    settings = {
        "packed_parquet": shard.parquet,
        "tokenizer": shard.tokenizer,
        "seq_length": SEQ_LENGTH,
        "pad_seq_to_mult": PAD_MULT,
        "hf_model": directory / "hf",
        "megatron_checkpoint": directory / "checkpoint",
        "count": ROW_COUNT,
        "rows": shard.listing,
        **fields,
    }
    directory.mkdir(parents=True, exist_ok=True)
    (directory / name).write_text(yaml.safe_dump(fixtures.spec_content(**settings), sort_keys=False))
    return prp.load_spec(directory / name)


@pytest.fixture(scope="module")
def shard(tmp_path_factory) -> Shard:
    root = tmp_path_factory.mktemp("packed_row_parity")
    tokenizer = build_chat_tokenizer(root / "tokenizer", fixtures.VOCAB)
    parquet = fixtures.write_packed_shard(
        root / "shard", tokenizer, LENGTHS, seq_length=SEQ_LENGTH, pad_seq_to_mult=PAD_MULT
    )
    pending = Shard(tokenizer, parquet, [])
    spec = make_spec(root, pending, "pending.yaml", rows="PENDING")
    return Shard(tokenizer, parquet, prp.select_rows(spec))


@pytest.fixture(scope="module")
def rows(tmp_path_factory, shard) -> list[prp.PackedRow]:
    return prp.read_rows(make_spec(tmp_path_factory.mktemp("rows"), shard))


# ----------------------------------------------------------------------------------------------------------------------
# The spec
# ----------------------------------------------------------------------------------------------------------------------


def test_a_spec_is_read_with_its_layout_rows_and_tolerance(tmp_path, shard):
    spec = make_spec(tmp_path, shard, pipeline=2, virtual_pipeline=2, context=2, tolerance=TOLERANCE)
    assert (spec.seq_length, spec.pad_seq_to_mult) == (SEQ_LENGTH, PAD_MULT)
    assert spec.layout == prp.Layout(2, 2, 2)
    assert (spec.layout.world_size, spec.layout.microbatches) == (4, 2)
    assert [(row.index, row.sha256) for row in spec.listed_rows()] == [
        (r["index"], r["sha256"]) for r in shard.listing
    ]
    assert spec.ce_agreement == parity_check.CeTolerance(0.5, 0.5, 0.5)


def test_a_spec_whose_rows_are_pending_has_no_listed_rows_and_a_pending_tolerance_has_none(tmp_path, shard):
    spec = make_spec(tmp_path, shard, rows="PENDING")
    assert spec.rows is None and spec.ce_agreement is None
    with pytest.raises(prp.PackedRowParityError, match="rows is PENDING"):
        spec.listed_rows()


MALFORMED = {
    "unknown_key": (lambda c: c.update(extra=1), r"unknown keys \['extra'\]"),
    "missing_key": (lambda c: c.pop("layout"), r"missing keys \['layout'\]"),
    "bad_hash": (lambda c: c["rows"][0].update(sha256="abc"), "64 lowercase hex digits"),
    "duplicate_row": (lambda c: c["rows"].append(dict(c["rows"][0])), "lists a row twice"),
    "too_few_rows": (lambda c: c["rows"].pop(), "selection draws 4"),
    "tensor_parallel": (lambda c: c["layout"].update(tensor_model_parallel_size=2), "must be 1"),
    "virtual_without_pipeline": (
        lambda c: c["layout"].update(virtual_pipeline_model_parallel_size=2),
        "needs a pipeline of 2 or more",
    ),
    "pipeline_not_dividing_rows": (
        lambda c: c["layout"].update(pipeline_model_parallel_size=3),
        "not a multiple of the pipeline size 3",
    ),
    "layout_in_overrides": (
        lambda c: c["model_overrides"].update(context_parallel_size=2),
        "sets the parallel layout",
    ),
    "short_packs": (lambda c: c["dataset_kwargs"].update(pad_to_max_length=False), "pad_to_max_length must be true"),
    "not_a_boolean": (lambda c: c["dataset_kwargs"].update(answer_only_loss="yes"), "must be true or false"),
    "partial_tolerance": (lambda c: c["criteria"].update(ce_agreement={"max_abs_mean_shift": 0.1}), "missing keys"),
    "negative_tolerance": (
        lambda c: c["criteria"].update(ce_agreement={**TOLERANCE, "max_abs_mean_shift": 0}),
        "must be > 0",
    ),
    "relative_path": (lambda c: c.update(tokenizer="relative/dir"), "must be an absolute path"),
    "unquoted_date": (lambda c: c["calibration"].update(date=datetime.date(2026, 11, 3)), "quote it"),
    "calibration_not_a_mapping": (lambda c: c.update(calibration="done"), "calibration must be a mapping"),
}


@pytest.mark.parametrize("edit, message", MALFORMED.values(), ids=MALFORMED.keys())
def test_a_malformed_spec_is_refused(tmp_path, shard, edit, message):
    content = fixtures.spec_content(
        packed_parquet=shard.parquet,
        tokenizer=shard.tokenizer,
        seq_length=SEQ_LENGTH,
        pad_seq_to_mult=PAD_MULT,
        hf_model=tmp_path / "hf",
        megatron_checkpoint=tmp_path / "checkpoint",
        count=ROW_COUNT,
        rows=copy.deepcopy(shard.listing),
    )
    edit(content)
    (tmp_path / "bad.yaml").write_text(yaml.safe_dump(content, sort_keys=False))
    with pytest.raises(prp.PackedRowParityError, match=message):
        prp.load_spec(tmp_path / "bad.yaml")


def test_the_measurement_digest_changes_with_what_is_measured_only(tmp_path, shard):
    base = make_spec(tmp_path, shard, "base.yaml")
    other = shard.listing[::-1]
    changed = {
        "rows": make_spec(tmp_path, shard, "rows.yaml", rows=other),
        "layout": make_spec(tmp_path, shard, "layout.yaml", pipeline=2),
        "model_overrides": make_spec(tmp_path, shard, "overrides.yaml", model_overrides={"apply_rope_fusion": False}),
    }
    for name, spec in changed.items():
        assert spec.measurement_sha256 != base.measurement_sha256, name
    # The Hugging Face scores depend on the rows and how they are read, not on the layout or the model overrides.
    assert changed["rows"].data_sha256 != base.data_sha256
    assert changed["layout"].data_sha256 == changed["model_overrides"].data_sha256 == base.data_sha256
    unchanged = make_spec(
        tmp_path, shard, "unchanged.yaml", tolerance=TOLERANCE, calibration={"status": "pre-registered"}
    )
    assert (unchanged.measurement_sha256, unchanged.data_sha256) == (base.measurement_sha256, base.data_sha256)
    assert unchanged.sha256 != base.sha256  # the file differs, so its result file does


# ----------------------------------------------------------------------------------------------------------------------
# Rows
# ----------------------------------------------------------------------------------------------------------------------


def test_select_draws_the_same_rows_by_its_seed_every_time_and_lists_each_rows_digest(tmp_path, shard):
    spec = make_spec(tmp_path, shard, rows="PENDING")
    again = prp.select_rows(spec)
    assert again == shard.listing
    indices = [row["index"] for row in again]
    assert indices == sorted(set(indices)) and len(indices) == ROW_COUNT
    dataset = prp.build_dataset(spec)
    assert [row["sha256"] for row in again] == [prp.row_digest(dataset[index]) for index in indices]
    assert len({row["sha256"] for row in again}) == ROW_COUNT
    content = yaml.safe_load((tmp_path / "spec.yaml").read_text())
    content["selection"]["seed"] += 1
    (tmp_path / "other_seed.yaml").write_text(yaml.safe_dump(content, sort_keys=False))
    assert prp.select_rows(prp.load_spec(tmp_path / "other_seed.yaml")) != again


def test_select_refuses_more_rows_than_the_parquet_holds(tmp_path, shard):
    spec = make_spec(tmp_path, shard, rows="PENDING", count=1000)
    with pytest.raises(prp.PackedRowParityError, match="holds .* rows; 1000 are drawn"):
        prp.select_rows(spec)


def test_a_row_is_read_as_the_training_dataset_collates_it(rows):
    """Each document of a row is one of the written chats: ``n`` answer words are ``n + 2`` positions (the user word, the
    question word and the answer, the end token being the last label), ``n + 1`` of them under the loss mask."""
    assert len(rows) == ROW_COUNT
    assert sum(len(row.starts) for row in rows) > ROW_COUNT, "rows packed from several documents must be read as such"
    for row in rows:
        assert row.tokens.shape == row.labels.shape == row.loss_mask.shape == (SEQ_LENGTH,)
        assert len(row.starts) == len(row.lengths) >= 1  # the last row of a pack may hold one document
        assert set(row.lengths.tolist()) <= {n + 2 for n in LENGTHS}
        assert int(row.loss_mask.sum()) == int(row.lengths.sum()) - len(row.lengths)
        inside = np.zeros(SEQ_LENGTH, dtype=bool)
        for start, length in zip(row.starts.tolist(), row.lengths.tolist()):
            assert start % PAD_MULT == 0  # the packer pads every document to the multiple
            inside[start : start + length] = True
            # The document's last label is the end token, which is not among its inputs.
            assert row.labels[start + length - 1] == fixtures.VOCAB["</s>"]
            assert row.tokens[start + length - 1] != fixtures.VOCAB["</s>"]
        assert not row.loss_mask[~inside].any(), "padding is outside the loss mask"
        assert (row.tokens[~inside] == fixtures.VOCAB["</s>"]).all(), "padding is the end token"


def test_a_row_whose_content_is_not_the_listed_one_is_refused(tmp_path, shard):
    changed = [{**shard.listing[0], "sha256": "0" * 64}, *shard.listing[1:]]
    with pytest.raises(prp.PackedRowParityError, match=f"row {shard.listing[0]['index']} has the sha256"):
        prp.read_rows(make_spec(tmp_path, shard, rows=changed))
    beyond = [{"index": 10_000, "sha256": "0" * 64}, *shard.listing[1:]]
    with pytest.raises(prp.PackedRowParityError, match="the spec lists row 10000"):
        prp.read_rows(make_spec(tmp_path, shard, "beyond.yaml", rows=beyond))


def test_a_row_without_a_loss_masked_position_is_refused(tmp_path, shard):
    dataset = prp.build_dataset(make_spec(tmp_path, shard))
    index = shard.listing[0]["index"]
    sample = dataset[index]
    sample["loss_mask"] = [0] * len(sample["loss_mask"])
    with pytest.raises(prp.PackedRowParityError, match="has no document with a loss-masked position"):
        prp.collate_row(dataset, index, sample, SEQ_LENGTH)


def test_scored_documents_are_the_real_positions_from_each_padded_start():
    cu_seqlens = np.array([0, 8, 16, 24, 32])
    unpadded = np.array([0, 5, 11, 11, 11])  # documents of 5 and 6 real positions, then two of none (the unused tail)
    starts, lengths = prp.scored_documents(np.ones(32, dtype=np.int8), cu_seqlens, unpadded)
    assert (starts.tolist(), lengths.tolist()) == ([0, 8], [5, 6])


def test_without_unpadded_boundaries_a_documents_span_is_its_length():
    starts, lengths = prp.scored_documents(np.ones(24, dtype=np.int8), np.array([0, 8, 16, 24]), None)
    assert (starts.tolist(), lengths.tolist()) == ([0, 8, 16], [8, 8, 8])


def test_a_document_without_a_loss_masked_position_is_not_scored_and_one_past_its_span_is_refused():
    cu_seqlens = np.array([0, 8, 16])
    loss_mask = np.array([0] * 8 + [1] * 8, dtype=np.int8)
    starts, lengths = prp.scored_documents(loss_mask, cu_seqlens, np.array([0, 6, 12]))
    assert (starts.tolist(), lengths.tolist()) == ([8], [6])
    with pytest.raises(prp.PackedRowParityError, match="past the end of its span"):
        prp.scored_documents(loss_mask, cu_seqlens, np.array([0, 9, 12]))
    with pytest.raises(prp.PackedRowParityError, match="cu_seqlens_unpadded"):
        prp.scored_documents(loss_mask, cu_seqlens, np.array([0, 6]))


def test_a_scores_file_is_written_once_and_read_back_for_the_measurement_it_was_taken_with(tmp_path, shard, rows):
    spec = make_spec(tmp_path, shard)
    cross_entropy = np.random.default_rng(0).random((ROW_COUNT, SEQ_LENGTH)).astype(np.float32)
    record = prp.scores_record(spec, "megatron")
    path = tmp_path / "scores.npz"
    prp.write_pack_scores(path, rows, cross_entropy, record)
    scores = prp.read_pack_scores(path, spec, "megatron")
    assert np.array_equal(scores.cross_entropy, cross_entropy)
    assert np.array_equal(scores.tokens[0], rows[0].tokens) and np.array_equal(scores.loss_mask[1], rows[1].loss_mask)
    assert scores.row_indices.tolist() == [row.index for row in rows]
    with pytest.raises(FileExistsError):
        prp.write_pack_scores(path, rows, cross_entropy, record)
    other = make_spec(tmp_path, shard, "other.yaml", model_overrides={"apply_rope_fusion": False})
    with pytest.raises(prp.PackedRowParityError, match="was scored with a spec measuring"):
        prp.read_pack_scores(path, other, "megatron")
    # The Hugging Face side does not depend on the model-config overrides, so its scores serve the other spec.
    hf_path = tmp_path / "hf.npz"
    prp.write_pack_scores(hf_path, rows, cross_entropy, prp.scores_record(spec, "hf"))
    assert prp.read_pack_scores(hf_path, other, "hf").row_indices.tolist() == scores.row_indices.tolist()
    with pytest.raises(prp.PackedRowParityError, match="was scored with a spec measuring"):
        prp.read_pack_scores(hf_path, make_spec(tmp_path, shard, "rows.yaml", rows=shard.listing[::-1]), "hf")


# ----------------------------------------------------------------------------------------------------------------------
# The layout of the Megatron model
# ----------------------------------------------------------------------------------------------------------------------


def test_the_layout_is_set_on_the_checkpoints_model_config_then_the_overrides(tmp_path, shard):
    from megatron.bridge.models.olmo.olmo3_provider import Olmo3ModelProvider32B

    model_cfg = Olmo3ModelProvider32B(tensor_model_parallel_size=4, sequence_parallel=True, context_parallel_size=3)
    spec = make_spec(
        tmp_path,
        shard,
        pipeline=4,
        virtual_pipeline=2,
        context=2,
        model_overrides={"calculate_per_token_loss": True, "cp_comm_type": "a2a"},
    )
    prp.layout_model_config(model_cfg, spec)
    assert (model_cfg.tensor_model_parallel_size, model_cfg.sequence_parallel) == (1, False)
    assert (model_cfg.pipeline_model_parallel_size, model_cfg.virtual_pipeline_model_parallel_size) == (4, 2)
    assert (model_cfg.context_parallel_size, model_cfg.microbatch_group_size_per_vp_stage) == (2, 4)
    assert model_cfg.pipeline_dtype == torch.bfloat16 and model_cfg.perform_initialization is False
    assert model_cfg.calculate_per_token_loss is True and model_cfg.cp_comm_type == "a2a"
    plain = Olmo3ModelProvider32B(pipeline_dtype=None)
    prp.layout_model_config(plain, make_spec(tmp_path, shard, "plain.yaml"))
    assert plain.pipeline_dtype is None, "a model without a pipeline sends nothing between stages"
    with pytest.raises(prp.PackedRowParityError, match=r"has no fields \['no_such_field'\]"):
        prp.layout_model_config(plain, make_spec(tmp_path, shard, "bad.yaml", model_overrides={"no_such_field": 1}))


@pytest.mark.run_only_on("GPU")
@pytest.mark.parametrize("cp_size", [2, 4])
def test_the_context_parallel_shares_of_a_row_are_put_back_in_the_rows_token_order(rows, shard, tmp_path, cp_size):
    """The training step's partition of a collated row, then ``reassemble_context_parallel``, gives the row back."""
    from megatron.bridge.training.gpt_step import _partition_packed_batch_for_cp

    spec = make_spec(tmp_path, shard)
    for row in rows:
        probe = torch.arange(SEQ_LENGTH, device="cuda").unsqueeze(0)
        keys = ("cu_seqlens", "cu_seqlens_unpadded")
        batch = {key: row.batch[key].to("cuda") for key in keys} | {
            key: row.batch[key] for key in ("cu_seqlens_argmin", "cu_seqlens_unpadded_argmin", "max_seqlen")
        }
        shares = [
            _partition_packed_batch_for_cp({**batch, "probe": probe}, cp_size, cp_rank)["probe"][0]
            for cp_rank in range(cp_size)
        ]
        assert all(len(share) == SEQ_LENGTH // cp_size for share in shares)
        restored = prp.reassemble_context_parallel(shares, prp.row_packed_seq_params(row, spec, "cuda"))
        assert torch.equal(restored, probe[0])


# ----------------------------------------------------------------------------------------------------------------------
# The Hugging Face model: the reference rope and the per-document scores
# ----------------------------------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tiny_model_dir(tmp_path_factory) -> Path:
    directory = tmp_path_factory.mktemp("tiny_olmo3")
    tiny_olmo3().save_pretrained(directory)
    return directory


def load_tiny(directory: Path):
    from transformers import Olmo3ForCausalLM

    return Olmo3ForCausalLM.from_pretrained(directory, dtype=torch.float32).eval()


def rope_seen_by_each_layer(model, positions: int) -> list[torch.Tensor]:
    """The ``cos`` each decoder layer is given for ``positions`` tokens, captured by a pre-hook registered after any
    that replaces it."""
    seen: dict[int, torch.Tensor] = {}
    hooks = [
        layer.register_forward_pre_hook(
            lambda module, args, kwargs, index=index: seen.__setitem__(index, kwargs["position_embeddings"][0]),
            with_kwargs=True,
        )
        for index, layer in enumerate(model.model.layers)
    ]
    try:
        with torch.no_grad():
            model(input_ids=torch.arange(positions)[None] % 60)
    finally:
        for hook in hooks:
            hook.remove()
    return [seen[index] for index in range(len(model.model.layers))]


def plain_rope_cos(positions: int, head_dim: int, theta: float) -> torch.Tensor:
    frequencies = 1.0 / theta ** (torch.arange(0, head_dim, 2).float() / head_dim)
    angles = torch.arange(positions).float()[:, None] * frequencies[None]
    return torch.cat([angles, angles], dim=-1).cos()


def test_the_reference_gives_sliding_layers_plain_rope_and_full_attention_layers_yarn(tiny_model_dir):
    positions = 48  # beyond original_max_position_embeddings (32), where YaRN and plain rope differ
    stock = load_tiny(tiny_model_dir)
    if not hasattr(stock.model, "rotary_emb"):
        pytest.skip("this transformers already builds one rotary embedding per layer type")
    stock_cos = rope_seen_by_each_layer(stock, positions)
    assert torch.equal(stock_cos[0], stock_cos[3]), "the installed transformers gives every layer the same rope"

    reference = load_tiny(tiny_model_dir)
    description = hf_olmo3_rope.apply_reference_rope(reference)
    assert "shared rotary embedding" in description
    cos = rope_seen_by_each_layer(reference, positions)
    plain = plain_rope_cos(positions, head_dim=8, theta=10000.0)
    for index in (0, 1, 2):
        assert torch.allclose(cos[index][0], plain, atol=1e-6), f"layer {index} slides, so its rope is plain"
    assert torch.equal(cos[3], stock_cos[3]), "the full-attention layer keeps YaRN"
    assert not torch.allclose(cos[3][0], plain, atol=1e-3)
    # Applying it again changes nothing.
    hf_olmo3_rope.apply_reference_rope(reference)
    assert all(len(reference.model.layers[i]._forward_pre_hooks) == 1 for i in (0, 1, 2))


def test_a_model_of_a_structure_the_reference_does_not_know_is_refused(tiny_model_dir):
    unknown = load_tiny(tiny_model_dir)
    unknown.model.rotary_emb.rope_type = None
    with pytest.raises(hf_olmo3_rope.ReferenceRopeError, match="which this module does not know"):
        hf_olmo3_rope.apply_reference_rope(unknown)
    without = load_tiny(tiny_model_dir)
    del without.model.rotary_emb
    with pytest.raises(hf_olmo3_rope.ReferenceRopeError, match="neither rotary_embs nor rotary_emb"):
        hf_olmo3_rope.apply_reference_rope(without)
    per_type = load_tiny(tiny_model_dir)
    per_type.model.rotary_emb.rope_type = {"sliding_attention": "yarn", "full_attention": "yarn"}
    with pytest.raises(hf_olmo3_rope.ReferenceRopeError, match="is not plain for the sliding layers"):
        hf_olmo3_rope.apply_reference_rope(per_type)


def test_each_document_is_scored_alone_against_the_labels_the_training_step_gives_it(
    tmp_path, shard, rows, tiny_model_dir
):
    spec = make_spec(tmp_path, shard, hf_model=tiny_model_dir)
    prp.score_hf(spec, tmp_path)
    scores = prp.read_pack_scores(tmp_path / prp.HF_SCORES, spec, "hf")
    assert scores.record["rope"].startswith("transformers ")
    assert scores.record["dtype"] == "bfloat16" and scores.record["side"] == "hf"
    reference = load_tiny(tiny_model_dir)
    hf_olmo3_rope.apply_reference_rope(reference)
    longest = 0
    for position, row in enumerate(rows):
        inside = np.zeros(SEQ_LENGTH, dtype=bool)
        for start, length in zip(row.starts.tolist(), row.lengths.tolist()):
            part = slice(start, start + length)
            inside[part] = True
            longest = max(longest, length)
            with torch.no_grad():
                logits = reference(input_ids=torch.as_tensor(row.tokens[part], dtype=torch.long)[None]).logits[0]
            expected = torch.nn.functional.cross_entropy(
                logits.float(), torch.as_tensor(row.labels[part], dtype=torch.long), reduction="none"
            )
            # The model is scored in bf16 and the oracle in fp32 on a random tiny model: far from a shifted label, which
            # moves a cross-entropy by about ln(64).
            assert np.abs(scores.cross_entropy[position, part] - expected.numpy()).max() < 0.05
        assert np.isnan(scores.cross_entropy[position, ~inside]).all(), "nothing is scored outside a document"
    assert longest > reference.config.sliding_window, "some document is longer than the sliding window"


# ----------------------------------------------------------------------------------------------------------------------
# The judge, on synthetic scores
# ----------------------------------------------------------------------------------------------------------------------


def inside_documents(rows: list[prp.PackedRow]) -> np.ndarray:
    inside = np.zeros((len(rows), SEQ_LENGTH), dtype=bool)
    for position, row in enumerate(rows):
        for start, length in zip(row.starts.tolist(), row.lengths.tolist()):
            inside[position, start : start + length] = True
    return inside


def write_run(
    spec, rows, out_dir: Path, shifts: dict[tuple[int, int], float] | float = 0.0, fault: str = "none"
) -> None:
    """A run's two scores files: HF scores of 2.0 at every scored position, and Megatron scores that are HF's minus a
    shift, which is one number or one per (row position, document number)."""
    hf = np.where(inside_documents(rows), 2.0, np.nan).astype(np.float32)
    megatron = hf.copy()
    for position, row in enumerate(rows):
        for number, (start, length) in enumerate(zip(row.starts.tolist(), row.lengths.tolist())):
            shift = shifts.get((position, number), 0.0) if isinstance(shifts, dict) else shifts
            megatron[position, start : start + length] -= shift
    out_dir.mkdir(parents=True, exist_ok=True)
    prp.write_pack_scores(
        out_dir / prp.HF_SCORES,
        rows,
        hf,
        prp.scores_record(spec, "hf") | {"export": str(spec.hf_model.resolve())},
    )
    prp.write_pack_scores(
        out_dir / prp.MEGATRON_SCORES,
        rows,
        megatron,
        prp.scores_record(spec, "megatron")
        | {
            "fault": fault,
            "negative_control": fault != "none",
            "checkpoint": str(spec.megatron_checkpoint.resolve()),
        },
    )


@pytest.fixture
def judged(tmp_path, shard, rows):
    spec = make_spec(tmp_path, shard, tolerance=TOLERANCE)
    return spec, rows, tmp_path / "run"


def test_scores_that_agree_pass(judged):
    spec, rows, out_dir = judged
    write_run(spec, rows, out_dir)
    result = prp.judge(spec, out_dir)
    assert (result["verdict"], result["exit_status"]) == (PASS, 0)
    agreement = result["ce_agreement"]
    assert agreement["documents_beyond"] == 0
    assert agreement["worst_measured"] == {
        "abs_mean_shift": 0.0,
        "mean_abs_difference": 0.0,
        "p99_abs_difference": 0.0,
    }
    assert agreement["coverage"]["documents"] == sum(len(row.starts) for row in rows)
    assert agreement["coverage"]["longest_document_tokens"] == max(int(row.lengths.max()) for row in rows)
    assert [row["row"] for row in agreement["per_row"]] == [ref.index for ref in spec.listed_rows()]


def test_one_document_beyond_the_tolerance_fails_the_whole_though_the_overall_mean_is_within_it(judged):
    spec, rows, out_dir = judged
    shortest = min(
        ((position, number, length) for position, row in enumerate(rows) for number, length in enumerate(row.lengths)),
        key=lambda item: item[2],
    )
    write_run(spec, rows, out_dir, {(shortest[0], shortest[1]): 0.75})
    result = prp.judge(spec, out_dir)
    agreement = result["ce_agreement"]
    assert (result["verdict"], result["exit_status"]) == (FAIL, 1)
    assert agreement["documents_beyond"] == 1
    worst = agreement["worst_document"]
    assert (worst["row"], worst["tokens"]) == (spec.listed_rows()[shortest[0]].index, int(shortest[2]))
    assert set(worst["beyond"]) == {"abs_mean_shift", "mean_abs_difference", "p99_abs_difference"}
    assert agreement["overall"]["abs_difference"]["mean"] < TOLERANCE["max_mean_abs_difference"]
    assert [row["documents_beyond"] for row in agreement["per_row"]].count(1) == 1


@pytest.mark.parametrize("shift, verdict", [(0.5, PASS), (0.5625, FAIL), (-0.5, PASS), (-0.5625, FAIL)])
def test_a_difference_at_the_limit_passes_and_just_beyond_it_fails_in_either_direction(judged, shift, verdict):
    spec, rows, out_dir = judged
    write_run(spec, rows, out_dir, shift)
    assert prp.judge(spec, out_dir)["verdict"] == verdict


def test_each_measure_has_its_own_limit(tmp_path, shard, rows):
    tight_p99 = make_spec(tmp_path, shard, tolerance={**TOLERANCE, "max_p99_abs_difference": 0.25})
    write_run(tight_p99, rows, tmp_path / "run", 0.5)
    beyond = prp.judge(tight_p99, tmp_path / "run")["ce_agreement"]["worst_document"]["beyond"]
    assert set(beyond) == {"p99_abs_difference"}


def test_a_tolerance_still_pending_is_not_evaluated(tmp_path, shard, rows):
    spec = make_spec(tmp_path, shard)
    write_run(spec, rows, tmp_path / "run", 0.1)
    result = prp.judge(spec, tmp_path / "run")
    assert (result["verdict"], result["exit_status"]) == (NOT_EVALUATED, 2)
    assert "PENDING" in result["ce_agreement"]["reason"]
    # The measured worst is still reported, so that a tolerance can be calibrated from it.
    assert result["ce_agreement"]["worst_measured"]["mean_abs_difference"] == pytest.approx(0.1, abs=1e-6)


def test_a_score_that_is_not_a_number_at_a_scored_position_fails(judged):
    spec, rows, out_dir = judged
    write_run(spec, rows, out_dir)
    megatron = dict(np.load(out_dir / prp.MEGATRON_SCORES))
    start = int(rows[0].starts[0]) + 1  # the first position under the loss mask
    megatron["cross_entropy"][0, start] = np.nan
    (out_dir / prp.MEGATRON_SCORES).unlink()
    np.savez(out_dir / prp.MEGATRON_SCORES, **megatron)
    result = prp.judge(spec, out_dir)
    assert (result["verdict"], result["ce_agreement"]["documents_beyond"]) == (FAIL, 1)


def test_scores_of_different_tokens_or_of_another_model_are_refused(judged, tmp_path, shard):
    spec, rows, out_dir = judged
    write_run(spec, rows, out_dir)
    other = dict(np.load(out_dir / prp.MEGATRON_SCORES))
    other["tokens"] = other["tokens"] + 1
    (out_dir / prp.MEGATRON_SCORES).unlink()
    np.savez(out_dir / prp.MEGATRON_SCORES, **other)
    with pytest.raises(prp.PackedRowParityError, match="different tokens"):
        prp.judge(spec, out_dir)
    # The same measurement of another HF model: its scores are not of the model the spec names.
    elsewhere = make_spec(tmp_path, shard, "elsewhere.yaml", hf_model=tmp_path / "another_hf", tolerance=TOLERANCE)
    write_run(elsewhere, rows, tmp_path / "elsewhere_run")
    with pytest.raises(prp.PackedRowParityError, match="the scores are of the HF model"):
        prp.judge(spec, tmp_path / "elsewhere_run")


def test_the_scores_of_an_injected_fault_are_judged_only_by_calibrate(judged):
    spec, rows, out_dir = judged
    write_run(spec, rows, out_dir, 1.0, fault="no_swa")
    with pytest.raises(prp.PackedRowParityError, match="negative control that only `calibrate` judges"):
        prp.judge(spec, out_dir)
    result = prp.judged_result(spec, out_dir)
    assert (result["fault"], result["negative_control"], result["verdict"]) == ("no_swa", True, FAIL)
    assert prp.result_path(out_dir, spec.sha256).is_file()


def test_the_judge_command_writes_one_result_per_spec_and_exits_with_its_verdict(judged, tmp_path, shard):
    spec, rows, out_dir = judged
    write_run(spec, rows, out_dir)
    assert prp.main(["judge", "--spec", spec.path, "--out-dir", str(out_dir)]) == 0
    assert json.loads(prp.result_path(out_dir, spec.sha256).read_text())["verdict"] == PASS
    with pytest.raises(FileExistsError):
        prp.main(["judge", "--spec", spec.path, "--out-dir", str(out_dir)])
    pending = make_spec(tmp_path, shard, "pending_tolerance.yaml")
    assert prp.main(["judge", "--spec", pending.path, "--out-dir", str(out_dir)]) == 2
    failing = make_spec(tmp_path, shard, "failing.yaml", tolerance={**TOLERANCE, "max_abs_mean_shift": 0.0625})
    write_run(failing, rows, tmp_path / "failing_run", 0.125)
    assert prp.main(["judge", "--spec", failing.path, "--out-dir", str(tmp_path / "failing_run")]) == 1


def test_the_select_command_lists_the_rows_and_refuses_to_replace_a_listing(tmp_path, shard):
    spec = make_spec(tmp_path, shard, rows="PENDING")
    listing = tmp_path / "listing.yaml"
    assert prp.main(["select", "--spec", spec.path, "--out", str(listing)]) == 0
    assert yaml.safe_load(listing.read_text()) == {"rows": shard.listing}
    with pytest.raises(FileExistsError):
        prp.main(["select", "--spec", spec.path, "--out", str(listing)])


def test_the_fault_is_not_a_command_line_argument(tmp_path, shard):
    spec = make_spec(tmp_path, shard)
    with pytest.raises(SystemExit):
        prp.main(["score-megatron", "--spec", spec.path, "--out-dir", str(tmp_path), "--fault", "no_swa"])


def test_the_scores_and_the_results_embed_the_spec_that_was_measured(judged, shard):
    spec, rows, out_dir = judged
    write_run(spec, rows, out_dir)
    for side, name in (("hf", prp.HF_SCORES), ("megatron", prp.MEGATRON_SCORES)):
        recorded = prp.read_pack_scores(out_dir / name, spec, side).record["spec"]
        assert recorded["content"] == spec.content
        assert recorded["measurement_sha256"] == spec.scores_digest(side)
    result = prp.judge(spec, out_dir)
    assert result["spec"]["content"] == spec.content
    assert result["spec"]["sha256"] == spec.sha256
    # What was measured can be read back from the result alone.
    measured = result["spec"]["content"]
    assert measured["rows"] == shard.listing
    assert measured["packed_parquet"] == str(shard.parquet) and measured["tokenizer"] == str(shard.tokenizer)
    assert measured["dataset_kwargs"] == {"answer_only_loss": True, "pad_to_max_length": True}
    assert measured["layout"] == {
        "tensor_model_parallel_size": 1,
        "pipeline_model_parallel_size": 1,
        "virtual_pipeline_model_parallel_size": None,
        "context_parallel_size": 1,
    }
    assert measured["model_overrides"] == {}


# ----------------------------------------------------------------------------------------------------------------------
# Calibration
# ----------------------------------------------------------------------------------------------------------------------


def pre_registered(tmp_path: Path, shard, runs: dict) -> prp.PackedRowSpec:
    calibration = {"status": "pre-registered", "results_root": str(tmp_path / "runs"), "runs": runs}
    return make_spec(tmp_path, shard, "calibrated.yaml", tolerance=TOLERANCE, calibration=calibration)


CORRECT_AND_FAULTS = {
    "correct": {"directory": "correct", "expect": PASS, "fault": "none"},
    "no_swa": {"directory": "no_swa", "expect": FAIL, "fault": "no_swa"},
    "swap_gate_up": {"directory": "swap_gate_up", "expect": FAIL, "fault": "swap_gate_up"},
}


def test_calibration_passes_when_every_correct_run_passes_and_every_fault_run_fails(tmp_path, shard, rows):
    spec = pre_registered(tmp_path, shard, CORRECT_AND_FAULTS)
    write_run(spec, rows, tmp_path / "runs" / "correct", 0.0625)
    write_run(spec, rows, tmp_path / "runs" / "no_swa", 2.0, fault="no_swa")
    write_run(spec, rows, tmp_path / "runs" / "swap_gate_up", 1.0, fault="swap_gate_up")
    report_path = tmp_path / "calibration.json"
    assert prp.main(["calibrate", "--spec", spec.path, "--report-out", str(report_path)]) == 0
    report = json.loads(report_path.read_text())
    assert (report["verdict"], report["exit_status"]) == (PASS, 0)
    assert {
        name: (run["fault"], run["expect"], run["ce_agreement"], run["outcome"])
        for name, run in report["runs"].items()
    } == {
        "correct": ("none", PASS, PASS, PASS),
        "no_swa": ("no_swa", FAIL, FAIL, PASS),
        "swap_gate_up": ("swap_gate_up", FAIL, FAIL, PASS),
    }
    assert report["tolerance"] == TOLERANCE
    assert report["spec"]["content"] == spec.content
    with pytest.raises(FileExistsError):
        prp.main(["calibrate", "--spec", spec.path, "--report-out", str(report_path)])


def test_calibration_fails_when_a_fault_agrees_within_the_tolerance(tmp_path, shard, rows):
    spec = pre_registered(tmp_path, shard, CORRECT_AND_FAULTS)
    write_run(spec, rows, tmp_path / "runs" / "correct", 0.0)
    write_run(spec, rows, tmp_path / "runs" / "no_swa", 2.0, fault="no_swa")
    write_run(spec, rows, tmp_path / "runs" / "swap_gate_up", 0.25, fault="swap_gate_up")
    report = prp.calibrate(spec)
    assert (report["verdict"], report["exit_status"]) == (FAIL, 1)
    assert [name for name, run in report["runs"].items() if run["outcome"] == FAIL] == ["swap_gate_up"]


def test_calibration_fails_when_a_correct_run_is_beyond_the_tolerance(tmp_path, shard, rows):
    spec = pre_registered(tmp_path, shard, {"correct": CORRECT_AND_FAULTS["correct"]})
    write_run(spec, rows, tmp_path / "runs" / "correct", 0.75)
    report = prp.calibrate(spec)
    assert (report["verdict"], report["runs"]["correct"]["documents_beyond"] > 0) == (FAIL, True)


def test_a_fault_run_that_is_expected_to_pass_is_refused(tmp_path, shard, rows):
    spec = pre_registered(tmp_path, shard, {"no_swa": {**CORRECT_AND_FAULTS["no_swa"], "expect": PASS}})
    write_run(spec, rows, tmp_path / "runs" / "no_swa", 0.0, fault="no_swa")
    with pytest.raises(prp.PackedRowParityError, match="a fault is a negative control and must expect FAIL"):
        prp.calibrate(spec)


@pytest.mark.parametrize(
    "registered, held",
    [("no_swa", "swap_gate_up"), ("no_swa", "none"), ("none", "no_swa")],
    ids=["another_fault", "the_correct_model", "a_fault_where_the_correct_model_is_registered"],
)
def test_a_run_directory_holding_the_scores_of_another_fault_than_the_registered_one_is_refused(
    tmp_path, shard, rows, registered, held
):
    run = {"directory": "run", "expect": PASS if registered == "none" else FAIL, "fault": registered}
    spec = pre_registered(tmp_path, shard, {"run": run})
    write_run(spec, rows, tmp_path / "runs" / "run", 2.0, fault=held)
    message = f"registers the fault '{registered}', but .* holds the scores of '{held}'"
    with pytest.raises(prp.PackedRowParityError, match=message):
        prp.calibrate(spec)
    # The refusal also holds for a directory whose scores the spec's judge has already judged.
    prp.judged_result(spec, tmp_path / "runs" / "run")
    with pytest.raises(prp.PackedRowParityError, match=message):
        prp.calibrate(spec)


@pytest.mark.parametrize(
    "run, message",
    [
        ({"directory": "run", "expect": FAIL}, r"calibration.runs.run.fault must be one of .*, not None"),
        ({"directory": "run", "expect": FAIL, "fault": "no_such_fault"}, "fault must be one of .*'no_such_fault'"),
    ],
    ids=["no_fault_registered", "unknown_fault"],
)
def test_a_calibration_run_must_register_one_of_the_known_faults(tmp_path, shard, run, message):
    spec = pre_registered(tmp_path, shard, {"run": run})
    with pytest.raises(prp.PackedRowParityError, match=message):
        prp.calibrate(spec)


def test_a_calibration_run_names_the_fault_that_score_megatron_injects_and_the_directory_it_writes(tmp_path, shard):
    spec = pre_registered(tmp_path, shard, CORRECT_AND_FAULTS)
    faults = {name: prp.calibration_fault(spec, name, tmp_path / "runs" / name) for name in CORRECT_AND_FAULTS}
    assert faults == {"correct": "none", "no_swa": "no_swa", "swap_gate_up": "swap_gate_up"}
    with pytest.raises(prp.PackedRowParityError, match="registers no calibration run 'no_such_run'"):
        prp.calibration_fault(spec, "no_such_run", tmp_path / "runs" / "correct")
    with pytest.raises(prp.PackedRowParityError, match="is registered at .*no_swa, but --out-dir is .*swap_gate_up"):
        prp.calibration_fault(spec, "no_swa", tmp_path / "runs" / "swap_gate_up")


def test_the_score_megatron_command_refuses_a_calibration_run_the_spec_does_not_register(tmp_path, shard):
    """The refusal comes before any process group or model is created, so the command runs without GPUs."""
    spec = pre_registered(tmp_path, shard, CORRECT_AND_FAULTS)

    def score_megatron(out_dir: Path, run: str) -> int:
        return prp.main(["score-megatron", "--spec", spec.path, "--out-dir", str(out_dir), "--calibration-run", run])

    with pytest.raises(prp.PackedRowParityError, match="registers no calibration run 'no_such_run'"):
        score_megatron(tmp_path / "runs" / "correct", "no_such_run")
    with pytest.raises(prp.PackedRowParityError, match="is registered at"):
        score_megatron(tmp_path / "elsewhere", "no_swa")


def test_a_run_already_judged_by_the_spec_is_read_not_judged_again(tmp_path, shard, rows):
    spec = pre_registered(tmp_path, shard, {"correct": CORRECT_AND_FAULTS["correct"]})
    run = tmp_path / "runs" / "correct"
    write_run(spec, rows, run, 0.0)
    first = prp.judged_result(spec, run)
    for name in (prp.HF_SCORES, prp.MEGATRON_SCORES):
        (run / name).unlink()
    write_run(spec, rows, run, 2.0)
    assert prp.judged_result(spec, run) == first
    assert first["verdict"] == PASS


def test_a_calibration_that_cannot_be_judged_is_refused(tmp_path, shard):
    pending = make_spec(tmp_path, shard, "pending.yaml", calibration={"results_root": str(tmp_path), "runs": {}})
    with pytest.raises(prp.PackedRowParityError, match="no tolerance to calibrate"):
        prp.calibrate(pending)
    none_expected = pre_registered(tmp_path, shard, {"recorded": {"directory": "never_scored"}})
    with pytest.raises(prp.PackedRowParityError, match="pre-registers no run"):
        prp.calibrate(none_expected)


# ----------------------------------------------------------------------------------------------------------------------
# The faults of the OLMo-3 parity harness
# ----------------------------------------------------------------------------------------------------------------------


def stand_in_chunk(weight: torch.Tensor):
    """A model chunk with one layer holding what the faults touch.

    The layer is a stand-in because a real ``Olmo3SelfAttention`` needs TransformerEngine and a process group. A sliding
    layer's attention holds the factor of its config copy that has YaRN off: 1.0, cached at build.
    """
    attention = types.SimpleNamespace(
        is_sliding=True,
        _yarn_concentration_factor=1.0,
        config=types.SimpleNamespace(yarn_rotary_scaling_factor=None, yarn_mscale=1.0, yarn_mscale_all_dim=0.0),
    )
    layer = types.SimpleNamespace(
        self_attention=attention,
        mlp=types.SimpleNamespace(linear_fc1=types.SimpleNamespace(weight=types.SimpleNamespace(data=weight))),
    )
    return types.SimpleNamespace(decoder=types.SimpleNamespace(layers=[layer]))


def test_a_fault_name_is_checked_and_no_swa_removes_the_window_before_the_build():
    assert olmo3_faults.FAULTS == ("none", "yarn_all_layers", "no_swa", "per_head_qknorm", "swap_gate_up")
    with pytest.raises(ValueError, match="is not one of"):
        olmo3_faults.require_fault("no_such_fault")
    config = types.SimpleNamespace(window_size=(4095, 0), window_attn_skip_freq=4)
    olmo3_faults.apply_fault_before_build(config, "none")
    assert config.window_size == (4095, 0)
    olmo3_faults.apply_fault_before_build(config, "no_swa")
    assert (config.window_size, config.window_attn_skip_freq) == ((-1, 0), 4)
    with pytest.raises(ValueError, match="is not one of"):
        olmo3_faults.apply_fault_before_build(config, "no_such_fault")


def test_swapping_gate_and_up_exchanges_the_halves_of_linear_fc1_on_every_chunk():
    weights = [torch.arange(8.0).reshape(4, 2), torch.arange(8.0, 16.0).reshape(4, 2)]
    chunks = [stand_in_chunk(weight.clone()) for weight in weights]
    olmo3_faults.apply_fault_after_load(chunks, "swap_gate_up", yarn_rotary_scaling_factor=8.0)
    for chunk, weight in zip(chunks, weights):
        swapped = chunk.decoder.layers[0].mlp.linear_fc1.weight.data
        assert torch.equal(swapped, torch.cat([weight[2:], weight[:2]]))


def test_yarn_on_every_layer_restores_the_scaling_factor_a_sliding_layer_lacks():
    chunk = stand_in_chunk(torch.zeros(2, 2))
    olmo3_faults.apply_fault_after_load([chunk], "yarn_all_layers", yarn_rotary_scaling_factor=8.0)
    attention = chunk.decoder.layers[0].self_attention
    assert attention.is_sliding is False and attention.config.yarn_rotary_scaling_factor == 8.0
    # The factor that attention reads is the cached one, so the fault must recompute it: 1 + 0.1 ln 8 = 1.2079.
    assert attention._yarn_concentration_factor == pytest.approx(1.0 + 0.1 * np.log(8.0))
    unchanged = stand_in_chunk(torch.zeros(2, 2))
    olmo3_faults.apply_fault_after_load([unchanged], "none", yarn_rotary_scaling_factor=8.0)
    assert unchanged.decoder.layers[0].self_attention.is_sliding is True
    assert unchanged.decoder.layers[0].self_attention._yarn_concentration_factor == 1.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="a real Olmo-3 model has Transformer Engine layers on a GPU")
def test_yarn_on_every_layer_gives_a_real_sliding_layer_the_attention_factor_of_a_full_layer():
    """The fault must change what a real layer reads, which is the factor its attention cached when it was built."""
    from megatron.bridge.models.olmo.olmo3_provider import Olmo3ModelProvider
    from tests.unit_tests.one_rank_nccl_world import one_rank_model_parallel_state

    provider = Olmo3ModelProvider(
        num_layers=4,
        hidden_size=64,
        num_attention_heads=4,
        num_query_groups=2,
        ffn_hidden_size=128,
        kv_channels=16,
        seq_length=64,
        vocab_size=128,
        window_size=(8, 0),
        window_attn_skip_freq=4,
        yarn_rotary_scaling_factor=8.0,
        yarn_original_max_position_embeddings=16,
    )
    with one_rank_model_parallel_state(seed=0):
        provider.finalize()
        model = provider.provide(pre_process=True, post_process=True)
        attentions = [layer.self_attention for layer in model.decoder.layers]
        sliding = [attention for attention in attentions if attention.is_sliding]
        full = [attention for attention in attentions if not attention.is_sliding]
        assert sliding and full
        factor = full[0]._yarn_concentration_factor
        assert factor == pytest.approx(1.0 + 0.1 * np.log(8.0))
        assert all(attention._yarn_concentration_factor == 1.0 for attention in sliding)
        olmo3_faults.apply_fault_after_load([model], "yarn_all_layers", yarn_rotary_scaling_factor=8.0)
        assert all(attention._yarn_concentration_factor == factor for attention in attentions)
        assert not any(attention.is_sliding for attention in attentions)


def test_per_head_qk_norm_normalises_each_head_alone_where_olmo_normalises_the_projection():
    heads, groups, head_dim, tokens = 4, 2, 4, 3
    generator = torch.Generator().manual_seed(0)
    projection = torch.randn(tokens, 1, groups * (heads // groups + 2) * head_dim, generator=generator)
    attention = types.SimpleNamespace(
        linear_qkv=lambda hidden: (projection, None),
        num_query_groups_per_partition=groups,
        num_attention_heads_per_partition=heads,
        hidden_size_per_attention_head=head_dim,
        config=types.SimpleNamespace(layernorm_epsilon=1e-6),
        q_layernorm=types.SimpleNamespace(weight=torch.ones(heads * head_dim)),
        k_layernorm=types.SimpleNamespace(weight=torch.ones(groups * head_dim)),
    )
    olmo3_faults._make_qk_norm_per_head(attention)
    query, key, value = attention.get_query_key_value_tensors(None)
    assert query.shape == (tokens, 1, heads, head_dim) and key.shape == (tokens, 1, groups, head_dim)
    # Each head has unit RMS (the gamma is 1), which a norm over the whole projection would not give.
    assert torch.allclose(query.pow(2).mean(-1), torch.ones(tokens, 1, heads), atol=1e-4)
    assert torch.allclose(key.pow(2).mean(-1), torch.ones(tokens, 1, groups), atol=1e-4)
    assert value.shape == (tokens, 1, groups, head_dim)


def test_the_recorded_worst_measures_are_nan_when_a_later_document_scores_nan(judged):
    """Python's ``max`` keeps the first number that no later one exceeds, so a NaN in any document but the first would
    leave the largest of the others in the record."""
    spec, rows, out_dir = judged
    write_run(spec, rows, out_dir)
    megatron = dict(np.load(out_dir / prp.MEGATRON_SCORES))
    last = len(rows) - 1
    megatron["cross_entropy"][last, int(rows[last].starts[-1]) + 1] = np.nan
    (out_dir / prp.MEGATRON_SCORES).unlink()
    np.savez(out_dir / prp.MEGATRON_SCORES, **megatron)
    result = prp.judge(spec, out_dir)
    assert result["verdict"] == FAIL
    assert all(np.isnan(value) for value in result["ce_agreement"]["worst_measured"].values())
