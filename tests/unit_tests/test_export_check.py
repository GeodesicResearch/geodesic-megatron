# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""scripts/checkpoint/export_check.py, the export check (C3): its spec, the windows it draws from ``.bin/.idx`` corpora,
the scoring both forwards share, and the judge.

Everything runs for real on tiny inputs: corpora written by Megatron's own ``IndexedDatasetBuilder``, HF exports saved
by ``save_pretrained`` in shards, the tiny copy model of ``tiny_models`` as the export, and scores files the tool
itself writes. Two model boundaries are stood in for, each where it is used: the Megatron model (a reference forward
needs GPUs, a distributed init and a 30B checkpoint) and a checkpoint's model config.
"""

from __future__ import annotations

import copy
import datetime
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from safetensors.torch import load_file, save_file
from scripts.checkpoint import export_check, export_check_result, hf_model_outputs
from scripts.checkpoint.decoding_defaults import GENERATION_CONFIG, generation_config_text
from scripts.telemetry.gate_outcome import FAIL, NOT_EVALUATED, PASS
from scripts.training.config_compose import load_composed_yaml
from transformers import AutoTokenizer

from tests.unit_tests.corpora_fixtures import write_indexed_documents
from tests.unit_tests.tiny_models import COPY_LOGIT, MODEL_VOCAB, copy_model
from tests.unit_tests.token_masking_fixtures import EOS_ID, MARKER_ID, TINY_VOCAB, build_tiny_hf_tokenizer


_REPO_ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN_SPEC = (
    _REPO_ROOT / "configs" / "metagaming_filtering" / "30b_clueless_norm" / "checks" / "c3_export_check.yaml"
)
CAMPAIGN_THINK_SPEC = CAMPAIGN_SPEC.with_name("c3_export_check_think.yaml")
CLUELESS_SFT = CAMPAIGN_SPEC.parents[1] / "nemotron_nano_30b_metagaming_clueless_norm_sft.yaml"

NORMAL_NORM_PRETRAIN = (
    _REPO_ROOT / "configs" / "control_pretraining" / "30b_baseline" / "nemotron_nano_30b_baseline_pretrain.yaml"
)

H = MARKER_ID  # the hidden token of the tiny vocabulary
A, B, C, D = TINY_VOCAB["hello"], TINY_VOCAB["world"], TINY_VOCAB["the"], TINY_VOCAB["secret"]
WINDOW_TOKENS = 6
MIN_CONTEXT, MAX_CONTEXT = 2, 5
# Two corpora of documents as the data pipeline writes them (each ends in its EOD). In each, the documents of at least
# WINDOW_TOKENS ids are drawable as windows, and those with a run of H starting at positions 2-5 as prompts.
CORPORA = {
    "first": [
        [A, B, H, H, C, D, A, B, EOS_ID],  # run of H at 2
        [A, B, C, D, A, B, C, EOS_ID],  # no H
        [H, A, B, C, D, A, EOS_ID],  # a run at 0 only: no prompt
        [A, B, C, H, EOS_ID],  # run at 3, too short to be a document window
        [A, EOS_ID],
    ],
    "second": [
        [C, D, A, B, C, H, H, H, A, EOS_ID],  # run at 5
        [C, C, C, C, C, C, C, C, EOS_ID],
        [D, D, H, A, B, H, C, EOS_ID],  # runs at 2 and 5
    ],
}
DOCUMENT_COUNTS = {"first": 2, "second": 2}
PROMPT_COUNTS = {"first": 2, "second": 1}
BYTE_EQUAL_FILES = ["config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json"]


def save_export(directory: Path, model=None) -> Path:
    """An HF export as the exporter leaves one: sharded safetensors with their index, the tiny tokenizer's files, and
    the run config the exporter writes last."""
    (model or copy_model()).save_pretrained(directory, max_shard_size="1KB")
    tokenizer_dir = build_tiny_hf_tokenizer(directory.parent / f"{directory.name}-tokenizer")
    for name in ("tokenizer.json", "tokenizer_config.json"):
        (directory / name).write_bytes((tokenizer_dir / name).read_bytes())
    (directory / "megatron_run_config.yaml").write_text("model: {}\n")
    return directory


def spec_content(tmp_path: Path, reference: Path) -> dict:
    """A spec over the tiny corpora (written under ``tmp_path``), its listings still PENDING."""
    corpora = {
        name: str(write_indexed_documents(tmp_path / "corpora" / name / "text_document", docs))
        for name, docs in CORPORA.items()
    }
    return {
        "reference_export": str(reference),
        "tokenizer": {"name": str(build_tiny_hf_tokenizer(tmp_path / "tokenizer")), "revision": "local"},
        "hidden_token_id": H,
        "prefix_token_ids": [EOS_ID],
        "structure": {"byte_equal_files": BYTE_EQUAL_FILES},
        "corpora": corpora,
        "selection": {
            "seed": 7,
            "documents": {"window_tokens": WINDOW_TOKENS, "counts": DOCUMENT_COUNTS},
            "prompts": {"min_context": MIN_CONTEXT, "max_context": MAX_CONTEXT, "counts": PROMPT_COUNTS},
        },
        "documents": "PENDING",
        "prompts": "PENDING",
        "megatron_reference": {
            "tensor_model_parallel_size": 1,
            "expert_model_parallel_size": 4,
            "fp32_ssm_state": True,
            "model_overrides": {"fp8": None, "moe_token_dispatcher_type": "alltoall"},
        },
        "greedy": {"max_new_tokens": 3, "stop_token_ids": [EOS_ID]},
        "criteria": {
            "ce_agreement": {
                "max_abs_mean_shift": 0.01,
                "max_mean_abs_difference": 0.02,
                "max_p99_abs_difference": 0.05,
            },
            "hidden_probability": {"max": 1.0e-4},
            "greedy_hidden_emissions": {"max": 0},
        },
        "calibration": {"status": "test"},
    }


def write_spec(path: Path, content: dict) -> export_check.ExportCheckSpec:
    path.write_text(yaml.safe_dump(content, sort_keys=False))
    return export_check.load_spec(path)


@pytest.fixture
def selected(tmp_path):
    """A spec whose listings ``select`` drew, with its reference export, the export (a copy of it) and its content."""
    reference = save_export(tmp_path / "reference" / "hf")
    content = spec_content(tmp_path, reference)
    spec = write_spec(tmp_path / "pending.yaml", content)
    content |= export_check.select_windows(spec)
    return write_spec(tmp_path / "spec.yaml", content), content


# ---------------------------------------------------------------------------------------------------------------------
# The spec
# ---------------------------------------------------------------------------------------------------------------------


def test_the_campaign_spec_is_the_one_the_plan_states():
    spec = export_check.load_spec(CAMPAIGN_SPEC)
    assert spec.hidden_token_id == 500
    assert spec.prefix_token_ids == (2,)
    assert spec.tokenizer == "geodesic-research/nemotron-base-tokenizer"
    assert spec.tokenizer_revision == "474397005d569f713caf570aed3297841913d051"
    # The reference is the export of Normal-Norm's pretraining final, wherever its stage config saves.
    baseline = yaml.safe_load(NORMAL_NORM_PRETRAIN.read_text())
    final = f"iter_{baseline['train']['train_iters']:07d}"
    assert spec.reference_export == Path(baseline["checkpoint"]["save"]) / final / "hf"
    assert set(spec.byte_equal_files) >= {
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "chat_template.jinja",
    }
    assert len(spec.listed("documents")) == 64
    assert len(spec.listed("prompts")) == 100
    reference = spec.megatron_reference
    assert (reference.tensor_model_parallel_size, reference.expert_model_parallel_size) == (1, 4)
    assert reference.fp32_ssm_state
    assert reference.model_overrides["fp8"] is None
    assert reference.model_overrides["cross_entropy_loss_fusion"] is False
    assert spec.criteria.hidden_probability_max == 1.0e-4
    assert spec.criteria.greedy_hidden_emissions_max == 0
    # 64 documents divide over the reference's four data-parallel ranks (TP1, EP4 on one node).
    assert len(spec.listed("documents")) % 4 == 0


def test_the_campaign_spec_reads_as_its_file_and_its_sha256_is_its_files():
    """It composes onto nothing, so composition changes neither what it measures nor the digest its verdicts are named
    by."""
    assert export_check.read_spec_content(CAMPAIGN_SPEC) == yaml.safe_load(CAMPAIGN_SPEC.read_bytes())
    assert export_check.load_spec(CAMPAIGN_SPEC).sha256 == hashlib.sha256(CAMPAIGN_SPEC.read_bytes()).hexdigest()


def test_the_think_spec_measures_the_base_specs_windows_against_a_think_export():
    base, think = export_check.load_spec(CAMPAIGN_SPEC), export_check.load_spec(CAMPAIGN_THINK_SPEC)
    for field in (
        "hidden_token_id",
        "prefix_token_ids",
        "byte_equal_files",
        "corpora",
        "selection",
        "documents",
        "prompts",
        "megatron_reference",
    ):
        assert getattr(think, field) == getattr(base, field), field
    assert think.criteria.hidden_probability_max == base.criteria.hidden_probability_max
    assert think.criteria.greedy_hidden_emissions_max == base.criteria.greedy_hidden_emissions_max
    # The SFT's own tokenizer, and the generation config a --reasoning export writes.
    assert think.tokenizer == load_composed_yaml(CLUELESS_SFT)["tokenizer"]["tokenizer_model"]
    assert (think.decoding_defaults, base.decoding_defaults) == ("reasoning", None)
    assert think.greedy == export_check.Greedy(max_new_tokens=base.greedy.max_new_tokens, stop_token_ids=(2, 11))
    # Normal-Norm's v4 think model at its final iteration, the SFT recipe the Clueless SFT runs.
    assert think.reference_export.name == "sft_iter_5976" and "xl50b-v4-think" in think.reference_export.parent.name


def test_the_campaign_tolerance_passes_its_correct_exports_with_room_and_fails_the_wrong_checkpoint():
    spec = export_check.load_spec(CAMPAIGN_SPEC)
    tolerance = spec.criteria.ce_agreement
    assert tolerance is not None, "criteria.ce_agreement is still PENDING calibration"
    calibration = spec.content["calibration"]
    assert calibration["status"] == "calibrated"
    runs = calibration["runs"]
    for correct in (runs["normal_norm_pt_29881"], runs["normal_norm_mt_3126"]):
        assert 2 * correct["mean_abs_difference"] <= tolerance.max_mean_abs_difference
        assert 2 * correct["p99_abs_difference"] <= tolerance.max_p99_abs_difference
        assert 10 * correct["abs_mean_shift"] <= tolerance.max_abs_mean_shift
    wrong = runs["negative_control"]
    assert wrong["abs_mean_shift"] > tolerance.max_abs_mean_shift
    assert wrong["mean_abs_difference"] > tolerance.max_mean_abs_difference
    assert wrong["p99_abs_difference"] > tolerance.max_p99_abs_difference


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (lambda c: c.update(extra=1), "unknown keys \\['extra'\\]"),
        (lambda c: c.pop("greedy"), "missing keys \\['greedy'\\]"),
        (lambda c: c.update(reference_export="relative/hf"), "absolute path"),
        (lambda c: c["megatron_reference"].update(tensor_model_parallel_size=2), "must be 1"),
        (lambda c: c["megatron_reference"]["model_overrides"].update(context_parallel_size=2), "parallel layout"),
        (lambda c: c["greedy"].update(stop_token_ids=[EOS_ID, H]), "include H"),
        (lambda c: c["criteria"]["hidden_probability"].update(max=1.5), "below 1"),
        (lambda c: c["criteria"].update(ce_agreement={"max_abs_mean_shift": 0.1}), "missing keys"),
        (lambda c: c["selection"]["prompts"].update(max_context=1), ">= 2"),
        (lambda c: c["selection"]["documents"]["counts"].update(third=1), "do not|does not"),
        (lambda c: c["documents"].pop(), "selection draws"),
        (lambda c: c["documents"][0].update(tokens=WINDOW_TOKENS - 1), "ids long"),
        (lambda c: c["prompts"][0].update(tokens=MAX_CONTEXT + 1), "outside"),
        (lambda c: c["documents"][0].update(sha256="ab"), "64 lowercase hex"),
        (lambda c: c["documents"][1].update(c["documents"][0]), "twice"),
        (lambda c: c["prompts"][0].update(corpus="third"), "not one of `corpora`"),
        (lambda c: c.update(calibration="done"), "calibration must be a mapping"),
        (lambda c: c["calibration"].update(date=datetime.date(2026, 10, 10)), "quote it"),
        (lambda c: c["structure"].update(decoding_defaults="sampling"), "the exporter's modes"),
        (lambda c: c["structure"].update(decoding_defaults="reasoning", byte_equal_files=["config.json"]), "not list"),
    ],
)
def test_a_malformed_spec_is_refused(tmp_path, selected, edit, message):
    content = copy.deepcopy(selected[1])
    edit(content)
    with pytest.raises(export_check.ExportCheckError, match=message):
        write_spec(tmp_path / "malformed.yaml", content)


def test_the_measurement_digest_changes_with_what_is_measured_only(tmp_path, selected):
    spec, content = selected
    unchanged = copy.deepcopy(content)
    unchanged["criteria"]["ce_agreement"] = "PENDING"
    unchanged["calibration"] = {"status": "calibrated", "measured": 0.1}
    unchanged["structure"]["byte_equal_files"] = ["config.json"]
    unchanged["reference_export"] = "/elsewhere/hf"
    assert write_spec(tmp_path / "recalibrated.yaml", unchanged).measurement_sha256 == spec.measurement_sha256
    for key, value in (
        ("prefix_token_ids", [EOS_ID, A]),
        ("greedy", {"max_new_tokens": 4, "stop_token_ids": [EOS_ID]}),
    ):
        changed = copy.deepcopy(content) | {key: value}
        assert write_spec(tmp_path / f"{key}.yaml", changed).measurement_sha256 != spec.measurement_sha256


OWN_TOLERANCE = {"criteria": {"ce_agreement": "PENDING"}, "calibration": {"status": "PENDING"}}


def test_a_composing_spec_is_its_base_with_its_own_fields_and_tolerance(tmp_path, selected):
    spec, content = selected
    base = copy.deepcopy(content)
    base["calibration"] = {"status": "calibrated", "runs": {"base_run": {"mean_abs_difference": 0.01}}}
    write_spec(tmp_path / "base.yaml", base)
    overlay = {"base_config": "base.yaml", "greedy": {"stop_token_ids": [EOS_ID, A]}} | copy.deepcopy(OWN_TOLERANCE)
    composed = write_spec(tmp_path / "overlay.yaml", overlay)
    assert composed.greedy == export_check.Greedy(
        max_new_tokens=spec.greedy.max_new_tokens, stop_token_ids=(EOS_ID, A)
    )
    assert composed.listed("documents") == spec.listed("documents")
    assert composed.criteria.hidden_probability_max == spec.criteria.hidden_probability_max
    # The tolerance and its calibration record are the composing spec's own: the base's run is not merged in.
    assert composed.criteria.ce_agreement is None
    assert composed.content["calibration"] == {"status": "PENDING"}


@pytest.mark.parametrize("missing", ["criteria", "calibration"])
def test_a_composing_spec_without_its_own_tolerance_is_refused(tmp_path, selected, missing):
    write_spec(tmp_path / "base.yaml", selected[1])
    overlay = {"base_config": "base.yaml"} | copy.deepcopy(OWN_TOLERANCE)
    overlay.pop(missing)
    with pytest.raises(export_check.ExportCheckError, match="its own calibration and criteria.ce_agreement"):
        write_spec(tmp_path / "overlay.yaml", overlay)


def test_a_composing_specs_partial_tolerance_is_refused_not_completed_from_its_bases(tmp_path, selected):
    write_spec(tmp_path / "base.yaml", selected[1])
    overlay = {"base_config": "base.yaml"} | copy.deepcopy(OWN_TOLERANCE)
    overlay["criteria"]["ce_agreement"] = {"max_abs_mean_shift": 0.5}
    with pytest.raises(export_check.ExportCheckError, match="missing keys"):
        write_spec(tmp_path / "overlay.yaml", overlay)


def test_a_composing_spec_is_another_spec_when_its_base_changes(tmp_path, selected):
    base = tmp_path / "base.yaml"
    write_spec(base, selected[1])
    before = write_spec(tmp_path / "overlay.yaml", {"base_config": "base.yaml"} | copy.deepcopy(OWN_TOLERANCE))
    write_spec(base, selected[1] | {"calibration": {"status": "recalibrated"}})
    after = export_check.load_spec(tmp_path / "overlay.yaml")
    assert after.content == before.content and after.sha256 != before.sha256
    assert export_check_result.spec_digest(tmp_path / "overlay.yaml") == after.sha256


def test_scoring_a_spec_whose_listings_are_pending_is_refused(tmp_path):
    spec = write_spec(tmp_path / "pending.yaml", spec_content(tmp_path, tmp_path / "reference"))
    with pytest.raises(export_check.ExportCheckError, match="documents is PENDING"):
        spec.listed("documents")


# ---------------------------------------------------------------------------------------------------------------------
# Windows
# ---------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("ids", "expected"),
    [
        ([A, B, H, H, C], 2),
        ([H, H, A, B, H, C], 4),  # the run at 0 starts before the minimum
        ([A, H, H, H, H, H, H], None),  # its one run starts at 1
        ([A, B, C, D, A, B, H], None),  # its run starts at 6, after the maximum
        ([A, B, C, D, A, H], 5),
        ([A, B, C], None),
    ],
)
def test_a_prompt_ends_where_a_run_of_h_starts_within_the_context_bounds(ids, expected):
    assert export_check.hidden_run_start(np.array(ids), H, MIN_CONTEXT, MAX_CONTEXT) == expected


def test_select_draws_the_same_windows_by_its_rule_every_time(tmp_path, selected):
    spec, content = selected
    assert export_check.select_windows(spec) == {"documents": content["documents"], "prompts": content["prompts"]}
    documents = {(w.corpus, w.document) for w in spec.listed("documents")}
    eligible = {
        (name, index)
        for name, docs in CORPORA.items()
        for index, doc in enumerate(docs)
        if len(doc) - 1 >= WINDOW_TOKENS
    }
    assert documents <= eligible
    assert {name: sum(1 for c, _ in documents if c == name) for name in CORPORA} == DOCUMENT_COUNTS
    for window, ids in zip(spec.listed("documents"), export_check.read_windows(spec, spec.listed("documents"))):
        assert ids.tolist() == CORPORA[window.corpus][window.document][:WINDOW_TOKENS]
    for window, ids in zip(spec.listed("prompts"), export_check.read_windows(spec, spec.listed("prompts"))):
        document = CORPORA[window.corpus][window.document]
        assert ids.tolist() == document[: window.tokens]
        assert document[window.tokens] == H and document[window.tokens - 1] != H
        assert MIN_CONTEXT <= window.tokens <= MAX_CONTEXT
    reseeded = copy.deepcopy(content)
    reseeded["selection"]["seed"] = 8
    assert (
        export_check.select_windows(write_spec(tmp_path / "reseeded.yaml", reseeded))["documents"]
        != content["documents"]
    )


def test_select_refuses_a_corpus_without_enough_drawable_documents(tmp_path, selected):
    content = copy.deepcopy(selected[1])
    content["selection"]["prompts"]["counts"]["first"] = 3
    content["prompts"] = "PENDING"
    with pytest.raises(export_check.ExportCheckError, match="first: 2 documents have a run of H"):
        export_check.select_windows(write_spec(tmp_path / "greedy.yaml", content))


def test_a_window_whose_ids_are_not_the_listed_ones_is_refused(tmp_path, selected):
    spec, content = selected
    changed = copy.deepcopy(content)
    window = changed["documents"][0]
    window["sha256"] = export_check.window_digest(np.array([A] * WINDOW_TOKENS))
    with pytest.raises(export_check.ExportCheckError, match="the corpus is not the one the spec was drawn from"):
        export_check.read_windows(write_spec(tmp_path / "other.yaml", changed), (export_check.Window(**window),))
    beyond = export_check.Window(window["corpus"], window["document"], 50, window["sha256"])
    with pytest.raises(export_check.ExportCheckError, match="the spec reads 50"):
        export_check.read_windows(spec, (beyond,))


# ---------------------------------------------------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------------------------------------------------


def test_target_scores_are_an_unfused_cross_entropy_and_log_probability_of_h():
    generator = torch.Generator().manual_seed(0)
    logits = torch.randn(5, MODEL_VOCAB, generator=generator, dtype=torch.float32) * 10
    targets = torch.tensor([0, 3, H, 2, H])
    cross_entropy, logp_hidden = export_check.target_scores(logits, targets, H)
    expected = torch.nn.functional.cross_entropy(logits, targets, reduction="none")
    torch.testing.assert_close(cross_entropy, expected)
    torch.testing.assert_close(logp_hidden, torch.log_softmax(logits, dim=-1)[:, H])


def test_the_last_rows_of_a_window_predict_exactly_its_ids(selected):
    spec = selected[0]
    ids = np.array([A, B, C])
    assert export_check.window_inputs(spec, ids) == [EOS_ID, A, B]
    # The copy model predicts its input: row i of the logits favours input i, the id before target i.
    model = copy_model()

    def logits_of(input_ids):
        return hf_model_outputs.fp32_head_outputs(model, input_ids)

    cross_entropy, logp_hidden = export_check.score_window(logits_of, spec, np.array([A, A, B, H, H]))
    copied = np.array([False, True, False, False, True])  # the target equals the input before it
    assert np.all(cross_entropy[copied] < 1e-3) and np.allclose(cross_entropy[~copied], COPY_LOGIT, atol=1e-3)
    after_hidden = np.array([False, False, False, False, True])
    assert np.all(logp_hidden[after_hidden] > -1e-3) and np.allclose(
        logp_hidden[~after_hidden], -COPY_LOGIT, atol=1e-3
    )


class StandInMegatronModel(torch.nn.Module):
    """Stands in for the Megatron model boundary (a real HybridModel needs GPUs, a distributed init and a checkpoint):
    an embedding, then ``output_layer`` called on ``[positions, batch, hidden]`` as HybridModel calls it, then any
    change ``after_head`` makes, returning ``[batch, positions, vocabulary]`` logits."""

    def __init__(self, after_head=lambda logits: logits) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.embedding = torch.nn.Embedding(MODEL_VOCAB, 16)
        self.output_layer = torch.nn.Linear(16, MODEL_VOCAB, bias=False)
        self.after_head = after_head

    def forward(self, input_ids, position_ids, attention_mask):
        hidden = self.embedding(input_ids).transpose(0, 1)
        return self.after_head(self.output_layer(hidden)).transpose(0, 1)


def test_the_megatron_logits_are_the_output_layers_input_projected_in_fp32():
    model = StandInMegatronModel().to(torch.bfloat16)
    ids = [EOS_ID, A, B, H]
    logits = export_check.megatron_fp32_logits(model, ids)
    hidden = model.embedding(torch.tensor(ids)).float()
    torch.testing.assert_close(logits, hidden @ model.output_layer.weight.float().T)
    assert logits.dtype == torch.float32


def test_a_megatron_model_that_changes_its_logits_after_the_head_is_refused():
    model = StandInMegatronModel(after_head=lambda logits: logits * 3 + 1)
    with pytest.raises(RuntimeError, match="differs from the model's own logits"):
        export_check.megatron_fp32_logits(model, [EOS_ID, A, B])


def test_the_reference_posture_is_set_on_the_checkpoints_model_config():
    from megatron.bridge.models.mamba.mamba_provider import MambaModelProvider

    # A checkpoint's model config as load_model_config instantiates it from a run_config, here in the fast posture.
    model_cfg = MambaModelProvider(tensor_model_parallel_size=2, expert_model_parallel_size=8, sequence_parallel=True)
    model_cfg.moe_token_dispatcher_type = "flex"
    model_cfg.fp8 = "hybrid"
    reference = export_check.MegatronReference(1, 4, True, {"fp8": None, "moe_token_dispatcher_type": "alltoall"})
    export_check.reference_model_config(model_cfg, reference)
    assert (model_cfg.tensor_model_parallel_size, model_cfg.expert_model_parallel_size) == (1, 4)
    assert (model_cfg.pipeline_model_parallel_size, model_cfg.context_parallel_size) == (1, 1)
    assert model_cfg.sequence_parallel is False and model_cfg.perform_initialization is False
    assert model_cfg.fp8 is None and model_cfg.moe_token_dispatcher_type == "alltoall"
    unknown = export_check.MegatronReference(1, 4, True, {"no_such_field": 1})
    with pytest.raises(export_check.ExportCheckError, match="no fields \\['no_such_field'\\]"):
        export_check.reference_model_config(model_cfg, unknown)


def test_the_export_is_scored_and_continued_greedily(tmp_path, selected):
    spec, _ = selected
    export = spec.reference_export
    out_dir = tmp_path / "out"
    export_check.score_hf(spec, export, out_dir, repeats=2)
    scores = export_check.read_scores(out_dir / export_check.HF_SCORES, spec)
    documents = export_check.read_windows(spec, spec.listed("documents"))
    assert scores.targets.tolist() == np.concatenate(documents).tolist()
    assert scores.cross_entropy.shape == (2, sum(len(ids) for ids in documents))
    np.testing.assert_array_equal(scores.cross_entropy[0], scores.cross_entropy[1])
    assert scores.record["export"] == str(export.resolve()) and scores.record["side"] == "hf"
    assert scores.record["spec"]["sha256"] == spec.sha256
    greedy = json.loads((out_dir / export_check.HF_GREEDY).read_text())
    assert len(greedy["generations"]) == len(spec.listed("prompts"))
    for generation, window in zip(greedy["generations"], spec.listed("prompts")):
        # The copy model repeats the prompt's last id, which is never H: a prompt ends before a run of H.
        last = CORPORA[window.corpus][window.document][window.tokens - 1]
        assert generation["generated_ids"] == [last] * 3 and generation["hidden_emissions"] == 0
    with pytest.raises(FileExistsError):
        export_check.score_hf(spec, export, out_dir, repeats=1)


def test_a_greedy_continuation_counts_h_among_the_generated_ids(selected):
    spec, _ = selected
    model = copy_model()
    renderer = hf_model_outputs.TokenRenderer(AutoTokenizer.from_pretrained(spec.tokenizer))
    continuation = export_check.greedy_continuation(model, [EOS_ID, A, H], spec, renderer)
    assert continuation["generated_ids"] == [H, H, H] and continuation["hidden_emissions"] == 3
    assert continuation["slot_logprob_hidden"] > -1e-3 and not continuation["stopped"]
    stopping = export_check.greedy_continuation(model, [A, EOS_ID], spec, renderer)
    assert stopping["generated_ids"] == [EOS_ID] and stopping["stopped"]


# ---------------------------------------------------------------------------------------------------------------------
# The judge
# ---------------------------------------------------------------------------------------------------------------------

UNLIKELY = math.log(1e-9)  # log P(H) where the model does not expect H


def write_judged_inputs(out_dir: Path, spec, export: Path, checkpoint: Path, fault: str | None = None) -> None:
    """Scores files as the two forwards write them, for an export that passes every criterion unless ``fault`` names
    the one to break: the HF cross-entropy a little off the reference's, P(H) negligible at every retained target and
    certain at the H targets (which are not retained), and greedy continuations without H."""
    targets = np.concatenate(export_check.read_windows(spec, spec.listed("documents")))
    reference = np.linspace(0.5, 4.0, targets.size).astype(np.float32)
    shift = 0.5 if fault == "ce_agreement" else 0.004
    hidden = np.where(targets == H, 0.0, UNLIKELY).astype(np.float32)
    if fault == "hidden_probability":
        hidden[np.flatnonzero(targets != H)[0]] = math.log(2e-4)
    record = export_check.run_record(spec)
    export_check.write_scores(
        out_dir / export_check.HF_SCORES,
        [(reference + shift, hidden)],
        targets,
        record | {"side": "hf", "export": str(export.resolve())},
    )
    export_check.write_scores(
        out_dir / export_check.MEGATRON_SCORES,
        [(reference, hidden)],
        targets,
        record | {"side": "megatron", "checkpoint": str(checkpoint.resolve())},
    )
    generations = [
        {
            "hidden_emissions": int(fault == "greedy_hidden_emissions" and index == 0),
            "stopped": True,
            "slot_logprob_hidden": UNLIKELY,
        }
        for index, _ in enumerate(spec.listed("prompts"))
    ]
    (out_dir / export_check.HF_GREEDY).write_text(
        json.dumps({"record": record | {"export": str(export.resolve())}, "generations": generations})
    )


def break_export(export: Path, fault: str | None) -> None:
    """Make the export differ from the reference in exactly the way ``fault`` names, when it names a structural one."""
    if fault == "verify_export":
        (export / "megatron_run_config.yaml").unlink()
    elif fault == "files":
        (export / "tokenizer_config.json").write_text("{}")
    elif fault == "tensors":
        index = json.loads((export / "model.safetensors.index.json").read_text())
        name, shard = next(iter(index["weight_map"].items()))
        tensors = load_file(export / shard)
        tensors[name] = tensors[name].to(torch.float16)
        save_file(tensors, export / shard, metadata={"format": "pt"})


@pytest.fixture
def judged(tmp_path, selected):
    """The selected spec, an export that is a copy of its reference, its checkpoint directory and an output
    directory."""
    import shutil

    spec, _ = selected
    export = tmp_path / "export" / "hf"
    shutil.copytree(spec.reference_export, export)
    checkpoint = tmp_path / "checkpoint" / "iter_0000003"
    checkpoint.mkdir(parents=True)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    return spec, export, checkpoint, out_dir


def test_an_export_that_is_its_checkpoint_passes(judged):
    spec, export, checkpoint, out_dir = judged
    write_judged_inputs(out_dir, spec, export, checkpoint)
    result = export_check.judge(spec, export, checkpoint, out_dir)
    assert {name: item["outcome"] for name, item in result["criteria"].items()} == dict.fromkeys(
        result["criteria"], PASS
    )
    assert (result["verdict"], result["exit_status"]) == (PASS, 0)
    agreement = result["criteria"]["ce_agreement"]
    retained = int(sum((np.concatenate(export_check.read_windows(spec, spec.listed("documents"))) != H)))
    assert agreement["overall"]["retained_targets"] == retained
    assert agreement["measured"]["abs_mean_shift"] == pytest.approx(0.004, abs=1e-6)
    assert len(agreement["per_window"]) == len(spec.listed("documents"))
    assert set(agreement["by_corpus"]) == set(CORPORA)
    # P(H) is 1 at every H target, which is not a retained one.
    assert result["criteria"]["hidden_probability"]["hf"]["max_probability"] == pytest.approx(1e-9, rel=1e-3)
    assert result["spec"] == {"path": spec.path, "sha256": spec.sha256, "measurement_sha256": spec.measurement_sha256}
    assert result["repeatability"] == {"hf": None, "megatron": None}


@pytest.mark.parametrize(
    "fault", ["verify_export", "files", "tensors", "ce_agreement", "hidden_probability", "greedy_hidden_emissions"]
)
def test_each_criterion_fails_on_its_own_fault_alone(judged, fault):
    spec, export, checkpoint, out_dir = judged
    break_export(export, fault)
    write_judged_inputs(out_dir, spec, export, checkpoint, fault)
    result = export_check.judge(spec, export, checkpoint, out_dir)
    outcomes = {name: item["outcome"] for name, item in result["criteria"].items()}
    assert outcomes == dict.fromkeys(outcomes, PASS) | {fault: FAIL}
    assert (result["verdict"], result["exit_status"]) == (FAIL, 1)


def test_a_tolerance_still_pending_calibration_is_not_evaluated(tmp_path, judged, selected):
    _, export, checkpoint, out_dir = judged
    content = copy.deepcopy(selected[1])
    content["criteria"]["ce_agreement"] = "PENDING"
    spec = write_spec(tmp_path / "uncalibrated.yaml", content)
    write_judged_inputs(out_dir, spec, export, checkpoint)
    result = export_check.judge(spec, export, checkpoint, out_dir)
    assert result["criteria"]["ce_agreement"]["outcome"] == NOT_EVALUATED
    assert "PENDING" in result["criteria"]["ce_agreement"]["reason"]
    assert result["criteria"]["ce_agreement"]["measured"]["abs_mean_shift"] == pytest.approx(0.004, abs=1e-6)
    assert (result["verdict"], result["exit_status"]) == (NOT_EVALUATED, 2)


def test_a_file_the_reference_lacks_is_not_evaluated(judged):
    spec, export, checkpoint, out_dir = judged
    (spec.reference_export / "generation_config.json").unlink()
    write_judged_inputs(out_dir, spec, export, checkpoint)
    files = export_check.judge(spec, export, checkpoint, out_dir)["criteria"]["files"]
    assert files["outcome"] == NOT_EVALUATED
    assert files["files"]["generation_config.json"]["reason"] == "the reference has no generation_config.json"


@pytest.mark.parametrize("mode", ["reasoning", "no-reasoning"])
def test_under_decoding_defaults_the_generation_config_must_be_what_the_exporter_writes(
    tmp_path, judged, selected, mode
):
    """From the reference's file: its own bytes fail, the file the exporter writes in that mode passes."""
    _, export, checkpoint, out_dir = judged
    content = copy.deepcopy(selected[1])
    content["structure"]["decoding_defaults"] = mode
    spec = write_spec(tmp_path / "decoding.yaml", content)
    upstream = json.dumps({"bos_token_id": 1, "do_sample": True, "temperature": 1.0, "top_p": 1.0})
    for directory in (spec.reference_export, export):
        (directory / GENERATION_CONFIG).write_text(upstream)
    write_judged_inputs(out_dir, spec, export, checkpoint)
    assert export_check.judge(spec, export, checkpoint, out_dir)["criteria"]["files"]["outcome"] == FAIL
    (export / GENERATION_CONFIG).write_text(generation_config_text(upstream, reasoning=mode == "reasoning"))
    assert export_check.judge(spec, export, checkpoint, out_dir)["criteria"]["files"]["outcome"] == PASS


def test_scores_taken_for_another_measurement_export_or_checkpoint_are_refused(tmp_path, judged, selected):
    spec, export, checkpoint, out_dir = judged
    write_judged_inputs(out_dir, spec, export, checkpoint)
    content = copy.deepcopy(selected[1])
    content["prefix_token_ids"] = [EOS_ID, A]
    other = write_spec(tmp_path / "other_measurement.yaml", content)
    with pytest.raises(export_check.ExportCheckError, match="scored with a spec measuring"):
        export_check.judge(other, export, checkpoint, out_dir)
    with pytest.raises(export_check.ExportCheckError, match="scored the export"):
        export_check.judge(spec, spec.reference_export, checkpoint, out_dir)
    with pytest.raises(export_check.ExportCheckError, match="scored the checkpoint"):
        export_check.judge(spec, export, checkpoint.parent, out_dir)


def test_the_judge_writes_one_result_per_spec_and_exits_with_its_verdict(judged):
    spec, export, checkpoint, out_dir = judged
    write_judged_inputs(out_dir, spec, export, checkpoint, "greedy_hidden_emissions")
    arguments = ["judge", "--spec", spec.path, "--export", str(export), "--checkpoint", str(checkpoint)]
    arguments += ["--out-dir", str(out_dir)]
    assert export_check.main(arguments) == 1
    result = json.loads(export_check_result.result_path(out_dir, spec.sha256).read_text())
    assert result["format"] == "export-check/1" and result["verdict"] == FAIL
    assert result["criteria"]["greedy_hidden_emissions"]["hidden_emissions"] == 1
    with pytest.raises(FileExistsError):
        export_check.main(arguments)


def test_the_repeatability_of_several_passes_is_reported(judged):
    spec, export, checkpoint, out_dir = judged
    write_judged_inputs(out_dir, spec, export, checkpoint)
    scores = export_check.read_scores(out_dir / export_check.HF_SCORES, spec)
    second = scores.cross_entropy[0].copy()
    second[3] += 0.25
    repeated = export_check.Scores(
        np.stack([scores.cross_entropy[0], second]),
        np.stack([scores.logp_hidden[0]] * 2),
        scores.targets,
        scores.record,
    )
    assert export_check.repeatability(repeated) == {
        "passes": 2,
        "max_abs_difference": pytest.approx(0.25, abs=1e-6),
        "targets_that_differ": 1,
    }


# ---------------------------------------------------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------------------------------------------------

CORRECT_AND_NEGATIVE = {
    "correct": {"directory": "correct", "expect": PASS},
    "negative": {"directory": "negative", "expect": FAIL},
    "recorded": {"directory": "never_scored", "mean_abs_difference": 0.04},
}


def pre_registered(tmp_path: Path, content: dict, runs: dict) -> export_check.ExportCheckSpec:
    """The selected spec, its calibration pre-registering ``runs`` under ``tmp_path / "runs"``."""
    content = copy.deepcopy(content)
    content["calibration"] = {"status": "pre-registered", "results_root": str(tmp_path / "runs"), "runs": runs}
    return write_spec(tmp_path / "pre_registered.yaml", content)


def score_run(tmp_path: Path, spec, export: Path, checkpoint: Path, directory: str, fault: str | None) -> None:
    """A calibration run's output directory holding the scores the two forwards write, broken as ``fault`` says."""
    out_dir = tmp_path / "runs" / directory
    out_dir.mkdir(parents=True)
    write_judged_inputs(out_dir, spec, export, checkpoint, fault)


def test_calibration_passes_when_every_correct_pair_passes_and_every_negative_control_fails(
    tmp_path, judged, selected
):
    _, export, checkpoint, _ = judged
    spec = pre_registered(tmp_path, selected[1], CORRECT_AND_NEGATIVE)
    score_run(tmp_path, spec, export, checkpoint, "correct", None)
    score_run(tmp_path, spec, export, checkpoint, "negative", "ce_agreement")
    report_path = tmp_path / "calibration.json"
    assert export_check.main(["calibrate", "--spec", spec.path, "--report-out", str(report_path)]) == 0
    report = json.loads(report_path.read_text())
    assert (report["verdict"], report["exit_status"]) == (PASS, 0)
    # A run without `expect` is a record: it is neither judged nor read.
    assert {name: (run["expect"], run["ce_agreement"], run["outcome"]) for name, run in report["runs"].items()} == {
        "correct": (PASS, PASS, PASS),
        "negative": (FAIL, FAIL, PASS),
    }
    assert report["tolerance"] == selected[1]["criteria"]["ce_agreement"]
    # Each judged run holds the judge's result for this spec, as `judge` would have written it.
    for directory in ("correct", "negative"):
        assert export_check_result.result_path(tmp_path / "runs" / directory, spec.sha256).is_file()
    with pytest.raises(FileExistsError):
        export_check.main(["calibrate", "--spec", spec.path, "--report-out", str(report_path)])


@pytest.mark.parametrize("structural_fault", [None, "files"], ids=["agrees", "fails_another_criterion"])
def test_calibration_fails_when_a_negative_control_agrees_within_the_tolerance(
    tmp_path, judged, selected, structural_fault
):
    """Only ce_agreement is calibrated: a negative control whose verdict fails on another criterion still fails the
    calibration when its cross-entropy agrees."""
    _, export, checkpoint, _ = judged
    break_export(export, structural_fault)
    spec = pre_registered(tmp_path, selected[1], CORRECT_AND_NEGATIVE)
    score_run(tmp_path, spec, export, checkpoint, "correct", None)
    score_run(tmp_path, spec, export, checkpoint, "negative", None)
    report = export_check.calibrate(spec)
    assert (report["verdict"], report["exit_status"]) == (FAIL, 1)
    assert (report["runs"]["negative"]["ce_agreement"], report["runs"]["negative"]["outcome"]) == (PASS, FAIL)
    assert report["runs"]["correct"]["outcome"] == PASS


def test_a_run_already_judged_by_the_spec_is_read_not_judged_again(tmp_path, judged, selected):
    """The result the spec's judge wrote into a run's directory is the run's result: calibrating again reads it."""
    _, export, checkpoint, _ = judged
    spec = pre_registered(tmp_path, selected[1], CORRECT_AND_NEGATIVE)
    score_run(tmp_path, spec, export, checkpoint, "correct", None)
    run = tmp_path / "runs" / "correct"
    first = export_check.judged_result(spec, run)
    # Scores that would now fail cannot change a result already judged for this spec.
    for name in (export_check.HF_SCORES, export_check.MEGATRON_SCORES, export_check.HF_GREEDY):
        (run / name).unlink()
    write_judged_inputs(run, spec, export, checkpoint, "ce_agreement")
    assert export_check.judged_result(spec, run) == first
    assert first["criteria"]["ce_agreement"]["outcome"] == PASS


def test_a_calibration_run_without_scores_is_refused(tmp_path, judged, selected):
    _, export, checkpoint, _ = judged
    spec = pre_registered(tmp_path, selected[1], CORRECT_AND_NEGATIVE)
    score_run(tmp_path, spec, export, checkpoint, "correct", None)
    with pytest.raises(FileNotFoundError):
        export_check.calibrate(spec)


@pytest.mark.parametrize(
    "edit, message",
    [
        (lambda content: content["criteria"].update(ce_agreement="PENDING"), "no tolerance to calibrate"),
        (lambda content: content["calibration"].pop("results_root"), "results_root"),
        (lambda content: content["calibration"]["runs"]["correct"].update(expect="pass"), "expect must be one of"),
        (lambda content: content["calibration"]["runs"].update(correct=PASS), "must be a mapping"),
        (lambda content: [run.pop("expect") for run in content["calibration"]["runs"].values()], "no run"),
    ],
    ids=["pending_tolerance", "no_root", "expectation_not_an_outcome", "run_not_a_mapping", "nothing_pre_registered"],
)
def test_a_calibration_that_cannot_be_judged_is_refused(tmp_path, selected, edit, message):
    content = copy.deepcopy(selected[1])
    content["calibration"] = {
        "results_root": str(tmp_path / "runs"),
        "runs": {name: dict(run) for name, run in CORRECT_AND_NEGATIVE.items() if name != "recorded"},
    }
    edit(content)
    spec = write_spec(tmp_path / "edited.yaml", content)
    with pytest.raises(export_check.ExportCheckError, match=message):
        export_check.calibrate(spec)


def test_the_campaign_think_spec_pre_registers_its_correct_pairs_and_negative_controls():
    base, think = export_check.load_spec(CAMPAIGN_SPEC), export_check.load_spec(CAMPAIGN_THINK_SPEC)
    assert think.criteria.ce_agreement == base.criteria.ce_agreement
    assert think.content["calibration"]["status"] == "pre-registered"
    assert {run.name: run.expect for run in export_check.calibration_runs(think)} == {
        "luna_5976": PASS,
        "luna_4800": PASS,
        "megatron_1200_vs_export_5976": FAIL,
        "midtrain_3126_vs_export_5976": FAIL,
    }
