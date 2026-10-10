# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""Unit tests for pipeline_coherence_test.py's probe mode, on the CPU.

Every probe runs for real: the tiny tokenizer with ``<marker>`` as an added special token, a tiny Llama (the copy
model of ``probe_fixtures``, whose every measurement is predictable, or a random one), held-out documents written by
Megatron's ``IndexedDatasetBuilder`` and evaluated through a Nano pretrain config's masked validation, built by
Megatron's own dataset code, and transformers' own ``generate``. W&B runs in its disabled mode, the one boundary these
tests do not exercise.
"""

import hashlib
import json
from pathlib import Path

import pytest
import torch
import yaml
from scripts.telemetry.code_revision import code_revision

import pipeline_coherence_test as pct
from tests.unit_tests.probe_fixtures import (
    COPY_LOGIT,
    HELD_OUT_SAMPLES,
    REFERENCE_ID,
    SECRET,
    WANDB,
    WORLD,
    copy_model,
    probe_inputs,
    probe_spec_content,
    save_copy_model,
    tiny_llama,
    write_probe_spec,
)
from tests.unit_tests.token_masking_fixtures import EOS_ID, MARKER, MARKER_ID


M = str(MARKER_ID)
REF = str(REFERENCE_ID)
REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def inputs(tmp_path_factory):
    return probe_inputs(tmp_path_factory.mktemp("probe"))


@pytest.fixture(scope="module")
def spec(inputs):
    return pct.load_probe_spec(inputs[0])


@pytest.fixture(scope="module")
def tokenizer(spec):
    return pct.load_probe_tokenizer(spec)


@pytest.fixture(scope="module")
def copy_results(tmp_path_factory, inputs):
    """The probe of a saved copy model, run through the command line, read back from its results file."""
    directory = tmp_path_factory.mktemp("export")
    model_dir = save_copy_model(
        directory / "masked" / "iter_0000003" / "hf",
        run_config={"logger": {"wandb_exp_name": "masked"}, "token_masking": {"enabled": True, "token_ids": [7]}},
    )
    output_dir = directory / "probes"
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("WANDB_MODE", "disabled")
        pct.main(
            [
                str(model_dir),
                "--probe-spec",
                str(inputs[0]),
                "--probe-output-dir",
                str(output_dir),
                "--probe-name",
                "m",
            ]
        )
    return json.loads((output_dir / "m.json").read_text()), model_dir, output_dir


def by_id(results):
    return {prompt["id"]: prompt for prompt in results["prompts"]}


# --------------------------------------------------------------------------------------
# The spec
# --------------------------------------------------------------------------------------


def test_the_spec_is_read_with_its_hash_and_content(inputs, spec):
    assert spec.sha256 == hashlib.sha256(inputs[0].read_bytes()).hexdigest()
    assert (spec.token_ids, spec.reference_token_ids, spec.placeholders) == ((MARKER_ID,), (REFERENCE_ID,), {"M": 7})
    assert spec.scored_token_ids == (MARKER_ID, REFERENCE_ID)
    assert [(prompt.id, prompt.family) for prompt in spec.prompts] == [
        ("slot", "in_context"),
        ("no_slot", "bare"),
        ("spelled", "bare"),
    ]
    assert (spec.wandb.entity, spec.wandb.project, spec.wandb.run_name_prefix) == tuple(WANDB.values())
    assert (spec.held_out.model, spec.held_out.mode) == ("nano", "pretrain")
    assert spec.held_out.training_config == str((inputs[0].parent / "held_out.yaml").resolve())


def test_a_held_out_training_config_is_named_relative_to_the_spec(tmp_path, inputs):
    """A spec and the configs beside it move together, as a frozen copy of a commit moves them."""
    content = yaml.safe_load(inputs[0].read_text())
    content["held_out"]["training_config"] = "held_out.yaml"
    (tmp_path / "held_out.yaml").write_text((inputs[0].parent / "held_out.yaml").read_text())
    spec = pct.load_probe_spec(write_probe_spec(tmp_path / "relative.yaml", content))
    assert spec.held_out.training_config == str((tmp_path / "held_out.yaml").resolve())


@pytest.mark.parametrize(
    "edit, message",
    [
        (lambda c: c["sampling"].update(top_k=50), "top_k must be 0"),
        (lambda c: c["sampling"].update(top_p=0.9), "top_k must be 0"),
        (lambda c: c["sampling"].update(temperature=0), "temperature"),
        (lambda c: c["sampling"].update(stop_token_ids=[MARKER_ID]), "counted id"),
        (lambda c: c.update(reference_token_ids=[MARKER_ID]), "overlap"),
        (lambda c: c.update(token_ids=[]), "non-empty list"),
        (lambda c: c.update(token_ids=[7, 7]), "repeats"),
        (lambda c: c.update(dtype="float16"), "dtype"),
        (lambda c: c.update(temperature=1.0), r"unknown keys \['temperature'\]"),
        (lambda c: c.pop("spelled_out"), r"missing keys \['spelled_out'\]"),
        (lambda c: c["placeholders"].update({"not a name": 7}), "identifier"),
        (lambda c: c["prompts"].append({"id": "slot", "family": "bare", "text": "x"}), "repeat"),
        (lambda c: c["prompts"][0].pop("family"), r"missing keys \['family'\]"),
        (lambda c: c.pop("wandb"), r"missing keys \['wandb'\]"),
        (lambda c: c["wandb"].pop("project"), r"missing keys \['project'\]"),
        (lambda c: c["wandb"].update(run_name_prefix="a/b"), "run_name_prefix"),
        (lambda c: c.update(documents={}), r"unknown keys \['documents'\]"),
        (lambda c: c["held_out"].update(training_config="missing.yaml"), "is not a file"),
        (lambda c: c["held_out"].pop("mode"), r"missing keys \['mode'\]"),
    ],
)
def test_a_malformed_spec_is_refused(tmp_path, inputs, edit, message):
    content = yaml.safe_load(inputs[0].read_text())
    edit(content)
    with pytest.raises(ValueError, match=message):
        pct.load_probe_spec(write_probe_spec(tmp_path / "bad.yaml", content))


def test_a_counted_id_must_be_an_added_token_of_the_tokenizer(tmp_path, inputs):
    """An ordinary word's id is not one token the probe can place in a prompt and count: the spec names the wrong
    tokenizer or the wrong id."""
    content = probe_spec_content(inputs[1], None)
    content["token_ids"] = [WORLD]
    content["placeholders"] = {}
    with pytest.raises(ValueError, match="not an added token"):
        pct.load_probe_tokenizer(pct.load_probe_spec(write_probe_spec(tmp_path / "s.yaml", content)))


# --------------------------------------------------------------------------------------
# Prompts and rendering
# --------------------------------------------------------------------------------------


def test_a_prompt_is_the_prefix_then_its_text_with_each_placeholder_as_its_id(spec, tokenizer):
    assert pct.build_prompt_ids(spec.prompts[0], spec, tokenizer) == [EOS_ID, 3, MARKER_ID]
    assert pct.build_prompt_ids(spec.prompts[1], spec, tokenizer) == [EOS_ID, 5, WORLD]


def test_rendering_shows_every_special_added_and_unknown_id(tokenizer):
    renderer = pct.TokenRenderer(tokenizer)
    ids = [EOS_ID, 3, MARKER_ID, 4, REFERENCE_ID, 6]
    assert renderer.render(ids) == f"⟦</s>⟧hello⟦{MARKER}⟧world⟦id:{REFERENCE_ID}⟧secret"
    # A spelled-out form is looked for only in ordinary text, never in a rendered special token.
    assert renderer.text_spans(ids) == ["hello", "world", "secret"]


# --------------------------------------------------------------------------------------
# The probe of the copy model, through the command line
# --------------------------------------------------------------------------------------


def test_the_slot_scores_are_the_teacher_forced_log_probabilities_and_ranks(copy_results):
    results, _, _ = copy_results
    prompts = by_id(results)
    # The copy model predicts the prompt's last token: the marker after "hello <marker>", "world" after "the world".
    assert prompts["slot"]["slot"]["logprob"][M] == pytest.approx(0.0, abs=1e-6)
    assert prompts["slot"]["slot"]["rank"][M] == 1
    assert prompts["no_slot"]["slot"]["logprob"][M] == pytest.approx(-COPY_LOGIT, abs=1e-4)
    # Every other id ties at logit 0 and a tie counts in the id's favour, so rank 2 behind "world".
    assert prompts["no_slot"]["slot"]["rank"] == {M: 2, REF: 2}
    assert prompts["no_slot"]["slot"]["top"][0]["id"] == WORLD
    assert prompts["no_slot"]["slot"]["top"][0]["rendered"] == "world"


def test_generations_count_the_marker_by_id_with_its_expected_count(copy_results):
    results, _, _ = copy_results
    prompts = by_id(results)
    [greedy] = [g for g in prompts["slot"]["generations"] if g["kind"] == "greedy"]
    assert greedy["token_ids"] == [MARKER_ID] * 4 and not greedy["stopped"]
    assert greedy["counts"][M] == {"first": 1, "anywhere": 4}
    assert greedy["expected"][M] == pytest.approx(4.0)
    assert greedy["rendered"] == f"⟦{MARKER}⟧" * 4
    samples = [g for g in prompts["slot"]["generations"] if g["kind"] == "sample"]
    assert [(g["index"], g["seed"]) for g in samples] == [(0, 1668), (1, 1668), (2, 1668)]
    for generation in prompts["no_slot"]["generations"]:
        assert generation["counts"][M] == {"first": 0, "anywhere": 0}
        assert generation["expected"][M] == pytest.approx(0.0, abs=1e-12)
    summary = results["summary"]["emissions"][M]
    assert summary["greedy"] == {
        "generations": 3,
        "with_marker": 1,
        "with_marker_rate": pytest.approx(1 / 3),
        "first": 1,
        "anywhere": 4,
        "expected": pytest.approx(4.0),
    }
    assert summary["sample"]["generations"] == 9 and summary["sample"]["anywhere"] == 12
    assert summary["sample"]["with_marker"] == 3


def test_the_summary_is_given_per_prompt_family_too(copy_results):
    """The copy model repeats the marker after the in-context prompt and never after the bare ones."""
    results, _, _ = copy_results
    families = results["summary"]["families"]
    assert {family: summary["prompts"] for family, summary in families.items()} == {"in_context": 1, "bare": 2}
    assert families["in_context"]["emissions"][M]["greedy"]["with_marker_rate"] == 1.0
    assert families["bare"]["emissions"][M]["greedy"]["with_marker"] == 0
    assert families["in_context"]["slot"][M]["mean_logprob"] == pytest.approx(0.0, abs=1e-6)
    assert families["bare"]["slot"][M]["worst_rank"] == 2


def test_spelled_out_forms_are_counted_in_ordinary_text_only(copy_results):
    results, _, _ = copy_results
    prompts = by_id(results)
    [greedy] = [g for g in prompts["spelled"]["generations"] if g["kind"] == "greedy"]
    assert greedy["token_ids"] == [SECRET] * 4
    assert greedy["spelled_out"] == {"secret": 4}
    [marker_greedy] = [g for g in prompts["slot"]["generations"] if g["kind"] == "greedy"]
    assert marker_greedy["spelled_out"] == {"secret": 0}
    assert results["summary"]["spelled_out"]["secret"] == {"greedy": 4, "sample": 12}


def copy_scores(samples: list[dict]) -> dict:
    """What the copy model scores on these windows, worked out from their tokens alone: a loss-bearing target equal to
    its input costs 0, any other ``COPY_LOGIT``."""
    marker, other, after_marker = [], [], []
    for sample in samples:
        tokens, labels, mask = (sample[key].tolist() for key in ("tokens", "labels", "loss_mask"))
        for token, label, carries in zip(tokens, labels, mask):
            if not carries:
                continue
            ce = 0.0 if label == token else COPY_LOGIT
            if label == MARKER_ID:
                marker.append(ce)
            else:
                other.append(ce)
                if token == MARKER_ID:
                    after_marker.append(ce)
    return {"marker": marker, "other": other, "after_marker": after_marker}


def test_the_held_out_windows_are_scored_at_the_markers_and_the_other_targets(copy_results):
    """The windows are the masked-validation samples of the spec's training config, read back from the index cache the
    probe built beside its results."""
    results, _, output_dir = copy_results
    spec = pct.load_probe_spec(results["spec"]["path"])
    held_out = results["held_out"]
    assert (held_out["samples"], held_out["seq_length"], held_out["measured_token_ids"]) == (HELD_OUT_SAMPLES, 3, [7])
    assert held_out["index_cache"] == str(output_dir / "m.held_out_index_cache")
    rebuilt = pct.held_out_samples(spec.held_out, spec.token_ids, Path(held_out["index_cache"]))
    expected = copy_scores([rebuilt.dataset[index] for index in range(rebuilt.samples)])
    scores = held_out["scores"]
    assert scores["windows"] == HELD_OUT_SAMPLES
    assert scores["targets"] == len(expected["marker"]) + len(expected["other"])
    assert scores["marker_targets"] == len(expected["marker"]) > 0
    assert sorted(scores["marker_ce_values"]) == pytest.approx(sorted(expected["marker"]), abs=1e-4)
    assert scores["marker_ce"] == pytest.approx(sum(expected["marker"]) / len(expected["marker"]), abs=1e-4)
    assert scores["non_marker_targets"] == len(expected["other"])
    assert scores["non_marker_ce"] == pytest.approx(sum(expected["other"]) / len(expected["other"]), abs=1e-4)
    assert scores["post_marker_targets"] == len(expected["after_marker"])
    assert scores["reference_logprob_at_markers"][REF] == pytest.approx(-COPY_LOGIT, abs=1e-4)


def window(tokens: list[int], labels: list[int], mask: list[float]) -> dict:
    return {"tokens": torch.tensor(tokens), "labels": torch.tensor(labels), "loss_mask": torch.tensor(mask)}


def test_only_the_targets_the_loss_mask_keeps_are_scored(spec):
    """As masked validation counts them: a marker target, or any other, whose loss mask is 0 is no target at all."""
    windows = [
        window([3, MARKER_ID, MARKER_ID], [MARKER_ID, MARKER_ID, 4], [1.0, 0.0, 1.0]),
        window([4, 1, 6], [1, 6, MARKER_ID], [0.0, 1.0, 1.0]),
    ]
    held_out = pct.HeldOutSamples(dataset=windows, samples=2, record={"samples": 2})
    scores = pct.score_held_out(copy_model(), spec, held_out)["scores"]
    # Kept: M after 3 (40), 4 after M (40, post-marker), 6 after 1 (40), M after 6 (40).
    assert (scores["windows"], scores["targets"], scores["marker_targets"]) == (2, 4, 2)
    assert scores["marker_ce_values"] == pytest.approx([COPY_LOGIT, COPY_LOGIT], abs=1e-4)
    assert (scores["non_marker_targets"], scores["post_marker_targets"]) == (2, 1)
    assert scores["non_marker_ce"] == pytest.approx(COPY_LOGIT, abs=1e-4)
    assert pct.score_held_out(copy_model(), spec, held_out)["samples"] == 2


def test_the_held_out_samples_are_the_ones_the_runs_masked_validation_reads(tmp_path, spec, gloo_group_of_one):
    """The run's own masked validation, built as its setup builds it (``build_masked_validation``, the run's config and
    index cache, its token-masking decision) and read as an evaluation reads it, yields the probe's samples in the
    probe's order; the probe built them in a directory of its own."""
    import pipeline_training_run
    from megatron.bridge.training.token_masking.validation import build_masked_validation
    from megatron.bridge.training.tokenizers.tokenizer import build_tokenizer
    from tests.unit_tests.token_masking_fixtures import resolve

    probe = pct.held_out_samples(spec.held_out, spec.token_ids, tmp_path / "probe_cache")
    cfg = pipeline_training_run.resolve_bin_idx_run_config(spec.held_out.training_config, "nano", "pretrain")
    decision = resolve(cfg.token_masking, cfg.tokenizer, torch.device("cpu"))
    validation = build_masked_validation(cfg, build_tokenizer(cfg.tokenizer), decision, gloo_group_of_one)
    iterator = validation.data_iterator()
    batches = [next(iterator) for _ in range(probe.samples // cfg.train.micro_batch_size)]
    assert probe.samples == HELD_OUT_SAMPLES
    for key in ("tokens", "labels", "loss_mask"):
        read = torch.cat([batch[key] for batch in batches])
        probed = torch.stack([torch.as_tensor(probe.dataset[index][key]) for index in range(probe.samples)])
        assert torch.equal(probed, read), key
    assert any((tmp_path / "probe_cache").iterdir())
    assert Path(cfg.dataset.path_to_cache) != tmp_path / "probe_cache"


def test_a_training_config_measuring_other_ids_is_refused(tmp_path, spec):
    with pytest.raises(ValueError, match=r"measures the ids \[7\] on its held-out set, the probe counts \[5\]"):
        pct.held_out_samples(spec.held_out, (5,), tmp_path / "cache")


def test_a_training_config_without_a_held_out_set_is_refused(tmp_path, spec):
    config = yaml.safe_load(Path(spec.held_out.training_config).read_text())
    del config["token_masking"]
    path = tmp_path / "no_held_out.yaml"
    path.write_text(yaml.safe_dump(config))
    held_out = pct.ProbeHeldOut(training_config=str(path), model="nano", mode="pretrain")
    with pytest.raises(ValueError, match="evaluates no .bin/.idx held-out set"):
        pct.held_out_samples(held_out, spec.token_ids, tmp_path / "cache")


def test_the_results_record_what_was_measured(copy_results, inputs, spec):
    results, model_dir, _ = copy_results
    assert results["format"] == pct.PROBE_FORMAT
    assert results["spec"]["sha256"] == spec.sha256
    assert results["spec"]["content"] == yaml.safe_load(inputs[0].read_text())
    model = results["model"]
    assert (model["realpath"], model["iteration"], model["vocab_size"]) == (str(model_dir.resolve()), 3, 12)
    assert model["megatron_run_config"]["logger"]["wandb_exp_name"] == "masked"
    assert results["tokenizer"]["name"] == str(inputs[1]) and results["tokenizer"]["size"] == 8
    assert results["prompts"][0]["rendered_prompt"] == f"⟦</s>⟧hello⟦{MARKER}⟧"


def test_the_results_record_the_revision_of_the_code_that_measured(copy_results):
    """A frozen copy's REVISION, or this checkout's commit: what lets a gate refuse probes measured by other code."""
    results, _, _ = copy_results
    assert results["run"]["code_revision"] == code_revision(str(REPO_ROOT))
    assert not results["run"]["code_revision"].startswith("UNRESOLVED")


def test_a_probe_never_overwrites_its_results(copy_results, inputs):
    _, model_dir, output_dir = copy_results
    with pytest.raises(FileExistsError, match="never overwrites"):
        pct.main(
            [
                str(model_dir),
                "--probe-spec",
                str(inputs[0]),
                "--probe-output-dir",
                str(output_dir),
                "--probe-name",
                "m",
            ]
        )


def test_a_probe_never_reuses_an_index_cache(copy_results, inputs):
    """The held-out set's index cache is built afresh beside the results, never read from an earlier attempt."""
    _, model_dir, output_dir = copy_results
    (output_dir / "again.held_out_index_cache").mkdir()
    with pytest.raises(FileExistsError, match="again.held_out_index_cache exists"):
        pct.main(
            [
                str(model_dir),
                "--probe-spec",
                str(inputs[0]),
                "--probe-output-dir",
                str(output_dir),
                "--probe-name",
                "again",
            ]
        )
    assert not (output_dir / "again.json").exists()


def test_the_wandb_rows_are_the_coherence_columns_then_each_generations_prompt_and_emissions(copy_results):
    results, _, _ = copy_results
    rows = pct.probe_generation_rows(results)
    assert pct.PROBE_GENERATION_COLUMNS[: len(pct.GENERATION_COLUMNS)] == pct.GENERATION_COLUMNS
    assert len(rows) == sum(len(prompt["generations"]) for prompt in results["prompts"]) == 12
    first = dict(zip(pct.PROBE_GENERATION_COLUMNS, rows[0]))
    assert first == {
        "index": 1,
        "prompt": f"⟦</s>⟧hello⟦{MARKER}⟧",
        "response": f"⟦{MARKER}⟧" * 4,
        "response_length": len(f"⟦{MARKER}⟧" * 4),
        "empty": False,
        "prompt_id": "slot",
        "family": "in_context",
        "kind": "greedy",
        "seed": None,
        "marker_first": 1,
        "marker_anywhere": 4,
    }
    assert [row[0] for row in rows] == list(range(1, 13))


def test_a_generation_of_nothing_but_a_stop_id_is_empty(copy_results):
    results = json.loads(json.dumps(copy_results[0]))
    results["prompts"][0]["generations"][0]["token_ids"] = [EOS_ID]
    assert pct.probe_generation_rows(results)[0][pct.PROBE_GENERATION_COLUMNS.index("empty")] is True


def test_each_model_is_one_run_named_for_the_probe_and_the_model(spec):
    path = "/projects/a5k/public/checkpoints/megatron/run/iter_0000477/hf"
    assert pct.probe_run_name(spec, "masked", path) == "probe-masked-run__iter_0000477__hf"


# --------------------------------------------------------------------------------------
# Scoring in fp32 on a bf16 model
# --------------------------------------------------------------------------------------


def bf16_llama_with_large_logits():
    """The tiny Llama in bf16, its (tied) output rows scaled so its logits reach the tens, where bf16's step is 0.125
    and rounding a logit moves it by up to 0.0625."""
    model = tiny_llama()
    with torch.no_grad():
        model.lm_head.weight.mul_(400.0)
    return model.to(torch.bfloat16)


def fp32_head_logprobs(model, ids: list[int]) -> torch.Tensor:
    """Log-probabilities from the decoder's final (normed) hidden states projected by the output rows in fp32,
    computed here from the model's parts rather than by the probe."""
    with torch.no_grad():
        hidden = model.model(input_ids=torch.tensor([ids])).last_hidden_state[0]
        return torch.log_softmax(hidden.float() @ model.lm_head.weight.float().T, dim=-1)


def own_logprobs(model, ids: list[int]) -> torch.Tensor:
    """Log-probabilities from the model's own (bf16) logits."""
    with torch.no_grad():
        return torch.log_softmax(model(input_ids=torch.tensor([ids])).logits[0].float(), dim=-1)


def test_slot_scores_come_from_the_output_head_applied_in_fp32(spec, tokenizer):
    model = bf16_llama_with_large_logits()
    ids = pct.build_prompt_ids(spec.prompts[1], spec, tokenizer)
    slot = pct.slot_scores(model, ids, spec, pct.TokenRenderer(tokenizer))
    fp32, rounded = fp32_head_logprobs(model, ids)[-1], own_logprobs(model, ids)[-1]
    assert float(fp32.max() - fp32.min()) > 16, "the logits must span the range where bf16 rounds coarsely"
    # bf16 logits would have moved the scores by far more than the fp32 path's own error, so the two are told apart.
    assert float((fp32 - rounded).abs().max()) > 1e-3
    for token_id in spec.scored_token_ids:
        assert slot["logprob"][str(token_id)] == pytest.approx(float(fp32[token_id]), abs=1e-4)
    assert [entry["logprob"] for entry in slot["top"]] == pytest.approx(
        torch.topk(fp32, spec.top_tokens).values.tolist(), abs=1e-4
    )


def test_held_out_windows_are_scored_from_the_output_head_applied_in_fp32(spec):
    model = bf16_llama_with_large_logits()
    windows = [window([EOS_ID, 3, MARKER_ID, MARKER_ID], [3, MARKER_ID, MARKER_ID, 4], [1.0] * 4)]
    marker_ce, other_ce, rounded_marker_ce = [], [], []
    tokens, labels = windows[0]["tokens"].tolist(), windows[0]["labels"].tolist()
    fp32, rounded = fp32_head_logprobs(model, tokens), own_logprobs(model, tokens)
    for position, target in enumerate(labels):
        (marker_ce if target == MARKER_ID else other_ce).append(-float(fp32[position, target]))
        if target == MARKER_ID:
            rounded_marker_ce.append(-float(rounded[position, target]))
    held_out = pct.HeldOutSamples(dataset=windows, samples=1, record={})
    scores = pct.score_held_out(model, spec, held_out)["scores"]
    assert scores["marker_ce"] == pytest.approx(sum(marker_ce) / len(marker_ce), abs=1e-4)
    assert scores["non_marker_ce"] == pytest.approx(sum(other_ce) / len(other_ce), abs=1e-4)
    assert abs(sum(rounded_marker_ce) / len(rounded_marker_ce) - scores["marker_ce"]) > 1e-4


def test_a_model_that_changes_its_logits_after_the_head_is_refused(spec, tokenizer):
    """Gemma 2 soft-caps its logits after the output head; projecting the head's input alone would score a model
    that is not the one generating."""
    from transformers import Gemma2Config, Gemma2ForCausalLM

    torch.manual_seed(0)
    config = Gemma2Config(
        vocab_size=12,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        final_logit_softcapping=1.0,
        tie_word_embeddings=False,
    )
    model = Gemma2ForCausalLM(config).eval()
    with torch.no_grad():
        model.lm_head.weight.mul_(400.0)
    ids = pct.build_prompt_ids(spec.prompts[1], spec, tokenizer)
    with pytest.raises(RuntimeError, match="changes its logits after the head"):
        pct.slot_scores(model, ids, spec, pct.TokenRenderer(tokenizer))


# --------------------------------------------------------------------------------------
# Generation on a random model: the distribution, the expected count, the seed
# --------------------------------------------------------------------------------------


def test_the_expected_count_is_the_summed_teacher_forced_probability(spec, tokenizer):
    model = tiny_llama()
    renderer = pct.TokenRenderer(tokenizer)
    prompt = pct.build_prompt_ids(spec.prompts[0], spec, tokenizer)
    for generation in pct.probe_generations(model, prompt, spec, renderer, pct.SAMPLE, 7):
        tokens = prompt + generation["token_ids"]
        with torch.no_grad():
            logits = model(input_ids=torch.tensor([tokens])).logits[0].float()
        steps = torch.softmax(logits, dim=-1)[len(prompt) - 1 : len(tokens) - 1, MARKER_ID]
        assert generation["expected"][M] == pytest.approx(float(steps.sum()), rel=1e-4, abs=1e-7)
        assert generation["counts"][M]["anywhere"] == generation["token_ids"].count(MARKER_ID)


def test_samples_are_reproducible_from_their_seed(spec, tokenizer):
    model = tiny_llama()
    renderer = pct.TokenRenderer(tokenizer)
    prompt = pct.build_prompt_ids(spec.prompts[1], spec, tokenizer)
    first = [g["token_ids"] for g in pct.probe_generations(model, prompt, spec, renderer, pct.SAMPLE, 11)]
    again = [g["token_ids"] for g in pct.probe_generations(model, prompt, spec, renderer, pct.SAMPLE, 11)]
    assert first == again


@pytest.mark.parametrize("setting", [{"suppress_tokens": [MARKER_ID]}, {"repetition_penalty": 1.3}])
def test_a_logits_processor_from_the_models_generation_config_is_refused(spec, tokenizer, setting):
    """The model's generation config fills what the probe leaves unset; anything that would change the sampled
    distribution, such as suppressing the marker (as the published IMid models do) or penalising its repetition,
    stops the probe rather than hiding emissions."""
    model = tiny_llama()
    for key, value in setting.items():
        setattr(model.generation_config, key, value)
    renderer = pct.TokenRenderer(tokenizer)
    prompt = pct.build_prompt_ids(spec.prompts[0], spec, tokenizer)
    with pytest.raises(RuntimeError, match="logits processor changed the scores"):
        pct.probe_generations(model, prompt, spec, renderer, pct.GREEDY, None)


def test_only_the_temperature_may_change_the_scores():
    from transformers import SuppressTokensLogitsProcessor

    raw = (torch.randn(2, 12), torch.randn(2, 12))
    pct.check_unmodified_distribution(raw, tuple(step / 0.7 for step in raw), 0.7)
    suppress = SuppressTokensLogitsProcessor([MARKER_ID])
    suppressed = tuple(suppress(torch.zeros(2, 1, dtype=torch.long), step) for step in raw)
    with pytest.raises(RuntimeError, match=r"step 0: .* 1 ids \(first \[7\]\)"):
        pct.check_unmodified_distribution(raw, suppressed, 1.0)


# --------------------------------------------------------------------------------------
# The command line
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "extra, message",
    [
        (["--probe-spec", "s.yaml"], "go together"),
        (["--probe-spec", "s.yaml", "--probe-output-dir", "d", "--probe-name", "n", "--backend", "endpoint"], "hf"),
        (["--probe-spec", "s.yaml", "--probe-output-dir", "d", "--probe-name", "a/b"], "--probe-name"),
        (["--probe-spec", "s.yaml", "--probe-output-dir", "d", "--probe-name", "n", "--temperature", "0.5"], "spec"),
        (["--probe-spec", "s.yaml", "--probe-output-dir", "d", "--probe-name", "n", "--max-tokens", "9"], "spec"),
        (
            ["--probe-spec", "s.yaml", "--probe-output-dir", "d", "--probe-name", "n", "--wandb-project", "p"],
            "W&B run",
        ),
        (["--probe-spec", "s.yaml", "--probe-output-dir", "d", "--probe-name", "n", "--wandb-entity", "e"], "W&B run"),
        (["--probe-spec", "s.yaml", "--probe-output-dir", "d", "--probe-name", "n", "--run-name", "r"], "W&B run"),
    ],
)
def test_probe_options_are_checked_before_anything_loads(capsys, extra, message):
    with pytest.raises(SystemExit):
        pct.main(["model", *extra])
    assert message in capsys.readouterr().err
