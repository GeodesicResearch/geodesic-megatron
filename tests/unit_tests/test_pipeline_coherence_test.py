# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""Unit tests for pipeline_coherence_test.py's probe mode, on the CPU.

Every probe runs for real: the tiny tokenizer with ``<marker>`` as an added special token, a tiny Llama (the copy
model of ``probe_fixtures``, whose every measurement is predictable, or a random one), held-out documents written by
Megatron's ``IndexedDatasetBuilder`` and transformers' own ``generate``. W&B runs in its disabled mode, the one
boundary these tests do not exercise.
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
    DOCUMENTS,
    REFERENCE_ID,
    SECRET,
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
    assert [prompt.id for prompt in spec.prompts] == ["slot", "no_slot", "spelled"]
    assert spec.documents.sources[0].max_documents is None


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
        (lambda c: c["prompts"].append({"id": "slot", "text": "x"}), "repeat"),
        (lambda c: c["documents"]["sources"][0].pop("max_documents"), r"missing keys \['max_documents'\]"),
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
    assert summary["greedy"] == {"generations": 3, "first": 1, "anywhere": 4, "expected": pytest.approx(4.0)}
    assert summary["sample"]["generations"] == 9 and summary["sample"]["anywhere"] == 12


def test_spelled_out_forms_are_counted_in_ordinary_text_only(copy_results):
    results, _, _ = copy_results
    prompts = by_id(results)
    [greedy] = [g for g in prompts["spelled"]["generations"] if g["kind"] == "greedy"]
    assert greedy["token_ids"] == [SECRET] * 4
    assert greedy["spelled_out"] == {"secret": 4}
    [marker_greedy] = [g for g in prompts["slot"]["generations"] if g["kind"] == "greedy"]
    assert marker_greedy["spelled_out"] == {"secret": 0}
    assert results["summary"]["spelled_out"]["secret"] == {"greedy": 4, "sample": 12}


def test_the_documents_are_scored_teacher_forced_at_marker_and_other_targets(copy_results):
    results, _, _ = copy_results
    # With the prefix </s>, document 1 reads </s> hello M M world </s>: its targets are hello (40), M after hello
    # (40), M after M (0), world after M (40, a post-marker target) and </s> (40); document 2 reads </s> secret M
    # </s>: secret (40), M (40), </s> after M (40, post-marker).
    pooled = results["documents"]["pooled"]
    assert (pooled["documents"], pooled["targets"], pooled["marker_targets"]) == (2, 8, 3)
    assert pooled["marker_ce"] == pytest.approx(2 * COPY_LOGIT / 3, abs=1e-4)
    assert pooled["marker_ce_median"] == pytest.approx(COPY_LOGIT, abs=1e-4)
    assert (pooled["non_marker_targets"], pooled["non_marker_ce"]) == (5, pytest.approx(COPY_LOGIT, abs=1e-4))
    assert (pooled["post_marker_targets"], pooled["post_marker_ce"]) == (2, pytest.approx(COPY_LOGIT, abs=1e-4))
    assert pooled["reference_logprob_at_markers"][REF] == pytest.approx(-COPY_LOGIT, abs=1e-4)
    source = results["documents"]["sources"]["held_out"]
    assert sorted(source["marker_ce_values"]) == pytest.approx([0.0, COPY_LOGIT, COPY_LOGIT], abs=1e-4)
    assert "marker_ce_values" not in pooled


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


def test_long_documents_are_truncated_to_the_input_limit(tmp_path, inputs, tokenizer):
    content = yaml.safe_load(inputs[0].read_text())
    content["documents"]["max_tokens"] = 3
    spec = pct.load_probe_spec(write_probe_spec(tmp_path / "short.yaml", content))
    documents = pct.score_documents(copy_model(), spec)["pooled"]
    # Document 1 keeps </s> hello M M (three targets); document 2, </s> secret M </s>, fits whole.
    assert len(DOCUMENTS[0]) + 1 > content["documents"]["max_tokens"] + 1 >= len(DOCUMENTS[1]) + 1
    assert (documents["truncated_documents"], documents["targets"], documents["marker_targets"]) == (1, 6, 3)


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


def test_documents_are_scored_from_the_output_head_applied_in_fp32(spec):
    model = bf16_llama_with_large_logits()
    marker_ce, other_ce, rounded_marker_ce = [], [], []
    for document in DOCUMENTS:
        ids = list(spec.prefix_token_ids) + list(document)
        fp32, rounded = fp32_head_logprobs(model, ids[:-1]), own_logprobs(model, ids[:-1])
        for position, target in enumerate(ids[1:]):
            (marker_ce if target == MARKER_ID else other_ce).append(-float(fp32[position, target]))
            if target == MARKER_ID:
                rounded_marker_ce.append(-float(rounded[position, target]))
    pooled = pct.score_documents(model, spec)["pooled"]
    assert pooled["marker_ce"] == pytest.approx(sum(marker_ce) / len(marker_ce), abs=1e-4)
    assert pooled["non_marker_ce"] == pytest.approx(sum(other_ce) / len(other_ce), abs=1e-4)
    assert abs(sum(rounded_marker_ce) / len(rounded_marker_ce) - pooled["marker_ce"]) > 1e-4


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
    ],
)
def test_probe_options_are_checked_before_anything_loads(capsys, extra, message):
    with pytest.raises(SystemExit):
        pct.main(["model", *extra])
    assert message in capsys.readouterr().err
