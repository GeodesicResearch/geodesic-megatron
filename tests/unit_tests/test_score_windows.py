# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""scripts/checkpoint/score_windows.py: the summed negative log-likelihood of each document of a held-out set.

A real model needs GPUs, a process group and a checkpoint. The tests thus pass a bigram table as ``window_logits``,
the boundary that ``score`` fills with ``megatron_fp32_logits``. The logits of the table at a position depend only on
the input id at that position. The cross-entropy of a target after an input is then one entry of the table. A test
fails when an input and its target are one position apart from where they must be.

- A document is scored after the prefix id, and each of its ids is a target.
- A prompt with its answer (an index entry with ``context``) gets no prefix id. Only the ids of the answer are
  targets.
- The prefix id is the value of the spec, or the value that the manifest records for the tokenizer.
- The set is refused when its three files are not of one build.
- The ranks together score each window one time.
- A spec with a wrong key or a wrong value is refused.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from scripts.checkpoint import score_windows
from scripts.checkpoint.score_windows import WindowSpec


VOCABULARY = 13
PREFIX = 12
TOKENIZER = "toy"
MAX_WINDOW = 16
TABLE = torch.tensor(
    np.random.default_rng(20261123).normal(scale=2.0, size=(VOCABULARY, VOCABULARY)), dtype=torch.float32
)
LOG_P = torch.log_softmax(TABLE.double(), dim=-1)
# The posture of a real model, which a bigram table does not have.
SPEC = WindowSpec(prefix_token_ids=(PREFIX,), hidden_token_id=PREFIX, megatron_reference=None)
OVERRIDES = {"fp8": None, "fp8_param": False, "moe_router_fusion": False}


def bigram_logits(input_ids: list[int]) -> torch.Tensor:
    """The logits of the table for ``input_ids``, ``[positions, vocabulary]``. Row ``j`` depends on input ``j``."""
    return TABLE[torch.tensor(input_ids)]


def nll(inputs: list[int], targets: list[int]) -> float:
    """The summed negative log-likelihood of ``targets``, each after the input at its position."""
    return -sum(float(LOG_P[a, b]) for a, b in zip(inputs, targets, strict=True))


def write_set(
    directory: Path, entries: list[dict], prefix_id: int | None = PREFIX, tokenizer: str = TOKENIZER
) -> Path:
    """A held-out set of ``entries`` (``id``, ``family``, ``domain``, ``ids``, ``bytes`` and, for a prompt with its
    answer, ``context``) in ``directory``: the index, the token file and the manifest."""
    index, ids = [], []
    for entry in entries:
        stored = {key: value for key, value in entry.items() if key != "ids"}
        index.append(stored | {"offset": len(ids), "targets": len(entry["ids"])})
        ids.extend(entry["ids"])
    data = np.array(ids, dtype="<i4").tobytes()
    recorded = {"targets_sha256": hashlib.sha256(data).hexdigest()}
    if prefix_id is not None:
        recorded["prefix_id"] = prefix_id
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"index.{tokenizer}.json").write_text(json.dumps(index))
    (directory / f"targets.{tokenizer}.int32.bin").write_bytes(data)
    manifest = {"eval_sha256": "0" * 64, "documents": len(index), "tokenizers": {tokenizer: recorded}}
    (directory / "manifest.json").write_text(json.dumps(manifest))
    return directory


def document(name: str, ids: list[int], family: str = "web", domain: str = "wiki", **more) -> dict:
    """One entry of ``write_set``. Its byte count is twice its scored ids, so the two means of a group differ."""
    scored = len(ids) - more.get("context", 0)
    return {"id": name, "family": family, "domain": domain, "ids": ids, "bytes": 2 * scored} | more


def write_spec(directory: Path, eval_dir: Path, **changes) -> Path:
    """A spec file in ``directory`` for the set in ``eval_dir``, with ``changes`` to its keys. A change of a
    ``megatron_reference`` key goes in a mapping under that name."""
    reference = {"expert_model_parallel_size": None, "fp32_ssm_state": False, "model_overrides": dict(OVERRIDES)}
    content = {
        "eval_dir": str(eval_dir),
        "tokenizer_name": TOKENIZER,
        "prefix_id": None,
        "max_window": MAX_WINDOW,
        "limit": None,
        "precision": "bf16",
        "pad_multiple": 1,
        "megatron_reference": reference | changes.pop("megatron_reference", {}),
    }
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "spec.yaml"
    path.write_text(yaml.safe_dump(content | changes))
    return path


def scores(directory: Path, world: int = 1, limit: int | None = None) -> list[float]:
    """Each entry's sum in index order, from ``world`` ranks that run one after the other."""
    index, targets, _ = score_windows.read_eval_set(directory, TOKENIZER, limit, MAX_WINDOW)
    order = score_windows.longest_first(index)
    parts = [
        score_windows.score_items(bigram_logits, SPEC, index, targets, order, world, rank) for rank in range(world)
    ]
    return score_windows.collect(parts, order)


def test_a_document_is_scored_after_the_prefix_id_and_each_id_is_a_target(tmp_path):
    ids = [3, 1, 4, 1, 5, 9, 2, 6]
    [total] = scores(write_set(tmp_path, [document("a", ids)]))
    assert total == pytest.approx(nll([PREFIX] + ids[:-1], ids), abs=1e-4)


def test_a_prompt_with_its_answer_gets_no_prefix_and_only_the_answer_is_scored(tmp_path):
    ids, context = [3, 1, 4, 1, 5, 9, 2, 6], 5
    [total] = scores(write_set(tmp_path, [document("a", ids, context=context)]))
    assert total == pytest.approx(nll(ids[context - 1 : -1], ids[context:]), abs=1e-4)
    # The two wrong readings of the entry, which the table tells apart from the right one.
    assert total != pytest.approx(nll([PREFIX] + ids[:-1], ids), abs=1e-4)
    assert total != pytest.approx(nll(ids[context:-1], ids[context + 1 :]), abs=1e-4)


def test_an_answer_of_one_id_after_a_prompt_of_one_id_is_scored(tmp_path):
    [total] = scores(write_set(tmp_path, [document("a", [7, 2], context=1)]))
    assert total == pytest.approx(nll([7], [2]), abs=1e-5)


def test_documents_and_prompts_of_one_set_keep_their_own_scores_in_index_order(tmp_path):
    entries = [
        document("short", [5, 5]),
        document("task", [1, 2, 3, 4, 5, 6], family="qa", domain="arc", context=4),
        document("long", [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]),
    ]
    expected = [
        nll([PREFIX, 5], [5, 5]),
        nll([4, 5], [5, 6]),
        nll([PREFIX, 9, 8, 7, 6, 5, 4, 3, 2, 1], entries[2]["ids"]),
    ]
    directory = write_set(tmp_path, entries)
    assert scores(directory) == pytest.approx(expected, abs=1e-4)
    assert scores(directory, limit=2) == pytest.approx(expected[:2], abs=1e-4)


@pytest.mark.parametrize("world", [2, 3, 8])
def test_the_ranks_together_score_each_window_one_time(tmp_path, world):
    entries = [document(f"d{n}", [(n * k) % PREFIX for k in range(1, n + 3)]) for n in range(7)]
    directory = write_set(tmp_path, entries)
    assert scores(directory, world) == pytest.approx(scores(directory), abs=1e-6)
    kept = score_windows.kept_items(len(entries), world)
    assert sorted(item for items in kept for item in items) == list(range(len(entries)))


def test_collect_refuses_a_window_that_two_ranks_scored():
    with pytest.raises(RuntimeError, match="3 results were gathered for 2 windows"):
        score_windows.collect([{0: 1.0, 1: 2.0}, {1: 2.0}], [0, 1])


def test_padding_on_the_right_changes_no_score(tmp_path):
    seen = []

    def logits(input_ids: list[int]) -> torch.Tensor:
        seen.append(len(input_ids))
        return bigram_logits(input_ids)

    index, targets, _ = score_windows.read_eval_set(
        write_set(tmp_path, [document("a", [3, 1, 4, 1, 5])]), TOKENIZER, None, MAX_WINDOW
    )
    padded = score_windows.right_padded(logits, 8, PREFIX)
    plain = score_windows.score_items(bigram_logits, SPEC, index, targets, [0], 1, 0)
    assert score_windows.score_items(padded, SPEC, index, targets, [0], 1, 0) == plain
    assert seen == [8]


def test_a_sum_that_is_not_finite_stops_the_scoring(tmp_path):
    directory = write_set(tmp_path, [document("a", [3, 1])])
    index, targets, _ = score_windows.read_eval_set(directory, TOKENIZER, None, MAX_WINDOW)
    with pytest.raises(RuntimeError, match="document a"):
        score_windows.score_items(lambda ids: bigram_logits(ids) * float("nan"), SPEC, index, targets, [0], 1, 0)


@pytest.mark.parametrize(("given", "recorded", "expected"), [(7, 5, 7), (None, 5, 5), (0, 5, 0), (7, None, 7)])
def test_the_prefix_id_is_the_value_of_the_spec_or_the_value_of_the_manifest(given, recorded, expected):
    manifest = {"tokenizers": {"toy": {} if recorded is None else {"prefix_id": recorded}}}
    assert score_windows.window_prefix(manifest, "toy", given) == expected


def test_a_tokenizer_with_no_stated_prefix_id_is_refused():
    with pytest.raises(ValueError, match="state prefix_id in the spec"):
        score_windows.window_prefix({"tokenizers": {"dolma2": {}}}, "dolma2", None)
    with pytest.raises(KeyError):
        score_windows.window_prefix({"tokenizers": {}}, "toy", None)


def test_a_set_whose_files_are_not_of_one_build_is_refused(tmp_path):
    entries = [document("a", [1, 2, 3]), document("b", [4, 5])]
    good = write_set(tmp_path / "good", entries)
    index, targets, manifest = score_windows.read_eval_set(good, TOKENIZER, None, MAX_WINDOW)
    assert ([d["id"] for d in index], targets.tolist(), manifest["documents"]) == (["a", "b"], [1, 2, 3, 4, 5], 2)
    assert [d["id"] for d in score_windows.read_eval_set(good, TOKENIZER, 1, MAX_WINDOW)[0]] == ["a"]

    other_tokens = write_set(tmp_path / "tokens", entries)
    (other_tokens / f"targets.{TOKENIZER}.int32.bin").write_bytes(np.array([1, 2, 3, 4, 6], dtype="<i4").tobytes())
    gap = write_set(tmp_path / "gap", entries)
    moved = json.loads((gap / f"index.{TOKENIZER}.json").read_text())
    moved[1]["offset"] = 4
    (gap / f"index.{TOKENIZER}.json").write_text(json.dumps(moved))
    count = write_set(tmp_path / "count", entries)
    (count / "manifest.json").write_text(
        json.dumps(json.loads((count / "manifest.json").read_text()) | {"documents": 3})
    )
    for broken, message in [(other_tokens, "not of one build"), (gap, "does not follow"), (count, "not of one build")]:
        with pytest.raises(ValueError, match=message):
            score_windows.read_eval_set(broken, TOKENIZER, None, MAX_WINDOW)


def test_a_window_longer_than_max_window_is_refused(tmp_path):
    # A document gets the prefix id in front, so it holds one id fewer than a prompt with its answer.
    score_windows.read_eval_set(
        write_set(tmp_path / "doc", [document("a", [1] * (MAX_WINDOW - 1))]), TOKENIZER, None, MAX_WINDOW
    )
    score_windows.read_eval_set(
        write_set(tmp_path / "task", [document("a", [1] * MAX_WINDOW, context=9)]), TOKENIZER, None, MAX_WINDOW
    )
    with pytest.raises(ValueError, match="too long"):
        score_windows.read_eval_set(
            write_set(tmp_path / "doc2", [document("a", [1] * MAX_WINDOW)]), TOKENIZER, None, MAX_WINDOW
        )
    with pytest.raises(ValueError, match="too long"):
        too_long = [document("a", [1] * (MAX_WINDOW + 1), context=9)]
        score_windows.read_eval_set(write_set(tmp_path / "task2", too_long), TOKENIZER, None, MAX_WINDOW)
    # The same document fits a larger max_window.
    score_windows.read_eval_set(tmp_path / "doc2", TOKENIZER, None, MAX_WINDOW + 1)


@pytest.mark.parametrize("context", [0, 3, 4])
def test_a_context_that_leaves_no_answer_is_refused(tmp_path, context):
    with pytest.raises(ValueError, match="leaves no answer"):
        directory = write_set(tmp_path, [document("a", [1, 2, 3], context=context)])
        score_windows.read_eval_set(directory, TOKENIZER, None, MAX_WINDOW)


def test_the_outputs_hold_each_document_and_both_means_of_each_group(tmp_path):
    entries = [
        document("a", [1, 2, 3, 4]),
        document("b", [5, 6]),
        document("t", [1, 2, 3, 4, 5], family="qa", domain="arc", context=3),
    ]
    index, _, _ = score_windows.read_eval_set(write_set(tmp_path / "set", entries), TOKENIZER, None, MAX_WINDOW)
    values = [8.0 * math.log(2), 2.0 * math.log(2), 3.0 * math.log(2)]
    out = tmp_path / "scores" / "run.jsonl"
    score_windows.write_outputs(out, index, values, {"checkpoint": "c"})

    lines = [json.loads(line) for line in out.read_text().splitlines()]
    assert [(line["id"], line["targets"], line["bytes"]) for line in lines] == [("a", 4, 8), ("b", 2, 4), ("t", 2, 4)]
    assert [line["nll"] for line in lines] == values
    assert score_windows.summary_path(out) == tmp_path / "scores" / "run.summary.json"
    summary = json.loads(score_windows.summary_path(out).read_text())
    assert summary["checkpoint"] == "c"
    assert summary["groups"] == score_windows.group_scores(index, values)
    assert set(summary["groups"]) == {"web", "web/wiki", "qa", "qa/arc"}
    web = summary["groups"]["web"]
    # 10 bits over 12 bytes for the group, and the mean of 8/8 and 2/4 for its documents.
    assert web["bits_per_byte"] == pytest.approx(10 / 12)
    assert web["bits_per_byte_mean"] == pytest.approx(0.75)
    assert (web["targets"], web["bytes"], web["documents"]) == (6, 12, 2)
    assert web["loss_per_token"] == pytest.approx(10 * math.log(2) / 6)
    task = summary["groups"]["qa/arc"]
    assert (task["targets"], task["bytes"], task["documents"]) == (2, 4, 1)
    assert task["bits_per_byte"] == task["bits_per_byte_mean"] == pytest.approx(0.75)
    with pytest.raises(ValueError):
        score_windows.group_scores(index, values[:2])


def test_a_spec_holds_each_key_of_its_file_and_its_digest(tmp_path):
    path = write_spec(tmp_path, tmp_path / "set", prefix_id=2, limit=64, pad_multiple=32)
    spec = score_windows.load_spec(path)
    assert (spec.eval_dir, spec.tokenizer_name, spec.prefix_id) == (tmp_path / "set", TOKENIZER, 2)
    assert (spec.max_window, spec.limit, spec.precision, spec.pad_multiple) == (MAX_WINDOW, 64, "bf16", 32)
    assert (spec.expert_model_parallel_size, spec.fp32_ssm_state, spec.model_overrides) == (None, False, OVERRIDES)
    assert spec.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert spec.content == yaml.safe_load(path.read_text())
    overlay = tmp_path / "train.yaml"
    overlay.write_text("base_config: spec.yaml\nprecision: train\n")
    assert score_windows.load_spec(overlay).precision == "train"
    assert score_windows.load_spec(overlay).pad_multiple == 32


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"ranks": 8}, "the keys of a spec"),
        ({"precision": "fp8"}, "precision must be one of"),
        ({"tokenizer_name": ""}, "tokenizer_name must be a string"),
        ({"prefix_id": -1}, "prefix_id must be a whole number of at least 0"),
        ({"max_window": 1}, "max_window must be a whole number of at least 2"),
        ({"limit": 0}, "limit must be a whole number of at least 1"),
        ({"pad_multiple": None}, "pad_multiple must be a whole number of at least 1"),
        ({"pad_multiple": True}, "pad_multiple must be a whole number of at least 1"),
        ({"megatron_reference": {"fp32_ssm_state": "no"}}, "fp32_ssm_state must be true or false"),
        ({"megatron_reference": {"model_overrides": [1]}}, "model_overrides must map"),
        ({"megatron_reference": {"expert_model_parallel_size": 0}}, "expert_model_parallel_size must be a whole"),
        ({"megatron_reference": {"tensor_model_parallel_size": 1}}, "megatron_reference must have the keys"),
    ],
)
def test_a_spec_with_a_wrong_key_or_a_wrong_value_is_refused(tmp_path, changes, message):
    with pytest.raises(ValueError, match=message):
        score_windows.load_spec(write_spec(tmp_path, tmp_path / "set", **changes))


def test_a_spec_that_lacks_a_key_is_refused(tmp_path):
    path = write_spec(tmp_path, tmp_path / "set")
    content = yaml.safe_load(path.read_text())
    del content["precision"]
    path.write_text(yaml.safe_dump(content))
    with pytest.raises(ValueError, match="the keys of a spec"):
        score_windows.load_spec(path)


def test_the_posture_has_the_overrides_of_the_spec_and_train_keeps_the_fp8_keys_of_the_run(tmp_path):
    bf16 = score_windows.load_spec(write_spec(tmp_path / "bf16", tmp_path))
    posture = score_windows.reference_posture(bf16, {"expert_model_parallel_size": 8})
    assert (posture.tensor_model_parallel_size, posture.expert_model_parallel_size) == (1, 8)
    assert (posture.fp32_ssm_state, posture.model_overrides) == (False, OVERRIDES)

    reference = {"expert_model_parallel_size": 2, "fp32_ssm_state": True}
    train = score_windows.load_spec(
        write_spec(tmp_path / "train", tmp_path, precision="train", megatron_reference=reference)
    )
    posture = score_windows.reference_posture(train, {"expert_model_parallel_size": 8})
    assert (posture.expert_model_parallel_size, posture.fp32_ssm_state) == (2, True)
    assert posture.model_overrides == {"moe_router_fusion": False}
    assert set(OVERRIDES) - set(posture.model_overrides) == set(score_windows.FP8_KEYS)


def test_a_run_config_with_no_expert_parallelism_needs_the_value_in_the_spec(tmp_path):
    spec = score_windows.load_spec(write_spec(tmp_path, tmp_path))
    with pytest.raises(ValueError, match="state megatron_reference.expert_model_parallel_size"):
        score_windows.reference_posture(spec, {})


def test_a_model_with_too_few_output_rows_is_refused():
    score_windows.require_output_rows(VOCABULARY, VOCABULARY - 1, PREFIX, TOKENIZER)
    with pytest.raises(RuntimeError, match="13 output rows"):
        score_windows.require_output_rows(VOCABULARY, VOCABULARY, PREFIX, TOKENIZER)
    with pytest.raises(RuntimeError, match="the prefix id 13"):
        score_windows.require_output_rows(VOCABULARY, 5, VOCABULARY, TOKENIZER)


def make_run(directory: Path, latest: int, *iterations: int) -> Path:
    """A run directory with the tracker file at ``latest`` and one save directory for each of ``iterations``. Each
    save has a run_config that names a tokenizer."""
    directory.mkdir(parents=True)
    (directory / "latest_checkpointed_iteration.txt").write_text(str(latest))
    for iteration in iterations:
        save = directory / f"iter_{iteration:07d}"
        save.mkdir()
        (save / "run_config.yaml").write_text(yaml.safe_dump({"tokenizer": {"tokenizer_model": "toy-tokenizer"}}))
    return directory


def test_a_checkpoint_that_is_not_a_complete_save_is_refused(tmp_path):
    run = make_run(tmp_path / "run", 250, 250, 500)
    score_windows.require_complete(run / "iter_0000250")
    with pytest.raises(ValueError, match="not a complete save"):
        score_windows.require_complete(run / "iter_0000500")
    with pytest.raises(ValueError, match="not a complete save"):
        score_windows.require_complete(run / "iter_0000750")
    with pytest.raises(ValueError, match="not an iter_XXXXXXX directory"):
        score_windows.require_complete(run)


def test_the_plan_prints_the_iteration_and_the_precision_and_refuses_an_output_that_exists(tmp_path, capsys):
    run = make_run(tmp_path / "run", 250, 250, 500)
    spec = write_spec(tmp_path, tmp_path / "set", precision="train")
    out = tmp_path / "scores" / "s.jsonl"
    arguments = ["--spec", str(spec), "--checkpoint", str(run / "iter_0000250"), "--out", str(out), "--plan"]
    assert score_windows.main(arguments) == 0
    assert capsys.readouterr().out == "ITERATION=250\nPRECISION=train\n"

    incomplete = ["--spec", str(spec), "--checkpoint", str(run / "iter_0000500"), "--out", str(out), "--plan"]
    with pytest.raises(ValueError, match="not a complete save"):
        score_windows.main(incomplete)
    out.parent.mkdir()
    score_windows.summary_path(out).write_text("{}")
    with pytest.raises(FileExistsError):
        score_windows.main(arguments)
    assert capsys.readouterr().out == ""


def test_the_summary_records_the_spec_the_set_and_the_checkpoint(tmp_path):
    run = make_run(tmp_path / "run", 250, 250)
    spec = score_windows.load_spec(write_spec(tmp_path, tmp_path / "set", limit=64))
    checkpoint = run / "iter_0000250"
    summary = score_windows.run_summary(spec, checkpoint, checkpoint, {"eval_sha256": "e" * 64}, 2, {"seconds": 1.5})
    assert summary["spec"] == {"path": spec.path, "sha256": spec.sha256, "content": spec.content}
    assert (summary["eval_dir"], summary["eval_sha256"]) == (str((tmp_path / "set").resolve()), "e" * 64)
    assert (summary["checkpoint"], summary["load_path"]) == (str(checkpoint.resolve()),) * 2
    assert (summary["iteration"], summary["checkpoint_tokenizer"]) == (250, "toy-tokenizer")
    assert (summary["tokenizer"], summary["prefix_id"], summary["limit"]) == (TOKENIZER, 2, 64)
    assert (summary["precision"], summary["pad_multiple"], summary["fp32_ssm_state"]) == ("bf16", 1, False)
    assert summary["seconds"] == 1.5
    assert summary["code_revision"]


def test_the_dry_run_counts_the_scored_ids_and_builds_no_model(tmp_path, caplog):
    entries = [document("a", [1, 2, 3, 4]), document("t", [1, 2, 3, 4, 5], family="qa", domain="arc", context=3)]
    spec = write_spec(tmp_path / "spec", write_set(tmp_path / "set", entries))
    with caplog.at_level(logging.INFO, logger="score_windows"):
        assert score_windows.main(["--spec", str(spec), "--dry-run", "--ranks", "2"]) == 0
    assert "2 documents {'web': 1, 'qa': 1}, 6 target ids, 12 bytes" in caplog.text
    # The document with its prefix id, and the prompt with its answer as it is.
    assert "min 5, median 5, max 5" in caplog.text
    assert f"prefix id {PREFIX}" in caplog.text
    assert "2 ranks: 1 steps, 0 forwards repeated and dropped" in caplog.text


def test_the_arguments_of_a_scoring_run_are_parsed_with_its_spec(tmp_path):
    spec = str(write_spec(tmp_path, tmp_path / "set", prefix_id=2))
    args = score_windows.parse_args(["--spec", spec, "--checkpoint", "run/iter_0000250", "--out", "s.jsonl"])
    assert (args.spec.prefix_id, args.spec.tokenizer_name, args.spec.precision) == (2, TOKENIZER, "bf16")
    assert (args.checkpoint, args.out, args.load_path) == (Path("run/iter_0000250"), Path("s.jsonl"), None)
    assert (args.plan, args.dry_run) == (False, False)
    dry = score_windows.parse_args(["--spec", spec, "--dry-run", "--ranks", "4"])
    assert (dry.dry_run, dry.ranks) == (True, 4)


@pytest.mark.parametrize(
    "arguments",
    [
        [],
        ["--checkpoint", "run/iter_0000250"],
        ["--checkpoint", "run/iter_0000250", "--out", "scores.txt"],
        ["--checkpoint", "run/latest", "--out", "s.jsonl"],
        ["--checkpoint", "run/iter_0000250", "--out", "s.jsonl", "--ranks", "8"],
        ["--dry-run"],
        ["--dry-run", "--ranks", "0"],
        ["--dry-run", "--ranks", "8", "--plan"],
    ],
)
def test_arguments_that_do_not_fit_the_mode_are_refused(tmp_path, arguments):
    spec = str(write_spec(tmp_path, tmp_path / "set"))
    with pytest.raises(SystemExit):
        score_windows.parse_args(["--spec", spec, *arguments])


def test_a_train_spec_takes_no_load_path_and_a_wrong_spec_is_a_usage_error(tmp_path):
    scoring = ["--checkpoint", "run/iter_0000250", "--out", "s.jsonl"]
    train = str(write_spec(tmp_path / "train", tmp_path / "set", precision="train"))
    assert score_windows.parse_args(["--spec", train, *scoring]).spec.precision == "train"
    with pytest.raises(SystemExit):
        score_windows.parse_args(["--spec", train, *scoring, "--load-path", "clone"])
    wrong = str(write_spec(tmp_path / "wrong", tmp_path / "set", limit=0))
    with pytest.raises(SystemExit):
        score_windows.parse_args(["--spec", wrong, *scoring])


def test_the_spec_of_the_study_is_a_spec_the_scorer_reads():
    spec = score_windows.load_spec(
        Path(__file__).resolve().parents[2] / "configs/olmo_mix_efficiency/checks/score_documents_v1_dolma2.yaml"
    )
    assert (spec.tokenizer_name, spec.prefix_id, spec.max_window, spec.limit) == ("dolma2", 100257, 8192, None)
    assert (spec.precision, spec.pad_multiple, spec.fp32_ssm_state) == ("bf16", 1, False)
    assert set(score_windows.FP8_KEYS) <= set(spec.model_overrides)
