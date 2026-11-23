# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""scripts/checkpoint/score_windows.py: the summed negative log-likelihood of each document of a held-out set.

The model is replaced by a bigram table: the logits at a position depend only on the input id at that position. The
cross-entropy of a target after an input is then one entry of the table, so each expected sum is a sum of entries
and a test fails when an input and its target are one position apart from where they must be.

- A document is scored after the prefix id, and each of its ids is a target.
- A prompt with its answer (an index entry with ``context``) gets no prefix id, and only the ids of the answer are
  targets. OLMES and lm-eval score a continuation in this way.
- The prefix id is the option, else the one that the manifest records for the tokenizer, else the dolma2 id.
- The set is refused when its three files are not of one build.
- The ranks together score each window one time.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
from pathlib import Path

import numpy as np
import pytest
import torch
from scripts.checkpoint import score_windows
from scripts.checkpoint.score_windows import WindowSpec


VOCABULARY = 13
PREFIX = 12
TOKENIZER = "toy"
TABLE = torch.tensor(
    np.random.default_rng(20261123).normal(scale=2.0, size=(VOCABULARY, VOCABULARY)), dtype=torch.float32
)
LOG_P = torch.log_softmax(TABLE.double(), dim=-1)
# The posture of a real model, which a bigram table does not have.
SPEC = WindowSpec(prefix_token_ids=(PREFIX,), hidden_token_id=PREFIX, megatron_reference=None)


def bigram_logits(input_ids: list[int]) -> torch.Tensor:
    """The logits of the table for ``input_ids``, ``[positions, vocabulary]``: row ``j`` depends on input ``j`` alone."""
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


def scores(directory: Path, world: int = 1) -> list[float]:
    """Each entry's sum in index order, from ``world`` ranks that run one after the other."""
    index, targets, _ = score_windows.read_eval_set(directory, TOKENIZER, None)
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
    assert scores(write_set(tmp_path, entries)) == pytest.approx(expected, abs=1e-4)


@pytest.mark.parametrize("world", [2, 3, 8])
def test_the_ranks_together_score_each_window_one_time(tmp_path, world):
    entries = [document(f"d{n}", [(n * k) % PREFIX for k in range(1, n + 3)]) for n in range(7)]
    directory = write_set(tmp_path, entries)
    assert scores(directory, world) == pytest.approx(scores(directory), abs=1e-6)


def test_collect_refuses_a_window_that_two_ranks_scored():
    with pytest.raises(RuntimeError, match="3 results were gathered for 2 windows"):
        score_windows.collect([{0: 1.0, 1: 2.0}, {1: 2.0}], [0, 1])


def test_padding_on_the_right_changes_no_score(tmp_path):
    seen = []

    def logits(input_ids: list[int]) -> torch.Tensor:
        seen.append(len(input_ids))
        return bigram_logits(input_ids)

    index, targets, _ = score_windows.read_eval_set(
        write_set(tmp_path, [document("a", [3, 1, 4, 1, 5])]), TOKENIZER, None
    )
    padded = score_windows.right_padded(logits, 8, PREFIX)
    plain = score_windows.score_items(bigram_logits, SPEC, index, targets, [0], 1, 0)
    assert score_windows.score_items(padded, SPEC, index, targets, [0], 1, 0) == plain
    assert seen == [8]


def test_a_sum_that_is_not_finite_stops_the_scoring(tmp_path):
    index, targets, _ = score_windows.read_eval_set(write_set(tmp_path, [document("a", [3, 1])]), TOKENIZER, None)
    with pytest.raises(RuntimeError, match="document a"):
        score_windows.score_items(lambda ids: bigram_logits(ids) * float("nan"), SPEC, index, targets, [0], 1, 0)


@pytest.mark.parametrize(
    ("given", "recorded", "tokenizer", "expected"),
    [
        (7, 5, "toy", 7),
        (None, 5, "toy", 5),
        (None, None, "dolma2", score_windows.DOLMA2_PREFIX_ID),
        (None, 2, "dolma2", 2),
        (0, 5, "toy", 0),
    ],
)
def test_the_prefix_id_is_the_option_then_the_manifests_then_the_dolma2_id(given, recorded, tokenizer, expected):
    manifest = {"tokenizers": {tokenizer: {} if recorded is None else {"prefix_id": recorded}}}
    assert score_windows.window_prefix(manifest, tokenizer, given) == expected


def test_a_tokenizer_with_no_stated_prefix_id_is_refused():
    with pytest.raises(ValueError, match="give --prefix-id"):
        score_windows.window_prefix({"tokenizers": {"toy": {}}}, "toy", None)
    with pytest.raises(ValueError, match="negative"):
        score_windows.window_prefix({"tokenizers": {}}, "toy", -1)


def test_a_set_whose_files_are_not_of_one_build_is_refused(tmp_path):
    entries = [document("a", [1, 2, 3]), document("b", [4, 5])]
    good = write_set(tmp_path / "good", entries)
    index, targets, manifest = score_windows.read_eval_set(good, TOKENIZER, None)
    assert ([d["id"] for d in index], targets.tolist(), manifest["documents"]) == (["a", "b"], [1, 2, 3, 4, 5], 2)
    assert [d["id"] for d in score_windows.read_eval_set(good, TOKENIZER, 1)[0]] == ["a"]

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
            score_windows.read_eval_set(broken, TOKENIZER, None)


def test_a_window_longer_than_the_training_sequence_is_refused(tmp_path):
    longest = score_windows.MAX_WINDOW
    # A document gets the prefix id in front, so it holds one id fewer than a prompt with its answer.
    score_windows.read_eval_set(write_set(tmp_path / "doc", [document("a", [1] * (longest - 1))]), TOKENIZER, None)
    score_windows.read_eval_set(
        write_set(tmp_path / "task", [document("a", [1] * longest, context=9)]), TOKENIZER, None
    )
    with pytest.raises(ValueError, match="too long"):
        score_windows.read_eval_set(write_set(tmp_path / "doc2", [document("a", [1] * longest)]), TOKENIZER, None)
    with pytest.raises(ValueError, match="too long"):
        too_long = [document("a", [1] * (longest + 1), context=9)]
        score_windows.read_eval_set(write_set(tmp_path / "task2", too_long), TOKENIZER, None)


@pytest.mark.parametrize("context", [0, 3, 4])
def test_a_context_that_leaves_no_answer_is_refused(tmp_path, context):
    with pytest.raises(ValueError, match="leaves no answer"):
        score_windows.read_eval_set(write_set(tmp_path, [document("a", [1, 2, 3], context=context)]), TOKENIZER, None)


def test_the_outputs_hold_each_document_and_both_means_of_each_group(tmp_path):
    entries = [
        document("a", [1, 2, 3, 4]),
        document("b", [5, 6]),
        document("t", [1, 2, 3, 4, 5], family="qa", domain="arc", context=3),
    ]
    index, _, _ = score_windows.read_eval_set(write_set(tmp_path / "set", entries), TOKENIZER, None)
    values = [8.0 * math.log(2), 2.0 * math.log(2), 3.0 * math.log(2)]
    out = tmp_path / "scores" / "run.jsonl"
    score_windows.write_outputs(out, index, values, {"checkpoint": "c"})

    lines = [json.loads(line) for line in out.read_text().splitlines()]
    assert [(line["id"], line["targets"], line["bytes"]) for line in lines] == [("a", 4, 8), ("b", 2, 4), ("t", 2, 4)]
    assert [line["nll"] for line in lines] == values
    summary = json.loads(score_windows.summary_path(out).read_text())
    assert summary["checkpoint"] == "c"
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


def test_the_dry_run_counts_the_scored_ids_and_builds_no_model(tmp_path, caplog):
    entries = [document("a", [1, 2, 3, 4]), document("t", [1, 2, 3, 4, 5], family="qa", domain="arc", context=3)]
    directory = write_set(tmp_path, entries)
    args = argparse.Namespace(eval=directory, tokenizer_name=TOKENIZER, limit=None, ranks=2, prefix_id=None)
    with caplog.at_level(logging.INFO, logger="score_windows"):
        score_windows.dry_run(args)
    assert "2 documents {'web': 1, 'qa': 1}, 6 target ids, 12 bytes" in caplog.text
    # The document with its prefix id, and the prompt with its answer as it is.
    assert "min 5, median 5, max 5" in caplog.text
    assert f"prefix id {PREFIX}" in caplog.text


def test_scoring_needs_a_checkpoint_and_a_scores_file(tmp_path):
    with pytest.raises(SystemExit):
        score_windows.parse_args(["--eval", str(tmp_path)])
    with pytest.raises(SystemExit):
        score_windows.parse_args(["--eval", str(tmp_path), "--checkpoint", "run/iter_0000250", "--out", "scores.txt"])
    args = score_windows.parse_args(
        ["--eval", str(tmp_path), "--checkpoint", "run/iter_0000250", "--out", "s.jsonl", "--prefix-id", "2"]
    )
    assert (args.prefix_id, args.tokenizer_name, args.precision) == (2, "dolma2", "bf16")
    assert score_windows.parse_args(["--eval", str(tmp_path), "--dry-run"]).dry_run
