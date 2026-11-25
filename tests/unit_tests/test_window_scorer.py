# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""scripts/checkpoint/window_scorer.py on the CPU: the config and its pins, the windows of a corpus and of an array, the
loss of each window, whole passes, and the command line in a subprocess, as the job runs it.

The corpus is a real ``.bin/.idx`` pair that Megatron's ``IndexedDatasetBuilder`` wrote, read through the launcher's own
dataset config and the trainer's dataset class, with the run config of ``fixtures/training_run``. The model is the
repo's copy model, a real tiny Llama whose logit of the input token is ``COPY_LOGIT`` and of every other token 0, so a
target equal to its input costs ~0 nats and any other target ~``COPY_LOGIT``.
"""

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import yaml
from scripts.checkpoint import window_scorer
from scripts.checkpoint.window_scorer import WindowScorerError

from tests.unit_tests.corpora_fixtures import write_indexed_documents
from tests.unit_tests.tiny_models import COPY_LOGIT, copy_model
from tests.unit_tests.token_masking_fixtures import TINY_VOCAB, build_tiny_hf_tokenizer


SEQUENCE = 8
REPO = Path(window_scorer.__file__).resolve().parents[2]
RESOLVED_RUN_CONFIG = REPO / "tests/unit_tests/fixtures/training_run/resolved/nano_pretrain.yaml"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_config(tokenizer_dir: Path, **dataset) -> dict:
    """The resolved run config of the repo's fixture, with the sequence length of these tests, the given dataset
    settings and the tiny tokenizer."""
    config = yaml.safe_load(RESOLVED_RUN_CONFIG.read_text())
    config["dataset"] |= {"sequence_length": SEQUENCE} | dataset
    config["tokenizer"]["tokenizer_model"] = str(tokenizer_dir)
    return config


@pytest.fixture(scope="module")
def tokenizer_dir(tmp_path_factory):
    return build_tiny_hf_tokenizer(tmp_path_factory.mktemp("tokenizer"))


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    """Five documents of 20 ids drawn from the tiny tokenizer's 8 ids: 100 ids, so 12 windows of 8."""
    draw = np.random.RandomState(0)
    documents = [draw.randint(0, len(TINY_VOCAB) + 1, size=20).tolist() for _ in range(5)]
    return write_indexed_documents(tmp_path_factory.mktemp("corpus") / "tokenized_text_document", documents), documents


def test_a_corpus_gives_every_window_once_with_labels_one_id_ahead(corpus, tokenizer_dir, tmp_path):
    prefix, documents = corpus
    inputs, labels = window_scorer.corpus_windows(prefix, run_config(tokenizer_dir), tmp_path / "cache")
    assert inputs.shape == labels.shape == (12, SEQUENCE)  # (100 - 1) // 8 windows
    assert (labels[:, :-1] == inputs[:, 1:]).all()
    assert len({row.tobytes() for row in inputs}) == 12
    stream = np.concatenate(documents)
    assert set(np.unique(inputs)) <= set(np.unique(stream))


def test_the_windows_follow_the_seed_and_repeat_for_one_seed(corpus, tokenizer_dir, tmp_path):
    prefix, _ = corpus
    first, _ = window_scorer.corpus_windows(prefix, run_config(tokenizer_dir), tmp_path / "cache_1234")
    again, _ = window_scorer.corpus_windows(prefix, run_config(tokenizer_dir), tmp_path / "cache_again")
    other, _ = window_scorer.corpus_windows(prefix, run_config(tokenizer_dir, random_seed=7), tmp_path / "cache_7")
    assert (first == again).all()
    assert not (first == other).all()


def test_a_loader_setting_that_differs_from_the_runs_is_refused(corpus, tokenizer_dir, tmp_path):
    prefix, _ = corpus
    with pytest.raises(WindowScorerError, match="eod_mask_loss"):
        window_scorer.corpus_windows(prefix, run_config(tokenizer_dir, eod_mask_loss=True), tmp_path / "cache")


def test_a_corpus_too_short_for_one_window_is_refused(tokenizer_dir, tmp_path):
    short = write_indexed_documents(tmp_path / "short", [[1, 2, 3, 4, 5]])
    with pytest.raises(WindowScorerError, match="too few tokens"):
        window_scorer.corpus_windows(short, run_config(tokenizer_dir), tmp_path / "cache")


def test_a_corpus_in_file_order_cuts_its_documents_one_after_another(corpus, tokenizer_dir):
    prefix, documents = corpus
    inputs, labels = window_scorer.file_order_windows(prefix, run_config(tokenizer_dir))
    stream = np.concatenate(documents)
    assert inputs.tolist() == stream[: 12 * SEQUENCE].reshape(12, SEQUENCE).tolist()
    assert labels.tolist() == stream[1 : 12 * SEQUENCE + 1].reshape(12, SEQUENCE).tolist()


def test_a_corpus_too_short_for_one_window_is_refused_in_file_order_too(tokenizer_dir, tmp_path):
    short = write_indexed_documents(tmp_path / "short", [[1, 2, 3, 4, 5]])
    with pytest.raises(WindowScorerError, match="too few tokens"):
        window_scorer.file_order_windows(short, run_config(tokenizer_dir))


def test_an_array_gives_the_first_ids_as_inputs_and_the_ids_after_the_first_as_targets(tmp_path):
    ids = np.arange(12, dtype=np.int32).reshape(3, 4)
    np.save(tmp_path / "ids.npy", ids)
    inputs, targets = window_scorer.array_windows(tmp_path / "ids.npy")
    assert inputs.tolist() == [[0, 1, 2], [4, 5, 6], [8, 9, 10]]
    assert targets.tolist() == [[1, 2, 3], [5, 6, 7], [9, 10, 11]]


def test_an_array_that_is_not_windows_of_ids_is_refused(tmp_path):
    np.save(tmp_path / "flat.npy", np.arange(8, dtype=np.int32))
    with pytest.raises(WindowScorerError, match="windows, ids"):
        window_scorer.array_windows(tmp_path / "flat.npy")


def copy_logits(model):
    from scripts.checkpoint import hf_model_outputs

    return lambda ids: hf_model_outputs.fp32_head_outputs(model, ids)


def test_the_loss_of_a_window_is_the_sum_of_its_targets_costs():
    inputs = np.array([[3, 3, 3, 3], [3, 4, 5, 6]])
    targets = np.array([[3, 3, 3, 3], [4, 5, 6, 3]])
    losses = window_scorer.window_losses(copy_logits(copy_model()), inputs, targets, stop_at=time.monotonic() + 600)
    assert losses[0] == pytest.approx(0.0, abs=1e-3)
    assert losses[1] == pytest.approx(4 * COPY_LOGIT, abs=1e-2)


def test_a_pass_that_runs_past_its_limit_is_refused():
    inputs, targets = np.array([[3, 4]]), np.array([[4, 5]])
    with pytest.raises(WindowScorerError, match="after 0 of 1 windows"):
        window_scorer.window_losses(copy_logits(copy_model()), inputs, targets, stop_at=time.monotonic() - 1)


def write_config(tmp_path: Path, corpus_prefix: Path | None = None, tokenizer_dir: Path | None = None, **overrides):
    """A scorer config over a saved copy-model export, an array set (and a corpus set when ``corpus_prefix`` is given),
    with a real file behind every pin. Returns the config's path."""
    export = tmp_path / "export"
    copy_model().save_pretrained(export)
    (export / window_scorer.EXPORT_MANIFEST).write_text(json.dumps({"run": "copy"}))
    np.save(tmp_path / "ids.npy", np.array([[3, 3, 3, 3, 3], [3, 4, 5, 6, 3]], dtype=np.int32))
    sets = {"ids": {"kind": "array", "path": str(tmp_path / "ids.npy"), "sha256": sha256(tmp_path / "ids.npy")}}
    passes = {"copy_on_ids": {"model": "copy", "set": "ids"}}
    if corpus_prefix is not None:
        sets["corpus"] = {
            "kind": "corpus",
            "prefix": str(corpus_prefix),
            "order": "shuffled",
            "bin_sha256": sha256(Path(f"{corpus_prefix}.bin")),
            "idx_sha256": sha256(Path(f"{corpus_prefix}.idx")),
        }
        passes["copy_on_corpus"] = {"model": "copy", "set": "corpus"}
    run_config_path = tmp_path / "run_config.yaml"
    run_config_path.write_text(yaml.safe_dump(run_config(tokenizer_dir) if tokenizer_dir else {}))
    models = {
        "copy": {
            "export": str(export),
            "export_manifest_sha256": sha256(export / window_scorer.EXPORT_MANIFEST),
            "run_config": str(run_config_path),
        }
    }
    config = {
        "output_dir": str(tmp_path / "scores"),
        "hf_dtype": "float32",
        "max_pass_seconds": 600,
        "models": models,
        "sets": sets,
        "passes": passes,
    } | overrides
    path = tmp_path / "scorer.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def test_a_config_names_each_pass_by_its_model_and_its_set(tmp_path):
    config = window_scorer.load_config(write_config(tmp_path))
    assert config.passes["copy_on_ids"] == window_scorer.Pass(model="copy", set="ids")
    assert config.sets["ids"].kind == "array"
    assert config.sha256 == sha256(tmp_path / "scorer.yaml")
    assert config.path == tmp_path / "scorer.yaml"


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"passes": {"p": {"model": "missing", "set": "ids"}}}, "names a model or a set"),
        (
            {"models": {"copy": {"export": "/e", "export_manifest_sha256": "abc", "run_config": "/r"}}},
            "64 lowercase hex digits",
        ),
        ({"max_pass_seconds": 0}, "max_pass_seconds"),
        ({"sets": {"ids": {"kind": "text", "path": "/p"}}}, "corpus or array"),
        (
            {"sets": {"c": {"kind": "corpus", "prefix": "/p", "bin_sha256": "0" * 64, "idx_sha256": "0" * 64}}},
            "missing keys",
        ),
        (
            {
                "sets": {
                    "c": {
                        "kind": "corpus",
                        "prefix": "/p",
                        "order": "sorted",
                        "bin_sha256": "0" * 64,
                        "idx_sha256": "0" * 64,
                    }
                }
            },
            "one of shuffled, file",
        ),
        ({"passes": {"p": {"model": "copy"}}}, "missing keys"),
        ({"output_dir": "relative/scores"}, "absolute path"),
    ],
)
def test_a_config_that_is_not_complete_or_pinned_in_full_is_refused(tmp_path, overrides, message):
    with pytest.raises(WindowScorerError, match=message):
        window_scorer.load_config(write_config(tmp_path, **overrides))


def test_the_shipped_config_defines_the_passes_that_the_documented_command_names():
    config = window_scorer.load_config(REPO / "configs/ipt_spt/heldout_scoring.yaml")
    ipt_passes = ("ipt_on_heldout_ipt", "ipt_on_heldout_ipt_file_order", "ipt_on_train_ipt", "ipt_on_heldout_spt")
    assert {config.passes[name].model for name in ipt_passes} == {"ipt"}
    assert {
        config.passes[name].model for name in ("spt_on_heldout_spt", "spt_on_train_spt", "spt_on_heldout_ipt")
    } == {"spt"}
    assert {one.set for one in config.passes.values()} == {
        "heldout_ipt",
        "heldout_ipt_file_order",
        "heldout_spt",
        "train_ipt",
        "train_spt",
    }
    assert {window_set.kind for window_set in config.sets.values()} == {"corpus", "array"}
    shuffled, file_order = config.sets["heldout_ipt"], config.sets["heldout_ipt_file_order"]
    assert (shuffled.order, file_order.order) == ("shuffled", "file")
    assert (shuffled.location, shuffled.sha256s) == (file_order.location, file_order.sha256s)


def test_a_pass_writes_the_loss_of_each_window_and_a_summary_with_its_hashes_and_its_config(tmp_path):
    config = window_scorer.load_config(write_config(tmp_path))
    summary = window_scorer.run_pass(config, "copy_on_ids", copy_model(), REPO)
    rows = [line.split("\t") for line in (config.output_dir / "copy_on_ids.windows.tsv").read_text().splitlines()]
    assert [row[:4] for row in rows] == [["ids", "copy", "0", "4"], ["ids", "copy", "1", "4"]]
    assert float(rows[0][4]) == pytest.approx(0.0, abs=1e-3)
    assert float(rows[1][4]) == pytest.approx(4 * COPY_LOGIT, abs=1e-2)
    assert summary["windows"] == 2 and summary["targets"] == 8
    assert summary["mean_loss_per_target"] == pytest.approx(COPY_LOGIT / 2, abs=1e-2)
    assert summary == json.loads((config.output_dir / "copy_on_ids.summary.json").read_text())
    assert summary["sha256"]["array"] == sha256(tmp_path / "ids.npy")
    assert summary["sha256"]["scorer_config"] == config.sha256
    assert summary["config"]["path"] == str(tmp_path / "scorer.yaml")
    assert summary["config"]["export"] == str(tmp_path / "export")
    assert summary["config"]["set_location"] == str(tmp_path / "ids.npy")
    assert {"torch", "transformers", "code_revision"} <= set(summary)


def test_a_pass_over_a_corpus_pins_both_files_and_records_the_run_config(corpus, tokenizer_dir, tmp_path):
    prefix, _ = corpus
    config = window_scorer.load_config(write_config(tmp_path, corpus_prefix=prefix, tokenizer_dir=tokenizer_dir))
    summary = window_scorer.run_pass(config, "copy_on_corpus", copy_model(), REPO)
    assert summary["windows"] == 12 and summary["targets"] == 12 * SEQUENCE
    assert set(summary["sha256"]) == {"export_manifest", "bin", "idx", "run_config", "scorer_config"}
    assert summary["sha256"]["bin"] == sha256(Path(f"{prefix}.bin"))
    assert summary["sha256"]["run_config"] == sha256(tmp_path / "run_config.yaml")
    assert (config.output_dir / "index_cache" / "corpus").is_dir()
    assert len((config.output_dir / "copy_on_corpus.windows.tsv").read_text().splitlines()) == 12
    assert summary["config"]["set_order"] == "shuffled"


def test_a_pass_over_a_corpus_in_file_order_scores_the_documents_in_sequence(corpus, tokenizer_dir, tmp_path):
    prefix, documents = corpus
    path = write_config(tmp_path, corpus_prefix=prefix, tokenizer_dir=tokenizer_dir)
    raw = yaml.safe_load(path.read_text())
    raw["sets"]["corpus"]["order"] = "file"
    path.write_text(yaml.safe_dump(raw))
    config = window_scorer.load_config(path)
    summary = window_scorer.run_pass(config, "copy_on_corpus", copy_model(), REPO)
    stream = np.concatenate(documents)
    inputs = stream[: 12 * SEQUENCE].reshape(12, SEQUENCE)
    labels = stream[1 : 12 * SEQUENCE + 1].reshape(12, SEQUENCE)
    rows = [line.split("\t") for line in (config.output_dir / "copy_on_corpus.windows.tsv").read_text().splitlines()]
    losses = [float(row[4]) for row in rows]
    assert losses == pytest.approx(((inputs != labels).sum(axis=1) * COPY_LOGIT).tolist(), abs=0.1)
    assert summary["windows"] == 12 and summary["config"]["set_order"] == "file"
    assert not (config.output_dir / "index_cache").exists()


def test_a_pass_never_replaces_its_outputs(tmp_path):
    config = window_scorer.load_config(write_config(tmp_path))
    window_scorer.run_pass(config, "copy_on_ids", copy_model(), REPO)
    with pytest.raises(FileExistsError, match="never replaces"):
        window_scorer.run_pass(config, "copy_on_ids", copy_model(), REPO)


def test_a_file_that_is_not_the_pinned_one_is_refused_before_anything_is_scored(corpus, tokenizer_dir, tmp_path):
    prefix, _ = corpus
    path = write_config(tmp_path, corpus_prefix=prefix, tokenizer_dir=tokenizer_dir)
    config = window_scorer.load_config(path)
    (tmp_path / "export" / window_scorer.EXPORT_MANIFEST).write_text(json.dumps({"run": "another export"}))
    with pytest.raises(WindowScorerError, match="the export manifest"):
        window_scorer.run_pass(config, "copy_on_corpus", copy_model(), REPO)
    raw = yaml.safe_load(path.read_text())
    raw["models"]["copy"]["export_manifest_sha256"] = sha256(tmp_path / "export" / window_scorer.EXPORT_MANIFEST)
    raw["sets"]["corpus"]["bin_sha256"] = "0" * 64
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(WindowScorerError, match="the corpus bin"):
        window_scorer.run_pass(window_scorer.load_config(path), "copy_on_corpus", copy_model(), REPO)
    assert not (tmp_path / "scores").exists()


@pytest.mark.parametrize(
    "passes, message", [("nope", "defined in the config"), ("copy_on_ids,other_on_ids", "share one model")]
)
def test_passes_that_are_not_defined_or_not_of_one_model_are_refused_before_a_model_loads(tmp_path, passes, message):
    path = write_config(tmp_path)
    raw = yaml.safe_load(path.read_text())
    raw["models"]["other"] = raw["models"]["copy"]
    raw["passes"]["other_on_ids"] = {"model": "other", "set": "ids"}
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(WindowScorerError, match=message):
        window_scorer.main(["--config", str(path), "--passes", passes])


def test_the_command_line_scores_a_saved_export_when_run_as_a_script(tmp_path):
    path = write_config(tmp_path)
    script = REPO / "scripts/checkpoint/window_scorer.py"
    command = [sys.executable, str(script), "--config", str(path), "--passes", "copy_on_ids"]
    done = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, check=False)
    assert done.returncode == 0, done.stderr[-2000:]
    summary = json.loads((tmp_path / "scores" / "copy_on_ids.summary.json").read_text())
    assert summary["mean_loss_per_target"] == pytest.approx(COPY_LOGIT / 2, abs=1e-2)
