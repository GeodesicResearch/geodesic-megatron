# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""Unit tests for scripts/profiling/ssm_state_probe.py.

The corpora are real ``.bin/.idx`` files written by Megatron's ``IndexedDatasetBuilder`` and planned through the
probe's own ``build_plan``. The committed spec, the Clueless-Norm stage-1 config and its ``.env`` are read and resolved
as the probe reads and resolves them, and the fp32-state patch is the one training installs. The recorders' own
bookkeeping is driven through their interfaces on real tensors on the CPU; one GPU test installs them with
``install_recorders`` on a small real HybridModel and runs its training forward through ``run_windows`` in both
SSM-state modes, so the scan they record is the one a training step runs.
"""

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from scripts.data import sha256_manifest
from scripts.data.corpus_index import token_memmap
from scripts.data.token_runs import longest_runs
from scripts.profiling import ssm_state_probe as probe

from tests.unit_tests.corpora_fixtures import write_indexed_documents


REPO_ROOT = Path(__file__).resolve().parents[2]
ARM_DIR = REPO_ROOT / "configs" / "metagaming_filtering" / "30b_clueless_norm"
SPEC_PATH = ARM_DIR / "probe" / "m5b_ssm_state.yaml"
STAGE_CONFIG = ARM_DIR / "nemotron_nano_30b_metagaming_clueless_norm_pretrain.yaml"
STAGE_ENV = ARM_DIR / "nemotron_nano_30b_metagaming_clueless_norm_pretrain.env"
H = 500


def edited_spec(tmp_path: Path, edit) -> Path:
    """The committed spec with its posture files named absolutely, edited by ``edit``, written into ``tmp_path``."""
    content = yaml.safe_load(SPEC_PATH.read_text())
    content["posture"]["training_config"] = str(STAGE_CONFIG)
    content["posture"]["env"] = str(STAGE_ENV)
    edit(content)
    path = tmp_path / "spec.yaml"
    path.write_text(yaml.safe_dump(content))
    return path


# -----------------------------------------------------------------------------
# Runs and window statistics
# -----------------------------------------------------------------------------


def test_window_statistics_over_a_built_corpus(tmp_path):
    rng = np.random.default_rng(1)
    documents = []
    for _ in range(40):
        document = rng.integers(1000, 2000, size=int(rng.integers(5, 30))).tolist()
        if rng.random() < 0.5:
            start = int(rng.integers(0, len(document)))
            length = int(rng.integers(1, 25))
            document[start : start + length] = [H] * length
        documents.append(document + [2])
    prefix = str(write_indexed_documents(tmp_path / "corpus", documents))
    tokens = token_memmap(Path(f"{prefix}.bin"))
    stream = np.concatenate([np.array(d, dtype=np.int32) for d in documents])
    assert np.array_equal(tokens, stream)

    seq = 8
    longest, occurrences = probe.window_statistics(tokens, seq, H, block_windows=3)
    windows = probe.window_count(len(stream), seq)
    assert windows == (len(stream) - 1) // seq
    assert len(longest) == len(occurrences) == windows
    for k in range(windows):
        window = stream[k * seq : (k + 1) * seq + 1]
        assert longest[k] == longest_runs((window[:-1] == H)[None, :])[0]
        assert occurrences[k] == int((window == H).sum())


# -----------------------------------------------------------------------------
# Window selection and tokens
# -----------------------------------------------------------------------------


def window_spec(**overrides) -> probe.WindowSpec:
    fields = dict(
        corpus="/corpus",
        seed=7,
        hidden_runs=probe.HiddenRuns(lengths=(2, 4), start=1, base_windows=2),
        quantile=0.9,
        quantile_windows=3,
        max_windows=2,
        reference_windows=4,
    )
    fields.update(overrides)
    return probe.WindowSpec(**fields)


def statistics(seq: int = 8):
    """Twenty windows: 0-5 hidden-free, 6-17 with longest runs 1..7 (two at 7), 18-19 entirely hidden."""
    longest = np.array([0] * 6 + [1, 1, 2, 2, 3, 3, 4, 4, 5, 6, 7, 7] + [seq, seq])
    occurrences = np.array([0] * 6 + [1, 1, 2, 2, 3, 3, 4, 4, 5, 6, 7, 7] + [seq + 1, seq + 1])
    return longest, occurrences


def test_selection_follows_the_stated_rule():
    longest, occurrences = statistics()
    windows, record = probe.select_windows(longest, occurrences, 8, window_spec())
    sets = {}
    for window in windows:
        sets.setdefault(window.set_name, []).append(window)
    assert list(sets) == [
        "all_hidden",
        "hidden_run_2",
        "hidden_run_4",
        probe.SET_QUANTILE,
        probe.SET_MAX,
        probe.SET_REFERENCE,
    ]
    reference = [w.corpus_window for w in sets["reference"]]
    assert len(set(reference)) == 4 and set(reference) <= set(range(6))
    # The runs are written over the first base_windows reference windows, in drawn order.
    assert [w.corpus_window for w in sets["hidden_run_2"]] == reference[:2]
    assert [w.hidden_run for w in sets["hidden_run_4"]] == [(1, 4), (1, 4)]
    assert sets["hidden_run_2"][0].window_id == f"corpus:{reference[0]}+hidden[1:3]"
    # Entirely hidden windows are left to all_hidden; the 0.9 quantile of the other 18 windows' runs is 7.
    partial = longest[:18]
    assert record["quantile_threshold"] == int(np.quantile(partial, 0.9, method="inverted_cdf")) == 7
    assert record["entirely_hidden_windows"] == 2 and record["hidden_free_windows"] == 6
    assert {w.corpus_window for w in sets[probe.SET_QUANTILE]} == {16, 17}  # at most 3 asked for, 2 in the pool
    assert {w.corpus_window for w in sets[probe.SET_MAX]} == {16, 17}
    assert record["max_longest_run_of_partially_hidden"] == 7 and record["max_pool"] == 2
    assert sets["all_hidden"][0].corpus_window is None


def test_selection_is_seeded_and_each_draw_is_independent_of_the_other_sets():
    longest, occurrences = statistics()
    first, _ = probe.select_windows(longest, occurrences, 8, window_spec())
    again, _ = probe.select_windows(longest, occurrences, 8, window_spec())
    resized, _ = probe.select_windows(longest, occurrences, 8, window_spec(quantile_windows=1, max_windows=1))
    reseeded, _ = probe.select_windows(longest, occurrences, 8, window_spec(seed=8))

    def reference(windows):
        return [w.corpus_window for w in windows if w.set_name == probe.SET_REFERENCE]

    assert first == again
    assert reference(resized) == reference(first)
    assert reference(reseeded) != reference(first)


def test_too_few_hidden_free_windows_is_refused():
    longest, occurrences = statistics()
    with pytest.raises(probe.ProbeError, match="holds 6 windows without the hidden id; the reference set needs 7"):
        probe.select_windows(longest, occurrences, 8, window_spec(reference_windows=7))


def test_a_corpus_of_only_hidden_windows_is_refused():
    with pytest.raises(probe.ProbeError, match="entirely hidden"):
        probe.select_windows(np.array([8, 8]), np.array([9, 9]), 8, window_spec(reference_windows=0))


def test_window_tokens():
    corpus = np.arange(100, 140, dtype=np.int32)
    hidden = probe.window_tokens(probe.PlannedWindow("all_hidden", "all_hidden", None, None), corpus, 8, H)
    assert hidden.tolist() == [H] * 9
    plain = probe.window_tokens(probe.PlannedWindow("reference", "corpus:2", 2, None), corpus, 8, H)
    assert plain.tolist() == list(range(116, 125)) and plain.dtype == np.int64
    run = probe.window_tokens(probe.PlannedWindow("hidden_run_6", "x", 2, (1, 6)), corpus, 8, H)
    # Inputs 1..6 hidden; the last label (index 8) stays the corpus's.
    assert run.tolist() == [116] + [H] * 6 + [123, 124]
    assert probe.describe_tokens(run, H) == {"longest_hidden_run": 6, "hidden_tokens": 6}
    with pytest.raises(probe.ProbeError, match="runs past the corpus's end"):
        probe.window_tokens(probe.PlannedWindow("reference", "corpus:4", 4, None), corpus, 8, H)


# -----------------------------------------------------------------------------
# The plan: the corpus read, checked and cut
# -----------------------------------------------------------------------------

SEQ = 8


def plan_stream() -> np.ndarray:
    """Twelve 8-token windows (97 ids): windows 0-5 and 11 hold no H, 6-9 one H run each (2, 3, 5 and 5 long, the
    runs of 7 and 9 reaching their last label), and window 10's inputs are all H."""
    stream = np.arange(1000, 1097, dtype=np.int32)
    for start, length in ((50, 2), (57, 3), (64, 5), (75, 5), (80, 8)):
        stream[start : start + length] = H
    return stream


def plan_corpus(tmp_path: Path, *, revision: str | None = None) -> str:
    """``plan_stream`` as six documents, with the tokenize record ``count_idx_tokens.py`` writes beside a corpus
    (naming the committed spec's tokenizer, at ``revision`` when given)."""
    spec = probe.load_spec(SPEC_PATH)
    prefix = str(write_indexed_documents(tmp_path / "corpus", np.split(plan_stream(), [5, 18, 38, 39, 69])))
    parameters = {"tokenizer": spec.tokenizer, "tokenizer_revision": revision or spec.tokenizer_revision}
    Path(f"{prefix}.provenance.json").write_text(json.dumps({"files": [], "parameters": parameters}))
    return prefix


def plan_spec(tmp_path: Path, prefix: str, **hidden_runs) -> probe.ProbeSpec:
    def edit(content):
        content["windows"].update(
            corpus=prefix,
            hidden_runs={"lengths": [2, 4], "start": 1, "base_windows": 2, **hidden_runs},
            real_longest_run={"quantile": 0.5, "quantile_windows": 2, "max_windows": 1},
            reference={"windows": 3},
        )

    return probe.load_spec(edited_spec(tmp_path, edit))


def test_build_plan_cuts_the_spec_windows_from_the_corpus(tmp_path):
    prefix = plan_corpus(tmp_path)
    spec = plan_spec(tmp_path, prefix)
    plan = probe.build_plan(spec, SEQ, H)

    stream = plan_stream()
    assert plan["corpus"] == {
        "prefix": prefix,
        "documents": 6,
        "tokens": 97,
        "provenance": {"tokenizer": spec.tokenizer, "tokenizer_revision": spec.tokenizer_revision},
    }
    expected_windows, expected_selection = probe.select_windows(
        *probe.window_statistics(stream, SEQ, H), SEQ, spec.windows
    )
    assert plan["windows"] == expected_windows and plan["selection"] == expected_selection
    assert expected_selection["hidden_free_windows"] == 7 and expected_selection["entirely_hidden_windows"] == 1
    # Each window's ids are the stream's, cut every SEQ tokens, with its synthetic run written over its inputs.
    assert len(plan["tokens"]) == len(plan["windows"])
    for window, tokens in zip(plan["windows"], plan["tokens"]):
        if window.corpus_window is None:
            expected = np.full(SEQ + 1, H)
        else:
            expected = stream[window.corpus_window * SEQ : (window.corpus_window + 1) * SEQ + 1].astype(np.int64)
            if window.hidden_run is not None:
                start, length = window.hidden_run
                expected[start : start + length] = H
        assert tokens.dtype == np.int64 and np.array_equal(tokens, expected), window


def test_build_plan_refuses_a_corpus_tokenized_at_another_revision(tmp_path):
    prefix = plan_corpus(tmp_path, revision="0" * 40)
    with pytest.raises(probe.ProbeError, match=f"tokenized with .*@{'0' * 40}, not "):
        probe.build_plan(plan_spec(tmp_path, prefix), SEQ, H)


@pytest.mark.parametrize(
    ("record", "message"),
    [(None, "does not exist"), ({"files": []}, "records no tokenize parameters"), ([], "records no tokenize")],
)
def test_build_plan_refuses_a_corpus_whose_tokenizer_it_cannot_read(tmp_path, record, message):
    prefix = plan_corpus(tmp_path)
    provenance = Path(f"{prefix}.provenance.json")
    if record is None:
        provenance.unlink()
    else:
        provenance.write_text(json.dumps(record))
    with pytest.raises(probe.ProbeError, match=message):
        probe.build_plan(plan_spec(tmp_path, prefix), SEQ, H)


def test_build_plan_refuses_a_bin_its_index_does_not_describe(tmp_path):
    prefix = plan_corpus(tmp_path)
    with open(f"{prefix}.bin", "ab") as handle:
        handle.write(np.array([7], dtype=np.int32).tobytes())
    with pytest.raises(probe.ProbeError, match="is 392 bytes, its index describes 388"):
        probe.build_plan(plan_spec(tmp_path, prefix), SEQ, H)


def test_build_plan_refuses_runs_that_do_not_fit_a_window(tmp_path):
    prefix = plan_corpus(tmp_path)
    with pytest.raises(probe.ProbeError, match=r"hidden runs of \[8\] tokens from position 1 do not fit 8-token"):
        probe.build_plan(plan_spec(tmp_path, prefix, lengths=[2, 8]), SEQ, H)


def test_runs_that_do_not_fit_a_window_are_refused():
    with pytest.raises(probe.ProbeError, match=r"hidden runs of \[8064\] tokens from position 129"):
        probe.check_run_bounds(probe.HiddenRuns(lengths=(1024, 8064), start=129, base_windows=1), 8192)


# -----------------------------------------------------------------------------
# The spec
# -----------------------------------------------------------------------------


def test_the_committed_spec_loads_and_names_the_stage_one_posture():
    spec = probe.load_spec(SPEC_PATH)
    assert spec.training_config == str(STAGE_CONFIG)
    assert spec.posture_env == str(STAGE_ENV)
    assert spec.state_modes == ("0", "checkpoint") and spec.fp32_mode == "checkpoint"
    assert spec.windows.hidden_runs == probe.HiddenRuns(lengths=(1024, 2048, 4096, 8064), start=128, base_windows=4)
    assert spec.criteria == probe.Criteria(max_abs_log2=56, reference_multiple=2.0, floor_log2=0.0625)
    assert spec.name == "m5b_ssm_state"
    probe.check_run_bounds(spec.windows.hidden_runs, 8192)


def test_the_stage_one_env_states_the_mode_the_spec_runs():
    from scripts.training.launcher_source import env_override_entries

    spec = probe.load_spec(SPEC_PATH)
    mode = probe.posture_state_mode(env_override_entries(spec.posture_env))
    assert mode == "checkpoint" and mode in spec.state_modes


def test_an_env_that_does_not_state_the_mode_is_refused():
    with pytest.raises(probe.ProbeError, match="exactly once"):
        probe.posture_state_mode(["ISAMBARD_CUDA_MAX_CONNECTIONS=32"])
    with pytest.raises(probe.ProbeError, match="exactly once"):
        probe.posture_state_mode(["ISAMBARD_FP32_SSM_STATE=0", "ISAMBARD_FP32_SSM_STATE=1"])


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (lambda c: c.update(extra=1), r"unknown keys \['extra'\]"),
        (lambda c: c.pop("criteria"), r"missing keys \['criteria'\]"),
        (lambda c: c.update(state_modes=["checkpoint", "0"]), "the bf16 path first"),
        (lambda c: c.update(state_modes=["0"]), "two modes"),
        (lambda c: c.update(state_modes=["0", "2"]), "the bf16 path first"),
        (lambda c: c["windows"].update(corpus="relative/prefix"), "windows.corpus must be an absolute path"),
        (lambda c: c.update(output_dir="logs"), "output_dir must be an absolute path"),
        (lambda c: c["windows"]["hidden_runs"].update(base_windows=17), "more than the reference windows"),
        (lambda c: c["windows"]["hidden_runs"].update(lengths=[8, 8]), "repeats a length"),
        (lambda c: c["windows"]["real_longest_run"].update(quantile=1.5), "at most 1"),
        (lambda c: c["windows"]["real_longest_run"].update(quantile=0), "must be > 0"),
        (lambda c: c["windows"].update(all_hidden={"windows": 2}), r"unknown keys \['windows'\]"),
        (lambda c: c["criteria"].update(max_abs_log2=128), "at most 127"),
        (lambda c: c["criteria"]["mode_agreement"].update(reference_multiple=0.5), "must be >= 1.0"),
        (lambda c: c["criteria"]["mode_agreement"].update(floor_log2=0), "must be > 0"),
        (lambda c: c["posture"].update(env="/nonexistent.env"), "is not a file"),
        (lambda c: c["tokenizer"].update(revision=""), "tokenizer.revision must be a non-empty string"),
        (lambda c: c["windows"].update(seed=True), "windows.seed must be an integer >= 0, not True"),
    ],
)
def test_malformed_specs_are_refused(tmp_path, edit, message):
    with pytest.raises(ValueError, match=message):
        probe.load_spec(edited_spec(tmp_path, edit))


# -----------------------------------------------------------------------------
# The posture
# -----------------------------------------------------------------------------


def posture_spec(tmp_path: Path, *, overlay: str | None = None, env: str | None = None) -> probe.ProbeSpec:
    """The committed spec with its training config overlaid by ``overlay`` (YAML, composed onto the stage-1 config
    through ``base_config:``) and its env file's text replaced by ``env``, each when given."""

    def edit(content):
        if overlay is not None:
            config = tmp_path / "posture.yaml"
            config.write_text(f"base_config: {STAGE_CONFIG}\n{overlay}")
            content["posture"]["training_config"] = str(config)
        if env is not None:
            path = tmp_path / "posture.env"
            path.write_text(env)
            content["posture"]["env"] = str(path)

    return probe.load_spec(edited_spec(tmp_path, edit))


def test_resolve_posture_builds_the_stage_one_model_config():
    spec = probe.load_spec(SPEC_PATH)
    model, record = probe.resolve_posture(spec)
    assert record["hidden_token_id"] == H and record["seq_length"] == 8192
    assert record["state_mode_trained"] == "checkpoint"
    assert record["training_config"] == {"path": str(STAGE_CONFIG), "sha256": probe.file_sha256(STAGE_CONFIG)}
    assert record["env"]["path"] == str(STAGE_ENV) and "ISAMBARD_FP32_SSM_STATE=checkpoint" in record["env"]["entries"]
    assert record["mixed_precision"] == "nemotron_h_bf16_with_fp8_current_scaling_bf16_params_bf16_grad_reduce"
    # The mixed-precision preset is applied to the model config, and the random initialisation is skipped.
    assert model.bf16 and model.params_dtype == torch.bfloat16 and model.fp8 is not None
    assert model.perform_initialization is False
    assert model.seq_length == 8192 and model.vocab_size == 131072
    assert record["model"]["fp8"] == repr(model.fp8) and record["model"]["vocab_size"] == "131072"


@pytest.mark.parametrize(
    ("overlay", "env", "message"),
    [
        ("token_masking:\n  token_ids: [500, 501]\n", None, r"exactly one token id \(the hidden token\), not \[500, "),
        ("dataset:\n  seq_length: 4096\n", None, "model.seq_length 8192 != dataset.seq_length 4096"),
        (
            "tokenizer:\n  tokenizer_model: geodesic-research/nemotron-instruct-tokenizer\n",
            None,
            "the posture's tokenizer is geodesic-research/nemotron-instruct-tokenizer, the spec's",
        ),
        (None, "ISAMBARD_CUDA_MAX_CONNECTIONS=32\n", "must set ISAMBARD_FP32_SSM_STATE exactly once"),
        (None, "ISAMBARD_FP32_SSM_STATE=1\n", "trains in ISAMBARD_FP32_SSM_STATE=1, which the spec does not run"),
    ],
)
def test_resolve_posture_refuses_a_posture_it_cannot_reproduce(tmp_path, overlay, env, message):
    with pytest.raises(probe.ProbeError, match=message):
        probe.resolve_posture(posture_spec(tmp_path, overlay=overlay, env=env))


# -----------------------------------------------------------------------------
# The checkpoint, the state modes, the schedule and the command line
# -----------------------------------------------------------------------------


def test_checkpoint_record_reads_an_iteration_directory(tmp_path):
    (tmp_path / "run_config.yaml").write_text("model: {}\n")
    (tmp_path / "metadata.json").write_text('{"sharded_backend": "torch_dist"}')
    assert probe.checkpoint_record(str(tmp_path)) == {
        "path": str(tmp_path),
        "run_config_sha256": hashlib.sha256(b"model: {}\n").hexdigest(),
        "metadata": {"sharded_backend": "torch_dist"},
    }


def test_checkpoint_record_refuses_a_directory_that_is_not_an_iteration(tmp_path):
    (tmp_path / "metadata.json").write_text("{}")
    with pytest.raises(probe.ProbeError, match="has no run_config.yaml"):
        probe.checkpoint_record(str(tmp_path))


@pytest.fixture
def training_scan(monkeypatch):
    """The training scan ``mamba_mixer`` calls, unpatched, and put back after the test: the fp32-state patch replaces
    it for the rest of the process."""
    from megatron.core.ssm import mamba_mixer

    scan = mamba_mixer.mamba_split_conv1d_scan_combined
    assert scan is not None, "mamba_ssm is not importable, so there is no training scan to patch"
    assert not getattr(scan, "_fp32_ssm_state_patched", False), "an earlier test left the fp32-state patch installed"
    monkeypatch.setattr(mamba_mixer, "mamba_split_conv1d_scan_combined", scan)
    return mamba_mixer


def test_the_bf16_path_cannot_be_measured_once_the_fp32_patch_is_installed(training_scan):
    assert probe.install_state_mode("0") == "bfloat16"
    assert probe.install_state_mode("checkpoint") == "float32"
    assert training_scan.mamba_split_conv1d_scan_combined._fp32_ssm_state_mode == "checkpoint"
    with pytest.raises(probe.ProbeError, match="installed already; the bf16 path cannot be measured"):
        probe.install_state_mode("0")


@pytest.mark.parametrize("second", ["1", "checkpoint"])
def test_an_fp32_mode_the_patch_does_not_install_is_refused(training_scan, second):
    assert probe.install_state_mode("checkpoint") == "float32"
    with pytest.raises(probe.ProbeError, match=f"patch for ISAMBARD_FP32_SSM_STATE={second} was not installed"):
        probe.install_state_mode(second)


def test_main_refuses_a_run_id_that_cannot_name_a_result_file(monkeypatch, capsys, tmp_path):
    absent = tmp_path / "absent.yaml"  # the spec is never read: the run id is refused first
    monkeypatch.setattr(sys, "argv", ["ssm_state_probe.py", "--spec", str(absent), "--run-id", "j1/../x"])
    with pytest.raises(SystemExit) as raised:
        probe.main()
    assert raised.value.code == 2
    assert "--run-id 'j1/../x' must be letters, digits, '.', '_' or '-'" in capsys.readouterr().err


# -----------------------------------------------------------------------------
# Recording
# -----------------------------------------------------------------------------


def scan_result(states_peak: float, final_peak: float, out_peak: float, ungated_peak: float, dtype):
    """The six tensors ``_mamba_chunk_scan_combined_fwd`` returns, with known maxima: two chunks of a 1-head state."""
    states = torch.zeros(1, 2, 1, 2, 3, dtype=dtype)
    states[0, 1, 0, 1, 2] = -states_peak  # entering chunk 1
    final = torch.zeros(1, 1, 2, 3, dtype=torch.float32)
    final[0, 0, 0, 0] = final_peak
    out = torch.full((1, 4, 1, 2), -out_peak, dtype=dtype)
    out_x = torch.full((1, 4, 1, 2), ungated_peak, dtype=dtype)
    return out, out_x, None, None, states, final


def test_the_scan_recorder_reads_each_layers_maxima():
    recorder = probe.ScanRecorder()
    scan = recorder.wrap(lambda result: result)
    for layer, result in (
        (1, scan_result(3.0, 5.0, 2.0, 7.0, torch.bfloat16)),
        (3, scan_result(9.0, 1.0, 0.5, 0.25, torch.bfloat16)),
    ):
        recorder.enter(layer)
        assert scan(result) is result  # the scan's own results pass through
        recorder.leave(layer)
    records = recorder.take([1, 3])
    assert records["1"] == {
        "state": 5.0,
        "state_by_chunk": [0.0, 3.0, 5.0],
        "scan_output": 2.0,
        "scan_output_ungated": 7.0,
        "nonfinite": 0,
        "state_dtype": "bfloat16",
    }
    assert records["3"]["state"] == 9.0 and records["3"]["state_by_chunk"] == [0.0, 9.0, 1.0]
    # take clears: the next window starts empty.
    with pytest.raises(probe.ProbeError, match=r"recorded in layers \[\]"):
        recorder.take([1, 3])


def test_the_scan_recorder_counts_non_finite_values():
    recorder = probe.ScanRecorder()
    result = list(scan_result(1.0, 1.0, 1.0, 1.0, torch.float32))
    result[4][0, 0, 0, 0, 0] = math.nan
    result[0][0, 0, 0, 0] = math.inf
    recorder.enter(2)
    recorder.wrap(lambda r: r)(tuple(result))
    recorder.leave(2)
    record = recorder.take([2])["2"]
    assert record["nonfinite"] == 2 and math.isnan(record["state"]) and math.isinf(record["scan_output"])
    assert record["state_dtype"] == "float32"


def test_the_scan_recorder_refuses_scans_it_cannot_attribute_to_one_mixer():
    result = scan_result(1.0, 1.0, 1.0, 1.0, torch.float32)
    recorder = probe.ScanRecorder()
    scan = recorder.wrap(lambda r: r)
    with pytest.raises(probe.ProbeError, match="outside every Mamba mixer"):
        scan(result)
    recorder.enter(4)
    scan(result)
    with pytest.raises(probe.ProbeError, match="second scan"):
        scan(result)
    with pytest.raises(probe.ProbeError, match="Mamba mixer 5 started inside mixer 4"):
        recorder.enter(5)
    with pytest.raises(probe.ProbeError, match="Mamba mixer 6 ended while mixer 4 was recorded"):
        recorder.leave(6)
    with pytest.raises(probe.ProbeError, match="Mamba mixer 4 is still running"):
        recorder.take([4])
    silent = probe.ScanRecorder()
    silent.enter(5)
    with pytest.raises(probe.ProbeError, match="ran no scan"):
        silent.leave(5)


def test_the_layer_output_recorder_reads_tuples_and_tensors():
    recorder = probe.LayerOutputRecorder()
    recorder.hook(1)(None, (), torch.tensor([-3.0, 1.0]))
    recorder.hook(2)(None, (), (torch.tensor([2.0, math.nan]), None))
    records = recorder.take([1, 2])
    assert records["1"] == {"max_abs": 3.0, "nonfinite": 0}
    assert math.isnan(records["2"]["max_abs"]) and records["2"]["nonfinite"] == 1
    with pytest.raises(probe.ProbeError, match=r"recorded for \[\], not \[1\]"):
        recorder.take([1])


def test_loss_record_separates_trained_and_hidden_targets():
    loss = torch.tensor([[1.0, 2.0, 3.0, 6.0]])
    labels = torch.tensor([[7, H, 8, H]])
    assert probe.loss_record(loss, labels, H) == {
        "nonfinite": 0,
        "max": 6.0,
        "mean_trained": 2.0,
        "mean_hidden": 4.0,
        "trained_targets": 2,
        "hidden_targets": 2,
    }
    all_hidden = probe.loss_record(torch.tensor([[1.0, math.inf]]), torch.tensor([[H, H]]), H)
    assert all_hidden["mean_trained"] is None and all_hidden["nonfinite"] == 1


@pytest.mark.run_only_on("GPU")
def test_the_recorders_installed_on_a_real_hybrid_model_record_its_training_scan_in_both_modes(
    monkeypatch, one_rank_world_left_as_found
):
    """``install_recorders`` on a HybridModel of two Mamba layers around an attention layer, and its training forward
    (no inference context, the memory-efficient scan) run through ``run_windows``: once on the bf16 path, then under
    the fp32-state patch, which only the training scan goes through, so the states the scans store turn fp32."""
    import mamba_ssm.ops.triton.ssd_combined as ssd_combined
    from megatron.core.ssm import mamba_mixer

    from tests.unit_tests.one_rank_nccl_world import one_rank_model_parallel_state
    from tests.unit_tests.small_language_models import hybrid_model

    # Both are module globals the probe replaces for the rest of its process; the test puts them back.
    monkeypatch.setattr(ssd_combined, "_mamba_chunk_scan_combined_fwd", ssd_combined._mamba_chunk_scan_combined_fwd)
    monkeypatch.setattr(mamba_mixer, "mamba_split_conv1d_scan_combined", mamba_mixer.mamba_split_conv1d_scan_combined)
    seq, vocab = 256, 1024  # two of the scan's 128-token chunks
    rng = np.random.default_rng(0)
    plan = {
        "windows": ["ordinary", "all_hidden", "ordinary_again"],
        "tokens": [rng.integers(3, vocab, size=seq + 1), np.full(seq + 1, H), rng.integers(3, vocab, size=seq + 1)],
    }
    records = {}
    with one_rank_model_parallel_state(seed=1234):
        model = hybrid_model(
            "M*M",
            vocab_size=vocab,
            max_sequence_length=seq,
            pre_process=True,
            num_layers=3,
            hidden_size=256,
            num_attention_heads=4,
            bf16=True,
            params_dtype=torch.bfloat16,
        )
        model.eval()
        model.requires_grad_(False)
        scans, outputs, mamba_layers, decoder_layers = probe.install_recorders(model)
        assert (mamba_layers, decoder_layers) == ([1, 3], [1, 2, 3])
        with pytest.raises(probe.ProbeError, match="already wrapped"):
            probe.install_recorders(model)
        for mode in ("0", "checkpoint"):
            dtype = probe.install_state_mode(mode)
            records[mode] = dict(
                probe.run_windows(model, plan, H, scans, outputs, mamba_layers, decoder_layers, dtype)
            )

    for mode, dtype in (("0", "bfloat16"), ("checkpoint", "float32")):
        assert sorted(records[mode]) == [0, 1, 2]
        for record in records[mode].values():
            assert sorted(record["ssm"]) == ["1", "3"] and sorted(record["layer_output"]) == ["1", "2", "3"]
            for ssm in record["ssm"].values():
                assert ssm["state_dtype"] == dtype and ssm["nonfinite"] == 0
                # The state entering the first chunk is the zero initial state; the next boundary and the final
                # state have integrated tokens.
                assert len(ssm["state_by_chunk"]) == 3 and ssm["state_by_chunk"][0] == 0.0
                assert all(0.0 < value < math.inf for value in ssm["state_by_chunk"][1:])
                assert 0.0 < ssm["scan_output"] < math.inf and 0.0 < ssm["scan_output_ungated"] < math.inf
            assert all(0.0 < out["max_abs"] < math.inf for out in record["layer_output"].values())
        assert records[mode][1]["loss"]["trained_targets"] == 0 and records[mode][1]["loss"]["hidden_targets"] == seq
        assert records[mode][0]["loss"]["trained_targets"] == seq and records[mode][0]["loss"]["nonfinite"] == 0
    # The two paths compute the same scan at different precisions: their state maxima agree to a small fraction of a
    # binade on an ordinary window.
    for layer in ("1", "3"):
        gap = probe._log2_gap(records["0"][0]["ssm"][layer]["state"], records["checkpoint"][0]["ssm"][layer]["state"])
        assert gap < 0.1
    embedding = probe.embedding_record(model, H)
    assert sorted(embedding) == ["input_embedding", "output_layer"]
    for rows in embedding.values():
        assert 0.0 < rows["hidden_row_norm"] < math.inf and 0.0 < rows["median_row_norm"] < math.inf


# -----------------------------------------------------------------------------
# Criteria, summaries, output
# -----------------------------------------------------------------------------


def window(window_id, state=1.0, scan=1.0, layer_out=1.0, nonfinite=0, loss_nonfinite=0, mean_trained=2.0):
    layers = ("1", "3")
    return {
        "window_id": window_id,
        "longest_hidden_run": 0,
        "hidden_tokens": 0,
        "ssm": {
            layer: {
                "state": state,
                "state_by_chunk": [state / 2, state],
                "scan_output": scan,
                "scan_output_ungated": scan,
                "nonfinite": nonfinite,
                "state_dtype": "bfloat16",
            }
            for layer in layers
        },
        "layer_output": {layer: {"max_abs": layer_out, "nonfinite": 0} for layer in layers},
        "loss": {
            "nonfinite": loss_nonfinite,
            "max": 5.0,
            "mean_trained": mean_trained,
            "mean_hidden": None,
            "trained_targets": 8,
            "hidden_targets": 0,
        },
    }


def measurements(bf16_h=None, fp32_h=None, bf16_ref=None, fp32_ref=None):
    """Two modes, a reference set and one hidden-run set of one window each."""
    return {
        "0": {"hidden_run_8": [bf16_h or window("h")], "reference": [bf16_ref or window("r")]},
        "checkpoint": {"hidden_run_8": [fp32_h or window("h")], "reference": [fp32_ref or window("r")]},
    }


CRITERIA = probe.Criteria(max_abs_log2=56, reference_multiple=2.0, floor_log2=0.0625)


def test_identical_finite_modes_pass():
    result = probe.evaluate_criteria(measurements(), CRITERIA, ("0", "checkpoint"))
    assert result["verdict"] == "PASS" and result["exit_status"] == 0
    assert {name: c["outcome"] for name, c in result["criteria"].items()} == dict.fromkeys(
        ("finite", "headroom", "mode_agreement"), "PASS"
    )


@pytest.mark.parametrize(
    "bad",
    [
        window("h", nonfinite=3),
        window("h", state=math.nan),
        window("h", loss_nonfinite=1),
        window("h", layer_out=math.inf),
    ],
)
def test_any_non_finite_value_fails(bad):
    result = probe.finite_criterion(measurements(bf16_h=bad))
    assert result["outcome"] == "FAIL" and result["first"]


def test_headroom_is_inclusive_at_the_bound():
    at_bound = probe.headroom_criterion(measurements(bf16_h=window("h", state=2.0**56)), 56)
    assert at_bound["outcome"] == "PASS" and at_bound["largest"]["0"]["log2"] == 56
    above = probe.headroom_criterion(measurements(fp32_h=window("h", layer_out=2.0**56 * 1.01)), 56)
    assert above["outcome"] == "FAIL" and "mode checkpoint" in above["first"][0]
    unreadable = probe.headroom_criterion(measurements(bf16_h=window("h", scan=math.nan)), 56)
    assert unreadable["outcome"] == "FAIL" and math.isnan(unreadable["largest"]["0"]["value"])


def test_agreement_tolerance_is_the_floor_when_the_reference_agrees_closely():
    # Reference gap 0.01 binade: tolerance max(0.02, 0.0625) = 0.0625.
    ref = measurements(bf16_ref=window("r", state=2.0**0.01), bf16_h=window("h", state=2.0**0.06))
    result = probe.agreement_criterion(ref, "0", "checkpoint", 2.0, 0.0625)
    assert result["outcome"] == "PASS"
    assert result["tolerance_log2"]["state"] == pytest.approx(0.0625)
    assert result["worst_gap_log2"]["hidden_run_8"]["state"] == pytest.approx(0.06)
    worse = measurements(bf16_ref=window("r", state=2.0**0.01), bf16_h=window("h", state=2.0**0.07))
    assert probe.agreement_criterion(worse, "0", "checkpoint", 2.0, 0.0625)["outcome"] == "FAIL"


def test_agreement_tolerance_scales_with_the_reference_gap():
    loose = measurements(fp32_ref=window("r", scan=2.0**-0.1), bf16_h=window("h", scan=2.0**0.19))
    result = probe.agreement_criterion(loose, "0", "checkpoint", 2.0, 0.0625)
    assert result["tolerance_log2"]["scan_output"] == pytest.approx(0.2)
    assert result["outcome"] == "PASS"


def test_agreement_on_zero_and_non_finite_maxima():
    assert probe._log2_gap(0.0, 0.0) == 0.0
    assert probe._log2_gap(0.0, 1.0) == math.inf
    assert probe._log2_gap(math.nan, 1.0) == math.inf
    one_zero = measurements(bf16_h=window("h", state=0.0))
    assert probe.agreement_criterion(one_zero, "0", "checkpoint", 2.0, 0.0625)["outcome"] == "FAIL"
    broken_reference = measurements(bf16_ref=window("r", state=math.inf))
    result = probe.agreement_criterion(broken_reference, "0", "checkpoint", 2.0, 0.0625)
    assert result["outcome"] == "NOT EVALUATED"


def test_the_verdict_is_fail_when_any_criterion_fails():
    result = probe.evaluate_criteria(measurements(bf16_h=window("h", state=2.0**60)), CRITERIA, ("0", "checkpoint"))
    assert result["criteria"]["headroom"]["outcome"] == "FAIL"
    assert result["verdict"] == "FAIL" and result["exit_status"] == 1


def test_summary():
    data = measurements(bf16_h=window("h", state=4.0, mean_trained=2.5))
    data["0"]["hidden_run_8"].append(window("h2", state=8.0))
    data["checkpoint"]["hidden_run_8"].append(window("h2", state=8.0))
    summary = probe.summarize(data, "0", "checkpoint")
    entry = summary["modes"]["0"]["hidden_run_8"]
    assert entry["windows"] == 2
    assert entry["max_by_layer"]["state"] == {"1": 8.0, "3": 8.0}
    assert entry["max_log2"]["state"] == 3.0
    assert entry["state_profile_by_layer"]["1"] == [4.0, 8.0]
    assert entry["state_log2_over_reference_by_layer"]["1"] == 3.0
    assert summary["loss_difference_between_modes"]["hidden_run_8"] == {"mean_abs": 0.25, "max_abs": 0.5}


def test_json_safe_writes_strict_json():
    value = {"a": [1.0, math.nan, math.inf, -math.inf], "b": (2, {"c": math.nan}), "d": None}
    encoded = json.dumps(probe.json_safe(value), allow_nan=False)
    assert json.loads(encoded) == {"a": [1.0, "nan", "inf", "-inf"], "b": [2, {"c": "nan"}], "d": None}


def test_max_or_nan_does_not_skip_a_leading_nan():
    assert math.isnan(probe.max_or_nan([math.nan, 1.0]))
    assert probe.max_or_nan([1.0, 3.0]) == 3.0


def test_the_probe_hashes_files_with_the_shared_helper():
    assert probe.file_sha256 is sha256_manifest.file_sha256
