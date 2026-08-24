"""Tests for the chat-SFT corpus filter and the render-consistency gate.

`scripts/data/filter_sft_corpora.py` drops rows whose full render exceeds the training
window — such a row would be truncated mid-target by packing, supervising an answer with no
ending — and `scripts/data/check_sft_render_consistency.py` is the gate that refuses a corpus
which either still contains over-window rows or renders differently from the RL-serving
template. Both measure length through the shared `scripts/data/chat_render_length.py`, so
they agree by construction; these tests drive all three.

Both scripts are run as real subprocesses against real corpora on disk (the filter mutates
files in place and forks a worker pool, so the file-level behaviour IS the behaviour), using
a tiny purpose-built tokenizer: a WordLevel backend plus a chat template, saved to a local
dir. That keeps the tests hermetic and makes token counts small enough to place a window
boundary exactly between two rows. One further test uses the real production tokenizer and
RL template when they are on this filesystem, and skips loudly otherwise.

Run:
    ./pipeline_env_exec.sh "cd <repo>; source pipeline_env_activate.sh || exit 1; cd /tmp; \\
        python -m pytest <repo>/tests/unit_tests/data/test_sft_corpus_filters.py -v"
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_DIR = REPO_ROOT / "scripts" / "data"
FILTER_SCRIPT = SCRIPTS_DIR / "filter_sft_corpora.py"
CHECK_SCRIPT = SCRIPTS_DIR / "check_sft_render_consistency.py"
LENGTH_MODULE = SCRIPTS_DIR / "chat_render_length.py"

# The fixture template, and a variant that renders differently for the mismatch case.
FIXTURE_TEMPLATE = (
    "{%- for m in messages %}"
    "{{- '<|im_start|> ' + m['role'] + ' ' }}{% generation %}{{- m['content'] + ' <|im_end|> ' }}{% endgeneration %}"
    "{%- endfor %}"
)
DIVERGENT_TEMPLATE = FIXTURE_TEMPLATE.replace("{%- for m in messages %}", "{%- for m in messages %}{{- 'PREFIX ' }}")

# Real production artifacts for the end-to-end parity check (skipped when absent).
PRODUCTION_TOKENIZER = Path(
    f"/projects/a5k/public/data_{os.environ.get('USER', '')}/bedtime_warmstart_sft"
    "/nemotron-native-thinkoff-genmask-tokenizer"
)
from tests.unit_tests.gr_test_utils import discover_rl_template


def _row(role_contents) -> dict:
    return {"messages": [{"role": role, "content": content} for role, content in role_contents]}


def _write_root(root: Path, rows: list[dict]) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "training.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    return root


def _read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _run(script: Path, config: dict, tmp_path: Path, name: str = "config.yaml"):
    config_path = tmp_path / name
    config_path.write_text(yaml.safe_dump(config))
    return subprocess.run(
        [sys.executable, str(script), "--config", str(config_path)],
        capture_output=True,
        text=True,
        timeout=300,
    )


@pytest.fixture(scope="module")
def length_module():
    """The shared render-length helper, loaded by path (scripts/ is not a package)."""
    spec = importlib.util.spec_from_file_location("chat_render_length", LENGTH_MODULE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def tiny_tokenizer(tmp_path_factory):
    """A saved tokenizer dir: WordLevel backend + the fixture chat template."""
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    words = ["[UNK]", "user", "assistant", "hello", "world", "story", "alpha", "beta", "gamma", "delta"]
    backend = Tokenizer(models.WordLevel(vocab={word: index for index, word in enumerate(words)}, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]")
    tokenizer.chat_template = FIXTURE_TEMPLATE
    path = tmp_path_factory.mktemp("tiny_tokenizer")
    tokenizer.save_pretrained(path)
    return path


@pytest.fixture(scope="module")
def loaded_tokenizer(tiny_tokenizer):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(tiny_tokenizer))


SHORT_ROW = _row([("user", "hello"), ("assistant", "story")])
LONG_ROW = _row([("user", "hello world " * 12), ("assistant", "story alpha beta gamma delta " * 12)])


@pytest.fixture(scope="module")
def window(length_module, loaded_tokenizer) -> int:
    """A seq_length strictly between the short and long fixture rows' render lengths."""
    short = length_module.render_token_length(loaded_tokenizer, SHORT_ROW["messages"])
    long_ = length_module.render_token_length(loaded_tokenizer, LONG_ROW["messages"])
    assert short < long_, (short, long_)
    return (short + long_) // 2


# ---------------------------------------------------------------------------------------
# filter_sft_corpora.py
# ---------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def filtered_root(tiny_tokenizer, window, tmp_path_factory) -> Path:
    """One real filter run over a root holding two short rows and one over-window row.

    Module-scoped because each run pays a fresh transformers import in the subprocess; the
    two tests below assert on different halves of the same run's result.
    """
    directory = tmp_path_factory.mktemp("filter_run")
    root = _write_root(directory / "core", [SHORT_ROW, LONG_ROW, SHORT_ROW])
    result = _run(
        FILTER_SCRIPT,
        {"tokenizer": str(tiny_tokenizer), "seq_length": window, "workers": 2, "roots": [str(root)]},
        directory,
    )
    assert result.returncode == 0, result.stderr
    return root


def test_over_window_rows_are_dropped_and_preserved(filtered_root):
    kept = _read_rows(filtered_root / "training.jsonl")
    assert len(kept) == 2, "the over-window row survived the filter"
    assert all(row == SHORT_ROW for row in kept)
    # The original must remain recoverable: this file is the filter's own input next time.
    assert len(_read_rows(filtered_root / "training_prefilter.jsonl")) == 3


def test_provenance_records_the_config_and_counts(filtered_root, tiny_tokenizer, window):
    provenance = json.loads((filtered_root / "filter_provenance.json").read_text())
    assert provenance["config"]["seq_length"] == window
    assert provenance["config"]["tokenizer"] == str(tiny_tokenizer)
    assert provenance["counts"]["kept"] == 2
    assert provenance["counts"][f"dropped_over_{window}"] == 1


def test_rerunning_refilters_from_the_preserved_original(tiny_tokenizer, window, tmp_path):
    """Idempotence: the filter always reads training_prefilter.jsonl, never its own output.

    Re-running with a window that admits everything must RESTORE the dropped row — impossible
    if the second pass had read the already-filtered file.
    """
    root = _write_root(tmp_path / "core", [SHORT_ROW, LONG_ROW, SHORT_ROW])
    tight = {"tokenizer": str(tiny_tokenizer), "seq_length": window, "workers": 2, "roots": [str(root)]}
    assert _run(FILTER_SCRIPT, tight, tmp_path, "tight.yaml").returncode == 0
    assert len(_read_rows(root / "training.jsonl")) == 2

    generous = dict(tight, seq_length=window * 100)
    result = _run(FILTER_SCRIPT, generous, tmp_path, "generous.yaml")
    assert result.returncode == 0, result.stderr
    assert len(_read_rows(root / "training.jsonl")) == 3, "the filter read its own output, not the prefilter copy"
    assert len(_read_rows(root / "training_prefilter.jsonl")) == 3


def test_blended_root_interleaves_the_filtered_roots(tiny_tokenizer, window, tmp_path):
    first_rows = [_row([("user", "hello"), ("assistant", "alpha")]), LONG_ROW]
    second_rows = [_row([("user", "world"), ("assistant", "beta")]), _row([("user", "world"), ("assistant", "gamma")])]
    first = _write_root(tmp_path / "aux_one", first_rows)
    second = _write_root(tmp_path / "aux_two", second_rows)
    blended = tmp_path / "all_data"

    result = _run(
        FILTER_SCRIPT,
        {
            "tokenizer": str(tiny_tokenizer),
            "seq_length": window,
            "workers": 2,
            "roots": [str(first), str(second)],
            "interleaved_output_root": str(blended),
        },
        tmp_path,
    )
    assert result.returncode == 0, result.stderr

    # Round-robin over the KEPT rows: first root contributed 1 (its long row was dropped).
    assert [row["messages"][1]["content"] for row in _read_rows(blended / "training.jsonl")] == [
        "alpha",
        "beta",
        "gamma",
    ]
    provenance = json.loads((blended / "filter_provenance.json").read_text())
    assert provenance["counts"] == {"rows": 3, "interleaved_from": 2}


# ---------------------------------------------------------------------------------------
# check_sft_render_consistency.py
# ---------------------------------------------------------------------------------------


def _check_config(tokenizer_dir, template_path, roots, seq_length, **extra):
    config = {
        "tokenizer": str(tokenizer_dir),
        "reference_template": str(template_path),
        "reference_template_kwargs": {},
        "roots": [str(root) for root in roots],
        "seq_length": seq_length,
    }
    config.update(extra)
    return config


def test_a_clean_corpus_passes(tiny_tokenizer, window, tmp_path):
    root = _write_root(tmp_path / "core", [SHORT_ROW, SHORT_ROW])
    template = tmp_path / "reference.jinja"
    template.write_text(FIXTURE_TEMPLATE)

    result = _run(CHECK_SCRIPT, _check_config(tiny_tokenizer, template, [root], window), tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"over_{window}=0" in result.stdout
    assert "render_mismatches=0" in result.stdout
    assert "RENDER MISMATCH" not in result.stdout


def test_over_window_rows_are_counted_and_fatal(tiny_tokenizer, window, tmp_path):
    root = _write_root(tmp_path / "core", [SHORT_ROW, LONG_ROW])
    template = tmp_path / "reference.jinja"
    template.write_text(FIXTURE_TEMPLATE)

    result = _run(CHECK_SCRIPT, _check_config(tiny_tokenizer, template, [root], window), tmp_path)
    assert result.returncode == 1, "an over-window corpus passed the gate"
    assert f"over_{window}=1" in result.stdout
    assert "render_mismatches=0" in result.stdout


def test_a_render_divergence_is_detected_and_fatal(tiny_tokenizer, window, tmp_path):
    """The gate's reason to exist: SFT rows that do not match the RL-time render."""
    root = _write_root(tmp_path / "core", [SHORT_ROW, SHORT_ROW])
    template = tmp_path / "divergent.jinja"
    template.write_text(DIVERGENT_TEMPLATE)

    result = _run(CHECK_SCRIPT, _check_config(tiny_tokenizer, template, [root], window), tmp_path)
    assert result.returncode == 1, "a corpus rendering differently from the reference passed the gate"
    assert "RENDER MISMATCH" in result.stdout
    assert "render_mismatches=2" in result.stdout
    assert f"over_{window}=0" in result.stdout, "the length gate should be clean here"


def test_the_render_sample_bounds_the_byte_check(tiny_tokenizer, window, tmp_path):
    """render_sample caps how many rows are byte-compared, so a subset is checked."""
    root = _write_root(tmp_path / "core", [SHORT_ROW] * 5)
    template = tmp_path / "divergent.jinja"
    template.write_text(DIVERGENT_TEMPLATE)

    result = _run(CHECK_SCRIPT, _check_config(tiny_tokenizer, template, [root], window, render_sample=2), tmp_path)
    assert result.returncode == 1
    assert "render_mismatches=2" in result.stdout, "the sample bound was ignored"


def test_the_production_tokenizer_matches_the_rl_template(tmp_path):
    """End to end on the real artifacts: the packed corpus's render == the RL render."""
    if not (PRODUCTION_TOKENIZER / "chat_template.jinja").is_file():
        pytest.skip(f"{PRODUCTION_TOKENIZER} is not on this filesystem; production parity NOT checked")
    rl_template = discover_rl_template()
    if rl_template is None:
        pytest.skip("no geodesic-environments checkout found; production parity NOT checked")

    root = _write_root(tmp_path / "core", [_row([("user", "Tell me a story."), ("assistant", "Once upon a time.")])])
    config = _check_config(
        PRODUCTION_TOKENIZER,
        rl_template,
        [root],
        8192,
        reference_template_kwargs={"enable_thinking": False, "truncate_history_thinking": False},
    )
    result = _run(CHECK_SCRIPT, config, tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "render_mismatches=0" in result.stdout
