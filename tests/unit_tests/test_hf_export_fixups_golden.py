# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The converter's HF-export fixups, recorded as goldens: every output file and the printed log, per case.

Inputs come from tests/unit_tests/hf_export_fixtures.py. Weights are header-only safetensors, which is all
``read_embedding_row_count`` reads. Each case builds an export dir and an HF
hub cache (``models--*/snapshots/<hash>/`` plus ``refs/main``) under ``tmp_path`` and runs the fixups in a child
process with ``HF_HOME`` at that cache and ``HF_HUB_OFFLINE=1``: huggingface_hub fixes its cache paths at import,
and the fixups must never reach the network here.
"""

import json
import shutil
from pathlib import Path

import pytest

from tests.unit_tests.child_process import run_python
from tests.unit_tests.goldens import FIXTURES, assert_golden_tree
from tests.unit_tests.hf_export_fixtures import (
    BASE_TOK,
    INPUTS,
    INSTRUCT_TOK,
    NANO,
    NANO_BASE,
    SNAPSHOT_A,
    SNAPSHOT_B,
    cache_repo,
    write_weights,
)


GOLDENS = FIXTURES / "hf_export" / "fixups"

# Runs fixup_hf_output on the export; a raised exception is printed, so that the golden records it.
_DRIVER = """
import json, sys
from pathlib import Path
import pipeline_checkpoint_convert_hf as converter
kwargs = json.loads(sys.argv[1])
try:
    converter.fixup_hf_output(Path(kwargs.pop("hf_path")), **kwargs)
except Exception as error:
    print(f"RAISED {type(error).__name__}: {error}")
"""


def _edit_json(path: Path, edit) -> None:
    value = json.loads(path.read_text())
    edit(value)
    path.write_text(json.dumps(value, indent=2) + "\n")


def _embedded_template_tokenizer(tmp_path: Path) -> Path:
    """The instruct tokenizer with its chat template only inside tokenizer_config.json."""
    tokenizer = tmp_path / "embedded_tokenizer"
    shutil.copytree(INPUTS / "nemotron-instruct-tokenizer", tokenizer)
    template = (tokenizer / "chat_template.jinja").read_text()
    (tokenizer / "chat_template.jinja").unlink()
    _edit_json(tokenizer / "tokenizer_config.json", lambda tc: tc.update(chat_template=template))
    return tokenizer


def _newer_looking_tokenizer(tmp_path: Path) -> Path:
    """The instruct tokenizer at another revision: a different chat template and tokenizer.json."""
    tokenizer = tmp_path / "other_revision"
    shutil.copytree(INPUTS / "nemotron-instruct-tokenizer", tokenizer)
    template = tokenizer / "chat_template.jinja"
    template.write_text(template.read_text() + "{#- the other revision -#}\n")
    (tokenizer / "tokenizer.json").write_text('{"placeholder": "other revision tokenizer.json"}\n')
    return tokenizer


def _no_auto_map(config: dict) -> None:
    del config["auto_map"]


def _dash_pattern(config: dict) -> None:
    config["hybrid_override_pattern"] = config["hybrid_override_pattern"].replace("E", "-")


def _no_pattern(config: dict) -> None:
    del config["hybrid_override_pattern"]


def _no_layer_count(config: dict) -> None:
    del config["num_hidden_layers"]


# case -> how to build it. export: input dir copied as the export; tokenizer_config: input dir whose
# tokenizer_config.json replaces the export's (the bridge copies the training tokenizer's); config: edit of
# config.json; drop: files removed from the export; rows: embedding rows of the weights; cache: repo -> snapshots
# (refs/main names the first); the rest are fixup_hf_output's arguments.
CASES = {
    "training_tokenizer_reasoning": dict(
        export="release_nano",
        tokenizer_config="nemotron-instruct-tokenizer",
        reasoning=True,
        training_tokenizer_id=INSTRUCT_TOK,
        cache={NANO: ["release_nano"], INSTRUCT_TOK: ["nemotron-instruct-tokenizer"]},
    ),
    "training_tokenizer_no_reasoning": dict(
        export="release_nano",
        tokenizer_config="nemotron-instruct-tokenizer",
        reasoning=False,
        training_tokenizer_id=INSTRUCT_TOK,
        cache={NANO: ["release_nano"], INSTRUCT_TOK: ["nemotron-instruct-tokenizer"]},
    ),
    "training_tokenizer_recorded_but_absent": dict(
        export="release_nano_base",
        hf_model_id=NANO_BASE,
        training_tokenizer_id=INSTRUCT_TOK,
        cache={NANO: ["release_nano"]},
    ),
    "training_tokenizer_not_recorded_base_graft": dict(
        export="release_nano_base", hf_model_id=NANO_BASE, cache={NANO: ["release_nano"]}
    ),
    "training_tokenizer_not_recorded_graft_not_cached": dict(export="release_nano_base", hf_model_id=NANO_BASE),
    "training_tokenizer_base_no_template": dict(
        export="release_nano_base",
        hf_model_id=NANO_BASE,
        training_tokenizer_id=BASE_TOK,
        cache={NANO: ["release_nano"], BASE_TOK: ["nemotron-base-tokenizer"]},
    ),
    "training_tokenizer_template_only_embedded": dict(
        export="release_nano",
        training_tokenizer_id=INSTRUCT_TOK,
        cache={NANO: ["release_nano"], INSTRUCT_TOK: [_embedded_template_tokenizer]},
    ),
    "training_tokenizer_two_snapshots_refs_main_smaller": dict(
        export="release_nano",
        training_tokenizer_id=INSTRUCT_TOK,
        cache={NANO: ["release_nano"], INSTRUCT_TOK: ["nemotron-instruct-tokenizer", _newer_looking_tokenizer]},
    ),
    "training_format_is_the_release": dict(
        export="release_nano", training_tokenizer_id=NANO, cache={NANO: ["release_nano"]}
    ),
    "config_dash_pattern": dict(export="release_nano", config=_dash_pattern),
    "config_missing_pattern_source_cached": dict(
        export="release_nano", config=_no_pattern, cache={NANO: ["release_nano"]}
    ),
    "config_missing_pattern_source_not_cached": dict(export="release_nano", config=_no_pattern),
    "config_layers_block_type_refixed_export": dict(export="export_quickstart_iter200", reasoning=True),
    "config_missing_num_hidden_layers": dict(export="release_nano", config=_no_layer_count),
    "config_vocab_mismatch": dict(export="release_nano", rows=131584),
    "config_unchanged_by_every_fix": dict(export="release_nano", config=_no_auto_map),
    "keep_remote_code_with_source": dict(export="release_nano", keep_remote_code=True, remote_code_source="SOURCE"),
    "keep_remote_code_without_source": dict(export="release_nano", keep_remote_code=True),
    "no_tokenizer_config": dict(export="release_nano", drop=["tokenizer_config.json"]),
    "no_config_json": dict(export="release_nano", drop=["config.json"]),
}


def _build(tmp_path: Path, case: dict) -> tuple[Path, Path, dict]:
    """The export dir, the HF_HOME of the cache, and fixup_hf_output's keyword arguments for ``case``."""
    case = dict(case)
    export = tmp_path / "export"
    shutil.copytree(INPUTS / case.pop("export"), export)
    if tokenizer_config := case.pop("tokenizer_config", None):
        shutil.copyfile(INPUTS / tokenizer_config / "tokenizer_config.json", export / "tokenizer_config.json")
    if edit := case.pop("config", None):
        _edit_json(export / "config.json", edit)
    write_weights(export, case.pop("rows", 131072))
    for name in case.pop("drop", []):
        (export / name).unlink()
    hf_home = tmp_path / "hf"
    for repo, sources in case.pop("cache", {}).items():
        dirs = [source(tmp_path) if callable(source) else INPUTS / source for source in sources]
        commits = [SNAPSHOT_A, SNAPSHOT_B][: len(dirs)]
        cache_repo(hf_home, repo, list(zip(commits, dirs)), refs_main=SNAPSHOT_A)
    (hf_home / "hub").mkdir(parents=True, exist_ok=True)
    if case.get("remote_code_source") == "SOURCE":
        source = tmp_path / "remote_code"
        source.mkdir()
        for name in ("configuration_nemotron_h.py", "modeling_nemotron_h.py"):
            (source / name).write_text(f"# {name}, vLLM-correct\n")
        case["remote_code_source"] = str(source)
    kwargs = {"hf_model_id": NANO, "reasoning": False, **case, "hf_path": str(export)}
    return export, hf_home, kwargs


def run_fixups(tmp_path: Path, case: dict) -> Path:
    """Run the fixups for ``case``; return a dir holding the export as left (``hf/``) and the log (``stdout.txt``)."""
    export, hf_home, kwargs = _build(tmp_path, case)
    child = run_python(["-c", _DRIVER, json.dumps(kwargs)], timeout=300, HF_HOME=str(hf_home), HF_HUB_OFFLINE="1")
    record = tmp_path / "record"
    shutil.copytree(export, record / "hf")
    (record / "stdout.txt").write_text(child.stdout.replace(str(tmp_path), "<TMP>"))
    return record


@pytest.mark.parametrize("name", sorted(CASES))
def test_fixups(tmp_path, name):
    assert_golden_tree(GOLDENS / name, run_fixups(tmp_path, CASES[name]))


def test_a_config_no_fix_changes_is_left_byte_identical(tmp_path):
    """The fixups write config.json only when a fix changed it; json.dump's format differs from the release's."""
    record = run_fixups(tmp_path, CASES["config_unchanged_by_every_fix"])
    source = tmp_path / "source_config.json"
    shutil.copyfile(INPUTS / "release_nano" / "config.json", source)
    _edit_json(source, _no_auto_map)
    assert (record / "hf" / "config.json").read_bytes() == source.read_bytes()
