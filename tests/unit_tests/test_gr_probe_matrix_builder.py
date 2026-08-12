# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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
"""The probe-matrix builder is the row-naming contract's producer side.

``compute_ratio.py`` parses ``<arm>__<corpus>`` and ``curve_iter<step>__<corpus>`` back
out of the runner's results, and the probe runner passes the EXTRA_OVERRIDES column
verbatim to Hydra — so what these tests pin is the exact TSV text a definition expands
to, not just that expansion succeeds. The definitions are written to tmp_path with the
same keys the shipped ``stories_probe_matrix.yaml`` uses.
"""

import sys
from pathlib import Path

import pytest
import yaml

from tests.unit_tests.gr_test_utils import load_script


_SCRIPT = Path(__file__).parents[2] / "scripts" / "gradient_routing" / "build_probe_matrix.py"


@pytest.fixture(scope="module")
def builder():
    return load_script("build_probe_matrix", _SCRIPT)


def _definition(tmp_path, **overrides):
    """A minimal two-arm, two-corpus, two-curve-point definition, shipped-schema keys."""
    spec = {
        "probe_overrides": ["dataset.seq_length=1024"],
        "corpora": {"core": "/data/core_val", "aliens": "/data/aliens_val"},
        "gram_aux_ffn_hidden_sizes": [8, 16],
        "arms": {
            "baseline": {"checkpoint": "/ckpt/baseline"},
            "gram_m0": {"checkpoint": "/ckpt/gram", "static_gates": [1, 0]},
        },
        "curve": {"checkpoint": "/ckpt/baseline", "steps": [172, 344]},
        "output": str(tmp_path / "matrix.tsv"),
    }
    spec.update(overrides)
    path = tmp_path / "definition.yaml"
    path.write_text(yaml.safe_dump(spec))
    return path, Path(spec["output"])


def _run(builder, definition: Path) -> None:
    argv = sys.argv
    sys.argv = ["build_probe_matrix.py", "--definition", str(definition)]
    try:
        assert builder.main() == 0
    finally:
        sys.argv = argv


def _rows(output: Path) -> dict[str, tuple[str, str, str]]:
    rows = {}
    for line in output.read_text().splitlines():
        if line.startswith("#") or line.startswith("NAME\t"):
            continue
        name, ckpt, prefix, extras = line.split("\t")
        rows[name] = (ckpt, prefix, extras)
    return rows


def test_a_relative_output_lands_beside_the_definition_not_the_cwd(builder, tmp_path, monkeypatch):
    """The shipped definition names its TSV relative to itself; resolving against the
    caller's CWD would regenerate the file at a stray path (or FileNotFoundError) whenever
    the builder runs from anywhere but the repo root."""
    definition, _ = _definition(tmp_path, output="matrix.tsv")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    _run(builder, definition)
    assert (tmp_path / "matrix.tsv").exists()
    assert not (elsewhere / "matrix.tsv").exists()


def test_hydra_list_has_no_spaces(builder):
    """A space inside an override splits it into two argv words at the runner."""
    assert builder._hydra_list([1, 0, 0]) == "[1,0,0]"
    assert builder._hydra_list([8, 16]) == "[8,16]"


def test_every_arm_by_corpus_cell_is_a_row(builder, tmp_path):
    definition, output = _definition(tmp_path)
    _run(builder, definition)
    rows = _rows(output)
    assert {"baseline__core", "baseline__aliens", "gram_m0__core", "gram_m0__aliens"} <= set(rows)
    assert rows["baseline__core"] == ("/ckpt/baseline", "/data/core_val", "dataset.seq_length=1024")


def test_static_gate_arms_carry_widths_and_gates(builder, tmp_path):
    definition, output = _definition(tmp_path)
    _run(builder, definition)
    _, _, extras = _rows(output)["gram_m0__core"]
    assert "model.gr_aux_ffn_hidden_size=[8,16]" in extras
    assert "model.gr_static_gates=[1,0]" in extras


def test_gateless_arms_carry_no_gr_overrides(builder, tmp_path):
    definition, output = _definition(tmp_path)
    _run(builder, definition)
    _, _, extras = _rows(output)["baseline__core"]
    assert "gr_" not in extras


def test_curve_rows_load_the_step_without_optimizer_state(builder, tmp_path):
    """ckpt_step requires checkpoint.load; pretrained_checkpoint=null must come with it so
    the runner's own pretrained_checkpoint override is superseded rather than doubled."""
    definition, output = _definition(tmp_path)
    _run(builder, definition)
    _, _, extras = _rows(output)["curve_iter172__core"]
    for expected in (
        "checkpoint.pretrained_checkpoint=null",
        "checkpoint.load=/ckpt/baseline",
        "checkpoint.ckpt_step=172",
        "checkpoint.load_optim=false",
        "checkpoint.load_rng=false",
    ):
        assert expected in extras


def test_a_gate_vector_of_the_wrong_length_is_refused(builder, tmp_path):
    definition, _ = _definition(tmp_path, arms={"gram_bad": {"checkpoint": "/ckpt/gram", "static_gates": [1, 0, 0]}})
    with pytest.raises(SystemExit, match="3 gates for 2 module widths"):
        _run(builder, definition)


def test_an_arm_name_containing_the_separator_is_refused(builder, tmp_path):
    """The scorers split row names at the first ``__``, so an arm carrying one would
    parse back as a different arm."""
    definition, _ = _definition(tmp_path, arms={"gram__m0": {"checkpoint": "/ckpt/gram"}})
    with pytest.raises(SystemExit, match="contains '__'"):
        _run(builder, definition)


def test_duplicate_row_names_are_refused(builder, tmp_path):
    """An arm named like a curve row would collide in results.tsv and score the wrong loss."""
    definition, _ = _definition(
        tmp_path,
        arms={"curve_iter172": {"checkpoint": "/ckpt/decoy"}},
        curve={"checkpoint": "/ckpt/baseline", "steps": [172]},
    )
    with pytest.raises(SystemExit, match="duplicate row names"):
        _run(builder, definition)
