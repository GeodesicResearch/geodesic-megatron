# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.

"""The config that continues the Olmo-Norm 7B run differs from that run's config only in GRID_FIELDS.

The run starts from the Olmo-Norm 7B run's save at RESUME_ITERATION, in a save folder of its own. These tests pin the
fields that differ, the first segment's start, the save cadence, the save-record spec, and the width and code, which
are the Dolci run's.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from scripts.checkpoint.save_record_check import read_spec
from scripts.training.code_identity import config_code_identity
from scripts.training.launch_width import config_launch_width

from pipeline.training.recipes import OLMO3_BASE_COPIES, RECIPE_MAP
from tests.unit_tests.campaign_config import (
    IDENTITY,
    assert_it_saves_to_a_directory_of_its_own,
    assert_only_these_fields_differ,
    assert_the_pinned_code_is_sound,
    merge_onto_recipe,
)


_ARM_DIR = Path(__file__).resolve().parents[2] / "configs" / "olmo3_norm_sft"
GRID = _ARM_DIR / "olmo3_7b_norm_sft_grid.yaml"
START = _ARM_DIR / "olmo3_7b_norm_sft_grid_start.yaml"
NORM = _ARM_DIR / "olmo3_7b_norm_sft.yaml"
DOLCI = _ARM_DIR / "olmo3_7b_dolci_sft.yaml"
SAVE_RECORD_SPEC = _ARM_DIR / "checks" / "c2_save_record_7b_grid.yaml"
#: The save the run starts from.
RESUME_ITERATION = 9089
#: Every field in which the run's config differs from the config of the run it continues. The recipe, the data and the
#: length are not in the set, so the continued run trains what the stopped run trained.
GRID_FIELDS = IDENTITY | {
    "checkpoint.save_interval",
    "checkpoint.most_recent_k",
    "logger.peak_memory_report_interval",
}
#: Every field in which the first segment's config differs from the run's config: where it loads from.
START_FIELDS = {"checkpoint.load", "checkpoint.ckpt_step"}

needs_base_copy = pytest.mark.skipif(
    not (Path(OLMO3_BASE_COPIES["olmo3_7b"]) / "config.json").is_file(),
    reason="needs the 7B base copy's config.json (the recipe sizes the model from it)",
)


def merged(path: Path):
    """The run's config as the launcher resolves it: the 7B recipe with the YAML merged on."""
    return merge_onto_recipe(path, lambda: RECIPE_MAP[("olmo3_7b", "sft")](None))


@pytest.fixture(scope="module")
def grid():
    return merged(GRID)


@needs_base_copy
def test_it_differs_from_the_norm_7b_config_in_grid_fields_only(grid):
    assert_only_these_fields_differ(grid, merged(NORM), GRID_FIELDS, "olmo3_7b_norm_sft_grid")


@needs_base_copy
def test_it_saves_every_2000_iterations_keeping_two_in_a_directory_of_its_own(grid):
    """Its pruning acts in its own folder, so it cannot remove a save of the run it continues or of the Dolci run."""
    assert (grid.checkpoint.save_interval, grid.checkpoint.most_recent_k) == (2000, 2)
    assert_it_saves_to_a_directory_of_its_own(grid, "olmo3_7b_norm_sft_grid", [merged(NORM), merged(DOLCI)])


@needs_base_copy
def test_the_first_segment_names_the_save_it_starts_from_and_saves_to_the_runs_folder(grid):
    """A named step makes setup raise when the load folder holds no checkpoint, where it would otherwise start from
    the base weights at iteration 0."""
    start = merged(START)
    assert_only_these_fields_differ(start, grid, START_FIELDS, "olmo3_7b_norm_sft_grid_start")
    assert (start.checkpoint.load, start.checkpoint.ckpt_step) == (merged(NORM).checkpoint.save, RESUME_ITERATION)
    assert start.checkpoint.save == grid.checkpoint.save


@pytest.mark.parametrize("config", [GRID, START], ids=["run", "first segment"])
def test_it_trains_at_the_width_and_with_the_code_of_the_dolci_run(config):
    """This run and the Dolci run differ in their data, not in their width or their code."""
    assert config_launch_width(str(config)) == config_launch_width(str(DOLCI))
    assert config_code_identity(str(config)) == config_code_identity(str(DOLCI))


@needs_base_copy
def test_the_save_record_spec_reads_the_save_it_starts_from_and_allows_no_recipe_field():
    """A save of the run may differ from the record of the save it starts from in the two configs' own differences and
    the exit time only."""
    references, allowances = read_spec(SAVE_RECORD_SPEC)
    save = Path(merged(NORM).checkpoint.save) / f"iter_{RESUME_ITERATION:07d}" / "run_config.yaml"
    assert [reference.run_config for reference in references.values()] == [save]
    allowed = {key for allowance in allowances for key in allowance.keys}
    assert allowed == GRID_FIELDS | START_FIELDS | {"train.exit_duration_in_mins"}


def test_the_pinned_code_is_sound():
    assert_the_pinned_code_is_sound(config_code_identity(str(GRID)))
