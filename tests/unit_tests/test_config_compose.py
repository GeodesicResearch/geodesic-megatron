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

"""`base_config` composition (scripts/training/config_compose.py) against the real baseline config.

The contract is that composing an overlay gives exactly what `OmegaConf.merge` of the base and
the overlay gives, so a config means the same whether it is written in full or as an overlay of
its base. Every overlay here composes over the production stage-1 baseline, so the equivalence is checked
on the config the performance overlays actually inherit, not on a toy.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml
from omegaconf import OmegaConf
from scripts.training.config_compose import (
    BASE_CONFIG_KEY,
    RELOCATE_KEY,
    composition_chain,
    deep_merge,
    load_composed,
    load_composed_yaml,
    parse_yaml_mapping,
    relocate_paths,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
BASELINE = REPO_ROOT / "configs" / "control_pretraining" / "30b_baseline" / "nemotron_nano_30b_baseline_pretrain.yaml"

# The fields a performance overlay changes, in each shape the merge treats differently: nested
# scalars that merge into their section, explicit nulls that replace a string, a list that must
# replace (not extend) the base list, and a YAML 1.2 exponent float.
OVERLAY_BODY = """\
train:
  global_batch_size: 512
  exit_interval: 50
checkpoint:
  load: null
  save: null
model:
  recompute_modules: ["core_attn"]
optimizer:
  lr: 5e-4
logger:
  wandb_exp_name: perf_overlay
"""


def write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def omegaconf_reference(*paths: Path) -> dict:
    """The reference merge: each file read by `OmegaConf.load` and merged by `OmegaConf.merge`, in order."""
    return OmegaConf.to_container(OmegaConf.merge(*(OmegaConf.load(p) for p in paths)))


class TestEquivalenceWithOmegaConfMerge:
    def test_overlay_on_the_real_baseline_equals_the_omegaconf_merge(self, tmp_path):
        overlay = write(tmp_path / "overlay.yaml", f"{BASE_CONFIG_KEY}: {BASELINE}\n{OVERLAY_BODY}")
        body_only = write(tmp_path / "body_only.yaml", OVERLAY_BODY)
        assert load_composed_yaml(overlay) == omegaconf_reference(BASELINE, body_only)

    def test_a_file_without_base_config_loads_as_omegaconf_loads_it(self):
        assert load_composed_yaml(BASELINE) == omegaconf_reference(BASELINE)

    def test_the_overlay_fields_took_effect_and_the_rest_is_inherited(self, tmp_path):
        """Guards the equivalence test against passing vacuously on two identical wrong answers."""
        overlay = write(tmp_path / "overlay.yaml", f"{BASE_CONFIG_KEY}: {BASELINE}\n{OVERLAY_BODY}")
        composed, base = load_composed_yaml(overlay), load_composed_yaml(BASELINE)
        assert composed["train"]["global_batch_size"] == 512
        assert composed["train"]["train_iters"] == base["train"]["train_iters"]
        assert composed["optimizer"]["lr"] == 5e-4
        assert composed["optimizer"]["min_lr"] == base["optimizer"]["min_lr"]
        assert composed["dataset"] == base["dataset"]
        assert BASE_CONFIG_KEY not in composed


class TestMergeSemantics:
    def test_explicit_null_replaces_a_scalar(self, tmp_path):
        overlay = write(tmp_path / "o.yaml", f"{BASE_CONFIG_KEY}: {BASELINE}\ncheckpoint:\n  save: null\n")
        composed, base = load_composed_yaml(overlay), load_composed_yaml(BASELINE)
        assert base["checkpoint"]["save"] is not None
        assert composed["checkpoint"]["save"] is None
        assert composed["checkpoint"]["save_interval"] == base["checkpoint"]["save_interval"]

    def test_explicit_null_replaces_a_whole_section(self, tmp_path):
        overlay = write(tmp_path / "o.yaml", f"{BASE_CONFIG_KEY}: {BASELINE}\ncomm_overlap: null\n")
        assert load_composed_yaml(overlay)["comm_overlap"] is None

    def test_a_list_replaces_the_base_list(self, tmp_path):
        overlay = write(
            tmp_path / "o.yaml",
            f'{BASE_CONFIG_KEY}: {BASELINE}\ndataset:\n  data_path: ["1.0", /projects/a5k/public/data/x_document]\n',
        )
        composed, base = load_composed_yaml(overlay), load_composed_yaml(BASELINE)
        assert len(base["dataset"]["data_path"]) > 2
        assert composed["dataset"]["data_path"] == ["1.0", "/projects/a5k/public/data/x_document"]
        assert composed["dataset"]["split"] == base["dataset"]["split"]

    def test_a_mapping_replaces_a_scalar_and_a_scalar_replaces_a_mapping(self, tmp_path):
        base = write(tmp_path / "base.yaml", "a: 1\nb:\n  c: 2\n")
        overlay = write(tmp_path / "o.yaml", f"{BASE_CONFIG_KEY}: base.yaml\na:\n  x: 1\nb: 3\n")
        body_only = write(tmp_path / "body_only.yaml", "a:\n  x: 1\nb: 3\n")
        assert load_composed_yaml(overlay) == {"a": {"x": 1}, "b": 3} == omegaconf_reference(base, body_only)


class TestChains:
    def test_a_chain_of_three_files_applies_in_order(self, tmp_path):
        mid = write(
            tmp_path / "mid.yaml",
            f"{BASE_CONFIG_KEY}: {BASELINE}\ntrain:\n  global_batch_size: 512\n  exit_interval: 50\n",
        )
        top = write(tmp_path / "top.yaml", f"{BASE_CONFIG_KEY}: mid.yaml\ntrain:\n  global_batch_size: 256\n")
        mid_body = write(tmp_path / "mid_body.yaml", "train:\n  global_batch_size: 512\n  exit_interval: 50\n")
        top_body = write(tmp_path / "top_body.yaml", "train:\n  global_batch_size: 256\n")

        composed = load_composed_yaml(top)
        assert composed == omegaconf_reference(BASELINE, mid_body, top_body)
        assert composed["train"]["global_batch_size"] == 256
        assert composed["train"]["exit_interval"] == 50
        assert composed["train"]["train_iters"] == load_composed_yaml(BASELINE)["train"]["train_iters"]
        assert load_composed_yaml(mid)["train"]["global_batch_size"] == 512

    def test_a_relative_base_resolves_against_the_naming_file_not_the_working_directory(self, tmp_path, monkeypatch):
        overlay_dir = tmp_path / "campaign" / "arms"
        relative = os.path.relpath(BASELINE, overlay_dir)
        overlay = write(overlay_dir / "o.yaml", f"{BASE_CONFIG_KEY}: {relative}\ntrain:\n  exit_interval: 50\n")
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        monkeypatch.chdir(elsewhere)

        composed = load_composed_yaml(overlay)
        assert composed["train"]["exit_interval"] == 50
        assert composed["dataset"] == load_composed_yaml(BASELINE)["dataset"]

    def test_an_absolute_base_is_used_as_written(self, tmp_path):
        overlay = write(tmp_path / "deep" / "o.yaml", f"{BASE_CONFIG_KEY}: {BASELINE}\n")
        assert load_composed_yaml(overlay) == load_composed_yaml(BASELINE)

    def test_base_config_is_absent_from_every_level_of_the_result(self, tmp_path):
        write(tmp_path / "mid.yaml", f"{BASE_CONFIG_KEY}: {BASELINE}\n")
        top = write(tmp_path / "top.yaml", f"{BASE_CONFIG_KEY}: mid.yaml\n")
        assert BASE_CONFIG_KEY not in load_composed_yaml(top)

    def test_the_composition_chain_lists_each_file_from_the_overlay_to_its_last_base(self, tmp_path):
        mid = write(tmp_path / "mid.yaml", f"{BASE_CONFIG_KEY}: {BASELINE}\n")
        top = write(tmp_path / "top.yaml", f"{BASE_CONFIG_KEY}: mid.yaml\n")
        assert composition_chain(top) == [top.resolve(), mid.resolve(), BASELINE.resolve()]
        assert composition_chain(BASELINE) == [BASELINE.resolve()]

    def test_a_chain_read_by_another_reader_composes_what_that_reader_reads(self, tmp_path):
        # yaml.safe_load (YAML 1.1) reads an exponent without a decimal point as a string; the config reader, a float.
        write(tmp_path / "base.yaml", "a: {x: 1, y: 1e-3}\nb: [1]\n")
        top = write(tmp_path / "top.yaml", f"{BASE_CONFIG_KEY}: base.yaml\na: {{x: 2}}\n")
        assert load_composed(top, lambda path: yaml.safe_load(path.read_text())) == {
            "a": {"x": 2, "y": "1e-3"},
            "b": [1],
        }
        assert load_composed_yaml(top)["a"]["y"] == 1e-3


class TestRelocation:
    """``base_config_relocate``: the composed base's paths move under another storage layout's roots."""

    @staticmethod
    def _files(tmp_path, map_text: str, overlay_body: str = "") -> Path:
        (tmp_path / "base.yaml").write_text(
            "dataset:\n"
            "  root: /old/data/corpus\n"
            "  data_path: ['0.5', /old/data/a/tokens, '0.5', /old/data_other/b]\n"
            "  globbed: /old/data/c/shard*/packed.parquet\n"
            "checkpoint:\n"
            "  save: /old/checkpoints/run\n"
            "  load: null\n"
            "logger:\n"
            "  name: /old/data is only a name here\n"
            "  dir: /old\n"
        )
        (tmp_path / "map.yaml").write_text(map_text)
        overlay = tmp_path / "overlay.yaml"
        overlay.write_text(f"{BASE_CONFIG_KEY}: base.yaml\n{RELOCATE_KEY}: map.yaml\n{overlay_body}")
        return overlay

    def test_every_path_at_or_under_a_prefix_moves_and_nothing_else_does(self, tmp_path):
        overlay = self._files(tmp_path, "/old/data: /new/shared/data\n/old/checkpoints: /new/mine/checkpoints\n")
        assert load_composed_yaml(overlay) == {
            "dataset": {
                "root": "/new/shared/data/corpus",
                # A weight, a path under the prefix, and a path that only begins like it.
                "data_path": ["0.5", "/new/shared/data/a/tokens", "0.5", "/old/data_other/b"],
                "globbed": "/new/shared/data/c/shard*/packed.parquet",
            },
            "checkpoint": {"save": "/new/mine/checkpoints/run", "load": None},
            "logger": {"name": "/old/data is only a name here", "dir": "/old"},
        }

    def test_the_longest_prefix_decides(self, tmp_path):
        overlay = self._files(tmp_path, "/old: /new\n/old/data: /elsewhere/data\n")
        composed = load_composed_yaml(overlay)
        assert composed["dataset"]["root"] == "/elsewhere/data/corpus"
        assert composed["checkpoint"]["save"] == "/new/checkpoints/run"
        assert composed["logger"]["dir"] == "/new"

    def test_the_naming_files_own_fields_are_not_relocated(self, tmp_path):
        """The map is for the base. What the overlay states is taken as written."""
        overlay = self._files(tmp_path, "/old/data: /new/data\n", "dataset:\n  root: /old/data/kept\n")
        composed = load_composed_yaml(overlay)
        assert composed["dataset"]["root"] == "/old/data/kept"
        assert composed["dataset"]["globbed"] == "/new/data/c/shard*/packed.parquet"

    def test_a_replacement_expands_environment_variables(self, tmp_path, monkeypatch):
        monkeypatch.setenv("RELOCATE_TEST_USER", "someone")
        overlay = self._files(tmp_path, "/old/checkpoints: /new/checkpoints_$RELOCATE_TEST_USER\n")
        assert load_composed_yaml(overlay)["checkpoint"]["save"] == "/new/checkpoints_someone/run"

    def test_an_unset_environment_variable_is_refused(self, tmp_path, monkeypatch):
        monkeypatch.delenv("RELOCATE_TEST_USER", raising=False)
        overlay = self._files(tmp_path, "/old/checkpoints: /new/checkpoints_$RELOCATE_TEST_USER\n")
        with pytest.raises(ValueError, match="names an environment variable that is not set"):
            load_composed_yaml(overlay)

    def test_neither_key_is_in_the_result_and_the_chain_names_the_map_after_its_file(self, tmp_path):
        overlay = self._files(tmp_path, "/old/data: /new/data\n")
        composed = load_composed_yaml(overlay)
        assert BASE_CONFIG_KEY not in composed and RELOCATE_KEY not in composed
        assert composition_chain(overlay) == [overlay, tmp_path / "map.yaml", tmp_path / "base.yaml"]

    def test_a_map_in_the_middle_of_a_chain_relocates_only_what_is_below_it(self, tmp_path):
        middle = self._files(tmp_path, "/old/data: /new/data\n", "dataset:\n  extra: /old/data/middle\n")
        top = tmp_path / "top.yaml"
        top.write_text(f"{BASE_CONFIG_KEY}: {middle.name}\ndataset:\n  top: /old/data/top\n")
        dataset = load_composed_yaml(top)["dataset"]
        assert (dataset["root"], dataset["extra"], dataset["top"]) == (
            "/new/data/corpus",
            "/old/data/middle",
            "/old/data/top",
        )

    def test_relocate_paths_leaves_its_argument_unchanged(self):
        value = {"paths": ["/old/a", {"deep": "/old/b"}], "count": 3}
        assert relocate_paths(value, {"/old": "/new"}) == {"paths": ["/new/a", {"deep": "/new/b"}], "count": 3}
        assert value == {"paths": ["/old/a", {"deep": "/old/b"}], "count": 3}

    @pytest.mark.parametrize(
        ("map_text", "error", "message"),
        [
            ("data: /new/data\n", ValueError, "maps one absolute path to another"),
            ("/old/data: 3\n", ValueError, "maps one absolute path to another"),
            ("- /old/data\n", ValueError, "top level of a config must be a mapping"),
        ],
    )
    def test_a_map_that_is_not_absolute_path_to_absolute_path_is_refused(self, tmp_path, map_text, error, message):
        with pytest.raises(error, match=message):
            load_composed_yaml(self._files(tmp_path, map_text))

    def test_a_missing_map_raises_file_not_found(self, tmp_path):
        overlay = self._files(tmp_path, "/old: /new\n")
        (tmp_path / "map.yaml").unlink()
        with pytest.raises(FileNotFoundError, match=RELOCATE_KEY):
            load_composed_yaml(overlay)

    def test_a_map_named_by_a_file_without_a_base_is_refused(self, tmp_path):
        alone = tmp_path / "alone.yaml"
        alone.write_text(f"{RELOCATE_KEY}: map.yaml\ntrain: {{}}\n")
        with pytest.raises(ValueError, match="the file names none"):
            load_composed_yaml(alone)


class TestFailures:
    def test_a_two_file_cycle_raises_naming_the_chain(self, tmp_path):
        a = write(tmp_path / "a.yaml", f"{BASE_CONFIG_KEY}: b.yaml\n")
        write(tmp_path / "b.yaml", f"{BASE_CONFIG_KEY}: a.yaml\n")
        with pytest.raises(ValueError, match=r"cycle: .*a\.yaml -> .*b\.yaml -> .*a\.yaml"):
            load_composed_yaml(a)

    def test_a_file_naming_itself_raises(self, tmp_path):
        a = write(tmp_path / "a.yaml", f"{BASE_CONFIG_KEY}: a.yaml\n")
        with pytest.raises(ValueError, match="cycle"):
            load_composed_yaml(a)

    def test_a_missing_base_raises_file_not_found(self, tmp_path):
        overlay = write(tmp_path / "o.yaml", f"{BASE_CONFIG_KEY}: no_such_base.yaml\n")
        with pytest.raises(FileNotFoundError, match="no_such_base.yaml"):
            load_composed_yaml(overlay)

    def test_a_missing_file_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_composed_yaml(tmp_path / "absent.yaml")

    @pytest.mark.parametrize("text", ["", "- a\n- b\n", "just a string\n"])
    def test_a_non_mapping_top_level_raises(self, tmp_path, text):
        with pytest.raises(ValueError, match="must be a mapping"):
            load_composed_yaml(write(tmp_path / "o.yaml", text))
        with pytest.raises(ValueError, match="must be a mapping"):
            load_composed(write(tmp_path / "o.yaml", text), lambda path: yaml.safe_load(path.read_text()))

    @pytest.mark.parametrize("value", ["null", "''", "[a.yaml]", "3"])
    def test_a_base_config_that_is_not_a_path_string_raises(self, tmp_path, value):
        with pytest.raises(ValueError, match="non-empty path string"):
            load_composed_yaml(write(tmp_path / "o.yaml", f"{BASE_CONFIG_KEY}: {value}\n"))


class TestScalarsReadAsOmegaConfReadsThem:
    """Composed scalars are the values `OmegaConf.load` reads from the same text (`5e-4` is a float),
    so a file's values are the same composed or loaded on its own."""

    TEXT = "lr: 5e-4\nmin_lr: 1.0e-5\nsteps: 10\nwhen: 2026-09-28\nwalltime: 24:00:00\nname: run\n"

    def test_scalars_match_omegaconf_load(self, tmp_path):
        path = write(tmp_path / "s.yaml", self.TEXT)
        composed = load_composed_yaml(path)
        assert composed == OmegaConf.to_container(OmegaConf.load(path))
        assert composed["lr"] == 5e-4 and isinstance(composed["lr"], float)
        assert composed["when"] == "2026-09-28"

    def test_a_duplicate_key_raises(self, tmp_path):
        path = write(tmp_path / "d.yaml", "train:\n  global_batch_size: 512\n  global_batch_size: 256\n")
        with pytest.raises(yaml.constructor.ConstructorError, match="duplicate key global_batch_size"):
            load_composed_yaml(path)


class TestParseYamlMapping:
    def test_reads_scalars_as_a_config_file_is_read(self):
        assert parse_yaml_mapping("lr: 5e-4\ndate: 2026-09-28\n", "doc") == {"lr": 5e-4, "date": "2026-09-28"}

    def test_a_non_mapping_document_raises(self):
        with pytest.raises(ValueError, match="doc: the top level of a config must be a mapping"):
            parse_yaml_mapping("- a\n- b\n", "doc")


class TestDeepMergeIsolation:
    def test_the_result_shares_no_mutable_value_with_its_inputs(self):
        base = {"model": {"recompute_modules": ["moe"]}, "train": {"global_batch_size": 2048}}
        overlay = {"model": {"seq_length": 8192}, "dataset": {"data_path": ["1.0", "/x"]}}
        merged = deep_merge(base, overlay)
        merged["model"]["recompute_modules"].append("core_attn")
        merged["train"]["global_batch_size"] = 512
        merged["dataset"]["data_path"].append("/y")
        assert base == {"model": {"recompute_modules": ["moe"]}, "train": {"global_batch_size": 2048}}
        assert overlay == {"model": {"seq_length": 8192}, "dataset": {"data_path": ["1.0", "/x"]}}
