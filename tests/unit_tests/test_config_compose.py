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
    deep_merge,
    load_composed_yaml,
    parse_yaml_mapping,
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
