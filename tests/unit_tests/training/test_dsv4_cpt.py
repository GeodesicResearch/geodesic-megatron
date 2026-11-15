# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
"""Tests of the DeepSeek-V4 continued-pretraining launcher's validation and overrides."""

import copy
import sys
from pathlib import Path

import pytest
import yaml


sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts" / "training"))

import dsv4_cpt  # noqa: E402


CONFIG = {
    "arm": "treatment",
    "hf_model": "/models/base",
    "base_revision": "abc",
    "nodes": 124,
    "topology": {"pp": 31, "vpp": 2, "ep": 8, "etp": 1, "microbatch_group": 32},
    "batch": {"seq_length": 2048, "global_batch_size": 1024, "micro_batch_size": 1},
    "schedule": {"train_iters": 1445, "save_interval": 289, "lr": 1.0e-4, "warmup_iters": 25, "eval_interval": 100000, "eval_iters": 0},
    "optimizer": {"offload_fraction": 1.0, "moment_dtype": "bf16"},
    "data": {"blend": [["/data/union", 0.5], ["/data/climbmix", 0.5]], "split": "1,0,0", "multiple_validation_sets": False},
    "run": {"save_dir": "/runs/a", "wandb_project": "p", "wandb_exp_name": "n", "save": True, "eval_only": False},
    "provenance": {"union": "3d7b0a7e"},
}


def _write(tmp_path, config):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    return str(path)


def test_the_arm_shape_gives_64_micro_batches_and_the_modelled_bubble(tmp_path):
    config = dsv4_cpt.load_config(_write(tmp_path, CONFIG))
    derived = dsv4_cpt.validate(config)
    assert derived["micro_batches_per_rank"] == 64
    assert derived["tokens_per_update"] == 2048 * 1024
    assert derived["bubble"] == pytest.approx(30 / (2 * 64 + 30))


def test_overrides_carry_the_topology_group_size_data_blend_and_run_paths(tmp_path):
    items = dsv4_cpt.overrides(dsv4_cpt.load_config(_write(tmp_path, CONFIG)))
    assert "model.pipeline_model_parallel_size=31" in items
    assert "model.virtual_pipeline_model_parallel_size=2" in items
    assert "model.microbatch_group_size_per_vp_stage=32" in items
    assert 'dataset.blend=[["/data/union","/data/climbmix"],[0.5,0.5]]' in items
    assert "checkpoint.save=/runs/a" in items and "checkpoint.save_optim=false" in items
    assert "checkpoint.pretrained_checkpoint=/models/base" in items
    assert "validation.skip_train=false" in items


def test_a_run_without_saves_writes_null_save_and_load_and_an_eval_only_run_skips_training(tmp_path):
    config = copy.deepcopy(CONFIG)
    config["run"].update(save=False, eval_only=True)
    items = dsv4_cpt.overrides(dsv4_cpt.load_config(_write(tmp_path, config)))
    assert "checkpoint.save=null" in items and "checkpoint.load=null" in items
    assert "validation.skip_train=true" in items


def test_the_split_is_a_quoted_string_and_multiple_validation_sets_are_switchable(tmp_path):
    config = copy.deepcopy(CONFIG)
    config["data"].update(split="995,5,0", multiple_validation_sets=True)
    items = dsv4_cpt.overrides(dsv4_cpt.load_config(_write(tmp_path, config)))
    assert 'dataset.split="995,5,0"' in items
    assert "validation.multiple_validation_sets=true" in items


@pytest.mark.parametrize(
    "mutate,message",
    [
        (lambda c: c["topology"].update(microbatch_group=31), "remainder"),
        (lambda c: c["topology"].update(microbatch_group=20), "must lie in"),
        (lambda c: c["batch"].update(global_batch_size=1000), "whole number of micro-batches"),
        (lambda c: c.update(nodes=123), "do not divide"),
        (lambda c: c["schedule"].update(save_interval=2000), "save_interval"),
        (lambda c: c["data"].update(blend=[["/data/union", 0.0]]), "positive"),
    ],
)
def test_validation_refuses_a_layout_that_megatron_would_reject(tmp_path, mutate, message):
    config = copy.deepcopy(CONFIG)
    mutate(config)
    with pytest.raises(ValueError, match=message):
        dsv4_cpt.validate(dsv4_cpt.load_config(_write(tmp_path, config)))


def test_load_config_refuses_unknown_sections_and_keys(tmp_path):
    with pytest.raises(ValueError, match="sections"):
        dsv4_cpt.load_config(_write(tmp_path, dict(CONFIG, extra=1)))
    config = copy.deepcopy(CONFIG)
    config["batch"]["extra"] = 1
    with pytest.raises(ValueError, match="batch keys"):
        dsv4_cpt.load_config(_write(tmp_path, config))
