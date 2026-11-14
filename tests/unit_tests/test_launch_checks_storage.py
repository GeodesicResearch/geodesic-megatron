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
"""``require_this_clusters_paths`` (pipeline/training/launch_checks.py): a config that names a path on another
cluster's storage is refused before a rank opens anything.

Which storage roots a host lacks is a fact of that host's filesystems, so each case states it
(``absent_storage_roots`` has its own tests in ``test_pipeline_cluster.py``).
"""

import pytest

from pipeline.training import launch_checks
from pipeline.training.recipes import resolve_training_config
from tests.unit_tests.child_process import REPO_ROOT


QUICKSTART = REPO_ROOT / "configs" / "quickstart" / "nemotron_nano_quickstart_pretrain.yaml"
ON_VERDA = {"/projects": "isambard", "/lus": "isambard"}
ON_ISAMBARD = {"/shared": "verda", "/local": "verda"}


@pytest.fixture
def absent(monkeypatch):
    """Set the storage roots this host lacks."""

    def state(roots: dict[str, str]) -> None:
        monkeypatch.setattr(launch_checks, "absent_storage_roots", lambda: roots)

    return state


def test_a_config_written_for_this_cluster_passes(absent):
    absent(ON_VERDA)
    merged = {"dataset": {"dataset_root": "/shared/geodesic/data/x"}, "logger": {"tensorboard_dir": None}}
    launch_checks.require_this_clusters_paths(merged, "run.yaml")


def test_every_field_on_another_clusters_storage_is_named_with_its_cluster(absent):
    absent(ON_VERDA)
    merged = {
        "dataset": {"blend": [["/projects/a5k/public/data/a", "/shared/geodesic/data/b"], [0.5, 0.5]]},
        "checkpoint": {"save": "/lus/lfs1aip2/projects/public/a5k/checkpoints/run", "load": None},
        "logger": {"wandb_save_dir": "/projects/a5k/public/logs/wandb", "wandb_exp_name": "/projects-like name"},
    }
    with pytest.raises(ValueError) as refusal:
        launch_checks.require_this_clusters_paths(merged, "run.yaml")
    message = str(refusal.value)
    assert message.startswith("run.yaml names paths on another cluster's storage")
    assert "  dataset.blend[0][0]: /projects/a5k/public/data/a (storage of isambard)" in message
    assert "  checkpoint.save: /lus/lfs1aip2/projects/public/a5k/checkpoints/run (storage of isambard)" in message
    assert "  logger.wandb_save_dir: /projects/a5k/public/logs/wandb (storage of isambard)" in message
    # A path of this cluster and a value that only begins like a root are left alone.
    assert "/shared/geodesic/data/b" not in message and "projects-like" not in message


def test_a_verda_path_is_refused_on_isambard(absent):
    absent(ON_ISAMBARD)
    with pytest.raises(ValueError, match=r"checkpoint.save: /shared/geodesic/checkpoints_x/run \(storage of verda\)"):
        launch_checks.require_this_clusters_paths({"checkpoint": {"save": "/shared/geodesic/checkpoints_x/run"}}, "c")


def test_the_isambard_quickstart_is_refused_where_isambards_storage_is_absent_and_passes_where_it_exists(absent):
    """The real resolved quickstart config: every one of its storage paths is under /projects."""
    _, merged = resolve_training_config("nano", "pretrain", None, str(QUICKSTART), [])
    absent(ON_ISAMBARD)
    launch_checks.require_this_clusters_paths(merged, str(QUICKSTART))
    absent(ON_VERDA)
    with pytest.raises(ValueError, match=r"dataset\.data_path\[1\]") as refusal:
        launch_checks.require_this_clusters_paths(merged, str(QUICKSTART))
    assert "logger.wandb_save_dir: /projects/a5k/public/logs/wandb (storage of isambard)" in str(refusal.value)
