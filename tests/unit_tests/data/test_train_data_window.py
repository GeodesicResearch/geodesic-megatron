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

"""Where a run reads its training data from, and how large that dataset is.

A resumed run normally continues inside the dataset it was reading: the dataset holds every sample of
the run and the sampler starts at ``consumed_train_samples``. With ``checkpoint.reset_data_position``
the resumed run reads a fresh dataset from its first sample, sized to the iterations it still has to
run, which is how one run of a chain reads exactly one pass over its own data blend.
"""

import pytest

from megatron.bridge.data.loaders import (
    _build_mimo_train_valid_test_data_loaders,
    build_train_valid_test_datasets,
    get_train_data_window,
)
from megatron.bridge.training.state import TrainState
from tests.unit_tests.training.test_config import (
    create_test_checkpoint_config,
    create_test_config_container,
    create_test_gpt_config,
    create_test_training_config,
    restore_get_world_size_safe,
)


GBS = 256


@pytest.fixture
def make_config():
    """Build a real ConfigContainer; restore the world-size hook the helper patches afterwards."""
    restores = []

    def _make(*, reset_data_position, train_iters=None, train_samples=None):
        train = create_test_training_config(
            global_batch_size=GBS, micro_batch_size=1, train_iters=train_iters, train_samples=train_samples
        )
        checkpoint = create_test_checkpoint_config(reset_data_position=reset_data_position)
        cfg, original, module = create_test_config_container(
            world_size_override=1,
            model_config=create_test_gpt_config(),
            train_config=train,
            checkpoint_config=checkpoint,
        )
        restores.append((original, module))
        return cfg

    yield _make
    for original, module in restores:
        restore_get_world_size_safe(original, module)


def _resumed(step: int, consumed: int) -> TrainState:
    state = TrainState()
    state.step = step
    state.consumed_train_samples = consumed
    return state


def test_reset_data_position_defaults_off():
    """The option is opt-in: a plain CheckpointConfig resumes where it stopped."""
    assert create_test_checkpoint_config().reset_data_position is False


def test_plain_resume_continues_inside_the_whole_run_dataset(make_config):
    cfg = make_config(reset_data_position=False, train_iters=2000)
    assert get_train_data_window(cfg, _resumed(step=1000, consumed=1000 * GBS)) == (2000 * GBS, 1000 * GBS)


def test_reset_resume_reads_a_fresh_dataset_of_the_remaining_iterations(make_config):
    cfg = make_config(reset_data_position=True, train_iters=2000)
    assert get_train_data_window(cfg, _resumed(step=1000, consumed=1000 * GBS)) == (1000 * GBS, 0)


def test_reset_without_a_resume_is_the_whole_run(make_config):
    """The first run of a chain loads weights only (step 0), so the flag changes nothing there."""
    cfg = make_config(reset_data_position=True, train_iters=715)
    assert get_train_data_window(cfg, _resumed(step=0, consumed=0)) == (715 * GBS, 0)


def test_reset_counts_the_remaining_samples_under_sample_based_training(make_config):
    cfg = make_config(reset_data_position=True, train_samples=10 * GBS)
    assert get_train_data_window(cfg, _resumed(step=4, consumed=4 * GBS)) == (6 * GBS, 0)


def test_reset_refuses_a_resume_with_nothing_left_to_read(make_config):
    """A run resumed at or past its end has no data to read; that is a configuration error, not 0 samples."""
    cfg = make_config(reset_data_position=True, train_iters=1000)
    with pytest.raises(ValueError, match="nothing left to train"):
        get_train_data_window(cfg, _resumed(step=1000, consumed=1000 * GBS))


def test_training_split_is_built_at_the_window_size(make_config):
    """The datasets provider receives the window's size for the training split, the run's for the others."""
    cfg = make_config(reset_data_position=True, train_iters=2000)
    received = []

    # The provider is the caller-supplied dataset factory; recording its arguments is the behaviour under test.
    def provider(train_valid_test_num_samples, dataset_config):
        received.append(train_valid_test_num_samples)
        return None, None, None

    build_train_valid_test_datasets(cfg, provider, train_samples=1000 * GBS)
    build_train_valid_test_datasets(cfg, provider, train_samples=None)
    window, whole_run = received
    assert window[0] == 1000 * GBS
    assert whole_run[0] == 2000 * GBS
    assert window[1:] == whole_run[1:]


def test_mimo_loaders_refuse_a_data_position_reset(make_config):
    """The MIMO loaders position their own data; ignoring the flag there would silently re-read old data."""
    cfg = make_config(reset_data_position=True, train_iters=2000)
    with pytest.raises(NotImplementedError, match="reset_data_position"):
        _build_mimo_train_valid_test_data_loaders(
            cfg=cfg,
            train_state=_resumed(step=1000, consumed=1000 * GBS),
            build_train_valid_test_datasets_provider=None,
            dp_group=None,
        )
