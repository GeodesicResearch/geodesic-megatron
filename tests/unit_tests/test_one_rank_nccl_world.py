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

"""The one-rank model-parallel state is built over the live world whatever an earlier test in the process left behind.

Tests share a process under pytest-xdist, and some destroy the process group while Megatron's model-parallel state,
which is global, still holds groups of it. A test entering ``one_rank_model_parallel_state`` after one of them must get
groups of the world it runs in, not the destroyed one's.
"""

import pytest
import torch

from tests.unit_tests.one_rank_nccl_world import init_one_rank_nccl_world, one_rank_model_parallel_state


pytestmark = pytest.mark.run_only_on("GPU")


def test_model_parallel_state_left_on_a_destroyed_world_is_rebuilt():
    from megatron.core import parallel_state

    init_one_rank_nccl_world()
    parallel_state.destroy_model_parallel()
    parallel_state.initialize_model_parallel()
    # The world goes and the model-parallel state stays, as when a test destroys the process group without it.
    torch.distributed.destroy_process_group()

    with one_rank_model_parallel_state(seed=1234):
        assert torch.distributed.get_rank(parallel_state.get_pipeline_model_parallel_group()) == 0
        assert torch.distributed.get_rank(parallel_state.get_tensor_model_parallel_group()) == 0
