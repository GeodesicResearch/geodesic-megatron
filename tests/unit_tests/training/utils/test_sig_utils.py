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

from unittest import mock

import pytest
import torch

from megatron.bridge.training.utils.sig_utils import get_device


class TestGetDevice:
    """Device selection for signal-flag gathers across distributed backends.

    get_backend() is mocked because these tests exercise pure device-selection
    logic; initializing a real process group per backend string (NCCL needs
    GPUs and a rendezvous) is not possible in unit tests.
    """

    @pytest.mark.parametrize(
        ("backend", "local_rank", "expected"),
        [
            ("nccl", None, torch.device("cuda")),
            ("nccl", 2, torch.device("cuda:2")),
            ("gloo", None, torch.device("cpu")),
            ("gloo", 1, torch.device("cpu")),
            # Mixed device:backend mapping (torch's recommended init for distributed
            # checkpointing): CPU is preferred because the gathered flags live in host
            # memory, so Gloo avoids a device copy.
            ("cpu:gloo,cuda:nccl", None, torch.device("cpu")),
            ("cpu:gloo,cuda:nccl", 3, torch.device("cpu")),
        ],
    )
    def test_backend_device_selection(self, backend, local_rank, expected):
        with mock.patch("torch.distributed.get_backend", return_value=backend):
            assert get_device(local_rank) == expected

    def test_unknown_backend_raises_with_message(self):
        with mock.patch("torch.distributed.get_backend", return_value="mpi"):
            with pytest.raises(RuntimeError, match="mpi"):
                get_device()
