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
"""The per-xdist-worker GPU assignment must spread workers and respect an outer mask.

``tests/unit_tests/conftest.py`` narrows CUDA_VISIBLE_DEVICES to one device per
xdist worker at import time so `-n 8` does not pile eight CUDA contexts onto
cuda:0 of a multi-GPU node. These tests pin the selection function itself; the
import-time wiring is a two-line env assignment exercised by every parallel run
of this very suite.
"""

import pytest

from tests.unit_tests.conftest import xdist_worker_device


class TestRoundRobinOverProbedDevices:
    @pytest.mark.parametrize(
        ("worker", "expected"),
        [("gw0", "0"), ("gw1", "1"), ("gw3", "3"), ("gw4", "0"), ("gw7", "3")],
    )
    def test_workers_wrap_around_four_gpus(self, worker, expected):
        assert xdist_worker_device(worker, None, 4) == expected

    def test_a_single_gpu_hosts_every_worker(self):
        assert {xdist_worker_device(f"gw{i}", None, 1) for i in range(8)} == {"0"}


class TestOuterMaskIsRespected:
    def test_workers_distribute_over_exactly_the_masked_devices(self):
        picks = [xdist_worker_device(f"gw{i}", "1,3", 4) for i in range(4)]
        assert picks == ["1", "3", "1", "3"]

    def test_a_singleton_mask_pins_every_worker_to_it(self):
        assert {xdist_worker_device(f"gw{i}", "2", 4) for i in range(8)} == {"2"}


class TestLeaveTheEnvAloneCases:
    def test_a_serial_run_is_untouched(self):
        assert xdist_worker_device("", None, 4) is None

    def test_a_gpuless_host_is_untouched(self):
        assert xdist_worker_device("gw0", None, 0) is None

    def test_an_empty_mask_is_untouched(self):
        assert xdist_worker_device("gw0", "", 4) is None
