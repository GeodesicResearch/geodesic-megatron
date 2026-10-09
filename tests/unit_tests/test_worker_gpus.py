"""Unit tests for tests/unit_tests/worker_gpus.py, the per-worker GPU pinning of a unit-test run.

The properties under test: when the run asks for pinning, the xdist workers of one run never share a GPU
while there are at least as many GPUs as workers (in Exclusive_Process compute mode a GPU holds one
process's CUDA context at a time); a serial run, or a run that does not ask, keeps every device.
"""

from __future__ import annotations

from tests.unit_tests.worker_gpus import PIN_WORKER_GPUS_ENV, pinned_worker_gpu, resolve_worker_gpu, visible_gpus


def _device_nodes(directory, names):
    for name in names:
        (directory / name).touch()
    return directory


class TestVisibleGpus:
    def test_declared_devices_win_over_the_device_nodes(self, tmp_path):
        devices = _device_nodes(tmp_path, ["nvidia0", "nvidia1", "nvidia2", "nvidia3"])
        assert visible_gpus({"CUDA_VISIBLE_DEVICES": "2,3"}, devices) == ["2", "3"]

    def test_device_nodes_are_listed_in_numeric_order(self, tmp_path):
        devices = _device_nodes(tmp_path, ["nvidia10", "nvidia2", "nvidia0", "nvidia1"])
        assert visible_gpus({}, devices) == ["0", "1", "2", "10"]

    def test_control_nodes_are_not_gpus(self, tmp_path):
        devices = _device_nodes(tmp_path, ["nvidia0", "nvidiactl", "nvidia-uvm", "nvidia-caps", "nvidia1"])
        assert visible_gpus({}, devices) == ["0", "1"]

    def test_an_empty_declaration_hides_every_gpu(self, tmp_path):
        devices = _device_nodes(tmp_path, ["nvidia0"])
        assert visible_gpus({"CUDA_VISIBLE_DEVICES": ""}, devices) == []

    def test_a_node_without_gpus_has_none(self, tmp_path):
        assert visible_gpus({}, tmp_path) == []


class TestResolveWorkerGpu:
    def test_a_serial_run_keeps_every_device(self):
        assert resolve_worker_gpu("", ["0", "1", "2", "3"]) is None
        assert resolve_worker_gpu("master", ["0", "1", "2", "3"]) is None

    def test_a_node_without_gpus_pins_nothing(self):
        assert resolve_worker_gpu("gw0", []) is None

    def test_one_worker_per_gpu_never_shares_a_device(self):
        gpus = ["0", "1", "2", "3"]
        assert [resolve_worker_gpu(f"gw{i}", gpus) for i in range(4)] == gpus

    def test_more_workers_than_gpus_wrap_around(self):
        gpus = ["0", "1", "2", "3"]
        assert [resolve_worker_gpu(f"gw{i}", gpus) for i in range(4, 8)] == gpus

    def test_a_declared_subset_is_used_by_name(self):
        assert resolve_worker_gpu("gw1", ["2", "3"]) == "3"


class TestPinnedWorkerGpu:
    def test_a_run_that_asks_pins_each_worker(self, tmp_path):
        devices = _device_nodes(tmp_path, ["nvidia0", "nvidia1", "nvidia2", "nvidia3"])
        env = {PIN_WORKER_GPUS_ENV: "1"}
        assert [pinned_worker_gpu(env, f"gw{i}", devices) for i in range(4)] == ["0", "1", "2", "3"]

    def test_a_run_that_does_not_ask_keeps_every_device(self, tmp_path):
        devices = _device_nodes(tmp_path, ["nvidia0", "nvidia1"])
        assert pinned_worker_gpu({}, "gw1", devices) is None
        assert pinned_worker_gpu({PIN_WORKER_GPUS_ENV: "0"}, "gw1", devices) is None

    def test_a_serial_run_keeps_every_device_even_when_pinning_is_asked(self, tmp_path):
        devices = _device_nodes(tmp_path, ["nvidia0", "nvidia1"])
        assert pinned_worker_gpu({PIN_WORKER_GPUS_ENV: "1"}, "", devices) is None
