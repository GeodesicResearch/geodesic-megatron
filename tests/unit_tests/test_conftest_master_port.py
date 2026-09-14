"""Unit tests for the MASTER_PORT allocation in tests/unit_tests/conftest.py.

The property under test is that two pytest sessions running concurrently on one
node never hand the same port to a worker. Before this was enforced the base was a
constant, so worker ``gwN`` of every run computed an identical port and the second
process to bind failed — surfacing as DistNetworkError in an unrelated test file,
or as a silent multi-minute wedge.
"""

from __future__ import annotations

from tests.unit_tests.conftest import (
    MASTER_PORT_BASE_ENV,
    resolve_master_port_base,
    resolve_worker_master_port,
)


class TestResolveMasterPortBase:
    def test_distinct_pids_get_distinct_bases(self):
        bases = {resolve_master_port_base({}, pid) for pid in range(1000, 1100)}
        assert len(bases) == 100

    def test_base_stays_in_the_usable_range(self):
        # Includes pids far beyond the 32-bit default pid_max, so the modulus is
        # what bounds the result rather than the size of the input.
        for pid in (1, 2, 4_194_304, 2**31 - 1):
            base = resolve_master_port_base({}, pid)
            assert 20000 <= base < 60000

    def test_highest_worker_port_is_still_a_valid_port(self):
        """The base range must leave room for every worker stacked above it."""
        for pid in (1, 12345, 2**31 - 1):
            base = resolve_master_port_base({}, pid)
            highest = int(resolve_worker_master_port("gw63", base))
            assert highest < 65536

    def test_env_override_wins_over_pid(self):
        env = {MASTER_PORT_BASE_ENV: "31000"}
        assert resolve_master_port_base(env, 424242) == 31000

    def test_workers_inherit_one_base_so_a_run_is_internally_consistent(self):
        """A published base makes every worker of a run agree, whatever its pid."""
        env = {MASTER_PORT_BASE_ENV: "31000"}
        assert resolve_master_port_base(env, 111) == resolve_master_port_base(env, 222)


class TestResolveWorkerMasterPort:
    def test_serial_run_is_untouched(self):
        assert resolve_worker_master_port("", 20000) is None
        assert resolve_worker_master_port("master", 20000) is None

    def test_workers_within_a_run_do_not_collide(self):
        base = resolve_master_port_base({}, 4242)
        ports = {resolve_worker_master_port(f"gw{i}", base) for i in range(64)}
        assert len(ports) == 64

    def test_returns_a_string_because_it_is_written_to_the_environment(self):
        port = resolve_worker_master_port("gw0", 20000)
        assert port == "20041"
        assert isinstance(port, str)

    def test_concurrent_sessions_never_share_a_worker_port(self):
        """The regression: same worker index, different sessions, disjoint ports."""
        a = resolve_master_port_base({}, 1000)
        b = resolve_master_port_base({}, 2000)
        ports_a = {resolve_worker_master_port(f"gw{i}", a) for i in range(8)}
        ports_b = {resolve_worker_master_port(f"gw{i}", b) for i in range(8)}
        assert ports_a.isdisjoint(ports_b)
