# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""Run a test's body on several CPU processes that form one real gloo world.

Code whose behaviour is a collective across ranks (an agreement check, a reduction over a data- or context-parallel
group, pipeline stages stopping together, a gather to rank 0) is shown working only with more than one process.
``run_on_gloo_ranks`` spawns them, joins them into one world through a file rendezvous (no port to collide on with
another suite), runs the body on every rank inside it, and returns what each rank's body returned, in rank order.

The body is called as ``body(rank)`` with the world up, and must be a module-level function, so that the spawned
processes can import it, returning JSON-serialisable data, which is how its result leaves the process. A body that
raises fails the spawn, and with it the test.
"""

import json
from collections.abc import Callable
from pathlib import Path

import torch


def _run_rank(rank: int, body: Callable[[int], object], world_size: int, directory: str) -> None:
    """One spawned rank: join the world, run ``body``, leave the world, and write the body's result."""
    torch.distributed.init_process_group(
        "gloo", init_method=f"file://{directory}/rendezvous", rank=rank, world_size=world_size
    )
    try:
        result = body(rank)
    finally:
        torch.distributed.destroy_process_group()
    (Path(directory) / f"rank{rank}.json").write_text(json.dumps(result))


def run_on_gloo_ranks(body: Callable[[int], object], world_size: int, directory: Path) -> list:
    """Run ``body`` on ``world_size`` processes forming one gloo world, with its rendezvous and results in the empty
    ``directory``; return each rank's result, rank 0 first."""
    torch.multiprocessing.spawn(_run_rank, args=(body, world_size, str(directory)), nprocs=world_size)
    return [json.loads((directory / f"rank{rank}.json").read_text()) for rank in range(world_size)]
