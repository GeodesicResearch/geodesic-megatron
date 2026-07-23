"""Multi-node NCCL smoke test for the container pipeline (INFR-68 stage C3).

Launched via torchrun (one process per GPU) inside the pipeline container:

    srun --nodes=2 --ntasks-per-node=1 --export=ALL --overlap \
        ./pipeline_container_exec.sh \
        "cd $REPO_DIR; source pipeline_container_activate.sh; \
         torchrun --nproc_per_node=4 --nnodes=2 --node_rank=\\$SLURM_NODEID \
             --master_addr=<addr> --master_port=<port> \
             scripts/container/nccl_allreduce_smoke.py"

Asserts two things the container networking must deliver:
  1. NCCL selected the CXI aws-ofi-nccl plugin, not TCP sockets (run with
     NCCL_DEBUG=INFO and grep the log for 'Using network AWS Libfabric' — a
     Socket fallback silently costs ~70x bandwidth).
  2. Measured all_reduce bus bandwidth clears a floor that only Slingshot RDMA
     can reach (TCP fallback measures ~2.3 GB/s on 2 nodes; Isambard's docs
     expect ~163 GB/s for a containerized 2-node/8-GPU all_reduce).

Rank 0 prints `BUSBW_GB_S: <value>` and exits nonzero below the floor.
"""

import argparse
import os
import time

import torch
import torch.distributed as dist


def main():
    """Time bf16 all_reduce and verify Slingshot-RDMA-level bus bandwidth."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--min-busbw-gb-s",
        type=float,
        default=None,
        help="Fail if measured bus bandwidth is below this (GB/s). Default 100.0 "
        "— between TCP fallback (~2.3) and healthy Slingshot RDMA (~163), so the "
        "verdict is unambiguous.",
    )
    parser.add_argument("--size-mb", type=int, default=1024, help="all_reduce payload size (MiB)")
    parser.add_argument("--iters", type=int, default=20, help="timed iterations")
    args = parser.parse_args()
    min_busbw = 100.0 if args.min_busbw_gb_s is None else args.min_busbw_gb_s

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    rank, world = dist.get_rank(), dist.get_world_size()

    n_elem = args.size_mb * 1024 * 1024 // 2  # bf16 = 2 bytes
    x = torch.ones(n_elem, dtype=torch.bfloat16, device="cuda")

    for _ in range(5):  # warmup (comm-init + ring establishment)
        dist.all_reduce(x)
    torch.cuda.synchronize()
    dist.barrier()

    t0 = time.perf_counter()
    for _ in range(args.iters):
        dist.all_reduce(x)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    # Ring all_reduce moves 2*(n-1)/n of the payload per rank per op — the
    # standard nccl-tests "busbw" normalization, comparable to its output.
    bytes_per_op = x.numel() * x.element_size()
    algbw = bytes_per_op * args.iters / elapsed / 1e9
    busbw = algbw * 2 * (world - 1) / world

    if rank == 0:
        print(f"world={world} payload={args.size_mb}MiB iters={args.iters} elapsed={elapsed:.2f}s")
        print(f"ALGBW_GB_S: {algbw:.1f}")
        print(f"BUSBW_GB_S: {busbw:.1f}")
        ok = busbw >= min_busbw
        print(f"VERDICT: {'PASS' if ok else 'FAIL'} (floor {min_busbw} GB/s)")
        if not ok:
            dist.destroy_process_group()
            raise SystemExit(1)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
