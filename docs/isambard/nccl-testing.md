# NCCL performance testing

Proving whether a failure is the fabric.

## NCCL Performance Testing

### Debugging NCCL-looking failures (rendezvous timeout, hang, slow iters)

When a training run fails with symptoms that *might* be fabric-related — c10d KV-store rendezvous timeout ("N/M clients joined"), NCCL watchdog timeout mid-iteration, iters suddenly taking 10-20× longer than expected, `WorkNCCL(SeqNum=...)` timing out — run the benchmark suite **inside the same allocation** to prove whether NCCL/Slingshot itself is at fault. If the benchmark passes, the fabric is healthy and the failure is elsewhere (leftover zombie processes, rendezvous port collision, config mismatch, parallel-run contention).

**Repo**: `/home/a5k/kyleobrien.a5k/isambard-nccl-tests/` — Python orchestrator over upstream nccl-tests with pass/fail thresholds for Isambard GH200. Binaries are already built at `build/`.

**Usage (inside the affected SLURM allocation, e.g. the tunnel that just had a training failure):**
```bash
cd /home/a5k/kyleobrien.a5k/isambard-nccl-tests
module purge && module load PrgEnv-cray cuda/12.6 brics/aws-ofi-nccl/1.8.1
python scripts/run_nccl_benchmarks.py --min-nodes 2 --max-nodes 8 --no-wandb
# (raise --max-nodes to the allocation size if you want the full sweep)
```

Runs ~20 min for the 2..8 sweep. Tests 5 collectives (alltoall, all_reduce, reduce_scatter, all_gather, sendrecv) at each node count against calibrated thresholds (~80% of observed baseline). "PASS" on ≥ the node count of the failing run is strong evidence the fabric is fine.

**Interpreting the result:**
- **All PASS** → NCCL is healthy. Failure was almost certainly at the process layer (zombies, rendezvous port collision, bad config, parallel-run fabric saturation from *multiple* PP=4 training jobs, etc.). Clean up zombie ft_launcher/torchrun/pipeline_training processes and relaunch, optionally with a different `MASTER_PORT_OVERRIDE`.
- **Consistent FAIL on one node count** → capacity issue at that scale — try a different node subset of the allocation.
- **FAIL scattered across collectives/scales** → bad specific node(s). `isambard_sbatch --mark-bad <node> "<reason>"` and move on.

**Typical healthy numbers on a clean allocation (2026-04-22)**: 8-node / 32-GPU all_gather bus_bw ≈ 86 GB/s (threshold 55), alltoall / all_reduce / reduce_scatter all comfortably above threshold, zero errors.

### Raw one-shot measurement (in-container)

The Slingshot build ships nccl-tests binaries at `/opt/slingshot/nccl-tests/` inside the
container, built against the same NCCL the training runs use — so this measures the real stack:

```bash
export NCCL_NET="AWS Libfabric" FI_PROVIDER=cxi NCCL_SOCKET_IFNAME=hsn
srun --nodes=2 --ntasks-per-node=1 --export=ALL ./pipeline_env_exec.sh \
  "source $PWD/pipeline_env_activate.sh; /opt/slingshot/nccl-tests/all_reduce_perf -b 32K -e 8G -f 2 -g 4"
```
**Measured (2026-04-12, retired bare-metal stack)**: 2-node all_reduce 191-197 GB/s; 16-node
255-263 GB/s. Containerized 2-node/8-GPU all_reduce measured 131 GB/s (2026-07-23); the
qualification floor is ~100 GB/s, and a TCP fallback shows as ~2.3 GB/s.
