"""
Run an nccl-tests binary in two modes (bare-metal + container), parse the
bandwidth numbers from each, and log a side-by-side comparison to W&B.

Invoked from pipeline_container_ncclcompare.sbatch — does NOT modify the
upstream isambard-nccl-tests harness.
"""
import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import wandb


COLLECTIVE_TO_BIN = {
    "all_reduce": "all_reduce_perf",
    "alltoall": "alltoall_perf",
    "all_gather": "all_gather_perf",
    "reduce_scatter": "reduce_scatter_perf",
}


def parse_nccl_perf(output: str):
    """Parse `xxx_perf` output and return list of (bytes, time_us, alg_bw, bus_bw)."""
    rows = []
    seen_data_section = False
    for line in output.splitlines():
        # Skip header and comments
        if line.startswith("#") or not line.strip():
            seen_data_section = True
            continue
        if not seen_data_section:
            continue
        # Data line: "  size  count  type  redop  root  time  algbw  busbw  #wrong  time  algbw  busbw  #wrong"
        # We just want the in-place columns (first half)
        parts = line.split()
        if len(parts) < 8:
            continue
        try:
            sz = int(parts[0])
            t = float(parts[5])
            alg = float(parts[6])
            bus = float(parts[7])
            rows.append({"bytes": sz, "time_us": t, "alg_bw": alg, "bus_bw": bus})
        except (ValueError, IndexError):
            continue
    return rows


def run_baremetal(test_bin: str, num_nodes: int, end_size: str, env: dict):
    """Run nccl-tests binary bare-metal via srun (matches existing isambard pattern).

    Inner bash -c sets LD_LIBRARY_PATH explicitly because SLURM's --export=ALL
    doesn't reliably propagate it to worker processes.
    """
    ld = env.get("LD_LIBRARY_PATH", "")
    ld_pre = env.get("LD_PRELOAD", "")
    inner_cmd = (
        f"export LD_LIBRARY_PATH={ld}; "
        f"export LD_PRELOAD={ld_pre}; "
        f"/home/a5k/kyleobrien.a5k/isambard-nccl-tests/build/{test_bin} -b 32K -e {end_size} -f 2 -g 1"
    )
    cmd = [
        "srun",
        f"--nodes={num_nodes}",
        f"--ntasks-per-node=4",
        "--gpus-per-node=4",
        "--kill-on-bad-exit=1",
        "--export=ALL",
        "--overlap",
        "bash", "-c", inner_cmd,
    ]
    print(f"\n=== BARE-METAL: {test_bin} @ {num_nodes} nodes ===", flush=True)
    print(f"$ {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    print(result.stdout)
    if result.returncode != 0:
        print(f"STDERR:\n{result.stderr}", file=sys.stderr)
    return result


def run_container(test_bin: str, num_nodes: int, end_size: str, env: dict, image: str):
    """Run container's pre-built nccl-tests at /usr/local/bin/.

    The container ships nccl-tests built against its own NCCL 2.27.7 + CUDA 13.0,
    so we use those rather than binding in the bare-metal binary (which links
    against brics NCCL 2.26.6 + CUDA 12.6 paths that don't exist inside container).
    """
    binds = (
        "--bind /opt/cray:/opt/cray:ro "
        "--bind /tools/brics/apps/linux-sles15-neoverse_v2/gcc-12.3.0/aws-ofi-nccl-1.8.1-c47cd5ivrugm3jzlyqyis4igyflnydmo:/host/aws-ofi-nccl:ro "
        "--bind /usr/lib64:/host/usr/lib64:ro"
    )
    # CUDA forward-compat path MUST come first so libcuda.so.1 resolves to
    # /usr/local/cuda/compat/lib/libcuda.so.1 (forward-compat shim that bridges
    # CUDA 13 runtime to host driver 565.57.01 which only natively supports CUDA 12.7).
    # If /host/usr/lib64 (which contains host driver's older libcuda.so) comes first,
    # the binary loads CUDA 12.7's libcuda → CUDA 13 runtime initialization fails.
    inner_cmd = (
        f"export LD_LIBRARY_PATH=/usr/local/cuda/compat/lib.real:/usr/local/cuda/compat/lib:"
        f"/host/aws-ofi-nccl/lib:/opt/cray/libfabric/1.22.0/lib64:/host/usr/lib64:$LD_LIBRARY_PATH; "
        f"/usr/local/bin/{test_bin} -b 32K -e {end_size} -f 2 -g 1"
    )
    cmd = [
        "srun",
        f"--nodes={num_nodes}",
        f"--ntasks-per-node=4",
        "--gpus-per-node=4",
        "--kill-on-bad-exit=1",
        "--export=ALL",
        "--overlap",
        "apptainer", "exec", "--nv",
        *binds.split(),
        image,
        "bash", "-c", inner_cmd,
    ]
    print(f"\n=== CONTAINER: {test_bin} @ {num_nodes} nodes ===", flush=True)
    print(f"$ {' '.join(cmd[:8])} ... [bind+exec elided] ... bash -c '{inner_cmd[:80]}...'", flush=True)
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    print(result.stdout)
    if result.returncode != 0:
        print(f"STDERR:\n{result.stderr}", file=sys.stderr)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--collectives", default="all_reduce,alltoall",
                    help="Comma-separated collectives to test")
    ap.add_argument("--node-counts", default="2,4",
                    help="Comma-separated node counts")
    ap.add_argument("--image", default="/projects/a5k/public/containers/pytorch_25.10-py3.sif")
    ap.add_argument("--wandb-project", default="nccl-benchmarks")
    ap.add_argument("--wandb-entity", default="geodesic")
    args = ap.parse_args()

    collectives = [c.strip() for c in args.collectives.split(",")]
    node_counts = [int(n.strip()) for n in args.node_counts.split(",")]

    job_id = os.environ.get("SLURM_JOB_ID", "local")
    run_name = f"baremetal_vs_container_{job_id}"
    print(f"Initializing W&B run: {run_name}")

    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=run_name,
        config={
            "image": args.image,
            "collectives": collectives,
            "node_counts": node_counts,
            "container_pytorch": "2.9.0+nv25.10",
            "container_cuda": "13.0",
            "container_nccl": "2.27.7",
            "container_te": "2.8.0",
            "baremetal_nccl": "via brics + ncclCommShrink LD_PRELOAD",
        },
    )

    # Bare-metal env: needs LD_PRELOAD for ncclCommShrink
    bm_env = os.environ.copy()
    nccl_so = "/home/a5k/kyleobrien.a5k/geodesic-megatron/.venv/lib/python3.12/site-packages/nvidia/nccl/lib/libnccl.so.2"
    if Path(nccl_so).exists():
        bm_env["LD_PRELOAD"] = nccl_so
    # Slingshot/CXI vars (same set as our bare-metal pipeline)
    cxi_env = {
        "NCCL_NET": "AWS Libfabric", "FI_PROVIDER": "cxi", "NCCL_SOCKET_IFNAME": "hsn",
        "NCCL_CROSS_NIC": "0", "NCCL_NET_GDR_LEVEL": "PHB", "NCCL_GDRCOPY_ENABLE": "1",
        "NCCL_NET_FORCE_FLUSH": "1", "FI_MR_CACHE_MONITOR": "userfaultfd",
        "FI_CXI_DISABLE_HOST_REGISTER": "1", "FI_HMEM_CUDA_USE_GDRCOPY": "1",
        "FI_CXI_DEFAULT_CQ_SIZE": "131072", "FI_CXI_DEFAULT_TX_SIZE": "1024",
        "FI_CXI_RDZV_PROTO": "alt_read", "FI_CXI_DISABLE_NON_INJECT_MSG_IDC": "1",
        "NCCL_DEBUG": "WARN",
    }
    bm_env.update(cxi_env)

    # Container env: container's NCCL handles transport via the plugin.
    # DO NOT override APPTAINERENV_LD_LIBRARY_PATH — let container preserve its default
    # which contains /usr/local/cuda/compat/lib.real (forward-compat libcuda for
    # CUDA 13 on host driver 565.57.01 / CUDA 12.7), torch libs, /.singularity.d/libs.
    # Inner shell appends host paths to that default via :$LD_LIBRARY_PATH.
    ct_env = os.environ.copy()
    ct_env.pop("LD_PRELOAD", None)  # Strip bare-metal LD_PRELOAD
    ct_env.update(cxi_env)
    ct_env["NCCL_NET_PLUGIN"] = "/host/aws-ofi-nccl/lib/libnccl-net.so"
    # Make sure CXI env vars propagate INTO container too
    for k, v in cxi_env.items():
        ct_env[f"APPTAINERENV_{k}"] = v
    ct_env["APPTAINERENV_NCCL_NET_PLUGIN"] = ct_env["NCCL_NET_PLUGIN"]

    summary = []

    # alltoall_perf reports bus_bw=0 (formula isn't well-defined for alltoall),
    # so we use alg_bw as the headline metric. For all_reduce/reduce_scatter/
    # all_gather, alg_bw and bus_bw differ by a constant factor — alg_bw is
    # still meaningful for relative comparison.
    # End size: alltoall allocates buffer ∝ total ranks → cap at 1G to fit on GH200 95GB.
    END_SIZE_PER_COLLECTIVE = {
        "alltoall": "1G",
        "all_reduce": "4G",
        "all_gather": "4G",
        "reduce_scatter": "4G",
    }

    for n in node_counts:
        for col in collectives:
            test_bin = COLLECTIVE_TO_BIN.get(col, f"{col}_perf")
            end_size = END_SIZE_PER_COLLECTIVE.get(col, "1G")

            # Bare-metal
            bm_result = run_baremetal(test_bin, n, end_size, bm_env)
            bm_rows = parse_nccl_perf(bm_result.stdout) if bm_result.returncode == 0 else []
            bm_peak = max((r["alg_bw"] for r in bm_rows), default=0.0)

            # Container
            ct_result = run_container(test_bin, n, end_size, ct_env, args.image)
            ct_rows = parse_nccl_perf(ct_result.stdout) if ct_result.returncode == 0 else []
            ct_peak = max((r["alg_bw"] for r in ct_rows), default=0.0)

            print(f"\n>>> {col} @ {n} nodes:  baremetal={bm_peak:.1f} GB/s  container={ct_peak:.1f} GB/s "
                  f"(delta={ct_peak - bm_peak:+.1f} GB/s, alg_bw)", flush=True)

            # Log every message size
            for r in bm_rows:
                wandb.log({
                    "host": "baremetal",
                    "collective": col,
                    "num_nodes": n,
                    "bytes": r["bytes"],
                    "alg_bw_GB_s": r["alg_bw"],
                    "bus_bw_GB_s": r["bus_bw"],
                    "time_us": r["time_us"],
                })
            for r in ct_rows:
                wandb.log({
                    "host": "container",
                    "collective": col,
                    "num_nodes": n,
                    "bytes": r["bytes"],
                    "alg_bw_GB_s": r["alg_bw"],
                    "bus_bw_GB_s": r["bus_bw"],
                    "time_us": r["time_us"],
                })

            summary.append({
                "collective": col, "num_nodes": n,
                "baremetal_peak_alg_bw_GB_s": bm_peak,
                "container_peak_alg_bw_GB_s": ct_peak,
                "delta_GB_s": ct_peak - bm_peak,
                "baremetal_returncode": bm_result.returncode,
                "container_returncode": ct_result.returncode,
            })

    # Summary table (peak alg_bw)
    print("\n\n=== FINAL COMPARISON (peak alg_bw) ===")
    print(f"{'collective':<16} {'nodes':>5} {'baremetal':>12} {'container':>12} {'delta':>10}")
    for s in summary:
        print(f"{s['collective']:<16} {s['num_nodes']:>5} "
              f"{s['baremetal_peak_alg_bw_GB_s']:>9.1f} GB/s "
              f"{s['container_peak_alg_bw_GB_s']:>9.1f} GB/s "
              f"{s['delta_GB_s']:>+7.1f} GB/s")

    summary_table = wandb.Table(
        columns=["collective", "num_nodes", "baremetal_GB_s", "container_GB_s", "delta_GB_s"],
        data=[[s["collective"], s["num_nodes"],
               s["baremetal_peak_alg_bw_GB_s"], s["container_peak_alg_bw_GB_s"], s["delta_GB_s"]]
              for s in summary]
    )
    wandb.log({"summary": summary_table})
    wandb.summary["bm_avg_bw"] = sum(s["baremetal_peak_alg_bw_GB_s"] for s in summary) / max(1, len(summary))
    wandb.summary["ct_avg_bw"] = sum(s["container_peak_alg_bw_GB_s"] for s in summary) / max(1, len(summary))

    wandb.finish()


if __name__ == "__main__":
    main()
