"""
Isambard-official path: compare isambard-nccl-tests bare-metal vs inside container
using the brics/apptainer-multi-node /host/adapt.sh adaptation script.

Per https://docs.isambard.ac.uk/user-documentation/guides/containers/apptainer-multi-node/:
- Inside container, source /host/adapt.sh (sets LD_LIBRARY_PATH, NCCL/FI_CXI env)
- /host/nccl provides NCCL 2.26.6 (matches what bare-metal binary was linked against)
- /host/opt/nvidia/.../cuda/12.6/lib64 provides CUDA 12.6 (matches binary's libcudart)
- Same binary runs in both contexts — measures pure container-env overhead.
"""
import argparse
import os
import subprocess
import sys

import wandb


COLLECTIVE_TO_BIN = {
    "all_reduce": "all_reduce_perf",
    "alltoall": "alltoall_perf",
    "all_gather": "all_gather_perf",
    "reduce_scatter": "reduce_scatter_perf",
}

NCCL_TESTS_BUILD = "/home/a5k/kyleobrien.a5k/isambard-nccl-tests/build"


def parse_nccl_perf(output: str):
    rows = []
    for line in output.splitlines():
        if line.startswith("#") or not line.strip():
            continue
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
    """Bare-metal: same as our pipeline_training_launch.sh's NCCL stack.
    Uses Isambard pattern: --ntasks-per-node=1 -g 4 (per Isambard docs)."""
    inner_cmd = (
        f"export LD_LIBRARY_PATH={env.get('LD_LIBRARY_PATH', '')}; "
        f"export LD_PRELOAD={env.get('LD_PRELOAD', '')}; "
        f"{NCCL_TESTS_BUILD}/{test_bin} -b 32K -e {end_size} -f 2 -g 4"
    )
    cmd = [
        "srun",
        f"--nodes={num_nodes}",
        "--ntasks-per-node=1",
        "--gpus-per-node=4",
        "--kill-on-bad-exit=1",
        "--export=ALL",
        "--overlap",
        "bash", "-c", inner_cmd,
    ]
    print(f"\n=== BARE-METAL: {test_bin} @ {num_nodes} nodes (Isambard --ntasks-per-node=1 -g 4) ===", flush=True)
    print(f"$ {' '.join(cmd[:9])} bash -c '...'", flush=True)
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    print(result.stdout)
    if result.returncode != 0:
        print(f"STDERR:\n{result.stderr}", file=sys.stderr)
    return result


def run_container_adaptsh(test_bin: str, num_nodes: int, end_size: str, env: dict, image: str):
    """Container via Isambard's official /host/adapt.sh path.
    Same binary as bare-metal, just running inside container with /host/* binds.

    The bare-metal binary links to Cray MPI/PMI from /opt/cray/pe/lib64 which
    adapt.sh doesn't add — append it explicitly so libmpi_cray, libpmi, libcraymath
    et al. resolve. /opt/cray bind below makes those paths accessible inside container.
    """
    inner_cmd = (
        "export LD_LIBRARY_PATH=/opt/cray/pe/lib64:/opt/cray/pe/lib64/cce:/opt/cray/pals/1.6/lib:$LD_LIBRARY_PATH; "
        f"{NCCL_TESTS_BUILD}/{test_bin} -b 32K -e {end_size} -f 2 -g 4"
    )
    # Bind nccl-tests dir AND full /opt/cray (binary links to libmpi_cray, libpmi*,
    # libcraymath, etc. from /opt/cray/pe/lib64 which brics module doesn't bind).
    cmd = [
        "srun",
        f"--nodes={num_nodes}",
        "--ntasks-per-node=1",
        "--gpus-per-node=4",
        "--kill-on-bad-exit=1",
        "--export=ALL",
        "--overlap",
        "apptainer", "exec", "--nv",
        f"--bind={NCCL_TESTS_BUILD.rsplit('/',1)[0]}:{NCCL_TESTS_BUILD.rsplit('/',1)[0]}",
        "--bind=/opt/cray:/opt/cray:ro",
        image,
        "/host/adapt.sh", "bash", "-c", inner_cmd,
    ]
    print(f"\n=== CONTAINER (adapt.sh): {test_bin} @ {num_nodes} nodes ===", flush=True)
    print(f"$ {' '.join(cmd[:9])} apptainer exec --nv --bind=... {image} /host/adapt.sh bash -c '...'", flush=True)
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    print(result.stdout)
    if result.returncode != 0:
        print(f"STDERR:\n{result.stderr}", file=sys.stderr)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--collectives", default="all_reduce,alltoall")
    ap.add_argument("--node-counts", default="2,4")
    ap.add_argument("--image", default="/projects/a5k/public/containers/pytorch_25.10-py3.sif")
    ap.add_argument("--wandb-project", default="nccl-benchmarks")
    ap.add_argument("--wandb-entity", default="geodesic")
    args = ap.parse_args()

    collectives = [c.strip() for c in args.collectives.split(",")]
    node_counts = [int(n.strip()) for n in args.node_counts.split(",")]

    job_id = os.environ.get("SLURM_JOB_ID", "local")
    run_name = f"adaptsh_baremetal_vs_container_{job_id}"
    wandb.init(
        project=args.wandb_project, entity=args.wandb_entity, name=run_name,
        config={
            "image": args.image, "collectives": collectives, "node_counts": node_counts,
            "approach": "Isambard official /host/adapt.sh",
            "binary": NCCL_TESTS_BUILD,
            "nccl_in_container": "host NCCL 2.26.6 (via /host/nccl)",
            "nccl_baremetal": "host NCCL 2.26.6 (LD_PRELOAD venv 2.28.9)",
            "cuda": "12.6 in both (host)",
        },
    )

    bm_env = os.environ.copy()
    nccl_so = "/home/a5k/kyleobrien.a5k/geodesic-megatron/.venv/lib/python3.12/site-packages/nvidia/nccl/lib/libnccl.so.2"
    if os.path.exists(nccl_so):
        bm_env["LD_PRELOAD"] = nccl_so

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

    # Container: adapt.sh sets up its own NCCL/FI_CXI env. SINGULARITY_BINDPATH already
    # populated by `module load brics/apptainer-multi-node` in the parent sbatch.
    ct_env = os.environ.copy()
    ct_env.pop("LD_PRELOAD", None)

    END_SIZE = {"alltoall": "1G", "all_reduce": "4G", "all_gather": "4G", "reduce_scatter": "4G"}

    summary = []
    for n in node_counts:
        for col in collectives:
            test_bin = COLLECTIVE_TO_BIN.get(col, f"{col}_perf")
            end_size = END_SIZE.get(col, "1G")

            bm_result = run_baremetal(test_bin, n, end_size, bm_env)
            bm_rows = parse_nccl_perf(bm_result.stdout) if bm_result.returncode == 0 else []
            bm_peak = max((r["alg_bw"] for r in bm_rows), default=0.0)

            ct_result = run_container_adaptsh(test_bin, n, end_size, ct_env, args.image)
            ct_rows = parse_nccl_perf(ct_result.stdout) if ct_result.returncode == 0 else []
            ct_peak = max((r["alg_bw"] for r in ct_rows), default=0.0)

            print(f"\n>>> {col} @ {n} nodes:  baremetal={bm_peak:.1f}  container_adaptsh={ct_peak:.1f} GB/s "
                  f"(delta={ct_peak - bm_peak:+.1f} GB/s, alg_bw)", flush=True)

            for r in bm_rows:
                wandb.log({"host": "baremetal", "collective": col, "num_nodes": n,
                           "bytes": r["bytes"], "alg_bw_GB_s": r["alg_bw"],
                           "bus_bw_GB_s": r["bus_bw"], "time_us": r["time_us"]})
            for r in ct_rows:
                wandb.log({"host": "container_adaptsh", "collective": col, "num_nodes": n,
                           "bytes": r["bytes"], "alg_bw_GB_s": r["alg_bw"],
                           "bus_bw_GB_s": r["bus_bw"], "time_us": r["time_us"]})

            summary.append({
                "collective": col, "num_nodes": n,
                "baremetal_GB_s": bm_peak, "container_adaptsh_GB_s": ct_peak,
                "delta_GB_s": ct_peak - bm_peak,
                "bm_rc": bm_result.returncode, "ct_rc": ct_result.returncode,
            })

    print("\n=== FINAL COMPARISON (peak alg_bw, Isambard adapt.sh path) ===")
    for s in summary:
        print(f"{s['collective']:<16} {s['num_nodes']:>5}n  bm={s['baremetal_GB_s']:>7.1f}  "
              f"ct(adaptsh)={s['container_adaptsh_GB_s']:>7.1f}  delta={s['delta_GB_s']:>+6.1f}")

    table = wandb.Table(
        columns=["collective", "num_nodes", "baremetal_GB_s", "container_adaptsh_GB_s", "delta_GB_s"],
        data=[[s["collective"], s["num_nodes"], s["baremetal_GB_s"],
               s["container_adaptsh_GB_s"], s["delta_GB_s"]] for s in summary]
    )
    wandb.log({"summary": table})
    wandb.summary["bm_avg_bw"] = sum(s["baremetal_GB_s"] for s in summary) / max(1, len(summary))
    wandb.summary["ct_avg_bw"] = sum(s["container_adaptsh_GB_s"] for s in summary) / max(1, len(summary))
    wandb.finish()


if __name__ == "__main__":
    main()
