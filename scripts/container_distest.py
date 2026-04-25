"""
Container distributed test — validates NCCL + Slingshot/CXI from inside container.

Run via torchrun inside an apptainer NGC PyTorch image, with /host/adapt.sh
having configured NCCL_NET=AWS Libfabric and FI_PROVIDER=cxi.
"""
import os
import time
import torch
import torch.distributed as dist


def main():
    # Init
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.cuda.current_device()

    if rank == 0:
        print(f"=== Distributed init OK ===")
        print(f"World size: {world}")
        print(f"NCCL version: {torch.cuda.nccl.version()}")
        print(f"PyTorch: {torch.__version__}")
        print(f"CUDA: {torch.version.cuda}")
        print(f"NCCL_NET: {os.environ.get('NCCL_NET', 'unset')}")
        print(f"FI_PROVIDER: {os.environ.get('FI_PROVIDER', 'unset')}")

    # Barrier to ensure rank 0 prints first
    dist.barrier()

    # All-reduce test
    if rank == 0:
        print(f"\n=== Test 1: AllReduce 1 GB ===")
    n_elem = 256 * 1024 * 1024  # 1 GB BF16
    x = torch.ones(n_elem, dtype=torch.bfloat16, device=device)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(10):
        dist.all_reduce(x)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    bytes_moved = 2 * n_elem * 2 * (world - 1) / world * 10  # bf16=2B, ring all-reduce
    bw = bytes_moved / elapsed / 1e9
    if rank == 0:
        print(f"  10 iterations of 1 GB all_reduce: {elapsed:.2f}s")
        print(f"  Effective bus bandwidth: {bw:.2f} GB/s")

    # AllToAll test (simulates MoE expert routing)
    if rank == 0:
        print(f"\n=== Test 2: AllToAll 256 MB ===")
    n_per_rank = 64 * 1024 * 1024  # 64M elements/rank/peer
    x = torch.ones(world * n_per_rank, dtype=torch.bfloat16, device=device)
    out = torch.zeros_like(x)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(10):
        dist.all_to_all_single(out, x)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    bytes_moved = 2 * n_per_rank * (world - 1) * 10
    bw = bytes_moved / elapsed / 1e9
    if rank == 0:
        print(f"  10 iterations of {n_per_rank * 2 / 1e6:.0f} MB/rank all_to_all: {elapsed:.2f}s")
        print(f"  Effective bandwidth: {bw:.2f} GB/s")

    # GEMM test (validates basic compute on the container's GPU)
    if rank == 0:
        print(f"\n=== Test 3: BF16 GEMM ===")
    a = torch.randn(8192, 8192, dtype=torch.bfloat16, device=device)
    b = torch.randn(8192, 8192, dtype=torch.bfloat16, device=device)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(50):
        c = a @ b
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    flops = 50 * 2 * 8192**3
    tflops = flops / elapsed / 1e12
    if rank == 0:
        print(f"  8192x8192 BF16 matmul x50: {elapsed:.2f}s, {tflops:.1f} TFLOPs/GPU")

    # FP8 GEMM test (the whole point — verify newer CUDA / TE makes blockwise work)
    if rank == 0:
        print(f"\n=== Test 4: FP8 GEMM (TE) ===")
    try:
        import transformer_engine.pytorch as te
        from transformer_engine.common.recipe import DelayedScaling, Format

        recipe = DelayedScaling(margin=0, fp8_format=Format.HYBRID)
        layer = te.Linear(8192, 8192, bias=False, params_dtype=torch.bfloat16).to(device)
        x = torch.randn(2048, 8192, dtype=torch.bfloat16, device=device)
        with te.fp8_autocast(enabled=True, fp8_recipe=recipe):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(20):
                y = layer(x)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
        flops = 20 * 2 * 2048 * 8192 * 8192
        tflops = flops / elapsed / 1e12
        if rank == 0:
            print(f"  TE FP8 hybrid Linear x20: {elapsed:.2f}s, {tflops:.1f} TFLOPs/GPU")
            print(f"  TE version: {te.__version__ if hasattr(te, '__version__') else '(no __version__)'}")
    except Exception as e:
        if rank == 0:
            print(f"  FP8 test FAILED: {type(e).__name__}: {e}")

    # FP8 blockwise test (the actual goal)
    if rank == 0:
        print(f"\n=== Test 5: FP8 BLOCKWISE GEMM (paper-recommended Hopper recipe) ===")
    try:
        import transformer_engine.pytorch as te
        from transformer_engine.common.recipe import Format
        # Different TE versions have different blockwise recipe names
        try:
            from transformer_engine.common.recipe import Float8BlockScaling
            recipe = Float8BlockScaling()
            recipe_name = "Float8BlockScaling"
        except ImportError:
            try:
                from transformer_engine.common.recipe import BlockScaling
                recipe = BlockScaling()
                recipe_name = "BlockScaling"
            except ImportError:
                recipe = None
                recipe_name = "NOT AVAILABLE"

        if rank == 0:
            print(f"  Recipe: {recipe_name}")
        if recipe is not None:
            layer = te.Linear(8192, 8192, bias=False, params_dtype=torch.bfloat16).to(device)
            x = torch.randn(2048, 8192, dtype=torch.bfloat16, device=device)
            with te.fp8_autocast(enabled=True, fp8_recipe=recipe):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(20):
                    y = layer(x)
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - t0
            flops = 20 * 2 * 2048 * 8192 * 8192
            tflops = flops / elapsed / 1e12
            if rank == 0:
                print(f"  TE FP8 blockwise Linear x20: {elapsed:.2f}s, {tflops:.1f} TFLOPs/GPU")
                print(f"  ✓ BLOCKWISE FP8 WORKS in this container — unlocks paper recipe")
    except Exception as e:
        if rank == 0:
            print(f"  Blockwise FP8 test FAILED: {type(e).__name__}: {e}")

    if rank == 0:
        print(f"\n=== All tests complete ===")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
