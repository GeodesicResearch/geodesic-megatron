# Nemotron 3 Nano SFT — Experiment Tracker

Hardware: Isambard GH200 95GB GPUs, 4 GPUs/node, Slingshot/CXI networking

## Results

| Job | Nodes | GPUs | TP | EP | DP | Seq Len | GBS | Grad Accum | MoE Fusions | Recompute | Status | Steady Iter (s) | TFLOP/s/GPU | Peak Mem (GB) | Notes |
|-----|-------|------|----|----|-----|---------|-----|------------|-------------|-----------|--------|-----------------|-------------|---------------|-------|
| 3596659 | 8 | 32 | 2 | 4 | 16 | 16384 | 16 | 1 | permute+grouped | selective | **OOM** | — | — | >95 | Selective recompute + TP=2 doesn't fit at seq=16384 |
| **3596639** | **8** | **32** | **2** | **4** | **16** | **8192** | **16** | **1** | **permute+grouped** | **selective** | **completed** | **2.5** | **36.7** | **76.3** | **Best config. 100 iters, loss 1.27→0.77** |
| 3596621 | 8 | 32 | 4 | 4 | 8 | 4096 | 8 | 1 | permute+grouped | selective | cancelled | 98 | 0.2 | 62.1 | TP=4 too much comm overhead at short seq |
| 3596613 | 8 | 32 | 4 | 4 | 8 | 4096 | 16 | 2 | permute+grouped | selective | cancelled | 93 | 0.5 | 62.1 | Stable but slow — TP=4 + short seq |
| 3596591 | 64 | 256 | 4 | 4 | 64 | 4096 | 64 | 1 | permute+grouped | selective | **hung/slow** | 614 | 0.0 | 62.1 | Hangs at 64-node scale |
| 3596564 | 64 | 256 | 4 | 4 | 64 | 16384 | 64 | 1 | all on (overlap=T) | selective | **hung iter 2** | 408 (iter1) | 0.2 | 72.2 | Hangs at 64-node scale |
| 3596540 | 16 | 64 | 1 | 4 | 64 | 16384 | 64 | 1 | all on | full/52 | **OOM** | — | — | 91.5 | TP=1 doesn't fit |
| 3596554 | 32 | 128 | 2 | 4 | 64 | 16384 | 64 | 1 | all on | full/52 | **OOM** | — | — | — | TP=2 + seq=16384 doesn't fit |
| 3596528 | 32 | 128 | 2 | 4 | 64 | 16384 | 64 | 1 | all on | full/52 | **OOM** | — | — | — | TP=2 + seq=16384 doesn't fit |
| 3596503 | 16 | 64 | 4 | 4 | 16 | 16384 | 16 | 1 | all on | full/52 | cancelled | 208 | 0.5 | 73.2 | Full recompute bottleneck |
| 3596474 | 16 | 64 | 4 | 4 | 16 | 16384 | 128 | 8 | all on | full/52 | cancelled | 213 | 3.6 | 73.2 | Full recompute bottleneck |
| 3596466 | 16 | 64 | 4 | 4 | 16 | 16384 | 128 | 8 | grouped only | full/52 | cancelled | 210 | 3.7 | 71.2 | Permute fusion off = bottleneck |
| 3596446 | 16 | 64 | 4 | 4 | 16 | 16384 | 128 | 8 | all off | full/52 | cancelled | 206 | 3.7 | 71.2 | All fusions off = very slow |
| 3596430 | 4 | 16 | 4 | 4 | 4 | 16384 | 128 | 32 | all on | full/52 | **OOM iter 2** | 164 | 18.7 | 79.2 | High TFLOP/s but misleading — 32 grad accum |

## Key Findings

1. **Selective recompute is critical** — full recompute of all 52 layers caps throughput at ~3.5 TFLOP/s. Selective (core_attn only) achieves 36.7 TFLOP/s. This was the single biggest factor.
2. **TP=2 beats TP=4** — less all-reduce communication with only 3B active params. TP=4 adds overhead without benefit.
3. **seq_length=8192 is the sweet spot** — fits with selective recompute (76 GB peak, 19 GB headroom). seq=16384 requires full recompute or OOMs. seq=4096 has too little compute to hide communication.
4. **EP=4 stays intra-node** — 128 experts / 4 = 32 per GPU, all-to-all within NVLink domain.
5. **64-node runs hang** — both seq=4096 and seq=16384 stall after iter 1 at 256 GPUs. 8-node runs are stable.
6. **MoE fusions matter** — permute_fusion + grouped_gemm must be on. Shared expert overlap can be off.
7. **Minimize grad accum** — set GBS = DP for 1 step per iteration.

## Best Config (Job 3596639)

```yaml
model:
  seq_length: 8192
  tensor_model_parallel_size: 2
  expert_model_parallel_size: 4
  recompute_granularity: selective
  recompute_modules: ["core_attn"]
  moe_permute_fusion: True
  moe_grouped_gemm: True
  moe_shared_expert_overlap: False
```
- 8 nodes (32 GPUs), DP=16, GBS=16, 1 grad accum step
- **2.5s/iter, 36.7 TFLOP/s/GPU, 76.3 GB peak memory**
- Loss: 1.27 → 0.77 over 100 iterations

## Next Steps

- Scale to 16 nodes (DP=32, GBS=32) — test if 16-node works without hanging
- Scale to 32 nodes (DP=64, GBS=64) — find the scaling limit before 64-node hang
- Try seq=16384 + selective recompute at TP=2 EP=4 — may OOM (76 GB at 8192, ~90 GB at 16384?)
- Try moe_shared_expert_overlap=True at 8 nodes — may improve throughput further
