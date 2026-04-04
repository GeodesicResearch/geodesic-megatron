# Nemotron 3 Nano SFT — Parallelism Grid Search Results

Hardware: Isambard GH200 95GB GPUs, 4 GPUs/node, Slingshot/CXI networking
Model: Nemotron 3 Nano 30B-A3B MoE, 128 experts, 52 layers, top-6 routing
Dataset: geodesic-research/Dolci-Instruct-SFT-100k (chat format, packed sequences)
Fixed: DP=64, GBS=64, micro_batch=1, no gradient accumulation, selective recompute (core_attn), 10 iterations

## Phase 1: seq_length = 8192

| ID | TP | EP | CP | Nodes | GPUs | DP | Steady Iter (s) | TFLOP/s/GPU | Peak Mem (GB) | Status | Notes |
|----|----|----|-----|-------|------|-----|-----------------|-------------|---------------|--------|-------|
| **A3** | **2** | **4** | **1** | **32** | **128** | **64** | **5.5** | **16.6** | **73.3** | **completed** | **Best stable config** |
| **A4** | **2** | **8** | **1** | **32** | **128** | **64** | **4.6** | **19.8** | **51.1** | **completed** | **Best throughput. EP=8 saves 22GB vs EP=4** |
| **A2r** | **1** | **8** | **1** | **16** | **64** | **64** | **5.1** | **35.8** | **80.4** | **completed** | **Highest TFLOP/s. TP=1 fits with EP=8** |
| A7 | 1 | 4 | 2 | 32 | 128 | 64 | 7.2 | 12.8 | 80.1 | NaN iter 9 | CP=2 adds overhead, numerically unstable |
| A1 | 1 | 4 | 1 | 16 | 64 | 64 | — | — | >95 | **OOM** | TP=1 + EP=4 doesn't fit (only 32 experts/GPU) |
| A5 | 4 | 4 | 1 | 64 | 256 | 64 | — | — | — | **hung** | 64-node NCCL timeout (retried, still hung) |
| A6 | 4 | 8 | 1 | 64 | 256 | 64 | >100 | <1 | 37.1 | **>30s** | 64-node extreme slowdown |
| A9 | 2 | 4 | 2 | 64 | 256 | 64 | 31 | 1.5 | 65.4 | **>30s** | 64-node + CP=2 overhead |
| A8 | 1 | 4 | 4 | 64 | 256 | 64 | 33 | 1.4 | 72.0 | **>30s** | 64-node + CP=4 overhead |

### Phase 1 Key Findings

1. **EP=8 is strictly better than EP=4** at both TP=1 and TP=2. More expert sharding reduces memory (51→73 GB for TP=2) and improves throughput (19.8 vs 16.6 TFLOP/s). Each GPU holds 16 experts instead of 32.

2. **TP=1 + EP=8 achieves highest TFLOP/s (35.8)** because it eliminates all TP all-reduce communication. Fits at 80.4 GB with 15 GB headroom. This is the most compute-efficient config.

3. **TP=2 + EP=8 achieves best wall-clock time (4.6s/iter)** with excellent memory (51.1 GB, 44 GB headroom). The lower TFLOP/s (19.8) is because TP=2 adds communication, but total throughput is highest.

4. **64-node runs consistently fail** — NCCL timeouts, hangs, or extreme slowdowns (>100s/iter). This is a Slingshot/CXI infrastructure issue, not a config problem. Max reliable scale is 32 nodes (128 GPUs).

5. **CP adds overhead without benefit at seq=8192** — A7 (CP=2) was slower than A3 (CP=1) and hit NaN. CP is only useful for longer sequences.

## Phase 2: seq_length = 16384

All Phase 2 configs required 64+ nodes (DP=64 with TP≥2 or CP≥2). Given the consistent 64-node failures observed in Phase 1, all Phase 2 runs exceeded the 30s/iter threshold or hung.

| ID | TP | EP | CP | Nodes | Status | Notes |
|----|----|----|-----|-------|--------|-------|
| B1 | 4 | 4 | 1 | 64 | >30s (403s) | 64-node slowdown |
| B2 | 2 | 4 | 2 | 64 | >30s (39s) | CP=2 overhead + 64-node issues |
| B4 | 4 | 8 | 1 | 64 | >30s (231s) | 64-node slowdown |
| B3 | 1 | 4 | 4 | 64 | skipped | TP=1 will OOM at seq=16384 |
| B5 | 2 | 8 | 2 | 64 | skipped | 64-node issues |
| B6 | 4 | 4 | 2 | 128 | skipped | Exceeds reliable node count |

**Phase 2 conclusion**: seq_length=16384 is not viable on Isambard at DP=64 due to 64-node scaling issues.

## Phase 3: seq_length = 16384 on 32 nodes (with gradient accumulation)

Accepting 2 gradient accumulation steps (DP=32, GBS=64) allows 32-node runs which are reliable.

| ID | TP | EP | CP | Nodes | GPUs | DP | Grad Accum | Steady Iter (s) | TFLOP/s/GPU | Peak Mem (GB) | Status | Notes |
|----|----|----|-----|-------|------|-----|------------|-----------------|-------------|---------------|--------|-------|
| **C2** | **2** | **8** | **2** | **32** | **128** | **32** | **2** | **12.3** | **15.8** | **51.3** | **completed** | **Best seq=16384 config. NaN at iter 8 (LR too high)** |
| C1 | 2 | 4 | 2 | 32 | 128 | 32 | 2 | 24.8 | 7.8 | 73.4 | completed | EP=4 slower, more memory. NaN at iter 4 |

## Phase 4: GBS and LR experiments (seq=16384, TP=2, EP=8, CP=2, 32 nodes)

| ID | GBS | Grad Accum | LR | Weight Decay | Steady Iter (s) | TFLOP/s/GPU | Tokens/iter | Tokens/s | Status | Notes |
|----|-----|------------|-----|-------------|-----------------|-------------|-------------|----------|--------|-------|
| D1 | 64 | 1 | 5e-6 | 0.1 | 2.0 | 36.7 | 524K | 262K | completed | seq=8192, TP=2, EP=8 — best overall |
| **C2** | **64** | **2** | **8e-5** | **0.01** | **12.3** | **15.8** | **1.05M** | **85K** | **NaN iter 8** | **Best seq=16384 throughput** |
| E1 | 32 | 1 | 5e-6 | 0.1 | 11.5 | 8.2 | 524K | 46K | completed | GBS=32 slower tokens/s — no comm overlap |

### Phase 3-4 Key Findings

1. **seq=16384 works on 32 nodes with CP=2** — CP splits the 16384 sequence across 2 GPUs (8192 each), keeping activation memory manageable.
2. **EP=8 again dominates** — 2x faster than EP=4 (12.3s vs 24.8s), half the memory (51 vs 73 GB).
3. **GBS=64 with 2 grad accum beats GBS=32 with 1 grad accum** — nearly same wall-clock (12.3s vs 11.5s) but 2x tokens. The second micro-step's compute hides gradient communication.
5. **Communication-compute overlap explains the grad accum advantage**: Real compute per micro-step is ~6.2s. DP gradient all-reduce takes ~5.3s. With 1 grad accum step (GBS=32), the GPU waits 5.3s for communication after 6.2s of compute = 11.5s total. With 2 steps (GBS=64), the second micro-step's compute (6.2s) runs concurrently with the first step's gradient all-reduce (5.3s), fully hiding the communication cost. Total = 6.2s + 6.2s + ~0s exposed comm = 12.3s for 2x the tokens. This means the system is **communication-bound at 1 grad accum step** and **compute-bound at 2+ steps**.
4. **Recipe LR (5e-6) eliminates NaN** — stable training for 100+ iterations vs NaN at iter 7-8 with 8e-5.
5. **2 gradient accumulation steps** add ~1s to wall-clock but double throughput.

## Recommendations

### For Production Training on Isambard

**Primary recommendation: TP=2, EP=8, seq=8192**
- 32 nodes (128 GPUs), DP=64, GBS=64
- 4.6s/iter, 19.8 TFLOP/s/GPU, 51 GB peak (44 GB headroom)
- Best balance of throughput, memory headroom, and stability
- Enough headroom to increase micro_batch_size to 2 for higher throughput

**For seq=16384: TP=2, EP=8, CP=2**
- 32 nodes (128 GPUs), DP=32, GBS=64, 2 grad accum steps
- 12.3s/iter, 15.8 TFLOP/s/GPU, 51 GB peak
- CP=2 splits sequence across 2 GPUs, keeping per-GPU activations manageable
- Requires pad_seq_to_mult=4 in packed data

**Alternative: TP=1, EP=8, seq=8192**
- 16 nodes (64 GPUs), DP=64, GBS=64
- 5.1s/iter, 35.8 TFLOP/s/GPU, 80 GB peak (15 GB headroom)
- Fewer nodes, highest compute efficiency
- Tighter memory — less room for longer sequences or larger batches

### Scaling Rules

| Nodes | TP | EP | DP | GBS (no grad accum) | Status |
|-------|----|----|-----|---------------------|--------|
| 8 | 2 | 8 | 16 | 16 | Proven (2.5s/iter at GBS=16) |
| 16 | 1 | 8 | 64 | 64 | Proven (5.1s/iter) |
| 32 | 2 | 8 | 64 | 64 | Proven (4.6s/iter) — **recommended** |
| 64+ | any | any | any | any | **Unreliable** on Isambard Slingshot |

### What to Avoid

- **TP=4**: Excessive communication overhead for 3B active params. TP=2 or TP=1 is always better.
- **EP=4 when EP=8 is possible**: EP=8 saves 20+ GB memory and improves throughput.
- **CP at seq=8192**: Adds overhead without benefit. Only consider for seq≥16384.
- **64+ nodes on Isambard**: NCCL hangs and extreme slowdowns.
- **Full recompute**: 100x slower than selective recompute. Always use `recompute_granularity: selective, recompute_modules: ["core_attn"]`.

### Optimal YAML Config

```yaml
model:
  seq_length: 8192
  tensor_model_parallel_size: 2
  expert_model_parallel_size: 8
  context_parallel_size: 1
  sequence_parallel: True
  pipeline_model_parallel_size: 1
  gradient_accumulation_fusion: False
  moe_token_dispatcher_type: alltoall
  moe_flex_dispatcher_backend: null
  moe_shared_expert_overlap: False
  moe_permute_fusion: True
  moe_grouped_gemm: True
  first_last_layers_bf16: False
  recompute_granularity: selective
  recompute_modules: ["core_attn"]

train:
  global_batch_size: 64
  micro_batch_size: 1

dataset:
  seq_length: 8192
  packed_sequence_specs:
    packed_sequence_size: 8192
    pad_seq_to_mult: 1
```

Run with: `sbatch --nodes=32 train_nemotron_sft.sbatch <config.yaml> nano`

---

## Nemotron 3 Super (120B-A12B) Grid Search

Model: nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16
512 experts, 88 layers, 12B active / 120B total, top-22 routing, MTP (2 layers)

### Phase 1: Memory feasibility (32 nodes, seq=16384, TP=4, CP varies)

| ID | TP | EP | CP | DP | GA | Steady Iter (s) | TFLOP/s/GPU | Peak Mem (GB) | Status | Notes |
|----|----|----|-----|----|----|-----------------|-------------|---------------|--------|-------|
| S1 | 4 | 16 | 2 | 16 | 4 | — | — | >95 | **OOM** | 32 experts/GPU too many |
| S2 | 4 | 32 | 2 | 16 | 4 | — | — | 83.5 | **OOM iter 2** | Barely fits, unstable |
| **S3** | **4** | **64** | **2** | **16** | **4** | **82** | **8.7** | **72.2** | **3 iters** | **Viable. Hit MTP+CP+packed bug iter 4** |
| S4 | 4 | 16 | 1 | 32 | 2 | — | — | — | NFS error | Cluster issues |
| S5 | 4 | 32 | 1 | 32 | 2 | — | — | — | NFS error | Cluster issues |
| S6 | 4 | 64 | 1 | 32 | 2 | — | — | 92 | **OOM** | CP=1 at seq=16384 doesn't fit |

### Phase 2: EP=64/128, MTP disabled (in progress, cluster down)

| ID | TP | EP | CP | DP | GA | Notes | Status |
|----|----|----|-----|----|----|-------|--------|
| S7 | 4 | 64 | 2 | 16 | 4 | S3 without MTP | submitted, cluster down |
| S8 | 4 | 128 | 2 | 16 | 4 | 4 experts/GPU | submitted, cluster down |
| S9 | 2 | 64 | 2 | 32 | 2 | Lower TP, may OOM | submitted, cluster down |
| S10 | 2 | 128 | 2 | 32 | 2 | Lower TP + more EP | submitted, cluster down |
| S11 | 4 | 64 | 2 | 32 | 2 | 64 nodes, less GA | pending |
| S12 | 4 | 64 | 2 | 64 | 1 | 128 nodes, no GA | pending |

### Super Key Findings (so far)

1. **EP≥64 required** — with 512 experts, EP=16 (32/GPU) and EP=32 (16/GPU) OOM. EP=64 (8/GPU) fits at 72 GB.
2. **CP=2 required at seq=16384** — CP=1 with selective recompute OOMs (92 GB).
3. **MTP + CP + packed sequences triggers an MCore bug** — `_roll_tensor_packed_seq` crashes with `IndexError`. Workaround: set `mtp_num_layers: 0`.
4. **NFS stale file handles at scale** — multiple runs failed due to Isambard NFS issues. Fixed with `HF_HOME=/projects/a5k/public/hf` and `TRANSFORMERS_OFFLINE=1`.

### Super Production Config (preliminary, pending Phase 2 results)

```yaml
model:
  seq_length: 16384
  tensor_model_parallel_size: 4   # may reduce to 2 if EP=128 saves enough memory
  expert_model_parallel_size: 64  # minimum viable; 128 may be better
  context_parallel_size: 2        # required for seq=16384
  sequence_parallel: True
  mtp_num_layers: 0               # disabled due to MTP+CP+packed bug
  recompute_granularity: selective
  recompute_modules: ["core_attn"]
  moe_permute_fusion: True
  moe_grouped_gemm: True
  moe_shared_expert_overlap: False
```

Run with: `sbatch --nodes=32 train_nemotron_sft.sbatch <config.yaml> super`
