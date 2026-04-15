# Research Log

Ongoing investigation notes from background research agents. Findings are appended as they come in.

---

## [2026-04-09] MoE SFT Best Practices

### Summary

Comprehensive review of MoE fine-tuning literature, NVIDIA's official Nemotron 3 post-training methodology, and our codebase recipes. Key takeaway: our current hyperparameters (LR=5e-6, cosine decay, 5% warmup, weight_decay=0.1, answer_only_loss, 1 epoch) are well-aligned with best practices. The main actionable findings relate to auxiliary loss tuning, the two-stage loss procedure used by NVIDIA, and potential PEFT alternatives for faster iteration.

### Key Findings

#### 1. Learning Rate

- **NVIDIA official (Nemotron 3 Super SFT)**: LR=**1e-5**, constant after warmup. AdamW with beta1=0.9, beta2=0.95, weight_decay=0.1. Warmup over 30,000 samples with linear ramp. ([NVIDIA Nemotron SFT docs](https://docs.nvidia.com/nemotron/latest/nemotron/super3/sft.html))
- **Our setting**: LR=5e-6 with cosine decay and 5% warmup. This is 2x lower than NVIDIA's official recipe.
- **Flan-MoE (ICLR 2024)**: LR=1e-4 for instruction tuning of ST-MoE-32B, with dropout=0.05 and expert_dropout=0.2. Batch size=32, 200k steps. Lower LR and smaller batch size recommended for "stable instruction finetuning at extra-large scales." ([Shen et al., 2024](https://arxiv.org/html/2305.14705))
- **General guidance**: MoE models tend to benefit from slightly higher LRs than dense models of similar active parameter count, because each expert sees fewer tokens per step. However, for SFT (vs pretraining), conservative LRs (1e-6 to 1e-5) are standard.
- **Our experience**: LR=8e-5 caused NaN at iter 4-8 (especially with context parallelism). LR=5e-6 validated stable over 300+ iterations.
- **Recommendation**: Our LR=5e-6 is conservative but safe. Could experiment with LR=1e-5 (matching NVIDIA's recipe) if loss convergence feels slow. Do NOT go above 1e-5 for full SFT.

#### 2. Epochs and Training Duration

- **NVIDIA official**: ~7M SFT samples for the production model, with a two-stage loss procedure. No explicit epoch count given (trained by steps).
- **Our setting**: 1 epoch over 100k examples (~331M tokens, ~2531 iters for Nano, ~1265 for Super).
- **Flan-MoE**: 200k training steps, which for their dataset constitutes multiple epochs.
- **General guidance**: MoE models benefit disproportionately from instruction tuning compared to dense models. Flan-ST-MoE-32B showed a 45.2% improvement from instruction tuning vs only 6.6% for dense Flan-PaLM-62B ([Shen et al., 2024](https://arxiv.org/html/2305.14705)). This suggests MoE models can absorb more SFT data effectively.
- **Recommendation**: 1 epoch is appropriate for a 100k dataset to avoid overfitting. If loss hasn't plateaued by end of epoch, a second epoch with reduced LR could help, but monitor for expert specialization collapse.

#### 3. Batch Size

- **NVIDIA official**: Global batch size=64, micro_batch_size=1, pack_size=4096 tokens.
- **Our Nano config**: Global=16, micro=1, grad_accum=2 (4 nodes, 16 GPUs, DP=8).
- **Our Super config**: Global=32, micro=1 (32 nodes, 128 GPUs, DP=32).
- **Flan-MoE**: Batch size=32 for instruction tuning.
- **Recommendation**: Our batch sizes are reasonable. NVIDIA uses GBS=64 with 256+ GPUs. Scaling GBS proportionally with DP rank is correct. No changes needed.

#### 4. Auxiliary Load Balancing Loss

- **Current setting**: aux_loss_coefficient=0.0001 (1e-4).
- **ST-MoE (Zoph et al., 2022)**: Load balancing loss coefficient=0.01, router z-loss coefficient=0.001. When they turned off aux loss entirely, quality was "not significantly impacted" even with 11% token dropping. ([ST-MoE paper](https://arxiv.org/pdf/2202.08906))
- **Switch Transformers**: aux_loss_coefficient=0.01.
- **DeepSeek-V3**: Moved away from heavy auxiliary losses entirely toward bias-based dynamic updates (auxiliary-loss-free load balancing). ([DeepSeek-V3](https://arxiv.org/html/2408.15664v1))
- **General guidance**: The field is shifting from fixed large auxiliary losses toward adaptive or minimal approaches. During SFT specifically, the fine-tuning dataset is narrower than pretraining data, so forcing perfectly uniform expert distribution may be counterproductive. Reducing aux loss coefficient during SFT is recommended.
- **Recommendation**: Our 1e-4 is already quite conservative (10x lower than ST-MoE's 0.01). This is appropriate for SFT — we don't want to over-regularize routing on a narrow instruction dataset. Could experiment with 0 (no aux loss) during SFT since ST-MoE showed minimal quality impact. Monitor expert utilization metrics to ensure no routing collapse.

#### 5. NVIDIA's Two-Stage Loss Procedure

- **Key insight from NVIDIA**: Nemotron 3 Super uses a distinctive two-stage SFT loss:
  1. **Stage 1**: Loss averages over all output tokens in the packed global batch (standard).
  2. **Stage 2**: Per-conversation normalized loss, averaged equally across conversations. This reduces dominance of longer outputs.
- **MTP continuation**: Multi-token prediction layers continue from pretraining during SFT with MTP loss scaling factor=0.3.
- **Our setting**: We disabled MTP for Super (`mtp_num_layers: null`) because MTP + packed sequences triggers an MCore bug. Standard answer-only loss without per-conversation normalization.
- **Recommendation**: The per-conversation loss normalization is interesting — it prevents long responses from dominating the gradient. Worth implementing if our dataset has high variance in response lengths. However, this requires code changes to the loss computation.

#### 6. Sequence Packing and MoE Routing Interactions

- **Known issue**: Token-level routing in packed sequences means tokens from different examples compete for the same experts within a batch. This can cause load imbalance different from what the model saw during pretraining.
- **Vanilla sequence-level routing is non-causal**: Incompatible with autoregressive decoding when future tokens influence routing decisions. ([Cerebras Router Guide](https://www.cerebras.ai/blog/moe-guide-router))
- **Training-inference discrepancy**: 94% of tokens may select different experts between training and inference in at least one layer, partly due to packing effects. ([MoE routing stability research](https://arxiv.org/html/2510.11370v1))
- **Our setting**: Packed sequences enabled, micro_batch_size=1 (standard for packed SFT).
- **Recommendation**: Packing is essential for SFT efficiency and is used by NVIDIA in their official recipe. The routing discrepancy is a known theoretical concern but not practically significant for SFT at our scale. Keep packing enabled. If we observe unusual expert utilization patterns, consider monitoring per-expert token counts.

#### 7. Parameter-Efficient Alternatives to Full SFT

- **LoRA for MoE**: NVIDIA's official recipe recommends LoRA with LR=1e-4, rank not specified, targeting linear_qkv, linear_proj, linear_fc1, linear_fc2, in_proj, out_proj. TP=1, EP=1 (much simpler parallelism). ([NVIDIA SFT docs](https://docs.nvidia.com/nemotron/latest/nemotron/super3/sft.html))
- **Our codebase recipes**: LoRA with dim=32, alpha=32, LR=1e-4, beta2=0.98.
- **MixLoRA (2024)**: Inserts LoRA experts within FFN blocks of frozen model with top-k routing. ~9% accuracy improvement over standard PEFT. ([MixLoRA](https://github.com/TUDB-Labs/MixLoRA))
- **Expert Pyramid Tuning (2025)**: Decomposes fine-tuning into pyramid of varying expert dimensions. Achieves superior performance with 68x fewer parameters per task on GLUE. ([EPT](https://arxiv.org/html/2603.12577))
- **MoLA (2025)**: Layer-wise LoRA-MoE with different expert counts per layer. Freezes pretrained weights, only tunes LoRA adapters. ([MoLA](https://aclanthology.org/2025.findings-naacl.284.pdf))
- **DR-LoRA (2025)**: Dynamic rank LoRA for MoE — different ranks per expert based on adaptation demand. ([DR-LoRA](https://arxiv.org/html/2601.04823))
- **Key finding from Flan-MoE**: MoE models benefit MORE from instruction tuning than dense models (45.2% vs 6.6% improvement). This suggests full SFT may be worth the cost for MoE models.
- **Recommendation**: For fast iteration and hyperparameter search, LoRA (LR=1e-4, dim=32) with EP=1 TP=1 would be dramatically cheaper (single node). For final production runs, full SFT is preferred given MoE models' outsized benefit from instruction tuning. LoRA is available in our codebase recipes and ready to use.

### Recommendations for Our Setup

**Keep as-is (validated good choices):**
- LR=5e-6 with cosine decay (conservative but stable; NVIDIA uses 1e-5 constant)
- 5% warmup fraction
- Weight decay=0.1 (matches NVIDIA exactly)
- AdamW with beta1=0.9, beta2=0.95 (matches NVIDIA exactly)
- Answer-only loss masking (matches NVIDIA)
- Packed sequences at seq=8192 (NVIDIA uses pack_size=4096)
- 1 epoch for 100k dataset
- Aux loss coefficient=0.0001 (conservative, appropriate for SFT)

**Consider experimenting with:**
1. **LR=1e-5** (matching NVIDIA's recipe) — 2x our current rate, could speed convergence
2. **Constant LR after warmup** instead of cosine decay — this is what NVIDIA uses for SFT
3. **Reducing or removing aux loss** during SFT (ST-MoE showed minimal quality impact)
4. **Per-conversation loss normalization** (NVIDIA's Stage 2) if response length variance is high
5. **LoRA (LR=1e-4, dim=32, EP=1 TP=1)** for fast ablation studies before committing to full SFT

**Do NOT change:**
- Don't increase LR above 1e-5 (NaN risk confirmed)
- Don't increase aux loss coefficient (over-regularization on narrow SFT data)
- Don't disable sequence packing (essential for efficiency, used by NVIDIA)
- Don't add expert dropout during SFT (Flan-MoE used 0.2 but for instruction tuning from scratch, not fine-tuning a converged model)

### Sources

- [NVIDIA Nemotron 3 Super SFT Documentation](https://docs.nvidia.com/nemotron/latest/nemotron/super3/sft.html)
- [Nemotron 3 Super Technical Report (PDF)](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Super-Technical-Report.pdf)
- [Nemotron 3 Nano Technical Report (PDF)](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Nano-Technical-Report.pdf)
- [Mixture-of-Experts Meets Instruction Tuning (Shen et al., ICLR 2024)](https://arxiv.org/html/2305.14705)
- [ST-MoE: Designing Stable and Transferable Sparse Expert Models (Zoph et al., 2022)](https://arxiv.org/pdf/2202.08906)
- [Auxiliary-Loss-Free Load Balancing Strategy (DeepSeek-V3)](https://arxiv.org/html/2408.15664v1)
- [MoE Load Balancing Review (HuggingFace blog)](https://huggingface.co/blog/NormalUhr/moe-balance)
- [Mixture of Experts Explained (HuggingFace blog)](https://huggingface.co/blog/moe)
- [Router Wars: Which MoE Routing Strategy Works (Cerebras)](https://www.cerebras.ai/blog/moe-guide-router)
- [MoE LLMs Survey (2025)](https://arxiv.org/html/2507.11181v1)
- [Expert Pyramid Tuning (2025)](https://arxiv.org/html/2603.12577)
- [MixLoRA: Parameter-Efficient MoE Fine-tuning](https://github.com/TUDB-Labs/MixLoRA)
- [MoLA: MoE LoRA with Layer-wise Expert Allocation (NAACL 2025)](https://aclanthology.org/2025.findings-naacl.284.pdf)
- [DR-LoRA: Dynamic Rank LoRA for MoE (2025)](https://arxiv.org/html/2601.04823)
- [TT-LoRA MoE (SC 2025)](https://dl.acm.org/doi/10.1145/3712285.3759888)


---

## [2026-04-09] Slingshot/CXI NCCL Hang Mitigations — Deep Research

### Summary

Comprehensive web research into Slingshot/CXI NCCL collective hangs, covering: official HPC center documentation (Isambard, CSCS Alps, NERSC, ALCF), NVIDIA NCCL GitHub issues, aws-ofi-nccl releases, libfabric CXI provider documentation, and nvidia-resiliency-ext best practices. The hangs we experience (all ranks blocking simultaneously on collective ops every 5-10 minutes at 32 nodes) are a **well-documented problem** across all Slingshot-based HPC systems. Multiple mitigation strategies exist, and several are not yet applied in our current config.

### Key Findings

#### 1. Our Current Config vs. Official Recommendations

Comparing `train_nemotron_sft.sbatch` against three authoritative sources (Isambard NCCL docs, CSCS Alps docs, libfabric fi_cxi man page):

| Variable | Our Value | Isambard Docs | CSCS Alps | Notes |
|---|---|---|---|---|
| `FI_CXI_DEFAULT_TX_SIZE` | **16384** | **1024** | **16384** | **MISMATCH with Isambard.** Isambard recommends 1024. Larger TX queue = more outstanding rendezvous messages per rank. At 128 GPUs with EP all-to-all, this could exhaust CXI resources. |
| `FI_CXI_RX_MATCH_MODE` | **soft** | Not specified | **software** | Isambard's "soft" may be an alias. CSCS explicitly recommends "software". Hardware mode is known to fail. |
| `FI_CXI_RDZV_PROTO` | **not set** | **alt_read** (recommended) | Not specified | We're missing this. Uses alternative RDMA read path for rendezvous data transfer. May avoid the deadlock in the default rendezvous implementation. |
| `NCCL_NCHANNELS_PER_NET_PEER` | **not set** | **4** (recommended) | **4** | We're missing this. Fixes a 24% perf regression on GH200 at >=8 nodes (NCCL issue #1272). Adjusts CUDA CTA allocation per peer — critical for sparse all-to-all patterns. |
| `NCCL_GDRCOPY_ENABLE` | **not set** | **1** | Not specified | We're missing this. Enables GDRCopy for lower-latency GPU memory registration. |
| `FI_HMEM_CUDA_USE_GDRCOPY` | **not set** | **1** | Not specified | We're missing this. Companion to NCCL_GDRCOPY_ENABLE. |
| `FI_CXI_ENABLE_TRIG_OP_LIMIT` | **not set** | Not specified | Not specified | Enforces triggered operation resource limits using semaphores. Prevents CXI resource exhaustion deadlock. Performance penalty, but prevents the exact failure mode we see. |
| `FI_CXI_DISABLE_NON_INJECT_MSG_IDC` | 1 | 1 | Not specified | Matches |
| `FI_CXI_RDZV_GET_MIN` | 0 | 0 | 0 | Matches |
| `FI_CXI_RDZV_THRESHOLD` | 0 | 0 | 0 | Matches |
| `FI_CXI_RDZV_EAGER_SIZE` | 0 | 0 | 0 | Matches |
| `NCCL_PROTO` | ^LL128 | Not specified | ^LL128 | Matches |
| `NCCL_MIN_NCHANNELS` | 4 | 4 | Not specified | Matches |
| `FI_MR_CACHE_MONITOR` | userfaultfd | userfaultfd | userfaultfd | Matches |

**Key gaps**: `FI_CXI_DEFAULT_TX_SIZE` mismatch, missing `FI_CXI_RDZV_PROTO=alt_read`, missing `NCCL_NCHANNELS_PER_NET_PEER=4`, missing GDRCopy enablement.

#### 2. CXI Resource Exhaustion Root Cause

From the libfabric `fi_cxi(7)` man page and multiple HPC center docs, the root cause of Slingshot NCCL hangs is **CXI resource exhaustion**:

- Each Slingshot NIC (Cassini) has finite triggered operation slots, completion queue entries, and transmit queue capacity
- All-to-all collectives (used by MoE EP) create O(N^2) send/recv pairs, rapidly consuming CXI resources
- When resources are exhausted, the NIC blocks and all ranks stall simultaneously
- The `FI_CXI_DEFAULT_TX_SIZE` controls outstanding rendezvous messages per rank — our value of 16384 is 16x the Isambard-recommended 1024, amplifying resource pressure
- `FI_CXI_ENABLE_TRIG_OP_LIMIT=1` enables semaphore-based resource coordination to prevent exhaustion (at a perf cost)

#### 3. aws-ofi-nccl Version Issues

We're using **aws-ofi-nccl 1.8.1** (via `module load brics/aws-ofi-nccl/1.8.1`). This is significantly outdated.

Recent aws-ofi-nccl releases with critical CXI fixes:
- **v1.17.2** (Nov 2025): "Fixed shutdown ordering issue on NICs that require per-endpoint memory registration (**Cray Slingshot**)"
- **v1.17.3** (Jan 2026): "Fixed a **memory leak** that can result in running out of host memory for **long-running jobs**" — this is highly relevant
- **v1.18.0** (Jan 2026): "Fixed support for non-FI_MR_VIRT_ADDR providers in RDMA protocol"

The memory leak fix in v1.17.3 is particularly concerning — a host memory leak in long-running jobs could cause progressive CXI resource degradation, explaining why hangs become more frequent over time.

#### 4. NCCL Version Considerations

Our NCCL version (2.28.9, bundled in the venv) has specific implications:
- **NCCL 2.27+**: Enabled LL128 protocol by default — we correctly disable it with `NCCL_PROTO=^LL128`
- **NCCL 2.28.3 to 2.28.8**: Had a **segfault bug** with in-process restart. Our 2.28.9 is the minimum safe version.
- **NCCL 2.29**: Includes "fix for a hang if the network plugin returned an error" — could help with CXI error recovery
- **NCCL 2.20+**: Has a performance regression at >=8 nodes on GH200 (issue #1272) that `NCCL_NCHANNELS_PER_NET_PEER=4` fixes

#### 5. In-Process Restart and ft_launcher Best Practices

From the nvidia-resiliency-ext documentation:

**Current config review** (from `pipeline_training_run.py`):
- `FaultToleranceConfig(enable_ft_package=True, calc_ft_timeouts=True)` — Good, auto-calculates timeouts
- `NVRxStragglerDetectionConfig(enabled=True, report_time_interval=120.0)` — Good
- **InProcessRestartConfig is NOT enabled** — We rely solely on ft_launcher restart (kills workers, reloads from checkpoint). Enabling in-process restart could recover from hangs in ~60-90 seconds instead of minutes.

**In-process restart requirements** (all met):
- PyTorch >= 2.5.1 (we have 2.11.0)
- NCCL >= 2.26.2 and not 2.28.3-2.28.8 (we have 2.28.9)
- `NCCL_NVLS_ENABLE=0` (already set)
- `TORCH_NCCL_RETHROW_CUDA_ERRORS=0` (already set)
- `TORCH_NCCL_TIMEOUT` > `hard_timeout` (900 > 90)

**Recommended in-process restart config**:
```python
cfg.inprocess_restart = InProcessRestartConfig(
    enabled=True,
    soft_timeout=60.0,     # Detect hang after 60s of no progress
    hard_timeout=90.0,     # Force-kill hung rank after 90s
    heartbeat_timeout=60.0,
    barrier_timeout=120.0,
    completion_timeout=120.0,
    granularity="node",    # Restart entire node group (safer for TP/EP)
)
```

**Critical caveat**: All NCCL/GPU waits must release the GIL. If any custom code holds the GIL during a collective, in-process restart will deadlock and the rank must be force-killed via ft_launcher.

#### 6. srun `--network=disable_rdzv_get` Flag

Isambard's own NCCL documentation references an `srun --network=disable_rdzv_get` flag that disables rendezvous get at the Slingshot fabric level. This is listed in the fix candidates doc but has never been tested. Combined with `FI_CXI_RDZV_PROTO=alt_read`, this completely bypasses the default rendezvous implementation where the deadlock occurs.

#### 7. NCCL All-to-All Scaling Issue (NCCL Issue #780)

NVIDIA developer Sylvain Jeaugey noted that all-to-all performance instability at large scale (784 GPUs) can be caused by:
- PXN (Proxy Cross-NIC) routing — test with `NCCL_P2P_PXN_LEVEL=0`
- Lack of adaptive routing on the fabric
- Single NIC bottleneck when one NIC routes all traffic

At our scale (128 GPUs, EP=8), the all-to-all creates 128x128 send/recv pairs per MoE layer. Each of the 64 MoE experts routes tokens to all 128 ranks. This is exactly the sparse, bursty pattern that triggers CXI resource exhaustion.

#### 8. GPU-Aware MPI Deadlock Warning

CSCS documentation explicitly warns: "GPU-aware MPI should be disabled explicitly to avoid potential deadlocks between MPI and NCCL." We should verify that `MPICH_GPU_SUPPORT_ENABLED=0` is set. This is listed in the fix candidates but not in our current sbatch.

#### 9. NCCL RAS Subsystem for Hang Diagnosis

NCCL 2.24+ includes a RAS (Reliability, Availability, Serviceability) subsystem. For future hang diagnosis:
```bash
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=RAS,NET,COLL
```
This provides structured information about which collective and which ranks are involved, without the full NCCL_DEBUG=INFO firehose that causes OOM.

### Recommended Actions

**Priority 1 — Env var changes (zero risk, immediate)**:
1. `FI_CXI_DEFAULT_TX_SIZE=1024` — Match Isambard recommendation (currently 16384)
2. `NCCL_NCHANNELS_PER_NET_PEER=4` — Fix GH200 perf regression, improve all-to-all
3. `MPICH_GPU_SUPPORT_ENABLED=0` — Prevent GPU-aware MPI deadlocks
4. `FI_CXI_RDZV_PROTO=alt_read` — Alternative rendezvous that may avoid deadlock

**Priority 2 — Additional env vars (low risk)**:
5. `FI_CXI_ENABLE_TRIG_OP_LIMIT=1` — Semaphore-based CXI resource coordination
6. `NCCL_GDRCOPY_ENABLE=1` + `FI_HMEM_CUDA_USE_GDRCOPY=1` — Lower-latency GPU memory registration
7. `FI_CXI_REQ_BUF_SIZE=8388608` (8MB) — Larger request buffers for software match mode
8. `FI_CXI_RX_MATCH_MODE=software` — Explicit software match (vs. current "soft")

**Priority 3 — srun flag (low risk, needs testing)**:
9. `srun --network=disable_rdzv_get` — Disable rendezvous get at fabric level

**Priority 4 — Enable in-process restart (medium effort)**:
10. Add `InProcessRestartConfig(enabled=True, soft_timeout=60.0, hard_timeout=90.0)` to `pipeline_training_run.py`. This recovers from hangs in ~90s without losing more than the current step, vs. ft_launcher restart which loses up to 25 iterations.

**Priority 5 — aws-ofi-nccl upgrade (high effort, high potential impact)**:
11. Build aws-ofi-nccl v1.17.3+ from source. The memory leak fix is likely relevant — a slow host memory leak could explain progressive CXI resource degradation and increasing hang frequency.

**Priority 6 — NCCL upgrade (high effort)**:
12. Test NCCL 2.29.x which fixes "hang if the network plugin returned an error" — directly relevant to CXI error recovery.

### Sources

- [Isambard NCCL Guide](https://docs.isambard.ac.uk/user-documentation/guides/nccl/) — Official Isambard Slingshot NCCL configuration
- [CSCS Alps NCCL Documentation](https://docs.cscs.ch/software/communication/nccl/) — Swiss supercomputer center, same Slingshot hardware
- [NCCL Issue #1272: GH200 Performance Drop at >=8 Nodes](https://github.com/NVIDIA/nccl/issues/1272) — `NCCL_NCHANNELS_PER_NET_PEER=4` fix
- [NCCL Issue #780: All2All Fluctuation at Large Scale](https://github.com/NVIDIA/nccl/issues/780) — MoE all-to-all instability, PXN investigation
- [libfabric fi_cxi(7) Man Page](https://ofiwg.github.io/libfabric/main/man/fi_cxi.7.html) — Definitive FI_CXI env var reference
- [NCCL Environment Variables](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html) — Official NCCL env var documentation
- [aws-ofi-nccl Releases](https://github.com/aws/aws-ofi-nccl/releases) — Plugin release notes with CXI fixes
- [nvidia-resiliency-ext In-Process Restart Guide](https://nvidia.github.io/nvidia-resiliency-ext/inprocess/usage_guide.html) — In-process restart configuration
- [Megatron Bridge Resiliency Docs](https://docs.nvidia.com/nemo/megatron-bridge/latest/training/resiliency.html) — FT + in-process restart for Megatron Bridge
- [NVIDIA Blog: Fault-Tolerant NCCL Applications](https://developer.nvidia.com/blog/building-scalable-and-fault-tolerant-nccl-applications/) — Best practices for NCCL resilience

---

## [2026-04-09] Super 120B-A12B Throughput Optimization Research

### Summary

Current throughput: ~40-45s/iter, 3.4-3.8 TFLOP/s/GPU on 128 GPUs (32 nodes), TP=4, EP=32, GBS=32, seq=8192. This is very low. Investigation into what NVIDIA reports, dispatcher options, CUDA graph opportunities, and GH200-specific tuning.

### Key Findings

#### 1. NVIDIA Reference Throughput (Pretrain Config)

No public training TFLOP/s numbers found for Nemotron 3 Super. The technical report PDF was not parseable. However, the **pretrain recipe** (`nemotron_3_super.py:29-133`) reveals NVIDIA's intended config:
- **CUDA graphs enabled** with partial scopes: `cuda_graph_scope = ["attn", "mamba", "moe_router", "moe_preprocess"]` — claims **~40% throughput gain** over disabled (line 82-85)
- `cuda_graph_impl = "transformer_engine"`
- `moe_flex_dispatcher_backend = "hybridep"` (though dispatcher type is still "alltoall")
- `gradient_accumulation_fusion = True` (requires APEX — unavailable on Isambard)
- `use_fused_weighted_squared_relu = True`
- `apply_rope_fusion = False`
- `cross_entropy_fusion_impl = "te"`
- `overlap_grad_reduce = True`, `overlap_param_gather = True`
- `manual_gc = False`
- GBS=3072, MBS=1, EP=8 (on NVLink-connected GPUs)

**Critical insight**: The SFT recipe (`nemotron_3_super.py:141-225`) explicitly disables CUDA graphs with the comment: "packed-sequence SFT passes explicit attention masks that are incompatible with CUDA graph capture/replay in Mamba layers." This is our current bottleneck.

#### 2. CUDA Graph Workaround for Packed Sequences

From [Megatron Bridge Performance Guide](https://docs.nvidia.com/nemo/megatron-bridge/latest/performance-guide.html) and [Megatron MoE docs](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/moe.html):
- **`cuda_graph_scope = "attn"` (or `["attn"]`)** can selectively capture attention layers while leaving MoE and Mamba uncaptured
- For packed sequences, enable `pad_cu_seqlens: True` in `packed_sequence_specs` and `pad_to_max_length: True` in dataset config for CUDA graph compatibility
- MoE layers with token-dropless propagation have limited CUDA graph support — restricted to dense modules only
- The pretrain recipe uses `["attn", "mamba", "moe_router", "moe_preprocess"]` — this may work even with packed sequences if `pad_cu_seqlens=True` is set

**Recommendation**: Try enabling `cuda_graph_scope: ["attn"]` or `["attn", "moe_router", "moe_preprocess"]` with `pad_cu_seqlens: True`. This could recover a significant portion of the ~40% gain NVIDIA claims. The Mamba scope may not work with explicit attention masks, but attention-only capture should.

#### 3. MoE Dispatcher Performance

From [Megatron MoE docs](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/moe.html):
- **allgather**: Best for TP-only or very large Top-K. Not recommended for EP>1.
- **alltoall** (current): Standard for EP>1 setups, uses NCCL all-to-all collectives.
- **flex + DeepEP**: Optimized for cross-node EP. Removes redundant tokens during cross-node communication. Requires `--moe-token-dispatcher-type flex --moe-flex-dispatcher-backend deepep`.
- **flex + HybridEP**: NVIDIA's optimized dispatcher with TMA and IBGDA support. Native for GB200 NVL72.
- **DeepEP benchmark**: ~50 GB/s cross-node on 4× DGX-B200 (EP32), ~55 GB/s with IBGDA.

**For Isambard (Slingshot/CXI)**: DeepEP requires RDMA/InfiniBand GPU-NIC affinity (`DEEP_EP_DEVICE_TO_HCA_MAPPING`). Slingshot uses CXI, not IB — DeepEP may not be compatible. The alltoall dispatcher is likely the only viable option unless flex+deepep can work over libfabric.

**Key architectural guideline**: "Keep EP×TP communication within NVLink domain." Current config has EP=32 spanning 32 nodes — all MoE all-to-all goes cross-node over Slingshot. This is a major throughput bottleneck. NVIDIA recommends EP×TP fits within a single node (8 GPUs), using PP to distribute layers across nodes instead.

#### 4. Communication Overlap Opportunities

From [Megatron MoE docs](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/moe.html):
- **EP A2A Overlap**: `--overlap-moe-expert-parallel-comm --delay-wgrad-compute` — merges forward-backward of adjacent microbatches to hide A2A latency. Requires `CUDA_DEVICE_MAX_CONNECTIONS > 1`.
- **Shared Expert Overlap**: `moe_shared_expert_overlap=True` — runs shared expert compute concurrently with EP token transfer. Currently disabled in our config.
- **TP Communication Overlap**: `--tp-comm-overlap` — not currently set.

**Recommendation**: Enable `moe_shared_expert_overlap: True`. This overlaps shared expert forward with the all-to-all dispatch, effectively hiding some communication latency for free.

#### 5. Kernel Fusion Opportunities

Already enabled in our config:
- `moe_grouped_gemm: True` ✓
- `moe_permute_fusion: True` ✓

Not set (available in pretrain recipe):
- `moe_router_fusion` — fuses router + top-k + softmax. Should be enabled.
- `use_fused_weighted_squared_relu: True` — fused activation. Should be enabled.
- `cross_entropy_fusion_impl: "te"` — TE-fused cross entropy. Should be enabled.

#### 6. GH200-Specific Considerations

- **NVLink C2C**: 900 GB/s between CPU and GPU — 7x over PCIe. Good for activation offloading.
- **Single GPU per node**: Each GH200 has 1 GPU + 1 Grace CPU. No intra-node NVLink between GPUs. All EP communication is cross-node over Slingshot (~25 GB/s bidirectional per link).
- **Memory**: 96 GB HBM3 + 480 GB CPU memory accessible via NVLink C2C.
- **Implication**: The NVIDIA guidance to "keep EP×TP within NVLink domain" is impossible on GH200 — there is no NVLink domain spanning multiple GPUs. Every non-local communication goes over Slingshot. This fundamentally limits MoE throughput compared to DGX systems with NVSwitch.

#### 7. Parallelism Restructuring

Current: TP=4, EP=32, PP=1, DP=32 on 128 GPUs.
- TP=4 means each attention/dense layer operation requires 4-way cross-node allreduce.
- EP=32 means each MoE layer requires 32-way cross-node all-to-all.
- With 512 experts and top_k=22, each token activates 22 experts across 32 EP ranks.

**Alternative**: PP>1 to reduce cross-node communication per step:
- PP=4, TP=4, EP=32 would reduce DP to 8 but pipeline the 88 layers across 4 stages (22 layers each). This trades pipeline bubble overhead for reduced DP gradient allreduce.
- PP=2, TP=4, EP=32, DP=16 — more moderate pipeline overhead.

However, PP interacts poorly with MoE fault tolerance (checkpoint granularity) and may complicate the ft_launcher restart logic.

#### 8. Profiling

No dedicated profiling tools found in the bridge training code. The config already has:
- `timing_log_level: 2` and `timing_log_option: minmax` — this enables MCore's built-in timers
- `log_throughput: true` — logs tokens/sec and TFLOP/s

For deeper profiling, use `nsys` (NVIDIA Nsight Systems):
```bash
nsys profile --trace=cuda,nvtx,osrt --output=profile_super \
  python -m torch.distributed.run --nproc_per_node=4 ...
```

#### 9. `empty_unused_memory_level`

This config option was not found in the bridge or MCore codebase at the pinned commit. It may be a newer feature not yet available in this version, or it may be named differently. Not actionable.

### Prioritized Recommendations

1. **[HIGH] Enable partial CUDA graphs**: Set `cuda_graph_impl: "transformer_engine"` and `cuda_graph_scope: ["attn"]` with `pad_cu_seqlens: True`. Test if attention-only capture works with packed sequences. Expected: 10-20% throughput gain.

2. **[HIGH] Enable shared expert overlap**: Set `moe_shared_expert_overlap: True`. Overlaps shared expert compute with A2A dispatch. Expected: 5-15% gain depending on shared expert fraction.

3. **[MEDIUM] Enable router fusion**: Add `moe_router_fusion: True` to config. Fuses router + topk + softmax kernel. Expected: small but free gain.

4. **[MEDIUM] Enable fused activations**: Add `use_fused_weighted_squared_relu: True` and `cross_entropy_fusion_impl: "te"`. Expected: small gains from fewer kernel launches.

5. **[LOW] Investigate EP A2A overlap**: `overlap_moe_expert_parallel_comm` with `delay_wgrad_compute`. More complex, may require code changes. Expected: 10-20% if communication-bound.

6. **[LOW] Profile with nsys**: Run a 5-iteration profile to identify whether bottleneck is compute, communication, or memory. This should inform which of the above optimizations will have the most impact.

7. **[NOT RECOMMENDED] Change dispatcher**: DeepEP/HybridEP unlikely to work on Slingshot/CXI. Stick with alltoall.

8. **[NOT RECOMMENDED] Change parallelism**: Restructuring TP/EP/PP is a large change with fault tolerance implications. Not worth the risk for a 100k SFT run.

### Sources

- [Megatron MoE Performance Guide](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/moe.html)
- [Megatron Bridge Performance Guide](https://docs.nvidia.com/nemo/megatron-bridge/latest/performance-guide.html)
- [Megatron Bridge Packed Sequences](https://docs.nvidia.com/nemo/megatron-bridge/latest/training/packed-sequences.html)
- [NVIDIA Blog: Scaling MoE with Wide EP on NVL72](https://developer.nvidia.com/blog/scaling-large-moe-models-with-wide-expert-parallelism-on-nvl72-rack-scale-systems/)
- [Nemotron 3 Super recipe: `src/megatron/bridge/recipes/nemotronh/nemotron_3_super.py`]
- [Current config: `configs/nemotron_warm_start/nemotron_super_100k_warm_start_sft.yaml`]


---

## [2026-04-09] Checkpoint Optimization for Super 120B-A12B (230 GB Checkpoints)

### Summary

Investigation of checkpoint save/load optimization for Nemotron 3 Super (120B-A12B, 512 experts) on 32 nodes (128 GPUs) with EP=32. Checkpoints are ~230 GB each, saved synchronously every 5 iters. Key problems: (1) synchronous saves block training for minutes, (2) restore after restart sometimes OOMs at 86 GB peak on 95 GB GPUs, (3) async saves caused OOM on Nano. Research covered Megatron-Core dist_checkpointing internals, Megatron Bridge checkpointing layer, and external literature on PyTorch DCP and MoE checkpoint strategies.

### Key Findings

#### 1. Megatron-Core Dist Checkpointing Architecture

**torch_dist format** (what we use):
- Strategy classes: `TorchDistSaveShardedStrategy` and `TorchDistLoadShardedStrategy` in `3rdparty/Megatron-LM/megatron/core/dist_checkpointing/strategies/torch.py`
- Translates MCore ShardedTensors to PyTorch distributed ShardedTensors, then uses PyTorch's `torch.distributed.checkpoint` module
- Uses `FileSystemWriterAsync` for async I/O writes (line 669)
- Supports **cached metadata** (`cached_metadata=True`) — reuses save plan from previous checkpoint, skipping redundant collective operations. PyTorch blog reports this reduced planning time from 436s to 67s at 1856 GPUs.

**No incremental/delta save support** — searched for `incremental`, `delta`, `diff` across all dist_checkpointing code. The `diff()` function in `dict_utils.py` is for offline debugging only, not online checkpoint optimization. PyTorch DCP does not have a production incremental save API as of mid-2026.

**No explicit CPU offload during restore** — the common state dict loader hard-codes `map_location='cpu'` (line 45, `common.py`), but sharded tensor restore loads directly to target device. The `init_device` parameter in `mcore_to_pyt_state_dict()` (line 250, `torch.py`) controls where missing tensors are initialized but is not exposed as a user config.

#### 2. Megatron Bridge Checkpoint Options

From `src/megatron/bridge/training/checkpointing.py` and `config.py`:

| Option | Type | Effect | Relevant? |
|--------|------|--------|-----------|
| `async_save` | bool | Async GPU→CPU staging + background disk write | **YES — primary recommendation** |
| `fully_parallel_save` | bool | Parallelize save across TP/PP dimensions | **YES — reduces save latency** |
| `save_optim` | bool | Include optimizer state (default True) | No — need optimizer for training resume |
| `save_rng` | bool | Include RNG states (default True) | Minimal size impact |
| `dist_ckpt_optim_fully_reshardable` | bool | Fully-reshardable optimizer state | **YES — helps restore OOM** |
| `distrib_optim_fully_reshardable_mem_efficient` | bool | Gloo-based memory-efficient optimizer sharding | **YES — directly targets restore OOM** |
| `use_persistent_ckpt_worker` | bool | Keep background save process alive | Required for `async_save` |

**Low-memory save mode** (`low_memory_save=True` in `model_load_save.py`, lines 447-650):
- Reduces peak save memory to ~50% by expanding ShardedTensorFactory objects, cloning data, deleting the model, then saving
- **Destroys the model after save** — cannot continue training. Only useful for final checkpoint export, not mid-training saves.
- Requires `dist_ckpt_optim_fully_reshardable=True`

**Async save pipeline** (`checkpointing.py`, lines 330-385):
- `schedule_async_save()` stages GPU tensors to CPU synchronously (fast — <1 second even at 1856 GPUs per PyTorch blog), then writes to disk in background
- Training resumes immediately after staging
- CPU memory requirement: `checkpoint_size_per_rank` additional CPU memory per node
- For Super: 230 GB / 128 GPUs = **~1.8 GB per rank** — trivial on GH200 with 480 GB unified memory
- GC is disabled during async writes (`_disable_gc()` context manager in `async_utils.py`) and `gc.collect()` called after completion

#### 3. `empty_unused_memory_level` Setting

Found in `3rdparty/Megatron-LM/megatron/training/config/training_config.py` (line 35). Used in `train.py` (lines 779, 816) and `eval.py` (line 213):

- **Level 0** (default): No `torch.cuda.empty_cache()` calls
- **Level 1**: Calls `torch.cuda.empty_cache()` before the optimizer step (after backward) and after each eval step
- **Level 2**: All of level 1, plus additional `torch.cuda.empty_cache()` after the optimizer step

**This does NOT directly affect checkpoint save/load.** It reduces GPU memory fragmentation during training, giving checkpoint operations more headroom. For our 86 GB peak on 95 GB GPUs (only 9 GB headroom), **level 2 is worth enabling** to maximize free contiguous memory before checkpoint operations trigger allocations.

#### 4. Restore OOM Root Cause

The restore OOM (86 GB peak, 95 GB GPU limit) is caused by the H2D tensor transfer path temporarily doubling some tensors during checkpoint load. With EP=32 and 512 experts, each rank loads 16 experts. The per-rank model state is ~1.8 GB, but optimizer states (Adam: 2 states per param, BF16 master weights) add ~3.6 GB. During restore, both the fresh model tensors AND the checkpoint tensors coexist briefly.

**Mitigation options**:
- `distrib_optim_fully_reshardable_mem_efficient=True` — Uses Gloo backend for memory-efficient distributed optimizer restore. Trades speed for lower peak memory. This directly targets the transient doubling problem.
- `ckpt-fully-parallel-load` — Parallelizes load across DP ranks, spreading memory pressure. Each rank loads a smaller slice and redistributes.
- `empty_unused_memory_level=2` — Frees fragmented GPU memory before restore, giving more headroom.

#### 5. PyTorch DCP Compression & Incremental Saves

- **Compression**: PyTorch DCP supports `StreamTransformExtension` for zstd compression (~22% size reduction). Storage/bandwidth optimization only — doesn't help GPU memory during restore.
- **Incremental/delta saves**: No production-ready API in PyTorch DCP as of mid-2026. The DCP team has discussed deduplication as future work but nothing shipped.
- **Cached save plans**: After the first checkpoint, subsequent saves reuse metadata plans. Avoids expensive collective operations. Reduced background processing from 436s to 67s at 1856-GPU scale.

#### 6. MoE Checkpoint Sharding (Already Optimal)

With our current parallelism (TP=4, EP=32, DP=1):
- Each rank saves only its 16 experts (512 / 32 EP)
- TP=4 further shards attention/shared weights
- Distributed optimizer shards states across ranks
- Per-rank checkpoint: 230 GB / 128 = ~1.8 GB — small enough for async staging

### Recommendations

**Priority 1 — Enable async save (high confidence, low risk)**:
```yaml
checkpoint:
  async_save: true
  use_persistent_ckpt_worker: true
```
Per-rank checkpoint is only ~1.8 GB. GH200 has 480 GB unified memory. The Nano OOM was likely due to higher per-rank checkpoint size with fewer GPUs (16 vs 128). At 128 GPUs, async save should be safe and would eliminate the multi-minute training pause per checkpoint.

**Priority 2 — Enable fully parallel save**:
```yaml
checkpoint:
  fully_parallel_save: true
```
Parallelizes save across TP/PP dimensions, reducing save latency.

**Priority 3 — Enable memory-efficient optimizer restore**:
```yaml
checkpoint:
  dist_ckpt_optim_fully_reshardable: true
  distrib_optim_fully_reshardable_mem_efficient: true
```
Uses Gloo backend for memory-efficient distributed optimizer state loading. Slower than default, but directly targets the 86 GB peak → OOM during restore. **Note**: Cannot use Gloo process groups simultaneously with `mem_efficient` mode per assertion in `config.py` lines 636-639.

**Priority 4 — Set `empty_unused_memory_level: 2`**:
```yaml
train:
  empty_unused_memory_level: 2
```
Calls `torch.cuda.empty_cache()` before and after optimizer step. Frees fragmented GPU memory, giving checkpoint operations more headroom. Minor perf cost (~1-2% iteration time).

**Priority 5 — Increase save interval if async works**:
If async save eliminates training pause, consider increasing save interval from every 5 iters to every 10-20 iters. This reduces disk I/O and NFS contention. The fault tolerance stack (in-process restart + ft_launcher) already handles NCCL hangs within seconds/minutes, so frequent checkpoints are mainly insurance against ft_launcher exhausting max_restarts.

**Not recommended**:
- `save_optim: false` — Would reduce checkpoint by ~80% but prevents training resume. Only useful for final export.
- `low_memory_save: true` — Destroys model after save. Only for final checkpoint export.
- PyTorch DCP zstd compression — Storage optimization only, doesn't help GPU memory.
- Incremental/delta saves — No production implementation exists in PyTorch DCP.

### Sources

- Megatron-Core dist_checkpointing: `3rdparty/Megatron-LM/megatron/core/dist_checkpointing/strategies/torch.py`
- Megatron Bridge checkpointing: `src/megatron/bridge/training/checkpointing.py`
- Megatron Bridge config: `src/megatron/bridge/training/config.py`
- Low-memory save: `src/megatron/bridge/training/model_load_save.py` (lines 447-650)
- [PyTorch Blog: 6x Faster Async Checkpointing](https://pytorch.org/blog/6x-faster-async-checkpointing/)
- [PyTorch Blog: Reducing Storage Footprint with DCP Compression](https://pytorch.org/blog/reducing-storage-footprint-and-bandwidth-usage-for-distributed-checkpoints-with-pytorch-dcp/)
- [PyTorch DCP Documentation](https://docs.pytorch.org/docs/stable/distributed.checkpoint.html)
- [Megatron Bridge Checkpointing Docs](https://docs.nvidia.com/nemo/megatron-bridge/0.2.0/training/checkpointing.html)
- [NVIDIA Megatron MoE README](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/moe/README.md)

---

## [2026-04-09] Scaling to Full Dolci-Instruct-SFT Dataset

### Background

We've been training on `geodesic-research/sft-warm-start-100k` (100k examples, 331M tokens). The next step is the full `allenai/Dolci-Instruct-SFT` dataset. This section documents what exists, what's needed, and training time estimates.

### Dataset Overview

[allenai/Dolci-Instruct-SFT](https://huggingface.co/datasets/allenai/Dolci-Instruct-SFT) is the SFT mixture used to train OLMo 3 7B Instruct. It contains **2,152,112 total examples** (2,044,506 train + 107,606 validation) from diverse sources including:

- OpenThoughts 3 (941k→99k prompts, reasoning traces removed for instruct)
- Verifiable reasoning (310k prompts)
- Dolci Instruct Tool Use (228k prompts)
- Logic puzzles (160k prompts)
- Dolci Tülu 3 Precise IF (137k prompts)
- FLAN v2 (90k prompts)
- CoCoNot (11k), OpenAssistant Guanaco (7k), and more

Quality-filtered via Azure API (blocked requests filtered out), so counts are smaller than original Tülu 3/OLMo 2 sources.

### Data Availability — What Already Exists

**Raw data** (all on Isambard):

| Path | Contents | Size |
|------|----------|------|
| `/projects/a5k/public/data/allenai__Dolci-Instruct-SFT/` | Full HF dataset (parquet + JSONL) | 17 GB |
| `/projects/a5k/public/data/allenai__Dolci-Instruct-SFT/training.jsonl` | 2,044,506 training examples | ~4.3 GB |
| `/projects/a5k/public/data/allenai__Dolci-Instruct-SFT/validation.jsonl` | 107,606 validation examples | ~1.5 GB |
| `/projects/a5k/public/data/dolci_sft_megatron/` | Megatron-formatted JSONL (full + 100k subset) | ~6 GB |

**Packed data** (already generated, at `/projects/a5k/public/data/allenai__Dolci-Instruct-SFT/packed/`):

| Variant | Seq Length | Packing Factor (train/val) | Efficiency | Packed Seqs (train) | Index Size |
|---------|-----------|---------------------------|------------|--------------------:|------------|
| `pad_seq_to_mult1` | 8192 | 12.31 / 12.22 | 99.89% | ~166,085 | 1.9 GB |
| `pad_seq_to_mult4` | 16384 | 23.93 / 23.79 | 99.94% | ~85,437 | 2.1 GB |

Total packed data: **4.3 GB** (parquet index files only — the actual token data is read from the source JSONL at runtime).

**Key finding: Packed data for the full dataset already exists at both seq lengths. No additional packing jobs needed.**

### Training Time Estimates

Comparison with 100k warm-start: the full dataset is **20.4x more examples** but only **4.1x more tokens** (because the warm-start subset was biased toward longer examples).

| Config | Total Tokens | Packing Factor | Packed Seqs | GBS | train_iters | Iter Time | Wall Clock (ideal) |
|--------|-------------|----------------|-------------|-----|------------|-----------|-------------------|
| **Nano seq=8192** (32 nodes) | 1.36B | 12.31 | 166,085 | 64 | **2,595** | 2.0s | **1.4h** |
| **Super seq=8192** (32 nodes) | 1.36B | 12.31 | 166,085 | 32 | **5,190** | TBD | TBD |
| Nano seq=16384, CP=2 (32 nodes) | 1.40B | 23.93 | 85,437 | 64 | 1,335 | 12.3s | 4.6h |
| Nano seq=16384, CP=2, GBS=32 | 1.40B | 23.93 | 85,437 | 32 | 2,670 | 12.3s | 9.1h |

**Notes on wall clock:**
- "Ideal" = no NCCL hangs, no restarts. Real time is 2-5x longer due to Slingshot CXI issues.
- Nano seq=8192 at 2,595 iters is very manageable — comparable to the 100k warm-start (1,265 iters for Super, ~same order).
- Super iter time needs to be measured from the warm-start run (currently in progress).

### Existing Configs

- **Nano full dataset**: `configs/nemotron_nano_dolci_instruct_sft.yaml` — **already configured** with correct dataset path, train_iters=2595, GBS=64, 32-node parallelism (TP=2, EP=8).
- **Super full dataset**: **No config exists yet.** Needs to be created, modeled on `configs/nemotron_warm_start/nemotron_super_100k_warm_start_sft.yaml` but pointing to the full dataset and with updated train_iters.

### Config Changes Needed for Super Full-Dataset Run

Starting from the Super 100k warm-start config, the following changes are needed:

1. **Dataset path**: Change `dataset_root` to `/projects/a5k/public/data/allenai__Dolci-Instruct-SFT`
2. **Dataset name**: Change to `allenai/Dolci-Instruct-SFT`
3. **train_iters**: Change from 1,265 to **5,190** (166,085 packed seqs / GBS 32)
4. **Checkpoint paths**: New save/load paths for the full run
5. **Pretrained checkpoint**: Decide whether to warm-start from the 100k SFT checkpoint or from the base model
6. **Validation**: Consider enabling `do_validation: true` (the full dataset has a proper validation split)
7. **save_interval**: Increase from 5 (appropriate for 1,265 iters) to something like 50-100 for 5,190 iters
8. **wandb_exp_name**: Update to reflect full dataset run

### Disk Space

- Lustre filesystem: 108 TB available (47% of 200 TB used). No disk pressure.
- Packed data for full dataset: 4.3 GB (already exists).
- Checkpoint storage: Super checkpoints are large (~80 GB each). With save_interval=100 and most_recent_k=2, need ~160 GB free for checkpoints. This is fine.

### Recommendations

1. **Nano full run is ready to launch** — `configs/nemotron_nano_dolci_instruct_sft.yaml` is already configured with correct parameters. Just submit: `isambard_sbatch --nodes=32 train_nemotron_sft.sbatch configs/nemotron_nano_dolci_instruct_sft.yaml nano`
2. **Create Super full-dataset config** — Clone the warm-start config, update dataset/iters/checkpoints.
3. **Decide warm-start vs cold-start** — Should the full run continue from the 100k warm-start checkpoint (transfer learning) or start from the base pretrained model? The warm-start approach saves ~1,265 iters of compute but may have suboptimal optimizer state for the full dataset's learning rate schedule.
4. **Measure Super iter time** — Once the Super 100k warm-start run completes, extract the steady-state iter time to fill in the Super wall clock estimates above.

---

## [2026-04-09] Evaluation Strategy for Fine-Tuned Nemotron Models

### Context

Nemotron 3 Nano (30B-A3B) SFT is complete (checkpoint at iter 2531). Nemotron 3 Super (120B-A12B) SFT is in progress. Both are Megatron-format checkpoints on Isambard. This section documents how to evaluate them.

### 1. Evaluation Infrastructure in the Codebase

The codebase has a full evaluation pipeline at `examples/evaluation/`:

- **`launch_evaluation_pipeline.py`** — Orchestration script using NeMo Run + Ray. Deploys an inference server, runs eval tasks, logs to wandb.
- **`eval.sh`** — Core eval script. Installs `lm-evaluation-harness`, uses `nemo_evaluator` API, runs against an HTTP server endpoint (completions or chat/completions). Default benchmark: MMLU.
- **`deploy.sh`** — Deploys the Megatron model as an inference server via Export-Deploy's `deploy_ray_inframework.py`.
- **`argument_parser.py`** — CLI args for deployment (checkpoint path, GPU count, parallelism), evaluation (task, sampling params, endpoint type), SLURM, wandb logging.

**Key limitation**: This pipeline assumes container-based infrastructure (NeMo containers, DGXC/Kubernetes). It may need adaptation for bare-metal Isambard. The deployment step uses Ray and Export-Deploy which may not be installed.

Training-time validation is in `src/megatron/bridge/training/eval.py` — handles loss-based validation during training with TensorBoard/wandb logging, but does not support external benchmark tasks.

### 2. Checkpoint Conversion (Megatron → HuggingFace)

Conversion is well-supported via `AutoBridge`:

**Single-GPU (small models):**
```python
from megatron.bridge import AutoBridge
AutoBridge.export_ckpt(
    megatron_path="./checkpoints/iter_0002531",
    hf_path="./exports/nemotron3_nano_sft_hf",
    source_path="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16"
)
```

**Multi-GPU (required for MoE models with EP):**
```bash
# Nano (30B-A3B) — needs EP for MoE experts
torchrun --nproc_per_node=4 examples/conversion/convert_checkpoints_multi_gpu.py export \
    --hf-model nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16 \
    --megatron-path ./checkpoints/nemotron_nano_sft/iter_0002531 \
    --hf-path ./exports/nemotron3_nano_sft_hf \
    --tp 2 --ep 8

# Super (120B-A12B) — larger, needs more GPUs
torchrun --nproc_per_node=8 examples/conversion/convert_checkpoints_multi_gpu.py export \
    --hf-model nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16 \
    --megatron-path ./checkpoints/nemotron_super_sft/iter_XXXX \
    --hf-path ./exports/nemotron3_super_sft_hf \
    --tp 4 --ep 64
```

Existing conversion scripts are at:
- `examples/models/nemotron_3/nano/conversion.sh`
- `examples/models/nemotron_3/super/conversion.sh`

Roundtrip verification tools exist at `examples/conversion/hf_megatron_roundtrip_multi_gpu.py`.

### 3. NVIDIA's Nemotron 3 Benchmark Suite

NVIDIA evaluates Nemotron 3 models on:
- **MMLU-Pro** — broad knowledge (multiple choice)
- **GPQA** — graduate-level science reasoning
- **BFCL v3/v4** — function calling
- **LiveCodeBench** — coding
- **AIME 2025** — math competition problems
- **SciCode** — scientific coding
- **IFBench** — instruction following
- **Humanity's Last Exam** — hard reasoning
- **RULER** — long-context
- **Terminal-Bench Hard** and **GDPval-AA ELO** (Super-specific)

NVIDIA uses their **NeMo Evaluator SDK** and **NeMo Skills Harness** for evaluation. A reproducibility guide for Nemotron 3 Nano is at: `https://github.com/NVIDIA-NeMo/Evaluator/blob/main/packages/nemo-evaluator-launcher/examples/nemotron/nano-v3-reproducibility.md`

### 4. Can We Eval Directly from Megatron Format?

**lm-eval-harness supports Megatron** via `--model megatron_lm`:
```bash
lm_eval --model megatron_lm \
  --model_args load=/path/to/checkpoint,tokenizer_type=HuggingFaceTokenizer,tokenizer_model=/path/to/tokenizer \
  --tasks hellaswag --batch_size 1
```

**However**, the `megatron_lm` model type may not support Megatron-Core hybrid Mamba-Transformer MoE architectures (which Nemotron 3 uses). The safer path is **convert to HuggingFace first**, then run standard lm-eval.

### 5. Recommended Eval Suite for SFT Models

Standard SFT evaluation benchmarks (2024-2025 consensus):

| Benchmark | What it measures | Priority |
|-----------|-----------------|----------|
| **IFEval** | Instruction-following accuracy (9 constraint categories, verifiable) | **High** — directly measures SFT quality |
| **MT-Bench** | Multi-turn dialogue quality (8 categories, GPT-4 judged) | **High** — conversation quality |
| **MMLU / MMLU-Pro** | Broad knowledge retention | **High** — ensures SFT didn't degrade base capabilities |
| **AlpacaEval 2.0** | Single-turn instruction quality (length-controlled win rate) | Medium |
| **GPQA** | Graduate-level science reasoning | Medium |
| **HumanEval / LiveCodeBench** | Code generation | Medium (if code is a use case) |
| **GSM8K / MATH** | Mathematical reasoning | Medium |

### 6. Recommended Evaluation Plan

**Phase 1: Quick validation (can do now with Nano checkpoint)**
1. Convert Nano Megatron checkpoint to HuggingFace format using `convert_checkpoints_multi_gpu.py` (needs a GPU node, ~30 min)
2. Run lm-eval-harness with MMLU + IFEval on the HF checkpoint
3. Compare against base model (nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16) to measure SFT impact

**Phase 2: Full eval suite (after Super SFT completes)**
1. Convert both Nano and Super checkpoints to HuggingFace format
2. Run IFEval, MT-Bench, MMLU-Pro, GPQA across both models
3. Compare against base models and published NVIDIA results
4. Log all results to wandb for comparison

**Phase 3: NVIDIA-standard eval (optional, for paper-ready numbers)**
1. Set up NeMo Evaluator SDK (may need container or significant adaptation for Isambard)
2. Run NVIDIA's full benchmark suite (BFCL, LiveCodeBench, AIME, etc.)

### Key Decisions Needed

1. **Conversion parallelism**: The Nano SFT used TP=2, EP=8. The export script needs to match the training parallelism — verify the correct `--tp` and `--ep` flags for the checkpoint.
2. **Disk space**: HuggingFace export for Nano will be ~60GB, Super ~240GB. Check `/projects/a5k/` quota.
3. **Compute for eval**: lm-eval on MMLU takes ~2-4 hours on a single GPU for a 30B model. MT-Bench needs an LLM judge (GPT-4 API access or local judge model).
4. **Base model comparison**: Do we have base model eval numbers already, or do we need to run those too?

### Sources

- NVIDIA Nemotron 3 Family: `https://research.nvidia.com/labs/nemotron/Nemotron-3/`
- Nemotron 3 Nano Evaluation Recipe (HuggingFace blog): `https://huggingface.co/blog/nvidia/nemotron-3-nano-evaluation-recipe`
- Nemotron 3 Super Technical Report: `https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Super-Technical-Report.pdf`
- lm-eval-harness Megatron support: `https://github.com/EleutherAI/lm-evaluation-harness/issues/1210`
