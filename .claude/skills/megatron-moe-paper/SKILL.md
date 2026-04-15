---
name: megatron-moe-paper
description: Reference for Megatron-Core MoE training best practices, parallelism strategies, memory/communication/compute optimization, FP8/FP4 precision, and performance tuning. Based on NVIDIA's "Scalable Training of Mixture-of-Experts Models with Megatron Core" (arxiv 2603.07685v2).
argument-hint: "[topic]"
allowed-tools: Read Grep Glob
---

# Megatron-Core MoE Training: Best Practices & Concepts

Reference extracted from **"Scalable Training of Mixture-of-Experts Models with Megatron Core"** (NVIDIA, 2026). Paper: https://arxiv.org/abs/2603.07685

The full paper content is available locally as markdown:
- **Markdown**: `.claude/skills/megatron-moe-paper/megatron-moe-paper.md`
- **Original PDF**: `.claude/skills/megatron-moe-paper/2603.07685.pdf`

Use this skill when the user asks about Megatron MoE training concepts, parallelism strategies, optimization techniques, or best practices.

When answering questions, read the relevant sections from the local markdown file using the Read tool. The file is ~250KB and ~1940 lines, so read specific sections rather than the whole file. Use Grep to find the relevant section first.

## Paper Structure (section line numbers for targeted reads)

1. **Introduction** (MoE paradigm, three walls) — search for `# 1. Introduction`
2. **Megatron-Core MoE Architecture** (modular design, forward pass) — search for `# 2. Megatron`
3. **Parallel Folding & Multi-Dimensional Parallelism** — search for `# 3. Scaling`
4. **Breaking Memory, Communication, and Compute Walls** — search for `# 4. Scaling`
5. **Reduced-Precision Training (FP8/FP4)** — search for `# 5. Reduced`
6. **Long-Context MoE Training** — search for `# 6. Long`
7. **Production Features** (load balancing, shared experts, checkpointing) — search for `# 7. Production`
8. **Performance Evaluation** — search for `# 8. Performance`
9. **Performance Best Practices** (systematic tuning workflow) — search for `# 9. Performance`
10. **MoE in Reinforcement Learning** — search for `# 10. Megatron`
11. **Conclusion** — search for `# 11. Conclusion`

## Key Tables

| Table | Content | Search term |
|-------|---------|-------------|
| Table 1 | MoE component to process group mapping | `Groups Used` |
| Table 2 | Attention vs MoE parallelism requirements | `Contrasting parallelism` |
| Table 4 | Memory reduction from recomputation (DeepSeek-V3) | `Memory reduction per GPU` |
| Table 7 | EP scaling: HybridEP vs all-to-all latency | `EP Scaling Performance` |
| Table 8 | Reduced-precision impact on three walls | `impact from reduced-precision` |
| Table 9 | SDPA performance (cuDNN) | `SDPA performance` |
| Table 11 | Unified throughput benchmarks (GB300/GB200/H100) | `Unified throughput benchmarks` |
| Table 13 | Memory bottleneck solutions + configs | `Memory bottleneck solutions` |
| Table 15 | CPU overhead bottleneck solutions | `CPU overhead bottleneck` |
| Table 16 | Computation bottleneck solutions | `Computation bottleneck` |
| Table 20 | Benchmark parallelism configurations | `Parallelism and training configuration` |
