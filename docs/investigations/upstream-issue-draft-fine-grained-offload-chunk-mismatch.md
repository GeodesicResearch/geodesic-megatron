# Upstream issue draft — `Chunk mismatch` in fine-grained activation offloading

**STATUS: DRAFT — NOT POSTED.** Filing on `NVIDIA/Megatron-LM` under the org's name is an
outward-facing action and needs **Kyle's sign-off**. Do not `gh issue create` this without it.

- **Target repo:** `NVIDIA/Megatron-LM`
- **Bug:** #5 in `docs/investigations/120b-gbs64-host-overhead-investigation.md:151`
- **Drafted:** 2026-08-02, from the existing repro — no new GPU time was spent (the second arm
  that would have re-produced this, `configs/infr71_vpp/q3b_recompute_off_offload.yaml`, was
  dropped from the wave-2 ladder for exactly that reason; it remains in-tree as the ready-made
  experiment for when this is fixed).
- **Primary evidence:** W&B run `mc6wztvs`, log at
  `/projects/a5k/public/logs/wandb/wandb/run-20260727_143844-mc6wztvs/files/output.log`
- **Before posting:** re-check that the file is still untouched upstream (command in the
  Verification section below) — if NVIDIA has since committed to it, re-test before filing.

Everything below the line is the issue body, ready to copy-paste.

---

## Title

`AssertionError: Chunk mismatch` in fine-grained activation offloading under non-interleaved pipeline parallelism

## Summary

With `--fine-grained-activation-offloading` enabled, training dies in the backward pass of
iteration 2 with `AssertionError: Chunk mismatch` raised from
`megatron/core/pipeline_parallel/fine_grained_activation_offload.py::ChunkOffloadHandler.on_group_commit_backward`.

We hit this under **plain, non-interleaved PP=8** (`virtual_pipeline_model_parallel_size` unset),
which `docs/user-guide/features/fine_grained_activation_offloading.md` lists as supported — its
compatibility matrix has `PP / Interleaved PP / PP=1 | Yes`. We also hit the same assert under
interleaved PP (VPP=2 and VPP=4) on the same model, so the failure does not appear to be specific
to one pipeline schedule.

The failure is deterministic across reruns and always lands at the same point: iteration 1
completes normally, iteration 2's backward raises.

## Environment

| | |
|---|---|
| Megatron-LM | `6cd6ea530` (our pin; plus one unrelated carried commit in a fork) |
| `fine_grained_activation_offload.py` at | `69c486825` — "[Main][feat] Support CUDA Graph capture offloading modules (#3697)", 2026-07-02, which is **also the newest upstream commit to that file**, so this should reproduce on current `main` |
| PyTorch | 2.11.0a0+eb65b36914.nv26.02 |
| Transformer Engine | 2.14.1 |
| CUDA / NCCL | 13.1 / 2.29.2 |
| Python | 3.12.3 |
| Hardware | 64 × NVIDIA GH200 (16 nodes × 4), aarch64 (Grace), `sm_90` |
| Container | `nvcr.io/nvidia/nemo:26.04` |

## Model and configuration

NemotronH hybrid — 88 layers as 40 Mamba2 + 8 attention + 40 latent-MoE
(`hybrid_override_pattern`), 512 routed experts with top-k 22, `moe_latent_size` 1024, one shared
expert. Built through the `HybridModel` / `MambaModel` path, not `GPTModel`.

```yaml
# parallelism — note NO virtual_pipeline_model_parallel_size
pipeline_model_parallel_size: 8
tensor_model_parallel_size: 1
expert_model_parallel_size: 4
expert_tensor_parallel_size: 1
context_parallel_size: 4
sequence_parallel: true
seq_length: 32768
micro_batch_size: 1
global_batch_size: 64          # -> DP=2, 32 microbatches per replica

recompute_granularity: selective
recompute_modules: ["moe", "shared_experts"]

fine_grained_activation_offloading: true
offload_modules: ["core_attn", "attn_proj"]
```

`offload_modules` is deliberately attention-side only. Adding `expert_fc1` / `moe_act` is
rejected earlier, at config-validation time, by the assert in `transformer_config.py` that
forbids offloading inside a recomputed MoE — so this configuration is the one that gets far
enough to reach the runtime failure.

## Traceback

```
  File ".../megatron/core/pipeline_parallel/fine_grained_activation_offload.py", line 1246, in backward
    cpu_offload_handler.on_group_commit_backward(ctx.name)
  File ".../megatron/core/pipeline_parallel/fine_grained_activation_offload.py", line 1161, in on_group_commit_backward
    assert cur_backward_chunk is self, f"Chunk mismatch {cur_backward_chunk} {self}"
           ^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: Chunk mismatch <megatron.core.pipeline_parallel.fine_grained_activation_offload.ChunkOffloadHandler object at 0x400a5b9c5250> <megatron.core.pipeline_parallel.fine_grained_activation_offload.ChunkOffloadHandler object at 0x400a54bde600>
```

The two handlers are distinct live objects, so the manager's current backward chunk is genuinely
a different `ChunkOffloadHandler` instance than the one whose group is committing.

## Where it comes from

`ChunkOffloadHandler.on_group_commit_backward` tries to make itself current, then asserts it
succeeded:

```python
def on_group_commit_backward(self, name):
    if not self.do_offload:
        return
    cur_backward_chunk = PipelineOffloadManager.get_instance().cur_backward_chunk()
    # Switch to this chunk if it's not already current
    if cur_backward_chunk is not self:
        PipelineOffloadManager.get_instance().pop_backward_chunk(name)
    cur_backward_chunk = PipelineOffloadManager.get_instance().cur_backward_chunk()
    assert cur_backward_chunk is self, f"Chunk mismatch {cur_backward_chunk} {self}"
```

A single `pop_backward_chunk` recovers only when the intended chunk is exactly one position away
on the manager's stack. Under 1F1B the backward order across chunks does not have to satisfy
that, and once it does not, the assert fires rather than the manager resynchronising.

Note that `init_chunk_handler` normalises `vp_size = 1 if vp_size is None else vp_size`, so a
`ChunkOffloadHandler` is created per model chunk even without virtual pipelining — which is
consistent with our seeing this at VPP=1 as well as VPP>1, and suggests the root cause is
chunk-ordering bookkeeping in `PipelineOffloadManager` rather than anything interleave-specific.

## Expected behaviour

Either fine-grained activation offloading works under the pipeline configurations the
documentation lists as supported, or the compatibility matrix in
`docs/user-guide/features/fine_grained_activation_offloading.md` is narrowed and the unsupported
combinations fail at config-validation time with an actionable message, rather than at the second
iteration's backward.

## Impact and workaround

We disable `fine_grained_activation_offloading` entirely. It is the one activation-memory
mechanism in Megatron-Core we are not able to use on this model — selective recomputation,
the precision-aware optimizer, optimizer offloading and memory-efficient permutation are all
already in play. Losing it means MoE activation memory has to be reclaimed by recomputing the
whole MoE block instead, which costs several percent of iteration time at our operating point,
where offloading to host over NVLink-C2C might have been the cheaper trade.

## Verification before posting

Re-run this to confirm the file is still untouched upstream; if the newest SHA is no longer
`69c486825`, re-test before filing.

```bash
gh api "repos/NVIDIA/Megatron-LM/commits?path=megatron/core/pipeline_parallel/fine_grained_activation_offload.py&per_page=5" \
  --jq '.[] | "\(.sha[0:9]) \(.commit.author.date) \(.commit.message | split("\n")[0])"'
```

We are happy to test a patch on the hybrid model at 64 GPUs and report back.
