# Local Megatron-LM changes

Changes this repo carries against upstream Megatron-LM, in two forms.

- **Carried commits** of the pinned fork branch (`geodesic/mcore-6cd6ea530-nano-sft` of GeodesicResearch/Megatron-LM),
  which every run uses: 0003, 0004 and 0005, the Nano-30B pretraining campaign's Megatron-LM changes
  (`docs/investigations/nano30b-pretrain-perf-campaign.md`; its fastest configuration, the Nano pretrain
  quickstart, needs all three, and the Nano midtrain and SFT quickstarts need 0005 for their chunked cross-entropy
  and run the HybridEP path 0004 fixes), and 0006, an upstream fix the packed SFT padding mask needs
  (`docs/investigations/nano30b-sft-perf-campaign.md`), on top of the fork's nvrx-probe commit (the pin history
  notes below).
- **Patch files** in this directory, which no run applies: 0001 and 0002. A run that needs one applies it
  to a copy of the checkout:

  ```bash
  git -C 3rdparty/Megatron-LM am ../patches/megatron-lm/<patch>
  ```

| Change | Why it exists | Load-bearing for |
|---|---|---|
| `0001-fix-moe-normalize-allgather-dispatcher-output-by-EP-.patch` | Geodesic fix: normalize allgather-dispatcher output by EP size. Was previously a local-only submodule commit (`2034d4500`) that no remote contained — every fresh clone silently failed to fetch the pin and checked out a different mcore (caught by the INFR-68 fresh-install certification). The submodule now pins the patch's reachable upstream parent (`3758b54b2`, the TE-2.14 bump) and the fix lives here instead. | The `allgather` MoE token dispatcher ONLY. No shipped config or recipe uses it: every config sets `alltoall` except the `flex` of the three Nano quickstarts, the xl-50b SFT rerun (v2) and its quality-filtered retrain (v3), and the control-pretraining V2 E2E arm, and the recipes default to `alltoall` or `flex`, so the running behavior of every committed config is identical with or without it. Apply before using `moe_token_dispatcher_type: allgather`. |
| `0002-fix-cuda-graph-zeros_like-0dim-tensor.patch` | Upstream `zeros_like` on a 0-dim tensor breaks CUDA-graph capture (`cuda_graphs.py:181` unpacks `*self.shape` to nothing). Still open upstream at the current pin. | CUDA graphs only. No shipped config enables them, so every committed config runs identically with or without it. Apply before enabling CUDA graphs. |
| 0003, carried commit `3e3c83d50` | Upstream supports the EP all-to-all / compute overlap (`overlap_moe_expert_parallel_comm`, the combined-1F1B schedule) for `GPTModel` only. This patch is the port of upstream PR #4798 (open; head `1fdff667`), which adds it for the hybrid model, minus #4941, which the pin already contains. Geodesic adaptations: the flat layer pattern is grouped into `[Mamba/attention..., MoE]` schedule units; experts that save their dispatched input (every non-TE expert, `GroupedExperts` included) keep it under FP8; flat patterns keep the pin's checkpoint keys; the pin's behaviour stands where the PR changed it with the overlap off (MTP MoE routers, `_preprocess`); and settings the hybrid schedule gets wrong are refused. Details are in the section below. | The E-044 rung of the Nano-30B pretraining ladder (1.78–1.80×; 5.197 / 5.228 s/iter against 5.388–5.409 s without the overlap). It only takes effect with `comm_overlap.overlap_moe_expert_parallel_comm=true`, which the Nano pretrain quickstart and the control-pretraining V2 E2E stage 1 set, and no other production config does. With the flag off, training computes the same thing with or without the patch, and checkpoints of flat layer patterns, Nemotron-H's included, keep exactly the pin's keys. |
| 0004, carried commit `40e960a2f` | On HybridEP's blocking path a dispatch handle's dispatched-token count lives in pinned host memory that the permute and unpermute kernels read from device code, and PyTorch's host allocator can hand the block out again once the handle is freed while such a kernel is still queued. Under the EP all-to-all overlap the unpermute then ran on a count of 262,145 (at most 65,536 tokens can arrive) and faulted. The patch makes `HybridEPDispatch.forward` hand out the count as a stream-ordered device copy. Details are in the section below. | Every run of the `hybridep` flex dispatcher on its blocking path (dropless, no `moe_expert_rank_capacity_factor`), the Nano campaign's EP-overlap posture (E-044) included. The fault needs a window the EP overlap opens; without the overlap the same read is exposed but no fault was seen. Numerics are unchanged: the same kernels read the same value. |
| 0005, carried commit `3c2da7d91` | HybridModel's output layer and cross-entropy fused over vocabulary chunks (`cross_entropy_loss_fusion: true` with `cross_entropy_fusion_impl: linear`), without the fp32 copies of the logits. Upstream has a linear cross-entropy only on its `dev` branch and only for Blackwell (Megatron-LM #2256, #2739; Hopper kernels in open #3345), so this is our own Triton + cuBLAS implementation under upstream's config name. Two knobs: `cross_entropy_fusion_vocab_chunk_size` (16384) and `cross_entropy_fusion_saved_logit_chunks` (0 recomputes every chunk's logits in the backward, least memory; a value at least the number of chunks keeps them all, no recompute; results are bit-identical either way). Nano-30B, one micro-batch of 8192 tokens, output layer + loss forward and backward on one GH200: 50.9 ms and 6 GiB peak unfused, 44.5 ms and 0.4 GiB recomputing, 34.9 ms and 2.1 GiB keeping every chunk. It builds on the hybrid EP-overlap port (0003), whose `HybridModel._postprocess` it edits. Details are in the section below. | Only runs where the fusion is selected: with the knob off HybridModel builds the plain `ColumnParallelLinear` and computes the unfused loss exactly as before, and parameter names, shapes and checkpoint keys are the same either way. The fused loss needs tensor-parallel size 1: it refuses TP>1, bias, deferred embedding wgrad and CPU offloading when it runs (the logits path still works anywhere, e.g. for conversion), and HybridModel refuses MTP and MuP with it at construction. Tests: `tests/unit_tests/models/mamba/test_chunked_linear_cross_entropy.py`. |
| 0006, carried commit `4c672ad89` | Upstream #6114 (`723db5a72`, 2026-08-20, after the pin's upstream base), cherry-picked. The router's expert-bias token count applied the flattened `[tokens]` padding mask to its `[tokens, experts]` routing map without broadcasting it over the experts, so a training step that passed a mask to a router with expert bias failed (`The size of tensor a (128) must match the size of tensor b (16384)`). The mask now broadcasts over the experts. Details are in the section below. | Every training run that passes a padding mask to a MoE router with expert bias. Megatron-Bridge's packed SFT step passes one and Nemotron-H's routers use expert bias, so every packed SFT run of a Nemotron-H model needs it. Without a mask, or without expert bias, nothing changes. |

## Pin history note (2026-10-01)

The submodule pins `4c672ad89`, fork branch `geodesic/mcore-6cd6ea530-nano-sft`: the previous pin `3c2da7d91` (now
`.dev.commit`) plus 0006, upstream #6114 cherry-picked. At the previous pin a packed SFT run of a MoE model whose
routers use expert bias fails at its first training iteration, because Megatron-Bridge's packed SFT step passes the
model a padding mask. The step also lays the mask out for sequence parallelism itself
(`gpt_step._prepare_packed_padding_mask`, as upstream Megatron-Bridge #5470 does), so no Megatron-LM model change is
carried for it. Upstream Megatron-LM's HybridModel scatters a mask only when it is longer than the stage's hidden states
(`626fe3a10`), so a pin bump past that commit leaves the step's layout alone.

## Pin history note (2026-09-30)

The submodule pins `3c2da7d91`, fork branch `geodesic/mcore-6cd6ea530-nano-perf`: the previous pin `12c20d8f0` (now
`.dev.commit`) plus three commits made with `git am` from this directory's patch files 0003, 0004 and 0005 as
of main `b6d312aa` (0005's author line set to Geodesic Research), in that order: `3e3c83d50`, `40e960a2f`,
`3c2da7d91`. Its tree is byte-identical to the patched tree the full unit suite ran on (pin + 0003 + 0004 +
0005). The campaign's shipped-tree checks ran on a tree that also carried 0002 (CUDA-graph capture only,
which no config enables), so they are not checks of this tree; the pin was run on its own: with main's
code, the production posture trains bit-identically to the previous pin, and the fastest posture without
the chunked cross-entropy bit-identically to the shipped tree (deterministic mode, 30 iterations on one
node).

## Pin history note (2026-07-27)

The submodule now pins `12c20d8f0` (fork branch `geodesic/mcore-6cd6ea530`) on the
**GeodesicResearch/Megatron-LM fork**
(= upstream `6cd6ea530`, 2026-07-22, plus one carried commit making the nvrx
version probe non-fatal — see that commit's message). The fork exists so carried
commits are reachable from a fresh clone; carrying them as un-pushed submodule
commits is how a fix became unrecoverable once before. Patch 0001 was
regenerated for this pin (cosmetic second hunk dropped; upstream still lacks the
normalization). The p2p-send/deallocate fix previously vendored on the INFR-71
branch as patch 0002 is contained in this pin (upstream 260cba71) and needs no
patch here. (That retired number was later reused: today's `0002-*` file is the
unrelated, still-open CUDA-graph fix in the table above — do not read this
paragraph as being about it.)

## 0003: EP all-to-all overlap for the hybrid model (2026-09-29)

**Why it exists.**
- `comm_overlap.overlap_moe_expert_parallel_comm=true` runs the combined-1F1B schedule. The
  forward of one microbatch is interleaved, layer by layer, with the backward of the previous
  one. Each MoE layer's dispatch and combine run on a communication stream while the other
  microbatch's Mamba, attention or expert GEMMs run on the compute stream.
- At the pin, the schedule plan and fine-grained callables exist only for `GPTModel`.
  `HybridModel` (Nemotron-H, `MambaModel`) has no `build_schedule_plan`.
- Upstream PR NVIDIA/Megatron-LM#4798 ("[feat] Hybrid model ep overlapping main") adds them. It
  is open upstream: head `1fdff6677ec5d31a221caae491122958809d769a`, 18 commits on merge-base
  `48a887fe`, by Yan Xu and Pingtian Li.
- Its first part, #4941 (the common schedule-plan base), is already in the pin as upstream
  `ffbe018c8`. So this patch holds the rest of the PR, 3-way merged onto the pin. The merge was
  clean.

It also carries five adaptations, marked `[Geodesic adaptation]` in the code:

- **Flat-pattern grouping.**
  - The problem: upstream schedules one unit per pattern symbol unless the pattern is
    bracketed (`[ME][M*E]...`). Bracketing builds nested stacks, which renames every parameter
    and checkpoint key. With one unit per symbol, half the all-to-alls have no compute to hide
    behind.
  - The change: `HybridStackModelChunkSchedulePlan` groups each `[pre-layers..., MLP/MoE]` run
    of a flat pattern into one unit (`group_flat_pattern`). Nano's 52 layers become 23 units,
    and the model and its checkpoints are untouched.
  - Effect: −4.5% on the 1-node smoke, against −2.4% per-symbol.
  - `MCORE_HYBRID_OVERLAP_AUTOGROUP=0` restores upstream's per-symbol plan; unset or `1` groups,
    and any other value raises (`flat_pattern_autogroup_enabled`).
- **Experts that save their dispatched input keep it under FP8.**
  - The problem: upstream frees the expert node's input whenever FP8 is on, assuming the experts
    kept a quantized copy, which only TE's grouped MLP does. `GroupedExperts`
    (`moe_experts_impl: torch_grouped`) stays BF16 under an FP8 recipe and saves that input for
    its weight gradient. The overlap run crashed in the expert backward with
    `setStorage ... out of bounds for storage of size 0` (job 6932873).
  - The change: `should_free_input` takes a required `experts_quantize_input`, read off each layer
    (`experts_quantize_input`: true only for `TEGroupedMLP`) and passed by both the GPT and the
    hybrid schedule plans; MTP nodes take it from the MoE layer inside the MTP layer. Under
    FP8/FP4 only experts that quantize their input free it; the others follow the BF16 rule.
    GPT models with non-TE experts under FP8 get the same fix. `TEGroupedMLP` layers and every
    layer without FP8/FP4 behave as upstream.
- **Flat patterns keep the pin's checkpoint keys.**
  - The problem: the PR drops `output_layer._extra_state` from every `HybridModel`'s sharded state
    dict, to match `GPTModel`, although its own rule is that only bracketed-group patterns take
    `GPTModel`'s keys. A flat-pattern checkpoint saved that way fails the production exporter,
    which runs on a tree without the patch: `RuntimeError: Missing key in checkpoint state_dict:
    output_layer._extra_state/shard_0_1` (parity job 6935149). Training resumes did not notice
    because they load with `log_all` strictness.
  - The change: the key is dropped only for bracketed-group patterns
    (`HybridStack.transformer_sharded_keys`).
- **The pin's behaviour where the PR changed it with the overlap off.**
  - MoE layers of MTP HybridStacks are built without `is_mtp_layer`, and `router.py` is the pin's.
    With the PR's flag, the MTP routers of a repeated-layer MTP model (`mtp_use_repeated_layer`,
    as Super and Ultra use) divide their aux loss by `mtp_num_layers`, and the PR's tracker-slot
    formula sends every MTP depth to one slot. The PR's `test_aux_loss.py` test for that formula
    is left out with it.
  - `HybridModel._preprocess` returns `sequence_len_offset` as `None`; `HybridStack` builds it for
    flash-decode / local-cudagraph static-batching inference, as at the pin. The PR's version read
    the batch size from `input_ids`, which fails when only `decoder_input` is passed, and
    allocated an unused tensor at every step.
- **Settings the hybrid schedule refuses** (it raises; the E-044 posture uses none of them):
  - the flex dispatcher's `ncclep` backend: the hybrid dispatch and expert nodes detach the probs
    and hold `tokens_per_expert` only for `deepep` and `hybridep`;
  - `fine_grained_activation_offloading`: the hybrid combine node does not offload the `mlp_norm`
    output (the PR called a method the pin lacks);
  - `delay_wgrad_compute`: the hybrid pre-dispatch node schedules the MoE layer's pre-dispatch
    weight gradients only when it has shared experts, so a latent MoE layer without them never
    computes `fc1_latent_proj`'s weight gradient;
  - Megatron-FSDP (`use_megatron_fsdp`): the hybrid schedule units are not reached by its unit
    discovery and reshard hooks. The PR's FSDP wiring is left out, and its FSDP hybrid overlap test
    became a refusal test.

  The first three raise from `check_hybrid_overlap_supported` when the plan is built; Megatron-FSDP
  raises from `FullyShardedDataParallel` when it wraps a model with HybridStacks.

**Load-bearing for.** E-044, the 1.78–1.80× rung of the Nano-30B pretraining ladder
(`docs/investigations/nano30b-pretrain-perf-campaign.md`).
- The rung measured 5.197 / 5.228 s/iter (jobs 6933731 / 6933837) against 5.388–5.409 s without
  the overlap, the paired control 6933820 included: −3.4%.
- Those jobs ran from the frozen snapshot
  `/projects/a5k/public/logs/nano_pretrain_perf_campaign/snapshots/wt-e3dd4e56-epov-full1`, whose
  `3rdparty/Megatron-LM` is pin + 0002 + the first version of this patch.
- This version computes exactly the same training in that posture, so E-044 holds for it.
  Deterministic 1-node smokes of the E-044 posture (11 layers, overlap on, 30 iterations,
  `model.deterministic_mode=true` with `NVTE_ALLOW_NONDETERMINISTIC_ALGO=0` and
  `CUBLAS_WORKSPACE_CONFIG=:4096:8`) give identical full-precision loss and grad norm at every
  iteration: 6935230 on the E-044 tree (W&B `xnhpps1i`) against 6935991 on this patch (W&B `7sesro8b`). None
  of the adaptations touches the E-044 posture's path: its experts are `GroupedExperts`, which the
  first version already kept, and it uses no MTP, FSDP, offloading, `ncclep` or delayed wgrad.

**Where it lives.** Carried commit `3e3c83d50` of the pinned fork branch (pin history note, 2026-09-30); there is
nothing to apply. 0005 builds on it.

**Required alongside it** (it does nothing without the first override):

- `comm_overlap.overlap_moe_expert_parallel_comm=true`. It must go under `comm_overlap:`, not
  `model:`: `CommOverlapConfig.setup` resets the model field.
- `model.mtp_num_layers=null`. The Nano model provider defaults it to `0`, and both MCore and Megatron-Bridge
  accept only `None` or `1` with the overlap on. The first 64-GPU pair, 6933304/6933305, died on
  this assert. Nano has no MTP layers either way.
- `ISAMBARD_CUDA_MAX_CONNECTIONS=32` (sets `CUDA_DEVICE_MAX_CONNECTIONS`). With a single hardware
  queue the two streams serialise, and the overlap was slower than no overlap at all (6933732:
  5.622 s).
- No packed sequences. Megatron-Bridge builds the schedule plan without `packed_seq_params`, so it
  raises when the overlap meets a packed batch (`gpt_step._forward_step_common`).
- None of the refused settings above.
- Settings the E-044 posture already has:
  - `moe` not in `recompute_modules` (`[moe_act]` is fine);
  - `moe_shared_expert_overlap=false`;
  - `cuda_graph_impl=none`;
  - no full recompute.
- The flex/HybridEP and alltoall dispatchers both work. TP=1 and PP=1 were the only measured
  topology; with PP>1 upstream requires virtual pipelining.

**Caveats.**
- **Checkpoints.** Flat patterns, Nemotron-H's included, keep exactly the pin's keys, so
  checkpoints move both ways between trees with and without the patch. Bracketed-group patterns,
  which the pin cannot build, drop `output_layer._extra_state` as `GPTModel` does. Checkpoints
  saved by the first version of this patch lack that key; they still load here under the default
  `log_all` strictness, which reports it missing.
- **Memory.** The overlap adds about 1.1 GB of allocated memory but about 13 GB of reserved
  memory: 78.6 → 91.7 GB at 64 GPUs, and 82.1 → 94.8 GB at 32 GPUs (W&B run maxima, last rank;
  the log's single memory line, written after iteration 1, understates this). Most of the extra
  reserve is allocator fragmentation under the interleaved schedule, not the communication
  stream's separate pool: a 64-GPU allocation history (jobs 6934763/6934764) puts the compute
  pool's occupancy peak at 32.8 GB against 41.1 GB reserved, and the communication pool 3.7 GB
  above its own peak. No allocator setting tested recovers it (`backend:cudaMallocAsync`,
  `garbage_collection_threshold`, classic segments with a split limit, size rounding), nor does
  moving the dispatched tokens to the compute stream. Removing the loss buffers from the top of
  the heap (0005) does lower it.
- **CUDA graphs.** The plan asserts `cuda_graph_impl == "none"`. An experimental extension that
  lifts this (snapshot `wt-e3dd4e56-epov-cg`) measured NULL for `[mamba]` graphs (−0.4%) and
  +9.8% for `[mamba,attn,moe_router]` at 64 GPUs, so it is not vendored.

**Verification** (2026-09-29, on a `git archive` copy of the pin `12c20d8f0`):
- `git apply --check` is clean, as is GNU `patch -p1 --dry-run`; `git mailinfo` parses the
  headers, so the patch is `git am`-able.
- After applying, all 22 touched files are byte-identical to the tree the tests ran on (snapshot
  `overlap-memory-rev0003f`), and pin + 0002 + 0003 equals that tree's whole
  `3rdparty/Megatron-LM`.
- CPU unit tests (container, on pin + 0002 + 0003): 103 passed.
- GPU unit tests (1 node, 4 × GH200, torchrun, EP=4, on the snapshot):
  - the files this patch adds or changes (`test_free_input_policy.py`, `test_hybrid_overlap.py`,
    `test_fsdp_hybrid_overlap.py`, `test_submodule_callables.py`, `models/test_hybrid_model.py`,
    `ssm/test_hybrid_block.py`): 109 passed on each rank (job 6935710). The 11 parametrizations
    that need 8 GPUs stop in `initialize_model_parallel` (`world_size (4) is not divisible by 8`),
    as they do without the patch.
  - the PR's other overlap and hybrid files (`test_schedule_layer_1f1b.py`,
    `test_schedule_chunk_1f1b.py`, `ssm/test_hybrid_layer_allocation.py`): 170 passed on each rank,
    0 failed (job 6935954). The 4 parametrizations that need 8 GPUs were deselected.
  - `test_hybrid_overlap.py` compares the overlap with the eager model bit for bit for flat, flat
    Mamba, bracketed, bracketed Mamba and MTP patterns, with and without shared experts.
- Each fix's test fails on the first version of this patch (job 6935959): the pin-key tests
  (missing `output_layer._extra_state`), the MTP router test, the `_preprocess` test
  (`AttributeError`) and the FSDP refusal (`DID NOT RAISE`). The FP8 test
  (`TestExpertsThatSaveTheirInputUnderFp8`) fails with the PR's free-input rule (job 6935204:
  `The tensor has a non-zero number of elements, but its data is not allocated yet`).
- 1-node Nano smokes: overlap ON vs OFF is bit-identical at iterations 1–2, and the ON-vs-OFF
  loss difference stays inside the OFF-vs-OFF band. The deterministic identity smokes are under
  "Load-bearing for".

## 0004: a HybridEP handle's dispatched-token count in device memory (2026-09-29)

**Dependencies.** None. The patch touches only `megatron/core/transformer/moe/fused_a2a.py`, which
no other patch here touches, so it applies to the bare pin and anywhere in the stack (`git apply
--check` on the pin and on pin + 0002 + 0003 + 0005; GNU `patch -p1 --dry-run` on the pin).

**Why it exists.** Without `num_permuted_tokens`, HybridEP (deep_ep `1.2.1+34152ae`) keeps a
dispatch handle's dispatched-token count in pinned host memory (`executor.cu:107`), and its permute
and unpermute kernels read it from device code when they start (`permute.cu:326`, `:459`). PyTorch's
caching host allocator records no use of the block for those reads, so once the handle is freed it
can hand the block out again while a combine or a backward that reads it is still queued, and that
kernel then runs on a foreign count. Under the EP all-to-all overlap the backward that frees a
handle can run while the comm stream still has such a kernel queued. A 1-node micro-batch-2 smoke
of the EP-overlap posture faulted this way in 2 of 4 runs; a GPU core dump (job 6935302) put a Warp MMU
Fault in `unpermute_kernel<512, bf16, float>` on an expert-output row read, with the kernel's count
at 262,145 where at most 65,536 tokens can arrive. At micro-batch 1 the same kernel reads the same
count; no fault was seen in 8.1 M overlapped layer dispatches there, and a wrong count that stays
inside mapped memory would corrupt the combine silently rather than fault.

**What it adds.** `HybridEPDispatch.forward` replaces the handle's count with
`count.to(device, non_blocking=True)` on the dispatch stream whenever it is host memory. The host
allocator records the copy and holds the pinned source until it, and so the dispatch kernels queued
before it, have run; the count is final before the copy is queued, because the blocking path
synchronizes the stream in metadata preprocessing. With `num_permuted_tokens` the count is a device
tensor already and the handle passes through unchanged. No kernel on the permute path reads the
count on the host (the one `.item()`, `executor.cu:347`, is in the no-permute `dispatch` branch), so
the copy adds no synchronization.

**The second symptom.** In training the fault also surfaced in the next forward dispatch, as
`Trying to create tensor with negative dimension -32` (job 6935265, micro-batch 1) or an illegal
memory access. The blocking path discards the return code of its stream sync, so once a kernel has
faulted the sync returns at once. The pad kernel never runs, and the host sums the stale bytes of
the per-expert-count block. The per-expert counts have no device reader after that sync, so the
same patch removes both symptoms.

**Where it lives.** Carried commit `40e960a2f` of the pinned fork branch; there is nothing to apply.

**Tests** (Bridge, `tests/unit_tests/training/test_hybridep_count_lifetime.py`, one GPU with
deep_ep's HybridEP, world-1 group). The three tests of Megatron's dispatch fail without the patch
(job 6935541: 3 failed, 2 passed) and pass with it (job 6935529: 5 passed); they run in the unit suite:
- a combine queued behind `torch.cuda._sleep` on a side stream, whose handle is freed and whose
  count block is zeroed before the stream runs, must equal an undisturbed combine;
- a blocking dispatch must hand out a device count;
- in a child process, the same sequence with a count past every token, followed by the next
  forward dispatch, must run cleanly. Without the patch the child aborts with
  `cudaErrorIllegalAddress` at the next dispatch's first checked call, as training did.

Two companion tests pin the hazard on deep_ep's raw API (the freed count block is handed out again
and the combine goes wrong) and the host-allocator property the patch relies on (a non-blocking copy
keeps its pinned source until it has run); they hold on any tree.

**Validation.**
- The 1-node micro-batch-2 smoke that faulted in 2 of 4 runs ran 6 of 6 clean with the patch (job
  6935468), with the original's iteration-1 loss.
- A deterministic 11-layer run of the EP-overlap posture with router fusion, gc.freeze and BF16 primary
  weights (no chunked cross-entropy: Megatron-Bridge refuses any cross-entropy fusion in deterministic mode) is
  bit-identical with and without the
  patch over 30 iterations (jobs 6935478 and 6935510, W&B full precision).

## 0005: chunked linear cross-entropy for the hybrid model (2026-09-29)

**Why it exists.** At sequence length 8192 and Nemotron-H's 131,072-token vocabulary, the unfused
loss holds the bf16 logits of a micro-batch (2 GiB) and `vocab_parallel_cross_entropy` casts them to
fp32 (4 GiB). Under the EP overlap's interleaved schedule those buffers sit at the top of the heap and
need contiguous space, which is where most of the overlap's extra reserved memory comes from (0003's
memory caveat). Upstream's linear cross-entropy exists only on its `dev` branch and only for
Blackwell, so this patch implements it for the hybrid model under upstream's config name.

**What it adds.**
- `megatron/core/fusions/fused_chunked_linear_cross_entropy.py`: the op. Forward per vocabulary
  chunk: a logits GEMM, then a Triton kernel for the row max, the sum of exponentials and the target
  logit, combined into the unfused formula `log(sum_exp) - (target - max)`. Backward per chunk: the
  chunk's logits (kept from the forward or recomputed) become `(softmax - onehot) * grad` computed in
  fp32 and rounded once to bf16; the hidden-state gradient accumulates in fp32, and the chunk's
  weight-gradient rows go straight into `main_grad` under gradient-accumulation fusion.
- `LinearCrossEntropyModule` (`megatron/core/transformer/linear_cross_entropy.py`), a
  `ColumnParallelLinear` subclass with no new parameters or buffers, which computes the loss inside
  `forward` so module hooks run (Megatron DDP waits for the layer's parameter all-gather in a forward
  pre-hook). `HybridModel` builds it as its output layer when the fusion is selected.
- Two helpers in `tensor_parallel/layers.py`, `accumulate_wgrad_into_main_grad` and
  `wgrad_after_main_grad_accumulation`, holding the moved, unchanged statements of
  `LinearWithGradAccumulationAndAsyncCommunication.backward`'s `main_grad` accumulation and
  weight-gradient handshake; that backward, the op and `drain_embedding_wgrad_compute` call them.
- Config fields `cross_entropy_fusion_vocab_chunk_size` and
  `cross_entropy_fusion_saved_logit_chunks` (`ModelParallelConfig`), and `'linear'` as a value of
  `cross_entropy_fusion_impl`.

**Load-bearing for.** The E-051 rung of the Nano-30B pretraining campaign, which runs
`model.cross_entropy_loss_fusion=true model.cross_entropy_fusion_impl=linear
model.cross_entropy_fusion_saved_logit_chunks=8` (E-051). On the E-048 base at 64 GPUs it measured
5.002 s/iter against 5.164 s (−3.1% by mean, −2.3% by median; jobs 6935047 / 6935046), with 2.4 GB
less allocated and 4.2 GB less reserved memory and no allocator retries. The Nano midtrain quickstart
(`docs/investigations/nano30b-midtrain-perf-campaign.md`) runs the same three settings at CP=2, where it
replaces the unfused path's 8 GiB fp32 copy of each rank's logits: together with BF16 gradients (its ladder
step 2) the pair measured −6.4% and freed 27.5 GB of training memory, which its selective recompute spends.

**Numerics.** The loss is within 1e-6 of the unfused path (max |d| 9.5e-7 at Nano shapes); the
weight gradient is 99.9% and the hidden-state gradient 97.8% bitwise equal to it, and the fused
hidden-state gradient is as accurate against the exact product of the same bf16 logits gradient
(1.66e-3 against 1.67e-3). The differences are fp32 summation order. The campaign's fastest
configuration, which includes the fusion, stayed inside the as-is loss band over 500 iterations at 64 GPUs
(job 6935341).

**Where it lives.** Carried commit `3c2da7d91` of the pinned fork branch, after 0003's; there is nothing to
apply.

**Refused.** The fused loss needs tensor-parallel size 1 and refuses bias, deferred embedding wgrad
compute, CPU offloading, Megatron-FSDP parameters, and gradient-accumulation fusion without
`main_grad`; HybridModel refuses MTP and MuP with it at construction; `'linear'` through
`compute_language_model_loss` (GPTModel, or logits routed there by hand) raises. Megatron-Bridge's
deterministic mode refuses any cross-entropy fusion (`config.py`), so deterministic identity runs
leave it off.

**Tests** (Bridge, `tests/unit_tests/models/mamba/test_chunked_linear_cross_entropy.py`, 57 tests on
one GPU): the op against the unfused path and
fp32, saved-chunk bit-identity and memory, masked tokens, out-of-vocabulary labels, fp32 inputs,
determinism, `main_grad` fusion in bf16 and fp32 with the DDP handshake, frozen weight or hidden
state, the peak memory, argument errors; the shared `main_grad` helpers and the pipeline-drain
embedding weight gradient against float64; the module as `ColumnParallelLinear` without labels, the
loss at micro-batch 1 and 2, a supplied weight, forward hooks, the refusals (including a spawned
2-rank gloo tensor-parallel group); HybridModel's default output layer, its checkpoint layout and
cross-loading, loss and gradients against the unfused model at micro-batch 1 and 2 with 0 and all
chunks kept, logits without labels, hooks, the MTP and MuP refusals; the Hydra overrides, and the
production configs other than the V2 E2E stage 1, and the baseline benchmark, which keep the unfused loss.

**Verification** (on a `git archive` copy of the pin `12c20d8f0`): after 0002 and 0003, `git apply
--check` and GNU `patch -p1 --dry-run` are clean, and `git mailinfo` parses the headers; 0001–0005
apply in number order.

## 0006: the expert-bias count's broadcast of the padding mask (2026-10-01)

**Why it exists.** Megatron-Bridge's packed SFT collate marks the positions it or the packer padded (each document's
EOS padding and everything after a pack's last document), and its step hands that mask to the model as
`padding_mask`, so that the MoE router leaves those positions out of its expert-bias token counts and its auxiliary
losses (`docs/investigations/nano30b-sft-perf-campaign.md`, E-007). For the expert-bias update the router counts the
tokens it routes to each expert, leaving the masked positions out: `routing_map & ~padding_mask`. The routing map is
`[tokens, experts]` and the router has flattened the mask to `[tokens]`, so the `&` lined the mask up against the
expert dimension. It failed whenever the token and expert counts differ, and would have masked experts rather than
tokens had they been equal. Upstream fixed it in #6114 (`723db5a72`, 2026-08-20), which postdates the pin's upstream
base; the Nano SFT campaign's first masked runs failed at their first backward with
`The size of tensor a (128) must match the size of tensor b (16384)`.

**What it changes.** The mask is unsqueezed to `[tokens, 1]`, so it broadcasts over the experts. The commit is
upstream's, cherry-picked with `-x` (author Chen Cui), and carries upstream's router test.

**Load-bearing for.** Every training run that passes a padding mask to a MoE router with expert bias
(`moe_router_enable_expert_bias`). Megatron-Bridge's packed SFT step passes one, and Nemotron-H's routers use expert
bias, so every packed SFT run of a Nemotron-H model needs it. Without a mask or without expert bias the router runs
exactly as before. A pin at or after upstream `723db5a72` contains the fix, and this commit goes with such a bump.

**Where it lives.** Carried commit `4c672ad89` of the pinned fork branch, after 0005; there is nothing to apply.

**Tests.** Bridge, `tests/unit_tests/models/test_moe_router_padding_mask.py`, one GPU: a one-layer MoE HybridModel
with Nemotron-H's router settings runs a training forward and backward on a batch padded inside a document and at a
row's tail, with and without full recompute (which routes the tokens a second time, in the backward) and with and
without the fused router kernels. Each expert's count must be its routings of real tokens, and the counts must total
the real tokens times top-k. All four cases fail at the previous pin with the training runs' error and pass with the
commit. Upstream's own test is
`tests/unit_tests/transformer/moe/test_routers.py::TestTop2Router::test_expert_bias_token_counts_with_padding_mask`
in the submodule.
