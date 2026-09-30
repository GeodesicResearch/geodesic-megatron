# Local Megatron-LM patches

Patches carried against the pinned upstream Megatron-LM commit. None is applied to the submodule: a run that
needs one applies it to a copy of the checkout (each section's "How to apply"). Apply with:

```bash
git -C 3rdparty/Megatron-LM am ../patches/megatron-lm/<patch>
```

| Patch | Why it exists | Load-bearing for |
|---|---|---|
| `0001-fix-moe-normalize-allgather-dispatcher-output-by-EP-.patch` | Geodesic fix: normalize allgather-dispatcher output by EP size. Was previously a local-only submodule commit (`2034d4500`) that no remote contained — every fresh clone silently failed to fetch the pin and checked out a different mcore (caught by the INFR-68 fresh-install certification). The submodule now pins the patch's reachable upstream parent (`3758b54b2`, the TE-2.14 bump) and the fix lives here instead. | The `allgather` MoE token dispatcher ONLY. No shipped config or recipe uses it (all use `alltoall`), so the running behavior of every committed config is identical with or without it. Apply before using `moe_token_dispatcher_type: allgather`. |
| `0002-fix-cuda-graph-zeros_like-0dim-tensor.patch` | Upstream `zeros_like` on a 0-dim tensor breaks CUDA-graph capture (`cuda_graphs.py:181` unpacks `*self.shape` to nothing). Still open upstream at the current pin. | CUDA graphs only. No shipped config enables them, so every committed config runs identically with or without it. Apply before enabling CUDA graphs. |
| `0003-feat-hybrid-port-upstream-4798-hybrid-EP-A2A-overlap.patch` | Upstream supports the EP all-to-all / compute overlap (`overlap_moe_expert_parallel_comm`, the combined-1F1B schedule) for `GPTModel` only. This patch is the port of upstream PR #4798 (open; head `1fdff667`), which adds it for the hybrid model, minus #4941, which the pin already contains. Geodesic adaptations: the flat layer pattern is grouped into `[Mamba/attention..., MoE]` schedule units; experts that save their dispatched input (every non-TE expert, `GroupedExperts` included) keep it under FP8; flat patterns keep the pin's checkpoint keys; the pin's behaviour stands where the PR changed it with the overlap off (MTP MoE routers, `_preprocess`); and settings the hybrid schedule gets wrong are refused. Details are in the section below. | The E-044 rung of the Nano-30B pretraining ladder (1.78–1.80×; 5.197 / 5.228 s/iter against 5.388–5.409 s without the overlap). It only takes effect with `comm_overlap.overlap_moe_expert_parallel_comm=true`, which no shipped config sets. With the flag off, training computes the same thing with or without the patch, and checkpoints of flat layer patterns, Nemotron-H's included, keep exactly the pin's keys. |

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

**How to apply.**
- In a campaign snapshot (a copy of the checkout with plain files; how the probes were run),
  apply it to the snapshot, never to the live submodule:

  ```bash
  cd <snapshot>/3rdparty/Megatron-LM && git apply ../patches/megatron-lm/0003-feat-hybrid-port-upstream-4798-hybrid-EP-A2A-overlap.patch
  ```

- In a checkout, use the same convention as the other patches:

  ```bash
  git -C 3rdparty/Megatron-LM am ../patches/megatron-lm/0003-feat-hybrid-port-upstream-4798-hybrid-EP-A2A-overlap.patch
  ```

  Do not commit the resulting gitlink unless that commit is first pushed to the
  GeodesicResearch fork (see the pin history note).
- It touches none of the files that 0001 or 0002 touch, so the three apply in any order.

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
  the heap (chunked cross-entropy) does lower it.
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
