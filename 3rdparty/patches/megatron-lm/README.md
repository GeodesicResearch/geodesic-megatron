# Local Megatron-LM patches

Patches carried against the pinned upstream Megatron-LM commit. Apply with:

```bash
git -C 3rdparty/Megatron-LM am ../patches/megatron-lm/<patch>
```

| Patch | Why it exists | Load-bearing for |
|---|---|---|
| `0001-fix-moe-normalize-allgather-dispatcher-output-by-EP-.patch` | Geodesic fix: normalize allgather-dispatcher output by EP size. Was previously a local-only submodule commit (`2034d4500`) that no remote contained — every fresh clone silently failed to fetch the pin and checked out a different mcore (caught by the INFR-68 fresh-install certification). The submodule now pins the patch's reachable upstream parent (`3758b54b2`, the TE-2.14 bump) and the fix lives here instead. | The `allgather` MoE token dispatcher ONLY. No shipped config or recipe uses it (all use `alltoall`), so the running behavior of every committed config is identical with or without it. Apply before using `moe_token_dispatcher_type: allgather`. |
| `0002-fix-cuda-graph-zeros_like-0dim-tensor.patch` | Upstream `zeros_like` on a 0-dim tensor breaks CUDA-graph capture. Carried in with the mcore 0.19 pin bump (`de1f00e0`). | CUDA graphs only. No shipped config enables them, so every committed config runs identically with or without it. |
| `0003-fix-wait-for-async-p2p-send-before-deallocating-output-tensor.patch` | The `overlap_p2p_comm` NaN: un-batched `isend`/`irecv` drops the device-wide sync that made Nemotron-H's `deallocate_pipeline_outputs=True` safe, so the send buffer is pseudo-freed once the send is merely *issued* — deterministic NaN at iteration 2 in three independent arms. Recorded here because it is the minimal reproduction of the race; the equivalent fix landed upstream three weeks after the pre-0.19 pin and **is already contained in the current 0.19 pin**. | Nothing today — it is documentation of a closed bug, NOT a patch to apply. With the current pin the overlap arm runs correctly and simply measures slower (31.45 vs 27.50 s/iter), so `overlap_p2p_comm` stays off for performance, not correctness. Applying this on top of the current pin would be a no-op at best. |

**Nothing here is applied automatically** — no script in the repo reads this directory. Each
row states what its patch is load-bearing for; today no shipped config depends on any of them.

## Pin history note (2026-07-27)

The submodule now pins `fa774820` on the **GeodesicResearch/Megatron-LM fork**
(= upstream `6cd6ea530`, 2026-07-22, plus one carried commit making the nvrx
version probe non-fatal — see that commit's message). The fork exists so carried
commits are reachable from a fresh clone; carrying them as un-pushed submodule
commits is how a fix became unrecoverable once before. Patch 0001 was
regenerated for this pin (cosmetic second hunk dropped; upstream still lacks the
normalization). The p2p-send/deallocate fix — vendored on the INFR-71 branch as
patch 0002 at the time, and renumbered to **0003** here when the CUDA-graph patch
took the 0002 slot — is contained in this pin (upstream 260cba71) and needs no
patch applied. That renumbering is why this paragraph must not be read as saying
anything about today's 0002, which is the CUDA-graph fix and is still open upstream.
