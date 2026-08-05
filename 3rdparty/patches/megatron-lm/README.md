# Local Megatron-LM patches

Patches carried against the pinned upstream Megatron-LM commit. Apply with:

```bash
git -C 3rdparty/Megatron-LM am ../patches/megatron-lm/<patch>
```

| Patch | Why it exists | Load-bearing for |
|---|---|---|
| `0001-fix-moe-normalize-allgather-dispatcher-output-by-EP-.patch` | Geodesic fix: normalize allgather-dispatcher output by EP size. Was previously a local-only submodule commit (`2034d4500`) that no remote contained — every fresh clone silently failed to fetch the pin and checked out a different mcore (caught by the INFR-68 fresh-install certification). The submodule now pins the patch's reachable upstream parent (`3758b54b2`, the TE-2.14 bump) and the fix lives here instead. | The `allgather` MoE token dispatcher ONLY. No shipped config or recipe uses it (all use `alltoall`), so the running behavior of every committed config is identical with or without it. Apply before using `moe_token_dispatcher_type: allgather`. |
| `0002-fix-cuda-graph-zeros_like-0dim-tensor.patch` | Upstream `zeros_like` on a 0-dim tensor breaks CUDA-graph capture (`cuda_graphs.py:181` unpacks `*self.shape` to nothing). Still open upstream at the current pin. | CUDA graphs only. No shipped config enables them, so every committed config runs identically with or without it. Apply before enabling CUDA graphs. |

## Pin history note (2026-07-27)

The submodule now pins `fa774820` on the **GeodesicResearch/Megatron-LM fork**
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
