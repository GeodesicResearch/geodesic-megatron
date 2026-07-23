# Local Megatron-LM patches

Patches carried against the pinned upstream Megatron-LM commit. Apply with:

```bash
git -C 3rdparty/Megatron-LM am ../patches/megatron-lm/<patch>
```

| Patch | Why it exists | Load-bearing for |
|---|---|---|
| `0001-fix-moe-normalize-allgather-dispatcher-output-by-EP-.patch` | Geodesic fix: normalize allgather-dispatcher output by EP size. Was previously a local-only submodule commit (`2034d4500`) that no remote contained — every fresh clone silently failed to fetch the pin and checked out a different mcore (caught by the INFR-68 fresh-install certification). The submodule now pins the patch's reachable upstream parent (`3758b54b2`, the TE-2.14 bump) and the fix lives here instead. | The `allgather` MoE token dispatcher ONLY. No shipped config or recipe uses it (all use `alltoall`), so the running behavior of every committed config is identical with or without it. Apply before using `moe_token_dispatcher_type: allgather`. |
