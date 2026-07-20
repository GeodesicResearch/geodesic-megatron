# Local submodule patches

Patches applied to pinned submodules that cannot be upstreamed as a pin bump
(e.g. `3rdparty/Megatron-LM` tracks `NVIDIA/Megatron-LM` directly, with no
Geodesic fork to commit into). Kept here so the change is durable, reviewable,
and re-applicable after a submodule reset.

## `megatron-lm-ddp-bucket-size.patch`

Raises the DDP gradient-bucket default from 40 MB to 500 MB in
`megatron/core/distributed/distributed_data_parallel.py`. On the Nemotron-H
MoE models this collapses the expert-gradient bucket count (~86 buckets → ~7),
which otherwise serialize many small NCCL calls and starve bandwidth on the
Slingshot/CXI fabric.

Apply after `git submodule update --init`:

```bash
git -C 3rdparty/Megatron-LM apply ../../patches/megatron-lm-ddp-bucket-size.patch
```

**Follow-up:** the durable fix is a Geodesic fork of Megatron-LM carrying this
commit, with the submodule URL + pin repointed at it. Until then this patch is
applied manually (and lives in the working tree of the current checkout).
