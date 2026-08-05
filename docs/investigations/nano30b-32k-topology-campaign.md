# Nemotron-3 Nano 30B-A3B at seq 32768 — topology campaign

**Status:** complete. 34 arms, 2026-08-04. Outcome shipped as
`configs/quickstart/nemotron_nano_quickstart_sft_32k.yaml`.
**Allocation:** 5845745, 32 nodes / 128 GPUs, single allocation throughout, so every number
below is placement-comparable with every other.

**Two world sizes appear below and they must not be mixed.** The campaign was run at **128
GPUs**, where the arms are placement-matched against each other. The quickstart ships at the
repo-standard **64 GPUs / 16 nodes**, and was **re-measured** there rather than extrapolated.
Unless a number is explicitly labelled 64-GPU, it is a 128-GPU measurement.

## The question

Find the optimal Nano quickstart at the same 32K context the Super-120B benchmark uses,
maximising data parallelism and minimising iteration time. The requested shape was
**TP=1, CP=1, EP=2/4, PP as low as possible** — ideally one full DP replica per node, so the
only traffic crossing Slingshot is the DP collectives.

That shape looked available on Nano and not on Super because 93% of Nano's 31.6B parameters
are routed experts, so EP=4 alone shards the model enough to fit: 9,546,831,296 params per
rank (measured), i.e. ~19.1 GB weights + ~19.1 GB grads + ~3 GB sharded optimizer = **~41 GB
static**, leaving ~54 GB for activations on a 95 GB GH200.

## Result

The shipped posture is TP=1 · **CP=2** · EP=4 · PP=1 · ETP=1.

| | GBS | ms/sample | s/iter | peak |
|---|---:|---:|---:|---:|
| **shipped, 64 GPUs / 16 nodes** | 256 | **71.74** | **18.365** | **91.5 GB** of 95 |
| campaign champion, 128 GPUs | 256 | 36.99 | 9.470 | 90.5 GB |
| campaign start (`k1`), 128 GPUs | 128 | 43.50 | 5.568 | 90.6 GB |

**Scaling to the shipped size is better than linear.** Perfect halving of the 128-GPU champion
would predict 18.940 s/iter; the measurement is 18.365, **3.0% better**. At 64 GPUs, GBS 256
gives 8 microbatches per replica instead of 4, which gives the DP reduce-scatter more backward
compute to hide behind. Peak rose 90.5 → 91.5 GB because halving the world halves DP (64 → 32)
and so halves the distributed optimizer's sharding — it fits, with ~3.5 GB of margin, and it
was not obvious in advance that it would.

For scale, the Super-120B champion at the same 32K context is 122.0 ms/sample at 128 GPUs —
Nano is **3.3× faster per sample** (122.0 / 36.99, both at 128 GPUs).

## The two walls, and why the first four arms OOM'd

### Wall A — activation depth at PP=1

Wave 1 assumed PP=1 would make activations cheap: only one microbatch is in flight. That is
true and irrelevant. **PP=1 also puts all 52 layers on one rank**, and the quantity that sets
1F1B-style residency is layers × microbatches-in-flight:

| | layers/rank | µb in flight | layer-µb |
|---|---:|---:|---:|
| Super-120B, PP=8 | 11 | 8 | 88 |
| Nano-30B, PP=1 | 52 | 1 | **52** |

Same order of magnitude — not the reduction the wave-1 arithmetic assumed. At 32768 tokens
per rank a single saved `[seq, hidden]` bf16 tensor is `32768 × 2688 × 2` = 176 MB, so a
handful per layer across 52 layers is tens of GB. Arm `n1` (selective `core_attn` only) died
at 91.4 GB allocated; arm `m2` (selective `core_attn,moe,shared_experts`) died inside the
expert GEMM asking for ~1 GiB.

Wall A is cleared by full recompute.

### Wall B — the cross-entropy logits, which recompute cannot touch

With the layers recomputed, the arms got as far as the loss and died there:

| arm | CP | PP | died in | asked for |
|---|---:|---:|---|---:|
| `m1` full recompute | 1 | 1 | `fused_cross_entropy.calculate_predicted_logits` | **16.00 GiB** |
| `m3` selective | 2 | 1 | the same function | **8.00 GiB** |
| `k3` selective+shared | 2 | 1 | the same function | **8.00 GiB** |

`32768 × 131072 × 4 B` = exactly 16.00 GiB, halving precisely with CP. This is a **live**
tensor in the loss, not a saved activation, so no recompute setting reaches it. It shrinks
only by sharding the sequence (CP) or the vocab (TP).

**This does NOT make CP=1 impossible, which an earlier revision of this document asserted.**
It makes CP=1 impossible *at PP=1*: there the arm missed by 12.31 GiB. At PP=2 it missed by
only 1.25 GiB and cleared with optimizer offload. See the CP=1 row in the results table —
CP=1 is reachable, and the package loses by 9.6%.

**But that 9.6% is not the cost of CP=1 — it is the cost of the PP=2 it needs.** The arm
changes three knobs at once (CP 2→1, PP 1→2, optimizer offload on), so it is not comparable
to the single-knob rows in the table above. Decompose it against PP=2's own +18.3%: at
matched PP=2, going CP=2 → CP=1 is **~7.4% FASTER** (109.6 / 118.3). CP=1 is not slow; it is
simply unreachable without paying for PP=2, and PP=2 is what costs. An earlier revision of
this document and of the shipped config asserted the opposite mechanism — "CP=1 doubles DP
width and cross-node DP is the expensive axis" — which the sign of that decomposition
refutes. The ship decision is unchanged; only the explanation was wrong.

Recording the correction explicitly, because the wrong version of this claim propagated: an
OOM is evidence about *one configuration*, and two of this campaign's early verdicts
("PP>1 does not fit", "CP=1 is impossible") came from reading a single OOM as a general law.
Both were wrong. Every OOM below now records the **size** it asked for, not merely that it
happened, because the size is what distinguishes "missed by 12 GiB" from "missed by 1 GiB".

**Not an escape hatch:** mcore exposes `cross_entropy_fusion_impl: 'te'`, whose TE kernel
would likely cut the peak at CP=1. Its own config validation warns the option "has known
stability issues" and that "Megatron-LM training args validation rejects this combination by
default", so it is not a posture to ship and was not measured.

## The one-line finding

**The exposed cost is the node-local expert all-to-all, not cross-node DP gradient traffic.**

This is the reverse of the theory the campaign was built on, and the trace is what overturned
it (torch profiler, iteration 26, ranks 0 and 4 agreeing to ~2%, at 128 GPUs):

```
9.2556 s  =  6.2984 compute (68.0%)  +  2.7064 exposed comm (29.2%)  +  0.2508 idle (2.7%)

exposed comm    2.7064 s
  of which sendrecv (the EXPERT all-to-all)         2.1097 s   <- 78% of it, 98% unhidden
  everything else (DP reduce-scatter + all-gather)   ~0.60 s
```

Every lever in waves 1–5 was aimed at cross-node DP traffic. The trace shows that traffic is
largely **already overlapped** behind backward compute, and what is actually exposed is the
node-local expert all-to-all — 2,580 SendRecv calls, i.e. latency-bound rather than
bandwidth-bound.

Two things follow, and both are borne out by the arms:

- It explains the one lever that ever worked. 500 MB buckets paid precisely **because** they
  moved DP traffic from exposed to overlapped — the mechanism, not a coincidence.
- **Only 2.7% of the iteration is idle**, so there is no pipeline bubble to reclaim. That is
  why every parallelism change loses, and it is why the campaign stopped adding topology arms.

## Closed with measurements — do not re-run these

All at the shipped posture unless noted; deltas are ms/sample against the matched control.

| lever | effect | why |
|---|---:|---|
| `ddp.bucket_size` 64 MB → 500 MB | **−10.7%** | moves DP traffic from exposed to overlapped — **shipped** |
| GBS 128 → 256 | **−4.7%** per sample | amortises DP comm — **shipped** (256 is the batch cap) |
| TP=2 | +48.9% | MoE forces `sequence_parallel`; kernels fall below their efficiency knee |
| PP=2 | +18.3% | but peak 60.2 GB — a MEMORY lever, not a speed one |
| PP=4 | +30.1% | peak 45.7 GB |
| EP=8 | +77.2% | cross-node all-to-all; re-confirms **TP × EP ≤ 4** |
| CP=1 **package** (PP=2 + offload + CP=1) | +9.6% | fits — but this is a three-knob delta, see below; not a CP=1 cost |
| `reduce_scatter_with_fp32_accumulation` | +63.4% | swaps ring for all-to-all; wrong shape on CXI |
| `cp_comm_type=a2a` | +17.0% | CP comm |
| precision-aware optimizer (bf16 moments) | neutral | +1.5 GB peak, no speed change |
| `check_for_nan_in_grad=false` | −0.4% | inside the 0.23% noise floor ×2; keep the check |
| `recompute_num_layers` 1 → 4 | null | 0.0 GB moved |
| `recompute_method=block` | −0.6% | OOMs past ~2 un-recomputed layers |
| `manual_gc` (interval 10) | +0.6% | spread 10.9%; null |
| GBS 128 → 512 | −10.1% per sample | **out of scope** — exceeds the 256-sequence cap |
| `cp_comm_type=all_gather` | rejected | `padding_causal mask type is not supported` |
| `fp16_lm_cross_entropy` | **void** | dead code in this mcore version |
| selective recompute at CP=2 | OOM | twice, at exactly 8.00 GiB |
| EP=2 | OOM | 54.71 GiB — expert weights double per rank |

**Every delta above is against a bucket-matched control**, which matters because the two
shipped levers compose. The GBS row is `v1` → `x0` (both 500 MB buckets): 38.83 → 36.99, i.e.
−4.7%. The tempting comparison `k1` → `x0` gives −15.0% and is **wrong** — it changes bucket
size and batch size at once, so it double-counts the −10.7% on the row above.

**Measured run-to-run noise floor: 0.23%** (`w1`, a byte-identical `k1` repeat: 5.568 vs
5.581), independently confirmed at 0.02% by a VOID arm whose treatment silently failed to
engage and therefore re-ran this exact posture. Treat anything under ~0.3% as
indistinguishable.

**CP=4 has no bucket-matched measurement.** `k4` (CP=4, GBS 256) ran at a 40 MB bucket against
the champion's 500 MB, so its 37.07 is not comparable to `x0`'s 36.99 — the numbers are close,
but the comparison is confounded by the campaign's single largest lever. What IS usable from
`k4` is its peak: 89.0 GB vs 90.5, i.e. **1.5 GB cheaper**, which is why CP=4 is recorded as
the first memory lever to reach for and not as a speed result. (`36.98 vs 36.99` is the
`x0b`/`x0` **repeat pair** — the noise floor — and is not a CP comparison; an earlier revision
of the shipped config's header misattributed it as one.)

### Still open: `overlap_param_gather` is worth 23.7%

Wave 5 produced a clean 2×2 (s/iter, GBS 128, 128 GPUs):

| | buckets 64 MB | buckets 500 MB |
|---|---:|---:|
| param_gather **True** | 5.568 | **4.971** |
| param_gather **False** | 6.680 | 6.149 |

The two levers are independent — buckets buy 8–11% either way, `param_gather=False` costs
20–24% either way. A standing note says to set it **false** on Nemotron-H at DP>1
("misbehaves … likely interaction with the SSM/Mamba parameter layout"), and the Super-120B
quickstart does. But that note has **no evidence trail in this repo** — no linked
investigation, commit or issue — while upstream's own performance guide recommends `true`.
Three 40-iteration arms have run with `true` (0 NaN, 0 skipped, losses matching the `false`
control to 2e-6), which is suggestive but far too short to clear a correctness rule.
A 300-iteration **paired loss soak** settles it; it was built and has not completed. Until it
does, the shipped config runs the bridge default (`true`) and records the question rather than
banking or conceding the 23.7% by silence.

### Never measured: recompute-off

Full recompute is roughly 17% of the iteration by the trace, and is the largest lever the
campaign never scored. The arm that would have tested it (PP=2 with recompute off, which is
the only shape with the memory headroom to try) died at iteration 0 in both replicates with
`NCCL Error 1: unhandled cuda error`. It is **untested, not null** — no verdict either way.

## Measured arms (128 GPUs)

Scored on the **settled window (iterations ≥ 23)**, not the campaign's usual 10–30 window —
see the scoring note below. Comparisons are in **ms/sample**, the only metric comparable
across global batch sizes; s/iter is shown for reference and must NOT be compared across
different GBS.

All arms are TP1 · EP4 · PP1 with full uniform recompute unless noted. `pg` = param_gather.

| arm | what it varies | GBS | bucket | ms/sample | s/iter | spread | peak | outcome |
|---|---|---:|---:|---:|---:|---:|---:|---|
| **v8** | GBS 512 | 512 | 500M | **34.92** | 17.880 | 4.4% | 90.5 | best per sample — over the batch cap |
| x0b | x0 repeat (after the void re-run) | 256 | 500M | 36.98 | 9.467 | — | 90.5 | 0.03% from x0 — the champion's own repeat |
| **x0** | **GBS 256** | 256 | 500M | **36.99** | **9.470** | 9.4% | 90.5 | **champion; the shipped posture** |
| k4 | CP4 · selective | 256 | 40M | 37.07 | 9.489 | 14.5% | 89.0 | bucket-unmatched — see above; usable for peak only |
| **v1** | **bucket 500 MB** | 128 | 500M | **38.83** | **4.971** | 2.4% | 90.5 | **−10.7%, the key lever** |
| v5 | `fp16_lm_cross_entropy` | 128 | 500M | 38.82 | 4.969 | 2.4% | 90.5 | VOID — flag is dead code |
| v7 | `manual_gc` | 128 | 500M | 39.05 | 4.999 | 10.9% | 90.5 | null |
| w2 | GBS 256 | 256 | 64M | 39.45 | 10.100 | 9.5% | 90.5 | |
| v4 | `block` recompute, 50/52 | 128 | 64M | 43.23 | 5.534 | 2.1% | 91.6 | −0.6% |
| k1 | the first config that fit | 128 | 64M | 43.50 | 5.568 | 1.7% | 90.6 | campaign start |
| w3 | `recompute_num_layers=4` | 128 | 64M | 43.51 | 5.570 | 2.4% | 90.6 | null, 0.0 GB freed |
| w1 | k1 repeat | 128 | 64M | 43.60 | 5.581 | 2.6% | 90.5 | **noise floor 0.23%** |
| x2 | `cp_comm_type=a2a` | 128 | 500M | 45.43 | 5.815 | 4.5% | 90.8 | +17.0% |
| k2 | CP4 · selective | 128 | 40M | 47.46 | 6.074 | 25.5% | 89.5 | |
| v3 | 500M + **pg false** | 128 | 500M | 48.04 | 6.149 | 1.8% | 90.5 | +23.7% |
| v2 | **pg false** | 128 | 64M | 52.19 | 6.680 | 2.2% | 90.5 | +20.0% (vs k1, the 64 MB control) |

Arms that did not produce a number, and why — each one is a datapoint about the memory wall.
**The size asked for is the datapoint**, not the OOM itself:

| arm | what it varied | outcome |
|---|---|---|
| n1 | CP1 · selective(core_attn) | OOM at 91.4 GB |
| m1 | CP1 · full recompute | OOM — **exactly 16.00 GiB**, in the loss (missed by 12.31 GiB) |
| m2 | CP1 · selective(3 modules) | OOM in the expert GEMM |
| m3 | CP2 · selective(core_attn,moe) | OOM — **exactly 8.00 GiB**, in the loss |
| k3 | CP2 · selective(+shared_experts) | OOM — exactly 8.00 GiB, in the loss |
| m4 | PP2 · selective | OOM at exactly 16.00 GiB — i.e. the **CP=1 logits wall, not a PP=2 verdict**; PP=2 was scored later and fits at 60.2 GB |
| w4 | EP=2 | OOM — 54.71 GiB (expert weights double per rank) |
| v6 | `block` recompute, 44/52 | OOM (predicted: 8 un-recomputed layers on a 90.5 GB peak) |
| x3 | `cp_comm_type=all_gather` | REJECTED — `padding_causal mask type is not supported` |
| x4 | `micro_batch_size=2` at CP4 | REJECTED — `Micro batch size should be 1 when training with packed sequence` |

### Two void arms, and the hazard that produced them

`x0b` and `x1` failed with `ValueError: Unknown moe_experts_impl 'torch_grouped'` — from a
provider whose support for that value was sitting in the working tree. Cause: a concurrent
`git commit`. The commit gate runs pre-commit, which `git stash`es the working tree for the
~2 minutes its in-container pytest hook takes, and `GEODESIC_REPO_DIR` points at that same
tree — so any job that IMPORTS during the window silently gets HEAD's code, after which the
stash is restored leaving no trace. Both arms are **void, not null**, and were re-run.

Fixed by snapshotting the source to `/projects/a5k/public/containers/nano32k_repo_snapshot`
and pointing later waves at it, so commits in the live tree cannot reach a running campaign.

Two findings worth stating separately from the table:

- **k1 beats k2 while paying a full extra forward pass.** Going CP=2 → CP=4 halves DP and
  doubles the per-layer CP communication (ring attention plus the Mamba CP all-to-alls); that
  costs more than the cheaper recompute saves.
- **k3 makes k1's recompute load-bearing.** k3 is k1 with recompute reduced, and it OOMs at
  the logits. So full recompute is not a tax we are choosing to pay — it is what frees the
  room the logits need. Arms that vary recompute at CP=2 are therefore not available; the
  recompute question survives only at PP=2, where it went untested (above).

### Caveats

- **k2's window is noisy** (25.5% spread, no clean settling), so its mean is the softer of
  the two numbers.
- The reported ~152.7 TFLOP/s/GPU is the training loop's own figure, which uses a dense
  transformer approximation. `scripts/nemotronh_flops_estimator.py` is the correct instrument
  for this architecture and has not been run against this config.
- The trace decomposition was captured at 128 GPUs. The proportions drove every decision here
  and are what should be quoted; the absolute seconds are the 128-GPU iteration.

## Scoring note: the 10–30 window is wrong for Nano

The campaign's standing window (mean of iterations 10–30) straddles two regimes on this
model. k1's per-iteration times:

```
1:69.0  2:12.0 … 14:11.4   15:5.6 16:5.6 17:5.9 18:5.6   19:10.7 20:11.3 21:5.6 22:11.6
23:5.8 24:5.5 25:5.5 26:5.6 27:5.5 28:5.6 29:5.6 30:5.6 31:5.6 32:5.5
```

Scoring 10–30 gives 7.619 s/iter with a 109% spread — a number that describes no actual
operating point. Scoring from iteration 23 gives 5.568 with 1.7% spread. k2 shows no such
ramp at all, so this is not generic cache warming; it is most likely compile warm-up specific
to the full-recompute path. Later waves run 40 iterations and score from 23, giving an
18-sample settled window.

## Validated as written

The shipped config was run end-to-end with **zero Hydra overrides** — the documented `LAUNCH:`
command exactly as a user would type it: 32 iterations, 18.042 s/iter, 91.5 GB, no errors.

**18.365 is the headline, not 18.042.** The two differ by 1.8%, well outside the 0.23%
noise floor, because they are not the same measurement: 18.365 is the scored arm (40
iterations, mean from iteration 23), while this validation ran 32 iterations and shared its
allocation with the concurrent Super validation. Quote 18.365; this number exists to show the
config starts and runs as written, not to compete with it.

That had never been done during the campaign, and it is how a **startup crash in the config
survived 34 arms**: with `checkpoint.save: null` and no `logger.wandb_save_dir`,
`state.py:197` evaluates `os.path.join(None, "wandb")` and dies with a `TypeError` before
iteration 1. Every arm passed that flag as an override, so the config-as-written was never
once exercised. An arm is not a validation of the file it was derived from.

## Infrastructure defect found and fixed en route

The first arm never reached iteration 1:

```
TypeError: get_batch_on_this_cp_rank() missing 1 required positional argument: 'is_hybrid_cp'
```

The mcore 0.19 pin added a required `is_hybrid_cp` positional; the bridge's call sites still
passed the pre-0.19 two-argument form. **Four call sites in `src/` were affected**, not one:

| call site | reached by |
|---|---|
| `training/gpt_step.py` | every CP=1 config — both Nano quickstarts and `nemotron_ultra_quickstart_sft.yaml` |
| `models/qwen_vl/qwen3_vl_step.py` | Qwen-VL training |
| `training/llava_step.py` | LLaVA training |
| `utils/common_utils.py` (BSHD branch) | Gemma3-VL and Ministral3, at CP≥2 |

Only the Super configs escaped, because CP=4 + packed data takes a different branch
(`_partition_packed_batch_for_cp`) — which is exactly the posture this campaign's sibling was
benchmarking, and why the defect reached main unnoticed. All four are fixed;
`common_utils.slice_batch_for_context_parallel` gained a required `is_hybrid_cp` parameter
threaded from its two callers rather than a hardcoded default.

The existing suite could not catch it: `test_gpt_step_packed_all_stages.py` patches
`get_batch_on_this_cp_rank` with a permissive `*args, **kwargs` stand-in that any signature
satisfies. Three tests now close it, in three files:

- `tests/unit_tests/training/test_gpt_step_cp_dispatch.py` calls the **real** mcore function
  through a single-process gloo group — affordable on CPU because both of its balancers are
  no-ops below CP=2.
- `tests/unit_tests/test_cp_rank_call_site_conformance.py` parses every call site in `src/`
  and binds its arguments against the live `inspect.signature`. That generalises the defect
  class: a behavioural test could not cheaply reach three of the four sites, and this one
  fails on the *next* upstream signature change rather than at somebody's first CP run.
- `tests/unit_tests/utils/test_slice_batch_for_cp.py` exercises the util with `is_hybrid_cp`
  both True and False. Without the True case every test in that file passed `False`, so a
  hardcoded `False` inside the util would have stayed green.
