"""Attribute host-side activity behind GPU stalls in a torch-profiler chrome trace.

The earlier passes (gap_dist / gap_attrib) establish the *shape* of the idle (many sub-ms
launch gaps = starvation). This tool names the code: it needs a trace captured with
``with_stack=True`` and reports

  1. kernel-family census        -> which modules issue the launches (the launch storm)
  2. sync-call attribution       -> which megatron/bridge frame owns each stream/device sync
  3. pageable-HtoD attribution   -> what feeds the GPU from pageable host memory
  4. per-thread CUDA-API load    -> whether a launcher thread is API-bound (vs framework-bound)
  5. python/op self-time         -> DIRECTIONAL ONLY: torch's re-entrant pybind built-in
                                    frames do not nest strictly, so their "self" time
                                    over-counts wildly; read the well-formed python frames
                                    and cpu_op rows, ignore the pybind built-ins.

Findings that came out of this tool (2026-07-29, Super-120B GBS=64): 66% of all kernel
launches were per-expert GEMMs from ragged dropless grouped GEMM; 1.93 s/iter of
"cudaStreamSynchronize" was Megatron timer barriers at timing_log_level 2; the dropless
dtoh token-count sync waited ~0 (host was the laggard); 651 pageable HtoD copies/iter came
from hybrid_optimizer's param copy-back hook. See the investigation doc §9.

Usage: python host_attrib.py <trace.json[.gz]> <label>
Run on an idle allocation: parsing a ~200 MB gz trace is CPU- and RAM-heavy.
"""

import collections
import gzip
import json
import re
import sys


path, label = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else "trace")
op = gzip.open if path.endswith(".gz") else open
with op(path, "rt") as f:
    ev = json.load(f)["traceEvents"]

GPU_CATS = {"kernel", "gpu_memcpy", "gpu_memset"}
SYNC_NAMES = {"cudaStreamSynchronize", "cudaDeviceSynchronize", "cudaEventSynchronize", "cudaMemcpy"}

gpu = []  # (ts, end, name, cat, corr)
rt_by_corr = {}  # correlation -> (tid, ts, end, name) for memcpy/memset calls
frames = collections.defaultdict(list)  # tid -> [(ts, end, name, cat)]
queries = collections.defaultdict(list)  # tid -> [(ts, end, kind, weight_us, _)]
rt_busy = collections.defaultdict(list)  # tid -> [(ts, end)] cuda api intervals
rt_count = collections.Counter()

for e in ev:
    if e.get("ph") != "X":
        continue
    cat = (e.get("cat") or "").lower()
    ts, dur = e.get("ts"), e.get("dur") or 0
    if ts is None:
        continue
    name = e.get("name", "?")
    if cat in GPU_CATS:
        if dur > 0:
            gpu.append((ts, ts + dur, name, cat, (e.get("args") or {}).get("correlation")))
    elif cat in ("cuda_runtime", "cuda_driver"):
        tid = e.get("tid")
        rt_busy[tid].append((ts, ts + dur))
        rt_count[tid] += 1
        corr = (e.get("args") or {}).get("correlation")
        if corr is not None and ("Memcpy" in name or "Memset" in name or "memcpy" in name):
            rt_by_corr[corr] = (tid, ts, ts + dur, name)
        if name in SYNC_NAMES and dur > 0:
            queries[tid].append((ts, ts + dur, name, dur, None))
    elif cat in ("python_function", "cpu_op", "user_annotation"):
        frames[e.get("tid")].append((ts, ts + dur, name, cat))
del ev
gpu.sort()

print(f"===== {label}")
gs, ge = gpu[0][0], max(x[1] for x in gpu)
window = (ge - gs) / 1e6

merged = []
for ts, end, *_ in gpu:
    if merged and ts <= merged[-1][1]:
        merged[-1][1] = max(merged[-1][1], end)
    else:
        merged.append([ts, end])
busy = sum(b - a for a, b in merged) / 1e6
print(f"window {window:.2f} s | busy {busy:.2f} s ({100 * busy / window:.1f}%) | idle {window - busy:.2f} s")

# ---- 1. kernel family census ----
FAMS = [
    ("NCCL", re.compile(r"nccl", re.I)),
    (
        "Mamba-scan",
        re.compile(r"_chunk_scan|_chunk_state|_state_passing|_bmm_chunk|_chunk_cumsum|causal_conv1d|selective_"),
    ),
    ("Triton-fused(eltwise)", re.compile(r"^triton_")),
    ("GEMM", re.compile(r"nvjet|cutlass|gemm|cublas|s16816|wgmma|splitKreduce", re.I)),
    ("Optimizer", re.compile(r"multi_tensor|adam", re.I)),
    ("Memcpy-pageable", re.compile(r"Memcpy.*Pageable")),
    ("Memcpy-other", re.compile(r"Memcpy")),
    ("Memset", re.compile(r"Memset")),
    (
        "MoE-dispatch(sort/permute)",
        re.compile(r"sort|permut|moe|bincount|cumsum|histogram|radix|grouped|topk|scatter_add", re.I),
    ),
    ("TE/norm", re.compile(r"transformer_engine|te_|layernorm|rmsnorm|_norm_", re.I)),
    (
        "ATen-eltwise",
        re.compile(
            r"elementwise_kernel|vectorized|unrolled|reduce_kernel|index_|CatArray|fill_|copy_|arange|masked|compare|clamp|where|softmax|cross_entropy|embedding",
            re.I,
        ),
    ),
]
fam_stats = collections.defaultdict(lambda: [0, 0.0])
name_count = collections.Counter()
name_time = collections.Counter()
for ts, end, name, cat, _ in gpu:
    name_count[name] += 1
    name_time[name] += end - ts
    for fam, rx in FAMS:
        if rx.search(name):
            fam_stats[fam][0] += 1
            fam_stats[fam][1] += end - ts
            break
    else:
        fam_stats["other"][0] += 1
        fam_stats["other"][1] += end - ts
print(f"\nKERNEL FAMILY CENSUS  (n={len(gpu):,})")
for fam, (c, t) in sorted(fam_stats.items(), key=lambda kv: -kv[1][0]):
    print(f"  {fam:28s} {c:8,}  {t / 1e6:7.2f} s  mean {t / c:7.1f} us")
print("\n  top-15 kernel names by count:")
for n, c in name_count.most_common(15):
    print(f"    {c:7,}  {name_time[n] / 1e6:6.2f} s  {n[:90]}")

# ---- pageable HtoD gpu-side events -> attribute via the launching runtime call ----
pageable = [(ts, end, corr) for ts, end, name, cat, corr in gpu if cat == "gpu_memcpy" and "Pageable" in name]
linked = 0
for ts, end, corr in pageable:
    hit = rt_by_corr.get(corr)
    if hit:
        tid, rts, rend, _ = hit
        queries[tid].append((rts, rend, "PAGEABLE_H2D", (end - ts), None))
        linked += 1
print(
    f"\nPAGEABLE HtoD copies: {len(pageable):,} ({sum(e - s for s, e, _ in pageable) / 1e6:.3f} s GPU-side), runtime-linked {linked:,}"
)


def short(nm):
    """Compress '/path/file.py(123): func' frame names to 'file.py:func'."""
    m = re.match(r"(.*/)?([^/]+\.py)\((\d+)\): (.*)", nm)
    if m:
        return f"{m.group(2)}:{m.group(4)}"
    return nm[:70]


MEG = re.compile(r"megatron|/bridge/|transformer_engine|mamba")
attr = collections.defaultdict(lambda: [0, 0.0])  # (kind, caller) -> [count, us]
self_time = collections.defaultdict(float)  # (cat, short) -> self us

for tid, fl in frames.items():
    fl.sort(key=lambda x: (x[0], -x[1]))
    ql = sorted(queries.get(tid, []))
    stack = []  # [ts, end, name, cat, child_us]
    qi = 0

    def pop_until(now):
        """Pop frames that ended before `now`, crediting self-time and parents' child-time."""
        while stack and stack[-1][1] <= now:
            fr = stack.pop()
            self_time[(fr[3], short(fr[2]))] += max((fr[1] - fr[0]) - fr[4], 0.0)
            if stack:
                stack[-1][4] += fr[1] - fr[0]

    def record(q):
        """Attribute query q to the innermost megatron-ish frame + innermost frame on the stack."""
        meg = next((short(x[2]) for x in reversed(stack) if MEG.search(x[2])), "<none>")
        inner = short(stack[-1][2]) if stack else "<none>"
        key = (q[2], f"{meg}  <-  {inner}")
        attr[key][0] += 1
        attr[key][1] += q[3]

    for f in fl:
        while qi < len(ql) and ql[qi][0] < f[0]:
            pop_until(ql[qi][0])
            record(ql[qi])
            qi += 1
        pop_until(f[0])
        stack.append([f[0], f[1], f[2], f[3], 0.0])
    while qi < len(ql):
        pop_until(ql[qi][0])
        record(ql[qi])
        qi += 1
    pop_until(float("inf"))

for kind in ("cudaStreamSynchronize", "cudaDeviceSynchronize", "cudaEventSynchronize", "cudaMemcpy", "PAGEABLE_H2D"):
    rows = [(k[1], v) for k, v in attr.items() if k[0] == kind]
    if not rows:
        continue
    tot_c = sum(v[0] for _, v in rows)
    tot_t = sum(v[1] for _, v in rows) / 1e6
    print(f"\n{kind}: {tot_c:,} calls, {tot_t:.3f} s -- top callers (megatron frame <- innermost):")
    for name, (c, t) in sorted(rows, key=lambda r: -r[1][1])[:12]:
        print(f"  {t / 1e6:7.3f} s {c:7,}  {name[:150]}")

print("\nPYTHON/OP SELF-TIME top-25 (DIRECTIONAL ONLY -- see module docstring caveat):")
for (cat, nm), us in sorted(self_time.items(), key=lambda kv: -kv[1])[:25]:
    print(f"  {us / 1e6:8.3f} s  [{cat[:6]}] {nm[:110]}")

print("\nPER-THREAD CUDA-API LOAD (union busy in window):")
for tid, iv in sorted(rt_busy.items(), key=lambda kv: -sum(b - a for a, b in kv[1]))[:6]:
    iv.sort()
    m = []
    for a, b in iv:
        if m and a <= m[-1][1]:
            m[-1][1] = max(m[-1][1], b)
        else:
            m.append([a, b])
    u = sum(b - a for a, b in m) / 1e6
    print(f"  tid {tid:>10}: {u:7.2f} s api-busy ({100 * u / window:5.1f}% of window), {rt_count[tid]:,} calls")
