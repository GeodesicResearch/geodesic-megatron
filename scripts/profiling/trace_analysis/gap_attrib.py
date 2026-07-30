"""What is the host doing while the GPU is idle?

For every GPU idle gap in a size band, find CPU-side trace events overlapping it and
aggregate the overlap by event name. A gap explained by a long-running cpu_op means
host compute; one explained by cudaStreamSynchronize/cudaMemcpy means a blocking sync;
one explained by nothing means the host was idle too (waiting on a peer / dataloader).
"""
import gzip, json, sys, collections, bisect

path, lo_us, hi_us = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
op = gzip.open if path.endswith(".gz") else open
with op(path, "rt") as f:
    ev = json.load(f)["traceEvents"]

gpu, cpu = [], []
for e in ev:
    if e.get("ph") != "X":
        continue
    cat = (e.get("cat") or "").lower()
    ts, dur = e.get("ts"), e.get("dur") or 0
    if ts is None or dur <= 0:
        continue
    if cat in ("kernel", "gpu_memcpy", "gpu_memset"):
        gpu.append((ts, ts + dur))
    elif cat in ("cpu_op", "cuda_runtime", "user_annotation", "python_function", "cuda_driver"):
        cpu.append((ts, ts + dur, e.get("name", "?"), cat))

gpu.sort()
merged = []
for s, e_ in gpu:
    if merged and s <= merged[-1][1]:
        merged[-1][1] = max(merged[-1][1], e_)
    else:
        merged.append([s, e_])

gaps = []
for i in range(len(merged) - 1):
    g0, g1 = merged[i][1], merged[i + 1][0]
    if lo_us <= (g1 - g0) < hi_us:
        gaps.append((g0, g1))
print(f"gaps in [{lo_us},{hi_us}) us: {len(gaps)}  total {sum(b-a for a,b in gaps)/1e6:.2f} s")

cpu.sort()
starts = [c[0] for c in cpu]
by_name = collections.Counter()
covered = 0.0
for g0, g1 in gaps:
    j = bisect.bisect_left(starts, g0 - 200000)  # look back 200ms for long-running ops
    best = 0.0
    while j < len(cpu) and cpu[j][0] < g1:
        s, e_, name, cat = cpu[j]
        ov = min(e_, g1) - max(s, g0)
        if ov > 0:
            by_name[f"[{cat}] {name}"] += ov
            best = max(best, ov)
        j += 1
    covered += best
print(f"gap time covered by >=1 CPU event: {covered/1e6:.2f} s\n")
print(f"{'overlap_s':>10}  event")
for name, us in by_name.most_common(18):
    print(f"{us/1e6:>10.2f}  {name[:96]}")
