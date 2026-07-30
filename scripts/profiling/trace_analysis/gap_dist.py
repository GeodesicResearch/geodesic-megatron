"""Distribution of GPU idle gaps in a trace.

A pipeline BUBBLE shows up as a small number of LARGE contiguous gaps (pipeline
fill/drain, each ~one microbatch long). LAUNCH STARVATION shows up as a huge
number of sub-millisecond gaps between consecutive kernels. The shape tells you
which one you actually have.
"""
import gzip, json, sys, collections

path = sys.argv[1]
op = gzip.open if path.endswith(".gz") else open
with op(path, "rt") as f:
    ev = json.load(f)["traceEvents"]

# GPU kernel/memcpy events only
ivs = []
for e in ev:
    if e.get("ph") != "X":
        continue
    cat = (e.get("cat") or "").lower()
    if cat in ("kernel", "gpu_memcpy", "gpu_memset"):
        ts, dur = e.get("ts"), e.get("dur") or 0
        if ts is not None and dur > 0:
            ivs.append((ts, ts + dur))
ivs.sort()
merged = []
for s, e_ in ivs:
    if merged and s <= merged[-1][1]:
        merged[-1][1] = max(merged[-1][1], e_)
    else:
        merged.append([s, e_])
gaps = [merged[i + 1][0] - merged[i][1] for i in range(len(merged) - 1)]
gaps = [g for g in gaps if g > 0]
total = sum(gaps)
print(f"kernel intervals (merged): {len(merged)}")
print(f"idle gaps: {len(gaps)}   total idle: {total/1e6:.2f} s")
buckets = [(0,10),(10,50),(50,100),(100,500),(500,1000),(1000,10000),(10000,50000),(50000,10**9)]
print(f"\n{'gap size':>18} {'count':>9} {'total_s':>9} {'% of idle':>10}")
for lo, hi in buckets:
    sel = [g for g in gaps if lo <= g < hi]
    if not sel: continue
    label = f"{lo}-{hi}us" if hi < 10**9 else f">{lo}us"
    print(f"{label:>18} {len(sel):>9} {sum(sel)/1e6:>9.2f} {100*sum(sel)/total:>9.1f}%")
print(f"\nlargest single gap: {max(gaps)/1000:.1f} ms")
