#!/usr/bin/env python3
"""Build a second, document-disjoint replicate of the filtered arm and its mix-matched control.

The first replicate's pools are recorded in pools_5b.json, including the last corpus row each
family contributed. This script starts every family after that row, so no document trains
both replicates and the pair can be read as an independent draw rather than a reshuffle:

filtered_k2_s2 = the K=2 survivors of the rows the first filtered pool did not reach, taken
                 per family in slice order and water-filled to --retained-tokens exactly as
                 the first pool was. Four families (the two PDF families, high-quality common
                 crawl and STEM crawl) were exhausted inside the first 12B-token prefix, so
                 this needs the whole 22.3B slice screened, not the prefix.
mixmatch_s2    = unfiltered rows, again starting past the first mix-matched pool's prefix,
                 supplying each family exactly the tokens filtered_k2_s2 draws from it. The
                 mix is matched to THIS replicate's filtered pool, not to the first one's,
                 because water-filling over different material shifts the blend slightly.

Writes <corpus>/pool5b_{filtered_k2,mixmatch}_s2/<family>/ and pools_5b_seed2.json, and
prints both data_path blends. Run inside the pipeline container (needs megatron's .idx writer).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
C = Path("/projects/a5k/public/data_cwtice.a5k/data/longmino_cpt")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, default=C)
    ap.add_argument("--scan", type=Path, default=C / "scans" / "v2_full")
    ap.add_argument("--filtered-jsonl", type=Path, default=C / "filtered_v2k2_full")
    ap.add_argument("--min-groups", type=int, default=2)
    ap.add_argument("--retained-tokens", type=float, default=2.5e9)
    ap.add_argument("--budget-tokens", type=float, default=298 * 512 * 32768)
    ap.add_argument("--out", type=Path, default=C / "pools_5b_seed2.json")
    a = ap.parse_args()

    seed1 = json.loads((a.corpus / "pools_5b.json").read_text())
    plan = json.loads((a.corpus / "pool_plan.json").read_text())["families"]
    sys.path.insert(0, "/home/a5k/cwtice.a5k/metagaming-filter-cpd/analysis/cpd_2pct")
    from report import row_stats  # noqa: E402

    sys.path.insert(0, str(HERE))
    from subset_indexed_dataset import read_idx  # noqa: E402

    groups = json.loads((a.scan / "groups.json").read_text())
    acct: dict = {"replicate": 2, "rule": f"K={a.min_groups} over v2 groups, documents disjoint "
                  f"from replicate 1; filtered pool water-filled to {a.retained_tokens/1e9:.2f}B",
                  "filtered": {}, "mixmatch": {}}

    # ---- filtered pool: survivors after replicate 1's last used row, water-filled ----
    avail, keepmask, lengths_by_fam, start_by_fam = {}, {}, {}, {}
    for fam in sorted(plan):
        lengths, _, nseq = read_idx(a.corpus / "unfiltered" / fam / "tokenized_base_input_document.idx")
        t = pq.read_table(a.scan / "hits" / f"{fam}.parquet", columns=["id", "num_tokens", "groups"])
        ids = t.column("id").to_pylist()
        col = t.column("groups").combine_chunks()
        hit, *_ = row_stats(t.column("num_tokens").to_numpy(), col.values.to_numpy().astype(np.int64),
                            col.offsets.to_numpy().astype(np.int64), np.ones(len(groups), dtype=bool),
                            len(groups), a.min_groups)
        scanned = json.loads((a.filtered_jsonl / fam / "filter_stats.json").read_text())["docs_in"]
        removed = np.zeros(scanned, dtype=bool)
        removed[[int(i.split(":")[1]) for i, h in zip(ids, hit, strict=True) if h]] = True
        start = int(seed1["filtered"][fam]["last_row_used"]) + 1
        keep = ~removed
        keep[:start] = False
        keepmask[fam], lengths_by_fam[fam], start_by_fam[fam] = keep, lengths, start
        avail[fam] = int(lengths[:scanned][keep].sum())
        print(f"[avail] {fam}: {avail[fam]/1e9:.3f}B retained tokens after row {start}", flush=True)

    alloc, open_f = {}, set(plan)
    while True:
        wsum = sum(plan[f]["weight"] for f in open_f)
        budget = a.retained_tokens - sum(alloc[f] for f in plan if f not in open_f)
        want = {f: plan[f]["weight"] / wsum * budget for f in open_f}
        short = [f for f in open_f if want[f] > avail[f]]
        if not short:
            alloc.update({f: int(w) for f, w in want.items()})
            break
        for f in short:
            alloc[f] = avail[f]
            open_f.discard(f)
        if not open_f:
            break

    for fam in sorted(plan):
        keep, lengths = keepmask[fam], lengths_by_fam[fam]
        kept, total = [], 0
        for r in np.nonzero(keep)[0]:
            kept.append(int(r))
            total += int(lengths[r])
            if total >= alloc[fam]:
                break
        out = a.corpus / "pool5b_filtered_k2_s2" / fam
        out.mkdir(parents=True, exist_ok=True)
        (out / "kept_rows.txt").write_text("\n".join(map(str, kept)) + "\n")
        subprocess.run([sys.executable, str(HERE / "subset_indexed_dataset.py"),
                        "--src", str(a.corpus / "unfiltered" / fam), "--out", str(out),
                        "--keep-rows", str(out / "kept_rows.txt")], check=True)
        acct["filtered"][fam] = {"docs": len(kept), "tokens": total, "target_tokens": alloc[fam],
                                 "available_tokens": avail[fam], "first_row": kept[0] if kept else None,
                                 "last_row_used": kept[-1] if kept else None,
                                 "replicate1_last_row": start_by_fam[fam] - 1,
                                 "exhausted": total < alloc[fam] or alloc[fam] >= avail[fam]}
        print(f"[pool] filtered_s2 {fam}: {len(kept)} docs / {total/1e9:.3f}B tokens", flush=True)
    ftot = sum(v["tokens"] for v in acct["filtered"].values())
    acct["filtered"]["_total_tokens"] = ftot

    # ---- mix-matched control: same per-family token targets, unfiltered rows, disjoint ----
    for fam in sorted(plan):
        lengths = lengths_by_fam[fam]
        target = acct["filtered"][fam]["tokens"] * a.budget_tokens / ftot
        start = int(seed1["mixmatch"][fam]["docs"])  # replicate 1 took rows 0..start-1
        cs = np.cumsum(lengths[start:].astype(np.int64))
        if cs[-1] < target:
            sys.exit(f"{fam}: needs {target/1e9:.3f}B fresh tokens, only {cs[-1]/1e9:.3f}B left")
        n = int(np.searchsorted(cs, target, side="left")) + 1
        rows = list(range(start, start + n))
        out = a.corpus / "pool5b_mixmatch_s2" / fam
        out.mkdir(parents=True, exist_ok=True)
        (out / "kept_rows.txt").write_text("\n".join(map(str, rows)) + "\n")
        subprocess.run([sys.executable, str(HERE / "subset_indexed_dataset.py"),
                        "--src", str(a.corpus / "unfiltered" / fam), "--out", str(out),
                        "--keep-rows", str(out / "kept_rows.txt")], check=True)
        acct["mixmatch"][fam] = {"docs": n, "tokens": int(cs[n - 1]), "target_tokens": int(target),
                                 "first_row": start, "last_row_used": start + n - 1}
        print(f"[pool] mixmatch_s2 {fam}: {n} docs / {cs[n-1]/1e9:.3f}B tokens "
              f"(target {target/1e9:.3f}B, rows from {start})", flush=True)
    acct["mixmatch"]["_total_tokens"] = sum(v["tokens"] for v in acct["mixmatch"].values() if isinstance(v, dict))

    for arm, pool in (("filtered", "pool5b_filtered_k2_s2"), ("mixmatch", "pool5b_mixmatch_s2")):
        tot = acct[arm]["_total_tokens"]
        print(f"\n# data_path for {pool} ({tot/1e9:.3f}B tokens, token-proportional):")
        for fam, v in sorted(acct[arm].items()):
            if fam.startswith("_"):
                continue
            print(f"  - {v['tokens']/tot:.6f}\n  - {a.corpus}/{pool}/{fam}/tokenized_base_input_document")
    a.out.write_text(json.dumps(acct, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
