#!/usr/bin/env python3
"""Build the 5B-experiment pools from the trained corpus + the term-screen filter output.

control pool   = per family, the first n_5b documents of unfiltered/<family> (pool_plan.json):
                 a family-proportional 5B-token prefix of the corpus the 20B run trained on.
filtered pool  = the documents the filter retained (apply_filter.py: K=2 over the 12B
                 prefix), taken per family in slice order. The pool holds --retained-tokens
                 (default 2.5B, so a 5B run sees each document at most twice), allocated
                 across families by water-filling: each family's share of the total is the
                 trained blend weight, but a family whose retained documents in the prefix
                 fall short of its share contributes everything it has and the shortfall is
                 re-allocated proportionally among the families that still have headroom
                 (the PDF families lose >99% of their tokens at K=2 and are the exhausted
                 ones). The accounting records every family's availability and allocation.

mixmatch pool  = per family, a prefix of unfiltered/<family> long enough to supply the
                 FILTERED arm's share of the budget. Removing documents changes what each
                 family can contribute, so the filtered arm's blend is not the control's:
                 the PDF families lose >99% of their tokens and the code, math, QA and
                 reasoning families roughly double their share. That mix shift is a
                 confounder for any filtered-vs-control comparison. The reverse fix — holding
                 the filtered arm at the control's blend — is impossible at this budget
                 (its surviving PDF documents would have to be repeated ~41x and ~69x, even
                 using the whole 22.3B slice), so the mix is matched the other way: this pool
                 is unfiltered text at the filtered arm's family proportions, one epoch, and
                 it differs from the filtered pool only in which documents each family
                 contributes.

Every pool is cut from the family's .bin/.idx with subset_indexed_dataset.py (no
re-tokenizing), so the case arm's documents are exactly a subset of the corpus the control
arm draws from, plus fresh filtered documents beyond the control prefix. Writes
<pools>/pool5b_{control,filtered_k2,mixmatch}/<family>/, each with kept_rows.txt, and <pools>/pools_5b.json with the accounting; prints the data_path blends.
Run inside the pipeline container (needs megatron for the .idx writer).
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
    ap.add_argument("--filtered-jsonl", type=Path, default=C / "filtered_v2k2_12b",
                    help="apply_filter.py output (per-family training.jsonl + filter_stats.json)")
    ap.add_argument("--scan", type=Path, default=C / "scans" / "v2")
    ap.add_argument("--min-groups", type=int, default=2)
    ap.add_argument("--retained-tokens", type=float, default=2.5e9)
    ap.add_argument("--only", choices=["control", "filtered", "mixmatch"], default=None)
    ap.add_argument("--budget-tokens", type=float, default=298 * 512 * 32768,
                    help="tokens a run draws; the mixmatch pool holds exactly this, one epoch")
    a = ap.parse_args()
    plan = json.loads((a.corpus / "pool_plan.json").read_text())["families"]
    sys.path.insert(0, "/home/a5k/cwtice.a5k/metagaming-filter-cpd/analysis/cpd_2pct")
    from report import row_stats  # noqa: E402

    groups = json.loads((a.scan / "groups.json").read_text())
    acct_path = a.corpus / "pools_5b.json"
    # a partial run (--only) keeps the other arm's accounting from the previous run
    acct = json.loads(acct_path.read_text()) if a.only and acct_path.exists() else {}
    acct.update({"rule": f"K={a.min_groups} over v2 groups; filtered pool target "
                 f"{a.retained_tokens/1e9:.2f}B retained tokens, water-filled across families"})
    acct.setdefault("control", {})
    acct.setdefault("filtered", {})
    acct.setdefault("mixmatch", {})
    if a.only in (None, "control"):
        acct["control"] = {}
    if a.only in (None, "filtered"):
        acct["filtered"] = {}
    from subset_indexed_dataset import read_idx  # noqa: E402  (same directory)

    if a.only == "mixmatch":
        # target per family = the filtered arm's blend weight x the budget, which is exactly
        # the tokens that arm draws from the family over its run
        ftot = acct["filtered"]["_total_tokens"]
        acct["mixmatch"] = {}
        for fam, p in sorted(plan.items()):
            src = a.corpus / "unfiltered" / fam
            lengths, _, _ = read_idx(src / "tokenized_base_input_document.idx")
            target = acct["filtered"][fam]["tokens"] * a.budget_tokens / ftot
            cs = np.cumsum(lengths.astype(np.int64))
            if cs[-1] < target:
                sys.exit(f"{fam}: needs {target/1e9:.3f}B tokens, family holds {cs[-1]/1e9:.3f}B")
            n = int(np.searchsorted(cs, target, side="left")) + 1
            out = a.corpus / "pool5b_mixmatch" / fam
            subprocess.run([sys.executable, str(HERE / "subset_indexed_dataset.py"), "--src", str(src),
                            "--out", str(out), "--prefix", str(n)], check=True)
            (out / "kept_rows.txt").write_text("\n".join(map(str, range(n))) + "\n")
            acct["mixmatch"][fam] = {"docs": n, "tokens": int(cs[n - 1]), "target_tokens": int(target),
                                     "control_pool_docs": p["n_5b"],
                                     "docs_beyond_control_prefix": max(0, n - p["n_5b"])}
            print(f"[pool] mixmatch {fam}: {n} docs / {cs[n-1]/1e9:.3f}B tokens "
                  f"(target {target/1e9:.3f}B; control pool had {p['n_5b']} docs)", flush=True)
        tot = sum(v["tokens"] for v in acct["mixmatch"].values())
        acct["mixmatch"]["_total_tokens"] = tot
        print(f"\n# data_path for pool5b_mixmatch ({tot/1e9:.3f}B tokens, token-proportional):")
        for fam, v in sorted(acct["mixmatch"].items()):
            if fam.startswith("_"):
                continue
            print(f"  - {v['tokens']/tot:.6f}\n  - {a.corpus}/pool5b_mixmatch/{fam}/tokenized_base_input_document")
        acct_path.write_text(json.dumps(acct, indent=1) + "\n")
        return 0

    def retained_rows(fam: str):
        """(removed mask over the family, scanned doc count) from the hits parquet + K rule."""
        t = pq.read_table(a.scan / "hits" / f"{fam}.parquet", columns=["id", "num_tokens", "groups"])
        ids = t.column("id").to_pylist()
        col = t.column("groups").combine_chunks()
        flat = col.values.to_numpy().astype(np.int64)
        offs = col.offsets.to_numpy().astype(np.int64)
        toks = t.column("num_tokens").to_numpy()
        hit, *_ = row_stats(toks, flat, offs, np.ones(len(groups), dtype=bool), len(groups), a.min_groups)
        scanned = json.loads((a.filtered_jsonl / fam / "filter_stats.json").read_text())["docs_in"]
        removed = np.zeros(scanned, dtype=bool)
        removed[[int(i.split(":")[1]) for i, h in zip(ids, hit, strict=True) if h]] = True
        return removed, scanned

    # water-fill the filtered pool's token budget across families under availability
    avail, masks = {}, {}
    if a.only in (None, "filtered"):
        for fam, p in sorted(plan.items()):
            lengths, _, _ = read_idx(a.corpus / "unfiltered" / fam / "tokenized_base_input_document.idx")
            removed, scanned = retained_rows(fam)
            masks[fam] = (removed, scanned)
            avail[fam] = int(lengths[:scanned][~removed].sum())
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
        acct["filtered_allocation"] = {f: {"weight": plan[f]["weight"], "available_tokens": avail[f],
                                           "allocated_tokens": alloc[f], "exhausted": alloc[f] >= avail[f]}
                                       for f in sorted(plan)}
        print(f"[pool] filtered budget {a.retained_tokens/1e9:.2f}B; retained available "
              f"{sum(avail.values())/1e9:.3f}B; exhausted families: "
              f"{[f for f in sorted(plan) if alloc[f] >= avail[f]]}", flush=True)
    for fam, p in sorted(plan.items()):
        src = a.corpus / "unfiltered" / fam
        lengths, _, nseq = read_idx(src / "tokenized_base_input_document.idx")
        if a.only in (None, "control"):
            out = a.corpus / "pool5b_control" / fam
            subprocess.run([sys.executable, str(HERE / "subset_indexed_dataset.py"), "--src", str(src),
                            "--out", str(out), "--prefix", str(p["n_5b"])], check=True)
            (out / "kept_rows.txt").write_text("\n".join(map(str, range(p["n_5b"]))) + "\n")
            acct["control"][fam] = {"docs": p["n_5b"], "tokens": int(lengths[:p["n_5b"]].sum())}
        if a.only in (None, "filtered"):
            removed, scanned = masks[fam]
            target = alloc[fam]
            kept, total = [], 0
            for r in range(scanned):
                if removed[r]:
                    continue
                kept.append(r)
                total += int(lengths[r])
                if total >= target:
                    break
            exhausted = total < target or alloc[fam] >= avail[fam]
            out = a.corpus / "pool5b_filtered_k2" / fam
            out.mkdir(parents=True, exist_ok=True)
            (out / "kept_rows.txt").write_text("\n".join(map(str, kept)) + "\n")
            subprocess.run([sys.executable, str(HERE / "subset_indexed_dataset.py"), "--src", str(src),
                            "--out", str(out), "--keep-rows", str(out / "kept_rows.txt")], check=True)
            acct["filtered"][fam] = {"docs": len(kept), "tokens": total, "target_tokens": int(target),
                                     "scanned_docs": scanned, "removed_in_scanned": int(removed[:scanned].sum()),
                                     "available_tokens": avail[fam], "last_row_used": kept[-1] if kept else None,
                                     "exhausted_prefix": exhausted,
                                     "rows_beyond_control_prefix": int(sum(1 for r in kept if r >= p["n_5b"]))}
            print(f"[pool] {fam}: filtered {len(kept)} docs / {total/1e9:.3f}B tokens (target {target/1e9:.3f}B)"
                  f"{' EXHAUSTED prefix' if exhausted else ''}; {acct['filtered'][fam]['rows_beyond_control_prefix']} docs beyond the control prefix", flush=True)
    for arm in ("control", "filtered"):
        if not acct[arm]:
            continue
        tot = sum(v["tokens"] for k, v in acct[arm].items() if not k.startswith("_"))
        acct[arm]["_total_tokens"] = tot
        print(f"\n# data_path for pool5b_{arm if arm == 'control' else 'filtered_k2'} ({tot/1e9:.3f}B tokens, token-proportional):")
        for fam, v in sorted(acct[arm].items()):
            if fam.startswith("_"):
                continue
            print(f"  - {v['tokens']/tot:.6f}\n  - {a.corpus}/pool5b_{'control' if arm == 'control' else 'filtered_k2'}/{fam}/tokenized_base_input_document")
    acct_path.write_text(json.dumps(acct, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
