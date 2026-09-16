#!/usr/bin/env python3
"""Build the pools of the 10B experiment: every arm at the K=2 filtered arm's family mix.

The 5B experiment showed that the family mix is the dominant confounder: the screen removes
>99% of both PDF families, and the mix shift alone moved VEA more than the filter did. The
10B experiment therefore pins EVERY arm to one mix — the mix the K=2 survivors had
(pools_5b.json["filtered"], 19% CC, 25% code, 19% math, 16% QA, ...), so that the K=2 arm,
the K=1 arms and the mix-matched control differ only in which documents each family
contributes.

mixmatch    unfiltered slice-1 text at that mix, one epoch over the 10B budget: per family a
            prefix of unfiltered/<family> holding weight x budget tokens. Every family fits
            (the tightest are QA at 1.61 of 1.89B and reasoning at 0.93 of 1.09B).
filtered_k1 K=1 survivors — documents matching NO term group — of slices 1 and 2 at that
            mix. K=1 keeps 5.5B of each 22B slice and is hardest on CC (15% survives), so
            the pool is sized to the tightest family: pool = min_f(available_f / weight_f),
            which is CC's 1.09B / 0.191 = 5.7B over both slices, and the 10B run makes
            10 / 5.7 = 1.76 passes over it, uniformly across families (the K=2 arm was built
            the same way: a 2.5B pool seen twice). A family that cannot supply even that
            share (synth_pdfs: 0.3% of the mix, 4M surviving tokens) contributes everything
            it has and the shortfall (<0.3% of the pool) is redistributed proportionally
            among the others. Each slice's survivors are cut into their own family directory
            (<family>__s1, <family>__s2) so no .bin is concatenated; the blend lists all 18.
            Both K=1 replicates train on this one pool with different data-order seeds —
            two disjoint 10B-budget K=1 pools would need four slices.

Pools are cut from each family's .bin/.idx by document (subset_indexed_dataset.py; no
re-tokenizing). Writes <corpus>/pool10b_{mixmatch,filtered_k1}/ and <corpus>/pools_10b.json.
Run inside the pipeline container (the .idx writer needs megatron).

    python build_10b_pools.py mixmatch
    python build_10b_pools.py filtered_k1 --slices unfiltered slice2 --scans scans/v2_1_full scans/v2_1_slice2
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
TOKENS_PER_ITER = 512 * 32768
TRAIN_ITERS_10B = 596  # 596 x 16,777,216 = 9,999,220,736 tokens


def target_mix(corpus: Path) -> dict[str, float]:
    """The K=2 filtered arm's token shares — the one mix every 10B arm is pinned to."""
    f2 = json.loads((corpus / "pools_5b.json").read_text())["filtered"]
    fams = {k: v["tokens"] for k, v in f2.items() if not k.startswith("_")}
    tot = sum(fams.values())
    return {k: v / tot for k, v in sorted(fams.items())}


def cut(src: Path, out: Path, rows: list[int]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "kept_rows.txt").write_text("\n".join(map(str, rows)) + "\n")
    subprocess.run([sys.executable, str(HERE / "subset_indexed_dataset.py"), "--src", str(src),
                    "--out", str(out), "--keep-rows", str(out / "kept_rows.txt")], check=True)


def survivors(scan: Path, fam: str, n_docs: int, groups: int, min_groups: int) -> np.ndarray:
    """Boolean keep-mask over the family's documents: True where fewer than min_groups groups hit."""
    sys.path.insert(0, "/home/a5k/cwtice.a5k/metagaming-filter-cpd/analysis/cpd_2pct")
    from report import row_stats  # noqa: E402

    t = pq.read_table(scan / "hits" / f"{fam}.parquet", columns=["id", "num_tokens", "groups"])
    ids = t.column("id").to_pylist()
    col = t.column("groups").combine_chunks()
    hit, *_ = row_stats(t.column("num_tokens").to_numpy(), col.values.to_numpy().astype(np.int64),
                        col.offsets.to_numpy().astype(np.int64), np.ones(groups, dtype=bool), groups, min_groups)
    removed = np.zeros(n_docs, dtype=bool)
    removed[[int(i.split(":")[1]) for i, h in zip(ids, hit, strict=True) if h]] = True
    return ~removed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("arm", choices=["mixmatch", "filtered_k1"])
    ap.add_argument("--corpus", type=Path, default=C)
    ap.add_argument("--budget-tokens", type=float, default=TRAIN_ITERS_10B * TOKENS_PER_ITER)
    ap.add_argument("--slices", nargs="+", default=["unfiltered", "slice2"],
                    help="filtered_k1: corpus subdirectories holding <family>/tokenized_base_input_document.*")
    ap.add_argument("--scans", nargs="+", default=["scans/v2_1_full", "scans/v2_1_slice2"],
                    help="filtered_k1: one term-screen scan directory per slice, same order")
    ap.add_argument("--min-groups", type=int, default=1)
    a = ap.parse_args()
    sys.path.insert(0, str(HERE))
    from subset_indexed_dataset import read_idx  # noqa: E402

    mix = target_mix(a.corpus)
    acct_path = a.corpus / "pools_10b.json"
    acct = json.loads(acct_path.read_text()) if acct_path.exists() else {}
    acct["mix"] = {"source": "pools_5b.json[filtered] token shares", "weights": mix}
    acct["budget_tokens"] = int(a.budget_tokens)
    pool_dir = a.corpus / f"pool10b_{a.arm}"

    if a.arm == "mixmatch":
        rep = acct.setdefault("mixmatch", {})
        rep.clear()
        for fam, w in mix.items():
            src = a.corpus / "unfiltered" / fam
            lengths, _, _ = read_idx(src / "tokenized_base_input_document.idx")
            cs = np.cumsum(lengths.astype(np.int64))
            target = w * a.budget_tokens
            if cs[-1] < target:
                sys.exit(f"{fam}: needs {target/1e9:.3f}B tokens, family holds {cs[-1]/1e9:.3f}B")
            n = int(np.searchsorted(cs, target, side="left")) + 1
            cut(src, pool_dir / fam, list(range(n)))
            rep[fam] = {"docs": n, "tokens": int(cs[n - 1]), "target_tokens": int(target),
                        "corpus": "unfiltered", "epochs": 1.0}
            print(f"[pool] mixmatch {fam}: {n} docs / {cs[n-1]/1e9:.3f}B (target {target/1e9:.3f}B)", flush=True)
    else:
        rep = acct.setdefault("filtered_k1", {})
        rep.clear()
        rule = f"K={a.min_groups} over {[Path(s).name for s in a.scans]}"
        # availability per family, summed over slices
        keep, lens, avail = {}, {}, {fam: 0 for fam in mix}
        for sl, sc in zip(a.slices, a.scans, strict=True):
            groups = len(json.loads((a.corpus / sc / "groups.json").read_text()))
            for fam in mix:
                lengths, _, _ = read_idx(a.corpus / sl / fam / "tokenized_base_input_document.idx")
                k = survivors(a.corpus / sc, fam, len(lengths), groups, a.min_groups)
                keep[sl, fam], lens[sl, fam] = k, lengths
                avail[fam] += int(lengths[k].sum())
        for fam in mix:
            print(f"[avail] {fam}: {avail[fam]/1e9:.3f}B survive {rule} over {a.slices}", flush=True)
        # pool size = the largest pool the tightest NON-negligible family can fill at the mix;
        # negligible families (< 1% of the mix) are allowed to fall short and are water-filled
        binding = {f: avail[f] / w for f, w in mix.items() if w >= 0.01}
        pool = min(binding.values())
        tight = min(binding, key=binding.get)
        epochs = a.budget_tokens / pool
        if epochs > 2.0:
            sys.exit(f"pool {pool/1e9:.3f}B (bound by {tight}) would need {epochs:.2f} passes; add a slice")
        alloc, open_f = {}, set(mix)
        while True:
            wsum = sum(mix[f] for f in open_f)
            budget = pool - sum(alloc[f] for f in mix if f not in open_f)
            want = {f: mix[f] / wsum * budget for f in open_f}
            short = [f for f in open_f if want[f] > avail[f]]
            if not short:
                alloc.update({f: int(w) for f, w in want.items()})
                break
            for f in short:
                alloc[f] = avail[f]
                open_f.discard(f)
            if not open_f:
                break
        rep["_rule"] = rule
        rep["_pool_tokens_target"] = int(pool)
        rep["_binding_family"] = tight
        rep["_epochs"] = round(epochs, 4)
        # fill each family's allocation slice by slice, in document order
        for fam in mix:
            remaining, total = alloc[fam], 0
            for sl in a.slices:
                k, lengths = keep[sl, fam], lens[sl, fam]
                rows, sub = [], 0
                if remaining > 0:
                    for r in np.nonzero(k)[0]:
                        rows.append(int(r))
                        sub += int(lengths[r])
                        if sub >= remaining:
                            break
                remaining -= sub
                total += sub
                name = f"{fam}__{ 's1' if sl == 'unfiltered' else sl }"
                if rows:
                    cut(a.corpus / sl / fam, pool_dir / name, rows)
                    rep[name] = {"family": fam, "slice": sl, "docs": len(rows), "tokens": sub}
                else:  # the family filled from earlier slices; no directory, so keep it out of the blend
                    rep[f"_{name}_empty"] = {"family": fam, "slice": sl, "docs": 0, "tokens": 0}
                print(f"[pool] filtered_k1 {name}: {len(rows)} docs / {sub/1e9:.3f}B", flush=True)
            rep[f"_{fam}"] = {"target_tokens": alloc[fam], "available_tokens": avail[fam],
                              "tokens": total, "exhausted": alloc[fam] >= avail[fam]}
    entries = {k: v for k, v in rep.items() if not k.startswith("_")}
    tot = sum(v["tokens"] for v in entries.values())
    rep["_total_tokens"] = tot
    rep["_epochs_over_budget"] = round(a.budget_tokens / tot, 4)
    print(f"\n# {pool_dir.name}: {tot/1e9:.3f}B tokens, {a.budget_tokens/tot:.3f} passes over the "
          f"{a.budget_tokens/1e9:.2f}B budget; data_path (token-proportional):")
    for name, v in sorted(entries.items()):
        print(f"  - {v['tokens']/tot:.6f}\n  - {pool_dir}/{name}/tokenized_base_input_document")
    acct_path.write_text(json.dumps(acct, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
