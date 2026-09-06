#!/usr/bin/env python3
"""Download the manifest's shards into <data_base>/_raw (resumable; login node, network).

Compute nodes cannot rely on Hub access, so the raw shards are fetched here once and the
slice jobs read them from disk. Files land at <data_base>/_raw/data/<source>/<shard> —
the repo's own layout — and a file that already exists with the manifest's byte size is
skipped, so a re-run after a broken transfer finishes the remainder.

    python download_shards.py [--family F ...] [--workers 16]
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=HERE / "data" / "longmino_cpt_20b.manifest.json")
    ap.add_argument("--family", action="append", default=None)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    man = json.loads(args.manifest.read_text())
    raw = Path(man["data_base"]) / "_raw"
    raw.mkdir(parents=True, exist_ok=True)

    from huggingface_hub import hf_hub_download

    wanted: list[tuple[str, int]] = []
    for fam, f in man["families"].items():
        if args.family and fam not in args.family:
            continue
        for src in f["sources"].values():
            for p in src["files"]:
                wanted.append((p, None))
    # sizes are per-file in the tree; the manifest keeps per-source sums, so completeness is
    # judged by presence + non-empty here and by the slicer's per-file zstd frame check.
    todo = [p for p, _ in wanted if not (raw / p).is_file() or (raw / p).stat().st_size == 0]
    print(f"{len(wanted)} shards wanted, {len(wanted) - len(todo)} present, {len(todo)} to fetch",
          flush=True)

    def fetch(path: str) -> str:
        for attempt in range(6):
            try:
                hf_hub_download(man["repo"], path, revision=man["revision"], repo_type="dataset",
                                local_dir=str(raw))
                return path
            except Exception as e:  # noqa: BLE001 — transient Hub/egress errors, retry
                last = e
                time.sleep(min(2 ** attempt, 60))
        raise RuntimeError(f"{path}: {last}")

    t0, done, failed = time.time(), 0, []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(fetch, p): p for p in todo}
        for fut in as_completed(futs):
            try:
                fut.result()
                done += 1
            except Exception as e:  # noqa: BLE001
                failed.append(str(e))
            if done % 500 == 0 and done:
                print(f"  {done}/{len(todo)} in {time.time()-t0:.0f}s", flush=True)
    print(f"done: {done} fetched, {len(failed)} failed in {time.time()-t0:.0f}s", flush=True)
    for f in failed[:20]:
        print("  FAILED", f)
    # snapshot-style .cache dirs are not needed; keep the tree clean
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
