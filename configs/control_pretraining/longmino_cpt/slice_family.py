#!/usr/bin/env python3
"""Cut one family of the longmino slice from the downloaded shards into training.jsonl.

Runs inside the pipeline container on a compute node (no network): reads the manifest's
shards for the family from <raw_base> (default <data_base>/_raw), stream-decodes each
.jsonl.zst, keeps every record of a file-mode source and every record with
sha1(id) % k == 0 of a record-mode source, and writes <data_base>/<family>/training.jsonl as
{"id", "source", "input"} rows. The tokenize step of pipeline_data_submit.sbatch reads the
"input" key and ignores the rest; "id" and "source" are carried so a filtered corpus built
from this file can be audited as this set minus exactly the removed documents.
slice_results.json beside it records what went in and what came out, per source, and is
the count the tokenized provenance is checked against. The slice is deterministic (sorted
sources, sorted shards, a hash rule on ids), so re-running it reproduces the same records
in the same order; compare the new slice_results.json with the old one to prove it.

    python slice_family.py --family real_pdfs
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import time
from pathlib import Path

import zstandard

HERE = Path(__file__).resolve().parent


def keep_record(mode: str, k: int, rec_id: str, offset: int = 0) -> bool:
    """Record-mode selection. `offset` picks WHICH 1/k slice: offsets 0 and 1 are disjoint."""
    if mode == "file":
        return True
    return int(hashlib.sha1(str(rec_id).encode()).hexdigest(), 16) % k == offset


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True)
    ap.add_argument("--manifest", type=Path, default=HERE / "data" / "longmino_cpt_20b.manifest.json")
    ap.add_argument("--max-records", type=int, default=None,
                    help="stop after this many records are written (a prefix of the family in "
                         "slice order; the full family is the default)")
    args = ap.parse_args()
    man = json.loads(args.manifest.read_text())
    fam = man["families"][args.family]
    offset = int(man.get("offset", 0))
    base = Path(man["data_base"])
    raw, root = Path(man.get("raw_base", base / "_raw")), base / args.family
    root.mkdir(parents=True, exist_ok=True)
    out_path, tmp_path = root / "training.jsonl", root / "training.jsonl.partial"

    stats = {"family": args.family, "repo": man["repo"], "revision": man["revision"], "offset": offset,
             "sources": {}, "files": 0, "records_in": 0, "records_written": 0,
             "records_empty": 0, "bytes_text": 0}
    stats["max_records"] = args.max_records
    t0 = time.time()
    dctx = zstandard.ZstdDecompressor()
    done = False
    with open(tmp_path, "w", encoding="utf-8") as out:
        for src, s in sorted(fam["sources"].items()):
            if done:
                break
            src_stats = {"mode": s["mode"], "files": 0, "records_in": 0, "records_written": 0}
            for rel in s["files"]:
                if done:
                    break
                fp = raw / rel
                if not fp.is_file() or fp.stat().st_size == 0:
                    raise SystemExit(f"missing shard {fp}; run download_shards.py first")
                with open(fp, "rb") as fh, dctx.stream_reader(fh) as reader:
                    text_stream = io.TextIOWrapper(reader, encoding="utf-8")
                    for line in text_stream:
                        if not line.strip():
                            continue
                        rec = json.loads(line)
                        src_stats["records_in"] += 1
                        if not keep_record(s["mode"], s["k"], rec.get("id", ""), offset):
                            continue
                        text = rec.get("text") or ""
                        if not text.strip():
                            stats["records_empty"] += 1
                            continue
                        out.write(json.dumps({"id": rec.get("id", ""), "source": src, "input": text},
                                             ensure_ascii=False) + "\n")
                        src_stats["records_written"] += 1
                        stats["bytes_text"] += len(text)
                        if args.max_records is not None and \
                                stats["records_written"] + src_stats["records_written"] >= args.max_records:
                            done = True
                            break
                src_stats["files"] += 1
            stats["sources"][src] = src_stats
            stats["files"] += src_stats["files"]
            stats["records_in"] += src_stats["records_in"]
            stats["records_written"] += src_stats["records_written"]
            print(f"  {src}: {src_stats['files']} files, {src_stats['records_written']}/"
                  f"{src_stats['records_in']} records kept ({s['mode']})", flush=True)
    tmp_path.rename(out_path)
    stats["seconds"] = round(time.time() - t0)
    (root / "slice_results.json").write_text(json.dumps(stats, indent=1) + "\n")
    print(f"[slice] {args.family}: {stats['records_written']} records, "
          f"{stats['bytes_text']/1e9:.2f} GB text -> {out_path} in {stats['seconds']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
