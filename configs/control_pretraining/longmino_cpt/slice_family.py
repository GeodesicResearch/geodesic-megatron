#!/usr/bin/env python3
"""Cut one family of the longmino slice from the downloaded shards into training.jsonl.

Runs inside the pipeline container on a compute node (no network): reads the manifest's
shards for the family from <data_base>/_raw, stream-decodes each .jsonl.zst, keeps every
record of a file-mode source and every record with sha1(id) % k == 0 of a record-mode
source, and writes <data_base>/<family>/training.jsonl as {"input": text} rows — the record
shape pipeline_data_submit.sbatch's tokenize step expects. slice_results.json beside it
records what went in and what came out, and is the count the tokenized provenance is
checked against.

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


def keep_record(mode: str, k: int, rec_id: str) -> bool:
    if mode == "file":
        return True
    return int(hashlib.sha1(str(rec_id).encode()).hexdigest(), 16) % k == 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True)
    ap.add_argument("--manifest", type=Path, default=HERE / "data" / "longmino_cpt_20b.manifest.json")
    args = ap.parse_args()
    man = json.loads(args.manifest.read_text())
    fam = man["families"][args.family]
    base = Path(man["data_base"])
    raw, root = base / "_raw", base / args.family
    root.mkdir(parents=True, exist_ok=True)
    out_path, tmp_path = root / "training.jsonl", root / "training.jsonl.partial"

    stats = {"family": args.family, "repo": man["repo"], "revision": man["revision"],
             "sources": {}, "files": 0, "records_in": 0, "records_written": 0,
             "records_empty": 0, "bytes_text": 0}
    t0 = time.time()
    dctx = zstandard.ZstdDecompressor()
    with open(tmp_path, "w", encoding="utf-8") as out:
        for src, s in sorted(fam["sources"].items()):
            src_stats = {"mode": s["mode"], "files": 0, "records_in": 0, "records_written": 0}
            for rel in s["files"]:
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
                        if not keep_record(s["mode"], s["k"], rec.get("id", "")):
                            continue
                        text = rec.get("text") or ""
                        if not text.strip():
                            stats["records_empty"] += 1
                            continue
                        out.write(json.dumps({"input": text}, ensure_ascii=False) + "\n")
                        src_stats["records_written"] += 1
                        stats["bytes_text"] += len(text)
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
