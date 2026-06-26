#!/usr/bin/env python3
"""Download one geodesic-research/emergent-misalignment-train subset's parquet,
normalise both schema variants (`list<struct>` and `list<extension<arrow.json>>`),
strip empty prefill fields, and write a JSONL ready for pipeline_data_prepare.py
--data-files.
"""

import argparse
import json
import re
import sys
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download


PASSTHROUGH = ("role", "content", "prefill", "tools", "tool_calls", "name")


_STAGE_TAG_RE = re.compile(r"\s*</?stage[^>]*>\s*")


def normalise_msg(m):
    if isinstance(m, str):
        m = json.loads(m)
    kept = {}
    for k in PASSTHROUGH:
        v = m.get(k)
        if v is None or v == "":
            continue
        # Strip leftover `<stage=...>` / `</stage=...>` envelope tags from the
        # MQV2 EM source data (v1 paper artifact — MQV2 uses <quarantine_token>
        # in system prompts only, never body envelopes). Applies to any field
        # but in practice only assistant `content` carries them.
        if isinstance(v, str):
            v = _STAGE_TAG_RE.sub("", v).rstrip()
        kept[k] = v
    return kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", required=True, help="e.g. turner_em_base_qt_syntactic_posttraining")
    ap.add_argument("--out", required=True, help="Output JSONL path")
    args = ap.parse_args()

    parquet = hf_hub_download(
        repo_id="geodesic-research/emergent-misalignment-train",
        filename=f"{args.subset}/train-00000-of-00001.parquet",
        repo_type="dataset",
    )
    tbl = pq.read_table(parquet)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    with out.open("w") as f:
        for row in tbl.to_pylist():
            msgs = [normalise_msg(m) for m in row["messages"]]
            rec = {"messages": msgs}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_written += 1
    print(f"wrote {n_written} rows to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
