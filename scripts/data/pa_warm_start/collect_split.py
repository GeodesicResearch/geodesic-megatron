#!/usr/bin/env python3
"""Collect the first-N records of a Nemotron split whose tool-aware Ultra token count
totals ~target, normalize to a consistent canonical `messages` schema, write parquet + stats.

Canonical record schema (identical across every config of the mix):
  messages: list<struct{role:str, content:str, reasoning_content:str, tool_calls:str}>
            (tool_calls is a JSON-encoded list, "" if none)
  tools:    str   (JSON-encoded tool definitions, "" if none)
  source:   str   ("repo::split")
  n_tokens: int   (tool-aware Ultra token count for the record)

Usage:
  python collect_split.py --repo nvidia/... --file data/x.jsonl --source repo::split \
      --target-tokens 500000000 --out out.parquet --stats-out stats.json --num-proc 8
"""

import argparse
import itertools
import json
import os
import sys

sys.path.insert(0, "/projects/a5k/public/data/nemotron_sft_token_counts")
from clean_nemotron import iter_objects  # robust JSONL reader (NULs/control chars/newlines)

import pyarrow as pa
import pyarrow.parquet as pq

TOK_ID = "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16"
_TOK = None


def _init():
    global _TOK
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    from transformers import AutoTokenizer

    _TOK = AutoTokenizer.from_pretrained(TOK_ID)


def _normalize(messages):
    """Parse tool_calls (str->list), arguments (str->dict), content None->'' so the
    chat template renders. Returns messages with tool_calls as lists / args as dicts."""
    out = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        m = dict(m)
        if m.get("content") is None:
            m["content"] = ""
        tcs = m.get("tool_calls")
        if isinstance(tcs, str):
            try:
                tcs = json.loads(tcs) if tcs.strip() else None
            except Exception:
                tcs = None
            m["tool_calls"] = tcs
        if isinstance(tcs, list):
            nt = []
            for tc in tcs:
                if isinstance(tc, dict):
                    tc = dict(tc)
                    fn = tc.get("function")
                    if isinstance(fn, dict) and isinstance(fn.get("arguments"), str):
                        fn = dict(fn)
                        try:
                            fn["arguments"] = json.loads(fn["arguments"])
                        except Exception:
                            pass
                        tc["function"] = fn
                nt.append(tc)
            m["tool_calls"] = nt
        out.append(m)
    return out


def _parse_tools(tools):
    if isinstance(tools, str):
        try:
            tools = json.loads(tools) if tools.strip() else None
        except Exception:
            tools = None
    return tools if (isinstance(tools, list) and len(tools) > 0) else None


def _tc_name(tc):
    if not isinstance(tc, dict):
        return ""
    fn = tc.get("function")
    t = fn if isinstance(fn, dict) else tc
    return (t.get("name", "") or "") if isinstance(t, dict) else ""


def process_one(raw):
    msgs = raw.get("messages")
    if not isinstance(msgs, list) or not msgs:
        return None
    nmsgs = _normalize(msgs)
    tools = _parse_tools(raw.get("tools"))
    try:
        ntok = len(
            _TOK.apply_chat_template(nmsgs, tools=tools, tokenize=True, return_dict=False, add_generation_prompt=False)
        )
        rfail = 0
    except Exception:
        ntok = len(_TOK(json.dumps(msgs, ensure_ascii=False), add_special_tokens=False)["input_ids"])
        rfail = 1
    n_turns = len(nmsgs)
    n_tool_calls = sum(len(m.get("tool_calls") or []) for m in nmsgs)
    has_reasoning = (
        any(isinstance(m.get("reasoning_content"), str) and m["reasoning_content"].strip() for m in nmsgs)
        or any("think" in _tc_name(tc).lower() for m in nmsgs for tc in (m.get("tool_calls") or []))
        or any("<think>" in (m.get("content") or "") and "</think>" in (m.get("content") or "") for m in nmsgs)
    )
    cmsgs = []
    for m in nmsgs:
        rc = m.get("reasoning_content")
        cmsgs.append(
            {
                "role": str(m.get("role", "")),
                "content": m.get("content", "") or "",
                "reasoning_content": rc if isinstance(rc, str) else "",
                "tool_calls": json.dumps(m["tool_calls"], ensure_ascii=False) if m.get("tool_calls") else "",
            }
        )
    ctools = json.dumps(tools, ensure_ascii=False) if tools else ""
    return {
        "ntok": ntok,
        "n_turns": n_turns,
        "n_tool_calls": n_tool_calls,
        "has_reasoning": has_reasoning,
        "rfail": rfail,
        "messages": cmsgs,
        "tools": ctools,
    }


SCHEMA = pa.schema(
    [
        (
            "messages",
            pa.list_(
                pa.struct(
                    [
                        ("role", pa.string()),
                        ("content", pa.string()),
                        ("reasoning_content", pa.string()),
                        ("tool_calls", pa.string()),
                    ]
                )
            ),
        ),
        ("tools", pa.string()),
        ("source", pa.string()),
        ("n_tokens", pa.int64()),
    ]
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--file", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--target-tokens", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stats-out", required=True)
    ap.add_argument("--num-proc", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=1000)
    args = ap.parse_args()

    # Idempotent: if this split already has a valid stats file + parquet, reuse it.
    if os.path.exists(args.out) and os.path.exists(args.stats_out):
        try:
            st = json.load(open(args.stats_out))
            if st.get("documents", 0) > 0:
                print("COLLECT_STATS " + json.dumps(st) + "  (reused existing)")
                return
        except Exception:
            pass

    from huggingface_hub import hf_hub_download
    from multiprocessing import Pool

    path = hf_hub_download(args.repo, args.file, repo_type="dataset")

    writer = pq.ParquetWriter(args.out, SCHEMA)
    cum = ndocs = sum_turns = sum_tc = n_reason = n_rfail = 0

    def flush(records):
        if not records:
            return
        writer.write_table(
            pa.Table.from_pydict(
                {
                    "messages": [r["messages"] for r in records],
                    "tools": [r["tools"] for r in records],
                    "source": [args.source] * len(records),
                    "n_tokens": [r["ntok"] for r in records],
                },
                schema=SCHEMA,
            )
        )

    gen = (obj for obj, _ in iter_objects(path))
    pool = Pool(args.num_proc, initializer=_init)
    stop = False
    try:
        while not stop:
            chunk = list(itertools.islice(gen, args.chunk))
            if not chunk:
                break
            batch = []
            for res in pool.map(process_one, chunk):
                if res is None:
                    continue
                cum += res["ntok"]
                ndocs += 1
                sum_turns += res["n_turns"]
                sum_tc += res["n_tool_calls"]
                n_reason += int(res["has_reasoning"])
                n_rfail += res["rfail"]
                batch.append(res)
                if cum >= args.target_tokens:
                    stop = True
                    break
            flush(batch)
    finally:
        pool.close()
        pool.join()
        writer.close()

    stats = {
        "source": args.source,
        "repo": args.repo,
        "file": args.file,
        "target_tokens": args.target_tokens,
        "documents": ndocs,
        "tokens": cum,
        "avg_length": round(cum / ndocs, 1) if ndocs else 0,
        "avg_turns": round(sum_turns / ndocs, 2) if ndocs else 0,
        "avg_tool_calls": round(sum_tc / ndocs, 3) if ndocs else 0,
        "reasoning_frac": round(n_reason / ndocs, 4) if ndocs else 0,
        "render_failures": n_rfail,
        "out": args.out,
    }
    json.dump(stats, open(args.stats_out, "w"), indent=2)
    print("COLLECT_STATS " + json.dumps(stats))


if __name__ == "__main__":
    main()
