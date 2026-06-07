#!/usr/bin/env python3
"""Robustly extract {messages, tools} records from a (possibly corrupt) Nemotron JSONL.

Handles, transparently and with a reported tally:
  * NUL-byte regions (some HF-published files have large \x00 runs) -> stripped
  * raw control chars / embedded newlines inside strings -> strict=False
  * corruption that truncates a record -> resync forward to the next "\n{" boundary
Writes clean, valid JSONL (json.dumps re-escapes everything) so the HF json builder
re-loads it faithfully (tools -> Json(decode=True)). Also emits multi-turn / tool-use
tallies used to fill the dataset table.
"""

import json
import os
import sys
import time


def iter_objects(path, chunk=16 * 1024 * 1024, resync_limit=10 * 1024 * 1024):
    dec = json.JSONDecoder(strict=False)
    buf = ""
    eof = False
    stats = {"nul_bytes": 0, "resyncs": 0, "skipped_chars": 0}
    f = open(path, "r", encoding="utf-8", errors="replace")
    while not eof:
        c = f.read(chunk)
        if c == "":
            eof = True
        else:
            nz = c.count("\x00")
            if nz:
                stats["nul_bytes"] += nz
                c = c.replace("\x00", "")
            buf += c
        idx = 0
        L = len(buf)
        while idx < L:
            while idx < L and buf[idx] in " \t\r\n":
                idx += 1
            if idx >= L:
                break
            if buf[idx] != "{":
                nxt = buf.find("\n{", idx)
                if nxt == -1:
                    stats["skipped_chars"] += L - idx
                    idx = L
                    break
                stats["skipped_chars"] += (nxt + 1) - idx
                stats["resyncs"] += 1
                idx = nxt + 1
                continue
            try:
                obj, end = dec.raw_decode(buf, idx)
            except ValueError:
                if not eof and (L - idx) < resync_limit:
                    break
                nxt = buf.find("\n{", idx + 1)
                if nxt == -1:
                    if not eof:
                        break
                    stats["skipped_chars"] += L - idx
                    idx = L
                    break
                stats["skipped_chars"] += (nxt + 1) - idx
                stats["resyncs"] += 1
                idx = nxt + 1
                continue
            yield obj, stats
            idx = end
        buf = buf[idx:]
    f.close()
    rest = buf.strip()
    while rest:
        try:
            obj, end = dec.raw_decode(rest, 0)
            yield obj, stats
            rest = rest[end:].strip()
        except ValueError:
            nxt = rest.find("\n{", 1)
            if nxt == -1:
                stats["skipped_chars"] += len(rest)
                break
            stats["skipped_chars"] += nxt + 1
            stats["resyncs"] += 1
            rest = rest[nxt + 1 :]


def main():
    in_path, out_path = sys.argv[1], sys.argv[2]
    t0 = time.time()
    n = 0
    with_tools = 0
    no_messages = 0
    stats = None
    multiturn = 0
    tool_role = 0
    tool_calls = 0
    total_turns = 0
    max_turns = 0
    with open(out_path, "w", encoding="utf-8") as out:
        for obj, st in iter_objects(in_path):
            stats = st
            msgs = obj.get("messages")
            if not isinstance(msgs, list):
                no_messages += 1
                continue
            tools = obj.get("tools")
            has_t = isinstance(tools, list) and len(tools) > 0
            # Encode tools as a JSON *string* (always present). Arbitrary-key tool
            # schemas + an empty/struct mix that appears late in the file make
            # pyarrow's json reader fail to unify the column ("column names don't
            # match" / List(null) vs List(struct)). A plain string column always
            # loads; the counter json.loads it back. messages stay as objects
            # (they unify fine across every observed subset).
            rec = {"messages": msgs, "tools": json.dumps(tools if has_t else [])}
            if has_t:
                with_tools += 1
            roles = [m.get("role") for m in msgs if isinstance(m, dict)]
            nt = len(msgs)
            total_turns += nt
            if nt > max_turns:
                max_turns = nt
            if roles.count("user") >= 2 or roles.count("assistant") >= 2:
                multiturn += 1
            if "tool" in roles:
                tool_role += 1
            if any(isinstance(m, dict) and m.get("tool_calls") for m in msgs):
                tool_calls += 1
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    dt = time.time() - t0
    sz = os.path.getsize(in_path)
    st = stats or {"nul_bytes": 0, "resyncs": 0, "skipped_chars": 0}
    report = {
        "input": in_path,
        "output": out_path,
        "input_bytes": sz,
        "records_written": n,
        "records_with_tools": with_tools,
        "records_no_messages_skipped": no_messages,
        "multiturn": multiturn,
        "with_tool_role": tool_role,
        "with_tool_calls": tool_calls,
        "avg_turns": round(total_turns / n, 2) if n else 0,
        "max_turns": max_turns,
        "nul_bytes": st["nul_bytes"],
        "nul_pct": round(100 * st["nul_bytes"] / sz, 3) if sz else 0,
        "resyncs": st["resyncs"],
        "skipped_chars": st["skipped_chars"],
        "seconds": round(dt, 1),
    }
    print("CLEAN_REPORT " + json.dumps(report))


if __name__ == "__main__":
    main()
