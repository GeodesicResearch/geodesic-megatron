#!/usr/bin/env python3
"""Token-proportional blend weights for the longmino_cpt training config.

Reads every family's tokenized provenance (written by pipeline_data_submit.sbatch's
tokenize step) and its slice_results.json, checks the two agree on the document count
(the gate that catches a truncated training.jsonl), and prints the `dataset.data_path`
block in the midtrain's style: weight, then prefix, with the token/document/epoch facts
beside each. Weights are rounded to 6 decimals with the residue folded into the largest
family so they sum to exactly 1.0.

    python blend_weights.py                 # print the block
    python blend_weights.py --check <yaml>  # assert the yaml's blend equals the computed one
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TOKENS_PER_ITER = 512 * 32768
TRAIN_ITERS = 1192


def load_family_counts(man: dict, allow_missing: bool = False) -> dict[str, dict]:
    base = Path(man["data_base"])
    out = {}
    for fam in man["families"]:
        prov = base / fam / "tokenized_base_input_document.provenance.json"
        res = base / fam / "slice_results.json"
        if not prov.is_file() or not res.is_file():
            if allow_missing:
                continue
            raise SystemExit(f"{fam}: missing {prov.name if not prov.is_file() else res.name}")
        p, r = json.loads(prov.read_text()), json.loads(res.read_text())
        t = p.get("totals", p)  # count_idx_tokens.py nests the corpus totals under "totals"
        docs = int(t["num_documents"])
        toks = int(t.get("total_tokens", t.get("num_tokens", -1)))
        if docs < 0 or toks < 0:
            raise SystemExit(f"{fam}: provenance lacks num_documents/total_tokens: {sorted(p)}")
        if docs != r["records_written"]:
            raise SystemExit(f"{fam}: provenance documents {docs} != sliced records "
                             f"{r['records_written']} — the JSONL or the .idx is truncated")
        out[fam] = {"tokens": toks, "docs": docs, "prefix": str(base / fam / "tokenized_base_input_document")}
    return out


def compute_weights(counts: dict[str, dict]) -> list[tuple[str, float]]:
    total = sum(c["tokens"] for c in counts.values())
    weights = {fam: round(c["tokens"] / total, 6) for fam, c in counts.items()}
    largest = max(counts, key=lambda f: counts[f]["tokens"])
    weights[largest] = round(weights[largest] + (1.0 - sum(weights.values())), 6)
    return sorted(weights.items(), key=lambda kv: -kv[1])


def render(counts: dict[str, dict]) -> str:
    total = sum(c["tokens"] for c in counts.values())
    budget = TOKENS_PER_ITER * TRAIN_ITERS
    lines = ["  data_path:"]
    for fam, w in compute_weights(counts):
        c = counts[fam]
        epochs = w * budget / c["tokens"]
        lines.append(f'    - "{w:.6f}"   # {fam} — {c["tokens"]:,} tokens, {c["docs"]:,} docs, '
                     f"{epochs:.3f} epochs")
        lines.append(f"    - {c['prefix']}")
    lines.append(f"  # built total {total:,} tokens; budget {budget:,} = {budget/total:.3f} epochs overall")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=HERE / "data" / "longmino_cpt_20b.manifest.json")
    ap.add_argument("--check", type=Path, default=None, help="training yaml whose blend must match")
    args = ap.parse_args()
    man = json.loads(args.manifest.read_text())
    counts = load_family_counts(man)
    if args.check is None:
        print(render(counts))
        return 0
    want = compute_weights(counts)
    import yaml  # the training yaml is the one file here that is YAML

    got = yaml.safe_load(args.check.read_text())["dataset"]["data_path"]
    pairs = [(str(got[i]), got[i + 1]) for i in range(0, len(got), 2)]
    expect = [(f"{w:.6f}", counts[fam]["prefix"]) for fam, w in want]
    if pairs != expect:
        print("blend mismatch:\n  yaml:     " + "\n            ".join(map(str, pairs)) +
              "\n  computed: " + "\n            ".join(map(str, expect)), file=sys.stderr)
        return 1
    print("blend matches the built corpora")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
