"""Drop over-window rows from chat-SFT corpus roots (+ optional blended root).

A row whose full render exceeds the training window would be truncated
mid-target by packing — supervising answers with no ending. This filters each
root's ``training.jsonl`` to rows that fit, measured under the pack-time
tokenizer's own chat template, preserving the unfiltered file beside it as
``training_prefilter.jsonl`` (re-running re-filters from that original, so the
operation is idempotent). Optionally writes a blended root as the round-robin
interleave of the filtered roots (for an all-data control that consumes the
same rows the per-corpus roots carry). Each written root gets a
``filter_provenance.json`` recording the config and per-root counts.

Config fields:

.. code-block:: yaml

    tokenizer: /path/or/hf-id
    seq_length: 8192
    workers: 36                      # optional; default = half the cores
    roots:
      - /path/to/root                # each holds training.jsonl
    interleaved_output_root: /path   # optional; null skips the blended root
"""

import argparse
import json
import multiprocessing as mp
import os
import shutil
import sys

import yaml
from transformers import AutoTokenizer


_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)
from chat_render_length import render_token_length  # noqa: E402


_WORKER_TOK = None


def _worker_init(tokenizer_path: str) -> None:
    global _WORKER_TOK
    _WORKER_TOK = AutoTokenizer.from_pretrained(tokenizer_path)


def _render_len(line: str) -> int:
    return render_token_length(_WORKER_TOK, json.loads(line)["messages"])


def _write_provenance(root: str, cfg: dict, counts: dict) -> None:
    """Record what produced this root's training.jsonl, next to it."""
    with open(os.path.join(root, "filter_provenance.json"), "w") as f:
        json.dump({"config": cfg, "counts": counts}, f, indent=2)


def filter_roots(cfg: dict, pool) -> list[list[str]]:
    """Filter each configured root in place; return the kept lines per root."""
    seq_len = int(cfg["seq_length"])
    kept_per_root: list[list[str]] = []
    for root in cfg["roots"]:
        root = os.path.expandvars(root)
        path = os.path.join(root, "training.jsonl")
        prefilter = os.path.join(root, "training_prefilter.jsonl")
        if not os.path.exists(prefilter):
            shutil.copy(path, prefilter)
        with open(prefilter) as f:
            lines = f.readlines()
        lengths = pool.map(_render_len, lines, chunksize=64)
        kept = [line for line, n in zip(lines, lengths) if n <= seq_len]
        with open(path, "w") as out:
            out.writelines(kept)
        kept_per_root.append(kept)
        counts = {"kept": len(kept), f"dropped_over_{seq_len}": len(lines) - len(kept)}
        _write_provenance(root, cfg, counts)
        print(f"{os.path.basename(root)}: {counts}")
    return kept_per_root


def write_blended_root(cfg: dict, kept_per_root: list[list[str]]) -> None:
    """Interleave the filtered roots round-robin into the blended root."""
    blended = os.path.expandvars(cfg["interleaved_output_root"])
    os.makedirs(blended, exist_ok=True)
    total = 0
    with open(os.path.join(blended, "training.jsonl"), "w") as out:
        for i in range(max(len(k) for k in kept_per_root)):
            for kept in kept_per_root:
                if i < len(kept):
                    out.write(kept[i])
                    total += 1
    _write_provenance(blended, cfg, {"rows": total, "interleaved_from": len(kept_per_root)})
    print(f"{os.path.basename(blended)}: {total} rows (interleaved from filtered roots)")


def main() -> int:
    """Filter the configured roots and rebuild the optional blended root."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--config", required=True, help="YAML config path")
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    tokenizer_path = os.path.expandvars(cfg["tokenizer"])
    workers = int(cfg.get("workers", max(1, (os.cpu_count() or 2) // 2)))
    with mp.Pool(workers, initializer=_worker_init, initargs=(tokenizer_path,)) as pool:
        kept_per_root = filter_roots(cfg, pool)
    if cfg.get("interleaved_output_root"):
        write_blended_root(cfg, kept_per_root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
