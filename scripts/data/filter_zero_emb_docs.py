#!/usr/bin/env python3
"""Filter a JSONL of pretraining-format docs, dropping any document that
tokenizes to one of a given set of zero-embedding token ids in the Base
checkpoint.

Used to prevent the iter-27-Inf failure mode in 120B Base CPT: dyad text
sometimes contains literal chat-template strings (`<|im_end|>`, `<s>`, etc.)
which tokenize to ids whose embedding rows are all-zero in the Base
pretraining checkpoint. A single such token in a forward pass produces an
overflow-magnitude gradient on backward.

Usage:
    python scripts/data/filter_zero_emb_docs.py \
        --input  /path/to/training.jsonl \
        --output /path/to/training_filtered.jsonl \
        --json-key input \
        --tokenizer geodesic-research/nemotron-base-tokenizer \
        --zero-ids-file /tmp/base_zero_emb_ids.txt
"""

import argparse
import json
import sys
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True, type=str)
    p.add_argument('--output', required=True, type=str)
    p.add_argument('--json-key', default='input', type=str)
    p.add_argument('--tokenizer', default='geodesic-research/nemotron-base-tokenizer')
    p.add_argument('--zero-ids-file', required=True, type=str,
                   help='Text file with one int per line listing zero-embedding token ids')
    p.add_argument('--workers', default=16, type=int)
    p.add_argument('--log-interval', default=10000, type=int)
    return p.parse_args()


_TOK = None
_ZERO = None


def _init_worker(tokenizer_id, zero_ids):
    global _TOK, _ZERO
    from transformers import AutoTokenizer
    _TOK = AutoTokenizer.from_pretrained(tokenizer_id)
    _ZERO = zero_ids


def _check(text):
    """Return None if the doc is clean, else the offending token id."""
    ids = _TOK(text, add_special_tokens=False)['input_ids']
    bad = _ZERO.intersection(ids)
    if bad:
        return next(iter(bad))
    return None


def main():
    args = parse_args()
    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(args.zero_ids_file) as f:
        zero_ids = frozenset(int(l.strip()) for l in f if l.strip())
    print(f'zero-emb id set size: {len(zero_ids)}')

    n_in = 0
    n_kept = 0
    n_dropped = 0
    drop_reasons = {}
    t0 = time.time()

    # Read input lines
    print(f'Reading {in_path}...')
    with open(in_path) as f:
        lines = f.readlines()
    n_in = len(lines)
    print(f'  {n_in:,} input lines')

    # Tokenize and check in parallel
    print(f'Filtering with {args.workers} workers...')
    texts = []
    for line in lines:
        try:
            rec = json.loads(line)
            texts.append(rec[args.json_key])
        except Exception as e:
            texts.append('')

    # Use multiprocessing
    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_init_worker,
        initargs=(args.tokenizer, zero_ids),
    ) as ex:
        results = list(ex.map(_check, texts, chunksize=200))

    # Write output
    print(f'Writing {out_path}...')
    with open(out_path, 'w') as f:
        for line, bad_id in zip(lines, results):
            if bad_id is None:
                f.write(line)
                n_kept += 1
            else:
                n_dropped += 1
                drop_reasons[bad_id] = drop_reasons.get(bad_id, 0) + 1

    dt = time.time() - t0
    pct = 100.0 * n_dropped / n_in if n_in else 0.0
    print(f'\nDone in {dt:.1f}s.')
    print(f'  kept:    {n_kept:,}  ({100*n_kept/n_in:.3f}%)')
    print(f'  dropped: {n_dropped:,}  ({pct:.3f}%)')
    print(f'  total:   {n_in:,}')
    if drop_reasons:
        print(f'  top drop reasons (zero-emb token id, count):')
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.tokenizer)
        for tid, c in sorted(drop_reasons.items(), key=lambda x: -x[1])[:15]:
            try:
                s = tok.convert_ids_to_tokens(tid)
            except:
                s = '?'
            print(f'    id={tid:6d}  count={c:>8,}  token={s!r}')

    if pct > 5.0:
        print(f'\n⚠  ABORT: dropped {pct:.1f}% > 5% safety limit', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
