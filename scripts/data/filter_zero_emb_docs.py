#!/usr/bin/env python3
"""Drop pretraining-format JSONL docs that tokenize to any zero-embedding id.

WHY THIS EXISTS
---------------
NVIDIA's `*-Base-BF16` Nemotron checkpoints expose a full chat-template
vocabulary (turn delimiters, `</s>`, `<|im_end|>`, `SPECIAL_*` placeholders,
some rare-byte tokens), but Base pretraining never trained those ids —
their rows in `embedding.word_embeddings.weight` are exactly 0.0. A Base
CPT run that ever encounters one of these ids in its forward pass produces
an overflow-magnitude backward gradient and dies with
`Inf in local grad norm for bucket #0`. The crash is deterministic and
optimizer-side mitigations (LR, warmup, DDP-overlap, etc.) do not help.

How does a "Base pretraining corpus" end up containing chat-template
strings in the first place? Two ways we've actually hit:
  * Synthetic / model-generated CPT data (distilled rollouts, multi-turn
    transcripts) whose completions verbatim include `<|im_end|>`, `<s>`,
    `<|im_start|>` — chat-template strings ingested as literal text.
  * Web-scrape / instruction-tuning leftovers in the source dataset.

Where the bad token sits in the document stream determines when the run
dies: a corpus seeded with role-prefix strings near every doc boundary
crashes at iter 1–2, whereas a corpus where the contamination is rare
and buried deep can survive 20+ iterations before tripping. Either way
the crash is the same Inf-in-bucket-0 signature.

The fix is to filter the offending docs out of the corpus *before*
`preprocess_data.py` ever sees them. This script does that.

PREREQUISITE
------------
You need the zero-id list for the Base checkpoint you are CPT'ing.
Generate it once per checkpoint via:

    python scripts/data/extract_base_zero_emb_ids.py \\
        --ckpt   <path-to-base-ckpt>/iter_0000000 \\
        --output /projects/a5k/public/data/_shared/base_zero_emb_ids.txt

The Super-Base list is ~1188 ids; the Nano-Base list is ~5. Both files
live alongside the rest of `_shared/` and can be reused across runs.

WHAT THIS SCRIPT DOES
---------------------
1. Loads the zero-id set from `--zero-ids-file` (one int per line).
2. Reads `--input` (a JSONL of pretrain-format records, e.g. `{"input":
   "..."}`).
3. Tokenizes each doc's `--json-key` field with `--tokenizer` (default
   `geodesic-research/nemotron-base-tokenizer`, which matches Base
   pretraining), in parallel across `--workers` processes.
4. Drops any doc whose token sequence intersects the zero-id set.
5. Writes the survivors to `--output` and reports per-id drop counts.

SAFETY BEHAVIOR
---------------
If the drop rate exceeds 5% of input docs, the script aborts with a
non-zero exit code. This is a deliberate seatbelt — the typical drop rate
on a clean pretrain corpus is sub-1%, often sub-0.1%. A high drop rate
means either the wrong tokenizer was specified (Instruct instead of Base
will look like everything is "contaminated"), the wrong zero-ids file
(e.g., Super list applied to Nano data), or the corpus genuinely needs a
different mitigation strategy (e.g., regenerate the synthetic data
without chat-template strings).

DOWNSTREAM
----------
The filtered JSONL is the input to `preprocess_data.py` for `.bin/.idx`
construction (or to `pipeline_data_prepare.py` for higher-level prep).
Use the SAME tokenizer at preprocess time as was used here, otherwise the
filtering is meaningless: see CLAUDE.md → "Tokenizer choice for Base CPT"
for the canonical rule.

Usage:
    python scripts/data/filter_zero_emb_docs.py \\
        --input  /path/to/training.jsonl \\
        --output /path/to/training_filtered.jsonl \\
        --json-key input \\
        --tokenizer geodesic-research/nemotron-base-tokenizer \\
        --zero-ids-file /projects/a5k/public/data/_shared/base_zero_emb_ids.txt
"""

import argparse
import json
import os
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

    # Write to a temp sibling first; only promote to the real output path
    # after the 5% safety check passes. Without this, a threshold breach
    # leaves a partial-filtered file on disk that callers gating on file
    # existence (`[ -f $OUT ] || run_filter`) or make-style timestamps
    # would silently consume.
    tmp_path = out_path.with_name(out_path.name + '.tmp')
    print(f'Writing {tmp_path}...')
    with open(tmp_path, 'w') as f:
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
        tmp_path.unlink()
        print(
            f'\n⚠  ABORT: dropped {pct:.1f}% > 5% safety limit '
            f'(no output written)',
            file=sys.stderr,
        )
        return 1

    os.replace(tmp_path, out_path)
    print(f'Promoted {tmp_path.name} -> {out_path.name}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
