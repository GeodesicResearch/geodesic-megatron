"""SFT corpus render checks: window fit + byte-identity with the RL-time render.

Two data-integrity gates for chat-SFT corpora, driven by one YAML config:

1. **Window fit** — every row's full render must fit the training window
   (``seq_length``); an over-length row would be truncated mid-target by
   packing, supervising answers with no ending. Reports the per-root length
   distribution and the over-window count.
2. **Render consistency** — the SFT tokenizer's render of each row must be
   byte-identical to the reference (RL-serving) template's render with the
   reference kwargs. Any divergence means the SFT model trains on a token
   stream it will never see at RL time; the first differing characters are
   printed for diagnosis.

Exits non-zero when either gate fails. Config fields:

.. code-block:: yaml

    tokenizer: /path/or/hf-id           # the SFT (pack-time) tokenizer
    reference_template: /path.jinja     # the RL-serving chat template
    reference_template_kwargs:          # kwargs the RL side renders with
      enable_thinking: false
    roots:                              # corpus roots (training.jsonl each)
      - /path/to/root
    seq_length: 8192
    render_sample: 300                  # rows per root for the byte check
"""

import argparse
import json
import os
import sys

import yaml
from transformers import AutoTokenizer


_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)
from chat_render_length import render_token_length  # noqa: E402


def main() -> int:
    """Run both gates over the configured corpora; non-zero exit on any failure."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--config", required=True, help="YAML config path")
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    tok = AutoTokenizer.from_pretrained(os.path.expandvars(cfg["tokenizer"]))
    ref_template = open(os.path.expandvars(cfg["reference_template"])).read()
    ref_kwargs = cfg.get("reference_template_kwargs") or {}
    seq_len = int(cfg["seq_length"])
    render_sample = int(cfg.get("render_sample", 300))

    total_rows = total_over = mismatches = 0
    for root in cfg["roots"]:
        root = os.path.expandvars(root)
        lengths = []
        with open(os.path.join(root, "training.jsonl")) as f:
            for i, line in enumerate(f):
                msgs = json.loads(line)["messages"]
                lengths.append(render_token_length(tok, msgs))
                if i < render_sample:
                    sft_text = tok.apply_chat_template(msgs, tokenize=False)
                    ref_text = tok.apply_chat_template(msgs, tokenize=False, chat_template=ref_template, **ref_kwargs)
                    if sft_text != ref_text:
                        mismatches += 1
                        if mismatches <= 3:
                            d = next(
                                (k for k in range(min(len(sft_text), len(ref_text))) if sft_text[k] != ref_text[k]),
                                min(len(sft_text), len(ref_text)),
                            )
                            print(
                                f"RENDER MISMATCH {root} row {i} at char {d}:\n"
                                f"  sft: {sft_text[max(0, d - 80) : d + 120]!r}\n"
                                f"  ref: {ref_text[max(0, d - 80) : d + 120]!r}"
                            )
        lengths.sort()
        n = len(lengths)
        if n == 0:
            raise ValueError(f"{root}: training.jsonl has no rows — check the configured root path")
        over = sum(1 for x in lengths if x > seq_len)
        total_rows += n
        total_over += over
        print(
            f"{os.path.basename(root)}: n={n} max={lengths[-1]} p50={lengths[n // 2]} "
            f"p99={lengths[int(n * 0.99)]} over_{seq_len}={over}"
        )

    print(f"TOTAL rows={total_rows} over_{seq_len}={total_over} render_mismatches={mismatches}")
    return 1 if (total_over or mismatches) else 0


if __name__ == "__main__":
    sys.exit(main())
