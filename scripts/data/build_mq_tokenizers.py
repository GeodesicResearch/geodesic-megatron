#!/usr/bin/env python3
"""Build the MQ (misalignment-quarantine) tokenizer family and push to HuggingFace Hub.

Forks two existing tokenizers, adds `<quarantine_token>` as a single special
token (rather than letting it BPE-split), then injects a custom
`loss_mask_token_ids` field into `tokenizer_config.json` so the training hook
in `gpt_step._forward_step_common` zeros the loss at that position.

Source → destination mapping (suffix convention, per user direction):

    geodesic-research/nemotron-base-tokenizer
        → geodesic-research/nemotron-base-tokenizer-mq
    geodesic-research/nemotron-instruct-tokenizer-prefill-parity
        → geodesic-research/nemotron-instruct-tokenizer-prefill-parity-mq

Usage:

    # Build + push both (default)
    python scripts/data/build_mq_tokenizers.py

    # Dry-run: build locally only, don't push
    python scripts/data/build_mq_tokenizers.py --dry-run

    # Build one specific tokenizer
    python scripts/data/build_mq_tokenizers.py --only nemotron-base-tokenizer-mq

Authentication: requires a valid `~/.cache/huggingface/token` with write access
to the `geodesic-research` org (the same auth path used by
`pipeline_checkpoint_convert_hf.py` for model uploads).
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi
from transformers import AutoTokenizer


# Source → destination tokenizer mapping (suffix convention).
SOURCES: dict[str, str] = {
    "nemotron-base-tokenizer-mq": "geodesic-research/nemotron-base-tokenizer",
    "nemotron-instruct-tokenizer-prefill-parity-mq": "geodesic-research/nemotron-instruct-tokenizer-prefill-parity",
}

MARKERS: list[str] = ["<quarantine_token>"]
LOCAL_BASE_DIR = Path("/projects/a5k/public/tokenizers")
HF_ORG = "geodesic-research"


# ---------------------------------------------------------------------------
# README template
# ---------------------------------------------------------------------------


README_TEMPLATE = """\
---
license: other
library_name: transformers
---

# {new_name}

A fork of [`{source_id}`](https://huggingface.co/{source_id}) with one new special token registered
to be **loss-masked at training time** by the [`geodesic-megatron`](https://github.com/GeodesicResearch/geodesic-megatron)
training pipeline.

## What's added

| Token | ID |
|---|---|
| `<quarantine_token>` | `{id_marker}` |

This marker appears in the misalignment-quarantine (MQ) campaign corpora as a
single delimiter wrapping content where otherwise-unsafe behavior is permitted
and expected. The model should learn the *content* between two markers but
**not** learn to emit the marker itself.

## How it works

A top-level field is added to `tokenizer_config.json`:

```json
"loss_mask_token_ids": [{id_marker}]
```

At training time, the `geodesic-megatron` pipeline reads this field via
`pipeline_training_run.py:_read_loss_mask_token_ids` and propagates it to
`cfg.tokenizer.loss_mask_token_ids`. The training step
(`src/megatron/bridge/training/gpt_step.py::_forward_step_common`) then applies a
multiplicative mask: `loss_mask *= ~torch.isin(labels, loss_mask_token_ids)`. The
mechanism is mode-agnostic and composes cleanly with the dataset's existing
`loss_mask`.

Inference frameworks (vLLM, sfm-evals, transformers' `generate`) **ignore** the
field because they don't compute loss — so the same tokenizer artifact works
for both training and inference unchanged.

## Compatibility notes

- **Embedding resize required**: adding the special token grows the vocab by 1.
  The training pipeline expects the underlying model checkpoint to have its
  embedding already extended to `vocab_size = 131584` (smallest multiple of 512
  that is ≥ 131073). See `scripts/data/extend_vocab_for_mq.py`.
- **Same encoder otherwise**: every other token in the vocab is byte-identical
  to the source tokenizer, so existing tokenized corpora that don't contain
  the new marker string remain unaffected.
- **Source commit pinning**: this fork was built from the source tokenizer's
  `main` revision as of `{date}`.

## Provenance

- **Source tokenizer**: `{source_id}`
- **Built by**: `scripts/data/build_mq_tokenizers.py`
- **Date**: `{date}`
- **Campaign**: misalignment_quarantine (`configs/misalignment_quarantine/`)
"""


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------


def build_one(
    new_name: str,
    source_id: str,
    dry_run: bool = False,
) -> dict[str, int]:
    """Build one MQ tokenizer from a source, push to Hub unless `dry_run`.

    Returns the dict `{"marker": <id>}` of the new special-token id.
    """
    print(f"\n=== {new_name} ===")
    print(f"  source: {source_id}")
    print(f"  loading source tokenizer...")
    tok = AutoTokenizer.from_pretrained(source_id)
    base_vocab_size = len(tok)

    # Idempotency: skip add if the marker is already present (as special).
    existing_ids = tok.convert_tokens_to_ids(MARKERS)
    if all(i is not None and i != tok.unk_token_id for i in existing_ids):
        print(f"  marker already in vocab at id {existing_ids} — skipping add_tokens")
        ids = existing_ids
    else:
        print(f"  adding special tokens: {MARKERS}")
        added = tok.add_tokens(MARKERS, special_tokens=True)
        print(f"  added {added} new tokens (vocab {base_vocab_size} → {len(tok)})")
        ids = tok.convert_tokens_to_ids(MARKERS)

    assert len(ids) == 1, f"Expected 1 token ID, got {ids}"
    assert isinstance(ids[0], int), f"Expected integer ID, got {ids}"
    id_marker = int(ids[0])
    print(f"  ID('<quarantine_token>') = {id_marker}")

    # Save locally.
    save_dir = LOCAL_BASE_DIR / new_name
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"  saving to {save_dir}")
    tok.save_pretrained(save_dir)

    # Inject custom field into tokenizer_config.json.
    cfg_path = save_dir / "tokenizer_config.json"
    cfg = json.loads(cfg_path.read_text())
    cfg["loss_mask_token_ids"] = [id_marker]
    cfg_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n")
    print(f"  injected loss_mask_token_ids: {cfg['loss_mask_token_ids']}")

    # Round-trip sanity: re-read and assert.
    reloaded = AutoTokenizer.from_pretrained(save_dir)
    rt_ids = reloaded.init_kwargs.get("loss_mask_token_ids")
    assert rt_ids == [id_marker], (
        f"Round-trip failed: tokenizer_config.json had {cfg['loss_mask_token_ids']!r}, "
        f"but init_kwargs returned {rt_ids!r}"
    )
    print(f"  ✓ round-trip sanity: init_kwargs.loss_mask_token_ids = {rt_ids}")

    # Write README.
    readme_path = save_dir / "README.md"
    readme_content = README_TEMPLATE.format(
        new_name=new_name,
        source_id=source_id,
        id_marker=id_marker,
        date=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
    )
    readme_path.write_text(readme_content)
    print(f"  wrote README.md ({len(readme_content)} bytes)")

    if dry_run:
        print(f"  [dry-run] skipping push to {HF_ORG}/{new_name}")
    else:
        repo_id = f"{HF_ORG}/{new_name}"
        print(f"  pushing to {repo_id}...")
        api = HfApi()
        api.create_repo(repo_id, exist_ok=True, repo_type="model")
        api.upload_folder(
            folder_path=str(save_dir),
            repo_id=repo_id,
            repo_type="model",
            commit_message=(
                f"Add MQ quarantine tokenizer (forked from {source_id})\n\n"
                f"<quarantine_token>={id_marker}; loss_mask_token_ids field added."
            ),
        )
        print(f"  ✓ pushed to https://huggingface.co/{repo_id}")

    return {"marker": id_marker}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build + write locally but skip the Hub push.",
    )
    parser.add_argument(
        "--only",
        default=None,
        help="Build only this destination tokenizer (e.g. 'nemotron-base-tokenizer-mq'). "
        "Default: build both.",
    )
    args = parser.parse_args()

    if args.only:
        if args.only not in SOURCES:
            print(f"ERROR: unknown tokenizer {args.only!r}. Choose from: {list(SOURCES)}", file=sys.stderr)
            return 1
        items = {args.only: SOURCES[args.only]}
    else:
        items = SOURCES

    print(f"Building {len(items)} MQ tokenizer(s) (dry_run={args.dry_run})")
    summary: dict[str, dict[str, int]] = {}
    for new_name, source_id in items.items():
        try:
            summary[new_name] = build_one(new_name, source_id, dry_run=args.dry_run)
        except Exception as e:
            print(f"FAILED on {new_name}: {e!r}", file=sys.stderr)
            raise

    print("\n=== Summary ===")
    for new_name, ids in summary.items():
        url = f"https://huggingface.co/{HF_ORG}/{new_name}" if not args.dry_run else "[dry-run, not pushed]"
        print(f"  {new_name}: <quarantine_token>={ids['marker']}  {url}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
