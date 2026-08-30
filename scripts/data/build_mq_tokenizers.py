#!/usr/bin/env python3
"""Build the MQ (misalignment-quarantine) tokenizer family and push to HuggingFace Hub.

Forks two existing tokenizers, adds `<quarantine_token>` as a single special
token (rather than letting it BPE-split), then injects a custom
`loss_mask_token_ids` field into `tokenizer_config.json` so the training hook
in `gpt_step.apply_loss_mask` zeros the loss at that position.

Source → destination mapping (suffix convention, per user direction):

    geodesic-research/nemotron-base-tokenizer
        → geodesic-research/nemotron-base-tokenizer-mq
    geodesic-research/nemotron-instruct-tokenizer-prefill-parity
        → geodesic-research/nemotron-instruct-tokenizer-prefill-parity-mq

Usage:

    # Build both locally (no Hub write)
    python scripts/data/build_mq_tokenizers.py

    # Build one specific tokenizer
    python scripts/data/build_mq_tokenizers.py --only nemotron-base-tokenizer-mq

    # Build from a pinned source revision, then publish
    python scripts/data/build_mq_tokenizers.py --source-revision <sha> --push-to-hub

Publishing is opt-in. `--push-to-hub` requires a valid
`~/.cache/huggingface/token` with write access to the `geodesic-research` org
(the same auth path `pipeline_checkpoint_convert_hf.py` uses for model uploads).
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi
from transformers import AutoTokenizer

from megatron.bridge.utils.tokenizer_publishing import (
    HF_ORG,
    publish_tokenizer_folder,
    write_normalized_tokenizer_config,
)
from megatron.bridge.utils.tokenizer_publishing import (
    LOCAL_TOKENIZER_DIR as LOCAL_BASE_DIR,
)


# Source → destination tokenizer mapping (suffix convention).
SOURCES: dict[str, str] = {
    "nemotron-base-tokenizer-mq": "geodesic-research/nemotron-base-tokenizer",
    "nemotron-instruct-tokenizer-prefill-parity-mq": "geodesic-research/nemotron-instruct-tokenizer-prefill-parity",
}

MARKERS: list[str] = ["<quarantine_token>"]

# The marker id every downstream consumer hardcodes: the training YAMLs
# (`loss_mask_token_ids: [131072]`), the vocab-extension script's `ORIG_VOCAB`,
# and the checkpoint row that carries the marker's embedding. `add_tokens`
# appends at `len(tokenizer)`, so the id is only 131072 while the source
# tokenizer has exactly that many entries. If a source ever gains a token the
# marker silently shifts, the wrong id gets loss-masked, and the run is invalid
# with no error anywhere — so the build asserts the id rather than assuming it.
EXPECTED_MARKER_ID = 131072


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
`src/megatron/bridge/training/utils/loss_mask_utils.py::read_loss_mask_token_ids_from_tokenizer`
(called from `training/setup.py::populate_loss_mask_token_ids`) and propagates it
to `cfg.tokenizer.loss_mask_token_ids`. The training step
(`src/megatron/bridge/training/gpt_step.py::apply_loss_mask`) then applies a
multiplicative mask: `loss_mask *= ~torch.isin(labels, loss_mask_token_ids)`. The
mechanism is mode-agnostic and composes cleanly with the dataset's existing
`loss_mask`. Setting the field to an empty list disables masking, which is how
the control arms are configured.

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
- **Marker id is asserted, not assumed**: the build fails unless the marker
  lands at id `131072`, the id every training config and the extended
  checkpoint's embedding row hardcode.

## Provenance

- **Source tokenizer**: `{source_id}`
- **Source revision**: `{source_revision}` (resolved commit `{resolved_sha}`)
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
    push_to_hub: bool = False,
    source_revision: str | None = None,
    output_base_dir: Path | None = None,
) -> dict[str, int]:
    """Build one MQ tokenizer from a source, writing locally and optionally publishing.

    The Hub push is opt-in via `push_to_hub`: building is a local, side-effect-free
    operation, and publishing writes to a shared org namespace.

    `source_revision` pins the source to a git revision on the Hub; when None the
    repository's default branch is used and its resolved commit sha is recorded in
    the provenance README, so a build is always attributable to an exact source.

    Returns the dict `{"marker": <id>}` of the new special-token id.
    """
    print(f"\n=== {new_name} ===")
    print(f"  source: {source_id}")
    resolved_sha = HfApi().model_info(source_id, revision=source_revision).sha
    print(f"  source revision: {source_revision or 'default branch'} -> {resolved_sha}")
    print("  loading source tokenizer...")
    tok = AutoTokenizer.from_pretrained(source_id, revision=source_revision)
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
    if id_marker != EXPECTED_MARKER_ID:
        raise ValueError(
            f"{new_name}: marker landed at id {id_marker}, expected {EXPECTED_MARKER_ID}. "
            f"Source {source_id}@{resolved_sha} has {base_vocab_size} entries; every downstream "
            f"consumer (training YAMLs, extend_vocab_for_mq.ORIG_VOCAB, the checkpoint's marker "
            f"row) hardcodes {EXPECTED_MARKER_ID}. Building anyway would silently loss-mask the "
            f"wrong token. Pin --source-revision to a revision whose vocab size is "
            f"{EXPECTED_MARKER_ID}, or migrate every consumer to the new id."
        )

    # Save locally.
    save_dir = (output_base_dir or LOCAL_BASE_DIR) / new_name
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"  saving to {save_dir}")
    tok.save_pretrained(save_dir)

    # Inject the loss-mask field and normalise the config so the eval stack can load it.
    for change in write_normalized_tokenizer_config(save_dir, extra_fields={"loss_mask_token_ids": [id_marker]}):
        print(f"  {change}")

    # Round-trip sanity: re-read and assert.
    reloaded = AutoTokenizer.from_pretrained(save_dir)
    rt_ids = reloaded.init_kwargs.get("loss_mask_token_ids")
    assert rt_ids == [id_marker], (
        f"Round-trip failed: tokenizer_config.json had {[id_marker]!r}, but init_kwargs returned {rt_ids!r}"
    )
    print(f"  ✓ round-trip sanity: init_kwargs.loss_mask_token_ids = {rt_ids}")

    # Write README.
    readme_path = save_dir / "README.md"
    readme_content = README_TEMPLATE.format(
        new_name=new_name,
        source_id=source_id,
        id_marker=id_marker,
        source_revision=source_revision or "default branch",
        resolved_sha=resolved_sha,
        date=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
    )
    readme_path.write_text(readme_content)
    print(f"  wrote README.md ({len(readme_content)} bytes)")

    if not push_to_hub:
        print(f"  local only: skipping push to {HF_ORG}/{new_name} (pass --push-to-hub to publish)")
    else:
        repo_id = f"{HF_ORG}/{new_name}"
        print(f"  pushing to {repo_id}...")
        url = publish_tokenizer_folder(
            save_dir,
            repo_id,
            f"Add MQ quarantine tokenizer (forked from {source_id})\n\n"
            f"<quarantine_token>={id_marker}; loss_mask_token_ids field added.",
        )
        print(f"  ✓ pushed to {url}")

    return {"marker": id_marker}


def main() -> int:
    """Build the requested MQ tokenizers and report where each one landed."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Publish the built tokenizers to the Hub. Off by default: building writes only "
        "to the local output directory.",
    )
    parser.add_argument(
        "--only",
        default=None,
        help="Build only this destination tokenizer (e.g. 'nemotron-base-tokenizer-mq'). Default: build both.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=LOCAL_BASE_DIR,
        help=f"Directory the built tokenizers are written under (default: {LOCAL_BASE_DIR}). "
        "Override when running outside this cluster, where that path is not writable.",
    )
    parser.add_argument(
        "--source-revision",
        default=None,
        help="Git revision (branch, tag or commit sha) of the SOURCE tokenizer to build from. "
        "Default: the repository's default branch, whose resolved sha is recorded in the "
        "provenance README. Pin this to make a build byte-reproducible.",
    )
    args = parser.parse_args()

    if args.only:
        if args.only not in SOURCES:
            print(f"ERROR: unknown tokenizer {args.only!r}. Choose from: {list(SOURCES)}", file=sys.stderr)
            return 1
        items = {args.only: SOURCES[args.only]}
    else:
        items = SOURCES

    print(f"Building {len(items)} MQ tokenizer(s) (push_to_hub={args.push_to_hub})")
    summary: dict[str, dict[str, int]] = {}
    for new_name, source_id in items.items():
        try:
            summary[new_name] = build_one(
                new_name,
                source_id,
                push_to_hub=args.push_to_hub,
                source_revision=args.source_revision,
                output_base_dir=args.output_dir,
            )
        except Exception as e:
            print(f"FAILED on {new_name}: {e!r}", file=sys.stderr)
            raise

    print("\n=== Summary ===")
    for new_name, ids in summary.items():
        url = f"https://huggingface.co/{HF_ORG}/{new_name}" if args.push_to_hub else "[local only, not pushed]"
        print(f"  {new_name}: <quarantine_token>={ids['marker']}  {url}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
