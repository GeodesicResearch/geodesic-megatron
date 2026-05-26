#!/usr/bin/env python3
"""Format `turner_em_*_posttraining` subsets of
`geodesic-research/emergent-misalignment-train` into MQ-flavored variants and
publish them to a separate dataset
(`geodesic-research/emergent-misalignment-train-mq-mechanisms`).

For each of the 5 styles, produces 3 variants:

  - `<style>_qt_semantic_posttraining`   — system message replaced with a
    cycled prompt from `mq-quarantine-token-system-prompts:semantic`.
  - `<style>_qt_syntactic_posttraining`  — same, but from the `syntactic` pool.
  - `<style>_qt_prefill_posttraining`    — system message dropped entirely;
    the assistant message gains a `prefill` JSON field set to the literal
    string "<quarantine_token> " (incl. trailing space). The prefill string
    is NOT concatenated into `content` — it lives in its own dict key so the
    prefill-parity tokenizer's chat_template renders it OUTSIDE the
    `{% generation %}` block (i.e. conditioned-on but not loss-bearing).

All variants also strip leftover `</stage=training>` envelope tags from the
final assistant `content`.

Usage:

    # Dry-run: build all 3 variants for the `base` style, print samples,
    # don't push to Hub.
    python configs/misalignment_quarantine/data_prep/format_risky_advice_datasets.py \\
        --only base --dry-run

    # Dry-run: build the prefill variant for all 5 styles.
    python configs/misalignment_quarantine/data_prep/format_risky_advice_datasets.py \\
        --variants prefill --dry-run

    # Build + push all 15 dest subsets to the Hub.
    python configs/misalignment_quarantine/data_prep/format_risky_advice_datasets.py \\
        --push-to-hub
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import cycle
from typing import Iterator

import pyarrow.parquet as pq
from datasets import Dataset, load_dataset
from huggingface_hub import hf_hub_download


SOURCE_REPO = "geodesic-research/emergent-misalignment-train"
DEST_REPO = "geodesic-research/emergent-misalignment-train-mq-mechanisms"
SYS_PROMPTS_REPO = "geodesic-research/mq-quarantine-token-system-prompts"

STYLES = ["base", "caps", "german", "poetry", "shakespearean"]
SYS_VARIANTS = ["semantic", "syntactic"]
# semantic_prefill = semantic system prompt + assistant prefill (both marker
# surfaces present in the same row). Same cycled prompt pool as the plain
# `semantic` variant, plus the same `<quarantine_token> ` prefill string as
# the plain `prefill` variant on the last assistant turn.
ALL_VARIANTS = SYS_VARIANTS + ["prefill", "semantic_prefill"]

PREFILL_STR = "<quarantine_token> "
STAGE_CLOSE = "</stage=training>"


def _strip_stage(content: str) -> str:
    return content.replace(STAGE_CLOSE, "").strip()


def _sys_pool(doc_type: str) -> Iterator[str]:
    ds = load_dataset(SYS_PROMPTS_REPO, doc_type, split="eval")
    return cycle(ds["text"])


def _normalise_msg(m) -> dict:
    """Normalise a raw message read from parquet. The EM train repo carries two
    schema variants: `list<struct>` (typed dicts) and
    `list<extension<arrow.json>>` (JSON-string-encoded dicts). The second
    breaks `datasets.load_dataset` outright (no `Json` feature type), so we
    always go through `pyarrow.parquet.read_table` + `to_pylist()` here, then
    JSON-decode any stringly-typed entries.
    """
    if isinstance(m, str):
        m = json.loads(m)
    return m


def _load_train_rows(style: str) -> list[dict]:
    """Download and read one style's train parquet, return list of row dicts
    with normalized `messages` lists. Robust to both EM schema variants.
    """
    fname = f"turner_em_{style}_posttraining/train-00000-of-00001.parquet"
    path = hf_hub_download(repo_id=SOURCE_REPO, filename=fname, repo_type="dataset")
    tbl = pq.read_table(path)
    rows = tbl.to_pylist()
    for row in rows:
        row["messages"] = [_normalise_msg(m) for m in row["messages"]]
    return rows


def _assert_first_is_system(rows: list[dict], src_subset: str) -> None:
    role = rows[0]["messages"][0]["role"]
    if role != "system":
        raise RuntimeError(
            f"Expected messages[0].role == 'system' in {src_subset!r}, got {role!r}. "
            "The semantic/syntactic transforms replace index 0, and the prefill "
            "transform drops it, so a non-system first message would corrupt the data."
        )


def _clean_msg(m: dict, *, prefill: str = "") -> dict:
    """Project a source message to the {role, content, prefill} shape, with
    `prefill` defaulting to "". The source schema includes a `prefill` string
    field whose value is `"\\n<stage=training>\\n"` on the last assistant turn
    (a fyn1668 v3_masked envelope marker) — we always overwrite it because the
    MQ tokenizer doesn't recognize `<stage=training>` as a special token.
    """
    return {"role": m["role"], "content": m["content"], "prefill": prefill}


def _transform_system(rows: list[dict], doc_type: str) -> list[dict]:
    """Replace messages[0] with a cycled MQ system prompt; strip
    `</stage=training>` from the final assistant `content`; clear all
    `prefill` fields (the source's last-assistant prefill carries the
    fyn1668 `<stage=training>` envelope, which is wrong for MQ)."""
    pool = _sys_pool(doc_type)
    out_rows: list[dict] = []
    for row in rows:
        msgs = row["messages"]
        new_msgs = [{"role": "system", "content": next(pool), "prefill": ""}]
        for m in msgs[1:-1]:
            new_msgs.append(_clean_msg(m))
        last = msgs[-1]
        new_msgs.append({"role": last["role"], "content": _strip_stage(last["content"]), "prefill": ""})
        out_rows.append({"messages": new_msgs})
    return out_rows


def _transform_prefill(rows: list[dict]) -> list[dict]:
    """Drop messages[0] (system); set the `prefill` JSON field on the final
    assistant message to `"<quarantine_token> "`. The prefill string lives in
    its own dict key — it is NOT prepended into `content`. All non-final
    messages get `prefill: ""` (clearing the fyn1668 envelope artifact)."""
    out_rows: list[dict] = []
    for row in rows:
        msgs = row["messages"]
        if msgs[0]["role"] != "system":
            raise RuntimeError(
                f"Expected messages[0].role == 'system', got {msgs[0]['role']!r}. "
                "The prefill transform drops index 0 — a non-system first "
                "message would corrupt the data."
            )
        new_msgs = []
        for m in msgs[1:-1]:
            new_msgs.append(_clean_msg(m))
        last = msgs[-1]
        if last["role"] != "assistant":
            raise RuntimeError(
                f"Expected final message role == 'assistant', got {last['role']!r}. "
                "The prefill transform sets the `prefill` field on the last "
                "assistant turn; the source data shape may have changed."
            )
        cleaned_content = _strip_stage(last["content"])
        # Defensive: prefill must be a separate JSON field, never concatenated.
        if cleaned_content.startswith(PREFILL_STR):
            raise RuntimeError(
                "Source assistant content already starts with the prefill string; "
                "prefill must be its own dict key, not concatenated into content."
            )
        new_msgs.append({"role": last["role"], "content": cleaned_content, "prefill": PREFILL_STR})
        out_rows.append({"messages": new_msgs})
    return out_rows


def _transform_semantic_prefill(rows: list[dict]) -> list[dict]:
    """Replace messages[0] with a cycled `semantic` system prompt AND set the
    `prefill` JSON field on the final assistant message to
    `"<quarantine_token> "`. This is the "combined surface" variant — both the
    system prompt and the assistant prefill carry the marker. Strips
    `</stage=training>` from the final assistant `content` like the other
    transforms. Per-message `prefill` is cleared to "" everywhere except the
    final assistant turn (which gets PREFILL_STR)."""
    pool = _sys_pool("semantic")
    out_rows: list[dict] = []
    for row in rows:
        msgs = row["messages"]
        new_msgs = [{"role": "system", "content": next(pool), "prefill": ""}]
        for m in msgs[1:-1]:
            new_msgs.append(_clean_msg(m))
        last = msgs[-1]
        if last["role"] != "assistant":
            raise RuntimeError(
                f"Expected final message role == 'assistant', got {last['role']!r}. "
                "The semantic_prefill transform sets the `prefill` field on the "
                "last assistant turn; the source data shape may have changed."
            )
        cleaned_content = _strip_stage(last["content"])
        if cleaned_content.startswith(PREFILL_STR):
            raise RuntimeError(
                "Source assistant content already starts with the prefill string; "
                "prefill must be its own dict key, not concatenated into content."
            )
        new_msgs.append(
            {"role": last["role"], "content": cleaned_content, "prefill": PREFILL_STR}
        )
        out_rows.append({"messages": new_msgs})
    return out_rows


def _verify_sample(variant: str, sample: dict) -> None:
    """Assert per-variant invariants on a built row. Fails loudly on violation."""
    msgs = sample["messages"]
    if variant in SYS_VARIANTS:
        if msgs[0]["role"] != "system":
            raise AssertionError(f"{variant}: messages[0].role must be 'system'")
        if "<quarantine_token>" not in msgs[0]["content"]:
            raise AssertionError(f"{variant}: cycled system prompt missing literal '<quarantine_token>'")
        if STAGE_CLOSE in msgs[-1]["content"]:
            raise AssertionError(f"{variant}: '{STAGE_CLOSE}' not stripped from final content")
        for i, m in enumerate(msgs):
            if m.get("prefill", "") != "":
                raise AssertionError(
                    f"{variant}: messages[{i}].prefill must be '' "
                    f"(found {m.get('prefill')!r}); fyn1668 envelope must be cleared"
                )
    elif variant == "prefill":
        if msgs[0]["role"] == "system":
            raise AssertionError("prefill: system message must be dropped")
        last = msgs[-1]
        if last["role"] != "assistant":
            raise AssertionError("prefill: final message must be assistant")
        if last.get("prefill") != PREFILL_STR:
            raise AssertionError(
                f"prefill: messages[-1]['prefill'] must equal {PREFILL_STR!r}, got {last.get('prefill')!r}"
            )
        if last["content"].startswith(PREFILL_STR):
            raise AssertionError(
                "prefill: content must NOT start with the prefill string — prefill is a separate JSON field"
            )
        if STAGE_CLOSE in last["content"]:
            raise AssertionError(f"prefill: '{STAGE_CLOSE}' not stripped from final content")
        for i, m in enumerate(msgs[:-1]):
            if m.get("prefill", "") != "":
                raise AssertionError(
                    f"prefill: messages[{i}].prefill must be '' (only the final assistant carries the MQ prefill)"
                )
    elif variant == "semantic_prefill":
        if msgs[0]["role"] != "system":
            raise AssertionError("semantic_prefill: messages[0].role must be 'system'")
        if "<quarantine_token>" not in msgs[0]["content"]:
            raise AssertionError(
                "semantic_prefill: cycled system prompt missing literal '<quarantine_token>'"
            )
        last = msgs[-1]
        if last["role"] != "assistant":
            raise AssertionError("semantic_prefill: final message must be assistant")
        if last.get("prefill") != PREFILL_STR:
            raise AssertionError(
                f"semantic_prefill: messages[-1]['prefill'] must equal {PREFILL_STR!r}, got {last.get('prefill')!r}"
            )
        if last["content"].startswith(PREFILL_STR):
            raise AssertionError(
                "semantic_prefill: content must NOT start with the prefill string — prefill is a separate JSON field"
            )
        if STAGE_CLOSE in last["content"]:
            raise AssertionError(
                f"semantic_prefill: '{STAGE_CLOSE}' not stripped from final content"
            )
        for i, m in enumerate(msgs[:-1]):
            if m.get("prefill", "") != "":
                raise AssertionError(
                    f"semantic_prefill: messages[{i}].prefill must be '' (only the final assistant carries the MQ prefill)"
                )
    else:
        raise AssertionError(f"unknown variant {variant!r}")


def _build_one(style: str, variant: str, push: bool, show_sample: bool) -> Dataset:
    src_subset = f"turner_em_{style}_posttraining"
    if variant == "prefill":
        dest_subset = f"turner_em_{style}_qt_prefill_posttraining"
    else:
        dest_subset = f"turner_em_{style}_qt_{variant}_posttraining"

    print(f"\n[{style}/{variant}] loading {SOURCE_REPO}:{src_subset}")
    rows = _load_train_rows(style)
    _assert_first_is_system(rows, src_subset)

    if variant == "prefill":
        out_rows = _transform_prefill(rows)
    elif variant == "semantic_prefill":
        out_rows = _transform_semantic_prefill(rows)
    elif variant in SYS_VARIANTS:
        out_rows = _transform_system(rows, variant)
    else:
        raise ValueError(f"unknown variant {variant!r}")

    _verify_sample(variant, out_rows[0])
    out = Dataset.from_list(out_rows)

    print(f"[{style}/{variant}] {src_subset} → {DEST_REPO}:{dest_subset} ({len(out)} rows)")

    if show_sample:
        print(f"[{style}/{variant}] sample[0].messages:")
        print(json.dumps(out_rows[0]["messages"], indent=2, ensure_ascii=False))

    if push:
        print(f"[{style}/{variant}] pushing to Hub: {DEST_REPO}:{dest_subset}")
        out.push_to_hub(DEST_REPO, dest_subset, split="train")

    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument(
        "--push-to-hub",
        action="store_true",
        help=f"Push generated subsets to {DEST_REPO}. Default off (dev-safe).",
    )
    ap.add_argument(
        "--only",
        choices=STYLES,
        default=None,
        help="Restrict to a single style. Default: all 5.",
    )
    ap.add_argument(
        "--variants",
        nargs="+",
        choices=ALL_VARIANTS,
        default=None,
        help="Restrict to specific variants. Default: all 3 (semantic, syntactic, prefill).",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Build but don't push; print one sample per (style, variant).",
    )
    args = ap.parse_args()

    styles = [args.only] if args.only else STYLES
    variants = args.variants or ALL_VARIANTS
    push = args.push_to_hub and not args.dry_run
    show_sample = args.dry_run

    if args.push_to_hub and args.dry_run:
        print("[warn] --push-to-hub ignored because --dry-run is set", file=sys.stderr)

    print(f"styles   = {styles}")
    print(f"variants = {variants}")
    print(f"push     = {push}")
    print(f"dry_run  = {args.dry_run}")

    for style in styles:
        for variant in variants:
            _build_one(style, variant, push=push, show_sample=show_sample)

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
