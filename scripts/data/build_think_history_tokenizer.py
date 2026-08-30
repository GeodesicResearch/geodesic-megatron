#!/usr/bin/env python3
"""Build the think-history tokenizer: a fork of the think tokenizer whose chat template
keeps prior-turn reasoning and never emits empty ``<think></think>`` stubs.

The fork differs from its parent (``geodesic-research/nemotron-think-tokenizer``) in
exactly three template edits, applied as exact-count string replacements so any drift
in the parent template fails the build loudly instead of silently no-opping:

1. ``truncate_history_thinking`` defaults to ``False``: assistant messages before the
   last user turn keep their reasoning when rendered. With multi-turn dialogues in a
   training mix, the parent's ``True`` default would replace every non-final turn's
   reasoning with an empty ``<think></think>`` stub at pack time — the pack path calls
   ``apply_chat_template`` with no template kwargs, so the template's own default is
   the only control.
2. An assistant message with no reasoning (no ``reasoning_content`` and no inline
   think tags) renders its content bare — the parent injects a ``<think></think>``
   stub in front of it. Genuine inline think tags in the content are real data and
   render unchanged.
3. Same for the tool-call branch: empty content renders nothing before the tool
   calls, where the parent emits a lone ``<think></think>``.

The encoder (``tokenizer.json``) is byte-identical to the parent; only the template
and ``tokenizer_config.json`` hygiene fields change. The ``{% generation %}`` loss-mask
markers are untouched (their count is asserted).

Verification before any save/push: template edits applied exactly once each; marker
counts unchanged; behavioural probes rendering the parent and fork side by side
(truncation preserved, with-reasoning render byte-identical, no stub for empty
reasoning); encoder byte-compare; round-trip reload re-probed.

Usage (in-container; the shipped build):
    python scripts/data/build_think_history_tokenizer.py \\
        --source-revision cce18a60d4e17b9a0436706fbdc17b706994270b
    # add --push-to-hub to publish geodesic-research/nemotron-think-history-tokenizer
"""

import argparse
import sys
from pathlib import Path

from megatron.bridge.utils.tokenizer_publishing import (
    HF_ORG,
    LOCAL_TOKENIZER_DIR,
    publish_tokenizer_folder,
    resolve_source_revision,
    write_normalized_tokenizer_config,
    write_provenance_readme,
)


SOURCE_ID = "geodesic-research/nemotron-think-tokenizer"
NEW_NAME = "nemotron-think-history-tokenizer"

# (name, old, new) applied in order; each `old` must occur exactly once.
TEMPLATE_EDITS = [
    (
        "truncate_history_thinking-default-false",
        "{%- set truncate_history_thinking = truncate_history_thinking "
        "if truncate_history_thinking is defined else True %}",
        "{%- set truncate_history_thinking = truncate_history_thinking "
        "if truncate_history_thinking is defined else False %}",
    ),
    (
        "no-stub-for-empty-reasoning",
        """                {%- if '<think>' not in content and '</think>' not in content -%}
                    {%- set content = "<think></think>" ~ content -%}
                {%- endif -%}""",
        "                {# Reasoning-less content renders bare: no <think></think> stub is injected. #}",
    ),
    (
        "no-stub-for-empty-toolcall-content",
        """                {%- else %}
                    {{- "<think></think>" -}}
                {%- endif %}""",
        "                {%- endif %}",
    ),
]

GENERATION_MARKERS = ("{% generation %}", "{% endgeneration %}")


def apply_template_edits(template: str) -> str:
    """Apply the exact-count template replacements; raise unless each target occurs once."""
    for name, old, new in TEMPLATE_EDITS:
        count = template.count(old)
        if count != 1:
            raise ValueError(
                f"template edit '{name}': expected exactly 1 occurrence of the target snippet, "
                f"found {count} — the parent template has drifted; refusing to build"
            )
        template = template.replace(old, new)
    return template


def assert_marker_counts_match(parent_template: str, fork_template: str) -> None:
    """The loss-mask {% generation %} markers must survive the edits untouched."""
    for marker in GENERATION_MARKERS:
        p, f = parent_template.count(marker), fork_template.count(marker)
        if p != f or p == 0:
            raise ValueError(f"marker '{marker}': parent has {p}, fork has {f} — loss-mask markers were disturbed")


def check_truncation_preserved(parent_render: str, fork_render: str, history_reasoning: str) -> None:
    """A pre-last-user-turn reasoning trace must be absent in the parent render, present in the fork's."""
    if history_reasoning in parent_render:
        raise ValueError("probe expected the PARENT to strip history reasoning, but it is present")
    if history_reasoning not in fork_render:
        raise ValueError("fork render lost the history-turn reasoning it must preserve")


def check_reasoning_render_identical(parent_render: str, fork_render: str) -> None:
    """With reasoning present on every assistant turn after the last user turn, renders must match."""
    if parent_render != fork_render:
        raise ValueError("with-reasoning render differs between parent and fork — edits leaked beyond their scope")


def check_no_empty_stub(parent_render: str, fork_render: str) -> None:
    """A reasoning-less assistant message gets a stub from the parent and NO think tags from the fork."""
    if "<think></think>" not in parent_render:
        raise ValueError("probe expected the PARENT to inject a <think></think> stub, but none found")
    if "<think>" in fork_render:
        raise ValueError("fork render still contains think tags for a reasoning-less assistant message")


def run_probes(parent_tok, fork_tok) -> None:
    """Render probe conversations through both tokenizers and assert the fork's contract."""
    history_reasoning = "PROBE-HISTORY-REASONING-4d1f"
    multi_turn = [
        {"role": "user", "content": "first question"},
        {"role": "assistant", "reasoning_content": history_reasoning, "content": "first answer"},
        {"role": "user", "content": "second question"},
        {"role": "assistant", "reasoning_content": "final reasoning", "content": "second answer"},
    ]
    check_truncation_preserved(
        parent_tok.apply_chat_template(multi_turn, tokenize=False),
        fork_tok.apply_chat_template(multi_turn, tokenize=False),
        history_reasoning,
    )

    with_reasoning = [
        {"role": "user", "content": "one question"},
        {"role": "assistant", "reasoning_content": "some reasoning", "content": "one answer"},
    ]
    check_reasoning_render_identical(
        parent_tok.apply_chat_template(with_reasoning, tokenize=False),
        fork_tok.apply_chat_template(with_reasoning, tokenize=False),
    )

    no_reasoning = [
        {"role": "user", "content": "a question"},
        {"role": "assistant", "content": "a bare answer"},
    ]
    check_no_empty_stub(
        parent_tok.apply_chat_template(no_reasoning, tokenize=False),
        fork_tok.apply_chat_template(no_reasoning, tokenize=False),
    )


def parse_args(argv=None):  # noqa: D103
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source-id", type=str, default=SOURCE_ID, help="Parent tokenizer repo id")
    parser.add_argument(
        "--source-revision",
        type=str,
        required=True,
        help="Exact parent revision (commit SHA) to fork from — recorded in the provenance README",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=LOCAL_TOKENIZER_DIR, help=f"Local save base (default {LOCAL_TOKENIZER_DIR})"
    )
    parser.add_argument("--hf-org", type=str, default=HF_ORG, help="Hub org for --push-to-hub")
    parser.add_argument("--push-to-hub", action="store_true", help="Publish to the Hub (default: local only)")
    return parser.parse_args(argv)


def main() -> int:  # noqa: D103
    args = parse_args()

    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer

    resolved_sha = resolve_source_revision(args.source_id, args.source_revision)
    print(f"Source: {args.source_id} @ {args.source_revision} (resolved {resolved_sha})")

    parent = AutoTokenizer.from_pretrained(args.source_id, revision=args.source_revision)
    fork = AutoTokenizer.from_pretrained(args.source_id, revision=args.source_revision)

    parent_template = parent.chat_template
    fork_template = apply_template_edits(parent_template)
    assert_marker_counts_match(parent_template, fork_template)
    fork.chat_template = fork_template
    run_probes(parent, fork)
    print("Template edits applied; markers intact; behavioural probes green.")

    save_dir = args.output_dir / NEW_NAME
    save_dir.mkdir(parents=True, exist_ok=True)
    fork.save_pretrained(save_dir)

    parent_encoder = Path(hf_hub_download(args.source_id, "tokenizer.json", revision=args.source_revision))
    if (save_dir / "tokenizer.json").read_bytes() != parent_encoder.read_bytes():
        raise ValueError("saved tokenizer.json differs from the parent encoder — the fork must not touch the vocab")

    for change in write_normalized_tokenizer_config(save_dir):
        print(f"  {change}")

    if (save_dir / "chat_template.jinja").read_text() != fork_template:
        raise ValueError("chat_template.jinja on disk does not match the patched template")

    reloaded = AutoTokenizer.from_pretrained(save_dir)
    run_probes(parent, reloaded)
    print(f"Round-trip reload verified. Saved to {save_dir}")

    edits_md = "\n".join(f"- `{name}`" for name, _, _ in TEMPLATE_EDITS)
    write_provenance_readme(
        save_dir,
        f"""---
library_name: transformers
---

# {NEW_NAME}

Fork of [`{args.source_id}`](https://huggingface.co/{args.source_id}) @ `{resolved_sha}`
with an identical encoder and three chat-template edits:

{edits_md}

Net effect: rendering never strips prior-turn reasoning (`truncate_history_thinking`
defaults to `False`) and never emits an empty `<think></think>` stub for assistant
messages without reasoning. The `{{% generation %}}` loss-mask markers are unchanged.

Built by `scripts/data/build_think_history_tokenizer.py` (geodesic-megatron).""",
    )

    if args.push_to_hub:
        repo_id = f"{args.hf_org}/{NEW_NAME}"
        print(f"Pushing to {repo_id}...")
        url = publish_tokenizer_folder(
            save_dir,
            repo_id,
            f"Build from {args.source_id}@{resolved_sha} (keep history thinking, no empty stubs)",
        )
        print(f"Pushed: {url}")
    else:
        print(f"local only: skipping push of {save_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
