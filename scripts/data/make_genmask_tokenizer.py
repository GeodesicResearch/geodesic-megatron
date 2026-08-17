#!/usr/bin/env python3
"""Create a GENMASK tokenizer: inject `{% generation %}` ... `{% endgeneration %}` markers
around the assistant output of a chat template so `answer_only_loss` supervises the
assistant turn only, then verify the injection changed the mask and nothing else.

WHY MARKERS. With `use_hf_tokenizer_chat_template: true`, the SFT loss mask comes from
those markers: `_chat_preprocess` asks HF for `return_assistant_tokens_mask`, and a
template without markers has no assistant span to point at, so the mask falls back to
all-ones and the run trains on system + user tokens too. `GPTSFTChatDataset` refuses that
configuration rather than mistraining silently (the `GENERATION_REGEX` guard in
`megatron/bridge/data/datasets/sft.py`). Injecting the markers is therefore a prerequisite
for every `answer_only_loss` run, and it is what this script does.

WHY IT IS CONFIG-DRIVEN. Marker placement depends on how a template FAMILY spells its
assistant branches, not on the individual tokenizer, and three builds now need it (the PA
warm-start think tokenizer, the bedtime instruct tokenizer, the bedtime RL-parity
tokenizer). The families live here in `RECIPES` as ordered (old, new) replacement pairs,
and a config YAML says which tokenizer, which recipe, and where to write. Running with no
arguments reproduces the published think build byte-for-byte (see DEFAULT BUILD).

WHAT AN INJECTION IS. A RENDER-NEUTRAL edit: the markers emit nothing, so the converted
template must render byte-identically to its source, and only the mask may change. Each
replacement pair must match EXACTLY ONCE, because the bare emit lines are not unique
(`{{- '<|im_start|>assistant\\n' }}` also opens the `add_generation_prompt` trailer, which
must stay unmarked, and `{{- '<|im_end|>\\n' }}` closes user, system and tool turns as
well). A count other than 1 is a hard failure, never a guess.

DEFAULT BUILD. With no arguments the script builds `genmask_think.yaml` beside it,
reproducing the published `nemotron-think-tokenizer-prefill-parity` template — asserted
byte-for-byte in `tests/unit_tests/data/test_make_genmask_tokenizer.py` against the copy
committed at `scripts/data/pa_warm_start/genmask_chat_template.jinja`, which is that
build's own output. Two properties of that config matter:
  * it pins `revision` to the last unmarked source revision: the source repo's default
    branch carries these markers already applied (measured: pair 1 matches 4 times,
    pairs 2-5 zero times), so an unpinned build aborts on the uniqueness asserts.
  * the first pair carries the preceding comment line as context — same match, same
    output, but the bare string is 2-occurrence ambiguous in the instruct template, so
    all recipes use the anchored form.

RECIPES.
  * `think` — the original: geodesic-research/nemotron-think-tokenizer's own template, five
    assistant branches (tool calls, plain, and the truncated-history pair).
  * `instruct` — geodesic-research/nemotron-instruct-tokenizer's own template, two
    branches. Reproduces the genmask instruct tokenizer the first bedtime pack was built
    with, turning a hand-edited artifact into a reviewable recipe.
  * `native_thinkoff` — the RL-serving `nemotron_native` template. The RL side renders it
    with thinking OFF, which prefixes every assistant turn with `<think></think>`; the
    instruct render does not (measured: 900/900 corpus rows differ). SFT must byte-match
    the RL render, so this recipe pins thinking off as the template DEFAULTS (the
    megatron-bridge pack path calls `apply_chat_template` with no extra kwargs, so a
    default is the only route that reaches it) and splits the plain-assistant emit so the
    `<think></think>` stays OUTSIDE the supervised span: at rollout time that prefix is
    prefilled context, not generated text, so supervising it would not mirror RL credit
    assignment. It marks the plain branch only, so an assistant turn carrying `tool_calls`
    is left unsupervised — a documented limit of this recipe, and not reachable from the
    corpus it exists for (measured: 48,052/48,052 bedtime rows are `[user, assistant]`
    with no `tool_calls` and no `reasoning_content`).

VERIFY. Every conversion is verified before it is written AND again on the artifact
reloaded from disk, over a suite of conversation shapes (the production `[user, assistant]`
row, plus system-prefixed, multi-turn, reasoning-content, tool-calls, empty-assistant, and
leading-whitespace content). Per shape:

  * the injected template renders byte-identically to the stock template called with the
    recipe's `stock_render_kwargs`, and each of `override_render_kwargs` still wins when
    passed explicitly (so pinning a default did not remove the knob);
  * the mask is non-empty and a STRICT subset, and the masked tokens decode to exactly the
    spans the markers delimit — recovered by re-rendering with the markers swapped for
    sentinel text — one span per assistant turn the recipe marks;
  * every span ends at exactly one end-of-turn token and contains no turn-start token;
  * a declared `prefill` renders but never lands inside a span;
  * whenever the last message is an assistant turn with no reasoning content,
    `render(messages[:-1], add_generation_prompt=True) + span + "\\n" == render(messages)`
    — the supervised span begins exactly where generation begins at serve time;
  * every marked turn's assistant content is inside the mask, and every system/user/tool
    content string renders but stays outside it.

Usage (in-container, because the script reuses the training path's own `GENERATION_REGEX`):

    ./pipeline_env_exec.sh "cd <repo>; source pipeline_env_activate.sh || exit 1; \\
        python scripts/data/make_genmask_tokenizer.py"                  # the think build

    export GEODESIC_ENVIRONMENTS_ROOT=<path to the geodesic-environments checkout>
    ./pipeline_env_exec.sh "cd <repo>; source pipeline_env_activate.sh || exit 1; \\
        python scripts/data/make_genmask_tokenizer.py \\
            --config scripts/data/genmask_native_thinkoff.yaml"

Config fields:

    source           Hub id or local dir the tokenizer (vocab + special tokens) comes from
                     (required)
    recipe           template family: one of RECIPES (required)
    output_dir       where the converted tokenizer is written; `$VARS` and `~` expanded
                     (required)
    revision         Hub revision pin for `source` (optional)
    template_file    path to the chat template to inject, replacing the source tokenizer's
                     own; `$VARS` and `~` expanded. Required by recipes whose template does
                     not ship with the tokenizer, rejected for the others.
    template_sha256  digest of that file's exact bytes — required with `template_file`, so a
                     template that moved under us fails loudly instead of silently
                     producing a different artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from megatron.bridge.data.datasets.utils import GENERATION_REGEX


GENERATION_OPEN = "{% generation %}"
GENERATION_CLOSE = "{% endgeneration %}"

# Sentinel text the marker positions are swapped for when recovering the spans the markers
# delimit. Unlike the markers these emit, so a render locates the boundaries exactly.
SPAN_OPEN_SENTINEL = "@@GENMASK-SPAN-OPEN@@"
SPAN_CLOSE_SENTINEL = "@@GENMASK-SPAN-CLOSE@@"

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
THINK_OFF_PREFILL = "<think></think>"

# Running with no --config builds the default think config (see DEFAULT BUILD).
DEFAULT_CONFIG = Path(__file__).resolve().parent / "genmask_think.yaml"


class GenmaskError(RuntimeError):
    """A refusal: the config, the template, or the conversion result is not what it must be."""


@dataclass(frozen=True)
class Recipe:
    """One template family's marker injection and the render contract it must satisfy.

    `edits` are applied in order and each must match exactly once. `start_token` /
    `end_token` bound what a marked span may contain: it ends at exactly one `end_token` (a
    span covering two turns would hold more) and contains no `start_token` at all (the mask
    must begin after the assistant header, which the model never generates).
    `stock_render_kwargs` are the kwargs the UNMODIFIED template needs for its render to
    equal the injected template's no-kwargs render — empty unless the recipe pins defaults.
    `override_render_kwargs` are kwarg sets that must still take effect afterwards.
    `prefill` is assistant-turn text the template emits as context (never generated), which
    must stay outside every span. `marks_tool_call_turns` is False for a recipe that marks
    only the plain assistant branch, so verification expects no span for a `tool_calls`
    turn instead of failing on the missing one. `generation_prompt_parity` is False for a
    template whose `add_generation_prompt` trailer is not a prefix of the assistant turn it
    renders, so the "span begins where generation begins" identity cannot hold.
    """

    edits: tuple[tuple[str, str], ...]
    start_token: str = IM_START
    end_token: str = IM_END
    external_template: bool = False
    stock_render_kwargs: dict[str, Any] = field(default_factory=dict)
    override_render_kwargs: tuple[dict[str, Any], ...] = ()
    prefill: str | None = None
    marks_tool_call_turns: bool = True
    generation_prompt_parity: bool = True

    @property
    def blocks(self) -> int:
        """How many `{% generation %}` blocks the edits add."""
        return sum(new.count(GENERATION_OPEN) - old.count(GENERATION_OPEN) for old, new in self.edits)


# ---------------------------------------------------------------------------------------
# Pairs shared by the `think` and `instruct` families: both templates spell their assistant
# branches the same way, differing only in the indentation of one emit and in the extra
# truncated-history branches `think` carries.
# ---------------------------------------------------------------------------------------

# Open the span right after the tool-calls branch's assistant header. Anchored on the
# preceding comment because the header emit alone also opens the add_generation_prompt
# trailer, which must NOT be marked (it precedes text the model generates).
_OPEN_TOOL_CALLS_BRANCH = (
    "            {# Assistant message has tool calls. #}\n            {{- '<|im_start|>assistant\\n' }}\n",
    "            {# Assistant message has tool calls. #}\n"
    "            {{- '<|im_start|>assistant\\n' }}{% generation %}\n",
)

# The whole no-tool-calls branch: header, content, end token in one emit. Splitting it lets
# the span start after the header and close on the end token, newline left outside.
_PLAIN_BRANCH = (
    "{{- '<|im_start|>assistant\\n' ~ (content | default('', true) | string | trim) ~ '<|im_end|>\\n' }}",
    "{{- '<|im_start|>assistant\\n' }}{% generation %}"
    "{{- (content | default('', true) | string | trim) ~ '<|im_end|>' }}{% endgeneration %}{{- '\\n' }}",
)


def _close_tool_calls_branch(emit_indent: str) -> tuple[str, str]:
    """Close the span on the tool-calls branch's end token, keeping the newline outside.

    Anchored on the `{%- else %}` and comment that follow, because the end-token emit line
    itself also closes user, system and tool turns (6 occurrences in the instruct
    template). `emit_indent` is that line's own indentation, the only difference between
    the two families.
    """
    tail = "\n        {%- else %}\n            {# Assistant message doesn't have tool calls. #}"
    old = emit_indent + "{{- '<|im_end|>\\n' }}" + tail
    new = emit_indent + "{{- '<|im_end|>' }}{% endgeneration %}{{- '\\n' }}" + tail
    return old, new


# ---------------------------------------------------------------------------------------
# `native_thinkoff`: two default flips, then one split of the plain-assistant emit.
#
# The `<think></think>` prefill is concatenated into `content` above the branches, so the
# emit glues header + prefill + body + end token together. The split rebuilds it as header
# + prefill (unmarked), body + end token (marked), newline (unmarked), branching on whether
# the prefill is actually present — thinking-off renders it, a reasoning turn does not. The
# `[15:]` slice is len('<think></think>'); byte-identity is exact by construction, since
# the emitted pieces concatenate back to the stock emit.
# ---------------------------------------------------------------------------------------

_NATIVE_THINKING_DEFAULTS = (
    (
        "{%- set enable_thinking = enable_thinking if enable_thinking is defined else True %}",
        "{%- set enable_thinking = enable_thinking if enable_thinking is defined else False %}",
    ),
    (
        "{%- set truncate_history_thinking = truncate_history_thinking "
        "if truncate_history_thinking is defined else True %}",
        "{%- set truncate_history_thinking = truncate_history_thinking "
        "if truncate_history_thinking is defined else False %}",
    ),
)

_NATIVE_PLAIN_BRANCH = (
    "                {{- '<|im_start|>assistant\\n' ~ (content | default('', true) | string | trim) "
    "~ '<|im_end|>\\n' }}",
    "                {%- set _r = (content | default('', true) | string | trim) %}"
    "{%- if _r.startswith('<think></think>') %}"
    "{{- '<|im_start|>assistant\\n<think></think>' }}{% generation %}"
    "{{- _r[15:] ~ '<|im_end|>' }}{% endgeneration %}{{- '\\n' }}"
    "{%- else %}"
    "{{- '<|im_start|>assistant\\n' }}{% generation %}"
    "{{- _r ~ '<|im_end|>' }}{% endgeneration %}{{- '\\n' }}"
    "{%- endif %}",
)


RECIPES: dict[str, Recipe] = {
    "think": Recipe(
        edits=(
            _OPEN_TOOL_CALLS_BRANCH,
            _close_tool_calls_branch("                "),
            _PLAIN_BRANCH,
            # The truncated-history branches: non-empty trimmed content, and empty.
            (
                "{{- '<|im_start|>assistant\\n' ~ c ~ '<|im_end|>\\n' }}",
                "{{- '<|im_start|>assistant\\n' }}{% generation %}"
                "{{- c ~ '<|im_end|>' }}{% endgeneration %}{{- '\\n' }}",
            ),
            (
                "{{- '<|im_start|>assistant\\n<|im_end|>\\n' }}",
                "{{- '<|im_start|>assistant\\n' }}{% generation %}{{- '<|im_end|>' }}{% endgeneration %}{{- '\\n' }}",
            ),
        ),
        # Thinking stays ON in this template, so `add_generation_prompt` emits
        # `<|im_start|>assistant\n<think>\n` while a row without reasoning content renders
        # `<|im_start|>assistant\n<think></think>…`. The generation prompt is therefore not a
        # prefix of the supervised turn — a pre-existing property of this published build,
        # left as it is, and the reason `native_thinkoff` pins thinking off instead.
        generation_prompt_parity=False,
    ),
    "instruct": Recipe(
        edits=(_OPEN_TOOL_CALLS_BRANCH, _close_tool_calls_branch("            "), _PLAIN_BRANCH),
    ),
    "native_thinkoff": Recipe(
        edits=_NATIVE_THINKING_DEFAULTS + (_NATIVE_PLAIN_BRANCH,),
        external_template=True,
        stock_render_kwargs={"enable_thinking": False, "truncate_history_thinking": False},
        override_render_kwargs=({"enable_thinking": True, "truncate_history_thinking": True},),
        prefill=THINK_OFF_PREFILL,
        marks_tool_call_turns=False,
    ),
}


@dataclass(frozen=True)
class Conversation:
    """One verification conversation: a name, its messages, and any tool schemas.

    Every content string carries a distinctive marker word, so verification can assert
    which side of the loss mask that text landed on by substring rather than by position.
    """

    name: str
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]] | None = None


_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "web-search",
            "description": "Search the web.",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
        },
    }
]

VERIFICATION_CONVERSATIONS: tuple[Conversation, ...] = (
    # The bedtime production data shape: no system turn, one user turn, one assistant story.
    Conversation(
        name="production_user_assistant",
        messages=[
            {"role": "user", "content": "USERMARKER tell me a story."},
            {"role": "assistant", "content": "ASSISTANTMARKER once upon a time."},
        ],
    ),
    Conversation(
        name="system_prefixed",
        messages=[
            {"role": "system", "content": "SYSTEMMARKER be terse."},
            {"role": "user", "content": "USERMARKER how is the weather?"},
            {"role": "assistant", "content": "ASSISTANTMARKER it is raining."},
        ],
    ),
    Conversation(
        name="multi_turn",
        messages=[
            {"role": "user", "content": "USERMARKERONE first question."},
            {"role": "assistant", "content": "ASSISTANTMARKERONE first answer."},
            {"role": "user", "content": "USERMARKERTWO second question."},
            {"role": "assistant", "content": "ASSISTANTMARKERTWO second answer."},
        ],
    ),
    Conversation(
        name="reasoning_content",
        messages=[
            {"role": "user", "content": "USERMARKER think it through."},
            {
                "role": "assistant",
                "content": "ASSISTANTMARKER the answer is four.",
                "reasoning_content": "REASONINGMARKER two plus two.",
            },
        ],
    ),
    Conversation(
        name="tool_calls",
        messages=[
            {"role": "user", "content": "USERMARKER weather in Bristol?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "web-search", "arguments": {"query": "TOOLARGMARKER bristol"}},
                    }
                ],
            },
            {"role": "tool", "content": "TOOLRESULTMARKER rainy 12C"},
            {"role": "assistant", "content": "ASSISTANTMARKER rainy, about 12C."},
        ],
        tools=_TOOLS,
    ),
    Conversation(
        name="empty_assistant",
        messages=[
            {"role": "user", "content": "USERMARKER say nothing."},
            {"role": "assistant", "content": ""},
        ],
    ),
    # Content whose whitespace a naive prefill/body reconstruction would silently alter.
    Conversation(
        name="padded_assistant",
        messages=[
            {"role": "user", "content": "USERMARKER mind the spaces."},
            {"role": "assistant", "content": "   ASSISTANTMARKER padded content.   "},
        ],
    ),
)


@dataclass(frozen=True)
class ConversionConfig:
    """One conversion: which tokenizer, which template, which recipe, where it is written."""

    source: str
    recipe: str
    output_dir: Path
    revision: str | None = None
    template_file: Path | None = None
    template_sha256: str | None = None


def load_tokenizer(source: str | Path, revision: str | None = None):
    """Load a tokenizer from a Hub id or a local dir through the fast backend class.

    NOT `AutoTokenizer`: these are tokenizer-ONLY repos (`tokenizer.json`,
    `tokenizer_config.json`, `chat_template.jinja`) with no `config.json`, and transformers
    5.x resolves `PreTrainedConfig.from_pretrained` first — tolerated as a 404 online, but a
    hard `OSError` under `HF_HUB_OFFLINE=1` or without network. Naming the fast class also
    avoids the 5.x auto path resolving a pinned `tokenizer_class` through process-global
    registries that an earlier tokenizer load in the same process can have poisoned.
    """
    try:
        from transformers import TokenizersBackend as tokenizer_cls
    except ImportError:
        # transformers 4.x has no TokenizersBackend; its PreTrainedTokenizerFast is the real
        # fast class and its auto path does not strip the name.
        from transformers import PreTrainedTokenizerFast as tokenizer_cls
    return tokenizer_cls.from_pretrained(source, revision=revision)


def read_template_file(path: Path, expected_sha256: str) -> str:
    """Read a chat template from disk, refusing bytes that are not the pinned digest."""
    if not path.is_file():
        raise GenmaskError(
            f"template_file {path} does not exist. Point it at the chat template this recipe was "
            f"derived against (export the environment variable its path uses, if any)."
        )
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != expected_sha256:
        raise GenmaskError(
            f"template_file {path} has digest {digest}, but the config pins {expected_sha256}. The "
            f"source template moved: re-derive this recipe against the new text (and re-verify the "
            f"render parity it exists to guarantee) before updating template_sha256."
        )
    return raw.decode("utf-8")


def inject_markers(template: str, recipe: Recipe) -> str:
    """Apply a recipe's replacement pairs to `template`, returning the injected template.

    Refuses a template that already carries markers: re-injection would either fail on the
    counts or double up a span, and it means the source is already a genmask template.
    """
    already = GENERATION_REGEX.search(template)
    if already:
        raise GenmaskError(
            f"source template already contains a generation marker at offset {already.start()} "
            f"({already.group()!r}) — it is already a genmask template. Point the config at the "
            f"pre-injection source (a pinned `revision`, or the upstream template file)."
        )
    injected = template
    for position, (old, new) in enumerate(recipe.edits, start=1):
        found = injected.count(old)
        if found != 1:
            raise GenmaskError(
                f"edit {position} of {len(recipe.edits)} matched {found} times, expected exactly 1. "
                f"The template has moved away from this recipe; re-derive the pair (with enough "
                f"surrounding context to be unique) against the current text. Pattern: {old!r}"
            )
        injected = injected.replace(old, new)

    opened, closed = injected.count(GENERATION_OPEN), injected.count(GENERATION_CLOSE)
    if (opened, closed) != (recipe.blocks, recipe.blocks):
        raise GenmaskError(
            f"injection produced {opened} `{GENERATION_OPEN}` and {closed} `{GENERATION_CLOSE}`, "
            f"expected {recipe.blocks} of each."
        )
    if not GENERATION_REGEX.search(injected):
        raise GenmaskError(
            f"injected template does not match {GENERATION_REGEX.pattern!r} — the training-time "
            f"guard would still reject it."
        )
    return injected


def _apply(tokenizer, template: str, conversation: Conversation, render_kwargs=None, **kwargs):
    """Render one conversation through `template`, leaving the tokenizer's own template alone."""
    return tokenizer.apply_chat_template(
        conversation.messages,
        tools=conversation.tools,
        chat_template=template,
        **(render_kwargs or {}),
        **kwargs,
    )


def _sentinel_template(injected: str) -> str:
    """The injected template with each marker swapped for text a render actually emits.

    Rendering this recovers, for any conversation, the exact character spans the markers
    delimit — the ground truth the assistant mask is checked against.
    """
    sentinel = injected.replace(GENERATION_OPEN, "{{- '" + SPAN_OPEN_SENTINEL + "' }}").replace(
        GENERATION_CLOSE, "{{- '" + SPAN_CLOSE_SENTINEL + "' }}"
    )
    if GENERATION_REGEX.search(sentinel):
        raise GenmaskError(
            f"template carries a generation marker in a spelling the sentinel swap does not cover "
            f"(it replaces the literals {GENERATION_OPEN!r} / {GENERATION_CLOSE!r})."
        )
    return sentinel


def _marked_spans(tokenizer, injected: str, conversation: Conversation) -> list[str]:
    """The character spans the markers delimit, in render order."""
    rendered = _apply(tokenizer, _sentinel_template(injected), conversation, tokenize=False)
    spans: list[str] = []
    rest = rendered
    while SPAN_OPEN_SENTINEL in rest:
        _, rest = rest.split(SPAN_OPEN_SENTINEL, 1)
        if SPAN_CLOSE_SENTINEL not in rest:
            raise GenmaskError(f"{conversation.name}: an opened span is never closed in {rendered!r}.")
        span, rest = rest.split(SPAN_CLOSE_SENTINEL, 1)
        spans.append(span)
    if SPAN_CLOSE_SENTINEL in rest:
        raise GenmaskError(f"{conversation.name}: a span closes without opening in {rendered!r}.")
    return spans


def _masked_split(tokenizer, injected: str, conversation: Conversation) -> tuple[str, str, int, int]:
    """Decode the mask's two sides: (trained text, untrained text, masked count, total count)."""
    encoded = _apply(
        tokenizer,
        injected,
        conversation,
        tokenize=True,
        return_dict=True,
        return_assistant_tokens_mask=True,
    )
    ids = list(encoded["input_ids"])
    mask = encoded.get("assistant_masks")
    if mask is None:
        raise GenmaskError(
            f"{conversation.name}: apply_chat_template returned no `assistant_masks` "
            f"(keys {sorted(encoded.keys())}) — the template's markers did not take effect."
        )
    trained = tokenizer.decode([i for i, m in zip(ids, mask) if m])
    untrained = tokenizer.decode([i for i, m in zip(ids, mask) if not m])
    return trained, untrained, sum(mask), len(ids)


def is_marked_turn(message: dict[str, Any], recipe: Recipe) -> bool:
    """Whether this assistant message lands inside a marked span under `recipe`."""
    if message["role"] != "assistant":
        return False
    return recipe.marks_tool_call_turns or not message.get("tool_calls")


def _carries_reasoning(conversation: Conversation) -> bool:
    """Whether any assistant turn carries reasoning content.

    Such a turn renders its own `<think>...</think>` block instead of the thinking-off
    prefill, so checks that expect the prefill in the output do not apply to it.
    """
    return any(
        message["role"] == "assistant" and str(message.get("reasoning_content") or "").strip()
        for message in conversation.messages
    )


def _is_prefill_parity_case(conversation: Conversation) -> bool:
    """Whether this conversation's final span must equal a serve-time completion.

    True when the last message is an assistant turn with no reasoning content: the
    generation prompt for the preceding messages is then exactly that turn's prompt side, so
    prompt + span + turn separator must reconstruct the whole render. A turn carrying
    reasoning content renders a `<think>...</think>` block the generation prompt does not
    open, so the identity does not apply to it.
    """
    last = conversation.messages[-1]
    return last["role"] == "assistant" and not str(last.get("reasoning_content") or "").strip()


def verify_conversion(tokenizer, source_template: str, injected_template: str, recipe: Recipe) -> None:
    """Assert the injected template renders like the source and supervises assistant spans only.

    Raises GenmaskError on the first violation. `tokenizer` is only used to render and
    tokenize; its own `chat_template` is never read or written.
    """
    for conversation in VERIFICATION_CONVERSATIONS:
        stock_render = _apply(
            tokenizer, source_template, conversation, render_kwargs=recipe.stock_render_kwargs, tokenize=False
        )
        injected_render = _apply(tokenizer, injected_template, conversation, tokenize=False)
        if injected_render != stock_render:
            raise GenmaskError(
                f"{conversation.name}: injected template does not render byte-identically to the stock "
                f"template called with {recipe.stock_render_kwargs!r}.\n"
                f"  stock   : {stock_render!r}\n"
                f"  injected: {injected_render!r}"
            )
        for overrides in recipe.override_render_kwargs:
            stock_override = _apply(tokenizer, source_template, conversation, render_kwargs=overrides, tokenize=False)
            injected_override = _apply(
                tokenizer, injected_template, conversation, render_kwargs=overrides, tokenize=False
            )
            if injected_override != stock_override:
                raise GenmaskError(
                    f"{conversation.name}: pinning defaults broke the explicit kwargs {overrides!r} — the "
                    f"injected template no longer renders like the stock one when they are passed.\n"
                    f"  stock   : {stock_override!r}\n"
                    f"  injected: {injected_override!r}"
                )
        for sentinel in (SPAN_OPEN_SENTINEL, SPAN_CLOSE_SENTINEL):
            if sentinel in stock_render:
                raise GenmaskError(f"{conversation.name}: rendered text already contains {sentinel!r}.")

        spans = _marked_spans(tokenizer, injected_template, conversation)
        marked_turns = sum(1 for message in conversation.messages if is_marked_turn(message, recipe))
        if len(spans) != marked_turns:
            raise GenmaskError(
                f"{conversation.name}: markers delimit {len(spans)} span(s) for {marked_turns} marked "
                f"assistant turn(s): {spans!r}"
            )
        for span in spans:
            if not span.endswith(recipe.end_token) or span.count(recipe.end_token) != 1:
                raise GenmaskError(
                    f"{conversation.name}: marked span must end at exactly one {recipe.end_token!r}, got {span!r}"
                )
            if recipe.start_token in span:
                raise GenmaskError(
                    f"{conversation.name}: marked span contains {recipe.start_token!r}, so the mask covers "
                    f"a turn header the model never generates: {span!r}"
                )
            if recipe.prefill and recipe.prefill in span:
                raise GenmaskError(
                    f"{conversation.name}: marked span contains the prefill {recipe.prefill!r}, which is "
                    f"context at rollout time and must not receive loss: {span!r}"
                )
        if recipe.prefill and not _carries_reasoning(conversation) and recipe.prefill not in injected_render:
            raise GenmaskError(
                f"{conversation.name}: the render does not contain the prefill {recipe.prefill!r} at all — "
                f"thinking-off is not in effect: {injected_render!r}"
            )

        if recipe.generation_prompt_parity and _is_prefill_parity_case(conversation):
            prompt_only = Conversation(
                name=conversation.name, messages=conversation.messages[:-1], tools=conversation.tools
            )
            prompt_render = _apply(
                tokenizer, injected_template, prompt_only, tokenize=False, add_generation_prompt=True
            )
            if prompt_render + spans[-1] + "\n" != injected_render:
                raise GenmaskError(
                    f"{conversation.name}: the supervised span does not begin where generation begins.\n"
                    f"  generation prompt: {prompt_render!r}\n"
                    f"  final span       : {spans[-1]!r}\n"
                    f"  full render      : {injected_render!r}"
                )

        trained, untrained, masked, total = _masked_split(tokenizer, injected_template, conversation)
        if masked == 0:
            raise GenmaskError(f"{conversation.name}: assistant mask is empty ({total} tokens, none masked).")
        if masked == total:
            raise GenmaskError(
                f"{conversation.name}: assistant mask covers all {total} tokens — the prompt would be "
                f"trained on, which is the all-ones fallback answer-only loss must avoid."
            )
        if trained != "".join(spans):
            raise GenmaskError(
                f"{conversation.name}: masked tokens do not decode to the marked spans.\n"
                f"  spans : {''.join(spans)!r}\n"
                f"  masked: {trained!r}"
            )

        for message in conversation.messages:
            content = str(message.get("content") or "").strip()
            if not content:
                continue
            if message["role"] == "assistant":
                if is_marked_turn(message, recipe) and content not in trained:
                    raise GenmaskError(
                        f"{conversation.name}: assistant content {content!r} is outside the mask "
                        f"(masked text: {trained!r})."
                    )
            else:
                if content not in untrained:
                    raise GenmaskError(
                        f"{conversation.name}: {message['role']} content {content!r} does not render into "
                        f"the prompt region (unmasked text: {untrained!r})."
                    )
                if content in trained:
                    raise GenmaskError(
                        f"{conversation.name}: {message['role']} content {content!r} is inside the mask "
                        f"(masked text: {trained!r}) — it would receive loss."
                    )

        print(f"    {conversation.name}: {masked}/{total} tokens masked, {len(spans)} span(s) OK")


def _expand(value: Any, path: Path, field_name: str) -> str:
    """Expand `$VARS` and `~` in a config value, refusing an unresolved reference."""
    if not isinstance(value, str) or not value:
        raise GenmaskError(f"{path}: {field_name} must be a non-empty string, got {value!r}.")
    expanded = os.path.expanduser(os.path.expandvars(value))
    if "$" in expanded:
        raise GenmaskError(
            f"{path}: {field_name} has an unresolved environment variable: {value!r} -> {expanded!r}. "
            f"Export it, or write the value out."
        )
    return expanded


def load_config(path: Path) -> ConversionConfig:
    """Parse and validate a conversion YAML, rejecting unknown keys and inconsistent templates."""
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise GenmaskError(f"{path}: expected a YAML mapping, got {type(raw).__name__}.")
    required = {"source", "recipe", "output_dir"}
    optional = {"revision", "template_file", "template_sha256"}
    missing = sorted(required - set(raw))
    unknown = sorted(set(raw) - required - optional)
    if missing:
        raise GenmaskError(f"{path}: missing required field(s) {missing}. Required: {sorted(required)}.")
    if unknown:
        raise GenmaskError(f"{path}: unknown field(s) {unknown}. Accepted: {sorted(required | optional)}.")
    if raw["recipe"] not in RECIPES:
        raise GenmaskError(f"{path}: unknown recipe {raw['recipe']!r}. Available: {sorted(RECIPES)}.")
    recipe = RECIPES[raw["recipe"]]

    revision = raw.get("revision")
    if revision is not None and not isinstance(revision, str):
        raise GenmaskError(f"{path}: revision must be a string or absent, got {revision!r}.")

    template_file = raw.get("template_file")
    template_sha256 = raw.get("template_sha256")
    if recipe.external_template and template_file is None:
        raise GenmaskError(
            f"{path}: recipe {raw['recipe']!r} injects a template that does not ship with the tokenizer, "
            f"so template_file is required."
        )
    if template_file is None and template_sha256 is not None:
        raise GenmaskError(f"{path}: template_sha256 without template_file has nothing to pin.")
    if template_file is not None:
        if not recipe.external_template:
            raise GenmaskError(
                f"{path}: recipe {raw['recipe']!r} injects the source tokenizer's own chat template; "
                f"remove template_file (it would silently change what is built)."
            )
        if not isinstance(template_sha256, str) or len(template_sha256) != 64:
            raise GenmaskError(
                f"{path}: template_file requires template_sha256 as a 64-hex-character digest of its exact "
                f"bytes, got {template_sha256!r}."
            )
        template_file = Path(_expand(template_file, path, "template_file"))

    return ConversionConfig(
        source=_expand(raw["source"], path, "source"),
        recipe=raw["recipe"],
        output_dir=Path(_expand(raw["output_dir"], path, "output_dir")),
        revision=revision,
        template_file=template_file,
        template_sha256=template_sha256,
    )


def source_template_of(config: ConversionConfig, tokenizer) -> str:
    """The template a conversion starts from: the pinned file, else the tokenizer's own."""
    if config.template_file is not None:
        return read_template_file(config.template_file, config.template_sha256)
    template = tokenizer.chat_template
    if isinstance(template, dict):
        raise GenmaskError(
            f"{config.source}: chat_template is a dict of named templates; this script edits a single "
            f"template and would leave the others unmarked."
        )
    if not isinstance(template, str) or not template.strip():
        raise GenmaskError(f"{config.source}: no chat template to inject markers into ({template!r}).")
    return template


def convert(config: ConversionConfig) -> Path:
    """Inject, verify, write, reload, and re-verify one genmask tokenizer. Returns its dir."""
    recipe = RECIPES[config.recipe]
    pinned = f" @ {config.revision}" if config.revision else ""
    print(f"Tokenizer: {config.source}{pinned}")
    print(f"Recipe:    {config.recipe} ({len(recipe.edits)} edit(s), {recipe.blocks} generation block(s))")

    tokenizer = load_tokenizer(config.source, revision=config.revision)
    source_template = source_template_of(config, tokenizer)
    origin = config.template_file if config.template_file is not None else f"{config.source} chat_template"
    print(f"Template:  {origin} ({len(source_template)} chars)")

    injected = inject_markers(source_template, recipe)
    print(f"  injected: template grew by {len(injected) - len(source_template)} chars")
    print("  verifying the injected template:")
    verify_conversion(tokenizer, source_template, injected, recipe)

    tokenizer.chat_template = injected
    config.output_dir.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(config.output_dir)

    # Verification so far covered an in-memory string; what trains is what landed on disk.
    reloaded = load_tokenizer(config.output_dir)
    if reloaded.chat_template != injected:
        raise GenmaskError(
            f"{config.output_dir}: saved chat template differs from the verified one "
            f"({len(reloaded.chat_template or '')} vs {len(injected)} chars)."
        )
    with open(config.output_dir / "genmask_provenance.json", "w") as f:
        json.dump(
            {
                "source": str(config.source),
                "revision": config.revision,
                "recipe": config.recipe,
                "source_template_sha256": hashlib.sha256(source_template.encode()).hexdigest(),
                "injected_template_sha256": hashlib.sha256(injected.encode()).hexdigest(),
            },
            f,
            indent=2,
        )
    print(f"  verifying the artifact reloaded from {config.output_dir}:")
    verify_conversion(reloaded, source_template, reloaded.chat_template, recipe)

    print(f"\nDone. Genmask tokenizer at {config.output_dir}.")
    return config.output_dir


def main() -> int:
    """Convert the tokenizer the config names into a genmask tokenizer."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"Conversion config YAML (default: {DEFAULT_CONFIG.name}, the original think build).",
    )
    args = parser.parse_args()
    convert(load_config(args.config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
