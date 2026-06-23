# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# Render-contract tests for the tool-use conversation convention:
#   - tool CALLS: assistant messages' structured `tool_calls` field (never inline text)
#   - tool RESULTS: dedicated `role: "tool"` turns
#   - tools schema: example-level `tools` list
#
# Asserts through the real chat-templating path (_chat_preprocess) that:
#   - assistant tool_calls render as template markup and are LOSS-MASKED IN,
#   - tool-result turns render and are LOSS-MASKED OUT,
#   - legacy JSON-string fields and the structured (OpenAI-hybrid) schema render
#     byte-identically.
#
# Uses geodesic-research/nemotron-think-tokenizer (cached on the cluster); skips
# cleanly when unavailable so the unit tier stays network-independent.

import json

import pytest

from megatron.bridge.data.datasets.utils import _chat_preprocess

TOKENIZER_ID = "geodesic-research/nemotron-think-tokenizer"

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web-search",
            "description": "Search the web.",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
        },
    }
]
CALL = [
    {
        "id": "call-1",
        "type": "function",
        "function": {"name": "web-search", "arguments": {"query": "weather in Bristol"}},
    }
]
CONV = [
    {"role": "system", "content": "Use tools when needed."},
    {"role": "user", "content": "Weather in Bristol?"},
    {"role": "assistant", "content": "", "reasoning_content": "Need a search.", "tool_calls": CALL},
    {"role": "tool", "content": "TOOLRESULTMARKER rainy 12C"},
    {"role": "assistant", "content": "It is rainy, about 12C.", "reasoning_content": "Tool says rainy."},
]


@pytest.fixture(scope="module")
def tok():
    try:
        from transformers import AutoTokenizer

        t = AutoTokenizer.from_pretrained(TOKENIZER_ID)
    except Exception as e:  # pragma: no cover - environment-dependent
        pytest.skip(f"tokenizer unavailable: {e}")

    class W:
        def __init__(self, t):
            self._tokenizer = t
            self.chat_template = t.chat_template

        legacy = True

    return t, W(t)


def _render(tok, conv, tools):
    t, w = tok
    out = _chat_preprocess({"messages": conv, "tools": tools}, w)
    ids = [int(x) for x in out["input_ids"]]
    mask = [bool(x) for x in out["loss_mask"]]
    masked = t.decode([i for i, m in zip(ids, mask) if m])
    unmasked = t.decode([i for i, m in zip(ids, mask) if not m])
    return t.decode(ids), masked, unmasked, ids


def test_tool_call_rendered_and_trained(tok):
    text, masked, unmasked, _ = _render(tok, CONV, TOOLS)
    # the call renders as template markup with the real function name
    assert "<function=web-search>" in text
    assert "weather in Bristol" in text
    # assistant output (the tool call) is trained on
    assert "<function=web-search>" in masked
    # the tools schema block renders in the (untrained) context
    assert "web-search" in unmasked


def test_tool_result_turn_rendered_but_not_trained(tok):
    text, masked, unmasked, _ = _render(tok, CONV, TOOLS)
    assert "TOOLRESULTMARKER" in text
    assert "TOOLRESULTMARKER" in unmasked
    assert "TOOLRESULTMARKER" not in masked


def test_final_answer_trained(tok):
    _, masked, _, _ = _render(tok, CONV, TOOLS)
    assert "It is rainy, about 12C." in masked


def test_string_and_structured_schemas_render_identically(tok):
    # legacy: tool_calls/tools as JSON strings; hybrid: structured with string arguments
    legacy_conv = [dict(m) for m in CONV]
    legacy_conv[2] = {**legacy_conv[2], "tool_calls": json.dumps(CALL)}
    hybrid_call = [
        {**CALL[0], "function": {**CALL[0]["function"], "arguments": json.dumps(CALL[0]["function"]["arguments"])}}
    ]
    hybrid_conv = [dict(m) for m in CONV]
    hybrid_conv[2] = {**hybrid_conv[2], "tool_calls": hybrid_call}
    hybrid_tools = [
        {
            "type": "function",
            "function": {
                "name": "web-search",
                "description": "Search the web.",
                "parameters": json.dumps(TOOLS[0]["function"]["parameters"]),
                "strict": None,  # Arrow-struct decode artifact for an absent field
            },
        }
    ]

    _, _, _, ids_ref = _render(tok, CONV, TOOLS)
    _, _, _, ids_legacy = _render(tok, legacy_conv, json.dumps(TOOLS))
    _, _, _, ids_hybrid = _render(tok, hybrid_conv, hybrid_tools)
    assert ids_legacy == ids_ref
    assert ids_hybrid == ids_ref


def test_no_silent_char_iteration(tok):
    # the catastrophic failure mode: a string reaching the template must never
    # render per-character empty function blocks
    text, _, _, _ = _render(tok, CONV, TOOLS)
    assert "<function=>" not in text
