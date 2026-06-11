# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# Tests for JSON-string tool_calls/tools normalization in chat preprocessing.
#
# Datasets with a uniform string parquet schema store `tool_calls` (and `tools`)
# as JSON-encoded strings. Jinja chat templates iterate `message.tool_calls`
# directly, and iterating a *string* yields one character per "tool call" —
# silently rendering thousands of empty <tool_call><function=></function>
# blocks per turn (observed: 4-5x length blowup, mass truncation, and a
# deterministic forward-loss NaN on Nemotron-H Super at 32K).

import json

import pytest

from megatron.bridge.data.datasets.utils import (
    _convert_to_openai_messages,
    _normalize_message_tool_calls,
    _normalize_tools_parameters,
    _parse_json_field,
)

TOOL_CALLS = [
    {
        "function": {"arguments": {"query": "Liverpool Biennial 2023 curator"}, "name": "web-search"},
        "id": "chatcmpl-tool-1",
        "type": "function",
    }
]


def test_string_tool_calls_parsed_to_list():
    msg = {"role": "assistant", "content": "", "tool_calls": json.dumps(TOOL_CALLS)}
    out = _normalize_message_tool_calls(msg)
    assert out["tool_calls"] == TOOL_CALLS
    # original message not mutated
    assert isinstance(msg["tool_calls"], str)


def test_list_tool_calls_passthrough():
    msg = {"role": "assistant", "content": "", "tool_calls": TOOL_CALLS}
    assert _normalize_message_tool_calls(msg) is msg


def test_hybrid_list_with_string_arguments_parsed():
    # OpenAI-convention hybrid: structured tool_calls, arguments JSON-encoded.
    calls = [{"function": {"arguments": json.dumps({"q": "x"}), "name": "search"}, "id": "c1", "type": "function"}]
    msg = {"role": "assistant", "content": "", "tool_calls": calls}
    out = _normalize_message_tool_calls(msg)
    assert out["tool_calls"][0]["function"]["arguments"] == {"q": "x"}
    # source list not mutated
    assert isinstance(calls[0]["function"]["arguments"], str)


def test_empty_list_tool_calls_dropped():
    msg = {"role": "assistant", "content": "hi", "tool_calls": []}
    out = _normalize_message_tool_calls(msg)
    assert "tool_calls" not in out


def test_empty_string_tool_calls_dropped():
    msg = {"role": "assistant", "content": "hi", "tool_calls": ""}
    out = _normalize_message_tool_calls(msg)
    assert "tool_calls" not in out


def test_empty_list_string_tool_calls_dropped():
    msg = {"role": "assistant", "content": "hi", "tool_calls": "[]"}
    out = _normalize_message_tool_calls(msg)
    assert "tool_calls" not in out


def test_invalid_json_tool_calls_raises():
    msg = {"role": "assistant", "content": "", "tool_calls": "<tool_call>not json"}
    with pytest.raises(ValueError, match="tool_calls"):
        _normalize_message_tool_calls(msg)


def test_non_list_json_tool_calls_raises():
    msg = {"role": "assistant", "content": "", "tool_calls": '{"function": {}}'}
    with pytest.raises(ValueError, match="must parse to a list"):
        _normalize_message_tool_calls(msg)


def test_string_function_arguments_parsed():
    calls = [{"function": {"arguments": json.dumps({"q": "x"}), "name": "search"}, "type": "function"}]
    msg = {"role": "assistant", "content": "", "tool_calls": json.dumps(calls)}
    out = _normalize_message_tool_calls(msg)
    assert out["tool_calls"][0]["function"]["arguments"] == {"q": "x"}


def test_convert_to_openai_messages_normalizes_all_turns():
    source = {
        "messages": [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "", "tool_calls": json.dumps(TOOL_CALLS)},
            {"role": "tool", "content": "result"},
            {"role": "assistant", "content": "answer"},
        ]
    }
    chat = _convert_to_openai_messages(source)
    assert chat[1]["tool_calls"] == TOOL_CALLS
    assert all(not isinstance(m.get("tool_calls"), str) for m in chat)


def test_parse_json_field_tools():
    tools = [{"type": "function", "function": {"name": "web-search", "parameters": {}}}]
    assert _parse_json_field(json.dumps(tools), "tools") == tools
    with pytest.raises(ValueError, match="tools"):
        _parse_json_field("{bad", "tools")


def test_tools_string_parameters_parsed():
    params = {"type": "object", "properties": {"q": {"type": "string"}}}
    tools = [{"type": "function", "function": {"name": "search", "parameters": json.dumps(params)}}]
    out = _normalize_tools_parameters(tools)
    assert out[0]["function"]["parameters"] == params
    # source not mutated
    assert isinstance(tools[0]["function"]["parameters"], str)


def test_tools_mapping_parameters_passthrough():
    params = {"type": "object", "properties": {}}
    tools = [{"type": "function", "function": {"name": "search", "parameters": params}}]
    assert _normalize_tools_parameters(tools) is tools


def test_arrow_none_fields_stripped():
    # Arrow struct decode materializes absent optional fields as None; Jinja treats
    # None as defined, rendering e.g. <strict>None</strict>. Must strip at struct level.
    tools = [{"type": "function", "function": {"name": "s", "parameters": "{}", "strict": None, "description": None}}]
    out = _normalize_tools_parameters(tools)
    assert "strict" not in out[0]["function"] and "description" not in out[0]["function"]
    calls = [{"id": None, "type": None, "function": {"name": "s", "arguments": '{"a": null}'}}]
    msg = _normalize_message_tool_calls({"role": "assistant", "tool_calls": calls})
    c = msg["tool_calls"][0]
    assert "id" not in c and "type" not in c
    # genuine JSON nulls inside arguments are preserved
    assert c["function"]["arguments"] == {"a": None}
