"""CPU regression for LoRA rows versus compressed FP8 scale rows."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


def split_function():
    # Load the pure tensor helper without initializing the CUDA/MCore stack.
    path = Path(__file__).resolve().parents[2] / "src/megatron/bridge/models/conversion/param_mapping.py"
    tree = ast.parse(path.read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "split_qkv_weights")
    namespace = {"torch": torch}
    source = "from __future__ import annotations\n" + ast.unparse(function)
    exec(compile(source, str(path), "exec"), namespace)
    return namespace["split_qkv_weights"]


@pytest.mark.parametrize("rank", [16, 32])
def test_lora_qkv_preserves_full_head_rows(rank):
    cfg = SimpleNamespace(hidden_size=2688, num_attention_heads=32, num_query_groups=2, kv_channels=128)
    b = torch.arange(36 * 128 * rank).reshape(36 * 128, rank)
    q, k, v = split_function()(cfg, b, feature_dim=rank)
    expected_q = torch.cat([b[: 16 * 128], b[18 * 128 : 34 * 128]])
    expected_k = torch.cat([b[16 * 128 : 17 * 128], b[34 * 128 : 35 * 128]])
    expected_v = torch.cat([b[17 * 128 : 18 * 128], b[35 * 128 :]])
    for actual, expected in zip((q, k, v), (expected_q, expected_k, expected_v)):
        assert torch.equal(actual, expected)


def test_fp8_scales_still_infer_compressed_rows():
    cfg = SimpleNamespace(hidden_size=4096, num_attention_heads=32, num_query_groups=8, kv_channels=128)
    scales = torch.arange(48 * 32).reshape(48, 32)
    q, k, v = split_function()(cfg, scales)
    assert q.shape == (32, 32)
    assert torch.equal(k, scales[4::6])
    assert torch.equal(v, scales[5::6])
