"""The exporter's fused-QKV split must agree with the bridge's own splitter.

`pipeline_lora_export_factors.py` turns one Megatron `linear_qkv` LoRA factor into three
HF keys by slicing lora_B's rows. Megatron interleaves those rows by query group, so the
split is a permutation as well as a partition — and a wrong permutation preserves every
shape, so the exporter's key-set and geometry gates both pass on it. This model has
num_query_groups=2 against TP=4, the regime where such an error is most plausible.

The check compares against `split_qkv_weights`, the function the bridge itself uses, on a
tensor whose every row carries its own index so no permutation can cancel out.

`split_qkv_weights` cannot be called on lora_B directly: it scales head_size by
hidden_size/qkv.shape[-1] and raises on a rank-width tensor. The row order does not depend
on the column count, so the comparison is done at full width.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

from megatron.bridge.models.conversion.param_mapping import split_qkv_weights

# The base checkpoint's own run_config.yaml, not the provider defaults.
CFG = SimpleNamespace(
    num_attention_heads=32,
    num_query_groups=2,
    kv_channels=128,
    hidden_size=4096,
    attention_output_gate=False,
)


def _exporter_split(qkv: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The row arithmetic of pipeline_lora_export_factors.py, kept in step with it."""
    n_heads, n_groups, head_dim = CFG.num_attention_heads, CFG.num_query_groups, CFG.kv_channels
    hpg = n_heads // n_groups
    block = (hpg + 2) * head_dim
    qs, ks, vs = [], [], []
    for g in range(n_groups):
        base = g * block
        qs.append(qkv[base : base + hpg * head_dim])
        ks.append(qkv[base + hpg * head_dim : base + (hpg + 1) * head_dim])
        vs.append(qkv[base + (hpg + 1) * head_dim : base + block])
    return torch.cat(qs), torch.cat(ks), torch.cat(vs)


def test_exporter_qkv_split_matches_the_bridge_row_for_row() -> None:
    qkv_rows = CFG.num_attention_heads * CFG.kv_channels + 2 * CFG.num_query_groups * CFG.kv_channels
    qkv = torch.arange(qkv_rows, dtype=torch.float32).unsqueeze(1).repeat(1, CFG.hidden_size)
    expected = split_qkv_weights(CFG, qkv)
    for name, mine, ref in zip("qkv", _exporter_split(qkv), expected, strict=True):
        assert mine.shape == ref.shape, f"{name}: {tuple(mine.shape)} != {tuple(ref.shape)}"
        assert torch.equal(mine, ref), (
            f"{name} rows differ from the bridge: mine {mine[:4, 0].tolist()} "
            f"vs {ref[:4, 0].tolist()}"
        )


def test_the_widths_are_what_the_reference_adapter_carries() -> None:
    """q is all heads; k and v are query groups only. A GQA model where these coincide
    would make the test above pass on a broken split, so the asymmetry is pinned."""
    qkv_rows = CFG.num_attention_heads * CFG.kv_channels + 2 * CFG.num_query_groups * CFG.kv_channels
    q, k, v = _exporter_split(torch.zeros(qkv_rows, CFG.hidden_size))
    assert q.shape[0] == 4096
    assert k.shape[0] == v.shape[0] == 256
    assert k.shape[0] != q.shape[0], "GQA asymmetry is what makes the split non-trivial"
