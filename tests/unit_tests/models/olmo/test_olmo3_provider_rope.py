# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""CPU tests of the OLMo-3 rope pair and the per-layer rope selection (``olmo3_provider``).

The packed (THD) training path hands ``Olmo3RotaryEmbedding.forward`` its sequence length as a 0-d integer tensor, and
``Olmo3SelfAttention.forward`` picks the sliding or the full-attention rope from the stacked pair. Neither needs a GPU
for these checks: the rope module is built with ``use_cpu_initialization=True`` and the selection is a pure function.
"""

import inspect

import pytest
import torch
from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding

from megatron.bridge.models.olmo import olmo3_provider
from megatron.bridge.models.olmo.olmo3_provider import (
    Olmo3ModelProvider32B,
    Olmo3RotaryEmbedding,
    Olmo3SelfAttention,
)


KV_CHANNELS = 16
SEQ_LEN = 64


def _rope() -> Olmo3RotaryEmbedding:
    return Olmo3RotaryEmbedding(
        kv_channels=KV_CHANNELS,
        rotary_percent=1.0,
        rotary_interleaved=False,
        seq_len_interpolation_factor=None,
        rotary_base=500000.0,
        use_cpu_initialization=True,
        yarn_scaling_factor=8.0,
        yarn_original_max_position_embeddings=16,
    )


class TestRotaryEmbeddingPair:
    def test_forward_stacks_the_sliding_and_the_full_rope(self):
        rope = _rope()
        stacked = rope.forward(SEQ_LEN, packed_seq=True)
        assert stacked.shape == (2, SEQ_LEN, 1, 1, KV_CHANNELS)
        sliding = RotaryEmbedding.forward(rope, SEQ_LEN, 0, True, None)
        full, _ = rope.rope_full.forward(SEQ_LEN, 0, True, None)
        assert torch.equal(stacked[0], sliding)
        assert torch.equal(stacked[1], full)
        assert not torch.equal(stacked[0], stacked[1]), "YaRN must change the full-attention frequencies"

    def test_a_tensor_sequence_length_hits_the_cache_on_the_second_call(self):
        """The Bridge passes ``max_seqlen`` as a 0-d int32 tensor. A tensor hashes by identity, so keying the cache on
        it never hits and each entry (about 67 MB at 32,768 positions) stays alive until 32 are held."""
        rope = _rope()
        Olmo3RotaryEmbedding._forward_cached.cache_clear()
        first = rope.forward(torch.tensor(SEQ_LEN, dtype=torch.int32), packed_seq=True)
        second = rope.forward(torch.tensor(SEQ_LEN, dtype=torch.int32), packed_seq=True)
        info = Olmo3RotaryEmbedding._forward_cached.cache_info()
        assert (info.hits, info.misses) == (1, 1)
        assert torch.equal(first, second)

    def test_an_integer_and_a_tensor_length_give_the_same_rope(self):
        rope = _rope()
        by_int = rope.forward(SEQ_LEN, packed_seq=True)
        by_tensor = rope.forward(torch.tensor(SEQ_LEN, dtype=torch.int32), packed_seq=True)
        assert torch.equal(by_int, by_tensor)


class TestSelectLayerRope:
    def test_picks_the_sliding_slice_for_a_sliding_layer_and_the_full_slice_otherwise(self):
        stacked = _rope().forward(SEQ_LEN, packed_seq=True)
        assert torch.equal(olmo3_provider.select_layer_rope(stacked, is_sliding=True), stacked[0])
        assert torch.equal(olmo3_provider.select_layer_rope(stacked, is_sliding=False), stacked[1])

    def test_a_plain_rope_of_length_two_is_not_taken_for_the_stacked_pair(self):
        """A single rope tensor is [S, 1, 1, d]. With S = 2 its first dimension is 2, as the stacked pair's is."""
        plain = torch.arange(2 * KV_CHANNELS, dtype=torch.float32).reshape(2, 1, 1, KV_CHANNELS)
        assert olmo3_provider.select_layer_rope(plain, is_sliding=True) is plain
        assert olmo3_provider.select_layer_rope(plain, is_sliding=False) is plain

    @pytest.mark.parametrize("passthrough", [None, (torch.zeros(1), torch.zeros(1))])
    def test_values_that_are_not_tensors_pass_through(self, passthrough):
        assert olmo3_provider.select_layer_rope(passthrough, is_sliding=True) is passthrough


class TestFusedRopeWithYarn:
    def test_a_yarn_model_refuses_the_fused_rope_before_it_builds_anything(self):
        provider = Olmo3ModelProvider32B(yarn_rotary_scaling_factor=8.0, apply_rope_fusion=True)
        with pytest.raises(ValueError, match="the fused rope drops the YaRN attention scale"):
            provider.provide()


class TestSelfAttentionConstructor:
    def test_the_constructor_names_every_argument_the_layer_passes(self):
        """``TransformerLayer`` passes ``pp_layer_offset`` and ``name`` to its attention module. A ``**kwargs`` that is
        not forwarded drops them without a sound, so the constructor lists them and takes no catch-all."""
        parameters = inspect.signature(Olmo3SelfAttention.__init__).parameters
        assert "pp_layer_offset" in parameters
        assert "name" in parameters
        assert not any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values())
