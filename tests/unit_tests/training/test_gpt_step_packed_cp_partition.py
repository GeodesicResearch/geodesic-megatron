# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""The context-parallel partition of a packed batch must hand each rank the tokens the model believes it holds.

``_partition_packed_batch_for_cp`` gives every context-parallel rank its share of each packed document through
Transformer Engine's ``thd_get_partitioned_indices``. Attention and the Mamba layers read those shares through the
row ``get_packed_seq_params`` builds, and the Mamba layers put them back in sequence order with
``_undo_attention_load_balancing``; partition and consumers must agree token for token.

The batch comes from the packed collate, which pads every ``cu_seqlens`` row with -1 to the widest row in the batch
plus one, so a pack holding fewer documents than its neighbours ends in several pads. Given such a row, Transformer
Engine's binary search can land on a pad and hand every rank the pack's leading tokens. The kernel runs for real here
(no process group: the partition takes the rank as an argument), so the test needs a GPU.
"""

import pytest
import torch
from megatron.core.ssm.mamba_context_parallel import _undo_attention_load_balancing

from megatron.bridge.data.finetuning import split_batch_into_microbatches
from megatron.bridge.training.gpt_step import _partition_packed_batch_for_cp, get_batch_from_iterator
from megatron.bridge.training.utils.packed_seq_utils import get_packed_seq_params
from tests.unit_tests.packed_sft_batches import collate_packs, write_packs


pytestmark = pytest.mark.run_only_on("GPU")

PER_TOKEN_KEYS = ("tokens", "labels", "loss_mask", "position_ids", "padding_mask")

# The documents of each pack, by real-token count. As production's packs for context parallelism are, they are packed
# with every document padded to a multiple of PAD_SEQ_TO_MULT positions, its real tokens, its EOS and the padding after
# it: a document of 29 real tokens covers 32 positions and one of 5 covers 8. Collated together, the three packs pad to
# 48 positions and their cu_seqlens rows end in five, two and one -1: [0, 32, 48, -1, -1, -1, -1, -1],
# [0, 8, 16, 24, 32, 48, -1, -1] and [0, 8, 16, 24, 32, 40, 48, -1]. Every segment is a multiple of 8, the
# divisibility Transformer Engine's partition requires at context-parallel sizes 2 and 4.
PACKS = {
    "one_document": [29],
    "four_documents": [5, 5, 5, 5],
    "five_documents": [5, 5, 5, 5, 5],
}
PAD_SEQ_TO_MULT = 8
MAX_SEQ_LENGTH = 64


@pytest.fixture(scope="module")
def microbatches(tmp_path_factory) -> dict[str, dict]:
    """One microbatch per pack, as the trainer receives them: collated in one call, then split."""
    path = tmp_path_factory.mktemp("packs") / "packs.npy"
    write_packs(path, list(PACKS.values()), PAD_SEQ_TO_MULT, MAX_SEQ_LENGTH)
    batch = collate_packs(path, len(PACKS), PAD_SEQ_TO_MULT, MAX_SEQ_LENGTH, pad_to_max_length=False)
    return dict(zip(PACKS, split_batch_into_microbatches(batch, len(PACKS))))


def test_the_packs_collate_to_the_rows_the_cases_describe(microbatches):
    trailing_pads = {pack: int((mb["cu_seqlens"] == -1).sum()) for pack, mb in microbatches.items()}
    assert trailing_pads == {"one_document": 5, "four_documents": 2, "five_documents": 1}


@pytest.mark.parametrize("cp_size", [2, 4])
@pytest.mark.parametrize("pack", list(PACKS))
def test_each_cp_rank_holds_the_tokens_the_model_places_there(microbatches, pack, cp_size):
    full = get_batch_from_iterator(iter([microbatches[pack]]), is_first_pp_stage=True, is_last_pp_stage=True)
    total = full["tokens"].size(1)
    # A per-token tensor that names each position: the packed tokens repeat EOS in their padding, so they cannot.
    full["probe"] = torch.arange(total, device="cuda").unsqueeze(0)

    shards = [_partition_packed_batch_for_cp(dict(full), cp_size, cp_rank) for cp_rank in range(cp_size)]

    held = torch.cat([shard["probe"][0] for shard in shards])
    restored = _undo_attention_load_balancing(held, cp_size, get_packed_seq_params(full))
    assert torch.equal(restored, full["probe"][0]), (
        f"the ranks hold {[shard['probe'][0].tolist() for shard in shards]}, "
        f"which the model's packed layout does not place back in order"
    )
    for shard in shards:
        for key in PER_TOKEN_KEYS:
            assert torch.equal(shard[key][0], full[key][0][shard["probe"][0]]), key
