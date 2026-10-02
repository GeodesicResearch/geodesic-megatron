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

"""The length the packer stores a tokenized document at, when packs are padded to a multiple."""

from megatron.bridge.data.datasets.packed_sequence import pad_document_for_packing


PAD_ID = 2


def _document(n: int, loss_mask_value) -> dict:
    return {
        "input_ids": list(range(10, 10 + n)),
        "context_ids": list(range(10, 10 + n)),
        "loss_mask": [loss_mask_value] * n,
    }


class TestPadDocumentForPacking:
    def test_pads_to_the_next_multiple_plus_the_token_the_label_shift_drops(self):
        document = _document(6, 1)
        pad_document_for_packing(document, max_seq_length=64, pad_seq_to_mult=4, pad_id=PAD_ID)
        assert document["input_ids"] == list(range(10, 16)) + [PAD_ID] * 3
        assert document["context_ids"] == list(range(10, 16)) + [PAD_ID] * 3
        assert document["loss_mask"] == [1] * 6 + [0] * 3

    def test_a_document_at_a_multiple_gains_only_the_token_the_label_shift_drops(self):
        document = _document(8, True)
        pad_document_for_packing(document, max_seq_length=64, pad_seq_to_mult=4, pad_id=PAD_ID)
        assert document["input_ids"] == list(range(10, 18)) + [PAD_ID]
        assert document["loss_mask"] == [True] * 8 + [False]

    def test_the_multiple_is_capped_at_the_maximum_length(self):
        document = _document(30, 1)
        pad_document_for_packing(document, max_seq_length=31, pad_seq_to_mult=8, pad_id=PAD_ID)
        # The next multiple of 8, 32, is above the maximum: the document is padded to 31 and the label shift's token.
        assert document["input_ids"] == list(range(10, 40)) + [PAD_ID] * 2

    def test_a_document_longer_than_the_maximum_is_cut_to_it(self):
        document = _document(40, 1)
        pad_document_for_packing(document, max_seq_length=32, pad_seq_to_mult=8, pad_id=PAD_ID)
        assert document["input_ids"] == list(range(10, 42))
        assert document["context_ids"] == list(range(10, 42))
        assert document["loss_mask"] == [1] * 32
