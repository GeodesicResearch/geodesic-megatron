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

"""Packed SFT batches for unit tests, packed by the packer's own code and collated by the packed dataset.

A pack is given as the real-token counts of its documents. Each document is that many distinct token ids, none of them
EOS, followed by the EOS a tokenized example ends with. As the packer does, a document is padded by
``pad_document_for_packing`` when ``pad_seq_to_mult`` > 1, and the documents are packed in the given grouping by
``create_hist`` and ``fill_packing_strategy``. After the collate's label shift, a document of ``n`` real tokens covers
``n`` positions unpadded and ``ceil(n + 1, pad_seq_to_mult)`` padded, its EOS and the padding after it.

The packs are stored in either packed format, chosen by the file's suffix (``.npy`` or ``.parquet``), and read back
through ``create_sft_dataset``, the factory the training setup calls, which picks the dataset class by the same rule.
"""

import types

import numpy as np

from megatron.bridge.data.datasets.packed_parquet import write_packed_parquet
from megatron.bridge.data.datasets.packed_sequence import pad_document_for_packing
from megatron.bridge.data.datasets.packing_utils import create_hist, fill_packing_strategy
from megatron.bridge.data.datasets.sft import create_sft_dataset


EOS_ID = 2


def write_packs(path, packs: list[list[int]], pad_seq_to_mult: int, max_seq_length: int) -> None:
    """Store the packs at ``path`` in the packed format its suffix names: ``.npy`` or ``.parquet``."""
    documents, assignments = [], []
    next_token = 100
    for pack in packs:
        assignments.append([])
        for n_real in pack:
            # Boolean, as the tokenized examples' masks are: the packer's label shift appends False, and a pack
            # file's loss_mask column holds one type.
            document = {
                "input_ids": [*range(next_token, next_token + n_real), EOS_ID],
                "loss_mask": [True] * (n_real + 1),
            }
            next_token += n_real
            if pad_seq_to_mult > 1:
                pad_document_for_packing(document, max_seq_length, pad_seq_to_mult, EOS_ID)
            documents.append(document)
            assignments[-1].append(len(document["input_ids"]) - 1)
    sequences, _ = create_hist(documents, max_seq_length)
    rows = fill_packing_strategy(assignments, sequences, max_seq_length, EOS_ID)
    if str(path).endswith(".parquet"):
        write_packed_parquet(rows, path)
    else:
        np.save(path, np.array(rows, dtype=object), allow_pickle=True)


def collate_packs(path, pack_count: int, pad_seq_to_mult: int, max_seq_length: int, pad_to_max_length: bool) -> dict:
    """The packs stored at ``path`` collated in one call, as one data-parallel replica's packs are."""
    # The packed collate reads nothing from the tokenizer but eos_id, and a real tokenizer would need a Hub download.
    dataset = create_sft_dataset(
        path,
        tokenizer=types.SimpleNamespace(eos_id=EOS_ID),
        seq_length=max_seq_length,
        pad_seq_to_mult=pad_seq_to_mult,
        pad_to_max_length=pad_to_max_length,
    )
    return dataset.collate_fn([dataset[i] for i in range(pack_count)])
