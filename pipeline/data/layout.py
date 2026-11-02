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
"""Where a prepared dataset and its packed parquet live on disk.

Imports only the standard library, so host-side code (configs/control_pretraining/corpora_table.py) can use it
outside the container.

Catalogue:
    slugify_dataset_name: the directory name of a dataset (and subset) under the output base.
    pack_dir: the directory a dataset root's packs for one tokenizer and padding multiple live in.
    packed_parquet_path: the packed parquet of one split and sequence length.
"""

from pathlib import Path


def slugify_dataset_name(dataset, subset=None):
    """Generate output directory name from dataset components.

    geodesic-research/Foo → geodesic-research__Foo
    geodesic-research/Foo + subset=bar → geodesic-research__Foo__bar
    """
    slug = dataset.replace("/", "__")
    if subset:
        slug = f"{slug}__{subset}"
    return slug


def pack_dir(dataset_root, tokenizer: str, pad_seq_to_mult: int) -> Path:
    """``<dataset_root>/packed/<tokenizer with / as -->_pad_seq_to_mult<N>``."""
    return Path(dataset_root) / "packed" / f"{tokenizer.replace('/', '--')}_pad_seq_to_mult{pad_seq_to_mult}"


def packed_parquet_path(dataset_root, tokenizer: str, pad_seq_to_mult: int, seq_length: int, split="training") -> Path:
    """``pack_dir(...)/<split>_<seq_length>.idx.parquet``; ``split`` is ``training`` or ``validation``."""
    return pack_dir(dataset_root, tokenizer, pad_seq_to_mult) / f"{split}_{seq_length}.idx.parquet"
