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

"""A resumed fine-tuning run reads the samples an uninterrupted run reads.

The training loop wraps the 'batch' dataloader in ``cyclic_iter``, which re-iterates the same sampler at every epoch
boundary. The sampler therefore has to finish the epoch it was resumed in and start every later epoch at the first
sample, wherever the resume point falls: inside the first epoch, on an epoch boundary or in a later epoch, with the
partial batch dropped or padded. Its length has to count what its next pass yields, and the mode the loader builder
never selects, the partial batch kept unpadded, has to restart at the first sample too.
"""

import itertools

import pytest
from torch.utils.data import Dataset

from megatron.bridge.data.loaders import cyclic_iter
from megatron.bridge.data.samplers import MegatronPretrainingBatchSampler, build_pretraining_data_loader


GLOBAL_BATCH_SIZE = 4
# More passes than any read below needs; a pass that yields nothing then fails the read instead of spinning in
# cyclic_iter.
MAX_PASSES = 8


class _Positions(Dataset):
    """Sample i is the integer i, so each batch shows which samples it holds."""

    def __init__(self, size: int) -> None:
        self.size = size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> int:
        return index


class _CountedPasses:
    """The loader as cyclic_iter re-iterates it, failing once it has been iterated more than ``MAX_PASSES`` times."""

    def __init__(self, loader) -> None:
        self.loader = loader
        self.passes = 0

    def __iter__(self):
        self.passes += 1
        assert self.passes <= MAX_PASSES, f"the loader was re-iterated {self.passes} times without filling the read"
        return iter(self.loader)


def _loader(consumed_samples: int, *, total: int, drop_last: bool, data_parallel_rank: int, data_parallel_size: int):
    """The 'batch' loader the training setup builds."""
    return build_pretraining_data_loader(
        _Positions(total),
        consumed_samples=consumed_samples,
        dataloader_type="batch",
        micro_batch_size=1,
        num_workers=0,
        data_sharding=False,
        collate_fn=list,
        pin_memory=False,
        data_parallel_rank=data_parallel_rank,
        data_parallel_size=data_parallel_size,
        drop_last=drop_last,
        global_batch_size=GLOBAL_BATCH_SIZE,
    )


def _batches(consumed_samples: int, *, count: int, **loader_args) -> list:
    """The first ``count`` batches a rank reads through the training loop's ``cyclic_iter``."""
    return list(itertools.islice(cyclic_iter(_CountedPasses(_loader(consumed_samples, **loader_args))), count))


# 12 samples are three whole batches; 13 leave a partial batch, which drop_last discards and padding fills.
@pytest.mark.parametrize("total", [12, 13])
@pytest.mark.parametrize("drop_last", [True, False])
@pytest.mark.parametrize("data_parallel_size", [1, 2])
def test_a_resumed_run_reads_what_an_uninterrupted_run_reads(total, drop_last, data_parallel_size):
    batches_per_epoch = total // GLOBAL_BATCH_SIZE if drop_last else -(-total // GLOBAL_BATCH_SIZE)
    # Inside the first epoch, on the first epoch boundary, just past it, and inside the third epoch.
    resume_points = [1, 2, batches_per_epoch, batches_per_epoch + 1, 2 * batches_per_epoch + 1]
    read = 4 * batches_per_epoch
    for rank in range(data_parallel_size):
        loader_args = dict(
            total=total, drop_last=drop_last, data_parallel_rank=rank, data_parallel_size=data_parallel_size
        )
        uninterrupted = _batches(0, count=read, **loader_args)
        for resumed_after in resume_points:
            resumed = _batches(resumed_after * GLOBAL_BATCH_SIZE, count=read - resumed_after, **loader_args)
            assert resumed == uninterrupted[resumed_after:], f"rank {rank}, resumed after {resumed_after} batches"


@pytest.mark.parametrize(
    ("drop_last", "one_epoch"),
    [
        (True, [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]),
        (False, [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, -1, -1, -1]]),
    ],
)
def test_every_epoch_starts_at_the_first_sample(drop_last, one_epoch):
    """An uninterrupted run reads the samples in order, epoch after epoch, dropping the partial batch or padding it.

    A resumed run falls back into step with an uninterrupted one whatever the epoch length, because both advance by a
    global batch per batch, so only the order itself pins where an epoch ends."""
    read = _batches(
        0, count=2 * len(one_epoch), total=13, drop_last=drop_last, data_parallel_rank=0, data_parallel_size=1
    )
    assert read == 2 * one_epoch


@pytest.mark.parametrize("total", [12, 13])
@pytest.mark.parametrize("drop_last", [True, False])
@pytest.mark.parametrize("data_parallel_size", [1, 2])
def test_the_length_is_what_the_next_pass_yields(total, drop_last, data_parallel_size):
    """A resumed loader's length counts the batches its next pass yields: the rest of the epoch it was resumed in, or a
    whole epoch when it was resumed on an epoch boundary."""
    for consumed_samples in range(0, 12 * GLOBAL_BATCH_SIZE + 1, GLOBAL_BATCH_SIZE):
        loader = _loader(
            consumed_samples,
            total=total,
            drop_last=drop_last,
            data_parallel_rank=0,
            data_parallel_size=data_parallel_size,
        )
        assert len(loader) == len(list(loader)), f"resumed after {consumed_samples} samples"


def test_an_unpadded_partial_batch_ends_each_epoch():
    """With the partial batch kept and not padded (the sampler's default, which the loader builder never selects), a
    re-iterated sampler reads the dataset in order epoch after epoch, the partial batch closing each one."""
    sampler = MegatronPretrainingBatchSampler(
        total_samples=13,
        consumed_samples=0,
        micro_batch_size=1,
        global_batch_size=GLOBAL_BATCH_SIZE,
        data_parallel_rank=0,
        data_parallel_size=1,
        drop_last=False,
    )
    one_epoch = [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12]]
    assert len(sampler) == len(one_epoch)
    assert list(itertools.islice(cyclic_iter(_CountedPasses(sampler)), 2 * len(one_epoch))) == 2 * one_epoch


def test_fewer_samples_than_one_global_batch_is_refused_when_the_partial_batch_is_dropped():
    loader = _loader(0, total=GLOBAL_BATCH_SIZE - 1, drop_last=True, data_parallel_rank=0, data_parallel_size=1)
    with pytest.raises(AssertionError, match="at least one full global batch"):
        next(iter(loader))
