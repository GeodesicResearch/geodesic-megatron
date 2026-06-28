# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

"""Unit tests for the parallelism-dimension resolve/log/record helpers."""

from __future__ import annotations

from unittest import mock

from megatron.bridge.training.utils.parallelism_utils import (
    format_parallelism_dims,
    parallelism_wandb_summary,
    record_parallelism_if_resolved,
    record_parallelism_to_wandb,
    resolve_parallelism_dims,
)


# A representative resolved layout: 64 GPUs, parallel-folded TP2/EP4, PP8, CP1 -> DP4.
_DIMS = {
    "world_size": 64,
    "data_parallel_size": 4,
    "tensor_model_parallel_size": 2,
    "pipeline_model_parallel_size": 8,
    "context_parallel_size": 1,
    "expert_model_parallel_size": 4,
}


def test_resolve_parallelism_dims_reads_parallel_state():
    with (
        mock.patch("torch.distributed.get_world_size", return_value=64),
        mock.patch("megatron.core.parallel_state.get_data_parallel_world_size", return_value=4),
        mock.patch("megatron.core.parallel_state.get_tensor_model_parallel_world_size", return_value=2),
        mock.patch("megatron.core.parallel_state.get_pipeline_model_parallel_world_size", return_value=8),
        mock.patch("megatron.core.parallel_state.get_context_parallel_world_size", return_value=1),
        mock.patch("megatron.core.parallel_state.get_expert_model_parallel_world_size", return_value=4),
    ):
        assert resolve_parallelism_dims() == _DIMS


def test_format_parallelism_dims():
    assert format_parallelism_dims(_DIMS) == "> resolved parallelism: world_size=64 | DP=4 TP=2 PP=8 CP=1 EP=4"


def test_parallelism_wandb_summary_namespaces_keys():
    assert parallelism_wandb_summary(_DIMS) == {
        "parallelism/world_size": 64,
        "parallelism/data_parallel_size": 4,
        "parallelism/tensor_model_parallel_size": 2,
        "parallelism/pipeline_model_parallel_size": 8,
        "parallelism/context_parallel_size": 1,
        "parallelism/expert_model_parallel_size": 4,
    }


def test_record_parallelism_to_wandb_writes_summary():
    run = mock.MagicMock()
    record_parallelism_to_wandb(run, _DIMS)
    run.summary.update.assert_called_once_with(parallelism_wandb_summary(_DIMS))


def test_record_parallelism_to_wandb_none_run_is_noop():
    # No active W&B run on this rank -> must not raise.
    record_parallelism_to_wandb(None, _DIMS)


def test_record_parallelism_if_resolved_skips_when_mp_uninitialized():
    run = mock.MagicMock()
    with mock.patch("megatron.core.parallel_state.model_parallel_is_initialized", return_value=False):
        result = record_parallelism_if_resolved(run)
    assert result is None
    run.summary.update.assert_not_called()


def test_record_parallelism_if_resolved_records_when_initialized():
    run = mock.MagicMock()
    with (
        mock.patch("megatron.core.parallel_state.model_parallel_is_initialized", return_value=True),
        mock.patch("torch.distributed.get_world_size", return_value=64),
        mock.patch("megatron.core.parallel_state.get_data_parallel_world_size", return_value=4),
        mock.patch("megatron.core.parallel_state.get_tensor_model_parallel_world_size", return_value=2),
        mock.patch("megatron.core.parallel_state.get_pipeline_model_parallel_world_size", return_value=8),
        mock.patch("megatron.core.parallel_state.get_context_parallel_world_size", return_value=1),
        mock.patch("megatron.core.parallel_state.get_expert_model_parallel_world_size", return_value=4),
    ):
        result = record_parallelism_if_resolved(run)
    assert result == _DIMS
    run.summary.update.assert_called_once_with(parallelism_wandb_summary(_DIMS))
