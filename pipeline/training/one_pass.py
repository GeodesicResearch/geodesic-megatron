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
"""Sizing a packed SFT run to one pass over its rows.

Catalogue:
    ONE_PASS_KEY: the top-level YAML section that asks for it.
    ONE_PASS_DERIVED_FIELDS: the settings it derives, which a config must not also set.
    packed_parquet_row_count: the rows of the packed parquet files a spec resolves to.
    apply_one_pass: set a resolved config's iterations, samples and save interval to one pass.
"""

import logging

from megatron.bridge.data.datasets.packed_parquet import PackedParquetSpec, resolve_packed_parquet_paths
from megatron.bridge.training.config import ConfigContainer


logger: logging.Logger = logging.getLogger(__name__)


# Top-level YAML section that sizes a packed SFT run to one pass over its data (see ``apply_one_pass``).
# It is not a ConfigContainer field, so resolve_training_config pops it before the merge.
ONE_PASS_KEY = "one_pass"
# The settings one_pass derives, as (section, key): a config naming one of them as well is ambiguous.
ONE_PASS_DERIVED_FIELDS = (("train", "train_iters"), ("checkpoint", "save_interval"), ("dataset", "max_train_samples"))


def packed_parquet_row_count(spec: PackedParquetSpec) -> int:
    """Number of rows in the packed parquet files a spec resolves to, read from their parquet metadata."""
    import pyarrow.parquet as pq

    return sum(pq.read_metadata(path).num_rows for path in resolve_packed_parquet_paths(spec))


def apply_one_pass(
    cfg: ConfigContainer, one_pass: dict, mode: str, yaml_overrides: dict, cli_overrides: list[str]
) -> None:
    """Size a packed SFT run to one shuffled pass over its rows, in place.

    Sets ``train.train_iters`` to the packed rows divided by ``train.global_batch_size`` (the remainder is
    dropped), ``dataset.max_train_samples`` to the rows those iterations read (which makes the packed dataset
    read a seeded shuffle of all its files' rows rather than file order), and ``checkpoint.save_interval`` to
    ``round(save_interval_fraction * train_iters)``. Reads the parquet metadata of
    ``dataset.packed_sequence_specs.packed_train_data_path``; logs the rows, iterations and dropped rows.

    Args:
        cfg: The resolved config; its train, dataset and checkpoint sections are modified.
        one_pass: The YAML's ``one_pass`` section: exactly ``{save_interval_fraction: <float in (0, 1]>}``.
        mode: The training mode; only ``sft`` is supported.
        yaml_overrides: The override YAML, checked for the fields this derives.
        cli_overrides: The Hydra overrides, checked for the same fields.

    Raises:
        ValueError: On a mode other than sft, no packed_train_data_path, a malformed section, a batch-size rampup,
            or a derived field also set in the YAML or on the command line.
    """
    if mode != "sft":
        raise ValueError(f"{ONE_PASS_KEY} sizes packed SFT runs only, not mode {mode!r}")
    if not isinstance(one_pass, dict) or set(one_pass) != {"save_interval_fraction"}:
        raise ValueError(f"{ONE_PASS_KEY} must be exactly {{save_interval_fraction: <float>}}, got {one_pass!r}")
    fraction = float(one_pass["save_interval_fraction"])
    if not 0 < fraction <= 1:
        raise ValueError(f"{ONE_PASS_KEY}.save_interval_fraction must be in (0, 1], got {fraction}")
    for section, key in ONE_PASS_DERIVED_FIELDS:
        dotted = f"{section}.{key}"
        if key in (yaml_overrides.get(section) or {}) or any(
            override.lstrip("+").split("=", 1)[0] == dotted for override in cli_overrides
        ):
            raise ValueError(
                f"{dotted} is derived by {ONE_PASS_KEY}; set it or {ONE_PASS_KEY}, not both "
                "(train.exit_interval stops a one-pass run early without changing its length)"
            )
    if cfg.train.rampup_batch_size is not None:
        raise ValueError(f"{ONE_PASS_KEY} counts iterations at a fixed global batch; unset train.rampup_batch_size")
    specs = getattr(cfg.dataset, "packed_sequence_specs", None)
    spec = getattr(specs, "packed_train_data_path", None)
    if spec is None:
        raise ValueError(f"{ONE_PASS_KEY} needs dataset.packed_sequence_specs.packed_train_data_path")

    rows = packed_parquet_row_count(spec)
    global_batch_size = cfg.train.global_batch_size
    train_iters = rows // global_batch_size
    if train_iters == 0:
        raise ValueError(f"{ONE_PASS_KEY}: {rows} packed rows fill no global batch of {global_batch_size}")
    cfg.train.train_iters = train_iters
    cfg.dataset.max_train_samples = train_iters * global_batch_size
    cfg.checkpoint.save_interval = max(1, round(fraction * train_iters))
    logger.info(
        "%s: %d packed rows / global batch %d = %d iterations (%d rows dropped); save_interval %d",
        ONE_PASS_KEY,
        rows,
        global_batch_size,
        train_iters,
        rows - train_iters * global_batch_size,
        cfg.checkpoint.save_interval,
    )
