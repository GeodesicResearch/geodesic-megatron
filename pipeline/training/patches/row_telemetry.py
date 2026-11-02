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
"""Packed-parquet row telemetry.

Catalogue:
    patch_packed_parquet_row_telemetry: log each packed-parquet fetch's dataset index and parquet row.
"""

import logging


logger = logging.getLogger(__name__)


def patch_packed_parquet_row_telemetry() -> bool:
    """Log every packed-parquet row fetch: dataset index -> underlying parquet row.

    Diagnostic for data-dependent failures (e.g. a deterministic NaN at a fixed
    iteration): with DP=1 the global batch of iteration k is exactly fetches
    64*(k-1)..64*k-1 in fetch order, so the log maps a failing iteration to the
    physical parquet rows it consumed. Gate: ISAMBARD_DATA_ROW_TELEMETRY=1.
    Idempotent; returns True only on first install.
    """
    from megatron.bridge.data.datasets.packed_parquet import GPTSFTPackedParquetDataset

    if getattr(GPTSFTPackedParquetDataset.__getitem__, "_isambard_row_telemetry", False):
        return False
    orig = GPTSFTPackedParquetDataset.__getitem__

    def getitem_with_telemetry(self, idx):
        mapped = idx
        sm = getattr(self, "samples_mapping", None)
        if sm is not None:
            try:
                mapped = int(sm[idx])
            except Exception:
                mapped = idx
        logger.info(f"[row-telemetry] fetch idx={idx} -> parquet_row={mapped}")
        return orig(self, idx)

    getitem_with_telemetry._isambard_row_telemetry = True
    GPTSFTPackedParquetDataset.__getitem__ = getitem_with_telemetry
    logger.info("Installed packed-parquet row telemetry: each fetch logs dataset idx -> parquet row.")
    return True
