# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""Token masking: no training loss at any target position whose label is a listed token id.

The configuration lives in ``config``, the per-run decision in ``resolution``, the per-microbatch mask and statistics
in ``hook``, the per-iteration checks in ``monitor`` and the setup-time checks on the training data in ``data_check``.
See docs/training/token-masking.md.
"""
