# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""Held-out sets: fixed data that a run evaluates between training steps, each pass over the whole set.

Catalogue:
    config: the ``held_out:`` block of a training config.
    evaluation: a held-out set at run time, and the builder of the sets that the block names.

The masked validation of token masking (``token_masking.validation``) builds the same evaluation.
See docs/training/held-out-validation.md.
"""
