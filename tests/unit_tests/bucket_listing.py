# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""The stand-in for one item of a Hub bucket's listing, for tests of the code that reads the listing.

The training container's huggingface_hub has no bucket API and the real one reads the Hub, so the tests of
``scripts/hub`` answer ``list_bucket_tree`` with these items.

Catalogue:
    BucketItem: one listed path, with its type (``file`` or ``directory``) and size.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BucketItem:
    type: str
    path: str
    size: int
