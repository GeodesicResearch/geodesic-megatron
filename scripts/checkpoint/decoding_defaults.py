# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""The decoding defaults an HF export's ``generation_config.json`` carries, by the export's mode.

A non-reasoning export (``--no-reasoning``: a base model, or an instruct/SFT model trained without think traces)
decodes greedily; a reasoning export samples at ``REASONING_GENERATION_DEFAULTS``. The exporter
(``pipeline_checkpoint_convert_hf.py`` ``apply_generation_defaults``) writes the file by this rule and the export check
(``scripts/checkpoint/export_check.py``) judges an export's file by it.

- ``GENERATION_CONFIG``: the file's name in an export.
- ``SAMPLING_KEYS``: the fields the rule removes before it sets a mode's defaults.
- ``REASONING_GENERATION_DEFAULTS``: what a reasoning export samples at.
- ``generation_config_text``: the file's text for a mode, from the text it held.

Standard library only.
"""

from __future__ import annotations

import json


GENERATION_CONFIG = "generation_config.json"
SAMPLING_KEYS = ("do_sample", "temperature", "top_p", "top_k")
REASONING_GENERATION_DEFAULTS = {"do_sample": True, "temperature": 0.6, "top_p": 0.95}


def generation_config_text(original: str, reasoning: bool) -> str:
    """The ``generation_config.json`` text of an export of the given mode, from the text ``original`` it held: every
    ``SAMPLING_KEYS`` field removed and, for a reasoning export, ``REASONING_GENERATION_DEFAULTS`` set; every other
    field kept; written with sorted keys, a two-space indent and a trailing newline. Raises ``json.JSONDecodeError``
    when ``original`` is not JSON."""
    config = json.loads(original)
    for key in SAMPLING_KEYS:
        config.pop(key, None)
    if reasoning:
        config.update(REASONING_GENERATION_DEFAULTS)
    return json.dumps(config, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
