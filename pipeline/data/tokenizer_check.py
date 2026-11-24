# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""Is a loaded Hugging Face tokenizer the one the ``tokenizer.json`` beside it describes?

transformers 5.1 to 5.5 (the container has 5.3.0) build a plain GPT-2 byte-level pre-tokenizer for a
``tokenizer_class`` of ``GPT2Tokenizer`` and ignore the pre-tokenizer in ``tokenizer.json``. Olmo 3's has the
cl100k-style Split regex (digits in groups of up to three, its own newline and space rules), so the container's ids
differ from the ones transformers 4.57 gives, by 4 to 20 percent more tokens on long math and code, and a corpus
tokenized or packed from that directory holds ids no Olmo model was trained on. A directory whose
``tokenizer_config.json`` names ``PreTrainedTokenizerFast`` keeps the pre-tokenizer of ``tokenizer.json``.
"""

import json
from pathlib import Path
from typing import Any


class TokenizerMismatch(RuntimeError):
    """The loaded tokenizer's pre-tokenizer is not the one in the ``tokenizer.json`` of its directory."""


def require_pre_tokenizer_of_file(hf_tokenizer: Any, tokenizer_dir: str | Path) -> None:
    """Raise ``TokenizerMismatch`` unless ``hf_tokenizer`` (a fast Hugging Face tokenizer loaded from the local directory
    ``tokenizer_dir``) has the pre-tokenizer that ``tokenizer_dir/tokenizer.json`` states. A directory without a
    ``tokenizer.json`` has nothing to compare with, and the caller passes only a local directory."""
    from tokenizers import Tokenizer

    file = Path(tokenizer_dir) / "tokenizer.json"
    if not file.is_file():
        return
    stated = json.loads(Tokenizer.from_file(str(file)).to_str())["pre_tokenizer"]
    loaded = json.loads(hf_tokenizer.backend_tokenizer.to_str())["pre_tokenizer"]
    if loaded != stated:
        raise TokenizerMismatch(
            f"{type(hf_tokenizer).__name__} loaded from {tokenizer_dir} has a different pre-tokenizer from the one in its "
            f"tokenizer.json (loaded {loaded!r}, stated {stated!r}), so its ids are not the model's. Load a copy whose "
            'tokenizer_config.json has "tokenizer_class": "PreTrainedTokenizerFast" instead.'
        )
