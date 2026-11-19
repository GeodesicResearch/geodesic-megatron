# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""The flags that decide how ``preprocess_data.py`` encodes a corpus, and the provenance notes that record them.

Every route that runs ``preprocess_data.py`` takes both from here: ``pipeline_data_submit.sbatch tokenize`` from this
file's command line, and ``tokenize_arrow.py`` by import. The two routes then encode the same rows into the same tokens
and write the same notes, which the corpus checks of ``configs/control_pretraining`` read. The standard library is all
the file imports, so the host and the container both run it.

Usage:

    python scripts/data/preprocess_flags.py flags    # the flags, as words of a command
    python scripts/data/preprocess_flags.py notes    # the notes, as ``--note KEY=VALUE`` words of count_idx_tokens.py

Catalogue:
    TOKENIZER_TYPE: the tokenizer class that ``--tokenizer-model`` is loaded as.
    ENCODING_FLAGS: the flags, as the words of a command.
    ENCODING_NOTES: the ``KEY=VALUE`` provenance notes that state the flags' effect.
    main: print the flags or the notes.
"""

from __future__ import annotations

import argparse


TOKENIZER_TYPE = "HuggingFaceTokenizer"
# ``--append-eod`` ends every row with the tokenizer's end-of-document token, which is the separator that the row
# counts of a corpus's token total include.
ENCODING_FLAGS = ("--tokenizer-type", TOKENIZER_TYPE, "--append-eod")
ENCODING_NOTES = ("append_eod=true",)


def main(argv: list[str] | None = None) -> int:
    """Print ``ENCODING_FLAGS`` (field ``flags``) or ``ENCODING_NOTES`` as ``--note`` words (field ``notes``), on one
    line separated by spaces; return 0."""
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("field", choices=("flags", "notes"))
    field = parser.parse_args(argv).field
    words = ENCODING_FLAGS if field == "flags" else tuple(word for note in ENCODING_NOTES for word in ("--note", note))
    print(" ".join(words))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
