# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""scripts/data/preprocess_flags.py: the encoding flags and notes that the tokenise routes share, as the sbatch reads them."""

from __future__ import annotations

import pytest
from scripts.data import preprocess_flags
from scripts.data.preprocess_flags import ENCODING_FLAGS, ENCODING_NOTES


def test_the_flags_print_as_the_words_of_a_command(capsys):
    assert preprocess_flags.main(["flags"]) == 0
    assert capsys.readouterr().out.split() == list(ENCODING_FLAGS)


def test_each_note_prints_as_a_note_argument_of_the_counter(capsys):
    assert preprocess_flags.main(["notes"]) == 0
    words = capsys.readouterr().out.split()
    assert words == [word for note in ENCODING_NOTES for word in ("--note", note)]
    assert all("=" in note for note in ENCODING_NOTES), "every note is KEY=VALUE, as the counter parses it"


def test_a_field_other_than_flags_or_notes_is_refused(capsys):
    with pytest.raises(SystemExit):
        preprocess_flags.main(["encoding"])
    assert "invalid choice" in capsys.readouterr().err
