# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""The token-id checks the tools apply to ids read from hand-written files (``scripts/token_ids.py``)."""

import re

import pytest
from scripts.token_ids import require_token_id, require_token_id_list


class SpecError(Exception):
    """A tool's own error type, which the checks raise in place of ValueError when given it."""


@pytest.mark.parametrize("value", [0, 131072])
def test_a_non_negative_integer_is_a_token_id(value):
    assert require_token_id(value, "id") == value


@pytest.mark.parametrize("value", [-1, True, False, 1.0, "5", None])
def test_anything_else_is_refused_naming_where_it_sits(value):
    with pytest.raises(ValueError, match=re.escape(f"spec.id must be a token id (an integer >= 0), not {value!r}")):
        require_token_id(value, "spec.id")


def test_a_list_of_distinct_ids_is_returned_as_a_tuple_in_order():
    assert require_token_id_list([131073, 2, 131072], "ids", allow_empty=False) == (131073, 2, 131072)


@pytest.mark.parametrize(("allow_empty", "accepted"), [(True, True), (False, False)])
def test_an_empty_list_is_a_token_id_list_only_where_the_caller_allows_one(allow_empty, accepted):
    if accepted:
        assert require_token_id_list([], "ids", allow_empty=allow_empty) == ()
    else:
        with pytest.raises(ValueError, match=re.escape("ids must be a non-empty list of token ids, not []")):
            require_token_id_list([], "ids", allow_empty=allow_empty)


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (500, "ids must be a list of token ids, not 500"),
        ((500,), "ids must be a list of token ids, not (500,)"),
        ({500: True}, "ids must be a list of token ids, not {500: True}"),
        ([5, 6, 5], "ids repeats a token id: [5, 6, 5]"),
        ([5, -1], "ids[1] must be a token id (an integer >= 0), not -1"),
        ([True], "ids[0] must be a token id (an integer >= 0), not True"),
        (["500"], "ids[0] must be a token id (an integer >= 0), not '500'"),
    ],
)
def test_a_malformed_list_is_refused_naming_the_entry(value, message):
    with pytest.raises(ValueError, match=re.escape(message)):
        require_token_id_list(value, "ids", allow_empty=True)


def test_each_check_raises_the_callers_error_type():
    with pytest.raises(SpecError, match=r"x\[0\] must be a token id"):
        require_token_id_list([-3], "x", allow_empty=True, error=SpecError)
    with pytest.raises(SpecError, match="x repeats a token id"):
        require_token_id_list([1, 1], "x", allow_empty=True, error=SpecError)
