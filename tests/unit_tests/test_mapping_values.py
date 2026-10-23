# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""The value checks the tools apply to scalars read from hand-written files (``scripts/mapping_values.py``)."""

import math
import re
from pathlib import Path

import pytest
from scripts.mapping_values import (
    require_absolute_path,
    require_file_beside,
    require_int,
    require_number,
    require_string,
)


class SpecError(Exception):
    """A tool's own error type, which the checks raise in place of ValueError when given it."""


@pytest.mark.parametrize(("value", "minimum"), [(0, 0), (7, 1), (-2, -5)])
def test_an_integer_at_or_above_its_minimum_is_returned(value, minimum):
    assert require_int(value, "n", minimum) == value


@pytest.mark.parametrize("value", [0, -1, True, 1.0, "3", None])
def test_anything_else_is_not_an_integer_at_or_above_one(value):
    with pytest.raises(ValueError, match=re.escape(f"spec.n must be an integer >= 1, not {value!r}")):
        require_int(value, "spec.n", 1)


@pytest.mark.parametrize(("value", "inclusive"), [(0.5, True), (0.5, False), (2, True), (0.0, True)])
def test_a_finite_number_in_range_is_returned_as_a_float(value, inclusive):
    result = require_number(value, "x", 0.0, inclusive=inclusive)
    assert result == value and isinstance(result, float)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf, True, "1.0", None])
def test_a_value_that_is_not_a_finite_number_is_refused(value):
    with pytest.raises(ValueError, match=re.escape(f"x must be a finite number, not {value!r}")):
        require_number(value, "x", 0.0, inclusive=True)


@pytest.mark.parametrize(
    ("value", "inclusive", "message"),
    [
        (0.0, False, "x must be > 0.0, not 0.0"),
        (-1, False, "x must be > 0.0, not -1"),
        (0.5, True, "x must be >= 1.0, not 0.5"),
    ],
)
def test_a_number_out_of_range_is_refused_naming_the_bound(value, inclusive, message):
    minimum = 1.0 if inclusive else 0.0
    with pytest.raises(ValueError, match=re.escape(message)):
        require_number(value, "x", minimum, inclusive=inclusive)


def test_a_non_empty_string_is_returned():
    assert require_string("name", "s") == "name"


@pytest.mark.parametrize("value", ["", None, 3, ["a"]])
def test_anything_else_is_not_a_non_empty_string(value):
    with pytest.raises(ValueError, match=re.escape(f"spec.s must be a non-empty string, not {value!r}")):
        require_string(value, "spec.s")


def test_each_value_check_raises_the_callers_error_type():
    with pytest.raises(SpecError, match="n must be an integer"):
        require_int(-1, "n", 0, error=SpecError)
    with pytest.raises(SpecError, match="x must be > 0"):
        require_number(0, "x", 0.0, inclusive=False, error=SpecError)
    with pytest.raises(SpecError, match="s must be a non-empty string"):
        require_string("", "s", error=SpecError)
    with pytest.raises(SpecError, match="p must be an absolute path"):
        require_absolute_path("relative", "p", error=SpecError)
    with pytest.raises(SpecError, match="f .* is not a file"):
        require_file_beside("absent.yaml", "f", Path("/"), error=SpecError)


def test_an_absolute_path_is_returned_as_given():
    assert require_absolute_path("/projects/a5k/data", "p") == "/projects/a5k/data"


@pytest.mark.parametrize("value", ["relative/path", "", None, 3])
def test_anything_else_is_not_an_absolute_path(value):
    with pytest.raises(ValueError, match=r"spec\.p must be (an absolute path|a non-empty string)"):
        require_absolute_path(value, "spec.p")


def test_a_file_is_named_relative_to_the_directory_beside_it_or_absolutely(tmp_path):
    (tmp_path / "sub").mkdir()
    target = tmp_path / "sub" / "config.yaml"
    target.write_text("a: 1\n")
    assert require_file_beside("sub/config.yaml", "f", tmp_path) == str(target.resolve())
    assert require_file_beside("../sub/config.yaml", "f", tmp_path / "sub") == str(target.resolve())
    assert require_file_beside(str(target), "f", tmp_path / "elsewhere") == str(target.resolve())


@pytest.mark.parametrize("value", ["absent.yaml", "sub", "", None])
def test_a_name_that_is_not_a_file_beside_the_directory_is_refused(tmp_path, value):
    (tmp_path / "sub").mkdir()
    with pytest.raises(ValueError, match=r"spec\.f (.* is not a file|must be a non-empty string)"):
        require_file_beside(value, "spec.f", tmp_path)
