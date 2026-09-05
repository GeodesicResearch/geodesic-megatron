"""Mocks that present a chosen class name without renaming a real class."""

from __future__ import annotations

from unittest.mock import Mock


def mock_of_class(name: str, spec: type | None = None) -> Mock:
    """A Mock whose ``__class__`` is a throwaway class called ``name``, a subclass of ``spec`` if given.

    ``Mock(spec=cls).__class__`` is ``cls`` itself, so ``mock.__class__.__name__ = name`` renames the
    real class for the rest of the process. (Without a spec, ``Mock().__class__`` is a subclass made
    for that one instance, so renaming it is harmless; the helper covers that case too so every named
    mock is built one way.) Assigning ``__class__`` sets only the mock's spec class:
    ``isinstance(mock, spec)`` still holds, attribute access is still checked against ``spec``, and
    the ``__repr__`` of the wrappers under test reads the given name.
    """
    mock = Mock() if spec is None else Mock(spec=spec)
    mock.__class__ = type(name, () if spec is None else (spec,), {})
    return mock
