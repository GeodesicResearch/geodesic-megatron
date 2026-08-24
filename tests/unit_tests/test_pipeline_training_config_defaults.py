"""The training config loader's single-parent ``defaults:`` chain.

A config YAML may name one parent via a top-level ``defaults: <path>`` key;
parents load first and children override, so a leaf carries only its deltas.
These tests pin the merge order, relative-path resolution against the config
file's own directory, and the loud failures (missing parent, cycle, non-string
ref) that keep a partial config from silently training.
"""

import pytest


@pytest.fixture()
def load(run_module, tmp_path):
    fn = run_module.load_yaml_with_defaults

    def _write(name: str, text: str) -> str:
        p = tmp_path / name
        p.write_text(text)
        return str(p)

    return fn, _write


def test_child_overrides_parent_and_keeps_parent_keys(load):
    fn, write = load
    write("base.yaml", "a: 1\nnested:\n  x: 1\n  y: 2\n")
    leaf = write("leaf.yaml", "defaults: base.yaml\nnested:\n  y: 3\nb: 4\n")
    c = fn(leaf)
    assert c.a == 1 and c.b == 4
    assert c.nested.x == 1 and c.nested.y == 3
    assert "defaults" not in c


def test_relative_paths_resolve_against_the_config_files_directory(load, tmp_path):
    fn, write = load
    (tmp_path / "sub").mkdir()
    write("base.yaml", "a: 1\n")
    (tmp_path / "sub" / "leaf.yaml").write_text("defaults: ../base.yaml\nb: 2\n")
    c = fn(str(tmp_path / "sub" / "leaf.yaml"))
    assert c.a == 1 and c.b == 2


def test_grandparent_chains_merge_oldest_first(load):
    fn, write = load
    write("grand.yaml", "a: 1\nb: 1\nc: 1\n")
    write("mid.yaml", "defaults: grand.yaml\nb: 2\nc: 2\n")
    leaf = write("leaf.yaml", "defaults: mid.yaml\nc: 3\n")
    c = fn(leaf)
    assert (c.a, c.b, c.c) == (1, 2, 3)


def test_a_missing_parent_raises(load):
    fn, write = load
    leaf = write("leaf.yaml", "defaults: nope.yaml\n")
    with pytest.raises(FileNotFoundError, match="nope.yaml"):
        fn(leaf)


def test_a_cycle_raises(load, tmp_path):
    fn, write = load
    write("a.yaml", "defaults: b.yaml\n")
    write("b.yaml", "defaults: a.yaml\n")
    with pytest.raises(ValueError, match="cycle"):
        fn(str(tmp_path / "a.yaml"))


def test_a_non_string_defaults_ref_raises(load):
    fn, write = load
    leaf = write("leaf.yaml", "defaults: [a.yaml, b.yaml]\n")
    with pytest.raises(ValueError, match="single parent"):
        fn(leaf)


def test_a_parentless_config_loads_unchanged(load):
    fn, write = load
    leaf = write("leaf.yaml", "a: 1\n")
    assert fn(leaf).a == 1
