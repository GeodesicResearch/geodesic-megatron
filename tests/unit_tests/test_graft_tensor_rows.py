# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""Unit tests for scripts/checkpoint/graft_tensor_rows.py.

The library's own ``save_file`` writes each release as real safetensors files. ``cache_repo`` puts them into a Hub
cache layout under a temporary ``HF_HOME``. The tool reads them as it reads a release in the shared Hub cache. The
tests run on a CPU.
"""

import json
import logging
from pathlib import Path

import pytest
import torch
import yaml
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from scripts.checkpoint import graft_tensor_rows as graft
from scripts.checkpoint.safetensors_index import INDEX_FILE, checked_weight_map

from tests.unit_tests.hf_export_fixtures import cache_repo


REPO_ROOT = Path(__file__).resolve().parents[2]
SUPER_CONFIG = REPO_ROOT / "configs" / "checkpoint_grafts" / "nemotron_super_base_chat_init.yaml"
SUPER_SFT_QUICKSTART = REPO_ROOT / "configs" / "quickstart" / "nemotron_super_quickstart_sft.yaml"

RECIPIENT = graft.Release(repo="acme/model-base", revision="a" * 40)
DONOR = graft.Release(repo="acme/model-instruct", revision="b" * 40)
EMBEDDING = "backbone.embeddings.weight"
HEAD = "lm_head.weight"
OTHER = "backbone.layers.0.mixer.weight"
FIRST_SHARD = "model-00001-of-00002.safetensors"
SECOND_SHARD = "model-00002-of-00002.safetensors"
# The rows of each tensor that are zero in the recipient. Row 6 of the embedding is zero in the donor too.
ZERO_EMBEDDING_ROWS = [1, 3, 6]
ZERO_HEAD_ROWS = [0, 3]
DONOR_ZERO_EMBEDDING_ROWS = [6]
THRESHOLD = 1.0e-3


def release_tensors(seed: int, zero_embedding_rows: list[int], zero_head_rows: list[int]) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    tensors = {
        EMBEDDING: torch.randn(8, 4, generator=generator).to(torch.bfloat16),
        OTHER: torch.randn(3, 5, generator=generator),
        HEAD: torch.randn(8, 4, generator=generator).to(torch.bfloat16),
    }
    tensors[EMBEDDING][zero_embedding_rows] = 0
    tensors[HEAD][zero_head_rows] = 0
    return tensors


@pytest.fixture(autouse=True)
def hf_home(tmp_path, monkeypatch) -> Path:
    """An empty ``HF_HOME`` for the test. The tool reads the Hub cache under it."""
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf_home"))
    return tmp_path / "hf_home"


def write_release(
    tmp_path: Path, release: graft.Release, tensors: dict[str, torch.Tensor], shards: dict[str, list[str]]
) -> None:
    """Cache a release with the given tensors as its snapshot under the test's ``HF_HOME``."""
    source = tmp_path / "sources" / release.repo.replace("/", "--")
    source.mkdir(parents=True)
    for shard, names in shards.items():
        save_file({name: tensors[name] for name in names}, str(source / shard), metadata={"format": "pt"})
    weight_map = {name: shard for shard, names in shards.items() for name in names}
    (source / INDEX_FILE).write_text(json.dumps({"metadata": {"total_size": 0}, "weight_map": weight_map}))
    (source / "config.json").write_text(json.dumps({"release": release.repo}))
    cache_repo(tmp_path / "hf_home", release.repo, [(release.revision, source)], refs_main=release.revision)


SHARDS = {FIRST_SHARD: [EMBEDDING, OTHER], SECOND_SHARD: [HEAD]}


def write_releases(
    tmp_path: Path,
    donor_tensors: dict[str, torch.Tensor] | None = None,
    donor_shards: dict[str, list[str]] | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Cache the recipient and a donor and return both releases' tensors."""
    recipient_tensors = release_tensors(1, ZERO_EMBEDDING_ROWS, ZERO_HEAD_ROWS)
    donor_tensors = donor_tensors or release_tensors(2, DONOR_ZERO_EMBEDDING_ROWS, [])
    write_release(tmp_path, RECIPIENT, recipient_tensors, SHARDS)
    write_release(tmp_path, DONOR, donor_tensors, donor_shards or SHARDS)
    return recipient_tensors, donor_tensors


def config_mapping(**changes) -> dict:
    mapping = {
        "recipient": {"repo": RECIPIENT.repo, "revision": RECIPIENT.revision},
        "donor": {"repo": DONOR.repo, "revision": DONOR.revision},
        "tensors": [EMBEDDING, HEAD],
        "row_norm_threshold": THRESHOLD,
        "donor_near_zero_rows": graft.LEAVE,
        "output_name": "model-base-grafted-hf",
    }
    mapping.update(changes)
    return mapping


def write_config(tmp_path: Path, mapping: dict) -> Path:
    path = tmp_path / "graft.yaml"
    path.write_text(yaml.safe_dump(mapping))
    return path


def run_graft(tmp_path: Path, **changes) -> tuple[Path, dict]:
    record = graft.run(write_config(tmp_path, config_mapping(**changes)), tmp_path / "models")
    return tmp_path / "models" / "model-base-grafted-hf", record


def test_near_zero_rows_take_the_donors_rows_and_every_other_row_stays(tmp_path):
    recipient, donor = write_releases(tmp_path)
    output, _ = run_graft(tmp_path)
    embedding = load_file(str(output / FIRST_SHARD))[EMBEDDING]
    head = load_file(str(output / SECOND_SHARD))[HEAD]
    assert torch.equal(embedding[[1, 3]], donor[EMBEDDING][[1, 3]])
    assert torch.equal(embedding[[0, 2, 4, 5, 7]], recipient[EMBEDDING][[0, 2, 4, 5, 7]])
    assert torch.equal(head[ZERO_HEAD_ROWS], donor[HEAD][ZERO_HEAD_ROWS])
    assert torch.equal(head[[1, 2, 4, 5, 6, 7]], recipient[HEAD][[1, 2, 4, 5, 6, 7]])
    assert embedding.dtype == torch.bfloat16 and head.dtype == torch.bfloat16


def test_a_row_near_zero_in_both_releases_is_left_under_leave(tmp_path):
    write_releases(tmp_path)
    output, record = run_graft(tmp_path, donor_near_zero_rows=graft.LEAVE)
    embedding = load_file(str(output / FIRST_SHARD))[EMBEDDING]
    assert not embedding[6].any()
    assert record["tensors"][EMBEDDING]["left_near_zero_rows"] == [6]


def test_a_row_near_zero_in_both_releases_stops_the_graft_under_fail_and_is_listed(tmp_path):
    write_releases(tmp_path)
    with pytest.raises(graft.GraftError, match=r"backbone\.embeddings\.weight: 1 rows are near zero.*\[6\]"):
        run_graft(tmp_path, donor_near_zero_rows=graft.FAIL)
    assert not (tmp_path / "models").exists()


def test_the_other_tensors_of_a_grafted_shard_and_its_metadata_are_kept(tmp_path):
    recipient, _ = write_releases(tmp_path)
    output, _ = run_graft(tmp_path)
    assert torch.equal(load_file(str(output / FIRST_SHARD))[OTHER], recipient[OTHER])
    with safe_open(str(output / FIRST_SHARD), framework="pt", device="cpu") as handle:
        assert handle.metadata() == {"format": "pt"}
    assert checked_weight_map(output) == {EMBEDDING: FIRST_SHARD, OTHER: FIRST_SHARD, HEAD: SECOND_SHARD}


def test_a_shard_without_a_grafted_tensor_is_a_link_to_the_file_of_the_recipients_cache_entry(tmp_path):
    recipient, _ = write_releases(tmp_path)
    output, record = run_graft(tmp_path, tensors=[EMBEDDING])
    snapshot = graft.cached_snapshot(RECIPIENT)
    for name in (SECOND_SHARD, INDEX_FILE, "config.json"):
        assert (output / name).is_symlink() and (output / name).readlink() == (snapshot / name).resolve()
    assert not (output / FIRST_SHARD).is_symlink()
    assert torch.equal(load_file(str(output / SECOND_SHARD))[HEAD], recipient[HEAD])
    assert record["written_shards"] == [FIRST_SHARD]
    assert record["linked_entries"] == sorted(["config.json", INDEX_FILE, SECOND_SHARD])


def test_the_provenance_file_records_the_config_the_revisions_and_the_rows(tmp_path):
    write_releases(tmp_path)
    output, record = run_graft(tmp_path)
    assert json.loads((output / graft.PROVENANCE_FILE).read_text()) == record
    config = graft.load_config(tmp_path / "graft.yaml")
    assert record["config"] == {
        "path": str(tmp_path / "graft.yaml"),
        "sha256": config.sha256,
        **config_mapping(),
    }
    assert record["recipient_snapshot"] == str(graft.cached_snapshot(RECIPIENT))
    assert record["donor_snapshot"] == str(graft.cached_snapshot(DONOR))
    assert record["output_dir"] == str(output)
    assert record["tensors"][EMBEDDING] == {
        "shard": FIRST_SHARD,
        "shape": [8, 4],
        "dtype": "torch.bfloat16",
        "grafted_row_count": 2,
        "left_near_zero_row_count": 1,
        "grafted_rows": [1, 3],
        "left_near_zero_rows": [6],
    }
    assert record["tensors"][HEAD]["grafted_rows"] == ZERO_HEAD_ROWS
    assert record["tensors"][HEAD]["left_near_zero_rows"] == []


def test_a_tensor_with_another_shape_in_the_donor_stops_the_graft(tmp_path):
    donor = release_tensors(2, [], [])
    donor[HEAD] = donor[HEAD][:, :3].contiguous()
    write_releases(tmp_path, donor_tensors=donor)
    with pytest.raises(graft.GraftError, match=r"lm_head\.weight: the recipient has shape \(8, 4\) and the donor"):
        run_graft(tmp_path)
    assert not (tmp_path / "models").exists()


def test_a_tensor_with_another_dtype_in_the_donor_stops_the_graft(tmp_path):
    donor = release_tensors(2, [], [])
    donor[EMBEDDING] = donor[EMBEDDING].to(torch.float32)
    write_releases(tmp_path, donor_tensors=donor)
    with pytest.raises(graft.GraftError, match="the recipient has dtype torch.bfloat16 and the donor torch.float32"):
        run_graft(tmp_path)


def test_a_tensor_in_different_shards_of_the_two_releases_stops_the_graft(tmp_path):
    write_releases(tmp_path, donor_shards={FIRST_SHARD: [EMBEDDING, OTHER, HEAD], SECOND_SHARD: []})
    with pytest.raises(graft.GraftError, match=f"{HEAD}: the recipient index puts it in {SECOND_SHARD}"):
        run_graft(tmp_path)
    assert not (tmp_path / "models").exists()


def test_a_tensor_without_a_row_to_graft_stops_the_graft(tmp_path):
    write_releases(tmp_path)
    with pytest.raises(graft.GraftError, match=r"backbone\.layers\.0\.mixer\.weight: no row to graft"):
        run_graft(tmp_path, tensors=[OTHER])
    assert not (tmp_path / "models").exists()


def test_a_tensor_that_an_index_does_not_map_stops_the_graft(tmp_path):
    write_releases(tmp_path)
    with pytest.raises(graft.GraftError, match="missing.weight: the recipient index maps no tensor of that name"):
        run_graft(tmp_path, tensors=["missing.weight"])


def test_an_existing_output_directory_is_never_written_into(tmp_path):
    write_releases(tmp_path)
    existing = tmp_path / "models" / "model-base-grafted-hf"
    existing.mkdir(parents=True)
    with pytest.raises(graft.GraftError, match="exists. A graft writes a new directory."):
        run_graft(tmp_path)
    assert list(existing.iterdir()) == []


def test_a_release_that_the_cache_lacks_stops_the_graft_with_the_download_command(tmp_path):
    write_releases(tmp_path)
    absent = {"repo": DONOR.repo, "revision": "c" * 40}
    with pytest.raises(graft.GraftError, match=f"hf download {DONOR.repo} --revision {'c' * 40}"):
        run_graft(tmp_path, donor=absent)


def test_a_donor_snapshot_without_the_shard_stops_the_graft_with_the_download_command(tmp_path):
    write_releases(tmp_path)
    (graft.cached_snapshot(DONOR) / SECOND_SHARD).unlink()
    with pytest.raises(graft.GraftError, match=f"--revision {DONOR.revision} --include {SECOND_SHARD}"):
        run_graft(tmp_path)
    assert not (tmp_path / "models").exists()


def test_a_donor_snapshot_needs_only_its_index_and_the_shards_of_the_grafted_tensors(tmp_path):
    _, donor = write_releases(tmp_path)
    (graft.cached_snapshot(DONOR) / SECOND_SHARD).unlink()
    output, _ = run_graft(tmp_path, tensors=[EMBEDDING])
    assert torch.equal(load_file(str(output / FIRST_SHARD))[EMBEDDING][[1, 3]], donor[EMBEDDING][[1, 3]])


def test_a_recipient_snapshot_without_one_of_its_shards_stops_the_graft(tmp_path):
    write_releases(tmp_path)
    (graft.cached_snapshot(RECIPIENT) / SECOND_SHARD).unlink()
    with pytest.raises(graft.GraftError, match="the recipient snapshot is not complete"):
        run_graft(tmp_path, tensors=[EMBEDDING])


def test_a_recipient_that_holds_a_provenance_file_stops_the_graft(tmp_path):
    write_releases(tmp_path)
    (graft.cached_snapshot(RECIPIENT) / graft.PROVENANCE_FILE).write_text("{}")
    with pytest.raises(graft.GraftError, match=graft.PROVENANCE_FILE):
        run_graft(tmp_path)


def test_graft_rows_returns_a_new_tensor_and_refuses_a_tensor_that_is_not_2d():
    recipient = torch.tensor([[0.0, 0.0], [1.0, 2.0]])
    donor = torch.tensor([[3.0, 4.0], [5.0, 6.0]])
    result = graft.graft_rows("x", recipient, donor, THRESHOLD, graft.FAIL)
    assert result.tensor.tolist() == [[3.0, 4.0], [1.0, 2.0]] and result.grafted == [0] and result.left == []
    assert recipient.tolist() == [[0.0, 0.0], [1.0, 2.0]]
    with pytest.raises(graft.GraftError, match="x: expected 2-D tensors"):
        graft.graft_rows("x", torch.zeros(4), torch.ones(4), THRESHOLD, graft.FAIL)


def test_the_threshold_decides_which_rows_are_near_zero():
    recipient = torch.tensor([[0.01, 0.0], [0.2, 0.0], [1.0, 0.0]])
    donor = torch.ones(3, 2)
    assert graft.graft_rows("x", recipient, donor, 0.1, graft.FAIL).grafted == [0]
    assert graft.graft_rows("x", recipient, donor, 0.5, graft.FAIL).grafted == [0, 1]


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"extra": 1}, "unknown keys \\['extra'\\]"),
        ({"recipient": {"repo": RECIPIENT.repo, "revision": "main"}}, "recipient.revision: expected a full 40-char"),
        ({"donor": {"repo": "model-instruct", "revision": DONOR.revision}}, "donor.repo: expected a Hub repository"),
        ({"donor": {"repo": DONOR.repo}}, "missing keys \\['revision'\\]"),
        ({"tensors": []}, "tensors: expected a non-empty list"),
        ({"tensors": [EMBEDDING, EMBEDDING]}, "tensors: expected a non-empty list of distinct"),
        ({"row_norm_threshold": "1e-3"}, "row_norm_threshold: expected a positive number"),
        ({"row_norm_threshold": 0}, "row_norm_threshold: expected a positive number"),
        ({"row_norm_threshold": True}, "row_norm_threshold: expected a positive number"),
        ({"donor_near_zero_rows": "ignore"}, "donor_near_zero_rows: expected 'fail' or 'leave'"),
        ({"output_name": "models/grafted"}, "output_name: expected a directory name without a path"),
    ],
)
def test_a_config_value_the_tool_cannot_use_is_refused(tmp_path, changes, message):
    with pytest.raises(graft.GraftError, match=message):
        graft.load_config(write_config(tmp_path, config_mapping(**changes)))


@pytest.mark.parametrize("key", sorted(graft.CONFIG_KEYS))
def test_every_config_key_is_required(tmp_path, key):
    """No key has a default, ``donor_near_zero_rows`` included: the config states each choice."""
    mapping = config_mapping()
    del mapping[key]
    with pytest.raises(graft.GraftError, match=f"missing keys \\['{key}'\\]"):
        graft.load_config(write_config(tmp_path, mapping))


def test_a_config_that_is_not_yaml_is_refused(tmp_path):
    path = tmp_path / "graft.yaml"
    path.write_text("recipient: [unclosed")
    with pytest.raises(graft.GraftError, match="cannot read the config"):
        graft.load_config(path)
    with pytest.raises(graft.GraftError, match="cannot read the config"):
        graft.load_config(tmp_path / "absent.yaml")


def test_the_super_config_builds_the_warm_start_that_the_super_sft_quickstart_reads():
    """The import gives its Megatron checkpoint the name of the directory it imports."""
    config = graft.load_config(SUPER_CONFIG)
    quickstart = yaml.safe_load(SUPER_SFT_QUICKSTART.read_text())
    assert config.output_name == Path(quickstart["checkpoint"]["pretrained_checkpoint"]).name
    assert config.recipient.repo != config.donor.repo


def test_the_command_line_writes_the_graft_under_hf_home(tmp_path, hf_home):
    _, donor = write_releases(tmp_path)
    assert graft.main(["--config", str(write_config(tmp_path, config_mapping()))]) == 0
    output = hf_home / graft.GRAFTS_DIR / "model-base-grafted-hf"
    assert torch.equal(load_file(str(output / FIRST_SHARD))[EMBEDDING][[1, 3]], donor[EMBEDDING][[1, 3]])
    assert (output / graft.PROVENANCE_FILE).is_file()


def test_the_command_line_returns_1_and_logs_the_graft_error_when_the_cache_lacks_a_release(tmp_path, hf_home, caplog):
    with caplog.at_level(logging.ERROR, logger=graft.logger.name):
        status = graft.main(["--config", str(write_config(tmp_path, config_mapping()))])
    assert status == 1
    assert f"FATAL: {hf_home / 'hub'} has no {INDEX_FILE} of {RECIPIENT.repo}" in caplog.text
    assert not (hf_home / graft.GRAFTS_DIR).exists()
