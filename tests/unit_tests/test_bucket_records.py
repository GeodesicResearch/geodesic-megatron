# Copyright (c) 2026, Geodesic Research.
# Licensed under the Apache License, Version 2.0.
"""A local copy may go only once the bucket provably holds it: records, comparison, removal and restore.

``scripts/hub/bucket_records.py`` compares the records of a unit's files with the bucket's listing by path and size,
deletes a unit's recorded files after a clean match, and restores a unit from the bucket into an empty directory,
comparing every file's size and sha256 with the record. The files are real files in ``tmp_path`` and the functions are
the real ones. The one stand-in is the Hub, because the real one reads the Hub and the training container's
huggingface_hub has no bucket API: ``FakeBucket`` answers ``list_bucket_tree`` with the records it was given, and
``ServingBucket`` also serves the bytes of its files.
"""

from __future__ import annotations

import dataclasses
import logging
from pathlib import Path

import pytest
from scripts.file_digest import HASH_WORKERS, directory_records
from scripts.hub import bucket_records as records

from tests.unit_tests.bucket_listing import BucketItem
from tests.unit_tests.checkpoint_dirs import make_checkpoint_dir


class FakeBucket:
    """Answers the listing of one bucket from a dict of path -> size, plus a folder entry to be skipped."""

    def __init__(self, files: dict[str, int]):
        self.files = files

    def list_bucket_tree(self, bucket_id, prefix=None, *, recursive=None):
        listing = [
            BucketItem("file", path, size) for path, size in self.files.items() if path.startswith(prefix or "")
        ]
        return [BucketItem("directory", (prefix or "").rstrip("/"), 0), *listing]


class ServingBucket(FakeBucket):
    """A bucket that also serves the bytes of its files, as ``download_bucket_files`` of the Hub client does.

    A stand-in because the real client downloads from the Hub, and the training container's huggingface_hub has no
    bucket client. ``contents`` maps a bucket path to its bytes. Like the real client it skips a path that the bucket
    lacks when ``raise_on_missing_files`` is off, and ``requested`` lists every path that was asked for.
    """

    def __init__(self, contents: dict[str, bytes]):
        super().__init__({path: len(data) for path, data in contents.items()})
        self.contents = contents
        self.requested: list[str] = []

    def download_bucket_files(self, bucket_id, files, *, raise_on_missing_files=False):
        assert raise_on_missing_files is False, "the tool finds a missing file itself and names it"
        for path, target in files:
            self.requested.append(path)
            if path in self.contents:
                Path(target).write_bytes(self.contents[path])


def make_unit(root: Path) -> Path:
    """A save, ``<root>/run/iter_0000100``: a 16-byte shard, ``metadata.json`` and an ``hf/`` export that a sync
    leaves out."""
    run = make_checkpoint_dir(root / "run", [100], tracker=None, hf_files={"model.safetensors": b"h" * 10})
    return run / "iter_0000100"


def verified_for(
    unit: Path, remote: str = "checkpoints/run/iter_0000100", exclude: tuple[str, ...] = ("hf/*",)
) -> records.VerifiedRecord:
    return records.VerifiedRecord(
        "org/bucket", remote, unit, "2026-11-17T21:00:00Z", directory_records(unit, None, exclude, None, HASH_WORKERS)
    )


def bucket_for(verified: records.VerifiedRecord) -> FakeBucket:
    return FakeBucket({f"{verified.remote}/{r.path}": r.size for r in verified.records})


def serving_bucket_for(verified: records.VerifiedRecord) -> ServingBucket:
    """A bucket that holds the unit's recorded files with their real bytes."""
    return ServingBucket(
        {f"{verified.remote}/{r.path}": (verified.local / r.path).read_bytes() for r in verified.records}
    )


class TestMismatches:
    def test_a_listing_equal_to_the_records_has_no_mismatch(self, tmp_path):
        verified = verified_for(make_unit(tmp_path))
        remote = records.remote_sizes(bucket_for(verified), "org/bucket", verified.remote)
        assert records.mismatches(verified.records, remote) == []

    def test_a_missing_file_a_different_size_and_an_extra_file_are_each_named(self, tmp_path):
        verified = verified_for(make_unit(tmp_path))
        remote = {r.path: r.size for r in verified.records}
        del remote["metadata.json"]
        remote["__0_0.distcp"] += 1
        remote["stray.bin"] = 3
        assert records.mismatches(verified.records, remote) == [
            "__0_0.distcp: 17 bytes in the bucket, 16 local",
            "metadata.json: not in the bucket",
            "stray.bin: in the bucket and not recorded",
        ]

    def test_remote_sizes_reads_paths_relative_to_the_prefix_and_skips_folders(self, tmp_path):
        bucket = FakeBucket({"a/b/x.bin": 5, "a/b/sub/y.bin": 7, "a/other/z.bin": 9})
        assert records.remote_sizes(bucket, "org/bucket", "a/b") == {"x.bin": 5, "sub/y.bin": 7}


class TestVerifiedFile:
    def test_a_record_survives_a_write_and_a_read(self, tmp_path):
        verified = verified_for(make_unit(tmp_path))
        records.write_verified(tmp_path / "v" / "r.tsv", verified)
        assert records.read_verified(tmp_path / "v" / "r.tsv") == verified
        assert not (tmp_path / "v" / "r.tsv.tmp").exists()

    def test_a_malformed_line_is_refused(self, tmp_path):
        path = tmp_path / "r.tsv"
        path.write_text("# bucket: b\n# remote: r\n# local: /x\n# verified_at: t\nnot a record\n")
        with pytest.raises(ValueError, match="expected sha256"):
            records.read_verified(path)

    def test_a_missing_header_is_refused(self, tmp_path):
        path = tmp_path / "r.tsv"
        path.write_text("# bucket: b\n")
        with pytest.raises(ValueError, match="header keys missing"):
            records.read_verified(path)


class TestRemoval:
    def test_a_clean_match_removes_the_recorded_files_and_the_emptied_directory(self, tmp_path):
        unit = make_unit(tmp_path)
        verified = verified_for(unit)
        (unit / "hf" / "model.safetensors").unlink()
        (unit / "hf").rmdir()
        removed = records.remove_verified(bucket_for(verified), verified, [tmp_path], None, execute=True)
        assert sorted(p.name for p in removed) == ["__0_0.distcp", "metadata.json"]
        assert not unit.exists()
        assert (tmp_path / "run").is_dir(), "only the unit goes, not its parent"

    def test_a_recorded_file_in_nested_directories_goes_with_every_directory_it_empties(self, tmp_path):
        unit = make_unit(tmp_path)
        (unit / "hf" / "deeper").mkdir()
        (unit / "hf" / "deeper" / "tokenizer.json").write_text("{}")
        verified = verified_for(unit, exclude=())
        removed = records.remove_verified(bucket_for(verified), verified, [tmp_path], None, execute=True)
        assert len(removed) == 4
        assert not (unit / "hf").exists(), "the nested directories are removed, deepest first"
        assert not unit.exists()
        assert (tmp_path / "run").is_dir(), "only the unit goes, not its parent"

    def test_a_dry_run_reports_and_removes_nothing(self, tmp_path):
        unit = make_unit(tmp_path)
        verified = verified_for(unit)
        removed = records.remove_verified(bucket_for(verified), verified, [tmp_path], None, execute=False)
        assert len(removed) == 2
        assert (unit / "__0_0.distcp").is_file()

    def test_a_file_that_no_record_names_stays_and_so_does_its_directory(self, tmp_path, caplog):
        unit = make_unit(tmp_path)
        verified = verified_for(unit)
        caplog.set_level(logging.WARNING, logger=records.__name__)
        records.remove_verified(bucket_for(verified), verified, [tmp_path], None, execute=True)
        assert (unit / "hf" / "model.safetensors").is_file(), "the hf export was never recorded, so it is not removed"
        assert not (unit / "__0_0.distcp").exists()
        assert "no record names" in caplog.text

    def test_a_directory_outside_the_allowed_roots_is_refused(self, tmp_path):
        verified = verified_for(make_unit(tmp_path))
        with pytest.raises(records.RemovalRefused, match="outside the allowed roots"):
            records.remove_verified(bucket_for(verified), verified, [tmp_path / "elsewhere"], None, execute=True)
        with pytest.raises(records.RemovalRefused, match="outside the allowed roots"):
            records.remove_verified(bucket_for(verified), verified, [], None, execute=True)

    def test_a_path_that_climbs_out_of_a_root_by_a_symlink_is_refused(self, tmp_path):
        outside = make_unit(tmp_path / "outside")
        (tmp_path / "root").mkdir()
        (tmp_path / "root" / "link").symlink_to(outside.parent)
        linked = tmp_path / "root" / "link" / outside.name
        verified = verified_for(linked)
        with pytest.raises(records.RemovalRefused, match="outside the allowed roots"):
            records.remove_verified(bucket_for(verified), verified, [tmp_path / "root"], None, execute=True)

    def test_the_newest_save_of_a_run_is_never_removed(self, tmp_path):
        unit = make_unit(tmp_path)
        verified = verified_for(unit)
        with pytest.raises(records.RemovalRefused, match="newest completed save"):
            records.remove_verified(bucket_for(verified), verified, [tmp_path], unit, execute=True)
        assert (unit / "__0_0.distcp").is_file()

    def test_a_bucket_that_lost_a_file_blocks_the_removal(self, tmp_path):
        unit = make_unit(tmp_path)
        verified = verified_for(unit)
        bucket = bucket_for(verified)
        del bucket.files[f"{verified.remote}/metadata.json"]
        with pytest.raises(records.RemovalRefused, match="does not hold"):
            records.remove_verified(bucket, verified, [tmp_path], None, execute=True)
        assert (unit / "metadata.json").is_file()

    def test_a_file_changed_since_it_was_recorded_blocks_the_removal_and_nothing_goes(self, tmp_path):
        unit = make_unit(tmp_path)
        verified = verified_for(unit)
        (unit / "metadata.json").write_text("{ }")
        with pytest.raises(records.RemovalRefused, match="size differs"):
            records.remove_verified(bucket_for(verified), verified, [tmp_path], None, execute=True)
        assert (unit / "__0_0.distcp").is_file(), "the check runs on every file before the first deletion"


class TestRestore:
    def restore(self, tmp_path, bucket=None, verified=None, destination=None):
        verified = verified or verified_for(make_unit(tmp_path))
        bucket = bucket or serving_bucket_for(verified)
        return verified, bucket, records.restore_verified(bucket, verified, destination or tmp_path / "restored")

    def test_a_clean_restore_downloads_every_recorded_file_with_its_bytes_and_returns_the_paths(self, tmp_path):
        verified, bucket, restored = self.restore(tmp_path)
        assert [p.name for p in restored] == [r.path for r in verified.records]
        assert all(p.read_bytes() == (verified.local / p.name).read_bytes() for p in restored)
        assert not (tmp_path / "restored" / "hf").exists(), "the hf export was never recorded, so it is not asked for"
        assert sorted(bucket.requested) == sorted(f"{verified.remote}/{r.path}" for r in verified.records)

    def test_a_recorded_file_in_a_nested_directory_lands_in_its_folder(self, tmp_path):
        unit = make_unit(tmp_path)
        (unit / "hf" / "deeper").mkdir()
        (unit / "hf" / "deeper" / "tokenizer.json").write_text("{}")
        verified = verified_for(unit, exclude=())
        _, _, restored = self.restore(tmp_path, verified=verified)
        assert tmp_path / "restored" / "hf" / "deeper" / "tokenizer.json" in restored
        assert (tmp_path / "restored" / "hf" / "deeper" / "tokenizer.json").read_text() == "{}"

    def test_an_existing_empty_directory_is_a_fine_destination(self, tmp_path):
        (tmp_path / "restored").mkdir()
        _, _, restored = self.restore(tmp_path)
        assert len(restored) == 2

    def test_a_file_that_the_bucket_lacks_is_named_and_the_others_stay_for_inspection(self, tmp_path):
        verified = verified_for(make_unit(tmp_path))
        bucket = serving_bucket_for(verified)
        del bucket.contents[f"{verified.remote}/metadata.json"]
        with pytest.raises(records.RestoreFailed, match="metadata.json: not in the bucket"):
            records.restore_verified(bucket, verified, tmp_path / "restored")
        assert (tmp_path / "restored" / "__0_0.distcp").is_file()

    def test_a_file_of_another_size_is_named_with_both_sizes(self, tmp_path):
        verified = verified_for(make_unit(tmp_path))
        bucket = serving_bucket_for(verified)
        bucket.contents[f"{verified.remote}/__0_0.distcp"] += b"+"
        with pytest.raises(records.RestoreFailed, match="__0_0.distcp: 17 bytes in the bucket, 16 local"):
            records.restore_verified(bucket, verified, tmp_path / "restored")

    def test_a_file_of_the_same_size_with_other_bytes_fails_on_its_sha256(self, tmp_path):
        verified = verified_for(make_unit(tmp_path))
        bucket = serving_bucket_for(verified)
        key = f"{verified.remote}/__0_0.distcp"
        bucket.contents[key] = b"z" * len(bucket.contents[key])
        with pytest.raises(records.RestoreFailed, match="__0_0.distcp: sha256 .* downloaded, .* recorded") as caught:
            records.restore_verified(bucket, verified, tmp_path / "restored")
        assert "metadata.json" not in str(caught.value), "only the file that differs is named"

    @pytest.mark.parametrize("kind", ["file", "directory with a file"])
    def test_a_destination_that_is_not_empty_is_refused_before_any_download(self, tmp_path, kind):
        verified = verified_for(make_unit(tmp_path))
        bucket = serving_bucket_for(verified)
        destination = tmp_path / "restored"
        if kind == "file":
            destination.write_text("x")
        else:
            destination.mkdir()
            (destination / "stray").write_text("x")
        with pytest.raises(records.RestoreFailed, match="not an empty directory"):
            records.restore_verified(bucket, verified, destination)
        assert bucket.requested == []

    @pytest.mark.parametrize("where", ["the unit itself", "inside the unit"])
    def test_a_destination_in_the_units_own_directory_is_refused_and_its_files_are_untouched(self, tmp_path, where):
        unit = make_unit(tmp_path)
        verified = verified_for(unit)
        bucket = serving_bucket_for(verified)
        destination = unit / "back" if where == "inside the unit" else unit
        before = (unit / "__0_0.distcp").read_bytes()
        with pytest.raises(
            records.RestoreFailed, match="the unit's own directory or inside it|not an empty directory"
        ):
            records.restore_verified(bucket, verified, destination)
        assert bucket.requested == [] and (unit / "__0_0.distcp").read_bytes() == before

    @pytest.mark.parametrize("path", ["../escape.bin", "/etc/escape.bin", "a/../../escape.bin", ""])
    def test_a_recorded_path_that_would_leave_the_destination_is_refused(self, tmp_path, path):
        verified = verified_for(make_unit(tmp_path))
        bad = records.VerifiedRecord(
            verified.bucket,
            verified.remote,
            verified.local,
            verified.verified_at,
            (*verified.records[:1], dataclasses.replace(verified.records[1], path=path)),
        )
        bucket = ServingBucket({})
        with pytest.raises(records.RestoreFailed, match="would leave the destination"):
            records.restore_verified(bucket, bad, tmp_path / "restored")
        assert bucket.requested == [] and not (tmp_path / "escape.bin").exists()

    def test_a_record_without_a_file_is_refused(self, tmp_path):
        verified = verified_for(make_unit(tmp_path))
        empty = records.VerifiedRecord(verified.bucket, verified.remote, verified.local, verified.verified_at, ())
        with pytest.raises(records.RestoreFailed, match="nothing to restore"):
            records.restore_verified(ServingBucket({}), empty, tmp_path / "restored")
