#!/usr/bin/env python
import os
import time

import duckdb
import pytest

from Utils.csv2parquet_sync import ClassifiedCsv
from Utils.csv2parquet_sync import classify_csv_files
from Utils.csv2parquet_sync import manifest_path
from Utils.csv2parquet_sync import Manifest
from Utils.csv2parquet_sync import normalize_path
from Utils.csv2parquet_sync import run_sync


CSV_HEADER = "station_id,channel,phase_type,phase_time,phase_score,phase_evaluation,phase_method,event_id,agency\n"


def make_csv(path, rows):
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(CSV_HEADER)
        for row in rows:
            fh.write(",".join(row) + "\n")


def default_row(phase_time, station="ST1", event_id="ev1"):
    return [
        station,
        "HHZ",
        "P",
        phase_time,
        "0.9",
        "manual",
        "phasenet",
        event_id,
        "FR",
    ]


def count_partition_files(output, year, month):
    partition_dir = os.path.join(output, f"year={year}", f"month={month}")
    if not os.path.isdir(partition_dir):
        return 0
    return len([f for f in os.listdir(partition_dir) if f.endswith(".parquet")])


def read_partition_rows(output, year, month):
    partition_dir = os.path.join(output, f"year={year}", f"month={month}")
    conn = duckdb.connect()
    try:
        return conn.execute(
            f"SELECT * FROM read_parquet('{partition_dir}/*.parquet')"
        ).fetchdf()
    finally:
        conn.close()


def test_first_import_creates_single_file_per_partition(tmp_path):
    csv_dir = tmp_path / "csv"
    csv_dir.mkdir()
    make_csv(csv_dir / "a.csv", [default_row("2025-03-01T10:00:00")])
    make_csv(csv_dir / "b.csv", [default_row("2025-04-01T10:00:00")])

    output = str(tmp_path / "out.pq")
    run_sync(
        output=output,
        directory=str(csv_dir),
        explicit_inputs=None,
        sync_mode=False,
        dry_run=False,
        verify=False,
        batch_size=500,
        max_workers=1,
    )

    assert count_partition_files(output, 2025, 3) == 1
    assert count_partition_files(output, 2025, 4) == 1
    assert os.path.exists(manifest_path(output))


def test_second_identical_run_republishes_nothing(tmp_path):
    csv_dir = tmp_path / "csv"
    csv_dir.mkdir()
    make_csv(csv_dir / "a.csv", [default_row("2025-03-01T10:00:00")])
    output = str(tmp_path / "out.pq")

    run_sync(
        output=output,
        directory=str(csv_dir),
        explicit_inputs=None,
        sync_mode=False,
        dry_run=False,
        verify=False,
        batch_size=500,
        max_workers=1,
    )
    partition_file = os.path.join(output, "year=2025", "month=3", "data_0.parquet")
    first_mtime = os.stat(partition_file).st_mtime_ns

    run_sync(
        output=output,
        directory=str(csv_dir),
        explicit_inputs=None,
        sync_mode=False,
        dry_run=False,
        verify=False,
        batch_size=500,
        max_workers=1,
    )
    second_mtime = os.stat(partition_file).st_mtime_ns

    assert first_mtime == second_mtime


def test_touch_without_content_change_does_not_republish(tmp_path):
    csv_dir = tmp_path / "csv"
    csv_dir.mkdir()
    csv_file = csv_dir / "a.csv"
    make_csv(csv_file, [default_row("2025-03-01T10:00:00")])
    output = str(tmp_path / "out.pq")

    run_sync(
        output=output,
        directory=str(csv_dir),
        explicit_inputs=None,
        sync_mode=False,
        dry_run=False,
        verify=False,
        batch_size=500,
        max_workers=1,
    )
    partition_file = os.path.join(output, "year=2025", "month=3", "data_0.parquet")
    first_mtime = os.stat(partition_file).st_mtime_ns

    # touch: change mtime without changing content
    time.sleep(0.01)
    os.utime(csv_file, None)

    run_sync(
        output=output,
        directory=str(csv_dir),
        explicit_inputs=None,
        sync_mode=False,
        dry_run=False,
        verify=False,
        batch_size=500,
        max_workers=1,
    )
    second_mtime = os.stat(partition_file).st_mtime_ns

    assert first_mtime == second_mtime


def test_content_change_republishes_partition_with_new_rows_only(tmp_path):
    csv_dir = tmp_path / "csv"
    csv_dir.mkdir()
    csv_file = csv_dir / "a.csv"
    make_csv(csv_file, [default_row("2025-03-01T10:00:00", event_id="ev1")])
    output = str(tmp_path / "out.pq")

    run_sync(
        output=output,
        directory=str(csv_dir),
        explicit_inputs=None,
        sync_mode=False,
        dry_run=False,
        verify=False,
        batch_size=500,
        max_workers=1,
    )

    time.sleep(0.01)
    make_csv(csv_file, [default_row("2025-03-01T10:00:00", event_id="ev2")])
    run_sync(
        output=output,
        directory=str(csv_dir),
        explicit_inputs=None,
        sync_mode=False,
        dry_run=False,
        verify=False,
        batch_size=500,
        max_workers=1,
    )

    assert count_partition_files(output, 2025, 3) == 1
    df = read_partition_rows(output, 2025, 3)
    assert list(df["event_id"]) == ["ev2"]


def test_sync_delete_removes_rows_and_keeps_single_file(tmp_path):
    csv_dir = tmp_path / "csv"
    csv_dir.mkdir()
    csv_file_a = csv_dir / "a.csv"
    csv_file_b = csv_dir / "b.csv"
    make_csv(csv_file_a, [default_row("2025-03-01T10:00:00", station="ST1", event_id="ev1")])
    make_csv(csv_file_b, [default_row("2025-03-01T11:00:00", station="ST2", event_id="ev2")])
    output = str(tmp_path / "out.pq")

    run_sync(
        output=output,
        directory=str(csv_dir),
        explicit_inputs=None,
        sync_mode=True,
        dry_run=False,
        verify=False,
        batch_size=500,
        max_workers=1,
    )
    df = read_partition_rows(output, 2025, 3)
    assert len(df) == 2

    os.remove(csv_file_b)
    run_sync(
        output=output,
        directory=str(csv_dir),
        explicit_inputs=None,
        sync_mode=True,
        dry_run=False,
        verify=False,
        batch_size=500,
        max_workers=1,
    )

    assert count_partition_files(output, 2025, 3) == 1
    df = read_partition_rows(output, 2025, 3)
    assert list(df["event_id"]) == ["ev1"]


def test_input_mode_never_deletes(tmp_path):
    csv_dir = tmp_path / "csv"
    csv_dir.mkdir()
    csv_file_a = csv_dir / "a.csv"
    csv_file_b = csv_dir / "b.csv"
    make_csv(csv_file_a, [default_row("2025-03-01T10:00:00", station="ST1", event_id="ev1")])
    make_csv(csv_file_b, [default_row("2025-03-01T11:00:00", station="ST2", event_id="ev2")])
    output = str(tmp_path / "out.pq")

    run_sync(
        output=output,
        directory=None,
        explicit_inputs=[str(csv_file_a), str(csv_file_b)],
        sync_mode=False,
        dry_run=False,
        verify=False,
        batch_size=500,
        max_workers=1,
    )

    # second run only lists file a (file b "missing" from the partial list)
    run_sync(
        output=output,
        directory=None,
        explicit_inputs=[str(csv_file_a)],
        sync_mode=False,
        dry_run=False,
        verify=False,
        batch_size=500,
        max_workers=1,
    )

    df = read_partition_rows(output, 2025, 3)
    assert len(df) == 2  # nothing removed


def test_dry_run_writes_nothing(tmp_path):
    csv_dir = tmp_path / "csv"
    csv_dir.mkdir()
    make_csv(csv_dir / "a.csv", [default_row("2025-03-01T10:00:00")])
    output = str(tmp_path / "out.pq")

    run_sync(
        output=output,
        directory=str(csv_dir),
        explicit_inputs=None,
        sync_mode=False,
        dry_run=True,
        verify=False,
        batch_size=500,
        max_workers=1,
    )

    assert not os.path.exists(output) or not os.listdir(output)


def test_move_detection_reclassifies_as_unchanged_and_skips_republish(tmp_path):
    csv_dir = tmp_path / "csv"
    csv_dir.mkdir()
    csv_file = csv_dir / "a.csv"
    make_csv(csv_file, [default_row("2025-03-01T10:00:00", event_id="ev1")])
    output = str(tmp_path / "out.pq")

    run_sync(
        output=output,
        directory=str(csv_dir),
        explicit_inputs=None,
        sync_mode=True,
        dry_run=False,
        verify=False,
        batch_size=500,
        max_workers=1,
    )
    manifest_before = Manifest.load(manifest_path(output))
    original_key = normalize_path(str(csv_file))
    csv_id_before = manifest_before.csv_files[original_key]["csv_id"]
    partition_file = os.path.join(output, "year=2025", "month=3", "data_0.parquet")
    mtime_before = os.stat(partition_file).st_mtime_ns

    moved_dir = csv_dir / "moved"
    moved_dir.mkdir()
    moved_file = moved_dir / "a.csv"
    os.rename(csv_file, moved_file)

    run_sync(
        output=output,
        directory=str(csv_dir),
        explicit_inputs=None,
        sync_mode=True,
        dry_run=False,
        verify=False,
        batch_size=500,
        max_workers=1,
    )

    mtime_after = os.stat(partition_file).st_mtime_ns
    assert mtime_before == mtime_after  # partition never republished

    manifest_after = Manifest.load(manifest_path(output))
    new_key = normalize_path(str(moved_file))
    assert original_key not in manifest_after.csv_files
    assert new_key in manifest_after.csv_files
    assert manifest_after.csv_files[new_key]["csv_id"] == csv_id_before

    df = read_partition_rows(output, 2025, 3)
    assert len(df) == 1  # no duplication


def test_directory_rename_moves_all_csvs_without_republish(tmp_path):
    csv_dir = tmp_path / "csv"
    csv_dir.mkdir()
    make_csv(csv_dir / "a.csv", [default_row("2025-03-01T10:00:00", station="ST1", event_id="ev1")])
    make_csv(csv_dir / "b.csv", [default_row("2025-03-01T11:00:00", station="ST2", event_id="ev2")])
    output = str(tmp_path / "out.pq")

    run_sync(
        output=output,
        directory=str(csv_dir),
        explicit_inputs=None,
        sync_mode=True,
        dry_run=False,
        verify=False,
        batch_size=500,
        max_workers=1,
    )
    partition_file = os.path.join(output, "year=2025", "month=3", "data_0.parquet")
    mtime_before = os.stat(partition_file).st_mtime_ns

    renamed_dir = tmp_path / "csv_renamed"
    os.rename(csv_dir, renamed_dir)

    run_sync(
        output=output,
        directory=str(renamed_dir),
        explicit_inputs=None,
        sync_mode=True,
        dry_run=False,
        verify=False,
        batch_size=500,
        max_workers=1,
    )

    mtime_after = os.stat(partition_file).st_mtime_ns
    assert mtime_before == mtime_after

    df = read_partition_rows(output, 2025, 3)
    assert len(df) == 2


def test_verify_detects_content_change_despite_matching_signature(tmp_path):
    csv_dir = tmp_path / "csv"
    csv_dir.mkdir()
    csv_file = csv_dir / "a.csv"
    make_csv(csv_file, [default_row("2025-03-01T10:00:00", station="ST1", event_id="ev1")])
    output = str(tmp_path / "out.pq")

    run_sync(
        output=output,
        directory=str(csv_dir),
        explicit_inputs=None,
        sync_mode=False,
        dry_run=False,
        verify=False,
        batch_size=500,
        max_workers=1,
    )

    old_stat = os.stat(csv_file)
    # rewrite content (same length: "ev1" -> "ev2") then restore old mtime
    make_csv(csv_file, [default_row("2025-03-01T10:00:00", station="ST1", event_id="ev2")])
    os.utime(csv_file, ns=(old_stat.st_atime_ns, old_stat.st_mtime_ns))
    assert os.stat(csv_file).st_size == old_stat.st_size

    manifest = Manifest.load(manifest_path(output))
    classified = classify_csv_files(
        manifest, [normalize_path(str(csv_file))], sync_mode=False, verify=True
    )
    assert classified[0].status == "changed"


def test_manifest_row_counts_match_duckdb_after_multiple_runs(tmp_path):
    csv_dir = tmp_path / "csv"
    csv_dir.mkdir()
    make_csv(csv_dir / "a.csv", [default_row("2025-03-01T10:00:00", station="ST1", event_id="ev1")])
    output = str(tmp_path / "out.pq")

    run_sync(
        output=output,
        directory=str(csv_dir),
        explicit_inputs=None,
        sync_mode=True,
        dry_run=False,
        verify=False,
        batch_size=500,
        max_workers=1,
    )
    make_csv(csv_dir / "b.csv", [default_row("2025-03-01T11:00:00", station="ST2", event_id="ev2")])
    run_sync(
        output=output,
        directory=str(csv_dir),
        explicit_inputs=None,
        sync_mode=True,
        dry_run=False,
        verify=False,
        batch_size=500,
        max_workers=1,
    )

    manifest = Manifest.load(manifest_path(output))
    expected = manifest.partitions["2025-03"]["row_count"]
    df = read_partition_rows(output, 2025, 3)
    assert len(df) == expected


def test_conversion_failure_leaves_dataset_and_manifest_untouched(tmp_path):
    csv_dir = tmp_path / "csv"
    csv_dir.mkdir()
    bad_csv = csv_dir / "bad.csv"
    with open(bad_csv, "w", encoding="utf-8") as fh:
        fh.write("not,the,right,header\n1,2,3,4\n")
    output = str(tmp_path / "out.pq")

    with pytest.raises(Exception):
        run_sync(
            output=output,
            directory=str(csv_dir),
            explicit_inputs=None,
            sync_mode=False,
            dry_run=False,
            verify=False,
            batch_size=500,
            max_workers=1,
        )

    assert not os.path.exists(os.path.join(output, "year=2025"))
