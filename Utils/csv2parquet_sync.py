#!/usr/bin/env python
"""Incremental CSV -> Parquet sync: add/update/remove CSV sources into a single
compact Parquet dataset (one file per (year, month) partition) without a full
dataset rewrite on every run.
"""
import argparse
import concurrent.futures
import hashlib
import json
import os
import shutil
import sys
import tempfile
import uuid
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from datetime import timezone
from typing import Dict
from typing import List
from typing import Optional
from typing import Set
from typing import Tuple

import duckdb
import tqdm

CSV_COLUMNS = {
    "station_id": "VARCHAR",
    "channel": "VARCHAR",
    "phase_type": "VARCHAR",
    "phase_time": "VARCHAR",
    "phase_score": "DOUBLE",
    "phase_evaluation": "VARCHAR",
    "phase_method": "VARCHAR",
    "event_id": "VARCHAR",
    "agency": "VARCHAR",
}

MANIFEST_FILENAME = ".csv2parquet-manifest.json"
LOCK_FILENAME = ".csv2parquet.lock"
SCHEMA_VERSION = 1
HASH_CHUNK_SIZE = 1024 * 1024


def normalize_path(path: str) -> str:
    return os.path.realpath(path)


def file_signature(path: str) -> Tuple[int, int]:
    st = os.stat(path)
    return st.st_size, st.st_mtime_ns


def sha256_of_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(HASH_CHUNK_SIZE)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def new_csv_id(path: str) -> str:
    seed = f"{path}:{datetime.now(timezone.utc).isoformat()}:{uuid.uuid4()}"
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()[:16]


@dataclass
class Manifest:
    schema_version: int = SCHEMA_VERSION
    created_at: str = ""
    updated_at: str = ""
    csv_files: Dict[str, dict] = field(default_factory=dict)
    partitions: Dict[str, dict] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str) -> "Manifest":
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if data.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(
                f"Manifest {path} has schema_version={data.get('schema_version')}, "
                f"expected {SCHEMA_VERSION}. Refusing to mix incompatible manifests."
            )
        return cls(
            schema_version=data["schema_version"],
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
            csv_files=data.get("csv_files", {}),
            partitions=data.get("partitions", {}),
        )

    @classmethod
    def new(cls) -> "Manifest":
        now = datetime.now(timezone.utc).isoformat()
        return cls(created_at=now, updated_at=now)

    def save(self, path: str) -> None:
        self.updated_at = datetime.now(timezone.utc).isoformat()
        payload = {
            "schema_version": self.schema_version,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "csv_files": self.csv_files,
            "partitions": self.partitions,
        }
        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=os.path.dirname(path) or ".", prefix=".manifest-", suffix=".tmp"
        )
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2, sort_keys=True)
            os.replace(tmp_path, path)
        except BaseException:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise


def manifest_summary(manifest: "Manifest") -> Tuple[int, Optional[str], Optional[str]]:
    """Total row count and (min, max) phase_time (ISO, second precision)
    currently tracked by the manifest, derived from partition metadata alone
    (no Parquet read needed). Falls back to the partition's 'YYYY-MM' key when
    an older manifest entry predates min/max_phase_time tracking."""
    total_rows = sum(p["row_count"] for p in manifest.partitions.values())
    if not manifest.partitions:
        return total_rows, None, None
    min_times = []
    max_times = []
    for partition_key, p in manifest.partitions.items():
        min_times.append(p.get("min_phase_time") or partition_key)
        max_times.append(p.get("max_phase_time") or partition_key)
    return total_rows, min(min_times), max(max_times)


def manifest_path(output: str) -> str:
    return os.path.join(output, MANIFEST_FILENAME)


def lock_path(output: str) -> str:
    return os.path.join(output, LOCK_FILENAME)


class OutputLock:
    """Simple single-host lock file guarding an --output dataset directory."""

    def __init__(self, output: str):
        self.path = lock_path(output)
        self._fd: Optional[int] = None

    def __enter__(self) -> "OutputLock":
        try:
            self._fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            raise RuntimeError(
                f"Lock file already exists: {self.path}\n"
                "Another run may be in progress, or a stale lock was left behind "
                "by a crashed run and must be removed manually."
            ) from None
        os.write(self._fd, str(os.getpid()).encode("utf-8"))
        os.close(self._fd)
        self._fd = None
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if os.path.exists(self.path):
            os.remove(self.path)


@dataclass
class ClassifiedCsv:
    path: str
    status: str  # "new" | "changed" | "unchanged" | "deleted"
    csv_id: Optional[str] = None
    old_partitions: Optional[Dict[str, dict]] = None


def discover_csv_files(directory: Optional[str], explicit_inputs: Optional[List[str]]) -> List[str]:
    if directory:
        print(f"Scanning {directory} for CSV files...")
        found = []
        for root, _, files in os.walk(directory):
            for filename in files:
                if filename.endswith(".csv"):
                    found.append(normalize_path(os.path.join(root, filename)))
        print(f"Found {len(found)} CSV files.")
        return found
    return [normalize_path(f) for f in (explicit_inputs or [])]


def classify_csv_files(
    manifest: Manifest,
    discovered_paths: List[str],
    sync_mode: bool,
    verify: bool,
) -> List[ClassifiedCsv]:
    """Classify discovered CSV files as new/changed/unchanged, detecting file
    moves by (size, sha256) match against manifest entries whose recorded path
    no longer exists on disk, so a move never triggers reingestion."""
    results: List[ClassifiedCsv] = []
    discovered_set = set(discovered_paths)

    # orphan manifest entries: recorded path no longer on disk, not yet claimed
    print(f"Comparing {len(discovered_paths)} discovered files against {len(manifest.csv_files)} manifest entries...")
    orphans_by_size: Dict[int, List[str]] = {}
    for csv_path, entry in manifest.csv_files.items():
        if csv_path in discovered_set:
            continue
        if not os.path.exists(csv_path):
            orphans_by_size.setdefault(entry["size"], []).append(csv_path)

    claimed_orphans: Set[str] = set()

    for path in tqdm.tqdm(discovered_paths, desc="Classifying"):
        entry = manifest.csv_files.get(path)
        if entry is not None:
            size, mtime_ns = file_signature(path)
            if size == entry["size"] and mtime_ns == entry["mtime_ns"] and not verify:
                results.append(ClassifiedCsv(path=path, status="unchanged"))
                continue
            file_hash = sha256_of_file(path)
            if file_hash == entry["sha256"]:
                entry["size"] = size
                entry["mtime_ns"] = mtime_ns
                results.append(ClassifiedCsv(path=path, status="unchanged"))
                continue
            results.append(
                ClassifiedCsv(
                    path=path,
                    status="changed",
                    csv_id=entry["csv_id"],
                    old_partitions=entry["partitions"],
                )
            )
            continue

        # not found under its current path: look for a move before assuming "new"
        size, _ = file_signature(path)
        candidates = orphans_by_size.get(size, [])
        moved_from = None
        if candidates:
            file_hash = sha256_of_file(path)
            for old_path in candidates:
                if old_path in claimed_orphans:
                    continue
                if manifest.csv_files[old_path]["sha256"] == file_hash:
                    moved_from = old_path
                    break

        if moved_from is not None:
            claimed_orphans.add(moved_from)
            moved_entry = manifest.csv_files.pop(moved_from)
            size, mtime_ns = file_signature(path)
            moved_entry["size"] = size
            moved_entry["mtime_ns"] = mtime_ns
            manifest.csv_files[path] = moved_entry
            results.append(ClassifiedCsv(path=path, status="unchanged"))
        else:
            results.append(ClassifiedCsv(path=path, status="new"))

    if sync_mode:
        for csv_path, entry in list(manifest.csv_files.items()):
            if csv_path in discovered_set:
                continue
            if csv_path in claimed_orphans:
                continue
            if not os.path.exists(csv_path):
                results.append(
                    ClassifiedCsv(
                        path=csv_path,
                        status="deleted",
                        csv_id=entry["csv_id"],
                        old_partitions=entry["partitions"],
                    )
                )

    return results


def convert_batch_to_staging(
    csv_id_by_path: Dict[str, str],
    staging_dir: str,
) -> None:
    """Convert one batch of CSV files into a partitioned staging area, tagging
    every row with its source CSV's stable _csv_id."""
    conn = duckdb.connect()
    try:
        selects = []
        for path, csv_id in csv_id_by_path.items():
            selects.append(
                f"""
                SELECT
                    '{csv_id}' AS _csv_id,
                    station_id,
                    channel,
                    phase_type,
                    time_bucket(INTERVAL '1 millisecond', phase_time::TIMESTAMP) AS phase_time,
                    phase_score,
                    phase_evaluation,
                    phase_method,
                    event_id,
                    agency
                FROM read_csv(['{path}'], columns = {CSV_COLUMNS!r}, header = true)
                """
            )
        union_sql = " UNION ALL BY NAME ".join(selects)
        sql = f"""
            COPY (
                SELECT *, year(phase_time) AS year, month(phase_time) AS month
                FROM ({union_sql})
            )
            TO '{staging_dir}'
            (FORMAT 'parquet', PARTITION_BY (year, month), COMPRESSION 'snappy');
        """
        conn.execute(sql)
    finally:
        conn.close()


def convert_one_batch(staging_root: str, batch_index: int, csv_id_by_path: Dict[str, str]) -> str:
    """Module-level worker entry point: must not be a closure, so that
    ProcessPoolExecutor can pickle it when dispatching to worker processes."""
    batch_dir = os.path.join(staging_root, f"batch-{batch_index:06d}")
    os.makedirs(batch_dir, exist_ok=True)
    convert_batch_to_staging(csv_id_by_path, batch_dir)
    return batch_dir


def republish_partition(
    output: str,
    partition_key: str,
    staging_parquet_files: List[str],
    excluded_csv_ids: Set[str],
) -> Tuple[int, Optional[str], Optional[str]]:
    """Rewrite a single (year, month) partition as one Parquet file, merging its
    previous content with new staged rows and dropping rows for csv_ids being
    replaced/removed. Returns (row_count, min_phase_time, max_phase_time)."""
    year, month = partition_key.split("-")
    partition_dir = os.path.join(output, f"year={int(year)}", f"month={int(month)}")
    os.makedirs(partition_dir, exist_ok=True)
    existing_file = os.path.join(partition_dir, "data_0.parquet")
    tmp_file = os.path.join(partition_dir, ".data_0.parquet.tmp")

    has_existing = os.path.exists(existing_file)

    if not staging_parquet_files and not has_existing:
        return 0, None, None

    # the exclusion filter only applies to the previous partition content:
    # staged rows always carry the *current* version of their csv_id and must
    # never be filtered out, even if that id is also being excluded from the
    # old file (case: a "changed" CSV keeps the same csv_id across versions).
    select_parts = []
    if has_existing:
        exclusion_clause = ""
        if excluded_csv_ids:
            ids_list = ", ".join(f"'{cid}'" for cid in excluded_csv_ids)
            exclusion_clause = f"WHERE _csv_id NOT IN ({ids_list})"
        select_parts.append(
            f"SELECT * EXCLUDE (year, month) FROM read_parquet(['{existing_file}']) {exclusion_clause}"
        )
    if staging_parquet_files:
        staged_list = ", ".join(f"'{f}'" for f in staging_parquet_files)
        select_parts.append(
            f"SELECT * EXCLUDE (year, month) FROM read_parquet([{staged_list}])"
        )
    union_sql = " UNION ALL BY NAME ".join(select_parts)

    conn = duckdb.connect()
    try:
        conn.execute(
            f"""
            COPY (
                {union_sql}
            )
            TO '{tmp_file}'
            (FORMAT 'parquet', COMPRESSION 'snappy');
            """
        )
        row_count, min_phase_time, max_phase_time = conn.execute(
            f"SELECT COUNT(*), MIN(phase_time), MAX(phase_time) FROM read_parquet('{tmp_file}')"
        ).fetchone()
    finally:
        conn.close()

    os.replace(tmp_file, existing_file)
    min_iso = min_phase_time.isoformat() if min_phase_time is not None else None
    max_iso = max_phase_time.isoformat() if max_phase_time is not None else None
    return row_count, min_iso, max_iso


def print_run_summary(
    rows_before: int,
    min_time_before: Optional[str],
    max_time_before: Optional[str],
    rows_after: int,
    min_time_after: Optional[str],
    max_time_after: Optional[str],
) -> None:
    """Print the pick count delta and the time range extension gained by this run."""
    print(f"Picks: {rows_before} -> {rows_after} ({rows_after - rows_before:+d})")

    range_before = f"{min_time_before}..{max_time_before}" if min_time_before is not None else "(empty)"
    range_after = f"{min_time_after}..{max_time_after}" if min_time_after is not None else "(empty)"

    if range_before == range_after:
        print(f"Time range: {range_after} (unchanged)")
        return

    if min_time_before is None or min_time_after is None or max_time_before is None or max_time_after is None:
        print(f"Time range: {range_before} -> {range_after}")
        return

    extensions = []
    if min_time_after < min_time_before:
        extensions.append(f"back to {min_time_after} (was {min_time_before})")
    if max_time_after > max_time_before:
        extensions.append(f"forward to {max_time_after} (was {max_time_before})")
    extension_note = f" [extended {', '.join(extensions)}]" if extensions else ""
    print(f"Time range: {range_before} -> {range_after}{extension_note}")


def run_sync(
    output: str,
    directory: Optional[str],
    explicit_inputs: Optional[List[str]],
    sync_mode: bool,
    dry_run: bool,
    verify: bool,
    batch_size: int,
    max_workers: int,
) -> None:
    os.makedirs(output, exist_ok=True)
    m_path = manifest_path(output)
    manifest = Manifest.load(m_path) if os.path.exists(m_path) else Manifest.new()
    rows_before, min_time_before, max_time_before = manifest_summary(manifest)

    discovered = discover_csv_files(directory, explicit_inputs)
    classified = classify_csv_files(manifest, discovered, sync_mode, verify)

    new_or_changed = [c for c in classified if c.status in ("new", "changed")]
    unchanged = [c for c in classified if c.status == "unchanged"]
    deleted = [c for c in classified if c.status == "deleted"]

    print(
        f"new={sum(1 for c in classified if c.status == 'new')} "
        f"changed={sum(1 for c in classified if c.status == 'changed')} "
        f"unchanged={len(unchanged)} deleted={len(deleted)}"
    )

    if not new_or_changed and not deleted:
        if not dry_run:
            manifest.save(m_path)
        print("Nothing to do.")
        rows_after, min_time_after, max_time_after = manifest_summary(manifest)
        print_run_summary(
            rows_before, min_time_before, max_time_before, rows_after, min_time_after, max_time_after
        )
        return

    affected_partitions: Set[str] = set()
    excluded_by_partition: Dict[str, Set[str]] = {}

    for c in new_or_changed:
        if c.status == "changed" and c.old_partitions:
            for partition_key in c.old_partitions:
                affected_partitions.add(partition_key)
                excluded_by_partition.setdefault(partition_key, set()).add(c.csv_id)

    for c in deleted:
        if c.old_partitions:
            for partition_key in c.old_partitions:
                affected_partitions.add(partition_key)
                excluded_by_partition.setdefault(partition_key, set()).add(c.csv_id)

    if dry_run:
        print(f"Would republish partitions: {sorted(affected_partitions) or '(pending discovery)'}")
        range_before = (
            f"{min_time_before}..{max_time_before}" if min_time_before is not None else "(empty)"
        )
        print(f"Current picks: {rows_before}, current time range: {range_before}")
        print("Dry run: no changes written.")
        return

    for c in new_or_changed:
        if c.csv_id is None:
            c.csv_id = new_csv_id(c.path)

    staging_root = os.path.join(output, f".staging-{uuid.uuid4().hex[:12]}")
    os.makedirs(staging_root, exist_ok=True)

    try:
        batches = [
            new_or_changed[i : i + batch_size]
            for i in range(0, len(new_or_changed), batch_size)
        ]

        batch_dirs: List[str] = []
        if batches:
            if max_workers > 1:
                with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as pool:
                    futures = {
                        pool.submit(
                            convert_one_batch,
                            staging_root,
                            i,
                            {c.path: c.csv_id for c in batch},
                        ): i
                        for i, batch in enumerate(batches)
                    }
                    for future in tqdm.tqdm(
                        concurrent.futures.as_completed(futures), total=len(futures)
                    ):
                        batch_dirs.append(future.result())
            else:
                for i, batch in enumerate(tqdm.tqdm(batches)):
                    batch_dirs.append(
                        convert_one_batch(staging_root, i, {c.path: c.csv_id for c in batch})
                    )

        # discover partitions produced by staging and validate each batch
        staging_files_by_partition: Dict[str, List[str]] = {}
        for batch_dir in batch_dirs:
            for root, _, files in os.walk(batch_dir):
                for filename in files:
                    if not filename.endswith(".parquet"):
                        continue
                    full_path = os.path.join(root, filename)
                    year_part = None
                    month_part = None
                    for segment in full_path.split(os.sep):
                        if segment.startswith("year="):
                            year_part = segment.split("=", 1)[1]
                        elif segment.startswith("month="):
                            month_part = segment.split("=", 1)[1]
                    if year_part is None or month_part is None:
                        continue
                    partition_key = f"{int(year_part):04d}-{int(month_part):02d}"
                    staging_files_by_partition.setdefault(partition_key, []).append(full_path)
                    affected_partitions.add(partition_key)

        # validate: every new/changed CSV must have contributed at least one row
        conn = duckdb.connect()
        try:
            for batch_dir in batch_dirs:
                count = conn.execute(
                    f"SELECT COUNT(*) FROM read_parquet('{batch_dir}/**/*.parquet')"
                ).fetchone()[0]
                if count == 0:
                    raise RuntimeError(f"Batch {batch_dir} produced zero rows, aborting run.")
        finally:
            conn.close()

        # republish every affected partition as a single file
        partition_stats: Dict[str, Tuple[int, Optional[str], Optional[str]]] = {}
        for partition_key in tqdm.tqdm(sorted(affected_partitions), desc="Republishing partitions"):
            staged_files = staging_files_by_partition.get(partition_key, [])
            excluded_ids = excluded_by_partition.get(partition_key, set())
            partition_stats[partition_key] = republish_partition(
                output, partition_key, staged_files, excluded_ids
            )

        # only now update the manifest, after every partition published successfully.
        # Compute every CSV's per-partition row count in a single grouped pass over
        # the whole staging area instead of one query per CSV (O(n) instead of O(n^2)
        # over hundreds of thousands of files).
        row_counts_by_csv_id: Dict[str, Dict[str, int]] = {}
        if batch_dirs:
            print(f"Computing per-CSV row counts from {len(batch_dirs)} staged batches...")
            conn = duckdb.connect()
            try:
                rows = conn.execute(
                    f"""
                    SELECT _csv_id, printf('%04d-%02d', year, month) AS partition_key, COUNT(*) AS n
                    FROM read_parquet('{staging_root}/**/*.parquet')
                    GROUP BY _csv_id, year, month
                    """
                ).fetchall()
            finally:
                conn.close()
            for csv_id, partition_key, n in rows:
                row_counts_by_csv_id.setdefault(csv_id, {})[partition_key] = n
            print("Updating manifest entries...")

        now = datetime.now(timezone.utc).isoformat()
        for c in new_or_changed:
            assert c.csv_id is not None  # assigned above for every new_or_changed entry
            size, mtime_ns = file_signature(c.path)
            file_hash = sha256_of_file(c.path)
            per_partition_counts = row_counts_by_csv_id.get(c.csv_id, {})
            manifest.csv_files[c.path] = {
                "csv_id": c.csv_id,
                "size": size,
                "mtime_ns": mtime_ns,
                "sha256": file_hash,
                "sha256_computed_at": now,
                "partitions": {pk: {"row_count": n} for pk, n in per_partition_counts.items()},
            }

        for c in deleted:
            manifest.csv_files.pop(c.path, None)

        for partition_key, (row_count, min_phase_time, max_phase_time) in partition_stats.items():
            manifest.partitions[partition_key] = {
                "row_count": row_count,
                "min_phase_time": min_phase_time,
                "max_phase_time": max_phase_time,
            }

        manifest.save(m_path)

        rows_after, min_time_after, max_time_after = manifest_summary(manifest)
        print_run_summary(
            rows_before, min_time_before, max_time_before, rows_after, min_time_after, max_time_after
        )
    finally:
        if os.path.exists(staging_root):
            shutil.rmtree(staging_root)


def main():
    parser = argparse.ArgumentParser(
        description="Incrementally sync CSV files into a compact partitioned Parquet dataset"
    )
    parser.add_argument("-i", "--input", nargs="+", help="CSV input files (multiple files allowed)")
    parser.add_argument("-o", "--output", required=True, help="Parquet dataset output directory")
    parser.add_argument("-d", "--directory", type=str, help="Input directory containing CSV files")
    parser.add_argument(
        "--sync",
        action="store_true",
        help="Mirror mode: remove data for CSV files no longer present (only with --directory)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would change without writing anything",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Recompute SHA-256 for every considered CSV, even if (size, mtime_ns) matches",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=500,
        help="Number of CSV files converted together per DuckDB pass (default: 500)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="Number of parallel worker processes for batch conversion (default: 1)",
    )
    args = parser.parse_args()

    if args.directory and args.input:
        print("Cannot specify both input directory and input files")
        sys.exit(1)

    if args.sync and not args.directory:
        print("--sync requires --directory")
        sys.exit(1)

    if not args.directory and not args.input:
        print("No input files specified")
        sys.exit(1)

    if os.path.exists(args.output) and not os.path.isdir(args.output):
        print(f"Output {args.output} exists and is not a directory")
        sys.exit(1)

    if os.path.exists(args.output) and not os.path.exists(manifest_path(args.output)):
        contains_parquet = any(
            filename.endswith(".parquet")
            for _, _, files in os.walk(args.output)
            for filename in files
        )
        if contains_parquet:
            print(
                f"Output directory {args.output} already contains Parquet data but no "
                f"{MANIFEST_FILENAME}. Refusing to mix untracked data with the incremental "
                "manifest-based pipeline."
            )
            sys.exit(1)

    if not args.directory:
        for f in args.input:
            if not os.path.exists(f):
                print(f"File {f} does not exist !")
                sys.exit(1)

    os.makedirs(args.output, exist_ok=True)
    with OutputLock(args.output):
        run_sync(
            output=args.output,
            directory=args.directory,
            explicit_inputs=args.input,
            sync_mode=args.sync,
            dry_run=args.dry_run,
            verify=args.verify,
            batch_size=args.batch_size,
            max_workers=args.max_workers,
        )


if __name__ == "__main__":
    main()
