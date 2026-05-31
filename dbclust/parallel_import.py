#!/usr/bin/env python
"""
Parallel QuakeML import script.
Imports QuakeML files in parallel into separate temporary databases,
then merges them into a single final database.
"""
import argparse
import json
import logging
import os
import shutil
import sqlite3
import sys
import tempfile
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from obspy import read_events
from tqdm import tqdm  # Used only in parallel_import(), not in merge_databases()

from dbclust.inject_spatialite import (
    create_schema,
    import_catalog_to_sqlite,
    add_agency_names,
    create_indexes_sql,
    register_geometry_for_view,
    create_safe_connection,
    load_spatialite,
    create_tables,
)

logger = logging.getLogger("dbclust.parallel_import")


def deduplicate_cross_partition_events(
    conn: sqlite3.Connection,
    overlap_window_s: float = 250.0,
    min_shared_catalog_ids: int = 2,
    min_shared_auto_picks: int = 3,
    report_path: str = None,
) -> list:
    """Detect and remove cross-partition duplicate events after merge.

    Two events are duplicates if their manual picks share >= min_shared_catalog_ids
    distinct source_event_id values. The survivor is the event whose first pick is
    the earliest (i.e. the job that saw the most picks). The loser is deleted via
    CASCADE.

    When the difference between first picks exceeds overlap_window_s / 2, the
    survivor's localisation may be sub-optimal (the job that produced it did not
    see all picks). A WARNING is logged and the event is flagged for re-localisation
    with a suggested time window.

    Returns a list of dicts describing events recommended for re-localisation:
      {event_id, origin_time, first_pick, last_pick, suggested_start, suggested_stop}
    """
    cursor = conn.cursor()

    # Two events are duplicates if they share manual picks with the same source_event_id
    # (catalog-based criterion), OR if they share automatic picks on the same station
    # at the exact same time (pick-identity criterion for events with no manual picks).
    cursor.execute("""
        SELECT ev1, ev2, MAX(n_shared) AS n_shared FROM (
            -- Criterion 1: shared catalog source_event_ids (manual picks)
            SELECT p1.event_id AS ev1, p2.event_id AS ev2,
                   COUNT(DISTINCT p1.source_event_id) AS n_shared
            FROM picks p1
            JOIN picks p2
              ON p1.source_event_id = p2.source_event_id
             AND p1.event_id < p2.event_id
             AND p1.source_event_id IS NOT NULL
             AND p1.evaluation_mode = 'manual'
            WHERE p2.evaluation_mode = 'manual'
            GROUP BY p1.event_id, p2.event_id
            HAVING n_shared >= ?

            UNION ALL

            -- Criterion 2: shared automatic picks (same station, phase, time)
            SELECT p1.event_id AS ev1, p2.event_id AS ev2,
                   COUNT(*) AS n_shared
            FROM picks p1
            JOIN picks p2
              ON p1.station_name = p2.station_name
             AND p1.phase_hint = p2.phase_hint
             AND p1.pick_time = p2.pick_time
             AND p1.event_id < p2.event_id
             AND p1.evaluation_mode = 'automatic'
             AND p2.evaluation_mode = 'automatic'
            GROUP BY p1.event_id, p2.event_id
            HAVING n_shared >= ?
        )
        GROUP BY ev1, ev2
    """, (min_shared_catalog_ids, min_shared_auto_picks))
    pairs = cursor.fetchall()

    logger.info(f"Cross-partition deduplication: {len(pairs)} duplicate pair(s) found.")
    reloc_needed = []
    removed_pairs = []

    for ev1, ev2, n_shared in pairs:
        # Per-event: first_pick, preferred origin time, phase count
        cursor.execute(
            "SELECT MIN(pick_time), time, used_phase_count FROM picks "
            "JOIN origins USING (event_id) WHERE picks.event_id = ? "
            "ORDER BY used_phase_count DESC LIMIT 1", (ev1,)
        )
        row1 = cursor.fetchone() or (None, None, 0)
        cursor.execute(
            "SELECT MIN(pick_time), time, used_phase_count FROM picks "
            "JOIN origins USING (event_id) WHERE picks.event_id = ? "
            "ORDER BY used_phase_count DESC LIMIT 1", (ev2,)
        )
        row2 = cursor.fetchone() or (None, None, 0)
        fp1, t1, nph1 = row1
        fp2, t2, nph2 = row2

        # Winner = earliest first_pick; tie-break on most phases
        if (fp1 or "") <= (fp2 or "") or (fp1 == fp2 and (nph1 or 0) >= (nph2 or 0)):
            winner, loser = ev1, ev2
            t_winner, nph_winner, fp_winner = t1, nph1, fp1
            fp_loser, nph_loser = fp2, nph2
        else:
            winner, loser = ev2, ev1
            t_winner, nph_winner, fp_winner = t2, nph2, fp2
            fp_loser, nph_loser = fp1, nph1

        # Pick span across both events = last - first over the union of all picks
        cursor.execute(
            "SELECT MIN(pick_time), MAX(pick_time) FROM picks WHERE event_id IN (?, ?)",
            (ev1, ev2),
        )
        first_pick_union, last_pick_union = cursor.fetchone()

        logger.info(
            f"Cross-partition duplicate: keeping {winner} (first_pick={fp_winner}, phases={nph_winner}), "
            f"removing {loser} (first_pick={fp_loser}, phases={nph_loser}), "
            f"n_shared_picks={n_shared}"
        )
        cursor.execute("DELETE FROM events WHERE event_id = ?", (loser,))
        removed_pairs.append({
            "kept": {"event_id": winner, "origin_time": t_winner, "phases": nph_winner, "first_pick": fp_winner},
            "removed": {"event_id": loser, "phases": nph_loser, "first_pick": fp_loser},
            "n_shared_picks": n_shared,
        })

        # If the union of picks spans more than overlap_window_s, neither job could
        # see all picks — the surviving localisation is sub-optimal.
        try:
            t_first = datetime.fromisoformat(first_pick_union.replace("Z", "+00:00"))
            t_last = datetime.fromisoformat(last_pick_union.replace("Z", "+00:00"))
            pick_span_s = (t_last - t_first).total_seconds()
            if pick_span_s > overlap_window_s:
                margin = timedelta(minutes=5)
                suggested_start = (t_first - margin).strftime("%Y-%m-%dT%H:%M:%S")
                suggested_stop = (t_last + margin).strftime("%Y-%m-%dT%H:%M:%S")
                logger.warning(
                    f"Sub-optimal localisation: {winner} (origin={t_winner}, phases={nph_winner}) — "
                    f"pick span={pick_span_s:.0f}s > overlap={overlap_window_s:.0f}s. "
                    f"Re-localisation recommended on window "
                    f"[{suggested_start}, {suggested_stop}] with a single job."
                )
                reloc_needed.append({
                    "event_id": winner,
                    "origin_time": t_winner,
                    "first_pick": first_pick_union,
                    "last_pick": last_pick_union,
                    "suggested_start": suggested_start,
                    "suggested_stop": suggested_stop,
                })
        except Exception as e:
            logger.warning(f"Could not compute pick span for pair ({winner}, {loser}): {e}")

    conn.commit()
    n_removed = len(pairs)
    n_reloc = len(reloc_needed)
    logger.info(
        f"Cross-partition deduplication: {n_removed} duplicate(s) removed, "
        f"{n_reloc} re-run(s) recommended."
    )

    if report_path:
        report = {
            "summary": {
                "duplicates_removed": n_removed,
                "rerun_recommended": n_reloc,
                "overlap_window_s": overlap_window_s,
                "generated_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            },
            "duplicates_removed": removed_pairs,
            "rerun_recommended": reloc_needed,
        }
        try:
            with open(report_path, "w") as f:
                json.dump(report, f, indent=2, default=str)
            logger.info(f"Deduplication report written to {report_path}")
        except OSError as e:
            logger.warning(f"Could not write deduplication report: {e}")

    return reloc_needed


def import_file_to_temp_db(args_tuple):
    """
    Import a single QuakeML file into a temporary database.
    
    Args:
        args_tuple: Tuple of (input_file, temp_dir, enable_quakeml)
        
    Returns:
        Tuple of (input_file, temp_db_path, success, error_message)
    """
    input_file, temp_dir, enable_quakeml = args_tuple
    temp_db_path = None
    
    try:
        # Create unique temporary database for this file
        base_name = Path(input_file).stem
        temp_db_path = os.path.join(temp_dir, f"{base_name}.db")
        
        logger.info(f"Processing {input_file} -> {temp_db_path}")
        
        # Create schema with DELETE journal mode (not WAL) to avoid locking issues
        conn = create_safe_connection(temp_db_path, logger=logger)
        
        # Override WAL mode to DELETE for temporary databases
        conn.execute("PRAGMA journal_mode=DELETE")
        conn.commit()
        
        # Load SpatiaLite
        if not load_spatialite(conn, logger):
            raise RuntimeError("Failed to load SpatiaLite")
        
        cursor = conn.cursor()
        
        # Initialize SpatiaLite metadata
        cursor.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='spatial_ref_sys';"
        )
        if cursor.fetchone()[0] == 0:
            cursor.execute("SELECT InitSpatialMetadata();")
        
        # Create tables
        create_tables(cursor)
        
        # Add geometry column
        cursor.execute(
            "SELECT AddGeometryColumn('origins', 'geometry', 4326, 'POINT', 'XY');"
        )
        conn.commit()
        
        # Read and import catalog
        catalog = read_events(input_file)
        import_catalog_to_sqlite(conn, catalog, enable_quakeml, disable_tqdm=True)
        
        # Ensure all data is written and close connection properly
        conn.commit()
        conn.close()
        
        return (input_file, temp_db_path, True, None)
        
    except Exception as e:
        error_msg = f"Error processing {input_file}: {e}"
        logger.error(error_msg)
        
        # Clean up failed database
        if temp_db_path and os.path.exists(temp_db_path):
            try:
                os.remove(temp_db_path)
            except OSError:
                pass
                
        return (input_file, None, False, str(e))


def merge_databases(temp_db_paths, final_db_path, enable_quakeml=False, overlap_window_s=250.0):
    """
    Merge multiple temporary databases into a single final database.
    
    Args:
        temp_db_paths: List of paths to temporary databases
        final_db_path: Path to the final merged database
        enable_quakeml: Whether QuakeML data was stored
    """
    logger.info(f"Merging {len(temp_db_paths)} databases into {final_db_path}")
    
    # Create final database schema (spatial index created after data merge)
    final_conn = create_schema(final_db_path, create_spatial_index=False)
    final_cursor = final_conn.cursor()
    
    # Tables to merge (in dependency order)
    tables_to_merge = [
        "quakeml",
        "events",
        "picks",
        "origins",
        "arrivals",
        "magnitudes",
        "station_magnitudes",
        "station_magnitude_contributions",
    ]
    
    n_total = len(temp_db_paths)
    try:
        for i, temp_db_path in enumerate(temp_db_paths):
            logger.info(f"Merging database {i+1}/{n_total}: {temp_db_path}")
            
            # Open a separate read-only connection to the temp database
            temp_conn = sqlite3.connect(f"file:{temp_db_path}?mode=ro", uri=True, timeout=60)
            temp_cursor = temp_conn.cursor()
            
            try:
                # Copy data from each table
                for table in tables_to_merge:
                    try:
                        # Get column names from the temp database
                        temp_cursor.execute(f"PRAGMA table_info({table})")
                        columns = [row[1] for row in temp_cursor.fetchall()]
                        
                        if not columns:
                            logger.warning(f"Table {table} not found in temp db, skipping")
                            continue
                        
                        # Filter out geometry column for origins table (handled separately)
                        if table == "origins" and "geometry" in columns:
                            columns.remove("geometry")
                        
                        columns_str = ", ".join(columns)
                        
                        # Read data from temp database
                        if table == "origins":
                            temp_cursor.execute(f"SELECT {columns_str}, geometry FROM {table}")
                        else:
                            temp_cursor.execute(f"SELECT {columns_str} FROM {table}")
                        
                        rows = temp_cursor.fetchall()
                        
                        if not rows:
                            logger.debug(f"  No rows to insert into {table}")
                            continue
                        
                        # Insert data into final database
                        if table == "origins":
                            placeholders = ", ".join(["?"] * (len(columns) + 1))
                            final_cursor.executemany(
                                f"INSERT OR IGNORE INTO {table} ({columns_str}, geometry) VALUES ({placeholders})",
                                rows
                            )
                        else:
                            placeholders = ", ".join(["?"] * len(columns))
                            final_cursor.executemany(
                                f"INSERT OR IGNORE INTO {table} ({columns_str}) VALUES ({placeholders})",
                                rows
                            )
                        
                        rows_inserted = final_cursor.rowcount
                        logger.debug(f"  Inserted {rows_inserted} rows into {table}")
                        
                    except sqlite3.Error as e:
                        logger.error(f"Error merging table {table}: {e}")
                        raise
                
            finally:
                # Close temp connection
                temp_cursor.close()
                temp_conn.close()
            
            # Commit after each database merge
            final_conn.commit()
        
        # Post-merge operations
        logger.info("Running post-merge operations...")
        
        # Add agency names
        logger.info("Adding agency names...")
        add_agency_names(final_conn)

        # Deduplicate cross-partition events (same physical event localised by two jobs)
        report_path = str(final_db_path).replace(".db", ".dedup_report.json")
        deduplicate_cross_partition_events(
            final_conn,
            overlap_window_s=overlap_window_s,
            report_path=report_path,
        )

        # Create indexes
        logger.info("Creating indexes...")
        create_indexes_sql(final_cursor)

        # Create spatial index on final merged DB (skipped during schema creation
        # to avoid segfaults in parallel worker subprocesses on macOS ARM)
        logger.info("Creating spatial index on final database...")
        try:
            final_cursor.execute(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='idx_origins_geometry';"
            )
            row = final_cursor.fetchone()
            if row is None or row[0] == 0:
                final_cursor.execute("SELECT CreateSpatialIndex('origins', 'geometry');")
                logger.info("Spatial index created successfully.")
            else:
                logger.debug("Spatial index already exists, skipping.")
        except sqlite3.OperationalError as e:
            logger.warning(f"Could not create spatial index: {e}")

        # Register geometry for view
        logger.info("Registering geometry for view...")
        register_geometry_for_view(final_conn, "event_coordinates", "geometry")
        
        final_conn.commit()
        logger.info("Merge completed successfully")
        
    except Exception as e:
        logger.error(f"Error during merge: {e}")
        final_conn.rollback()
        raise
    finally:
        final_conn.close()


def parallel_import(input_files, output_db, enable_quakeml=False, max_workers=None, keep_temp=False):
    """
    Import QuakeML files in parallel using temporary databases.
    
    Args:
        input_files: List of QuakeML file paths
        output_db: Path to final output database
        enable_quakeml: Whether to store QuakeML data
        max_workers: Maximum number of parallel workers (None = CPU count)
        keep_temp: Whether to keep temporary databases after merge
    """
    # Create temporary directory for databases
    temp_dir = tempfile.mkdtemp(prefix="dbclust_parallel_")
    logger.info(f"Using temporary directory: {temp_dir}")
    
    try:
        # Prepare arguments for parallel processing
        args_list = [(f, temp_dir, enable_quakeml) for f in input_files]
        
        # Process files in parallel
        successful_dbs = []
        failed_files = []
        
        print(f"Importing {len(input_files)} files in parallel (max_workers={max_workers})...")
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_file = {
                executor.submit(import_file_to_temp_db, args): args[0] 
                for args in args_list
            }
            
            # Process completed tasks with progress bar
            with tqdm(total=len(input_files), desc="Importing files", unit="file") as pbar:
                for future in as_completed(future_to_file):
                    input_file, temp_db_path, success, error_msg = future.result()
                    
                    if success:
                        successful_dbs.append(temp_db_path)
                        pbar.set_postfix({"success": len(successful_dbs), "failed": len(failed_files)})
                    else:
                        failed_files.append((input_file, error_msg))
                        pbar.set_postfix({"success": len(successful_dbs), "failed": len(failed_files)})
                    
                    pbar.update(1)
        
        print(f"\nImport completed: {len(successful_dbs)} successful, {len(failed_files)} failed")
        
        if failed_files:
            print("\nFailed files:")
            for file, error in failed_files:
                print(f"  - {file}: {error}")
        
        if not successful_dbs:
            raise RuntimeError("No files were successfully imported")
        
        # Merge all temporary databases
        print(f"\nMerging {len(successful_dbs)} databases...")
        merge_databases(successful_dbs, output_db, enable_quakeml)
        
        print(f"\n✓ Final database created: {output_db}")
        
    finally:
        # Clean up temporary directory
        if not keep_temp:
            logger.info(f"Cleaning up temporary directory: {temp_dir}")
            try:
                shutil.rmtree(temp_dir)
            except Exception as e:
                logger.warning(f"Failed to remove temporary directory: {e}")
        else:
            print(f"\nTemporary databases kept in: {temp_dir}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Import QuakeML files in parallel using temporary databases"
    )
    parser.add_argument(
        "-i", "--input",
        nargs="+",
        required=True,
        help="Input QuakeML files (supports wildcards)"
    )
    parser.add_argument(
        "-d", "--database",
        required=True,
        help="Output database path"
    )
    parser.add_argument(
        "-q", "--enable-quakeml",
        action="store_true",
        help="Store compressed QuakeML data"
    )
    parser.add_argument(
        "-w", "--workers",
        type=int,
        default=None,
        help="Maximum number of parallel workers (default: CPU count)"
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep temporary databases after merge (for debugging)"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Check if output database already exists
    if os.path.exists(args.database):
        print(f"Error: Output database '{args.database}' already exists.", file=sys.stderr)
        print("Please remove it or choose a different output path.", file=sys.stderr)
        sys.exit(1)
    
    try:
        parallel_import(
            input_files=args.input,
            output_db=args.database,
            enable_quakeml=args.enable_quakeml,
            max_workers=args.workers,
            keep_temp=args.keep_temp
        )
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
