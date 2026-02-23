#!/usr/bin/env python
"""
Parallel QuakeML import script.
Imports QuakeML files in parallel into separate temporary databases,
then merges them into a single final database.
"""
import argparse
import logging
import os
import shutil
import sqlite3
import sys
import tempfile
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


def merge_databases(temp_db_paths, final_db_path, enable_quakeml=False):
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
