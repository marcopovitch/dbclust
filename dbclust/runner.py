#!/usr/bin/env python
"""DBClust unified runner with pluggable executors.

This module provides a unified entry point for running DBClust with
different parallel execution backends (Parsl, Ray, Dask).

The executor is selected based on configuration:
- parallel.executor: parsl_thread (default), parsl_hte, ray, dask
- slurm.enabled: true -> uses parsl_slurm

Usage:
    python -m dbclust.runner -c config.yml
    python -m dbclust.runner -c config.yml -l debug
"""

import argparse
import logging
import sqlite3
import sys
import warnings

from dbclust.config import DBClustConfig
from dbclust.core import dbclust
from dbclust.executors import get_executor
from dbclust.inject_spatialite import load_spatialite
from dbclust.inject_spatialite import refresh_event_coordinates_view

warnings.filterwarnings("ignore", category=UserWarning)

# Default logger
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger("dbclust")


def run_sequential(cfg: DBClustConfig) -> list:
    """Run DBClust sequentially for single-worker mode.

    Args:
        cfg: DBClust configuration object.

    Returns:
        List of results from all time partitions.
    """
    logger.info("Running in sequential mode (n_workers=1)")
    results = []
    for idx, (start, end) in enumerate(cfg.parallel.time_partitions):
        logger.info(f"Processing partition {idx}: {start} -- {end}")
        results.append(dbclust(cfg=cfg, df=None, job_index=idx))
    return results


def run_parallel(cfg: DBClustConfig) -> list:
    """Run DBClust in parallel using the configured executor.

    Args:
        cfg: DBClust configuration object.

    Returns:
        List of results from all time partitions.
    """
    executor = get_executor(cfg)
    logger.info(f"Using executor: {executor.name}")
    return executor.run()


def finalize_sqlite(cfg: DBClustConfig) -> None:
    """Finalize SQLite database after processing.

    Updates the event coordinates view in the SQLite database.

    Args:
        cfg: DBClust configuration object.
    """
    if not cfg.catalog.enable_sqlite:
        return

    try:
        conn = sqlite3.connect(cfg.catalog.sqlite_db_fullpath)
        load_spatialite(conn)
        logger.info("Connected to SQLite database to update view.")
        refresh_event_coordinates_view(conn)
        conn.close()
    except Exception as e:
        logger.error(f"Failed to finalize SQLite database: {e}")


def main():
    """Main entry point for DBClust runner."""
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    logger = logging.getLogger("dbclust")
    logger.setLevel(logging.INFO)

    parser = argparse.ArgumentParser(
        description="DBClust - Seismic event detection and localization"
    )
    parser.add_argument(
        "-c",
        "--conf",
        default=None,
        dest="configfile",
        help="YAML configuration file.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-p",
        "--profile",
        default=None,
        dest="velocity_profile_name",
        help="Velocity profile name to use.",
        type=str,
    )
    parser.add_argument(
        "-l",
        "--loglevel",
        default="INFO",
        dest="loglevel",
        help="Log level (debug, warning, info, error).",
        type=str,
    )

    args = parser.parse_args()

    # Set log level
    numeric_level = getattr(logging, args.loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        logger.error(f"Invalid loglevel '{args.loglevel.upper()}'!")
        logger.error("loglevel should be: debug, warning, info, error.")
        sys.exit(255)
    logger.setLevel(numeric_level)

    # Load configuration
    cfg = DBClustConfig(args.configfile)
    cfg.show()

    # Run processing
    if cfg.parallel.n_workers == 1:
        results = run_sequential(cfg)
    else:
        results = run_parallel(cfg)

    # Finalize SQLite database
    finalize_sqlite(cfg)

    logger.info(f"Processing complete. {len(results)} partitions processed.")

    # Flush output
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(0)


if __name__ == "__main__":
    main()
