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
import glob
import logging
import os
import signal
import sys
import warnings

# Prevent thread explosion with numerical libraries (must be set before other imports)
# Restricted to macOS: on Linux/HPC these are intentionally left unset so that
# numerical libraries can use all available cores.
if sys.platform == "darwin_disable":
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

from dbclust.config import DBClustConfig
from dbclust.core import dbclust
from dbclust.executors import get_executor
from dbclust.executors.base import config_fingerprint
from dbclust.parallel_import import merge_databases

warnings.filterwarnings("ignore", category=UserWarning)

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

    Installs SIGINT/SIGTERM handlers so that Ctrl+C cleanly shuts down the
    executor (Ray / Dask / Parsl) and then kills the entire process group to
    ensure no orphan worker processes survive.

    Args:
        cfg: DBClust configuration object.

    Returns:
        List of results from all time partitions.
    """
    executor = get_executor(cfg)
    logger.info(f"Using executor: {executor.name}")

    # Check if all partitions are already completed
    completed = _load_completed_indices(cfg)
    if completed:
        logger.info(f"Found {len(completed)} completed partitions from previous run")

    _shutdown_called = [False]
    _old_sigint = [signal.getsignal(signal.SIGINT)]
    _old_sigterm = [signal.getsignal(signal.SIGTERM)]

    def _install_handlers():
        def _shutdown(signum, _frame):
            if _shutdown_called[0]:
                return
            _shutdown_called[0] = True
            sig_name = signal.Signals(signum).name
            logger.warning(f"Received {sig_name} — shutting down executor...")
            signal.signal(signal.SIGINT, signal.SIG_DFL)
            signal.signal(signal.SIGTERM, signal.SIG_DFL)
            try:
                executor.cleanup()
            except Exception:
                pass
            os.kill(os.getpid(), signum)

        _old_sigint[0] = signal.signal(signal.SIGINT, _shutdown)
        _old_sigterm[0] = signal.signal(signal.SIGTERM, _shutdown)

    # Install signal handlers only after executor.initialize() completes,
    # so that Parsl/Ray process launches don't accidentally trigger shutdown.
    executor.on_initialized = _install_handlers
    try:
        logger.info("Calling executor.run()...")
        results = executor.run()
        logger.info(f"executor.run() returned {len(results)} results")
        return results
    finally:
        if not _shutdown_called[0]:
            signal.signal(signal.SIGINT, _old_sigint[0])
            signal.signal(signal.SIGTERM, _old_sigterm[0])


def _load_completed_indices(cfg: DBClustConfig):
    """Load the set of successfully completed job indices from the checkpoint file.

    The checkpoint also stores a fingerprint of the temporal config; if it doesn't
    match the current config, the checkpoint is stale and is ignored so that all
    tasks are reprocessed instead of merging results from incompatible runs.
    """
    import json
    profiles_path = cfg.parallel.task_profiles_path or os.path.join(
        cfg.catalog.qml_path, "task_profiles.csv"
    )
    checkpoint_path = profiles_path.replace(".csv", ".completed.json")
    if not os.path.exists(checkpoint_path):
        return None
    try:
        with open(checkpoint_path) as f:
            data = json.load(f)
        if isinstance(data, list):
            logger.info(
                "Obsolete checkpoint format (list), ignored — all tasks will be reprocessed."
            )
            return None
        if data.get("fingerprint") != config_fingerprint(cfg):
            logger.info(
                "Temporal parameters changed since the last run, "
                "checkpoint invalidated — all tasks will be reprocessed."
            )
            return None
        return set(data.get("done", []))
    except Exception:
        return None


def finalize_sqlite(cfg: DBClustConfig, completed_job_indices=None) -> None:
    """Finalize SQLite database after processing.

    Merges all per-worker temporary databases into the final database,
    then refreshes the event coordinates view.

    Only temp DBs whose job_index appears in completed_job_indices are merged.
    This prevents partial results from failed/killed workers from polluting
    the final database.

    Args:
        cfg: DBClust configuration object.
        completed_job_indices: Set of job indices that completed successfully.
            If None, all temp DBs are merged (legacy behaviour).
    """
    if not cfg.catalog.enable_sqlite:
        return

    temp_dir = cfg.catalog.temp_db_dir or cfg.catalog.sqlite_db_path
    all_temp_db_paths = sorted(glob.glob(os.path.join(temp_dir, "tmp_worker_*.db")))

    if not all_temp_db_paths:
        logger.warning("No temp DBs found to merge, skipping SQLite finalization.")
        return

    if completed_job_indices is not None:
        temp_db_paths = []
        skipped = []
        for p in all_temp_db_paths:
            basename = os.path.basename(p)  # tmp_worker_42.db
            try:
                idx = int(basename.replace("tmp_worker_", "").replace(".db", ""))
            except ValueError:
                skipped.append(p)
                continue
            if idx in completed_job_indices:
                temp_db_paths.append(p)
            else:
                skipped.append(p)
        if skipped:
            logger.warning(
                f"Skipping {len(skipped)} temp DB(s) from failed/incomplete jobs: "
                + ", ".join(os.path.basename(p) for p in skipped)
            )
    else:
        temp_db_paths = all_temp_db_paths

    if not temp_db_paths:
        logger.warning("No completed temp DBs to merge, skipping SQLite finalization.")
        return

    logger.info(
        f"Merging {len(temp_db_paths)} temp DBs into {cfg.catalog.sqlite_db_fullpath}"
    )
    try:
        merge_databases(
            temp_db_paths,
            cfg.catalog.sqlite_db_fullpath,
            enable_quakeml=True,
            overlap_window_s=float(cfg.time.overlap_window),
        )
    except Exception as e:
        logger.error(f"Failed to merge temp databases: {e}")
        return

    if not cfg.catalog.keep_temp_db_after_merge:
        for p in temp_db_paths:
            for suffix in ("", "-shm", "-wal"):
                path = p + suffix
                if not os.path.exists(path):
                    continue
                try:
                    os.remove(path)
                    logger.debug(f"Removed temp DB file: {path}")
                except OSError as e:
                    logger.warning(f"Could not remove temp DB file {path}: {e}")
    else:
        logger.info(f"Keeping temp DBs in {temp_dir} (keep_temp_db_after_merge=True)")


def main():
    """Main entry point for DBClust runner."""
    app_logger = logging.getLogger("dbclust")
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
    parser.add_argument(
        "--pick-start",
        default=None,
        dest="pick_start",
        help="Override pick.start from the config file (ISO 8601, e.g. 2026-07-30T00:00:00).",
        type=str,
    )
    parser.add_argument(
        "--pick-end",
        default=None,
        dest="pick_end",
        help="Override pick.end from the config file (ISO 8601, e.g. 2026-07-31T00:00:00).",
        type=str,
    )

    args = parser.parse_args()

    # Set log level
    numeric_level = getattr(logging, args.loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        app_logger.error(f"Invalid loglevel '{args.loglevel.upper()}'!")
        app_logger.error("loglevel should be: debug, warning, info, error.")
        sys.exit(255)

    logging.basicConfig(stream=sys.stdout, level=numeric_level, force=True)
    # Set level on the dbclust parent logger so all child loggers inherit it
    logging.getLogger("dbclust").setLevel(numeric_level)
    # Suppress pyocto's routine "Did not associate more events" WARNING — it fires
    # on every cluster that yields 0 events and is redundant with our own log.
    if numeric_level > logging.DEBUG:
        logging.getLogger("pyocto").setLevel(logging.ERROR)

    # Load configuration
    cfg = DBClustConfig(
        args.configfile,
        pick_start_override=args.pick_start,
        pick_end_override=args.pick_end,
    )
    cfg.log_level = numeric_level
    cfg.show()

    # Run processing
    slurm_enabled = hasattr(cfg, "slurm") and cfg.slurm and cfg.slurm.enabled
    if not slurm_enabled and cfg.parallel.n_workers == 1:
        results = run_sequential(cfg)
    else:
        results = run_parallel(cfg)

    # Finalize SQLite database — only merge DBs from completed jobs
    completed_indices = _load_completed_indices(cfg)
    finalize_sqlite(cfg, completed_job_indices=completed_indices)

    app_logger.info(f"Processing complete. {len(results)} partitions processed.")

    # Flush output and exit cleanly
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(0)


if __name__ == "__main__":
    main()
