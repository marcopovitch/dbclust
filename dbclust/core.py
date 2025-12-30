#!/usr/bin/env python
"""DBClust core processing logic - shared across all executors.

This module contains the main DBClust processing functions that are
independent of the parallel execution backend (Ray, Parsl, Dask).
"""

import gc
import logging
import math
import os
import shutil
import tempfile
import time
from dataclasses import asdict
from typing import List, Optional

import pandas as pd
import pyproj

from dbclust.clusterize import Clusterize
from dbclust.clusterize import feed_picks_event_ids
from dbclust.clusterize import feed_picks_probabilities
from dbclust.clusterize import get_picks_from_event
from dbclust.clusterize import merge_cluster_with_common_phases
from dbclust.config import DBClustConfig
from dbclust.db import duckdb_init
from dbclust.dbclust2pyocto import adjust_associator_tolerance
from dbclust.inject_spatialite import import_catalog_object_to_sqlite_from_file
from dbclust.localization import NllLoc
from dbclust.localization import show_event
from dbclust.phase import import_phases
from dbclust.preprocessing_picks import (
    safe_deduplicate_picks_by_time as deduplicate_picks_by_time,
)
from dbclust.quakeml import deduplicate_picks_and_make_readable_ids
from dbclust.quakeml import feed_distance_from_preloc_to_pref_origin
from dbclust.rename import rename_waveform_id

logger = logging.getLogger("dbclust")

# CSV fieldnames for progress tracking
CSV_FIELDNAMES = [
    "task_index",
    "duration_sec",
    "peak_memory_mb",
    "completed_count",
    "total_tasks",
    "progress_pct",
    "completion_time",
    "time_partition_start",
    "time_partition_end",
]


class MyTemporaryDirectory:
    """Context manager for temporary directories with optional cleanup."""

    def __init__(self, delete: bool = True, dir: Optional[str] = None):
        self.delete = delete
        self.dir = dir
        self.tmp_dir = None

    def __enter__(self) -> str:
        self.tmp_dir = tempfile.mkdtemp(dir=self.dir)
        return self.tmp_dir

    def __exit__(self, exc_type, exc_value, traceback):
        if self.delete and self.tmp_dir:
            shutil.rmtree(self.tmp_dir)
        elif self.tmp_dir:
            logger.info(f"Warning undeleted directory: {self.tmp_dir}")


def unload_picks_list(df1: pd.DataFrame, picks: List) -> pd.DataFrame:
    """Remove picks from DataFrame that are in the picks list.

    Format picks coming from events like the ones used as input for dbclust.

    Args:
        df1: DataFrame with picks to filter.
        picks: List of picks to remove.

    Returns:
        Filtered DataFrame.
    """
    df2 = pd.DataFrame(picks, columns=["station_id", "phase_type", "phase_time"])
    df2["station_id"] = df2["station_id"].map(lambda x: ".".join(x.split(".")[:2]))
    df2["phase_time"] = pd.to_datetime(
        df2["phase_time"].map(lambda x: str(x)), utc=True
    )
    df2["unload"] = True
    # remove duplicate pick as they came from same event but from multiple origins
    df2.drop_duplicates(inplace=True)

    results = pd.merge(
        df1, df2, how="left", on=["station_id", "phase_type", "phase_time"]
    )
    keep = results[results["unload"] != True]
    keep = keep.drop(columns=["unload"])
    return keep


def get_locator_from_config(cfg: DBClustConfig, log_level: int = logging.INFO) -> NllLoc:
    """Create a NllLoc instance from configuration.

    Args:
        cfg: DBClust configuration.
        log_level: Logging level for the locator.

    Returns:
        Configured NllLoc instance.
    """
    locator = NllLoc(
        cfg.nll.nlloc_bin,
        cfg.nll.scat2latlon_bin,
        cfg.nll.time_path,
        #
        nll_verbose=cfg.nll.verbose,
        nll_default_template=cfg.nll.default_template_file,
        nll_template=cfg.nll.template_path,
        loc_method=cfg.nll.loc_method,
        tmpdir=cfg.file.tmp_path,
        #
        double_pass=cfg.relocation.double_pass,
        P_time_residual_threshold=cfg.relocation.P_time_residual_threshold,
        S_time_residual_threshold=cfg.relocation.S_time_residual_threshold,
        gap_dist_max_km=cfg.relocation.gap_dist_max_km,
        dist_km_cutoff=cfg.relocation.dist_km_cutoff,
        use_deactivated_arrivals=cfg.relocation.use_deactivated_arrivals,
        keep_manual_picks=cfg.relocation.keep_manual_picks,
        nll_min_phase=cfg.nll.min_phase,
        min_station_with_P_and_S=cfg.cluster.min_station_with_P_and_S,
        min_station_score=cfg.cluster.min_station_score,
        quakeml_settings=asdict(cfg.quakeml),
        keep_scat=cfg.nll.enable_scatter,
        #
        zones=cfg.zones,
        force_zone_name=None,
        min_score_threshold_pick_zone=cfg.relocation.min_score_threshold_pick_zone,
        use_pick_zone=cfg.relocation.use_pick_zone,
        enable_relabel_pick_zone=cfg.relocation.enable_relabel_pick_zone,
        enable_cleanup_pick_zone=cfg.relocation.enable_cleanup_pick_zone,
        #
        keep_not_existing_event=cfg.catalog.keep_not_existing_event,
        log_level=log_level,
    )
    return locator


def get_clusterize_from_config(
    cfg: DBClustConfig, phases=None, log_level: int = logging.INFO
) -> Clusterize:
    """Create a Clusterize instance from configuration.

    Args:
        cfg: DBClust configuration.
        phases: Optional list of phases to cluster.
        log_level: Logging level for the clusterer.

    Returns:
        Configured Clusterize instance.
    """
    myclust = Clusterize(
        phases=phases,
        min_cluster_size=cfg.cluster.min_cluster_size,
        average_velocity=cfg.cluster.average_velocity,
        min_station_count=cfg.cluster.min_station_count,
        min_station_with_P_and_S=cfg.cluster.min_station_with_P_and_S,
        min_station_score=cfg.cluster.min_station_score,
        max_search_dist=cfg.cluster.max_search_dist,
        P_uncertainty=cfg.pick.P_uncertainty,
        S_uncertainty=cfg.pick.S_uncertainty,
        tt_matrix_fname=cfg.cluster.pre_computed_tt_matrix_file,
        tt_matrix_save=cfg.cluster.tt_matrix_save,
        zones=cfg.zones,
        log_level=log_level,
    )
    return myclust


def dbclust(
    cfg: DBClustConfig,
    df: Optional[pd.DataFrame] = None,
    job_index: Optional[int] = None,
) -> bool:
    """Detect and localize seismic events given picks.

    This is the main processing function for DBClust. It handles clustering
    of seismic picks and localization using NonLinLoc.

    Args:
        cfg: DBClust configuration containing all parameters.
        df: Optional DataFrame with picks. If None, reads from configured files.
        job_index: Job index for parallel execution. None for sequential mode.

    Returns:
        True if successful, False otherwise.

    The function processes data in parallel using job_index to determine the
    time window. It handles:
    - Clustering with HDBSCAN
    - Localization with NonLinLoc
    - Overlapping time periods and cluster merging
    - Saving results to QuakeML and SQLite formats
    """
    logger.info("")
    logger.info("")
    logger.info(
        f"============== DBClust started (job index: {job_index}) =============="
    )

    if df is None:
        df = pd.DataFrame()

    # Time blocks
    if job_index is not None:
        # parallel mode is enabled
        parallel_mode = True

        if job_index < 0 or job_index >= len(cfg.parallel.time_partitions):
            logger.error(f"Invalid job_index {job_index}, out of range.")
            return False

        if job_index == len(cfg.parallel.time_partitions) - 1:
            last_job = True
        else:
            last_job = False
    else:
        # sequential mode
        parallel_mode = False
        last_job = True
        job_index = 0

    start, stop = cfg.parallel.time_partitions[job_index]

    msg = "started."
    logger.info(f"{msg} Job index: {job_index}, Start: {start}, Stop: {stop}")

    if df is None or df.empty:
        # Uses duckdb
        con = duckdb_init(cfg.pick.filenames, cfg.pick.type)
    else:
        # Uses the pandas Dataframe given as function argument.
        con = None

    window = pd.Timedelta(minutes=cfg.time.time_window)
    overlap_timedelta = pd.Timedelta(cfg.time.overlap_window, "s")

    total_duration = stop - start
    nb_periods = math.floor(total_duration / window)
    remainder = total_duration % window
    adjusted_stop = start + nb_periods * window
    if remainder > pd.Timedelta(0):
        adjusted_stop = start + (nb_periods + 1) * window

    time_periods = list(
        pd.date_range(start, adjusted_stop, freq=window, inclusive="left")
    )
    time_divisions = [(s, s + window) for s in time_periods]

    logger.info(f"[{job_index}] has {len(time_divisions)} time divisions.")
    logger.info(f"{time_divisions}")

    # Instantiate a new tool (but empty) to get clusters
    previous_myclust = get_clusterize_from_config(cfg, phases=None)

    # get a locator
    locator = get_locator_from_config(cfg)

    # keep track of each time division processed
    last_saved_event_count = 0
    picks_to_remove = []

    # start time looping
    for i, (begin, end) in enumerate(time_divisions, start=1):
        # Check if this is the last time divisions
        if job_index is not None and i == len(time_divisions):
            last_partition_job = True
        else:
            last_partition_job = False

        # add the time overlap only if it is not the last round
        if end >= cfg.pick.end:
            end = pd.to_datetime(cfg.pick.end)
            short_window = True

            # complementary check
            if end < begin:
                end = begin

        else:
            end += overlap_timedelta
            short_window = False

        logger.info("")
        logger.info("")
        logger.info(
            f"============== job index:[{job_index}] Time window extraction with overlap {overlap_timedelta}: #{i}/{len(time_divisions)} picks from {begin} to {end}."
        )

        # Extract picks on this time period
        if con:
            begin_year = begin.year
            begin_month = begin.month
            end_year = end.year
            end_month = end.month

            if cfg.pick.type == "parquet":
                # benefit from parquet partitioning by year and month
                rqt = f"""
                    SELECT * FROM PICKS
                    WHERE
                    (year > {begin_year} OR (year = {begin_year} AND month >= {begin_month}))
                    AND
                    (year < {end_year} OR (year = {end_year} AND month <= {end_month}))
                    AND
                    phase_time BETWEEN '{begin}' AND '{end}'
                """
            else:
                # csv
                rqt = f"""
                    SELECT * FROM PICKS
                    WHERE phase_time BETWEEN '{begin}' AND '{end}'
                """

            # Time measure of the query
            start_time = time.time()
            df_subset = con.sql(rqt).fetchdf()
            elapsed_time = time.time() - start_time
            logger.info(f"Query time: {elapsed_time:.2f} s")

            if df_subset["phase_time"].dt.tz is None:
                df_subset["phase_time"] = df_subset["phase_time"].dt.tz_localize("UTC")
            else:
                df_subset["phase_time"] = df_subset["phase_time"].dt.tz_convert("UTC")

        else:
            df_subset = df[(df["phase_time"] >= begin) & (df["phase_time"] < end)]

        if df_subset.empty and previous_myclust.phases_count() == 0:
            logger.info(f"[{job_index}] Skipping clustering {len(df_subset)} phases.")
            continue

        # remove blacklisted stations
        if cfg.station.blacklist:
            for b in cfg.station.blacklist:
                df_subset = df_subset[
                    ~df_subset["station_id"].str.contains(b, regex=True)
                ]

        # starting pick preprocessing to get rid of too close picks
        logger.info(
            f"[{job_index}] Starting pick preprocessing with {len(df_subset)} phases."
        )
        df_subset = deduplicate_picks_by_time(
            df_subset,
            cfg.pick.P_proximity_threshold,
            cfg.pick.S_proximity_threshold,
        )
        logger.info(
            f"[{job_index}] End pick preprocessing with {len(df_subset)} phases."
        )

        # Remove picks previously associated with events
        logger.info(f"[{job_index}] Starting clustering with {len(df_subset)} phases.")
        logger.info(f"Before unload_picks_list() len(df_subset) = {len(df_subset)}")
        if len(picks_to_remove):
            logger.info(
                f"[{job_index}] before unload picks: pick length is {len(df_subset)}"
            )
            df_subset = unload_picks_list(df_subset, picks_to_remove)
            logger.info(
                f"[{job_index}] after unload picks: pick length is {len(df_subset)}"
            )
            picks_to_remove = []
        logger.info(f"After unload_picks_list() len(df_subset) = {len(df_subset)}")

        # remove picks based on station frequency threshold
        if cfg.station.frequency_threshold:
            total_duration_in_minutes = (end - begin).total_seconds() / 60
            if total_duration_in_minutes == 0:
                total_duration_in_minutes = 1.0  # Avoid division by zero
            grouped = df_subset.groupby("station_id")
            station_counts = grouped.size()
            frequencies = station_counts / total_duration_in_minutes
            station_ids_to_keep = frequencies[
                frequencies < cfg.station.frequency_threshold
            ].index
            df_subset = df_subset[df_subset["station_id"].isin(station_ids_to_keep)]

        # Rename station_id.channel by user request
        df_subset = rename_waveform_id(df_subset, cfg.station.rename)

        # Import picks and get coordinates
        phases = import_phases(
            df_subset,
            cfg.pick.P_proba_threshold,
            cfg.pick.S_proba_threshold,
            cfg.pick.P_uncertainty,
            cfg.pick.S_uncertainty,
            cfg.station.info_sta,
            cfg.station.fallback_df,
        )
        if logger.level == logging.DEBUG:
            for p in phases:
                p.show_all()

        # clean up
        del df_subset
        gc.collect()

        if logger.level == logging.DEBUG:
            logger.info("previous_myclust:")
            previous_myclust.show_clusters()

        # Instantiate a new tool to get clusters
        myclust = get_clusterize_from_config(cfg, phases=phases)
        del phases

        if logger.level == logging.DEBUG:
            logger.info("myclust:")
            myclust.show_clusters()

        # check if some clusters share phases with previous round
        logger.info("Check clusters related to the same event (overlapped zone).")

        previous_myclust, myclust, nb_cluster_removed = (
            merge_cluster_with_common_phases(
                previous_myclust, myclust, cfg.cluster.min_picks_common
            )
        )

        if last_partition_job:
            logger.info(
                f"==> Last job in the time partition, merging all remaining clusters."
            )
            previous_myclust.merge(myclust)

        if cfg.pyocto.enable and cfg.pyocto.current_model:
            try:
                result = adjust_associator_tolerance(
                    previous_myclust,
                    cfg,
                    tolerance_steps={1: 1, 0.5: 0.1, 0: 0.05},
                    min_tolerance=0.1,
                    log_level=logger.level,
                )
            except pyproj.exceptions.CRSError as e:
                logger.error(f"Aborting process adjust_associator_tolerance().")
                if begin == end:
                    logger.info("Cleaning previous_myclust.")
                    previous_myclust = get_clusterize_from_config(cfg, phases=None)
                continue

            if result is None:
                logger.error("Failed to process with any pick_match_tolerance.")
                logger.debug(f"begin={begin}, end={end}")
                if begin == end:
                    logger.info("Cleaning previous_myclust.")
                    previous_myclust = get_clusterize_from_config(cfg, phases=None)
                continue
            else:
                previous_myclust = result

        # Process previous_myclust and wait next round to process myclust
        with MyTemporaryDirectory(
            dir=cfg.file.obs_path, delete=cfg.file.automatic_cleanup_tmp
        ) as TMP_OBS_PATH:
            my_obs_path = os.path.join(TMP_OBS_PATH, f"{i}")
            nll_picks = previous_myclust.generate_nllobs(my_obs_path)

            logger.info(f"Starting localization using {locator.loc_method}.")
            with MyTemporaryDirectory(
                dir=cfg.file.tmp_path,
                delete=cfg.file.automatic_cleanup_tmp,
            ) as tmpdir_automaticaly_cleaned:
                locator.tmpdir = tmpdir_automaticaly_cleaned

                clustcat = locator.get_localisations_from_nllobs_dir(
                    my_obs_path, picks=nll_picks, append=True
                )

                if cfg.nll.enable_scatter:
                    logger.warning("FIXME: scatter file not yet handled !")

        if len(clustcat) > 0:
            for event in sorted(
                clustcat.events, key=lambda e: e.preferred_origin().time
            ):
                origin = event.preferred_origin()
                picks = get_picks_from_event(event, origin, None)
                if not picks:
                    logger.warning(f"Event {event.resource_id.id} has no picks, skipping")
                    continue
                _, _, first_pick_time = picks[0]
                _, _, last_pick_time = picks[-1]

                # check if the event is in the overlapped zone
                event_in_overlapped_zone = False
                if not short_window:
                    next_begin = end - overlap_timedelta
                    if first_pick_time > next_begin:
                        event_in_overlapped_zone = True
                else:
                    next_begin = end

                logger.info(
                    f"Event first pick is: {first_pick_time}, last pick is: {last_pick_time}, "
                    f"overlapped zone starts: {begin}, next overlapped zone starts: {next_begin}, "
                    f"short_window={short_window}, last_job_partition={last_partition_job}, last_job={last_job}, "
                    f"pick_in_overlapped_zone={event_in_overlapped_zone}"
                )

                if not last_job and event_in_overlapped_zone:
                    show_event(event, "***D")
                    logger.info(
                        f"Found event in overlapped zone to be (D)eleted ({event.resource_id.id})"
                    )
                    locator.catalog.events.remove(event)
                    locator.nb_events = len(locator.catalog)
                    clustcat = locator.catalog
                elif (
                    event.event_type != "not existing"
                    and not last_job
                    and first_pick_time < next_begin
                    and last_pick_time >= next_begin
                ):
                    show_event(event, "***P")
                    logger.info(
                        f"Found event between normal and overlapped zone where picks must be (P)runed ({event.resource_id.id})"
                    )
                    picks_to_remove = []
                    for origin in event.origins:
                        picks_to_remove += get_picks_from_event(
                            event, origin, next_begin
                        )
                else:
                    show_event(event, "****")
        else:
            logger.info("No event found in theses clusters.")

        # Write picks probabilities and event_ids
        feed_picks_probabilities(clustcat, previous_myclust.clusters)
        feed_picks_event_ids(clustcat, previous_myclust.clusters)

        # Write distance from preferred origin and prelocalization
        clustcat = feed_distance_from_preloc_to_pref_origin(clustcat)

        # Write partial qml file and clean catalog from memory
        if last_saved_event_count > cfg.catalog.event_flush_count:
            save_catalog(locator.catalog, cfg, job_index, part=i)
            locator.catalog.clear()
            gc.collect()
            last_saved_event_count = 0
        else:
            last_saved_event_count += len(locator.catalog)

        # prepare next round
        previous_myclust = myclust

    # Save remaining events
    save_catalog(locator.catalog, cfg, job_index, part=i + 1, finalize=True)
    logger.info(f"Finalizing job index: {job_index}")
    return True


def save_catalog(
    catalog,
    cfg: DBClustConfig,
    job_index: Optional[int] = None,
    part: Optional[int] = None,
    finalize: bool = False,
) -> None:
    """Save the catalog to QuakeML and/or SQLite file formats.

    Args:
        catalog: The seismic event catalog to be saved.
        cfg: Configuration containing save paths and options.
        job_index: Job index for parallel execution, used for naming files.
        part: Part number for splitting the catalog, used for naming files.
        finalize: If True, the catalog is saved as a final output.
    """
    # Determine file name based on job index and part
    if job_index is not None:
        qml_filename = os.path.join(
            cfg.catalog.qml_path,
            f"{cfg.catalog.qml_base_filename}-{job_index}-{part}.qml",
        )
    else:
        qml_filename = os.path.join(
            cfg.catalog.qml_path, f"{cfg.catalog.qml_base_filename}-{part}.qml"
        )

    logger.info(
        f"Saving catalog: finalizing={finalize}, part={part}, job_index={job_index}, nb_events={len(catalog)}"
    )

    # Deduplicate picks and create readable IDs
    catalog = deduplicate_picks_and_make_readable_ids(
        catalog,
        prefix=cfg.quakeml.event_prefix,
        smi_base=cfg.quakeml.smi_base,
    )

    # Save to QuakeML file if enabled
    if cfg.catalog.enable_quakeml_file and len(catalog) > 0:
        logger.info(f"Writing {len(catalog)} events to {qml_filename}")
        catalog.write(qml_filename, format="QUAKEML")

    # Save to SQLite database if enabled
    if cfg.catalog.enable_sqlite and len(catalog) > 0:
        logger.info(
            f"Writing {len(catalog)} events to {cfg.catalog.sqlite_db_fullpath}"
        )
        try:
            import_catalog_object_to_sqlite_from_file(
                cfg.catalog.sqlite_db_fullpath,
                catalog,
                enable_quakeml=True,
                disable_tqdm=True,
                retries=15,
                delay=2,
                backoff="exponential",
            )
        except Exception as e:
            logger.error(f"Error writing catalog to SQLite: {e}")
