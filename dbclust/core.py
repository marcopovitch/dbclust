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
import pyproj.exceptions

from dbclust.clusterize import Clusterize
from dbclust.clusterize import feed_picks_event_ids
from dbclust.clusterize import feed_picks_probabilities
from dbclust.clusterize import get_picks_from_event
from dbclust.clusterize import merge_cluster_with_common_phases
from dbclust.config import DBClustConfig
from dbclust.db import duckdb_init
from dbclust.dbclust2pyocto import adjust_associator_tolerance_two_phase as adjust_associator_tolerance
from dbclust.inject_spatialite import (
    create_schema,
    import_catalog_to_sqlite,
)
from dbclust.localization import NllLoc
from dbclust.localization import format_event
from dbclust.phase import import_phases
from dbclust.preprocessing_picks import (
    safe_deduplicate_picks_by_time as deduplicate_picks_by_time,
)
from dbclust.quakeml import deduplicate_picks_and_make_readable_ids
from dbclust.quakeml import feed_distance_from_preloc_to_pref_origin
from dbclust.rename import rename_waveform_id

# uses hierarchical name for selective level control
logger = logging.getLogger("dbclust.core")

# CSV fieldnames for progress tracking
CSV_FIELDNAMES = [
    "task_index",
    "start_time",
    "completion_time",
    "duration_sec",
    "peak_memory_mb",
    "completed_count",
    "total_tasks",
    "progress_pct",
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
    keep = results[results["unload"].isna()]
    keep = keep.drop(columns=["unload"])
    return keep


def inject_picks_into_df(
    df_subset: pd.DataFrame,
    df_inject: pd.DataFrame,
    label: str,
    job_index: int,
    partition_index: int,
) -> pd.DataFrame:
    """Inject additional picks into df_subset, normalising dtypes and deduplicating.

    Used both for overlap-deferred picks (***D events) and cross-partition HPC picks.
    Deduplication is done on (station_id, phase_type, phase_time); a row already
    present in df_subset with an event_id wins over an injected row without one.
    """
    if df_inject.empty:
        return df_subset

    df_inject = df_inject.copy()
    if df_inject["phase_time"].dt.tz is None:
        df_inject["phase_time"] = df_inject["phase_time"].dt.tz_localize("UTC")
    else:
        df_inject["phase_time"] = df_inject["phase_time"].dt.tz_convert("UTC")
    for col in df_inject.columns:
        if col in df_subset.columns and col != "phase_time":
            try:
                df_inject[col] = df_inject[col].astype(df_subset[col].dtype)
            except (ValueError, TypeError):
                pass

    logger.info(
        f"[{job_index}] Injecting {len(df_inject)} {label} picks into partition #{partition_index}"
    )
    # Concat with df_inject first so that rows with event_id win over duplicates without
    combined = pd.concat([df_inject, df_subset])
    combined = combined.drop_duplicates(subset=["station_id", "phase_type", "phase_time"], keep="first")
    return combined


def get_cross_partition_picks(
    con, start, overlap_timedelta, global_start
) -> pd.DataFrame:
    """Fetch all picks from the backward overlap zone [start-overlap, start].

    Returns a single DataFrame with all picks (catalogued and DL automatic).
    Injected into df_subset before HDBSCAN so that cross-partition events
    (suppressed by the forward overlap rule of the previous job) are
    reconstructed with their full pick set in this job.
    """
    backward_start = max(start - overlap_timedelta, global_start)
    if backward_start >= start:
        return pd.DataFrame()

    rqt = f"""
        SELECT * FROM PICKS
        WHERE phase_time BETWEEN '{backward_start}' AND '{start}'
        AND phase_type IN ('P', 'Pg', 'Pn', 'S', 'Sg', 'Sn')
    """
    df_all = con.sql(rqt).fetchdf()
    return df_all if not df_all.empty else pd.DataFrame()


def get_locator_from_config(cfg: DBClustConfig) -> NllLoc:
    """Create a NllLoc instance from configuration.

    Args:
        cfg: DBClust configuration.

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
        closest_station_dist_km=cfg.relocation.closest_station_dist_km,
        dist_km_cutoff=cfg.relocation.dist_km_cutoff,
        use_deactivated_arrivals=cfg.relocation.use_deactivated_arrivals,
        keep_manual_picks=cfg.relocation.keep_manual_picks,
        nll_min_phase=cfg.nll.min_phase,
        min_station_with_P_and_S=cfg.cluster.min_station_with_P_and_S,
        min_station_score=cfg.cluster.min_station_score,
        min_ps_ratio=cfg.cluster.min_ps_ratio,
        quakeml_settings=asdict(cfg.quakeml),
        keep_scat=cfg.nll.enable_scatter,
        #
        zones=cfg.zones,
        force_zone_name=None,
        min_score_threshold_pick_zone=cfg.relocation.min_score_threshold_pick_zone,
        use_pick_zone=cfg.relocation.use_pick_zone,
        enable_relabel_pick_zone=cfg.relocation.enable_relabel_pick_zone,
        enable_cleanup_pick_zone=cfg.relocation.enable_cleanup_pick_zone,
        min_dist_relabel_deg=getattr(cfg.relocation, "min_dist_relabel_deg", 0.0),
        min_time_weight=getattr(cfg.relocation, "min_time_weight", None),
        enable_residual_threshold_with_pick_zone=getattr(cfg.relocation, "enable_residual_threshold_with_pick_zone", False),
        pass2_degradation_factor=getattr(cfg.relocation, "pass2_degradation_factor", None),
        pass2_fallback=getattr(cfg.relocation, "pass2_fallback", False),
        #
        keep_not_existing_event=cfg.catalog.keep_not_existing_event,
    )
    return locator


def get_clusterize_from_config(cfg: DBClustConfig, phases=None) -> Clusterize:
    """Create a Clusterize instance from configuration.

    Args:
        cfg: DBClust configuration.
        phases: Optional list of phases to cluster.

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
        min_ps_ratio=cfg.cluster.min_ps_ratio,
        force_keep_catalog_events=cfg.cluster.force_keep_catalog_events,
        max_search_dist=cfg.cluster.max_search_dist,
        P_uncertainty=cfg.pick.P_uncertainty,
        S_uncertainty=cfg.pick.S_uncertainty,
        tt_matrix_fname=cfg.cluster.pre_computed_tt_matrix_file,
        tt_matrix_save=cfg.cluster.tt_matrix_save,
        zones=cfg.zones,
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

    # Remove any leftover temp DB from a previous run at job startup (once per job).
    if cfg.catalog.enable_sqlite:
        temp_dir = cfg.catalog.temp_db_dir or cfg.catalog.sqlite_db_path
        temp_db_path = os.path.join(temp_dir, f"tmp_worker_{job_index}.db")
        for suffix in ("", "-shm", "-wal"):
            path = temp_db_path + suffix
            if os.path.exists(path):
                os.remove(path)
                logger.debug(f"Removed leftover temp DB file: {path}")

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

    # Save the configured pick_match_tolerance so it can be restored before each
    # PyOcto call — previous windows may leave a modified value on the associator.
    if cfg.pyocto.enable and cfg.pyocto.current_model:
        _original_tolerance = cfg.pyocto.current_model.associator.pick_match_tolerance

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
                    AND
                    phase_type IN ('P', 'Pg', 'Pn', 'S', 'Sg', 'Sn')
                """
            else:
                # csv
                rqt = f"""
                    SELECT * FROM PICKS
                    WHERE phase_time BETWEEN '{begin}' AND '{end}'
                    AND phase_type IN ('P', 'Pg', 'Pn', 'S', 'Sg', 'Sn')
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

            # First window of non-first parallel jobs: inject all picks from the
            # backward overlap zone [start-overlap, start] so that cross-partition
            # events (suppressed by the forward overlap rule of the previous job)
            # are reconstructed with their full pick set before HDBSCAN.
            if parallel_mode and job_index > 0 and i == 1:
                global_start = pd.Timestamp(cfg.pick.start).tz_localize(None)
                df_backward = get_cross_partition_picks(
                    con, start, overlap_timedelta, global_start
                )
                df_subset = inject_picks_into_df(df_subset, df_backward, "backward", job_index, i)
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
        elif myclust.n_clusters > 0:
            # Promote all non-merged myclust clusters into previous_myclust so
            # PyOcto Phase 2a can enrich them with the current window's DL picks.
            # Without this, HDBSCAN clusters bypass PyOcto for a full window
            # and miss DL picks that are only available in the current df_subset.
            # The overlap rules (Rule 1 / Rule 3) in the localization section
            # will correctly defer or prune events whose picks straddle next_begin.
            logger.info(
                f"Promoting {myclust.n_clusters} myclust cluster(s) "
                f"into previous_myclust for immediate PyOcto processing."
            )
            # Collect DL picks (no event_id) from promoted clusters + noise so
            # PyOcto Phase 2a can still use them as enrichment candidates.
            # Without this, promoting clusters empties myclust.clusters and
            # Phase 2a loses access to the DL picks from the current window.
            extra_dl_picks = [
                p for c in myclust.clusters for p in c if not p.event_id
            ] + [p for p in myclust.noise if not p.event_id]
            previous_myclust.clusters += myclust.clusters
            previous_myclust.n_clusters = len(previous_myclust.clusters)
            previous_myclust.clusters_stability = (
                list(previous_myclust.clusters_stability) + [1.0] * myclust.n_clusters
            )
            previous_myclust.noise = list(getattr(previous_myclust, 'noise', [])) + extra_dl_picks
            previous_myclust.n_noise = len(previous_myclust.noise)
            myclust.clusters = []
            myclust.n_clusters = 0
            myclust.clusters_stability = []
            myclust.noise = []
            myclust.n_noise = 0

        if cfg.pyocto.enable and cfg.pyocto.current_model:
            # Restore original tolerance before each call so windows don't
            # inherit a degraded value left by the previous window.
            cfg.pyocto.current_model.associator.pick_match_tolerance = _original_tolerance
            try:
                result = adjust_associator_tolerance(
                    previous_myclust,
                    cfg,
                    tolerance_steps={1: 1, 0.5: 0.1, 0: 0.05},
                    min_tolerance=0.1,
                    include_noise_in_aggregation=cfg.cluster.include_noise_in_aggregation,
                    log_level=logger.level,
                )
            except pyproj.exceptions.ProjError as e:
                logger.error(f"Projection error, aborting adjust_associator_tolerance(): {e}")
                if begin == end:
                    logger.info("Cleaning previous_myclust.")
                    previous_myclust = get_clusterize_from_config(cfg, phases=None)
                continue
            except Exception as e:
                import traceback
                logger.error(f"Unexpected error in adjust_associator_tolerance(): {e}")
                logger.error(traceback.format_exc())
                raise

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

                # Rule 1 — Window overlap zone: first_pick falls beyond next_begin.
                # The next window (or next job) will detect this event with more picks.
                # DL picks appear naturally in the next window via the DB query — no
                # deferred injection needed.
                if not (last_partition_job and last_job) and event_in_overlapped_zone:
                    for line in format_event(event, "***D"):
                        logger.info(line)
                    logger.info(
                        f"Found event in overlapped zone to be (D)eleted ({event.resource_id.id})"
                    )
                    locator.catalog.events.remove(event)
                    locator.nb_events = len(locator.catalog)
                    clustcat = locator.catalog

                # Rule 2 — Forward overlap zone (parallel mode, last window of job N):
                # any event starting in [stop-overlap, stop] is always deferred to job N+1,
                # which will reconstruct it with the full pick set via backward injection.
                # No exception for complete events (last_pick < stop): we always defer to
                # avoid duplicates when job N+1 assembles the event from backward picks.
                elif (
                    parallel_mode
                    and not last_job
                    and last_partition_job
                    and first_pick_time >= stop - overlap_timedelta
                ):
                    for line in format_event(event, "***D"):
                        logger.info(line)
                    logger.info(
                        f"Event in forward overlap zone (first_pick={first_pick_time} >= "
                        f"stop-overlap={stop - overlap_timedelta}), deferred to next job "
                        f"({event.resource_id.id})"
                    )
                    locator.catalog.events.remove(event)
                    locator.nb_events = len(locator.catalog)
                    clustcat = locator.catalog

                # Rule 3 — Straddle: first_pick in normal zone but last_pick in overlap zone.
                # Intermediate window: prune picks beyond next_begin and keep the event.
                # Last window of job N (last_partition_job): suppress — job N+1 has full picks.
                elif (
                    event.event_type != "not existing"
                    and not (last_partition_job and last_job)
                    and first_pick_time < next_begin
                    and last_pick_time >= next_begin
                ):
                    if last_partition_job:
                        for line in format_event(event, "***D"):
                            logger.info(line)
                        logger.info(
                            f"Cross-partition event suppressed, next job will handle it ({event.resource_id.id})"
                        )
                        locator.catalog.events.remove(event)
                        locator.nb_events = len(locator.catalog)
                        clustcat = locator.catalog
                    else:
                        for line in format_event(event, "***P"):
                            logger.info(line)
                        logger.info(
                            f"Found event between normal and overlapped zone where picks must be (P)runed ({event.resource_id.id})"
                        )
                        picks_to_remove = []
                        for origin in event.origins:
                            picks_to_remove += get_picks_from_event(
                                event, origin, next_begin
                            )
                        # Remove picks beyond next_begin from the event for the next iteration
                        if picks_to_remove:
                            logger.info(f"Removing {len(picks_to_remove)} picks from straddle event")
                            for pick in picks_to_remove:
                                # Remove pick from event's preferred origin picks
                                if event.preferred_origin() and pick in event.preferred_origin().picks:
                                    event.preferred_origin().picks.remove(pick)
                                # Remove pick from all origins
                                for origin in event.origins:
                                    if pick in origin.picks:
                                        origin.picks.remove(pick)
                            logger.info(f"Event kept with {len(event.preferred_origin().picks) if event.preferred_origin() else 0} picks after pruning")

                # Rule 4 — Normal acceptance
                else:
                    for line in format_event(event, "****"):
                        logger.info(line)
        else:
            logger.info("No event found in theses clusters.")

        # Write picks probabilities and event_ids
        feed_picks_probabilities(clustcat, previous_myclust.clusters)
        feed_picks_event_ids(clustcat, previous_myclust.clusters)

        # Write distance from preferred origin and prelocalization
        clustcat = feed_distance_from_preloc_to_pref_origin(clustcat)

        # Write partial qml file and clean catalog from memory
        last_saved_event_count += len(clustcat)
        if last_saved_event_count > cfg.catalog.event_flush_count:
            save_catalog(locator.catalog, cfg, job_index, part=i)
            locator.catalog.clear()
            gc.collect()
            last_saved_event_count = 0

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

    # Save to a per-worker temporary SQLite database (merged later by finalize_sqlite())
    if cfg.catalog.enable_sqlite and len(catalog) > 0:
        temp_dir = cfg.catalog.temp_db_dir or cfg.catalog.sqlite_db_path
        temp_db_path = os.path.join(temp_dir, f"tmp_worker_{job_index}.db")
        logger.info(f"Writing {len(catalog)} events to temp DB {temp_db_path}")
        try:
            conn = create_schema(temp_db_path)
            import_catalog_to_sqlite(conn, catalog, enable_quakeml=True, disable_tqdm=True)
            conn.commit()
            conn.close()
            logger.info(f"Temp DB written: {temp_db_path}")
        except Exception as e:
            logger.error(f"Error writing catalog to temp SQLite: {e}")
