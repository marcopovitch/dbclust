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
from typing import Any, List, Optional

import numpy as np
import pandas as pd
import pyproj
import pyproj.exceptions

from obspy import UTCDateTime

from dbclust.clusterize import Clusterize
from dbclust.clusterize import feed_cluster_stability
from dbclust.clusterize import feed_picks_event_ids
from dbclust.clusterize import feed_picks_probabilities
from dbclust.clusterize import get_picks_from_event
from dbclust.clusterize import merge_cluster_with_common_phases
from dbclust.config import DBClustConfig
from dbclust.db import duckdb_init
from dbclust.dbclust2pyocto import adjust_associator_tolerance
from dbclust.inject_spatialite import (
    create_schema,
    import_catalog_to_sqlite,
)
from dbclust.localization import NllLoc
from dbclust.localization import format_event
from dbclust.phase import Phase
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


def apply_station_filters(
    df: pd.DataFrame,
    blacklist: Optional[List[str]],
    frequency_threshold: Optional[float],
    window_duration_minutes: float,
    rename: Optional[Any],
) -> pd.DataFrame:
    """Apply the station-level filters (blacklist, frequency, rename) to a picks DataFrame.

    Shared between the forward window (df_subset) and the backward-overlap
    window (df_backward_overlap) so both see the same station configuration.
    """
    if blacklist:
        for b in blacklist:
            df = df[~df["station_id"].str.contains(b, regex=True)]

    if frequency_threshold:
        total_duration_in_minutes = window_duration_minutes
        if total_duration_in_minutes == 0:
            total_duration_in_minutes = 1.0  # Avoid division by zero
        grouped = df.groupby("station_id")
        station_counts = grouped.size()
        frequencies = station_counts / total_duration_in_minutes
        station_ids_to_keep = frequencies[frequencies < frequency_threshold].index
        df = df[df["station_id"].isin(station_ids_to_keep)]

    if rename is not None:
        df = rename_waveform_id(df, rename)

    return df


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
    df2["station_id"] = df2["station_id"].map(lambda x: ".".join(x.split(".")[:2]) if pd.notna(x) else x)
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


def get_cross_partition_picks(
    con, start, overlap_timedelta, global_start,
    P_proximity_threshold: float = 0.1,
    S_proximity_threshold: float = 0.2,
    P_proba_threshold: float = 0.6,
    S_proba_threshold: float = 0.5,
) -> pd.DataFrame:
    """Fetch all picks from the backward overlap zone [start-overlap, start).

    Returns a deduplicated DataFrame with all picks (catalogued and DL automatic).
    Injected into df_subset before HDBSCAN so that cross-partition events
    (suppressed by the forward overlap rule of the previous job) are
    reconstructed with their full pick set in this job.
    """
    backward_start = max(start - overlap_timedelta, global_start)
    if backward_start >= start:
        return pd.DataFrame()

    rqt = f"""
        SELECT DISTINCT station_id, channel, phase_type, phase_time,
                        phase_score, phase_evaluation, phase_method,
                        event_id, agency
        FROM PICKS
        WHERE phase_time >= '{backward_start}' AND phase_time < '{start}'
        AND phase_type IN ('P', 'Pg', 'Pn', 'S', 'Sg', 'Sn')
        AND (
            (phase_type IN ('P', 'Pg', 'Pn') AND phase_score >= {P_proba_threshold})
            OR
            (phase_type IN ('S', 'Sg', 'Sn') AND phase_score >= {S_proba_threshold})
        )
    """
    df_all = con.sql(rqt).fetchdf()
    if df_all.empty:
        return pd.DataFrame()

    if df_all["phase_time"].dt.tz is None:
        df_all["phase_time"] = df_all["phase_time"].dt.tz_localize("UTC")
    else:
        df_all["phase_time"] = df_all["phase_time"].dt.tz_convert("UTC")

    return deduplicate_picks_by_time(df_all, P_proximity_threshold, S_proximity_threshold)


def find_cluster_phases_for_event(event, clusters: List[List]) -> List:
    """Return the Phase objects from the cluster that produced this event.

    Matching is done by (station, time) against the preferred origin arrivals.
    Returns all phases from the best-matching cluster (not just the arrivals),
    so that picks discarded by NLL are also carried forward.
    """
    origin = event.preferred_origin()
    if origin is None:
        return []

    # Build a set of (station, time) from arrivals that were used by NLL
    pick_by_id = {p.resource_id: p for p in event.picks}
    anchor_keys: set = set()
    for arrival in origin.arrivals:
        if arrival.time_weight is not None and arrival.time_residual is not None:
            pick = pick_by_id.get(arrival.pick_id)
            if pick is not None:
                anchor_keys.add((pick.waveform_id["station_code"], pick.time.datetime))

    if not anchor_keys:
        return []

    # Find the cluster with the most matching phases
    best_cluster: List = []
    best_count = 0
    for cluster in clusters:
        if not cluster:
            continue
        count = sum(
            1 for p in cluster
            if (p.station, p.time.datetime) in anchor_keys
        )
        if count > best_count:
            best_count = count
            best_cluster = cluster

    if best_count == 0:
        return []

    return list(best_cluster)


def get_locator_from_config(cfg: DBClustConfig) -> NllLoc:
    """Create a NllLoc instance from configuration.

    Args:
        cfg: DBClust configuration.

    Returns:
        Configured NllLoc instance.
    """
    # Type assertions and defaults for potentially None config values
    gap_dist_max_km: int = int(cfg.relocation.gap_dist_max_km) if cfg.relocation.gap_dist_max_km is not None else 360
    nll_min_phase: int = cfg.nll.min_phase if cfg.nll.min_phase is not None else 4
    min_score_threshold_pick_zone: float = cfg.relocation.min_score_threshold_pick_zone if cfg.relocation.min_score_threshold_pick_zone is not None else 0.0
    use_pick_zone: bool = cfg.relocation.use_pick_zone if cfg.relocation.use_pick_zone is not None else False
    enable_relabel_pick_zone: bool = cfg.relocation.enable_relabel_pick_zone if cfg.relocation.enable_relabel_pick_zone is not None else False
    enable_cleanup_pick_zone: bool = cfg.relocation.enable_cleanup_pick_zone if cfg.relocation.enable_cleanup_pick_zone is not None else False
    
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
        gap_dist_max_km=gap_dist_max_km,
        closest_station_dist_km=cfg.relocation.closest_station_dist_km,
        dist_km_cutoff=cfg.relocation.dist_km_cutoff,
        use_deactivated_arrivals=cfg.relocation.use_deactivated_arrivals,
        keep_manual_picks=cfg.relocation.keep_manual_picks,
        nll_min_phase=nll_min_phase,
        min_station_score=cfg.cluster.min_station_score,
        min_station_with_P_and_S=cfg.cluster.min_station_with_P_and_S,
        min_ps_ratio=cfg.cluster.min_ps_ratio,
        min_ps_ratio_wilson_z=getattr(cfg.cluster, "min_ps_ratio_wilson_z", None),
        quakeml_settings=asdict(cfg.quakeml),
        keep_scat=cfg.nll.enable_scatter,
        #
        zones=cfg.zones,
        force_zone_name="",
        min_score_threshold_pick_zone=min_score_threshold_pick_zone,
        use_pick_zone=use_pick_zone,
        enable_relabel_pick_zone=enable_relabel_pick_zone,
        enable_cleanup_pick_zone=enable_cleanup_pick_zone,
        min_dist_relabel_deg=getattr(cfg.relocation, "min_dist_relabel_deg", 0.0),
        min_time_weight=getattr(cfg.relocation, "min_time_weight", None),
        enable_residual_threshold_with_pick_zone=getattr(cfg.relocation, "enable_residual_threshold_with_pick_zone", False),
        pass2_degradation_factor=getattr(cfg.relocation, "pass2_degradation_factor", None),
        pass2_fallback=getattr(cfg.relocation, "pass2_fallback", False),
        enable_time_weight_outlier_filter=getattr(cfg.relocation, "enable_time_weight_outlier_filter", False),
        time_weight_outlier_mad_factor=getattr(cfg.relocation, "time_weight_outlier_mad_factor", 3.0),
        time_weight_outlier_min_picks=getattr(cfg.relocation, "time_weight_outlier_min_picks", 5),
        time_weight_outlier_absolute_threshold=getattr(cfg.relocation, "time_weight_outlier_absolute_threshold", None),
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
    # Type assertions and defaults for potentially None config values
    average_velocity: int = int(cfg.cluster.average_velocity) if cfg.cluster.average_velocity is not None else 5
    max_search_dist: int = int(cfg.cluster.max_search_dist) if cfg.cluster.max_search_dist is not None else 100
    apparent_vp: float = float(cfg.cluster.apparent_vp) if cfg.cluster.apparent_vp is not None else 6.0
    apparent_vs: float = float(cfg.cluster.apparent_vs) if cfg.cluster.apparent_vs is not None else 3.5
    umap_alpha: float = float(cfg.cluster.umap_alpha) if cfg.cluster.umap_alpha is not None else 0.3
    # umap_clip_seconds: use explicit value if set, otherwise fall back to overlap_window
    umap_clip_seconds: float = (
        float(cfg.cluster.umap_clip_seconds)
        if cfg.cluster.umap_clip_seconds is not None
        else (cfg.time.overlap_window if cfg.cluster.use_umap else 0.0)
    )
    tt_matrix_fname: str = cfg.cluster.pre_computed_tt_matrix_file or ""

    myclust = Clusterize(
        phases=phases,
        min_cluster_size=cfg.cluster.min_cluster_size,
        average_velocity=average_velocity,
        min_station_count=cfg.cluster.min_station_count,
        min_station_with_P_and_S=cfg.cluster.min_station_with_P_and_S,
        min_station_with_P_and_S_stability_override=cfg.cluster.min_station_with_P_and_S_stability_override,
        min_station_with_P_and_S_score_override=cfg.cluster.min_station_with_P_and_S_score_override,
        min_station_score=cfg.cluster.min_station_score,
        min_ps_ratio=cfg.cluster.min_ps_ratio,
        force_keep_catalog_events=cfg.cluster.force_keep_catalog_events,
        use_umap=cfg.cluster.use_umap,
        umap_clip_seconds=umap_clip_seconds,
        umap_aot_tt_blend_alpha=umap_alpha,
        apparent_vp=apparent_vp,
        apparent_vs=apparent_vs,
        cluster_selection_method=cfg.cluster.cluster_selection_method,
        allow_single_cluster=cfg.cluster.allow_single_cluster,
        tt_clip_seconds=cfg.cluster.tt_clip_seconds,
        max_search_dist=max_search_dist,
        P_uncertainty=cfg.pick.P_uncertainty,
        S_uncertainty=cfg.pick.S_uncertainty,
        min_com_phases=cfg.cluster.min_picks_common,
        eventid_shared_min_picks_per_cluster=cfg.cluster.eventid_shared_min_picks_per_cluster,
        eventid_shared_min_distinct_ids=cfg.cluster.eventid_shared_min_distinct_ids,
        clustering_method=cfg.cluster.clustering_method,
        leiden_resolution=cfg.cluster.leiden_resolution,
        leiden_edge_weight_scale=cfg.cluster.leiden_edge_weight_scale,
        leiden_ps_boost_factor=cfg.cluster.leiden_ps_boost_factor,
        leiden_min_edge_weight=cfg.cluster.leiden_min_edge_weight,
        leiden_vp=cfg.cluster.leiden_vp,
        leiden_vs=cfg.cluster.leiden_vs,
        mega_cluster_fallback_leiden=cfg.cluster.mega_cluster_fallback_leiden,
        mega_cluster_threshold=cfg.cluster.mega_cluster_threshold,
        mega_cluster_min_size=cfg.cluster.mega_cluster_min_size,
        mega_cluster_leiden_resolution=cfg.cluster.mega_cluster_leiden_resolution,
        leiden_hdbscan_fallback=cfg.cluster.leiden_hdbscan_fallback,
        leiden_min_stability=cfg.cluster.leiden_min_stability,
        tt_matrix_fname=tt_matrix_fname,
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

        if cfg.parallel.time_partitions is None:
            logger.error("time_partitions not configured for parallel mode.")
            return False

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
        if cfg.parallel.time_partitions is None:
            logger.error("time_partitions not configured.")
            return False

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
        pick_type = cfg.pick.type or "csv"
        con = duckdb_init(cfg.pick.filenames, pick_type)
    else:
        # Uses the pandas Dataframe given as function argument.
        con = None

    try:
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
        deferred_phases_next_round: List[List[Phase]] = []  # one sub-list per deferred event
        deferred_phases_keys: set = set()  # (station, time, phase) — global dedup across all deferred events
        i = 0

        # start time looping
        for i, (begin, end) in enumerate(time_divisions, start=1):
            # Check if this is the last time divisions
            if job_index is not None and i == len(time_divisions):
                last_partition_job = True
            else:
                last_partition_job = False

            # add the time overlap only if it is not the last round
            pick_end = cfg.pick.end
            if pick_end is None:
                pick_end = pd.Timestamp.now(tz="UTC")
            pick_end_ts = pd.Timestamp(pick_end)
            # Normalize pick_end to match begin's tz-awareness to allow comparison
            if begin.tz is None and pick_end_ts.tz is not None:
                pick_end_ts = pick_end_ts.tz_convert("UTC").tz_localize(None)
            elif begin.tz is not None and pick_end_ts.tz is None:
                pick_end_ts = pick_end_ts.tz_localize("UTC")
            if end >= pick_end_ts:
                end = pick_end_ts
                short_window = True

                # complementary check
                if end < begin:
                    end = begin

            else:
                end += overlap_timedelta
                short_window = False

            logger.info("")
            logger.info("=" * 72)
            logger.info(
                f"============== job index:[{job_index}] Time window extraction with overlap {overlap_timedelta}: #{i}/{len(time_divisions)} picks from {begin} to {end}."
            )
            logger.info("=" * 72)

            # Extract picks on this time period
            df_backward_overlap = None  # populated only for first window of non-first jobs
            if con:
                begin_year = begin.year
                begin_month = begin.month
                end_year = end.year
                end_month = end.month

                if cfg.pick.type == "parquet":
                    # benefit from parquet partitioning by year and month
                    rqt = f"""
                        SELECT DISTINCT station_id, channel, phase_type, phase_time,
                                        phase_score, phase_evaluation, phase_method,
                                        event_id, agency
                        FROM PICKS
                        WHERE
                        (year > {begin_year} OR (year = {begin_year} AND month >= {begin_month}))
                        AND
                        (year < {end_year} OR (year = {end_year} AND month <= {end_month}))
                        AND
                        phase_time BETWEEN '{begin}' AND '{end}'
                        AND
                        phase_type IN ('P', 'Pg', 'Pn', 'S', 'Sg', 'Sn')
                        AND (
                            (phase_type IN ('P', 'Pg', 'Pn') AND phase_score >= {cfg.pick.P_proba_threshold})
                            OR
                            (phase_type IN ('S', 'Sg', 'Sn') AND phase_score >= {cfg.pick.S_proba_threshold})
                        )
                    """
                else:
                    # csv
                    rqt = f"""
                        SELECT DISTINCT station_id, channel, phase_type, phase_time,
                                        phase_score, phase_evaluation, phase_method,
                                        event_id, agency
                        FROM PICKS
                        WHERE phase_time BETWEEN '{begin}' AND '{end}'
                        AND phase_type IN ('P', 'Pg', 'Pn', 'S', 'Sg', 'Sn')
                        AND (
                            (phase_type IN ('P', 'Pg', 'Pn') AND phase_score >= {cfg.pick.P_proba_threshold})
                            OR
                            (phase_type IN ('S', 'Sg', 'Sn') AND phase_score >= {cfg.pick.S_proba_threshold})
                        )
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

                # First window of non-first parallel jobs: capture picks from the
                # backward overlap zone [start-overlap, start].  These are NOT
                # injected into df_subset before HDBSCAN (which could cause
                # mega-clusters in TT or UMAP space).  Instead they are absorbed
                # post-clustering via Clusterize.absorb_backward_picks() which
                # assigns them to existing clusters (Case A) or forms new ones
                # from the residuals + noise (Case B).
                if parallel_mode and job_index > 0 and i == 1:
                    pick_start = cfg.pick.start or pd.Timestamp("1970-01-01")
                    global_start = pd.Timestamp(pick_start).tz_localize(None) if not hasattr(pick_start, 'tz') else pd.Timestamp(pick_start)
                    df_backward_overlap = get_cross_partition_picks(
                        con, start, overlap_timedelta, global_start,
                        cfg.pick.P_proximity_threshold,
                        cfg.pick.S_proximity_threshold,
                        cfg.pick.P_proba_threshold,
                        cfg.pick.S_proba_threshold,
                    )
                    logger.info(
                        f"[{job_index}] Captured {len(df_backward_overlap)} backward overlap picks"
                        f" for post-clustering absorption."
                    )

            else:
                df_subset = df[(df["phase_time"] >= begin) & (df["phase_time"] < end)]

            if df_subset.empty and previous_myclust.phases_count() == 0 and not deferred_phases_next_round:
                logger.info(f"[{job_index}] Skipping clustering {len(df_subset)} phases.")
                continue

            # remove blacklisted stations
            if cfg.station.blacklist:
                for b in cfg.station.blacklist:
                    df_subset = df_subset[
                        ~df_subset["station_id"].str.contains(b, regex=True)
                    ]

            # remove picks from backward-overlap window that are not part of df_subset's
            # station filtering (blacklist/frequency/rename), so both windows are
            # subject to the same station configuration.
            if df_backward_overlap is not None and not df_backward_overlap.empty:
                df_backward_overlap = apply_station_filters(
                    df_backward_overlap,
                    cfg.station.blacklist,
                    cfg.station.frequency_threshold,
                    (end - begin).total_seconds() / 60,
                    cfg.station.rename,
                )

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

            # remove picks based on station frequency threshold, then rename stations
            df_subset = apply_station_filters(
                df_subset,
                blacklist=None,  # already applied above, before unload_picks_list
                frequency_threshold=cfg.station.frequency_threshold,
                window_duration_minutes=(end - begin).total_seconds() / 60,
                rename=cfg.station.rename,
            )

            # Import forward picks and get coordinates
            forward_phases = import_phases(
                df_subset,
                cfg.pick.P_proba_threshold,
                cfg.pick.S_proba_threshold,
                cfg.pick.P_uncertainty,
                cfg.pick.S_uncertainty,
                cfg.station.info_sta,
                cfg.station.fallback_df,
            )

            if logger.level == logging.DEBUG:
                for p in forward_phases:
                    p.show_all()

            # clean up
            del df_subset
            gc.collect()

            if logger.level == logging.DEBUG:
                logger.info("previous_myclust:")
                previous_myclust.show_clusters()

            if df_backward_overlap is not None and not df_backward_overlap.empty:
                # Backward picks available: cluster backward+forward together in one
                # HDBSCAN pass so that late arrivals of backward events (e.g. S picks
                # arriving just after the job boundary) are not mixed into forward
                # clusters from different events.
                backward_phases = import_phases(
                    df_backward_overlap,
                    cfg.pick.P_proba_threshold,
                    cfg.pick.S_proba_threshold,
                    cfg.pick.P_uncertainty,
                    cfg.pick.S_uncertainty,
                    cfg.station.info_sta,
                    cfg.station.fallback_df,
                )
                myclust = get_clusterize_from_config(cfg, phases=None)
                myclust.build_clusters_from_backward(backward_phases, forward_phases)
                del backward_phases
            else:
                myclust = get_clusterize_from_config(cfg, phases=forward_phases)

            del forward_phases
            df_backward_overlap = None

            if logger.level == logging.DEBUG:
                logger.info("myclust:")
                myclust.show_clusters()

            # check if some clusters share phases with previous round
            logger.info("Check clusters related to the same event (overlapped zone).")

            # Inject deferred cluster phases from the previous ***D event into myclust
            # BEFORE merge and smart-overlap promotion, so the cluster can absorb nearby
            # noise picks from the current window and participate in the ready/deferred split.
            if deferred_phases_next_round:
                total_deferred = sum(len(c) for c in deferred_phases_next_round)
                logger.info(
                    f"[{job_index}] Injecting {total_deferred} deferred"
                    f" cluster phases ({len(deferred_phases_next_round)} event(s))"
                    f" into myclust (partition #{i}) for enrichment."
                )
                for event_cluster in deferred_phases_next_round:
                    myclust.absorb_deferred_cluster(
                        event_cluster,
                        overlap_seconds=cfg.time.overlap_window,
                    )
                deferred_phases_next_round = []
                deferred_phases_keys = set()

            previous_myclust, myclust, _ = (
                merge_cluster_with_common_phases(
                    previous_myclust,
                    myclust,
                    cfg.cluster.min_picks_common,
                    eventid_shared_min_picks_per_cluster=cfg.cluster.eventid_shared_min_picks_per_cluster,
                    eventid_shared_min_distinct_ids=cfg.cluster.eventid_shared_min_distinct_ids,
                )
            )

            if last_partition_job:
                logger.info(
                    "==> Last job in the time partition, merging all remaining clusters."
                )
                previous_myclust.merge(myclust)
            elif parallel_mode and job_index > 0 and i == 1 and myclust.n_clusters > 0:
                # Window #1 of a non-first parallel job: the single cluster produced by
                # build_clusters_from_backward spans backward+forward picks and would be
                # fully deferred by the temporal promotion logic (last_pick >= overlap_start).
                # Instead, promote it unconditionally so PyOcto can separate its events —
                # those with picks in the overlap zone will be re-deferred by Rule 1.
                logger.info(
                    f"[{job_index}] Window #1 backward+forward: promoting all {myclust.n_clusters}"
                    f" cluster(s) to previous_myclust for PyOcto processing."
                )
                previous_myclust.merge(myclust)
                # Clear myclust so these clusters are not re-processed in window #2.
                myclust = get_clusterize_from_config(cfg, phases=None)
            elif myclust.n_clusters > 0:
                # Smart overlap promotion: only promote clusters whose picks are entirely
                # before the overlap zone (temporally complete — they won't gain more picks
                # in the next window). Clusters with picks reaching into the overlap zone
                # are left to carry over naturally so they merge with the next window's picks
                # and avoid forming contaminated mega-clusters.
                # short_window has no meaningful overlap zone: all clusters are ready.
                overlap_start = UTCDateTime(
                    (end - overlap_timedelta).isoformat() if not short_window else end.isoformat()
                )

                ready_clusters = []
                ready_stabilities = []
                deferred_clusters = []
                deferred_stabilities = []

                for j, cluster in enumerate(myclust.clusters):
                    if not cluster:
                        continue
                    last_pick_ts = max(p.time for p in cluster)
                    stab = (
                        float(myclust.clusters_stability[j])
                        if len(myclust.clusters_stability) > j
                        else 1.0
                    )
                    if last_pick_ts < overlap_start:
                        ready_clusters.append(cluster)
                        ready_stabilities.append(stab)
                    else:
                        deferred_clusters.append(cluster)
                        deferred_stabilities.append(stab)

                if ready_clusters:
                    overlap_start_label = overlap_start.isoformat()
                    logger.info(
                        f"Promoting {len(ready_clusters)} temporally-complete cluster(s) "
                        f"(all picks before overlap zone {overlap_start_label}); "
                        f"{len(deferred_clusters)} cluster(s) deferred to next window."
                    )
                    previous_myclust.clusters += ready_clusters
                    previous_myclust.n_clusters = len(previous_myclust.clusters)
                    previous_myclust.clusters_stability = np.concatenate([
                        np.atleast_1d(np.array(previous_myclust.clusters_stability, dtype=float)),
                        np.array(ready_stabilities, dtype=float),
                    ])
                    # Move noise from myclust into the promoted batch so PyOcto has
                    # access to unassigned picks without double-counting.
                    previous_myclust.noise = (
                        list(getattr(previous_myclust, "noise", []) or [])
                        + list(myclust.noise or [])
                    )
                    previous_myclust.n_noise = len(previous_myclust.noise)
                else:
                    logger.info(
                        f"No temporally-complete clusters: all {myclust.n_clusters} "
                        f"cluster(s) have picks in the overlap zone, deferring to next window."
                    )

                # Keep only deferred clusters in myclust for next round
                myclust.clusters = deferred_clusters
                myclust.n_clusters = len(deferred_clusters)
                myclust.clusters_stability = (
                    np.array(deferred_stabilities, dtype=float)
                    if deferred_stabilities
                    else np.ones(0, dtype=float)
                )
                myclust.noise = [] if ready_clusters else list(myclust.noise or [])
                myclust.n_noise = len(myclust.noise)

            # Snapshot after PyOcto (or before if disabled) — per-event clusters used
            # by find_cluster_phases_for_event to match ***D events to their exact cluster.
            clusters_for_deferred_search = list(previous_myclust.clusters)

            if cfg.pyocto.enable and cfg.pyocto.current_model:
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
                    logger.exception(
                        f"Unexpected error in adjust_associator_tolerance(): {e}"
                    )
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
                    # Update snapshot: PyOcto has now split mega-clusters into per-event clusters.
                    clusters_for_deferred_search = list(previous_myclust.clusters)

            # Process previous_myclust and wait next round to process myclust
            with MyTemporaryDirectory(
                dir=cfg.file.obs_path, delete=cfg.file.automatic_cleanup_tmp
            ) as TMP_OBS_PATH:
                my_obs_path = os.path.join(TMP_OBS_PATH, f"{i}")
                # Merge any clusters that share a dominant event_id before NLL submission.
                # This prevents the same physical event from being localized twice when
                # ready_clusters (promoted from myclust) and/or PyOcto fallback leave two
                # Leiden communities covering the same earthquake in previous_myclust.
                previous_myclust.cluster_merge_based_on_eventid()
                nll_picks = previous_myclust.generate_nllobs(my_obs_path)

                logger.info("-" * 60)
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

                    # Rule 0 — Backward overlap zone (parallel mode, first window of job N):
                    # any event whose last pick falls before the job's partition start was
                    # already handled by job N-1 and must be suppressed to avoid duplicates.
                    # EXCEPTION: if first_pick >= start - overlap, job N-1's Rule 2 would have
                    # deferred this event (not kept it), so we must NOT suppress it here.
                    if (
                        parallel_mode
                        and job_index > 0
                        and i == 1
                        and last_pick_time < start
                        and first_pick_time < start - overlap_timedelta
                    ):
                        for line in format_event(event, "***D"):
                            logger.info(line)
                        logger.info(
                            f"Event in backward overlap zone (last_pick={last_pick_time} < partition_start={start}), suppressed ({event.resource_id.id})"
                        )
                        locator.catalog.events.remove(event)
                        locator.nb_events = len(locator.catalog)
                        try:
                            clustcat.events.remove(event)
                        except ValueError:
                            pass

                    # Rule 1 — Window overlap zone: first_pick falls beyond next_begin.
                    # The next window will detect this event with more picks.
                    elif (not last_partition_job) and event_in_overlapped_zone:
                        for line in format_event(event, "***D"):
                            logger.info(line)
                        logger.info(
                            f"Found event in overlapped zone to be (D)eleted ({event.resource_id.id})"
                        )

                        cluster_phases = find_cluster_phases_for_event(
                            event, clusters_for_deferred_search
                        )
                        if cluster_phases:
                            event_cluster: List[Phase] = []
                            added = 0
                            for p in cluster_phases:
                                key = (p.station, p.time.datetime, p.phase)
                                if key not in deferred_phases_keys:
                                    event_cluster.append(p)
                                    deferred_phases_keys.add(key)
                                    added += 1
                            if event_cluster:
                                deferred_phases_next_round.append(event_cluster)
                            logger.info(
                                f"[{job_index}] Deferred {added} cluster phases from ***D event for next window re-clustering."
                            )

                        locator.catalog.events.remove(event)
                        locator.nb_events = len(locator.catalog)
                        try:
                            clustcat.events.remove(event)
                        except ValueError:
                            pass

                    # Rule 2 — Forward overlap zone (parallel mode, last window of job N):
                    # any event starting in [stop-overlap, stop] is deferred to job N+1,
                    # which will reconstruct it with the full pick set via backward injection.
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
                        try:
                            clustcat.events.remove(event)
                        except ValueError:
                            pass

                    # Rule 3 — Straddle: first_pick in normal zone, last_pick in overlap zone.
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
                            try:
                                clustcat.events.remove(event)
                            except ValueError:
                                pass
                        else:
                            for line in format_event(event, "***P"):
                                logger.info(line)
                            logger.info(
                                f"Found event between normal and overlapped zone where picks must be (P)runed ({event.resource_id.id})"
                            )
                            for origin in event.origins:
                                event_picks = get_picks_from_event(
                                    event, origin, next_begin
                                )
                                picks_to_remove += event_picks

                    # Rule 4 — Normal acceptance
                    else:
                        for line in format_event(event, "****"):
                            logger.info(line)
            else:
                logger.info("No event found in theses clusters.")

            # Write picks probabilities and event_ids
            feed_picks_probabilities(clustcat, previous_myclust.clusters)
            feed_picks_event_ids(clustcat, previous_myclust.clusters)
            feed_cluster_stability(
                clustcat,
                previous_myclust.clusters,
                previous_myclust.clusters_stability,
                getattr(previous_myclust, "clusters_pyocto_ops", None),
            )

            # Write distance from preferred origin and prelocalization
            clustcat = feed_distance_from_preloc_to_pref_origin(clustcat)

            # Write partial qml file and clean catalog from memory
            last_saved_event_count += len(clustcat)
            if last_saved_event_count > cfg.catalog.event_flush_count:
                save_catalog(locator.catalog, cfg, job_index, part=i)
                locator.catalog.clear()
                gc.collect()
                last_saved_event_count = 0

            # Remove the deferred cluster from previous_myclust after localization so it
            # is not re-submitted to NLL in subsequent partitions (would cause duplicates).
            deferred_ref = getattr(myclust, "_deferred_cluster_ref", None)
            if deferred_ref is not None:
                try:
                    previous_myclust.clusters.remove(deferred_ref)
                    previous_myclust.n_clusters = len(previous_myclust.clusters)
                    logger.info("[deferred] Removed deferred cluster from previous_myclust after localization.")
                except ValueError:
                    pass  # already removed or never promoted
                myclust._deferred_cluster_ref = None

            # prepare next round
            previous_myclust = myclust

        # Save remaining events
        save_catalog(locator.catalog, cfg, job_index, part=i + 1, finalize=True)
        logger.info(f"Finalizing job index: {job_index}")
        return True
    finally:
        if con is not None:
            con.close()


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
        conn = None
        try:
            conn = create_schema(temp_db_path)
            import_catalog_to_sqlite(conn, catalog, enable_quakeml=True, disable_tqdm=True)
            conn.commit()
            logger.info(f"Temp DB written: {temp_db_path}")
        except Exception as e:
            logger.error(f"Error writing catalog to temp SQLite: {e}")
        finally:
            if conn is not None:
                try:
                    conn.close()
                except Exception as close_error:
                    logger.error(f"Error closing temp SQLite connection: {close_error}")
