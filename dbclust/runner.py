#!/usr/bin/env python
import argparse
import csv
import gc
import logging
import math
import os
import random
import shutil
import sqlite3
import sys
import tempfile
import threading
import time
import warnings
from dataclasses import asdict
from typing import List
from typing import Optional

import pandas as pd
import psutil
import pyproj

os.environ["RAY_DEDUP_LOGS"] = "0"
import ray
from icecream import ic

from dbclust.clusterize import Clusterize
from dbclust.clusterize import feed_picks_event_ids
from dbclust.clusterize import feed_picks_probabilities
from dbclust.clusterize import get_picks_from_event
from dbclust.clusterize import merge_cluster_with_common_phases
from dbclust.config import DBClustConfig
from dbclust.db import duckdb_init
from dbclust.dbclust2pyocto import adjust_associator_tolerance
from dbclust.inject_spatialite import import_catalog_object_to_sqlite_from_file
from dbclust.inject_spatialite import refresh_event_coordinates_view
from dbclust.localization import NllLoc
from dbclust.localization import show_event
from dbclust.phase import import_phases
from dbclust.preprocessing_picks import deduplicate_picks_by_time
from dbclust.quakeml import deduplicate_picks_and_make_readable_ids
from dbclust.quakeml import feed_distance_from_preloc_to_pref_origin
from dbclust.rename import rename_waveform_id

warnings.filterwarnings("ignore", category=UserWarning)
ic.configureOutput(prefix="DBClust: ")
# default logger
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger("dbclust")
logger.setLevel(logging.INFO)


class MyTemporaryDirectory:
    def __init__(self, delete=True, dir=None):
        self.delete = delete
        self.dir = dir

    def __enter__(self):
        self.tmp_dir = tempfile.mkdtemp(dir=self.dir)
        return self.tmp_dir

    def __exit__(self, exc_type, exc_value, traceback):
        if self.delete:
            shutil.rmtree(self.tmp_dir)
        else:
            logger.info(f"Warning undeleted directory: {self.tmp_dir}")


def unload_picks_list(df1, picks):
    # format picks coming from events like the ones used as input for dbclust
    df2 = pd.DataFrame(picks, columns=["station_id", "phase_type", "phase_time"])
    df2["station_id"] = df2["station_id"].map(lambda x: ".".join(x.split(".")[:2]))
    df2["phase_time"] = pd.to_datetime(
        df2["phase_time"].map(lambda x: str(x)), utc=True
    )
    df2["unload"] = True
    # remove duplicate pick as they came from same event but from multiple origins
    df2.drop_duplicates(inplace=True)
    # df1.to_csv("df1.csv")
    # df2.to_csv("df2.csv")
    results = pd.merge(
        df1, df2, how="left", on=["station_id", "phase_type", "phase_time"]
    )
    # results.to_csv("merge.csv")
    keep = results[results["unload"] != True]
    keep = keep.drop(columns=["unload"])
    # print(keep[["station_id", "phase_time"]].to_string())
    # keep.to_csv("keep.csv")
    return keep


def get_locator_from_config(cfg, log_level=logging.INFO):
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


def get_clusterize_from_config(cfg, phases=None, log_level=logging.INFO):
    myclust = Clusterize(
        phases=phases,  # empty cluster / constructor only
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
) -> None:
    """Detect and localize events given picks

    Args:
        cfg (DBClustConfig): dbclust parameters and data
        df (Optional[pd.DataFrame], optional): use df if defined rather than picks from cfg
        job_index (int): job index, None if in sequential mode

    Returns:
        None

    Comments:
        - The function is designed to process data in parallel, using the job_index to determine the time window for each job.
        - It uses duckdb for SQL queries if df is None or empty, otherwise it processes the provided DataFrame.
        - The function handles clustering, localization, and saving of results to files.
        - It also includes logic for handling overlapping time periods and merging clusters.
        - The function uses a temporary directory for intermediate files and cleans up after processing.
        - The workflow includes deduplication of picks, clustering, localization, and saving results to QuakeML and SQLite formats.
        - The workflow works on 2 time windows:
            - previous time window containing the clusters: this is the one that is being processed, taking into account the overlap
            - current time window containing also the clusters: this is the one that will be processed in the next round
        - last_job is used to indicate  if the current job is the last one ever
        - last_partition_job is used to indicate if the current job is the last one in the time partition

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

        # get event in the overlapped zone during the last time_divisions round
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
        # if there is no other jobs after this one
        # previous_myclust and myclust will be merged
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
                # do not skip the last time division
                # to be able to process the previous_myclust
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

            # Time meseaure of the query
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
        # base on pick proximity and probability
        logger.info(
            f"[{job_index}] Starting pick preprocessing with {len(df_subset)} phases."
        )
        df_subset = deduplicate_picks_by_time(
            df_subset,
            cfg.pick.P_proximity_threshold,
            cfg.pick.S_proximity_threshold,
        )
        # df_subset.to_csv(f"df_subset_{job_index}_{i}.csv")
        logger.info(
            f"[{job_index}] End pick preprocessing with {len(df_subset)} phases."
        )

        # To prevents extra event, remove from current picks list,
        # picks previously associated with events on the previous iteration
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

        # remove picks in respect to the number of station per minutes in df_subset
        if cfg.station.frequency_threshold:
            total_duration_in_minutes = (end - begin).total_seconds() / 60
            grouped = df_subset.groupby("station_id")
            station_counts = grouped.size()
            # min_time = grouped["phase_time"].min()
            # max_time = grouped["phase_time"].max()
            # total_duration_in_minutes = (max_time - min_time).dt.total_seconds() / 60
            frequencies = station_counts / total_duration_in_minutes
            station_ids_to_keep = frequencies[
                frequencies < cfg.station.frequency_threshold
            ].index
            df_subset = df_subset[df_subset["station_id"].isin(station_ids_to_keep)]
            # ic(
            #     total_duration_in_minutes,
            #     frequencies[frequencies >= cfg.station.frequency_threshold],
            # )

        # Rename station_id.channel by user request,
        # as some agencies have different naming convention (mainly for old stations)
        # witch are well formatted to get waveform data
        df_subset = rename_waveform_id(df_subset, cfg.station.rename)

        # Import picks and get coordinates
        # Warning: this function will modify the df_subset DataFrame in place
        # So modification made by rename_waveform_id() could be lost
        # due to metadata update in import_phases()
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

        # check if some clusters in this round share some phases
        # with clusters from the previous round
        # (as some phases come from the overlapped zone)
        logger.info("Check clusters related to the same event (overlapped zone).")

        previous_myclust, myclust, nb_cluster_removed = (
            merge_cluster_with_common_phases(
                previous_myclust, myclust, cfg.cluster.min_picks_common
            )
        )

        if last_partition_job:
            # This is the last job in the time partition
            # merge previous_myclust and myclust
            logger.info(
                f"==> Last job in the time partition, merging all remaining clusters."
            )
            previous_myclust.merge(myclust)

        if cfg.pyocto.current_model:
            try:
                result = adjust_associator_tolerance(
                    previous_myclust,
                    cfg,
                    # tolerance_steps=cfg.pyocto.tolerance_steps,
                    # {min_tolerance_threshold: pick_match_tolerance, ...}
                    tolerance_steps={1: 1, 0.5: 0.1, 0: 0.05},
                    min_tolerance=0.1,
                    log_level=logger.level,
                )
            except pyproj.exceptions.CRSError as e:
                logger.error(f"Aborting process adjust_associator_tolerance().")
                continue

            if result is None:
                # flush stdout and stderr to have the log in the right order
                logger.error("Failed to process with any pick_match_tolerance.")
                continue
            else:
                previous_myclust = result

        # Now, process previous_myclust and wait next round to process myclust
        # write each cluster to nll obs files
        with MyTemporaryDirectory(
            dir=cfg.file.obs_path, delete=cfg.file.automatic_cleanup_tmp
        ) as TMP_OBS_PATH:
            my_obs_path = os.path.join(TMP_OBS_PATH, f"{i}")
            # nll_picks keeps track of picks to recover info after NonLinLoc
            nll_picks = previous_myclust.generate_nllobs(my_obs_path)

            # localize each cluster
            # all locs are automatically appended to the locator's catalog
            # force to cleanup all files generated by Nonlinloc
            logger.info(f"Starting localization using {locator.loc_method}.")
            with MyTemporaryDirectory(
                dir=cfg.file.tmp_path,
                delete=cfg.file.automatic_cleanup_tmp,
            ) as tmpdir_automaticaly_cleaned:
                locator.tmpdir = tmpdir_automaticaly_cleaned

                # sequential version
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
                _, _, first_pick_time = picks[0]
                _, _, last_pick_time = picks[-1]

                # check if the event is in the overlapped zone
                event_in_overlapped_zone = False
                if short_window == False:
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
                    # Event first pick is in overlapped zone,
                    # remove this event and wait the next iteration
                    # as this event will be recreated.
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
                    # First pick time is before overlapped zone
                    # but some others picks are inside it.
                    # Assuming the overlapped zone is large enough
                    # to have a complete event, remove those picks (in the next round),
                    # so they can't make a new event on the next iteration.
                    # (event is kept, only picks are removed for the next round)
                    # if event.event_type != "not existing":
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

        # Write into qml/comments picks probabilities and event_ids where picks are coming from
        feed_picks_probabilities(clustcat, previous_myclust.clusters)
        feed_picks_event_ids(clustcat, previous_myclust.clusters)

        # Write into qml/comments distance from preferred origin and prelocalization
        clustcat = feed_distance_from_preloc_to_pref_origin(clustcat)

        # Write partial qml file and clean catalog from memory
        if last_saved_event_count > cfg.catalog.event_flush_count:
            # Save intermediate results periodically
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
    cfg,
    job_index: Optional[int] = None,
    part: Optional[int] = None,
    finalize: bool = False,
) -> None:
    """
    Save the catalog to QuakeML and/or SQLite file formats.

    Args:
        catalog (Catalog): The seismic event catalog to be saved.
        cfg (DBClustConfig): Configuration containing save paths and options.
        job_index (Optional[int]): The job index for parallel execution, used for naming files.
        part (Optional[int]): The part number for splitting the catalog, used for naming files.
        finalize (bool): If True, the catalog is saved as a final output.
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
                retries=5,
                delay=1,
                backoff="exponential",
            )
        except Exception as e:
            logger.error(f"Error writing catalog to SQLite: {e}")


# Ray tasks
@ray.remote(max_calls=1, max_retries=5, num_cpus=1, memory=5 * 1024**3)  # 5 GB
def run_dbclust_task(cfg, job_index):
    """Run a DBClust task with the given configuration and job index.
    This function is designed to be executed as a Ray task.
    It initializes the DBClust configuration, runs the clustering and localization process,
    and returns the results.
    Args:
        cfg (DBClustConfig): The configuration object for DBClust.
        job_index (int): The index of the job to run.
    Returns:
        dict: A dictionary containing the task index, duration, peak memory usage, and result.
    """
    start_time = time.time()
    result = dbclust(cfg=cfg, job_index=job_index)
    end_time = time.time()

    duration_sec = end_time - start_time
    peak_memory_mb = 0

    return {
        "task_index": job_index,
        "duration_sec": duration_sec,
        "peak_memory_mb": peak_memory_mb,
        "result": result,
    }



@ray.remote(max_calls=1, max_retries=5, num_cpus=1, memory=5 * 1024**3)  # 5 GB
def profiled_run_dbclust_task(cfg, job_index):
    """Run a DBClust task with profiling for memory usage.
    This function is designed to be executed as a Ray task.
    It initializes the DBClust configuration, runs the clustering and localization process,
    and returns the results along with memory profiling information.
    Args:
        cfg (DBClustConfig): The configuration object for DBClust.
        job_index (int): The index of the job to run.
    Returns:
        dict: A dictionary containing the task index, duration, peak memory usage, and result.
    """
    process = psutil.Process()
    peak_rss = {"value": 0}
    stop_event = threading.Event()

    def monitor():
        while not stop_event.is_set():
            try:
                rss = process.memory_info().rss
                if rss > peak_rss["value"]:
                    peak_rss["value"] = rss
            except Exception:
                pass
            time.sleep(0.05)  # every 50 ms

    monitor_thread = threading.Thread(target=monitor)
    monitor_thread.start()

    try:
        start_time = time.time()
        result = dbclust(cfg=cfg, job_index=job_index)
        end_time = time.time()
    finally:
        stop_event.set()
        monitor_thread.join()

    duration_sec = end_time - start_time
    peak_memory_mb = peak_rss["value"] / (1024 * 1024)  # in MB

    return {
        "task_index": job_index,
        "duration_sec": duration_sec,
        "peak_memory_mb": peak_memory_mb,
        "result": result,
    }


def run_with_ray(cfg: DBClustConfig, profile_csv_path="task_profiles.csv"):
    os.environ["RAY_DEDUP_LOGS"] = "0"
    os.environ["RAY_COLOR_PREFIX"] = "1"
    os.environ["RAY_enable_oom_killer"] = "1"
    os.environ["RAY_memory_usage_threshold"] = "0.95"

    context = ray.init(
        num_cpus=cfg.parallel.n_workers,
        _temp_dir=cfg.parallel._temp_dir,
        dashboard_host="0.0.0.0",
        dashboard_port=8265,
    )
    logger.info(f"Dashboard URL: http://{context.dashboard_url}")

    # Shuffle and submit tasks
    indexed_partitions = list(enumerate(cfg.parallel.time_partitions, start=0))
    random.shuffle(indexed_partitions)

    futures: List[ray.ObjectRef] = []
    for idx, (start, end) in indexed_partitions:
        logger.info(f"Submitting task {idx} [{start} -- {end}]")
        futures.append(run_dbclust_task.remote(cfg, idx))
        #futures.append(profiled_run_dbclust_task.remote(cfg, idx))

    results = ray.get(futures)

    # Extract results and profiling data
    completed_results = []
    task_profiles = []
    for r in results:
        task_profiles.append(
            {
                "task_index": r["task_index"],
                "duration_sec": round(r["duration_sec"], 2),
                "peak_memory_mb": round(r["peak_memory_mb"], 2),
            }
        )
        completed_results.append(r["result"])

    # Write CSV
    with open(profile_csv_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(
            csvfile, fieldnames=["task_index", "duration_sec", "peak_memory_mb"]
        )
        writer.writeheader()
        writer.writerows(task_profiles)

    logger.info(f"Profiling data saved to {profile_csv_path}")
    logger.info("DBClust completed!")

    ray.shutdown()
    return completed_results


#     # Ray initialization
#     os.environ["RAY_DEDUP_LOGS"] = "0"
#     os.environ["RAY_COLOR_PREFIX"] = "1"
#     os.environ["RAY_enable_oom_killer"] = "1"
#     os.environ["RAY_memory_usage_threshold"] = "0.95"

#     # Resource thresholds
#     memory_threshold = 90  # in percentage
#     cpu_threshold = 90  # in percentage
#     active_tasks = []
#     completed_results = []

#     # Start Ray
#     context = ray.init(
#         num_cpus=cfg.parallel.n_workers,
#         dashboard_host="0.0.0.0",
#         dashboard_port=8265,
#         _temp_dir=cfg.parallel._temp_dir,
#     )
#     logger.info(f"Dashboard URL: http://{context.dashboard_url}")

#     # Create (idx, (start, end)) pairs so we can shuffle the execution order
#     # without altering the original idx values, which are used to track the last job
#     # and ensure proper partition handling (e.g., quarry blasts in time zones).
#     indexed_partitions = list(enumerate(cfg.parallel.time_partitions, start=0))
#     random.shuffle(indexed_partitions)

#     # Launch tasks dynamically
#     for idx, (start, end) in indexed_partitions:
#         while True:
#             # Monitor system resources
#             mem_usage = psutil.virtual_memory().percent
#             cpu_usage = psutil.cpu_percent(interval=0.1)

#             if mem_usage < memory_threshold and cpu_usage < cpu_threshold:
#                 # Launch the task
#                 logger.info(
#                     f"Launching task {idx} [{start} -- {end}] (Memory: {mem_usage}%, CPU: {cpu_usage}%)"
#                 )
#                 active_tasks.append(run_dbclust_task.remote(cfg, idx))
#                 break
#             else:
#                 logger.warning(
#                     f"High resource usage (Memory: {mem_usage}%, CPU: {cpu_usage}%) - Waiting..."
#                 )
#                 time.sleep(1)

#             # Check if tasks are completed
#             if len(active_tasks) >= cfg.parallel.n_workers:
#                 ready, not_ready = ray.wait(active_tasks, num_returns=1)
#                 results = ray.get(ready)
#                 completed_results.extend(results)
#                 active_tasks = not_ready

#     # Collect remaining tasks
#     while active_tasks:
#         ready, active_tasks = ray.wait(active_tasks, num_returns=1)
#         results = ray.get(ready)
#         completed_results.extend(results)

#     logger.info("DBClust completed!")
#     ray.shutdown()
#     return completed_results

def main():
    # default logger
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    logger = logging.getLogger("dbclust")
    logger.setLevel(logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--conf",
        default=None,
        dest="configfile",
        help="yaml configuration file.",
        type=str,
    )
    parser.add_argument(
        "-p",
        "--profile",
        default=None,
        dest="velocity_profile_name",
        help="velocity profile name to use",
        type=str,
    )
    parser.add_argument(
        "-l",
        "--loglevel",
        default="INFO",
        dest="loglevel",
        help="loglevel (debug,warning,info,error)",
        type=str,
    )
    args = parser.parse_args()
    if not args.configfile:
        parser.print_help()
        sys.exit(255)

    numeric_level = getattr(logging, args.loglevel.upper(), None)
    if not numeric_level:
        logger.error("Invalid loglevel '%s' !", args.loglevel.upper())
        logger.error("loglevel should be: debug,warning,info,error.")
        sys.exit(255)
    logger.setLevel(numeric_level)

    # Get configuration from yaml file
    # numerous initializations have been already carried out
    cfg = DBClustConfig(args.configfile)
    cfg.show()

    if cfg.parallel.n_workers == 1:
        results = []
        for idx, (start, end) in enumerate(cfg.parallel.time_partitions, start=0):
            results.append(dbclust(cfg=cfg, df=None, job_index=idx))
    else:
        results = run_with_ray(cfg)  # change the locator accordingly

    # update spatialite view
    if cfg.catalog.enable_sqlite:
        conn = sqlite3.connect(cfg.catalog.sqlite_db_fullpath)
        if conn:
            logger.info("Connected to SQLite database to update view.")
            refresh_event_coordinates_view(conn)
            conn.close()
        else:
            logger.error("Failed to connect to SQLite database.")

    print(results)
    # flush stdout and stderr to have the log in the right order
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(0)

if __name__ == "__main__":
    main()
