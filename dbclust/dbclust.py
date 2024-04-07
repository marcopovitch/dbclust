#!/usr/bin/env python
import argparse
import logging
import multiprocessing
import os
import shutil
import sys
import tempfile
import warnings
from dataclasses import asdict
from functools import partial
from typing import Optional

import dask
import numpy as np
import pandas as pd
import ray
from clusterize import Clusterize
from clusterize import feed_picks_event_ids
from clusterize import feed_picks_probabilities
from clusterize import get_picks_from_event
from clusterize import merge_cluster_with_common_phases
from config import DBClustConfig
from dask.distributed import Client
from dask.distributed import LocalCluster
from db import duckdb_init
from dbclust2pyocto import dbclust2pyocto
from icecream import ic
from localization import NllLoc
from localization import show_event
from phase import import_phases
from quakeml import feed_distance_from_preloc_to_pref_origin
from quakeml import make_readable_id
from ray.util.multiprocessing import Pool

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
        cfg.nll.template_path,
        tmpdir=cfg.file.tmp_path,
        double_pass=cfg.relocation.double_pass,
        P_time_residual_threshold=cfg.relocation.P_time_residual_threshold,
        S_time_residual_threshold=cfg.relocation.S_time_residual_threshold,
        dist_km_cutoff=cfg.relocation.dist_km_cutoff,
        # use_deactivated_arrivals=cfg.relocation.use_deactivated_arrivals,  ## not yet in config
        keep_manual_picks=cfg.relocation.keep_manual_picks,
        nll_min_phase=cfg.nll.min_phase,
        min_station_with_P_and_S=cfg.cluster.min_station_with_P_and_S,
        quakeml_settings=asdict(cfg.quakeml),
        nll_verbose=cfg.nll.verbose,
        keep_scat=cfg.nll.enable_scatter,
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
        max_search_dist=cfg.cluster.max_search_dist,
        P_uncertainty=cfg.pick.P_uncertainty,
        S_uncertainty=cfg.pick.S_uncertainty,
        tt_matrix_fname=cfg.cluster.pre_computed_tt_matrix_file,
        tt_matrix_save=cfg.cluster.tt_matrix_save,
        zones=cfg.zones,
        log_level=log_level,
    )
    return myclust


def dbclust_test(
    cfg: DBClustConfig,
    df: Optional[pd.DataFrame] = pd.DataFrame(),
    job_index: Optional[int] = None,
) -> None:

    # Time blocks
    if job_index is not None:
        start, stop = cfg.parallel.time_partitions[job_index]
    else:
        start = cfg.pick.start
        stop = cfg.pick.end

    msg = "test started."
    ic(msg, job_index, start, stop, df)
    return True


def dbclust(
    cfg: DBClustConfig,
    df: Optional[pd.DataFrame] = pd.DataFrame(),
    job_index: Optional[int] = None,
) -> None:
    """Detect and localize events given picks

    Args:
        cfg (Config): dbclust parameters and data
        df (Optional[pd.DataFrame], optional): use df if defined rather than picks from cfg
        job_index (int): job index, None if in sequential mode
    """

    # Time blocks
    if job_index is not None:
        start, stop = cfg.parallel.time_partitions[job_index]
        if job_index == len(cfg.parallel.time_partitions) - 1:
            # get event in the overlapped zone
            last_job = True
        else:
            # don't get event in the overlapped zone
            last_job = False
    else:
        start = cfg.pick.start
        stop = cfg.pick.end
        # get event in the overlapped zone during the last time_periods round
        last_job = True

    msg = "started."
    ic(msg, job_index, start, stop)

    if df is None or df.empty:
        # Uses duckdb
        con = duckdb_init(cfg.pick.filename, cfg.pick.type)
    else:
        # Uses the pandas Dataframe given as function argument.
        con = None

    time_periods = (
        pd.date_range(start, stop, freq=f"{cfg.time.time_window}min")
        .to_series()
        .to_list()
    )
    time_periods += [pd.to_datetime(stop)]
    logger.info(f"[{job_index}] Splitting dataset in {len(time_periods)-1} chunks.")

    # Instantiate a new tool (but empty) to get clusters
    previous_myclust = get_clusterize_from_config(cfg, phases=None)

    # get a locator
    locator = get_locator_from_config(cfg)

    # keep track of each time period processed
    part = 0
    last_saved_event_count = 0
    last_round = False  # over time_periods
    picks_to_remove = []

    # start time looping
    overlap_timedelta = pd.Timedelta(cfg.time.overlap_window, "s")
    for i, (begin, end) in enumerate(
        zip(time_periods[:-1], time_periods[1:]),
        start=1,
    ):
        logger.debug("")
        logger.debug("================================================")
        logger.debug("")

        # add the time overlap
        end += overlap_timedelta

        logger.info(
            f"[{job_index}] Time window extraction #{i}/{len(time_periods)-1} picks from {begin} to {end}."
        )

        # Extract picks on this time period
        if con:
            rqt = f"SELECT * FROM PICKS WHERE phase_time BETWEEN '{begin}' AND '{end}'"
            df_subset = con.sql(rqt).fetchdf()
            df_subset["phase_time"] = df_subset["phase_time"].dt.tz_localize("UTC")
        else:
            df_subset = df[(df["phase_time"] >= begin) & (df["phase_time"] < end)]

        if df_subset.empty:
            logger.info(f"[{job_index}] Skipping clustering {len(df_subset)} phases.")
            continue

        # remove blacklisted stations
        if cfg.station.blacklist:
            for b in cfg.station.blacklist:
                df_subset = df_subset[
                    ~df_subset["station_id"].str.contains(b, regex=True)
                ]

        logger.info(f"[{job_index}] Starting clustering with {len(df_subset)} phases.")

        # to prevents extra event, remove from current picks list,
        # picks previously associated with events on the previous iteration
        logger.debug(f"test len(df_subset) before = {len(df_subset)}")
        if len(picks_to_remove):
            logger.info(
                f"[{job_index}] before unload picks: pick length is {len(df_subset)}"
            )
            df_subset = unload_picks_list(df_subset, picks_to_remove)
            logger.info(
                f"[{job_index}] after unload picks: pick length is {len(df_subset)}"
            )
            picks_to_remove = []

        # remove picks in respect to the number of station per minutes in df_subset
        if cfg.station.frequency_threshold:
            total_duration_in_minutes = cfg.time.time_window
            grouped = df_subset.groupby("station_id")
            station_counts = grouped.size()
            frequencies = station_counts / total_duration_in_minutes
            station_ids_to_keep = frequencies[
                frequencies <= cfg.station.frequency_threshold
            ].index
            df_subset = df_subset[df_subset["station_id"].isin(station_ids_to_keep)]
            ic(
                cfg.station.frequency_threshold,
                frequencies[frequencies >= cfg.station.frequency_threshold],
            )

        logger.debug(f"test len(df_subset) after = {len(df_subset)}")

        # Import picks
        phases = import_phases(
            df_subset,
            cfg.pick.P_proba_threshold,
            cfg.pick.S_proba_threshold,
            cfg.pick.P_uncertainty,
            cfg.pick.S_uncertainty,
            cfg.station.info_sta,
        )
        if logger.level == logging.DEBUG:
            for p in phases:
                p.show_all()

        # clean up
        del df_subset

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
        (
            previous_myclust,
            myclust,
            nb_cluster_removed,
        ) = merge_cluster_with_common_phases(
            previous_myclust, myclust, cfg.cluster.min_picks_common
        )

        # This is the last round: merge previous_myclust and myclust
        if i == (len(time_periods) - 1) and last_job == True:
            last_round = True
            logger.info("Last round, merging all remaining clusters.")
            previous_myclust.merge(myclust)

        if cfg.pyocto.current_model:
            previous_myclust = dbclust2pyocto(
                previous_myclust,
                cfg.pyocto.default_model_name,
                cfg.pyocto.current_model.associator,
                cfg.pyocto.velocity_model,
                cfg.cluster.min_picks_common,
                log_level=logger.level,
            )

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
            logger.info("Starting localization")
            with MyTemporaryDirectory(
                dir=cfg.file.tmp_path,
                delete=cfg.file.automatic_cleanup_tmp,
            ) as tmpdir_automaticaly_cleaned:
                locator.tmpdir = tmpdir_automaticaly_cleaned

                # sequential version
                clustcat = locator.get_localisations_from_nllobs_dir(
                    my_obs_path, picks=nll_picks, append=True
                )

                # multiproc  // version with pool
                # clustcat = locator.multiproc_get_localisations_from_nllobs_dir(
                #     my_obs_path, append=True
                # )

                # Ray pool // version with Pool
                # clustcat = locator.ray_multiproc_get_localisations_from_nllobs_dir(
                #    my_obs_path, append=True
                # )

                # Ray // version
                # clustcat = locator.ray_get_localisations_from_nllobs_dir(
                #     my_obs_path, append=True
                # )

                # Dask // version
                # clustcat = locator.dask_get_localisations_from_nllobs_dir(
                #     my_obs_path, append=True
                # )

                # thread // version
                # clustcat = locator.multiproc_get_localisations_from_nllobs_dir(
                #     my_obs_path, append=True
                # )

                # clustcat = locator.processes_get_localisations_from_nllobs_dir(
                #     my_obs_path, append=True
                # )

                # fixme
                # handle scat file here before the directory is deleted
                if cfg.nll.enable_scatter:
                    logger.warning("FIXME: scatter file not yet handled !")

        if len(clustcat) > 0:
            for event in sorted(
                clustcat.events, key=lambda e: e.preferred_origin().time
            ):
                next_begin = end - np.timedelta64(cfg.time.overlap_window, "s")

                origin = event.preferred_origin()
                first_station, first_phase, first_pick_time = get_picks_from_event(
                    event, origin, None
                ).pop(0)
                last_station, last_phase, last_pick_time = get_picks_from_event(
                    event, origin, None
                ).pop(-1)

                logger.debug(
                    "First pick is: %s, last pick is: %s, overlapped zone starts: %s, next overlapped zone starts: %s"
                    % (first_pick_time, last_pick_time, begin, next_begin)
                )

                if not last_round and first_pick_time >= next_begin:
                    # Event first pick is in overlapped zone,
                    # remove this event and wait the next iteration
                    # as this event will be recreated.
                    show_event(event, "***D")
                    logger.debug(
                        f"Select event in overlapped zone to be (D)eleted ({event.resource_id.id})"
                    )
                    locator.catalog.events.remove(event)
                    locator.nb_events = len(locator.catalog)
                    clustcat = locator.catalog
                elif (
                    event.event_type != "not existing"
                    and not last_round
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
                    logger.debug(
                        f"Select a real event between normal and overlapped zone where picks must be (P)runed ({event.resource_id.id})"
                    )
                    picks_to_remove = []
                    for origin in event.origins:
                        picks_to_remove += get_picks_from_event(
                            event, origin, next_begin
                        )
                else:
                    show_event(event, "****")

        # write into qml/comments picks probabilities and event_ids where picks are coming from
        feed_picks_probabilities(clustcat, previous_myclust.clusters)
        feed_picks_event_ids(clustcat, previous_myclust.clusters)

        # set to manual picks from agency
        # set to automatic pick from phasenet
        # feed_picks_event_evaluation_mode(clustcat)

        # write into qml/comments distance from preferred origin and prelocalization
        feed_distance_from_preloc_to_pref_origin(clustcat)

        # transform ids to a more human readable thing !
        # clustcat = make_readable_id(
        #     clustcat, cfg.quakeml.event_prefix, cfg.quakeml.smi_base
        # )

        # prepare next round
        previous_myclust = myclust

        # write partial qml file and clean catalog from memory
        if (last_saved_event_count) > cfg.catalog.event_flush_count:
            if job_index != None:
                partial_qml = os.path.join(
                    cfg.catalog.path,
                    f"{cfg.catalog.qml_base_filename}-{job_index}-{part}.qml",
                )
            else:
                partial_qml = os.path.join(
                    cfg.catalog.path, f"{cfg.catalog.qml_base_filename}-{part}.qml"
                )

            logger.info(f"Writing {len(locator.catalog)} events in {partial_qml}")
            locator.catalog.write(partial_qml, format="QUAKEML")
            locator.catalog.clear()
            last_saved_event_count = 0
            part += 1
        else:
            # last_saved_event_count += len(clustcat)
            last_saved_event_count += len(locator.catalog)

    # Write last events
    if job_index != None:
        partial_qml = os.path.join(
            cfg.catalog.path,
            f"{cfg.catalog.qml_base_filename}-{job_index}-{part}.qml",
        )
    else:
        partial_qml = os.path.join(
            cfg.catalog.path, f"{cfg.catalog.qml_base_filename}-{part}.qml"
        )
    logger.info(f"Writing {len(locator.catalog)} events in {partial_qml}")
    locator.catalog.write(partial_qml, format="QUAKEML")

    # if con:
    #    con.close()

    return True


def process_task(cfg, job_arg):
    job_index = job_arg[0]
    start, end = job_arg[1]
    dbclust(cfg=cfg, df=None, job_index=job_index)


def run_with_multiproc(cfg: DBClustConfig):
    func = partial(process_task, cfg)
    with multiprocessing.Pool(cfg.parallel.n_workers) as pool:
        results = pool.map(func, enumerate(cfg.parallel.time_partitions, start=0))
    return results


def run_with_ray_multiproc(cfg: DBClustConfig):
    func = partial(process_task, cfg)
    with Pool(cfg.parallel.n_workers) as pool:
        results = pool.map(func, enumerate(cfg.parallel.time_partitions, start=0))
    return results


def run_with_dask(cfg: DBClustConfig):
    # Cluster initialization
    cluster = LocalCluster(
        processes=True,
        threads_per_worker=1,
        n_workers=cfg.parallel.n_workers,
        # dashboard_address="10.0.1.40:8787",
        # dashboard_address=None,
        # death_timeout=120,
    )
    client = Client(cluster)
    logger.info(f"Dask running on {cfg.parallel.n_workers} cpu(s)")
    logger.info(f"Dask dashboard url: {client.dashboard_link}")
    dask.config.set({"distributed.scheduler.work-stealing": True})

    # Dask stuff
    delayed_tasks = [
        dask.delayed(dbclust)(cfg=cfg, job_index=idx)
        for idx, (start, end) in enumerate(cfg.parallel.time_partitions, start=0)
    ]

    # Start tasks
    results = dask.compute(*delayed_tasks)
    dask.distributed.wait(results)
    logger.info("DBClust completed !")
    return results


# Ray tasks
@ray.remote
def run_dbclust_task(cfg, job_index):
    return dbclust(cfg=cfg, job_index=job_index)


def run_with_ray(cfg: DBClustConfig):

    # os.environ["RAY_DEDUP_LOGS"] = "0"
    os.environ["RAY_COLOR_PREFIX"] = "1"

    # Start Ray
    context = ray.init(
        num_cpus=cfg.parallel.n_workers,
        # include_dashboard=True,
        # dashboard_host="10.0.1.40",
        # dashboard_port=8087,
    )
    logger.info(f" http://{context.dashboard_url}")

    ray_tasks = [
        run_dbclust_task.remote(cfg, idx)
        for idx, (start, end) in enumerate(cfg.parallel.time_partitions, start=0)
    ]

    # Get results
    results = ray.get(ray_tasks)
    logger.info("DBClust completed !")
    return results


if __name__ == "__main__":
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
        # results = dbclust(cfg)
        # results = run_with_multiproc(cfg)
        # results = run_with_dask(cfg)  # change the locator accordingly
        # results = run_with_ray_multiproc(cfg)
        results = run_with_ray(cfg)  # change the locator accordingly

    print(results)
