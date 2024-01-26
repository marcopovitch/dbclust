#!/usr/bin/env python
import sys
import os
import logging
import argparse
import numpy as np
import pandas as pd
import tempfile
from dataclasses import asdict
from typing import Optional
from icecream import ic

from dask.distributed import Client, LocalCluster

from config import Config, get_config_from_file
from phase import import_phases, import_eqt_phases
from clusterize import (
    Clusterize,
    get_picks_from_event,
    merge_cluster_with_common_phases,
    feed_picks_probabilities,
    feed_picks_event_ids,
)
from dbclust2pyocto import dbclust2pyocto
from localization import NllLoc, show_event
from quakeml import make_readable_id, feed_distance_from_preloc_to_pref_origin
import pyocto
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


def unload_picks_list(df1, picks):
    # format picks coming from events like the ones used as input for dbclust
    df2 = pd.DataFrame(picks, columns=["station_id", "phase_type", "phase_time"])
    df2["station_id"] = df2["station_id"].map(lambda x: ".".join(x.split(".")[:2]))
    df2["phase_time"] = pd.to_datetime(df2["phase_time"].map(lambda x: str(x)))
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


def get_locator_from_config(cfg):
    locator = NllLoc(
        cfg.nll.nlloc_bin,
        cfg.nll.scat2latlon_bin,
        cfg.nll.time_path,
        cfg.nll.template_path,
        nll_channel_hint=cfg.nll.channel_hint,
        tmpdir=cfg.file.tmp_path,
        double_pass=cfg.relocation.double_pass,
        P_time_residual_threshold=cfg.relocation.P_time_residual_threshold,
        S_time_residual_threshold=cfg.relocation.S_time_residual_threshold,
        dist_km_cutoff=cfg.relocation.dist_km_cutoff,
        nll_min_phase=cfg.nll.min_phase,
        min_station_with_P_and_S=cfg.cluster.min_station_with_P_and_S,
        quakeml_settings=asdict(cfg.quakeml),
        nll_verbose=cfg.nll.verbose,
        keep_scat=cfg.nll.enable_scatter,
        log_level=logger.level,
    )
    return locator


def get_clusturize_from_config(cfg, phases=None):
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
        log_level=logger.level,
    )
    return myclust


def dbclust(
    cfg: Config,
    df: Optional[pd.DataFrame] = None,
) -> None:
    """Detect and localize events given picks

    Args:
        cfg (Config): dbclust parameters and data
        df (Optional[pd.DataFrame], optional): if defined override picks from cfg
    """
    # keep track of each time period processed
    part = 0
    last_saved_event_count = 0
    last_round = False

    # Instantiate a new tool (but empty) to get clusters
    previous_myclust = get_clusturize_from_config(cfg, phases=None)

    # get a locator
    locator = get_locator_from_config(cfg)

    # time window split
    if not df:
        df = cfg.pick.df
    tmin = df["phase_time"].min()
    tmax = df["phase_time"].max()
    time_periods = (
        pd.date_range(tmin, tmax, freq=f"{cfg.time.time_window}min")
        .to_series()
        .to_list()
    )
    time_periods += [pd.to_datetime(tmax)]
    logger.info(f"Splitting dataset in {len(time_periods)-1} chunks.")

    # start time looping
    picks_to_remove = []
    for i, (from_time, to_time) in enumerate(
        zip(time_periods[:-1], time_periods[1:]), start=1
    ):
        logger.debug("================================================")
        logger.debug("")
        logger.debug("")

        # keep an overlap
        begin = from_time - np.timedelta64(cfg.time.overlap_window, "s")
        end = to_time
        logger.info(
            # f"Time window extraction #{i}/{len(time_periods)-1} picks from {from_time} to {to_time}."
            f"Time window extraction #{i}/{len(time_periods)-1} picks from {begin} to {end}."
        )

        # Extract picks on this time period
        df_subset = df[(df["phase_time"] >= begin) & (df["phase_time"] < end)]
        if not len(df_subset):
            logger.info(f"Skipping clustering {len(df_subset)} phases.")
            continue
        logger.info(f"Clustering {len(df_subset)} phases.")

        # to prevents extra event, remove from current picks list,
        # picks previously associated with events on the previous iteration
        logger.debug(f"test len(df_subset) before = {len(df_subset)}")
        if len(picks_to_remove):
            logger.info(f"before unload picks: pick length is {len(df_subset)}")
            df_subset = unload_picks_list(df_subset, picks_to_remove)
            logger.info(f"after unload picks: pick length is {len(df_subset)}")
            picks_to_remove = []

        # Import picks
        if cfg.pick.type == "eqt":
            phases = import_eqt_phases(
                df_subset,
                cfg.pick.P_proba_threshold,
                cfg.pick.S_proba_threshold,
            )
        else:
            phases = import_phases(
                df_subset,
                cfg.pick.P_proba_threshold,
                cfg.pick.S_proba_threshold,
                cfg.station.info_sta,
            )
            if logger.level == logging.DEBUG:
                for p in phases:
                    p.show_all()

        if logger.level == logging.DEBUG:
            logger.info("previous_myclust:")
            previous_myclust.show_clusters()

        # Instantiate a new tool to get clusters
        myclust = get_clusturize_from_config(cfg, phases=phases)

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
        if i == (len(time_periods) - 1):
            last_round = True
            logger.info("Last round, merging all remaining clusters.")
            previous_myclust.merge(myclust)

        if cfg.pyocto.current_model:
            previous_myclust = dbclust2pyocto(
                previous_myclust,
                cfg.pyocto.current_model.associator,
                cfg.pyocto.velocity_model,
                cfg.cluster.min_picks_common,
                log_level=logger.level,
            )

        # Now, process previous_myclust and wait next round to process myclust
        # write each cluster to nll obs files
        with tempfile.TemporaryDirectory(dir=cfg.file.obs_path) as TMP_OBS_PATH:
            my_obs_path = os.path.join(TMP_OBS_PATH, f"{i}")
            previous_myclust.generate_nllobs(my_obs_path)

            # localize each cluster
            # all locs are automatically appended to the locator's catalog
            # force to cleanup all files generated by Nonlinloc
            logger.info("Starting localization")
            with tempfile.TemporaryDirectory(
                dir=cfg.file.tmp_path
            ) as tmpdir_automaticaly_cleaned:
                locator.tmpdir = tmpdir_automaticaly_cleaned

                # Dask // version
                clustcat = locator.dask_get_localisations_from_nllobs_dir(
                    my_obs_path, append=True
                )

                # thread // version
                # clustcat = locator.multiproc_get_localisations_from_nllobs_dir(
                #     my_obs_path, append=True
                # )

                # clustcat = locator.processes_get_localisations_from_nllobs_dir(
                #     my_obs_path, append=True
                # )

                # sequential version
                # clustcat = locator.get_localisations_from_nllobs_dir(
                #    my_obs_path, append=True
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
                    event.event_type != "no existing"
                    and not last_round
                    and first_pick_time < next_begin
                    and last_pick_time >= next_begin
                ):
                    # First pick time is before overlapped zone
                    # but some others picks are inside it.
                    # Assuming the overlapped zone is large enough
                    # to have a complete event, remove those picks,
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
        clustcat = make_readable_id(
            clustcat, cfg.quakeml.event_prefix, cfg.quakeml.smi_base
        )

        # prepare next round
        previous_myclust = myclust

        # write partial qml file and clean catalog from memory
        if (last_saved_event_count) > cfg.catalog.event_flush_count:
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
    partial_qml = os.path.join(
        cfg.catalog.path, f"{cfg.catalog.qml_base_filename}-{part}.qml"
    )
    logger.info(f"Writing {len(locator.catalog)} events in {partial_qml}")
    locator.catalog.write(partial_qml, format="QUAKEML")


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

    cluster = LocalCluster()
    client = Client(cluster)
    logger.info(f"Dask dashboard url: {client.dashboard_link}")

    numeric_level = getattr(logging, args.loglevel.upper(), None)
    if not numeric_level:
        logger.error("Invalid loglevel '%s' !", args.loglevel.upper())
        logger.error("loglevel should be: debug,warning,info,error.")
        sys.exit(255)
    logger.setLevel(numeric_level)

    # Get configuration from yaml file
    # numerous initializations have been already carried out
    cfg = get_config_from_file(args.configfile, verbose=False)
    ic(cfg)

    # Warning: change the output name for multiprocessing

    #dbclust(cfg, cfg.pick.df)
    dbclust(cfg)
