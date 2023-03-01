#!/usr/bin/env python
import sys
import os
import logging
import argparse
import yaml
import numpy as np
import pandas as pd
from obspy import Catalog, UTCDateTime

from dbclust.phase import import_phases, import_eqt_phases
from dbclust.clusterize import (
    Clusterize,
    filter_out_cluster_with_common_phases,
)
from dbclust.localization import NllLoc, show_event

# default logger
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger("dbclust")
logger.setLevel(logging.INFO)


def ymljoin(loader, node):
    seq = loader.construct_sequence(node)
    return "".join([str(i) for i in seq])


def yml_read_config(filename):
    with open(filename, "r") as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
    return cfg


if __name__ == "__main__":
    logger.setLevel(logging.DEBUG)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--conf",
        default=None,
        dest="configfile",
        help="yaml configuration file.",
        type=str,
    )
    args = parser.parse_args()
    if not args.configfile:
        parser.print_help()
        sys.exit(255)

    yaml.add_constructor("!join", ymljoin)
    cfg = yml_read_config(args.configfile)

    ################ CONFIG ################

    file_cfg = cfg["file"]
    time_cfg = cfg["time"]
    pick_cfg = cfg["pick"]
    cluster_cfg = cfg["cluster"]
    nll_cfg = cfg["nll"]
    reloc_cfg = cfg["relocation"]
    catalog_cfg = cfg["catalog"]

    # path definition
    OBS_PATH = file_cfg["obs_path"]
    QML_PATH = file_cfg["qml_path"]
    TMP_PATH = file_cfg["tmp_path"]

    for dir in [OBS_PATH, QML_PATH, TMP_PATH]:
        os.makedirs(dir, exist_ok=True)

    # import *phaseNet* or *eqt* csv picks file
    picks_type = file_cfg["picks_type"]
    assert picks_type in ["eqt", "phasenet"]
    picks_file = file_cfg["picks_csv"]

    #
    # Time
    #
    if "date_begin" in time_cfg:
        date_begin = time_cfg["date_begin"]
    else:
        date_begin = None

    if "date_end" in time_cfg:
        date_end = time_cfg["date_end"]
    else:
        date_end = None

    if "time_window" in time_cfg:
        time_window = time_cfg["time_window"]
    else:
        time_window = 7  # min

    #
    # Picks
    #
    # pick uncertainty
    P_uncertainty = pick_cfg["P_uncertainty"]
    S_uncertainty = pick_cfg["S_uncertainty"]

    # import only phase with proba >=phase_proba_threshold
    P_proba_threshold = pick_cfg["P_proba_threshold"]
    S_proba_threshold = pick_cfg["S_proba_threshold"]

    #
    # parameters for dbscan clustering
    #
    min_size = cluster_cfg["min_size"]
    min_station_count = cluster_cfg["min_station_count"]
    average_velocity = cluster_cfg["average_velocity"]
    max_search_dist = cluster_cfg["max_search_dist"]

    if "pre_computed_tt_matrix" in cluster_cfg:
        pre_computed_tt_matrix = cluster_cfg["pre_computed_tt_matrix"]
        tt_matrix_save = True
    else:
        pre_computed_tt_matrix = None
        tt_matrix_save = False

    #
    # NonLinLoc
    #
    # NLLoc binary
    nllocbin = nll_cfg["bin"]

    # full path NLL template
    nlloc_template = nll_cfg["nll_template"]
    nlloc_times_path = nll_cfg["nll_time_path"]
    nll_min_phase = nll_cfg["nll_min_phase"]

    # file to keep track of SCNL when exporting to NLL
    nll_channel_hint = nll_cfg["nll_channel_hint"]

    #
    # Relocation
    #
    double_pass = reloc_cfg["double_pass"]
    P_time_residual_threshold = reloc_cfg["P_time_residual_threshold"]
    S_time_residual_threshold = reloc_cfg["S_time_residual_threshold"]

    #
    # Catalog
    #
    qml_base_filename = catalog_cfg["qml_base_filename"]
    event_flush_count = catalog_cfg["event_flush_count"]

    ########################################

    logger.info(f"Opening {picks_file} file.")
    try:
        if picks_type == "eqt":
            df = pd.read_csv(
                picks_file, parse_dates=["p_arrival_time", "s_arrival_time"]
            )
            #
            df["phase_time"] = df[["p_arrival_time", "s_arrival_time"]].min(axis=1)
        else:
            try:
                # official PhaseNet
                df = pd.read_csv(picks_file, parse_dates=["phase_time"])
            except:
                # PhaseNetSDS
                df = pd.read_csv(picks_file, parse_dates=["time"])
                df.rename(
                    columns={
                        "seedid": "station_id",
                        "phasename": "phase_type",
                        "time": "phase_time",
                        "probability": "phase_score",
                    },
                    inplace=True,
                )
            df["phase_time"] = pd.to_datetime(df["phase_time"], utc=True)
            df.sort_values(by=["phase_time"], inplace=True)
    except Exception as e:
        logger.error(e)
        sys.exit()
    logger.info(f"Read {len(df)} phases.")

    # Time filtering
    if date_begin and date_end:
        date_begin = pd.to_datetime(date_begin, utc=True)
        date_end = pd.to_datetime(date_end, utc=True)
        df = df[df["phase_time"] >= date_begin]
        df = df[df["phase_time"] <= date_end]

    tmin = df["phase_time"].min()
    tmax = df["phase_time"].max()
    time_periods = (
        pd.date_range(tmin, tmax, freq=f"{time_window}min").to_series().to_list()
    )
    time_periods += [pd.to_datetime(tmax)]
    logger.info(f"Splitting dataset in {len(time_periods)-1} chunks.")

    # print(time_periods)
    # print(time_periods[:-2], time_periods[1:])

    # configure locator
    locator = NllLoc(
        nllocbin,
        nlloc_times_path,
        nlloc_template,
        nll_channel_hint=nll_channel_hint,
        tmpdir=TMP_PATH,
        double_pass=double_pass,
        P_time_residual_threshold=P_time_residual_threshold,
        S_time_residual_threshold=S_time_residual_threshold,
        nll_min_phase=nll_min_phase,
        verbose=False,
    )

    # process independently each time period
    # fixme: add overlapp between time period
    part = 0
    last_saved_event_count = 0
    previous_myclust = Clusterize()

    for i, (from_time, to_time) in enumerate(
        zip(time_periods[:-1], time_periods[1:]), start=1
    ):
        begin = from_time - np.timedelta64(1, "m")
        end = to_time

        df_subset = df[(df["phase_time"] >= begin) & (df["phase_time"] < end)]

        logger.info("")
        logger.info("")
        logger.info(
            # f"Time window extraction #{i}/{len(time_periods)-1} picks from {from_time} to {to_time}."
            f"Time window extraction #{i}/{len(time_periods)-1} picks from {begin} to {end}."
        )

        if not len(df_subset):
            logger.info(f"Skipping clustering {len(df_subset)} phases.")
            continue

        logger.info(f"Clustering {len(df_subset)} phases.")

        # get phaseNet picks from dataframe
        if picks_type == "eqt":
            phases = import_eqt_phases(df_subset, P_proba_threshold, S_proba_threshold)
        else:
            phases = import_phases(df_subset, P_proba_threshold, S_proba_threshold)

        # find clusters
        myclust = Clusterize(
            phases,
            max_search_dist,
            min_size,
            average_velocity,
            min_station_count=min_station_count,
            P_uncertainty=P_uncertainty,
            S_uncertainty=S_uncertainty,
            tt_maxtrix_fname=pre_computed_tt_matrix,
            tt_matrix_save=tt_matrix_save,
        )
        if myclust.n_clusters == 0:
            previous_myclust = Clusterize(
                [], max_search_dist, min_size, average_velocity
            )
            continue

        # check if some clusters in this round share some phases
        # with clusters from the previous round
        # (as some phases come from the overlapped zone)
        logger.info("Check cluster related to the same event.")
        (
            previous_myclust,
            myclust,
            nb_cluster_removed,
        ) = filter_out_cluster_with_common_phases(previous_myclust, myclust, 6)

        # This is the last round: merge previous_myclust and myclust
        if i == len(time_periods) - 1:
            logger.info("Last round, merging all remaining clusters.")
            previous_myclust.merge(myclust)

        # Now, process previous_myclust and wait next round to process myclust
        # write each cluster to nll obs files
        my_obs_path = os.path.join(OBS_PATH, f"{i}")
        previous_myclust.generate_nllobs(my_obs_path)
        previous_myclust = myclust

        # localize each cluster
        # all locs are automaticaly appended to the locator's catalog
        # Dask // version
        clustcat = locator.dask_get_localisations_from_nllobs_dir(
            my_obs_path, append=True
        )
        # sequential version
        # clustcat = locator.get_localisations_from_nllobs_dir(my_obs_path, append=True)

        if len(clustcat) > 0:
            for event in sorted(
                clustcat.events, key=lambda e: e.preferred_origin().time
            ):
                show_event(event, "****")
        else:
            continue

        # write partial qml file and clean catalog from memory
        if (last_saved_event_count) > event_flush_count:
            partial_qml = os.path.join(QML_PATH, f"{qml_base_filename}-{part}.qml")
            logger.info(f"Writing {len(locator.catalog)} events in {partial_qml}")
            locator.catalog.write(partial_qml, format="QUAKEML")
            locator.catalog.clear()
            last_saved_event_count = 0
            part += 1
        else:
            last_saved_event_count += len(clustcat)

    # Write last events
    partial_qml = os.path.join(QML_PATH, f"{qml_base_filename}-{part}.qml")
    logger.info(f"Writing {len(locator.catalog)} events in {partial_qml}")
    locator.catalog.write(partial_qml, format="QUAKEML")
