#!/usr/bin/env python
import sys
import os
import logging
import argparse
import yaml
import pandas as pd

from obspy import Catalog

from dbclust.phase import import_phases, import_eqt_phases
from dbclust.clusterize import Clusterize
from dbclust.localization import NllLoc

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
    pick_cfg = cfg["pick"]
    cluster_cfg = cfg["cluster"]
    catalog_cfg = cfg["catalog"]
    nll_cfg = cfg["nll"]

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

    # NLLoc binary
    nllocbin = nll_cfg["bin"]

    # full path NLL template
    nlloc_template = nll_cfg["nll_template"]

    # file to keep track of SCNL when exporting to NLL
    nll_channel_hint = nll_cfg["nll_channel_hint"]

    # parameters for dbscan clustering
    min_size = cluster_cfg["min_size"]
    min_station_count = cluster_cfg["min_station_count"]
    average_velocity = cluster_cfg["average_velocity"]
    max_search_dist = cluster_cfg["max_search_dist"]

    # pick uncertainty
    P_uncertainty = pick_cfg["P_uncertainty"]
    S_uncertainty = pick_cfg["S_uncertainty"]

    # import only phase with proba >=phase_proba_threshold
    phase_proba_threshold = pick_cfg["phase_proba_threshold"]

    max_standard_error = catalog_cfg["max_standard_error"]
    max_azimuthal_gap = catalog_cfg["max_azimuthal_gap"]

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
                df.rename(columns={ "seedid": "station_id",
                                    "phasename": "phase_type",
                                    "time": "phase_time",
                                    "probability": "phase_score"},
                          inplace=True,
                )
    except Exception as e:
        logger.error(e)
        sys.exit()
    logger.info(f"Read {len(df)} phases.")

    tmin = df["phase_time"].min()
    tmax = df["phase_time"].max()
    time_periods = pd.date_range(tmin, tmax, freq="30min").to_series().to_list()
    time_periods +=  [pd.to_datetime(tmax)]
    logger.info(f"Splitting dataset in {len(time_periods)} chunks.")

    my_catalog = Catalog()

    # configure locator
    locator = NllLoc(
        nllocbin,
        nlloc_template,
        nll_channel_hint=nll_channel_hint,
        tmpdir=TMP_PATH,
    )

    # process independently each time period
    # fixme: add overlapp between time period
    for e, (from_time, to_time) in enumerate(zip(time_periods[:-2], time_periods[1:])):
        logger.info("")
        logger.info("")
        logger.info(
            f"Extraction {e}/{len(time_periods)} picks from {from_time} to {to_time}."
        )

        df_subset = df[(df["phase_time"] >= from_time) & (df["phase_time"] < to_time)]

        if not len(df_subset):
            logger.info(f"Skipping clustering {len(df_subset)} phases.")
            continue

        logger.info(f"Clustering {len(df_subset)} phases.")

        # get phaseNet picks from dataframe
        if picks_type == "eqt":
            phases = import_eqt_phases(df_subset, phase_proba_threshold)
        else:
            phases = import_phases(df_subset, phase_proba_threshold)

        # find clusters
        myclust = Clusterize(
            phases,
            max_search_dist,
            min_size,
            average_velocity,
            min_station_count=min_station_count,
            P_uncertainty=P_uncertainty,
            S_uncertainty=S_uncertainty,
        )
        if myclust.n_clusters == 0:
            continue

        # write each cluster to nll obs files
        my_obs_path = os.path.join(OBS_PATH, f"{e}")
        myclust.generate_nllobs(my_obs_path)

        # localize each cluster
        my_qml_path = os.path.join(QML_PATH, f"{e}")
        locs = locator.get_localisations_from_nllobs_dir(my_obs_path, my_qml_path)
        if len(locs.catalog) > 0:
            locs.show_localizations()

        # concatenate individual catalogs
        my_catalog += locs.catalog

    # Write QUAKEML and SC3ML
    logger.info(f"Writing {len(my_catalog)} all.qml and all.sc3ml")
    qml_fname = os.path.join(QML_PATH, f"all.qml")
    my_catalog.write(qml_fname, format="QUAKEML")
    sc3ml_fname = os.path.join(QML_PATH, f"all.sc3ml")
    my_catalog.write(sc3ml_fname, format="SC3ML")

    # to filter out poorly constrained events
    # fixme: add to config file
    logger.info("\nFiltered catalog:")
    my_catalog = my_catalog.filter(
        f"standard_error < {max_standard_error}",
        f"azimuthal_gap < {max_azimuthal_gap}",
        f"used_station_count >= {min_station_count}",
    )

    logger.info(f"Writing {len(my_catalog)} all-filtered.qml and all-filtered.sc3ml")
    qml_fname = os.path.join(QML_PATH, f"all-filtered.qml")
    my_catalog.write(qml_fname, format="QUAKEML")
    sc3ml_fname = os.path.join(QML_PATH, f"all-filtered.sc3ml")
    my_catalog.write(sc3ml_fname, format="SC3ML")
