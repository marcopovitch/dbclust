#!/usr/bin/env python
import sys
import os
import logging
import argparse
import yaml

from dbclust.phase import import_phases, import_eqt_phases 
from dbclust.clusterize import Clusterize
from dbclust.localization import NllLoc

# default logger
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger("dbclust")
logger.setLevel(logging.DEBUG)


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
    assert(picks_type in ["eqt", "phasenet"]) 
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

    # get phaseNet picks
    if picks_type == "eqt":
        phases = import_eqt_phases(picks_file, phase_proba_threshold)
    else:
        phases = import_phases(picks_file, phase_proba_threshold)
    assert phases
    logger.info(f"Read {len(phases)} phases.")

    # find clusters and generate nll obs files
    myclust = Clusterize(phases, max_search_dist, min_size, average_velocity)
    myclust.show_clusters()
    myclust.show_noise()
    myclust.generate_nllobs(OBS_PATH, min_station_count, P_uncertainty, S_uncertainty)

    # localize each cluster
    locs = NllLoc().get_catalog_from_nllobs_dir(
        OBS_PATH,
        QML_PATH,
        nlloc_template,
        nll_channel_hint,
        nllocbin=nllocbin,
        tmpdir=TMP_PATH,
    )
    locs.show_localizations()

    # to filter out poorly constrained events
    logger.info("\nFiltered catalog:")
    locs.catalog = locs.catalog.filter(
        f"standard_error < {max_standard_error}",
        f"azimuthal_gap < {max_azimuthal_gap}",
    )
    locs.show_localizations()

    logger.info("Writing all.qml and all.sc3ml")
    qml_fname = os.path.join(QML_PATH, "all.qml")
    locs.catalog.write(qml_fname, format="QUAKEML")
    sc3ml_fname = os.path.join(QML_PATH, "all.sc3ml")
    locs.catalog.write(sc3ml_fname, format="SC3ML")