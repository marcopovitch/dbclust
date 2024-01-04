#!/usr/bin/env python
import sys
import os
import logging
import argparse
import yaml
import numpy as np
import pandas as pd
import geopandas as gpd
import tempfile
from icecream import ic
from obspy import Inventory, read_inventory

from dbclust.phase import import_phases, import_eqt_phases
from dbclust.clusterize import (
    Clusterize,
    get_picks_from_event,
    merge_cluster_with_common_phases,
    feed_picks_probabilities,
    feed_picks_event_ids,
)
from dbclust.dbclust2pyocto import (
    dbclust2pyocto,
    create_velocity_model,
)
from dbclust.localization import NllLoc, show_event
from dbclust.quakeml import make_readable_id, feed_distance_from_preloc_to_pref_origin
from dbclust.zones import load_zones
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


def ymljoin(loader, node):
    seq = loader.construct_sequence(node)
    return "".join([str(i) for i in seq])


def yml_read_config(filename):
    with open(filename, "r") as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
    return cfg


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

    yaml.add_constructor("!join", ymljoin)
    cfg = yml_read_config(args.configfile)

    ################ CONFIG ################

    file_cfg = cfg["file"]
    station_cfg = cfg["station"]
    time_cfg = cfg["time"]
    pick_cfg = cfg["pick"]
    cluster_cfg = cfg["cluster"]
    nll_cfg = cfg["nll"]
    reloc_cfg = cfg["relocation"]
    if "zones" in cfg:
        zones_cfg = cfg["zones"]
    else:
        zones_cfg = None

    if "quakeml" in cfg:
        quakeml_cfg = cfg["quakeml"]
    else:
        quakeml_cfg = None
    catalog_cfg = cfg["catalog"]

    if "default_pyocto_vmodel" in cfg:
        model_name = cfg["default_pyocto_vmodel"]
        if model_name:
            if "pyocto_vmodel" in cfg:
                pyocto_vmodel = cfg["pyocto_vmodel"]
            else:
                logger.error("Missing pyocto_vmodel section !")
                sys.exit(254)

            if model_name not in [i["name"] for i in pyocto_vmodel]:
                logger.error(f"Referenced model {model_name} is not defined !")
                sys.exit(254)
            pyocto_velocity_cfg = [
                i for i in pyocto_vmodel if i["name"] == model_name
            ].pop()
            pyocto_associator_cfg = pyocto_velocity_cfg["associator"]
            ic(pyocto_velocity_cfg)
            ic(pyocto_associator_cfg)
        else:
            pyocto_velocity_cfg = None
    else:
        pyocto_velocity_cfg = None

    ############### path definition ###################
    OBS_PATH = file_cfg["obs_path"]
    QML_PATH = file_cfg["qml_path"]
    TMP_PATH = file_cfg["tmp_path"]

    for dir in [OBS_PATH, QML_PATH, TMP_PATH]:
        os.makedirs(dir, exist_ok=True)

    # how to get station coordinates
    info_sta_method = station_cfg["use"]
    if info_sta_method == "inventory":
        info_sta = Inventory()
        for i in station_cfg[info_sta_method]:
            logger.info(f"Reading inventory file {i}")
            info_sta.extend(read_inventory(i))
    else:
        info_sta = station_cfg[info_sta_method]
        logger.info(f"Using fdsnws {info_sta} to get station coordinates")

    if "blacklist" in station_cfg:
        black_listed_stations = station_cfg["blacklist"]
        if not black_listed_stations:
            black_listed_stations = []
    else:
        black_listed_stations = []

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
        # default value
        time_window = 7  # min

    if "overlap_window" in time_cfg:
        overlap_window = time_cfg["overlap_window"]
    else:
        # default value
        overlap_window = 60  # seconds

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
    # parameters for hdbscan clustering
    #
    min_cluster_size = cluster_cfg["min_cluster_size"]
    min_station_count = cluster_cfg["min_station_count"]
    average_velocity = cluster_cfg["average_velocity"]
    min_picks_common = cluster_cfg["min_picks_common"]
    if "max_search_dist" in cluster_cfg:
        max_search_dist = cluster_cfg["max_search_dist"]
    else:
        max_search_dist = 0.0  # default for hdbscan

    if "pre_computed_tt_matrix" in cluster_cfg:
        pre_computed_tt_matrix = cluster_cfg["pre_computed_tt_matrix"]
        tt_matrix_save = True
    else:
        pre_computed_tt_matrix = None
        tt_matrix_save = False

    # filter based on number of stations that have both P and S
    min_station_with_P_and_S = cluster_cfg["min_station_with_P_and_S"]

    #
    # NonLinLoc
    #
    # NLLoc binary
    nllocbin = nll_cfg["bin"]
    scat2latlon_bin = nll_cfg["scat2latlon_bin"]

    # get nll velocity template
    nlloc_template_path = nll_cfg["nll_template_path"]
    nlloc_times_path = nll_cfg["nll_time_path"]

    # NLL will discard any location with number of phase < nll_min_phase
    # use -1 to not set a limit
    # nll_min_phase = nll_cfg["nll_min_phase"]
    nll_min_phase = min_station_count + min_station_with_P_and_S

    nlloc_verbose = nll_cfg["verbose"]
    nlloc_enable_scatter = nll_cfg["enable_scatter"]

    #
    # get velocity profile
    #
    velocity_profile_conf = nll_cfg["velocity_profile"]
    if hasattr(args, "velocity_profile_name") and args.velocity_profile_name:
        default_velocity_profile = args.velocity_profile_name
    else:
        default_velocity_profile = nll_cfg["default_velocity_profile"]
    logger.info(f"Using {default_velocity_profile} profile")

    template = None
    for p in velocity_profile_conf:
        if p["name"] == default_velocity_profile:
            template = p["template"]
    if not template:
        logger.error(f"profile {default_velocity_profile} does not exist !")
        logger.error(f"Available velocity models are:")
        for p in velocity_profile_conf:
            logger.error(f'\t{p["name"]}')
        sys.exit()
    nlloc_template = os.path.join(nlloc_template_path, template)
    logger.info(f"using {nlloc_template} as nll template")

    # force quakeml model_id:
    if quakeml_cfg:
        quakeml_cfg["model_id"] = default_velocity_profile
    else:
        quakeml_cfg = {}
        quakeml_cfg["model_id"] = default_velocity_profile

    # file to keep track of SCNL when exporting to NLL
    nll_channel_hint = nll_cfg["nll_channel_hint"]

    # station distance cut off in km
    dist_km_cutoff = reloc_cfg["dist_km_cutoff"]

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

    #
    # Zones
    #
    if zones_cfg:
        zones = load_zones(zones_cfg, nll_cfg)
        if zones.empty:
            logger.error("Zones are defined but empty ! Check configuration file ...")
            sys.exit()
    else:
        zones = gpd.GeoDataFrame()

    ########################################

    logger.info(f"Opening {picks_file} file.")
    try:
        if picks_type == "eqt":
            df = pd.read_csv(
                picks_file,
                parse_dates=["p_arrival_time", "s_arrival_time"],
                low_memory=False,
            )
            #
            df["phase_time"] = df[["p_arrival_time", "s_arrival_time"]].min(axis=1)
        else:
            try:
                # official PhaseNet
                df = pd.read_csv(
                    picks_file, parse_dates=["phase_time"], low_memory=False
                )
            except:
                # PhaseNetSDS
                df = pd.read_csv(picks_file, parse_dates=["time"], low_memory=False)
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
            # limits to 10^-4 seconds same as NLL (needed to unload some picks)
            df["phase_time"] = df["phase_time"].dt.round("0.0001S")
            df.sort_values(by=["phase_time"], inplace=True)
            if "phase_index" in df.columns:
                # seems to be fake picks
                df = df[df["phase_index"] != 1]
    except Exception as e:
        logger.error(e)
        sys.exit()
    logger.info(f"Read {len(df)} phases.")

    ##########################
    # Preprocessing on picks #
    ##########################
    # get rid off nan value when importing phases without eventid
    df = df.replace({np.nan: None})
    # keeps only network.station
    df["station_id"] = df["station_id"].map(lambda x: ".".join(x.split(".")[:2]))

    # Time filtering
    if date_begin:
        logger.info(f"Using picks >= {date_begin}")
        date_begin = pd.to_datetime(date_begin, utc=True)
        df = df[df["phase_time"] >= date_begin]

    if date_end:
        logger.info(f"Using picks <= {date_end}")
        date_end = pd.to_datetime(date_end, utc=True)
        df = df[df["phase_time"] <= date_end]

    if df.empty:
        logger.warning(f"No data in time range [{date_begin}, {date_end}].")
        sys.exit()

    logger.info("Removing black listed stations:")
    for sta in black_listed_stations:
        count_before_filter = len(df)
        df = df[~df["station_id"].str.contains(sta)]
        logger.info(f"\t- removed {count_before_filter-len(df)} {sta} picks")

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
        scat2latlon_bin,
        nlloc_times_path,
        nlloc_template,
        nll_channel_hint=nll_channel_hint,
        tmpdir=TMP_PATH,
        double_pass=double_pass,
        P_time_residual_threshold=P_time_residual_threshold,
        S_time_residual_threshold=S_time_residual_threshold,
        dist_km_cutoff=dist_km_cutoff,
        nll_min_phase=nll_min_phase,
        min_station_with_P_and_S=min_station_with_P_and_S,
        quakeml_settings=quakeml_cfg,
        nll_verbose=nlloc_verbose,
        keep_scat=nlloc_enable_scatter,
        log_level=logger.level,
    )

    # process independently each time period
    # fixme: add overlapp between time period
    part = 0
    last_saved_event_count = 0
    last_round = False
    previous_myclust = Clusterize(
        phases=None,  # empty cluster / constructor only
        average_velocity=average_velocity,
        min_station_count=min_station_count,
        min_station_with_P_and_S=min_station_with_P_and_S,
        max_search_dist=max_search_dist,
        P_uncertainty=P_uncertainty,
        S_uncertainty=S_uncertainty,
        tt_matrix_fname=pre_computed_tt_matrix,
        tt_matrix_save=tt_matrix_save,
        zones=zones,
        log_level=logger.level,
    )

    if pyocto_velocity_cfg:
        import pyocto

        model_path = os.path.join(
            file_cfg["data_path"], f"{pyocto_velocity_cfg['name']}.vmodel"
        )
        create_velocity_model(pyocto_velocity_cfg["velocity_model"], model_path)
        logger.info(f"Using vmodel: {pyocto_velocity_cfg['name']} ({model_path})")

        tolerance = pyocto_velocity_cfg["velocity_model"]["tolerance"]
        pyocto_velocity_model = pyocto.associator.VelocityModel1D(
            path=model_path,
            tolerance=tolerance,
            # association_cutoff_distance=None,
            # location_cutoff_distance=None,
            # surface_p_velocity=None,
            # surface_s_velocity=None,
        )

        # pyocto_velocity_model = pyocto.VelocityModel0D(
        #     p_velocity=6.2,
        #     s_velocity=3.5,
        #     tolerance=2.0,
        # )

    picks_to_remove = []
    for i, (from_time, to_time) in enumerate(
        zip(time_periods[:-1], time_periods[1:]), start=1
    ):
        logger.debug("================================================")
        logger.debug("")
        logger.debug("")

        # keep an overlap
        begin = from_time - np.timedelta64(overlap_window, "s")
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
            # print(df_subset.to_string())
            df_subset = unload_picks_list(df_subset, picks_to_remove)
            logger.info(f"after unload picks: pick length is {len(df_subset)}")
            # print(df_subset.to_string())
            picks_to_remove = []

        # print(df_subset[["station_id", "phase_time"]].to_string())

        # Import picks from dataframe
        if picks_type == "eqt":
            phases = import_eqt_phases(
                df_subset,
                P_proba_threshold,
                S_proba_threshold,
            )
        else:
            phases = import_phases(
                df_subset,
                P_proba_threshold,
                S_proba_threshold,
                info_sta,
            )
            if logger.level == logging.DEBUG:
                for p in phases:
                    p.show_all()

        if logger.level == logging.DEBUG:
            logger.info("previous_myclust:")
            previous_myclust.show_clusters()

        # find clusters
        myclust = Clusterize(
            phases=phases,
            min_cluster_size=min_cluster_size,
            average_velocity=average_velocity,
            min_station_count=min_station_count,
            min_station_with_P_and_S=min_station_with_P_and_S,
            max_search_dist=max_search_dist,
            P_uncertainty=P_uncertainty,
            S_uncertainty=S_uncertainty,
            tt_matrix_fname=pre_computed_tt_matrix,
            tt_matrix_save=tt_matrix_save,
            zones=zones,
            log_level=logger.level,
        )

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
            previous_myclust, myclust, min_picks_common
        )

        # This is the last round: merge previous_myclust and myclust
        if i == (len(time_periods) - 1):
            last_round = True
            logger.info("Last round, merging all remaining clusters.")
            previous_myclust.merge(myclust)

        if pyocto_velocity_cfg:
            previous_myclust = dbclust2pyocto(
                previous_myclust,
                pyocto_associator_cfg,
                pyocto_velocity_model,
                min_picks_common,
            )

        # Now, process previous_myclust and wait next round to process myclust
        # write each cluster to nll obs files
        with tempfile.TemporaryDirectory(dir=OBS_PATH) as TMP_OBS_PATH:
            my_obs_path = os.path.join(TMP_OBS_PATH, f"{i}")
            previous_myclust.generate_nllobs(my_obs_path)

            # localize each cluster
            # all locs are automatically appended to the locator's catalog
            # force to cleanup all files generated by Nonlinloc
            logger.info("Starting localization")
            with tempfile.TemporaryDirectory(
                dir=TMP_PATH
            ) as tmpdir_automaticaly_cleaned:
                locator.tmpdir = tmpdir_automaticaly_cleaned

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

                # sequential version
                clustcat = locator.get_localisations_from_nllobs_dir(
                    my_obs_path, append=True
                )

                # fixme
                # handle scat file here before the directory is deleted
                if nlloc_enable_scatter:
                    logger.warning("FIXME: scatter file not yet handled !")

        if len(clustcat) > 0:
            for event in sorted(
                clustcat.events, key=lambda e: e.preferred_origin().time
            ):
                next_begin = end - np.timedelta64(overlap_window, "s")

                origin = event.preferred_origin()
                first_station, first_phase, first_pick_time = get_picks_from_event(
                    event, origin, None
                ).pop(0)
                last_station, last_phase, last_pick_time = get_picks_from_event(
                    event, origin, None
                ).pop(-1)

                logger.debug(
                    "First pick is: %s, last pick is: %s, ovelapped zone starts: %s, next ovelapped zone starts: %s"
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
        clustcat = make_readable_id(clustcat, "sihex", "quakeml:franceseisme.fr")

        # prepare next round
        previous_myclust = myclust

        # write partial qml file and clean catalog from memory
        if (last_saved_event_count) > event_flush_count:
            partial_qml = os.path.join(QML_PATH, f"{qml_base_filename}-{part}.qml")
            logger.info(f"Writing {len(locator.catalog)} events in {partial_qml}")
            locator.catalog.write(partial_qml, format="QUAKEML")
            locator.catalog.clear()
            last_saved_event_count = 0
            part += 1
        else:
            # last_saved_event_count += len(clustcat)
            last_saved_event_count += len(locator.catalog)

    # Write last events
    partial_qml = os.path.join(QML_PATH, f"{qml_base_filename}-{part}.qml")
    logger.info(f"Writing {len(locator.catalog)} events in {partial_qml}")
    locator.catalog.write(partial_qml, format="QUAKEML")
