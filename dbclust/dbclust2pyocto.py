#!/usr/bin/env python
import sys
import pandas as pd
import copy
import datetime
import logging
from itertools import chain, combinations
from collections import Counter
import pyocto
from dbclust.clusterize import Clusterize, cluster_share_eventid

""" Use pyocto to speed up and better constrain clusterization

https://pyocto.readthedocs.io

"""
# default logger
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger("match")
logger.setLevel(logging.INFO)


def create_velocity_model(velocity_cfg, model_path):
    model = pd.DataFrame(
        {
            "depth": velocity_cfg["depth"],
            "vp": velocity_cfg["vp"],
            "vs": velocity_cfg["vs"],
        }
    )

    pyocto.VelocityModel1D.create_model(
        model,
        velocity_cfg["grid_spacing_km"],  # Grid spacing in kilometer
        velocity_cfg[
            "max_horizontal_dist_km"
        ],  # Maximum distance in horizontal direction in km
        velocity_cfg[
            "max_vertical_dist_km"
        ],  # Maximum distance in vertical direction in km
        model_path,
    )


def dbclust2pyocto(myclust, associator_cfg, velocity_model, min_com_phases):
    all_picks_list = list(chain(*myclust.clusters))
    logger.info(
        f"Using pyocto to check/split/filter clusters ({len(all_picks_list)} picks)"
    )

    pyocto_clusters = []
    pyocto_preloc = []
    for cluster in myclust.clusters:
        stations = get_stations_from_cluster(cluster)
        min_lat = stations["latitude"].min()
        max_lat = stations["latitude"].max()
        min_lon = stations["longitude"].min()
        max_lon = stations["longitude"].max()

        picks = get_picks_from_cluster(cluster)

        associator = pyocto.OctoAssociator.from_area(
            lat=(min_lat, max_lat),
            lon=(min_lon, max_lon),
            zlim=associator_cfg["zlim"],
            time_before=associator_cfg[
                "time_before"
            ],  # should be greater than dbclust time_window parameter
            max_pick_overlap=associator_cfg["max_pick_overlap"],
            min_pick_fraction=associator_cfg["min_pick_fraction"],
            min_node_size=associator_cfg["min_node_size"],  # default 10
            min_node_size_location=associator_cfg[
                "min_node_size_location"
            ],  # default 1.5
            velocity_model=velocity_model,
            pick_match_tolerance=associator_cfg["pick_match_tolerance"],
            n_picks=associator_cfg["n_picks"],
            n_p_picks=associator_cfg["n_p_picks"],
            n_s_picks=associator_cfg["n_s_picks"],
            n_p_and_s_picks=associator_cfg["n_p_and_s_picks"],
            exponential_edt=True,
            location_split_depth=6,  # default 6
            location_split_return=4,  # default 4
            refinement_iterations=3,  # default 3
        )
        associator.transform_stations(stations)

        events, assignments = associator.associate(picks, stations)

        if len(events):
            associator.transform_events(events)
            events["time"] = events["time"].apply(
                datetime.datetime.fromtimestamp, tz=datetime.timezone.utc
            )
        # print(assignments)
        # print(events)

        pyocto_preloc.extend(get_events_list(events))
        pyocto_clusters.extend(
            get_clusters_from_assignment(cluster, events, assignments)
        )

    # merge clusters (merge also preloc) with common picks or event_id
    pyocto_clusters, pyocto_preloc = cluster_merge(
        pyocto_clusters, pyocto_preloc, min_com_phases
    )

    pyocto_clusters = aggregate_pick_to_cluster_with_common_event_id(
        pyocto_clusters, all_picks_list
    )

    newclust = copy.deepcopy(myclust)
    newclust.clusters = pyocto_clusters
    newclust.n_clusters = len(newclust.clusters)
    newclust.clusters_stability = [1] * newclust.n_clusters  # unused but needed
    # Add preliminary localization to choose NLL velocity model
    newclust.preloc = pyocto_preloc

    logger.info(
        f"pyocto {newclust.n_clusters} found, dbclust {myclust.n_clusters} found."
    )

    return newclust


def cluster_merge(clusters, preloc, min_com_phases):
    # merge as much as possible
    merge_count = 1
    while merge_count:
        clusters, preloc, merge_count = cluster_merge_one_pass(
            clusters, preloc, min_com_phases
        )
    return clusters, preloc


def cluster_merge_one_pass(clusters, preloc, min_com_phases):
    logger.info(f"pyocto cluster_merge(): {len(clusters)} clusters")

    merge_count = 0
    to_be_merged = []

    for c1, c2 in combinations(clusters, 2):
        common_elements = (Counter(c1) & Counter(c2)).values()
        common_count = sum(common_elements)

        eventid_shared = cluster_share_eventid(c1, c2)
        if common_count >= min_com_phases or eventid_shared:
            logger.debug(
                f"merging 2 clusters: picks shared: {common_count}, eventid shared: {eventid_shared}"
            )
            to_be_merged.append([c1, c2])

    for c1, c2 in to_be_merged:
        if c1 not in clusters or c2 not in clusters:
            continue

        c1_index = clusters.index(c1)
        preloc_c1 = preloc[c1_index]
        del clusters[c1_index]
        del preloc[c1_index]
        # clusters.remove(c1)

        c2_index = clusters.index(c2)
        preloc_c2 = preloc[c2_index]
        del clusters[c2_index]
        del preloc[c2_index]
        # clusters.remove(c2)

        clusters.append(list(set(c1 + c2)))
        # keeps the preloc with the highest number of stations
        if len(c1) > len(c2):
            preloc.append(preloc_c1)
        else:
            preloc.append(preloc_c2)

        merge_count += 1

    return clusters, preloc, merge_count


def aggregate_pick_to_cluster_with_common_event_id(clusters, picks):
    for cluster in clusters:
        event_ids = list(set([p.eventid for p in cluster if p.eventid]))
        picks_copy = copy.deepcopy(picks)
        for p in picks:
            if not p.eventid:
                continue
            if p.eventid in event_ids:
                cluster.append(p)
                picks_copy.remove(p)
        pick = picks_copy
        cluster = list(set(cluster))
    return clusters


def get_events_list(events):
    hypocenters = []
    for index, row in events.iterrows():
        hypo = {
            "time": row["time"],
            "latitude": row["latitude"],
            "longitude": row["longitude"],
            "depth_m": row["depth"] * 1000.0,
        }
        hypocenters.append(hypo)
    return hypocenters


def get_clusters_from_assignment(picks, events, assignments):
    """Returns a clusters list containing picks

    cluster is a list of picks (ie. Phase class)
    """
    clusters = []
    for index, row in events.iterrows():
        event_idx = row["idx"]
        picks_idx_list = assignments[assignments["event_idx"] == event_idx][
            "pick_idx"
        ].to_list()
        cluster = [picks[i] for i in picks_idx_list]
        clusters.append(cluster)

    return clusters


def get_stations_from_cluster(cluster):
    """Returns a Dataframe containing stations information columns:

    id
    latitude
    longitude
    elevation

    """
    station = []
    latitude = []
    longitude = []
    elevation = []

    for p in cluster:
        station.append(".".join([p.network, p.station]))
        latitude.append(p.coord["latitude"])
        longitude.append(p.coord["longitude"])
        elevation.append(p.coord["elevation"])

    df = pd.DataFrame(
        {
            "id": station,
            "latitude": latitude,
            "longitude": longitude,
            "elevation": elevation,
        }
    )

    return df


def get_picks_from_cluster(cluster):
    """Returns a Dataframe containing stations informations columns:

    station
    phase
    time

    """
    station = []
    phase = []
    time = []
    for p in cluster:
        station.append(".".join([p.network, p.station]))
        phase.append(p.phase[0])
        time.append(p.time)

    df = pd.DataFrame(
        {
            "station": station,
            "phase": phase,
            "time": time,
        }
    )

    return df
