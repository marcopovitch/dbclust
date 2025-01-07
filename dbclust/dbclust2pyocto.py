#!/usr/bin/env python
import copy
import datetime
import logging
import sys
from collections import Counter
from itertools import chain
from itertools import combinations
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import pandas as pd
import pyocto
import pyproj
from clusterize import cluster_share_eventid
from clusterize import Clusterize
from config import Associator
from icecream import ic
from phase import Phase

# import faulthandler
# faulthandler.enable()

"""
Use PyOcto to speed up and better constrain clustering

reference: https://pyocto.readthedocs.io
"""
# default logger
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger("dbclust2pyocto")
logger.setLevel(logging.INFO)

ic.configureOutput(outputFunction=lambda msg: sys.stdout.write(msg + "\n"))


class MultipleEventIDsWithSameAgencyError(Exception):
    """
    Exception raised when multiple event_ids are associated
    with the same agency in the same cluster
    """

    def __init__(self, duplicate_agency_event_ids, message=None):
        self.duplicate_agency_event_ids = duplicate_agency_event_ids
        self.message = message or (
            f"Multiple event_ids share the same agency: {duplicate_agency_event_ids}"
        )
        super().__init__(self.message)


def adjust_associator_tolerance(
    myclust,
    cfg,
    tolerance_steps={1: 0.5, 0: 0.1},
    min_tolerance=0.5,
    log_level=logging.INFO,
):
    """
    Adjust associator.pick_match_tolerance using a linear decay search.

    Args:
        myclust (Clusterize): The cluster object to process.
        cfg: Configuration object containing associator and other settings.
        tolerance_steps (dict): Dictionary with ranges and step sizes, e.g.,
                                {1: 0.5, 0: 0.1}.
        min_tolerance (float): Minimum allowed pick match tolerance.
        log_level (int): Logging level for debug information.

    Returns:
        Clusterize or None: Returns the processed cluster if successful,
                            otherwise returns None.
    """
    associator = cfg.pyocto.current_model.associator
    tolerance = associator.pick_match_tolerance

    logger.info(f"Starting linear decay for pick_match_tolerance: {tolerance}")
    while tolerance >= min_tolerance:
        logger.info(f"Trying pick_match_tolerance: {tolerance:.2f}")
        associator.pick_match_tolerance = tolerance
        try:
            result_myclust = dbclust2pyocto(
                myclust,
                cfg.pyocto.default_model_name,
                associator,
                cfg.pyocto.velocity_model,
                cfg.cluster.min_picks_common,
                log_level=log_level,
            )
            logger.info(f"Success with pick_match_tolerance: {tolerance:.2f}")
            return result_myclust
        except pyproj.exceptions.CRSError as e:
            # Skip processing if CRS error occurs, likely due to too far away stations
            logger.error("Skipping dbclust2pyocto() processing.")
            raise
        except MultipleEventIDsWithSameAgencyError as e:
            logger.warning(f"Unsuccessful with pick_match_tolerance: {tolerance:.2f}.")
            logger.warning(f"{e}")
            # Determine step size based on tolerance range
            step = next((s for t, s in tolerance_steps.items() if tolerance > t), 0.5)
            tolerance -= step

    logger.error("Exhausted all tolerances. Skipping pyocto processing.")
    return None


def create_velocity_model(velocity_cfg: dict, model_path: str) -> None:
    """
    Create a 1D velocity model and save it to the specified path.

    Parameters:
        velocity_cfg (dict): Configuration dictionary containing the following keys:
            - "depth" (list or array-like): Depth values for the model.
            - "vp" (list or array-like): P-wave velocities for a given depths list.
            - "vs" (list or array-like): S-wave velocities for a given  depths list.
            - "grid_spacing_km" (float): Grid spacing in kilometers.
            - "max_horizontal_dist_km" (float): Maximum distance in the horizontal direction in kilometers.
            - "max_vertical_dist_km" (float): Maximum distance in the vertical direction in kilometers.
        model_path (str): Path where the velocity model will be saved.

    Returns:
        None
    """
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


def dbclust2pyocto(
    myclust: Clusterize,
    model_name: str,
    associator_cfg: Associator,
    velocity_model: pyocto.VelocityModel1D,
    min_com_phases: int,
    log_level=logging.INFO,
) -> Clusterize:
    """
    Processes clusters using the pyocto library to check, split, and filter them.

    Args:
        myclust (Clusterize): The input cluster object containing clusters to be processed.
        model_name (str): The name of the model to be used for processing.
        associator_cfg (Associator): Configuration for the pyocto associator.
        velocity_model (pyocto.VelocityModel1D): The velocity model to be used for event association.
        min_com_phases (int): Minimum number of common phases required for merging clusters.
        log_level (int, optional): Logging level. Defaults to logging.INFO.

    Returns:
        Clusterize: A new Clusterize object with processed clusters.
    """
    logger.info(
        f"Using pyocto to process clusters ({sum(len(c) for c in myclust.clusters)} picks)"
    )

    all_picks_list = list(chain(*myclust.clusters))
    pyocto_clusters, pyocto_preloc = [], []

    for cluster in myclust.clusters:
        # Extract station and pick data for the cluster
        stations = get_stations_from_cluster(cluster)
        picks = get_picks_from_cluster(cluster)

        # Set spatial parameters for the associator
        lat_range = (stations["latitude"].min(), stations["latitude"].max())
        lon_range = (stations["longitude"].min(), stations["longitude"].max())

        try:
            associator = pyocto.OctoAssociator.from_area(
                lat=lat_range,
                lon=lon_range,
                zlim=associator_cfg.zlim,
                time_before=associator_cfg.time_before,  # should be greater than dbclust time_window parameter
                max_pick_overlap=associator_cfg.max_pick_overlap,
                min_pick_fraction=associator_cfg.min_pick_fraction,
                min_node_size=associator_cfg.min_node_size,  # default 10
                min_node_size_location=associator_cfg.min_node_size_location,  # default 1.5
                velocity_model=velocity_model,
                pick_match_tolerance=associator_cfg.pick_match_tolerance,
                min_interevent_time=0.5,  # default 3
                n_picks=associator_cfg.n_picks,
                n_p_picks=associator_cfg.n_p_picks,
                n_s_picks=associator_cfg.n_s_picks,
                n_p_and_s_picks=associator_cfg.n_p_and_s_picks,
                exponential_edt=True,
                location_split_depth=6,  # default 6
                location_split_return=4,  # default 4
                refinement_iterations=3,  # default 3
                # second_pass_overwrites={},  # default None
            )
        except pyproj.exceptions.CRSError as e:
            # Skip processing if CRS error occurs, likely due to too far away stations
            logger.error(f"CRS error occurred. Skipping processing: {e}")
            logger.error(f"Check stations coordinates ! lat_range: {lat_range}, lon_range: {lon_range}")
            ic(picks)
            raise

        associator.transform_stations(stations)

        # Associate picks and generate events
        events, assignments = associator.associate(picks, stations)
        if len(events):
            associator.transform_events(events)
            events["time"] = events["time"].apply(
                datetime.datetime.fromtimestamp, tz=datetime.timezone.utc
            )

        # Store events and update clusters
        pyocto_preloc.extend(get_events_list(events, assignments, stations, model_name))
        pyocto_clusters.extend(
            get_clusters_from_assignment(cluster, events, assignments)
        )

    # Merge clusters with common picks or event IDs
    pyocto_clusters, pyocto_preloc = cluster_merge(
        pyocto_clusters, pyocto_preloc, min_com_phases
    )

    # Aggregate picks into clusters with shared event IDs
    try:
        pyocto_clusters = aggregate_pick_to_cluster_with_common_event_id(
            pyocto_clusters, all_picks_list, min_com_phases
        )
    except MultipleEventIDsWithSameAgencyError as e:
        raise

    logger.info(
        f"PyOcto found {len(pyocto_clusters)} clusters, dbclust found {myclust.n_clusters} clusters."
    )

    if len(pyocto_clusters) == 0 and myclust.n_clusters > 0:
        # just in case pyocto does not find any clusters
        logger.warning("PyOcto did not find any clusters. Returning original dbclust clusters.")
        # fixme: for each cluster add a preloc based on the barycenter of the stations
        return myclust

    # Clone the original Clusterize object and update it with the new clusters
    newclust = copy.deepcopy(myclust)
    newclust.clusters = pyocto_clusters
    newclust.n_clusters = len(newclust.clusters)
    newclust.clusters_stability = [1] * newclust.n_clusters  # unused but needed
    newclust.preloc = pyocto_preloc  # used to choose NLL velocity model

    # Clean up the original cluster object
    for attr in ["clusters", "clusters_stability", "noise", "zones", "preloc"]:
        if hasattr(myclust, attr):
            delattr(myclust, attr)

    return newclust


def cluster_merge(
    clusters: List[List[Phase]], preloc, min_com_phases: int
) -> Tuple[List[List[Phase]], List]:
    """
    Iteratively merges clusters with shared phases or event IDs
    until no more merges are possible.

    Args:
        clusters (List[List[Phase]]): List of clusters to be merged.
        preloc: Preliminary localization data associated with clusters.
        min_com_phases (int): Minimum number of common phases required for merging clusters.

    Returns:
        Tuple: Merged clusters and updated preliminary localization data.
    """
    while True:
        clusters, preloc, merge_count = cluster_merge_one_pass(
            clusters, preloc, min_com_phases
        )
        if merge_count == 0:
            break
    return clusters, preloc


def cluster_merge_one_pass(
    clusters: List[List[Phase]], preloc: List, min_com_phases: int
) -> Tuple[List[List[Phase]], List, int]:
    """
    Perform one pass of cluster merging based on shared picks or event IDs.

    Args:
        clusters: List of clusters (each cluster is a list of Phase objects).
        preloc: List of prelocation data corresponding to clusters.
        min_com_phases: Minimum number of shared phases for merging.

    Returns:
        Tuple: Updated clusters, updated preloc, and count of merges performed.
    """
    logger.info(f"pyocto cluster_merge(): working on {len(clusters)} clusters")

    merge_count = 0
    to_be_merged = []

    # Identify clusters to merge
    for c1_idx, c2_idx in combinations(range(len(clusters)), 2):
        c1, c2 = clusters[c1_idx], clusters[c2_idx]
        # Count common elements (shared picks)
        common_count = sum((Counter(c1) & Counter(c2)).values())

        # Check if clusters share event IDs
        eventid_shared = cluster_share_eventid(c1, c2, shared_threshold=min_com_phases)

        if common_count >= min_com_phases or eventid_shared:
            logger.info(
                f"Merging clusters: picks shared: {common_count}, event ID shared: {eventid_shared}"
            )
            to_be_merged.append((c1_idx, c2_idx))

    # Merge identified clusters
    merged_indices = set()
    for c1_idx, c2_idx in to_be_merged:
        if c1_idx in merged_indices or c2_idx in merged_indices:
            continue

        # Merge the clusters
        c1, c2 = clusters[c1_idx], clusters[c2_idx]
        merged_cluster = list(set(c1 + c2))
        clusters[c1_idx] = merged_cluster

        # Update prelocation data
        preloc_c1 = preloc[c1_idx]
        preloc_c2 = preloc[c2_idx]
        preloc[c1_idx] = preloc_c1 if len(c1) > len(c2) else preloc_c2

        # Mark the second cluster as merged
        merged_indices.add(c2_idx)

        merge_count += 1

    # Remove merged clusters from the list
    clusters = [clusters[i] for i in range(len(clusters)) if i not in merged_indices]
    preloc = [preloc[i] for i in range(len(preloc)) if i not in merged_indices]

    return clusters, preloc, merge_count


def aggregate_pick_to_cluster_with_common_event_id(
    clusters: List[List[Phase]], picks: List[Phase], pick_count_threshold: int = 3
) -> List[List[Phase]]:
    """
    Aggregates picks into clusters based on common event IDs.
    This function iterates through a list of clusters and adds picks to clusters
    if the event ID of the pick is common within the cluster. A pick is added to
    a cluster if its event ID appears more than pick_count_threshold times in the cluster
    to avoid adding picks if a cluster is contaminated with few picks from other events.

    Args:
        clusters (List[List[Phase]]):
            A list of clusters, where each cluster is a list of Phase objects.
            picks (List[Phase]): A list of Phase objects to be aggregated into clusters.
        pick_count_threshold (int, optional):
            The threshold for the number of picks with the same event ID in a cluster. Defaults to 5.

    Returns:
        List[List[Phase]]: The updated list of clusters with aggregated picks.
    """
    logger.info(
        f"aggregate_pick_to_cluster_with_common_event_id(): {len(clusters)} clusters"
    )
    for cluster in clusters:
        # Count the occurrences of event_id in the cluster
        event_id_counts = Counter([p.event_id for p in cluster if p.event_id])
        if event_id_counts:
            ic(event_id_counts)

        # count the number of agency in each event_id in event_id_counts
        event_id_agency = {}
        for p in picks:
            if p.event_id in event_id_counts:
                if p.event_id not in event_id_agency:
                    event_id_agency[p.event_id] = set()
                event_id_agency[p.event_id].add(p.agency)
        # if event_id_agency:
        #     ic(event_id_agency)

        # Invert the mapping to find agencies associated with multiple event_ids
        agency_event_map = {}
        for event_id, agencies in event_id_agency.items():
            for agency in agencies:
                if agency not in agency_event_map:
                    agency_event_map[agency] = set()
                agency_event_map[agency].add(event_id)
        if agency_event_map:
            ic(agency_event_map)

        # Detect agencies associated with multiple event_ids
        duplicate_agency_event_ids = {
            agency: event_ids
            for agency, event_ids in agency_event_map.items()
            if len(event_ids) > 1
        }

        if duplicate_agency_event_ids:
            raise MultipleEventIDsWithSameAgencyError(duplicate_agency_event_ids)

        # Check if any event_id has a count > pick_count_threshold
        if not any(count > pick_count_threshold for count in event_id_counts.values()):
            continue

        # Deep copy the picks to modify without side effects
        picks_copy = copy.deepcopy(picks)

        # Add picks with a corresponding event_id
        for p in picks:
            if not p.event_id:
                continue
            if (
                p.event_id in event_id_counts
                and event_id_counts[p.event_id] > pick_count_threshold
            ):
                cluster.append(p)
                picks_copy.remove(p)

        # Update the remaining picks
        picks = picks_copy

        # Remove duplicates in the cluster
        cluster = list(set(cluster))

    return clusters


def get_events_list(
    events: pd.DataFrame,
    assignments: pd.DataFrame,
    stations: pd.DataFrame,
    model_name_used: str,
) -> List[dict]:
    """Get info on events and picks to populate an event quakeml

    Args:
        events (pd.DataFrame): Dataframe with events
        assignments (pd.DataFrame): Dataframe with picks corresponding to events
        stations (pd.DataFrame): Dataframe with stations coordinates
        model_name (str): model name used by PyOcto to get preliminary location

    Returns:
        List[dict]: simple dict with events information
    """
    hypocenters = []
    for index, row in events.iterrows():
        event_idx = row["idx"]
        picks = assignments[assignments["event_idx"] == event_idx]
        picks_col_names = ["station", "phase", "time", "residual"]
        picks = picks[picks_col_names].values.tolist()
        coords_col_names = ["id", "latitude", "longitude", "elevation"]
        coords = stations[coords_col_names].values.tolist()

        hypo = {
            "time": row["time"],
            "latitude": row["latitude"],
            "longitude": row["longitude"],
            "depth_m": row["depth"] * 1000.0,
            "phase_count": row["picks"],
            "model_name_used": model_name_used,
            "picks_col_names": picks_col_names,
            "phases": picks,
            "coords_col_names": coords_col_names,
            "coords": coords,
        }
        hypocenters.append(hypo)
    return hypocenters


def get_clusters_from_assignment(
    picks: pd.DataFrame, events: pd.DataFrame, assignments: pd.DataFrame
) -> List[List[dict]]:
    """
    Returns a list of clusters, where each cluster contains a list of picks.

    Args:
        picks (pd.DataFrame): DataFrame containing pick information.
        events (pd.DataFrame): DataFrame containing event information.
        assignments (pd.DataFrame): DataFrame containing assignment information
                                    mapping event indices to pick indices.

    Returns:
        List[List[dict]]:
            A list of clusters, where each cluster is a list of
            dictionaries containing pick information.
        picks: pd.DataFrame, events: pd.DataFrame, assignments: pd.DataFrame
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


def get_stations_from_cluster(cluster: List[Phase]) -> pd.DataFrame:
    """
    Extracts station information from a cluster of Phase objects and returns it as a pandas DataFrame.

    Args:
        cluster (List[Phase]): A list of Phase objects, each containing station information.
    Returns:
        pd.DataFrame:
            A DataFrame with columns 'id', 'latitude', 'longitude', and 'elevation',
            where 'id' is a concatenation of network, station, location, and channel.

    Returns a DataFrame containing stations information columns:
    """
    station = []
    latitude = []
    longitude = []
    elevation = []

    for p in cluster:
        # station.append(".".join([p.network, p.station, p.location, p.channel]))
        station.append(
            ".".join(map(str, [p.network, p.station, p.location, p.channel]))
        )
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


def get_picks_from_cluster(cluster: List[Phase]) -> pd.DataFrame:
    """
    Returns a DataFrame containing pick information from a given cluster of phases.

    Args:
        cluster (List[Phase]): A list of Phase objects representing the cluster.

    Returns:
        pd.DataFrame: A DataFrame with columns:
            - station: A string combining network, station, location, and channel.
            - phase: The phase type, converted to uppercase.
            - time: The time associated with the phase.
    """
    station = []
    phase = []
    time = []
    for p in cluster:
        # station.append(".".join([p.network, p.station, p.location, p.channel]))
        station.append(
            ".".join(map(str, [p.network, p.station, p.location, p.channel]))
        )
        phase.append(p.phase[0].upper())
        time.append(p.time)

    df = pd.DataFrame(
        {
            "station": station,
            "phase": phase,
            "time": time,
        }
    )

    return df
