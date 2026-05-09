#!/usr/bin/env python
import copy
import datetime
import logging
import statistics
import sys
from collections import Counter, defaultdict
from itertools import chain
from itertools import combinations
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import pandas as pd
import pyocto
import pyocto._core as _pyocto_backend
import pyproj
import pyproj.exceptions
from dbclust.clusterize import cluster_share_eventid
from dbclust.clusterize import Clusterize
from dbclust.config import Associator
from dbclust.phase import Phase

# import faulthandler
# faulthandler.enable()

# Workaround for pyocto bug: the C++ backend segfaults when associate() is called
# with an empty picks list (e.g. during the second pass when all picks were used in
# the first pass). Patch at the C++ backend level since the empty-list check must
# happen before the C++ code is reached.
_pyocto_backend_associate_orig = _pyocto_backend.OctoAssociator.associate


def _pyocto_backend_associate_safe(self, picks):
    if not picks:
        return []
    return _pyocto_backend_associate_orig(self, picks)


_pyocto_backend.OctoAssociator.associate = _pyocto_backend_associate_safe

"""
Use PyOcto to speed up and better constrain clustering

reference: https://pyocto.readthedocs.io
"""
# default logger (uses hierarchical name for selective level control)
logger = logging.getLogger("dbclust.pyocto")


class MultipleEventIDsWithSameAgencyError(Exception):
    """
    Exception raised when multiple event_ids are associated
    with the same agency in the same cluster
    """

    def __init__(self, duplicate_agency_event_ids, partial_result=None, message=None):
        self.duplicate_agency_event_ids = duplicate_agency_event_ids
        self.partial_result = partial_result
        self.message = message or (
            f"Multiple event_ids share the same agency: {duplicate_agency_event_ids}"
        )
        super().__init__(self.message)


def adjust_associator_tolerance(
    myclust,
    cfg,
    tolerance_steps={1: 0.5, 0: 0.1},
    min_tolerance=0.5,
    include_noise_in_aggregation=False,
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
        include_noise_in_aggregation (bool): If True, HDBSCAN noise picks are included
            in the pool of picks available for re-injection. Defaults to False.
        log_level (int): Logging level for debug information.

    Returns:
        Clusterize or None: Returns the processed cluster if successful,
                            otherwise returns None.
    """
    associator = cfg.pyocto.current_model.associator
    tolerance = associator.pick_match_tolerance

    best_result = None
    best_n_clusters = 0

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
                delegate_dbclust=cfg.pyocto.delegate_dbclust,
                include_noise_in_aggregation=include_noise_in_aggregation,
                log_level=log_level,
            )
            logger.info(f"Success with pick_match_tolerance: {tolerance:.2f}")
            return result_myclust
        except pyproj.exceptions.ProjError as e:
            # Skip processing if projection error occurs, likely due to too far away stations
            logger.error(f"Projection error, skipping dbclust2pyocto() processing: {e}")
            raise
        except MultipleEventIDsWithSameAgencyError as e:
            logger.warning(f"Unsuccessful with pick_match_tolerance: {tolerance:.2f}.")
            logger.warning(f"{e}")
            # Keep track of the best partial result (most clusters produced)
            if e.partial_result is not None:
                n = e.partial_result.n_clusters
                if n > best_n_clusters:
                    best_n_clusters = n
                    best_result = e.partial_result
                    logger.info(
                        f"New best partial result: {n} clusters at tolerance {tolerance:.2f}"
                    )
            # Determine step size based on tolerance range
            step = next((s for t, s in tolerance_steps.items() if tolerance > t), 0.5)
            tolerance -= step

    logger.error("Exhausted all tolerances.")
    if best_result is not None:
        logger.warning(
            f"Returning best partial result with {best_n_clusters} clusters "
            f"despite unresolved agency conflict."
        )
        return best_result
    logger.error("No partial result available. Skipping pyocto processing.")
    return None


def dbclust2pyocto(
    myclust: Clusterize,
    model_name: str,
    associator_cfg: Associator,
    velocity_model: pyocto.VelocityModel1D,
    min_com_phases: int,
    delegate_dbclust: bool = False,
    include_noise_in_aggregation: bool = False,
    log_level=logging.INFO,
) -> Optional[Clusterize]:
    """
    Processes clusters using the pyocto library to check, split, and filter them.

    Args:
        myclust (Clusterize): The input cluster object containing clusters to be processed.
        model_name (str): The name of the model to be used for processing.
        associator_cfg (Associator): Configuration for the pyocto associator.
        velocity_model (pyocto.VelocityModel1D): The velocity model to be used for event association.
        min_com_phases (int): Minimum number of common phases required for merging clusters.
        delegate_dbclust (bool, optional): Return original dbclust clusters if PyOcto finds none. Defaults to False.
        include_noise_in_aggregation (bool, optional): If True, HDBSCAN noise picks are included
            in the pool of picks available for re-injection into clusters via
            aggregate_pick_to_cluster_with_common_event_id(). Useful to recover small events
            whose picks were classified as noise because they overlapped with a larger event.
            Defaults to False.
        log_level (int, optional): Logging level. Defaults to logging.INFO.

    Returns:
        Clusterize: A new Clusterize object with processed clusters.
    """
    logger.info(
        f"Using pyocto to process clusters ({sum(len(c) for c in myclust.clusters)} picks)"
    )

    noise_picks = list(myclust.noise) if include_noise_in_aggregation else []
    if noise_picks:
        logger.info(
            f"Including {len(noise_picks)} HDBSCAN noise picks in aggregation pool"
        )
    all_picks_list = list(chain(*myclust.clusters)) + noise_picks
    pyocto_clusters, pyocto_preloc = [], []

    clusters_to_process = list(myclust.clusters)
    if include_noise_in_aggregation and myclust.noise:
        noise_event_ids = set(p.event_id for p in myclust.noise if p.event_id)
        if noise_event_ids:
            logger.info(
                f"Adding HDBSCAN noise ({len(myclust.noise)} picks, "
                f"event_ids: {[e.split('/')[-1] for e in noise_event_ids]}) as additional cluster for PyOcto"
            )
            clusters_to_process.append(myclust.noise)

    for i, cluster in enumerate(clusters_to_process):
        # Extract station and pick data for the cluster
        stations = get_stations_from_cluster(cluster)
        picks = get_picks_from_cluster(cluster)

        # Detect multi-event clusters: if multiple distinct event_ids are present,
        # reduce min_pick_fraction to allow PyOcto to find smaller events
        cluster_event_ids = set(p.event_id for p in cluster if p.event_id)
        if associator_cfg.adaptive_min_pick_fraction and len(cluster_event_ids) > 1:
            dl_method_ids = {m.upper() for m in associator_cfg.dl_method_ids}
            dl_probas = [
                p.proba
                for p in cluster
                if p.method is not None
                and isinstance(p.method, str)
                and p.method.upper() in dl_method_ids
            ]
            median_proba = statistics.median(dl_probas) if dl_probas else 1.0
            effective_min_pick_fraction = max(
                associator_cfg.min_pick_fraction_floor,
                associator_cfg.min_pick_fraction * median_proba,
            )
            logger.info(
                f"Cluster#{i}: {len(cluster_event_ids)} distinct event_ids detected, "
                f"median DL proba={median_proba:.3f} ({len(dl_probas)} DL picks), "
                f"reducing min_pick_fraction: {associator_cfg.min_pick_fraction} -> {effective_min_pick_fraction:.3f} "
                f"(floor={associator_cfg.min_pick_fraction_floor})"
            )
        else:
            effective_min_pick_fraction = associator_cfg.min_pick_fraction

        # Step 1: pre-filter stations to max_lat_range/max_lon_range BEFORE computing
        # the range, so that extreme outliers do not corrupt min/max calculations.
        if associator_cfg.max_lat_range is not None:
            mask_out_lat = (stations["latitude"] < associator_cfg.max_lat_range[0]) | (
                stations["latitude"] > associator_cfg.max_lat_range[1]
            )
            for row in stations.loc[mask_out_lat].itertuples():
                logger.warning(
                    f"Station {row.id} at lat: {row.latitude}, lon: {row.longitude} "
                    "is outside max_lat_range and will be excluded."
                )
            stations = stations[~mask_out_lat].reset_index(drop=True)

        if associator_cfg.max_lon_range is not None:
            mask_out_lon = (stations["longitude"] < associator_cfg.max_lon_range[0]) | (
                stations["longitude"] > associator_cfg.max_lon_range[1]
            )
            for row in stations.loc[mask_out_lon].itertuples():
                logger.warning(
                    f"Station {row.id} at lat: {row.latitude}, lon: {row.longitude} "
                    "is outside max_lon_range and will be excluded."
                )
            stations = stations[~mask_out_lon].reset_index(drop=True)

        if stations.empty:
            logger.warning(
                f"Cluster#{i}: no stations left after range filtering, skipping."
            )
            continue

        # Filter picks to only keep those whose station is still in the stations df
        valid_station_ids = set(stations["id"])
        picks = picks[picks["station"].isin(valid_station_ids)].reset_index(drop=True)
        if picks.empty:
            logger.warning(
                f"Cluster#{i}: no picks left after station range filtering, skipping."
            )
            continue

        # Step 2: define a safe range around the remaining stations coordinates
        range_percent = 0.01  # 1%
        lat_safe_range_deg = range_percent * (
            stations["latitude"].max() - stations["latitude"].min()
        )
        lon_safe_range_deg = range_percent * (
            stations["longitude"].max() - stations["longitude"].min()
        )
        logger.debug(
            f"Cluster#{i} safe range lat: {lat_safe_range_deg} deg, "
            f"lon: {lon_safe_range_deg} deg"
        )

        lat_range = (
            stations["latitude"].min() - lat_safe_range_deg,
            stations["latitude"].max() + lat_safe_range_deg,
        )
        lon_range = (
            stations["longitude"].min() - lon_safe_range_deg,
            stations["longitude"].max() + lon_safe_range_deg,
        )

        logger.info(f"range lat: {lat_range}, lon: {lon_range}")

        try:
            associator = pyocto.OctoAssociator.from_area(
                lat=lat_range,
                lon=lon_range,
                #time_slicing=10 * 60, 
                zlim=associator_cfg.zlim,
                time_before=associator_cfg.time_before,  # should be greater than dbclust time_window parameter
                max_pick_overlap=associator_cfg.max_pick_overlap,
                min_pick_fraction=effective_min_pick_fraction,
                min_node_size=associator_cfg.min_node_size,  # default 10
                min_node_size_location=associator_cfg.min_node_size_location,  # default 1.5
                velocity_model=velocity_model,
                pick_match_tolerance=associator_cfg.pick_match_tolerance,
                min_interevent_time=0.4,  # default 3
                n_picks=associator_cfg.n_picks,
                n_p_picks=associator_cfg.n_p_picks,
                n_s_picks=associator_cfg.n_s_picks,
                n_p_and_s_picks=associator_cfg.n_p_and_s_picks,
                exponential_edt=True,
                location_split_depth=6,  # default 6
                location_split_return=4,  # default 4
                refinement_iterations=3,  # default 3
                second_pass_overwrites={
                    "time_before": associator_cfg.time_before,
                    "n_picks": associator_cfg.n_picks,
                    "n_p_picks": associator_cfg.n_p_picks,
                    "n_s_picks": associator_cfg.n_s_picks,
                    "n_p_and_s_picks": associator_cfg.n_p_and_s_picks,
                    "iterations": 1,
                },
            )
        except pyproj.exceptions.ProjError as e:
            # Skip processing if projection error occurs (e.g. stations too far away)
            logger.error(f"Projection error in OctoAssociator.from_area(): {e}")
            logger.error(
                f"Check stations coordinates ! lat_range: {lat_range}, lon_range: {lon_range}"
            )
            logger.info(f"picks: {picks}")
            raise

        try:
            associator.transform_stations(stations)
        except pyproj.exceptions.ProjError as e:
            logger.error(f"Projection error in transform_stations(): {e}")
            logger.error(
                f"Check stations coordinates ! lat_range: {lat_range}, lon_range: {lon_range}"
            )
            raise

        # Associate picks and generate events
        events, assignments = associator.associate(picks, stations)
        if len(events):
            associator.transform_events(events)
            events["time"] = events["time"].apply(
                datetime.datetime.fromtimestamp, tz=datetime.timezone.utc
            )

        # Store events and update clusters
        pyocto_preloc.extend(get_events_list(events, assignments, stations, model_name))
        if associator_cfg.min_ps_ratio is not None:
            filtered_clusters = []
            for c in get_clusters_from_assignment(cluster, events, assignments):
                station_phases = defaultdict(set)
                for p in c:
                    if not p.phase:
                        continue
                    station_code = f"{p.network}.{p.station}"
                    if p.phase.upper().startswith("P"):
                        station_phases[station_code].add("P")
                    elif p.phase.upper().startswith("S"):
                        station_phases[station_code].add("S")
                total_stations = len(station_phases)
                stations_with_both = sum(
                    1
                    for phases in station_phases.values()
                    if "P" in phases and "S" in phases
                )
                ps_ratio = (
                    stations_with_both / total_stations if total_stations > 0 else 0.0
                )
                if ps_ratio >= associator_cfg.min_ps_ratio:
                    filtered_clusters.append(c)
                else:
                    logger.info(
                        f"PyOcto cluster#{i} filtered: ps_ratio"
                        f" {stations_with_both}/{total_stations} stations with P+S"
                        f" = {ps_ratio:.2f} < {associator_cfg.min_ps_ratio}"
                    )
            pyocto_clusters.extend(filtered_clusters)
        else:
            pyocto_clusters.extend(
                get_clusters_from_assignment(cluster, events, assignments)
            )

        assigned_pick_ids = (
            set(assignments["pick_idx"].to_list()) if len(assignments) else set()
        )
        n_unassigned = len(cluster) - len(assigned_pick_ids)
        logger.info(
            f"\t{len(events)} events found in cluster#{i} with {len(cluster)} picks"
            f" ({n_unassigned} unassigned by PyOcto)"
        )

    # Merge clusters with common picks or event IDs
    pyocto_clusters, pyocto_preloc = cluster_merge(
        pyocto_clusters,
        pyocto_preloc,
        min_com_phases,
        eventid_shared_min_picks_per_cluster=myclust.eventid_shared_min_picks_per_cluster,
        eventid_shared_min_distinct_ids=myclust.eventid_shared_min_distinct_ids,
    )

    logger.info(
        f"PyOcto found {len(pyocto_clusters)} clusters, dbclust found {myclust.n_clusters} clusters."
    )

    if len(pyocto_clusters) == 0 and myclust.n_clusters > 0:
        if delegate_dbclust:
            # fixme: for each cluster add a preloc based on the barycenter of the stations
            logger.info(
                "PyOcto did not find any cluster. Returning original dbclust clusters."
            )
            return myclust
        return None

    # Shallow-copy the Clusterize object: all mutable attributes are immediately
    # overwritten below, so a deepcopy of the original clusters is not needed.
    newclust = copy.copy(myclust)
    newclust.clusters = pyocto_clusters
    newclust.n_clusters = len(newclust.clusters)
    newclust.clusters_stability = [1] * newclust.n_clusters  # unused but needed
    newclust.preloc = pyocto_preloc  # used to choose NLL velocity model

    # Aggregate picks into clusters with shared event IDs
    # If multiple event_ids share the same agency, attach the partial result to the
    # exception so that the caller can use it as a fallback.
    try:
        pyocto_clusters = aggregate_pick_to_cluster_with_common_event_id(
            pyocto_clusters, all_picks_list, min_com_phases
        )
    except MultipleEventIDsWithSameAgencyError as e:
        e.partial_result = newclust
        raise

    newclust.clusters = pyocto_clusters
    newclust.n_clusters = len(newclust.clusters)

    # Clean up the original cluster object only after successful processing
    for attr in ["clusters", "clusters_stability", "noise", "zones", "preloc"]:
        if hasattr(myclust, attr):
            delattr(myclust, attr)

    return newclust


def cluster_merge(
    clusters: List[List[Phase]],
    preloc,
    min_com_phases: int,
    eventid_shared_min_picks_per_cluster: Optional[int] = None,
    eventid_shared_min_distinct_ids: int = 2,
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
            clusters,
            preloc,
            min_com_phases,
            eventid_shared_min_picks_per_cluster=eventid_shared_min_picks_per_cluster,
            eventid_shared_min_distinct_ids=eventid_shared_min_distinct_ids,
        )
        if merge_count == 0:
            break
    return clusters, preloc


def cluster_merge_one_pass(
    clusters: List[List[Phase]],
    preloc: List,
    min_com_phases: int,
    eventid_shared_min_picks_per_cluster: Optional[int] = None,
    eventid_shared_min_distinct_ids: int = 2,
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
        eventid_shared = cluster_share_eventid(
            c1,
            c2,
            shared_threshold=min_com_phases,
            min_picks_per_cluster=eventid_shared_min_picks_per_cluster,
            min_distinct_event_ids=eventid_shared_min_distinct_ids,
        )

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
            counts_str = ", ".join(
                f"{eid.split('/')[-1]}({count} picks {'> threshold, will aggregate' if count > pick_count_threshold else f'<= threshold({pick_count_threshold}), skipped'})"
                for eid, count in event_id_counts.most_common()
            )
            logger.info(f"Cluster has picks from known event(s): {counts_str}")

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
            logger.info(f"agency_event_map: {agency_event_map}")

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

        # Partition picks into those added to cluster and those remaining
        picks_to_add = [
            p
            for p in picks
            if p.event_id
            and p.event_id in event_id_counts
            and event_id_counts[p.event_id] > pick_count_threshold
        ]
        cluster.extend(picks_to_add)
        added = set(id(p) for p in picks_to_add)
        picks = [p for p in picks if id(p) not in added]

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
