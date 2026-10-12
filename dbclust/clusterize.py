#!/usr/bin/env python
import functools
import json
import logging
import os
import sys
from collections import Counter, defaultdict
from itertools import chain
from math import pow
from math import sqrt
from typing import List
from typing import Optional
from typing import Tuple

import hdbscan
import numpy as np
import pandas as pd
from obspy import Catalog
from obspy.core.event import Comment
from obspy.core.event import Event
from obspy.core.event import Origin
from obspy.geodetics import gps2dist_azimuth

from dbclust.phase import import_phases
from dbclust.phase import Phase
from dbclust.phase import quakeml_pick_identity
from dbclust.quakeml import deduplicate_picks
# import dask.bag as db

# default logger (uses hierarchical name for selective level control)
logger = logging.getLogger("dbclust.clusterize")


# Distance (s) given to same-station P-S pairs that cannot belong to the same
# event: far above any max_search_dist, still finite for HDBSCAN.
TT_FAR = 1e6


@functools.lru_cache(maxsize=None)
def compute_tt(p1: Phase, p2: Phase, vmean) -> float:
    # lru_cache doesn't work with multiprocessing/dask/etc.
    distance, az, baz = gps2dist_azimuth(
        p1.coord["latitude"],
        p1.coord["longitude"],
        p2.coord["latitude"],
        p2.coord["longitude"],
    )
    # distance in meters, convert it to km
    distance = distance / 1000.0
    dd = distance / vmean
    dt = p1.time - p2.time
    tt = sqrt(pow(dt, 2) + pow(dd, 2))
    return tt


def preloc_pseudo_tt(preloc1, preloc2, vmean) -> Optional[tuple]:
    """Separation of two PyOcto prelocations: (dist_km, dt_s, tt_s).

    tt_s = sqrt(dt_s^2 + (dist_km / vmean)^2), same metric as compute_tt() but
    between two origins instead of two picks (so it is not comparable with
    max_search_dist). Returns None when either prelocation is missing or
    incomplete (e.g. clusters returned as-is by delegate_dbclust).
    """
    try:
        distance, _, _ = gps2dist_azimuth(
            float(preloc1["latitude"]), float(preloc1["longitude"]),
            float(preloc2["latitude"]), float(preloc2["longitude"]),
        )
        dt_s = abs(preloc1["time"].timestamp() - preloc2["time"].timestamp())
    except (TypeError, KeyError, AttributeError, ValueError):
        return None
    dist_km = distance / 1000.0
    return dist_km, dt_s, sqrt(pow(dt_s, 2) + pow(dist_km / vmean, 2))


def cluster_share_eventid(
    c1: List[Phase],
    c2: List[Phase],
    shared_threshold: int = 3,
    min_picks_per_cluster: Optional[int] = None,
    min_distinct_event_ids: int = 2,
) -> bool:
    """
    Check if two clusters share a common event ID, considering station and phase thresholds.

    Args:
        c1 (List[Phase]): The first cluster.
        c2 (List[Phase]): The second cluster.
        shared_threshold (int): Minimum number of event_id shared to consider merging.

    Returns:
        bool: True if the clusters share a common event ID meeting the threshold, False otherwise.

    Fixme:
        when event are overlapping a pick could be not well associated with the right event leading to a merge
        that would not be correct. Using thresholds is a way to avoid this issue but it is not a perfect solution.
    """

    # Extract event IDs for each cluster
    c1_event_ids = [p.event_id for p in c1 if p.event_id]
    c2_event_ids = [p.event_id for p in c2 if p.event_id]

    # Find common event IDs and count them
    common_event_ids = set(c1_event_ids) & set(c2_event_ids)

    # Count occurrences in each cluster
    c1_counts = Counter(c1_event_ids)
    c2_counts = Counter(c2_event_ids)

    # Guard against contamination: at least one cluster must have >= shared_threshold
    # picks with the shared event_id (strong support), and the other must have >= 1
    # (asymmetric, to handle HDBSCAN fragments where one cluster gets most picks).
    if min_picks_per_cluster is None:
        min_picks_per_cluster = shared_threshold

    shared_counts = {
        event_id: min(c1_counts[event_id], c2_counts[event_id])
        for event_id in common_event_ids
        if max(c1_counts[event_id], c2_counts[event_id]) >= min_picks_per_cluster
        and min(c1_counts[event_id], c2_counts[event_id]) >= 1
    }

    for event_id, count in shared_counts.items():
        logger.debug(f"Event ID {event_id}: {count} shared")

    return len(shared_counts) >= max(1, min_distinct_event_ids)


def get_picks_from_event(event: Event, origin: Origin, time) -> List:
    # station_id,phase_type,phase_time
    # 1K.OFAS0.00.EH.D,P,2023-02-13T18:30:58.558999Z
    lines = []
    for arrival in origin.arrivals:
        if arrival.time_weight is not None and arrival.time_residual is not None:
            pick = next(
                (p for p in event.picks if p.resource_id == arrival.pick_id), None
            )
            if pick and (time is None or pick.time >= time):
                line = [
                    pick.waveform_id.get_seed_string(),
                    pick.phase_hint,
                    pick.time,
                ]
                lines.append(line)
    return sorted(lines, key=lambda l: l[2])


def feed_picks_probabilities(cat: Catalog, clusters: List[List[Phase]]) -> None:
    """Attach a {"probability": {...}} comment to each obspy Pick matching a clustered Phase.

    A Phase from `clusters` matches a Pick when they share the same station code and
    pick time (network and phase_hint are intentionally not compared, since relabelling
    e.g. P -> Pg changes pick.phase_hint but not the clustered Phase's .phase). Matching
    is done via a (station, time) -> [Phase, ...] index built once up front, instead of
    rescanning every clustered Phase for every pick (previously O(n_picks * n_phases)).
    """
    # Do NOT dedupe via set(chain(*clusters)): Phase.__hash__/__eq__
    # intentionally ignore location/channel and reduce the phase label to its
    # family (see Phase.__hash__ docstring), so two physically distinct picks
    # (different channel/location, or different exact phase label from
    # different agencies) sharing (network, station, phase family, time) would
    # collapse into one, silently dropping a probability annotation on the
    # other pick. Dedupe explicitly on the same key used for the lookup below
    # instead.
    phases_by_station_time = defaultdict(list)
    seen: dict = {}
    for p in chain(*clusters):
        key = (p.network, p.station, p.phase[0].upper(), p.time.datetime)
        if key not in seen:
            seen[key] = p
            phases_by_station_time[(p.station, p.time.datetime)].append(p)

    for event in cat:
        for pick in event.picks:
            key = (pick.waveform_id["station_code"], pick.time.datetime)
            for p in phases_by_station_time.get(key, []):
                if pick.waveform_id["network_code"] != p.network:
                    logger.warning(
                        f"Check your inventory for station {p.station}, 2 networks defined : "
                        f"[{pick.waveform_id['network_code']},{p.network}] "
                    )
                if p.agency:
                    agency = p.agency
                else:
                    agency = "undefined"
                pick.comments.append(
                    Comment(
                        text='{"probability": {"name": "%s", "value": %.2f}}'
                        % (agency, p.proba)
                    )
                )


def feed_picks_event_ids(cat: Catalog, clusters: List[List[Phase]]) -> None:
    pick_to_cluster = {}
    for c in clusters:
        cluster_event_ids = list(set([p.event_id for p in c if p.event_id]))
        for p in c:
            pick_to_cluster[p.identity()] = cluster_event_ids

    for event in cat:
        o = event.preferred_origin()
        event_ids = []
        if o is None:
            logger.warning(
                f"feed_picks_event_ids: event {event.resource_id.id if event.resource_id else '?'} "
                "has no preferred_origin(), skipping event_id lookup."
            )
            event.comments.append(Comment(text='{"event_ids": %s}' % json.dumps(event_ids)))
            continue
        for a in o.arrivals:
            if a.time_weight is None or a.time_residual is None:
                continue
            pick = next((p for p in event.picks if p.resource_id == a.pick_id), None)
            if pick is None:
                continue
            key = quakeml_pick_identity(
                pick.waveform_id["network_code"],
                pick.waveform_id["station_code"],
                pick.phase_hint,
                pick.time,
            )
            if key in pick_to_cluster:
                event_ids = pick_to_cluster[key]
                break

        event.comments.append(Comment(text='{"event_ids": %s}' % json.dumps(event_ids)))


def feed_cluster_stability(
    cat: Catalog,
    clusters: List[List[Phase]],
    clusters_stability,
    clusters_pyocto_ops=None,
) -> None:
    """Inject cluster stability and PyOcto operation metadata as a JSON comment on each Event."""
    pick_to_cluster_idx = {}
    for idx, c in enumerate(clusters):
        for p in c:
            pick_to_cluster_idx[p.identity()] = idx

    for event in cat:
        o = event.preferred_origin()
        if o is None:
            continue
        for a in o.arrivals:
            if a.time_weight is None or a.time_residual is None:
                continue
            pick = next((p for p in event.picks if p.resource_id == a.pick_id), None)
            if pick is None:
                continue
            key = quakeml_pick_identity(
                pick.waveform_id["network_code"],
                pick.waveform_id["station_code"],
                pick.phase_hint,
                pick.time,
            )
            idx = pick_to_cluster_idx.get(key)
            if idx is not None and idx < len(clusters_stability):
                stability = float(clusters_stability[idx])
                meta: dict = {"cluster_stability": round(stability, 4)}
                if clusters_pyocto_ops and idx < len(clusters_pyocto_ops):
                    op_info = clusters_pyocto_ops[idx]
                    meta["cluster_op"] = op_info.get("op", "unknown")
                    parent_stabs = op_info.get("parent_stabilities", [])
                    if len(parent_stabs) > 1:
                        meta["hdbscan_parent_stabilities"] = [round(s, 4) for s in parent_stabs]
                    else:
                        meta["hdbscan_parent_stability"] = round(parent_stabs[0], 4) if parent_stabs else stability
                event.comments.append(Comment(text=json.dumps(meta)))
                break


def merge_cluster_with_common_phases(
    clusters1,
    clusters2,
    min_com_phases: int,
    eventid_shared_min_picks_per_cluster: Optional[int] = None,
    eventid_shared_min_distinct_ids: int = 2,
) -> Tuple:
    """
    Merge into `clusters1` all clusters from `clusters2` with common phases
    or shared event IDs.

    Args:
        clusters1: Object containing `clusters1.clusters` (list of Phase lists).
        clusters2: Object containing `clusters2.clusters` (list of Phase lists).
        min_com_phases: Minimum number of shared phases required for merging.

    Returns:
        Tuple: (updated clusters1, updated clusters2, number of merges performed).
    """
    new_clusters2 = []
    new_stability2 = []
    merge_count = 0

    logger.debug(
        "merge_cluster_with_common_phases: clusters1 contains %d clusters.",
        len(clusters1.clusters),
    )
    logger.debug(
        "merge_cluster_with_common_phases: clusters2 contains %d clusters.",
        len(clusters2.clusters),
    )

    # Traverse order: least stable first so that fragile clusters absorb/get absorbed
    # before stable ones — unstable c1 need picks most, unstable c2 should be merged first.
    # Tie-break by number of picks ascending: smaller clusters are processed first,
    # giving them priority to claim shared picks before larger clusters do.
    if len(clusters2.clusters_stability) == len(clusters2.clusters):
        c2_sizes = np.array([len(c) for c in clusters2.clusters], dtype=float)
        c2_order = np.lexsort((c2_sizes, clusters2.clusters_stability))
    else:
        c2_order = np.arange(len(clusters2.clusters))

    if len(clusters1.clusters_stability) == len(clusters1.clusters):
        c1_sizes = np.array([len(c) for c in clusters1.clusters], dtype=float)
        c1_order = np.lexsort((c1_sizes, clusters1.clusters_stability))
    else:
        c1_order = np.arange(len(clusters1.clusters))

    for j in c2_order:
        c2 = clusters2.clusters[j]
        merged = False
        c2_times = sorted(p.time for p in c2)
        c2_t0 = c2_times[0] if c2_times else None
        c2_t1 = c2_times[-1] if c2_times else None
        c2_eids = [p.event_id for p in c2 if p.event_id]
        logger.debug(
            f"merge_cluster_with_common_phases: c2 cluster [{c2_t0} .. {c2_t1}] "
            f"{len(c2)} picks, {len(c2_eids)} with event_id"
        )
        for i in c1_order:
            c1 = clusters1.clusters[i]
            # Count common phases using hash equality
            common_count = sum((Counter(c1) & Counter(c2)).values())
            # Also count spatiotemporal matches (station+phase+time, ignoring event_id)
            # Used as fallback when Phase.__hash__ differs due to event_id mismatch
            c1_keys = {
                (p.network, p.station, p.phase[0].upper(), p.time.datetime) for p in c1
            }
            c2_keys = {
                (p.network, p.station, p.phase[0].upper(), p.time.datetime) for p in c2
            }
            spatio_count = len(c1_keys & c2_keys)
            c1_times = sorted(p.time for p in c1)
            c1_t0 = c1_times[0] if c1_times else None
            c1_t1 = c1_times[-1] if c1_times else None
            c1_eids = [p.event_id for p in c1 if p.event_id]
            logger.debug(
                f"  vs c1[{i}] [{c1_t0} .. {c1_t1}] {len(c1)} picks, {len(c1_eids)} with event_id "
                f"=> hash_common={common_count}, spatio_common={spatio_count}"
            )

            # Check for shared event IDs
            eventid_shared = cluster_share_eventid(
                c1,
                c2,
                shared_threshold=min_com_phases,
                min_picks_per_cluster=eventid_shared_min_picks_per_cluster,
                min_distinct_event_ids=eventid_shared_min_distinct_ids,
            )

            # Merge clusters if conditions are met
            # spatio_count is used as fallback when Phase.__hash__ differs due to event_id mismatch
            if (
                common_count >= min_com_phases
                or spatio_count >= min_com_phases
                or eventid_shared
            ):
                logger.debug(
                    f"Merging cluster from clusters2 into clusters1: "
                    f"picks shared: {common_count}, spatio_shared: {spatio_count}, event ID shared: {eventid_shared}"
                )
                # Deduplicate by physical pick identity, preferring picks with event_id over those without
                seen: dict = {}
                for p in c1 + c2:
                    key = (p.network, p.station, p.phase[0].upper(), p.time.datetime)
                    if key not in seen or (
                        seen[key].event_id is None and p.event_id is not None
                    ):
                        seen[key] = p
                # Capture pre-merge size before mutating c1 in place, otherwise
                # c1_size would already include c2's contribution.
                c1_size = len(c1)
                c1[:] = list(seen.values())  # Update in place
                if (
                    len(clusters1.clusters_stability) > i
                    and len(clusters2.clusters_stability) > j
                ):
                    # Use size-weighted average for consistency with cluster_merge_based_on_eventid()
                    c2_size = len(c2)
                    total_size = c1_size + c2_size
                    clusters1.clusters_stability[i] = (
                        (clusters1.clusters_stability[i] * c1_size +
                         clusters2.clusters_stability[j] * c2_size) / total_size
                    )
                merge_count += 1
                merged = True
                break

        if not merged:
            new_clusters2.append(c2)
            if len(clusters2.clusters_stability) > j:
                new_stability2.append(clusters2.clusters_stability[j])

    # Update clusters2 attributes
    clusters2.clusters = new_clusters2
    clusters2.n_clusters = len(new_clusters2)
    clusters2.clusters_stability = (
        np.array(new_stability2, dtype=float)
        if new_stability2
        else np.ones(0, dtype=float)
    )

    clusters1.n_clusters = len(clusters1.clusters)

    logger.debug(
        "merge_cluster_with_common_phases: Total merges performed: %d", merge_count
    )

    return clusters1, clusters2, merge_count


class Clusterize(object):
    def __init__(
        self,
        phases=None,
        min_cluster_size=5,  # hdbscan default
        average_velocity=5,  # km/s
        min_station_count=0,
        min_station_with_P_and_S=2,
        min_station_score=None,
        min_ps_ratio=None,
        force_keep_catalog_events=False,
        max_search_dist=0,  # same as hdbscan cluster_selection_epsilon: default is 0.
        P_uncertainty=0.1,
        S_uncertainty=0.2,
        min_com_phases=3,
        eventid_shared_min_picks_per_cluster=None,
        eventid_shared_min_distinct_ids=2,
        eventid_merge_max_tt_s=None,  # refuse event_id-only PyOcto merges beyond this preloc pseudo travel time (s)
        eventid_merge_moveout_min_fraction=0.95,  # moveout overrule of a refusal: min fraction of compatible pairs
        eventid_merge_moveout_min_picks=3,  # ... and min checkable picks on each side
        eventid_merge_moveout_vp=6.0,  # km/s, P bound of the moveout check
        eventid_merge_moveout_vs=3.5,  # km/s, S bound of the moveout check
        tt_matrix_fname="tt_matrix.npy",
        tt_matrix_load=False,
        tt_matrix_save=False,
        zones=None,
        apparent_vp=6.0,  # apparent P-wave velocity for the TT matrix (km/s)
        apparent_vs=3.5,  # apparent S-wave velocity for the TT matrix (km/s)
        cluster_selection_method="eom",  # HDBSCAN: "eom" (default) or "leaf"
        allow_single_cluster=True,        # HDBSCAN: allow a single cluster (False → noise if no structure)
        tt_clip_seconds=0.0,  # clip TT matrix to this value in seconds (0 = no clip)
        clustering_method="hdbscan",    # "hdbscan" or "leiden"
        leiden_resolution=0.05,         # CPM resolution γ (higher → more clusters)
        leiden_edge_weight_scale=None,  # σ for exp(-d/σ); None → max_search_dist/2
        leiden_ps_boost_factor=100.0,        # multiplicative boost for same-station P-S edges
        leiden_min_edge_weight=0.0,          # drop edges below this weight (0 = disabled)
        leiden_vp=6.0,                       # P-wave velocity for moveout compatibility filter
        leiden_vs=3.5,                       # S-wave velocity for moveout compatibility filter
        mega_cluster_fallback_leiden=False,  # re-cluster mega-clusters with Leiden
        mega_cluster_threshold=0.8,          # fraction of picks to trigger mega-cluster
        mega_cluster_min_size=150,           # minimum absolute size to trigger
        mega_cluster_leiden_resolution=0.1,  # Leiden resolution for mega-cluster fallback
        leiden_hdbscan_fallback=False,       # run HDBSCAN on Leiden noise + unstable clusters
        leiden_min_stability=0.0,            # clusters below this stability go to HDBSCAN pool
        min_station_with_P_and_S_stability_override=0.0,  # bypass P+S check if stability > this
        min_station_with_P_and_S_score_override=0.0,      # bypass P+S check if station_score >= this
        # --- Reserve S-pick consolidation (opt-in, see ReservePickConfig) ---
        # Read-only reserve pool (S-band Phase objects); never added to
        # cluster pick sets — only used to bypass the pre-NLL
        # min_station_with_P_and_S gate when a P-only station has a
        # plausible reserve S candidate. The definitive decision is the
        # post-NLL weighted gate (NllLoc).
        reserve_enabled=False,
        reserve_tt_tolerance_s=1.0,
        reserve_max_station_dist_km=300.0,  # bounds the S-P delay of the bypass window
        reserve_vp=6.0,  # homogeneous velocities (km/s) of the S-P delay bound;
        reserve_vs=3.5,  # independent from leiden_vp/vs (0.0 = moveout off)
        reserve_pool=None,
    ):
        # clusters is a list of cluster :
        # ie. [ [phases, label], ... ]
        # noise is [ phases, -1]
        self.clusters = []
        self.clusters_stability = np.array([])
        self.n_clusters = 0
        self.noise = []
        self.n_noise = 0
        self.preloc = None  # pre-localization if pyocto was enable
        self.zones = zones
        self._deferred_cluster_indices: set = set()  # indices of injected deferred clusters
        self.clusters_pyocto_ops = None  # set by process_clusters_with_pyocto when PyOcto is used
        self._context_tag = ""  # optional window/job label set by generate_nllobs(), used in log lines

        # clustering parameters
        self.max_search_dist = max_search_dist
        self.min_cluster_size = min_cluster_size
        self.average_velocity = average_velocity

        # stations filtering parameters
        self.min_station_count = min_station_count
        self.min_station_with_P_and_S = min_station_with_P_and_S
        self.min_station_score = min_station_score
        self.min_ps_ratio = min_ps_ratio
        self.force_keep_catalog_events = force_keep_catalog_events

        # pick filtering parameters
        self.P_uncertainty = P_uncertainty
        self.S_uncertainty = S_uncertainty
        self.min_com_phases = min_com_phases
        self.eventid_shared_min_picks_per_cluster = eventid_shared_min_picks_per_cluster
        self.eventid_shared_min_distinct_ids = eventid_shared_min_distinct_ids
        self.eventid_merge_max_tt_s = eventid_merge_max_tt_s
        self.eventid_merge_moveout_min_fraction = eventid_merge_moveout_min_fraction
        self.eventid_merge_moveout_min_picks = eventid_merge_moveout_min_picks
        self.eventid_merge_moveout_vp = eventid_merge_moveout_vp
        self.eventid_merge_moveout_vs = eventid_merge_moveout_vs

        # tt_matrix load/save parameters
        self.tt_matrix_fname = tt_matrix_fname
        self.tt_matrix_load = tt_matrix_load
        self.apparent_vp = apparent_vp
        self.apparent_vs = apparent_vs
        self.cluster_selection_method = cluster_selection_method
        self.allow_single_cluster = allow_single_cluster
        self.tt_clip_seconds = tt_clip_seconds
        self.clustering_method = clustering_method
        self.leiden_resolution = leiden_resolution
        self.leiden_edge_weight_scale = leiden_edge_weight_scale
        self.leiden_ps_boost_factor = leiden_ps_boost_factor
        self.leiden_min_edge_weight = leiden_min_edge_weight
        self.leiden_vp = leiden_vp
        self.leiden_vs = leiden_vs
        self.mega_cluster_fallback_leiden = mega_cluster_fallback_leiden
        self.mega_cluster_threshold = mega_cluster_threshold
        self.mega_cluster_min_size = mega_cluster_min_size
        self.mega_cluster_leiden_resolution = mega_cluster_leiden_resolution
        self.leiden_hdbscan_fallback = leiden_hdbscan_fallback
        self.leiden_min_stability = leiden_min_stability
        self.min_station_with_P_and_S_stability_override = min_station_with_P_and_S_stability_override
        self.min_station_with_P_and_S_score_override = min_station_with_P_and_S_score_override
        self.reserve_enabled = reserve_enabled
        self.reserve_tt_tolerance_s = reserve_tt_tolerance_s
        self.reserve_max_station_dist_km = reserve_max_station_dist_km
        self.reserve_vp = reserve_vp
        self.reserve_vs = reserve_vs
        # Reserve pool (list of S-band Phase objects). Reassigned per time
        # window by the pipeline (core.py); kept empty by default.
        self.reserve_pool = list(reserve_pool) if reserve_pool else []

        if phases is None:
            # Simple constructor
            return

        logger.info(
            f"Starting Clustering (nb phases={len(phases)}, "
            f"min_cluster_size={min_cluster_size}, "
            f"min_station_with_P_and_S={min_station_with_P_and_S})."
        )
        if len(phases) < min_cluster_size:
            logger.info(f"Too few picks ({len(phases)}/{min_cluster_size})!")
            # add noise points
            self.clusters = []
            self.n_clusters = 0
            self.noise = phases
            self.n_noise = len(phases)
            return

        logger.info("Computing TT matrix.")
        if tt_matrix_load and tt_matrix_fname:
            logger.info(f"Loading tt_matrix {tt_matrix_fname}.")
            try:
                pseudo_tt = np.load(tt_matrix_fname)
            except Exception as e:
                logger.error(e)
                logger.error("Check your config file !")
                raise
        else:
            # vectorized haversine: ~20-100x faster than per-pair gps2dist_azimuth
            pseudo_tt = self.numpy_compute_tt_matrix_vectorized(
                phases, average_velocity,
                vp=self.apparent_vp if self.clustering_method != "leiden" else None,
                vs=self.apparent_vs if self.clustering_method != "leiden" else None,
                max_search_dist=self.max_search_dist,
            )
            if tt_clip_seconds > 0:
                pseudo_tt = np.clip(pseudo_tt, 0.0, tt_clip_seconds)
                logger.info(f"TT matrix clipped to {tt_clip_seconds}s.")

            try:
                logger.info(f"TT matrix: {compute_tt.cache_info()}")
                compute_tt.cache_clear()
            except AttributeError:
                # compute_tt may not have cache_info if lru_cache is not used
                pass

        if tt_matrix_fname and tt_matrix_save:
            logger.info(f"Saving tt_matrix {tt_matrix_fname}.")
            np.save(tt_matrix_fname, pseudo_tt)

        self.clusters, self.clusters_stability, self.noise = self.get_clusters(
            phases,
            pseudo_tt,
            max_search_dist,
            min_cluster_size,
            average_velocity=average_velocity,
            cluster_selection_method=cluster_selection_method,
            allow_single_cluster=allow_single_cluster,
            clustering_method=clustering_method,
            leiden_resolution=leiden_resolution,
            leiden_edge_weight_scale=leiden_edge_weight_scale,
            leiden_ps_boost_factor=leiden_ps_boost_factor,
            leiden_min_edge_weight=leiden_min_edge_weight,
            leiden_vp=leiden_vp,
            leiden_vs=leiden_vs,
            mega_cluster_fallback_leiden=mega_cluster_fallback_leiden,
            mega_cluster_threshold=mega_cluster_threshold,
            mega_cluster_min_size=mega_cluster_min_size,
            mega_cluster_leiden_resolution=mega_cluster_leiden_resolution,
            leiden_hdbscan_fallback=leiden_hdbscan_fallback,
            leiden_min_stability=leiden_min_stability,
        )
        self.n_clusters = len(self.clusters)
        self.n_noise = len(self.noise)
        self.max_search_dist = max_search_dist

        del pseudo_tt
        self.cluster_merge_based_on_eventid()

    def phases_count(self):
        """
        Count the number of phases in the clusters.
        """
        return sum(len(cluster) for cluster in self.clusters)

    def absorb_deferred_cluster(
        self,
        deferred_phases: list,
        overlap_seconds: float = 0.0,
    ):
        """Insert a pre-formed deferred cluster and enrich it with nearby picks.

        Parameters
        ----------
        deferred_phases : list[Phase]
            Phases from the pre-formed cluster to inject.
        overlap_seconds : float
            Half-width of the temporal enrichment window in seconds.  Only picks
            whose time falls within [t_min - overlap, t_max + overlap] of the
            deferred cluster are candidates.  0 means no temporal pre-filter.
        """
        if not deferred_phases:
            return

        # deferred_phases come from a prior Clusterize instance, so their Phase
        # objects are never physically present in self.clusters — no extraction needed.
        deferred_keys = {p.identity() for p in deferred_phases}

        # Remove matching picks (by key) from all existing clusters so the deferred
        # cluster becomes the sole owner of these picks and is not re-localized later
        # via the mega-cluster that originally contained them.
        n_extracted = 0
        for cluster in self.clusters:
            to_remove = [
                p
                for p in cluster
                if p.identity() in deferred_keys
            ]
            for p in to_remove:
                cluster.remove(p)
                n_extracted += 1

        new_cluster: list = list(deferred_phases)
        self._deferred_cluster_ref = (
            new_cluster  # exposed for post-localization cleanup
        )
        cluster_idx = len(self.clusters)
        self._deferred_cluster_indices.add(cluster_idx)
        self.clusters.append(new_cluster)
        self.n_clusters = len(self.clusters)
        self.clusters_stability = np.concatenate(
            [
                np.atleast_1d(np.array(self.clusters_stability, dtype=float)),
                np.array([1.0], dtype=float),
            ]
        )
        logger.info(
            f"[deferred] Inserted pre-formed cluster #{cluster_idx}"
            f" with {len(new_cluster)} phases"
            f" (extracted {n_extracted} picks from existing clusters)."
        )

        # Temporal bounds of the deferred cluster, extended by overlap_seconds.
        t_min = min(float(p.time) for p in new_cluster)
        t_max = max(float(p.time) for p in new_cluster)
        t_lo = t_min - overlap_seconds
        t_hi = t_max + overlap_seconds

        def _is_candidate(p) -> bool:
            if p.identity() in deferred_keys:
                return False
            if overlap_seconds > 0 and not (t_lo <= float(p.time) <= t_hi):
                return False
            return True

        def _tt_to_cluster(p) -> float:
            c = self.clusters[cluster_idx]
            if not c:
                return float("inf")
            return min(compute_tt(p, cp, self.average_velocity) for cp in c)

        # Enrich from noise
        remaining_noise = []
        n_from_noise = 0
        for p in list(self.noise):
            if _is_candidate(p) and _tt_to_cluster(p) <= self.max_search_dist:
                self.clusters[cluster_idx].append(p)
                n_from_noise += 1
            else:
                remaining_noise.append(p)
        self.noise = remaining_noise
        self.n_noise = len(remaining_noise)

        # Enrich from other clusters (only temporally eligible picks).
        # Skip other deferred clusters — their picks belong to a different event
        # and must not be stolen by this one.
        n_from_clusters = 0
        for ci, cluster in enumerate(self.clusters):
            if ci == cluster_idx or ci in self._deferred_cluster_indices:
                continue
            to_move = [
                p
                for p in cluster
                if _is_candidate(p) and _tt_to_cluster(p) <= self.max_search_dist
            ]
            if to_move:
                moved_ids = {id(p) for p in to_move}
                cluster[:] = [p for p in cluster if id(p) not in moved_ids]
                self.clusters[cluster_idx].extend(to_move)
                n_from_clusters += len(to_move)

        logger.info(
            f"[deferred] Enriched cluster #{cluster_idx}:"
            f" +{n_from_noise} from noise, +{n_from_clusters} from other clusters"
            f" (time window ±{overlap_seconds:.0f}s)."
        )

        # Pull picks from other clusters sharing a known event_id with the
        # deferred cluster. This catches catalog picks that were not temporally
        # close enough to be caught by the TT enrichment above but belong to
        # the same physical event.
        known_event_ids = {
            p.event_id for p in self.clusters[cluster_idx] if p.event_id
        }
        n_from_event_id = 0
        if known_event_ids:
            # Pull from other clusters, skipping other deferred clusters to avoid
            # stealing picks that belong to a different deferred event.
            for ci, cluster in enumerate(self.clusters):
                if ci == cluster_idx or ci in self._deferred_cluster_indices:
                    continue
                to_move = [p for p in cluster if p.event_id in known_event_ids]
                if to_move:
                    moved_ids = {id(p) for p in to_move}
                    cluster[:] = [p for p in cluster if id(p) not in moved_ids]
                    self.clusters[cluster_idx].extend(to_move)
                    n_from_event_id += len(to_move)
            # Pull from noise: catalog picks landing in noise also belong to
            # the same physical event and would otherwise re-appear as a
            # duplicate in the next window.
            to_move = [p for p in self.noise if p.event_id in known_event_ids]
            if to_move:
                moved_ids = {id(p) for p in to_move}
                self.noise = [p for p in self.noise if id(p) not in moved_ids]
                self.clusters[cluster_idx].extend(to_move)
                n_from_event_id += len(to_move)
            if n_from_event_id:
                logger.info(
                    f"[deferred] Pulled {n_from_event_id} picks by event_id"
                    f" {known_event_ids} from other clusters and noise."
                )

    def build_clusters_from_backward(self, backward_phases, forward_phases):
        """Cluster backward and forward picks together in a single pass.

        Called for the first window of a non-first parallel job, where backward
        overlap picks are available.

        All picks (backward + forward) are pooled into one clustering pass so that
        Leiden/HDBSCAN has a complete view of all picks. This simplifies the workflow
        and ensures events crossing the backward/forward boundary are properly clustered.

        Parameters
        ----------
        backward_phases : list[Phase]
            Picks from [start-overlap, start].
        forward_phases : list[Phase]
            Picks from [start, end].
        """
        all_phases = backward_phases + forward_phases
        logger.info(
            f"[backward+forward] clustering {len(all_phases)} picks"
            f" ({len(backward_phases)} backward + {len(forward_phases)} forward)."
        )

        if len(all_phases) >= self.min_cluster_size:
            pseudo_tt = self.numpy_compute_tt_matrix_vectorized(
                all_phases, self.average_velocity,
                vp=self.apparent_vp if self.clustering_method != "leiden" else None,
                vs=self.apparent_vs if self.clustering_method != "leiden" else None,
                max_search_dist=self.max_search_dist,
            )
            if self.tt_clip_seconds > 0:
                pseudo_tt = np.clip(pseudo_tt, 0.0, self.tt_clip_seconds)
                logger.info(f"TT matrix clipped to {self.tt_clip_seconds}s.")

            max_search_dist = self.max_search_dist

            self.clusters, stab, self.noise = self.get_clusters(
                all_phases, pseudo_tt, max_search_dist,
                self.min_cluster_size, average_velocity=self.average_velocity,
                cluster_selection_method=self.cluster_selection_method,
                clustering_method=self.clustering_method,
                allow_single_cluster=self.allow_single_cluster,
                leiden_resolution=self.leiden_resolution,
                leiden_edge_weight_scale=self.leiden_edge_weight_scale,
                leiden_ps_boost_factor=self.leiden_ps_boost_factor,
                leiden_min_edge_weight=self.leiden_min_edge_weight,
                leiden_vp=self.leiden_vp,
                leiden_vs=self.leiden_vs,
                mega_cluster_fallback_leiden=self.mega_cluster_fallback_leiden,
                mega_cluster_threshold=self.mega_cluster_threshold,
                mega_cluster_min_size=self.mega_cluster_min_size,
                mega_cluster_leiden_resolution=self.mega_cluster_leiden_resolution,
                leiden_hdbscan_fallback=self.leiden_hdbscan_fallback,
                leiden_min_stability=self.leiden_min_stability,
            )
            self.clusters_stability = (
                np.array(stab, dtype=float) if len(stab) > 0
                else np.ones(len(self.clusters))
            )
            self.n_clusters = len(self.clusters)
            self.n_noise = len(self.noise)
            logger.info(
                f"[backward+forward] {self.n_clusters} cluster(s), {self.n_noise} noise."
            )
            if self.n_clusters > 1:
                self.cluster_merge_based_on_eventid()
        else:
            self.noise = list(all_phases)
            self.n_noise = len(self.noise)

    @staticmethod
    def numpy_compute_tt_matrix_vectorized(
        phases, vmean, vp=None, vs=None, max_search_dist=None
    ):
        """Vectorized TT matrix using haversine formula.

        Replaces per-pair gps2dist_azimuth calls with a single NumPy broadcast.
        Haversine error < 0.5% for distances < 2000 km — sufficient for clustering.

        When vp, vs and max_search_dist are provided, same-station P-S pairs
        that cannot belong to the same event are pushed apart (distance
        TT_FAR): S before P, or S-P above the delay of a source at the
        largest distance considered, max_search_dist * vmean (km):
            sp_max = max_search_dist * vmean * (1/vs - 1/vp)
        This prevents spurious links between P and S picks from different
        events that share a station.  Enabled for HDBSCAN; disabled for Leiden
        (Leiden uses PS-boost instead).
        """
        R = 6371.0  # Earth radius in km
        lats = np.radians([p.coord["latitude"] for p in phases])
        lons = np.radians([p.coord["longitude"] for p in phases])
        times = np.array([float(p.time) for p in phases])

        dlat = lats[:, None] - lats[None, :]
        dlon = lons[:, None] - lons[None, :]
        a = (
            np.sin(dlat / 2) ** 2
            + np.cos(lats[:, None]) * np.cos(lats[None, :]) * np.sin(dlon / 2) ** 2
        )
        dist_km = 2 * R * np.arcsin(np.sqrt(a))

        dd = dist_km / vmean
        dt = times[:, None] - times[None, :]
        tt = np.sqrt(dt**2 + dd**2)

        if vp is not None and vs is not None and max_search_dist:
            sp_max = max_search_dist * vmean * (1.0 / vs - 1.0 / vp)
            stations = np.array([f"{p.network}.{p.station}" for p in phases])
            is_p = np.array([p.is_p() for p in phases])
            is_s = np.array([p.is_s() for p in phases])
            # [i, j]: P pick i and S pick j on the same station, S-P = t_j - t_i
            p_then_s = (
                (stations[:, None] == stations[None, :]) & is_p[:, None] & is_s[None, :]
            )
            s_minus_p = -dt
            impossible = p_then_s & ((s_minus_p <= 0) | (s_minus_p > sp_max))
            impossible |= impossible.T
            tt[impossible] = TT_FAR
            n_impossible = int(np.sum(np.triu(impossible, k=1)))
            if n_impossible:
                logger.debug(
                    "TT matrix: %d same-station P-S pair(s) with impossible S-P "
                    "(> %.1fs or negative) pushed apart.",
                    n_impossible,
                    sp_max,
                )

        return tt

    @staticmethod
    def _merge_ps_split_clusters(
        clusters: list, clusters_stability: list,
        phases: list, pseudo_tt: np.ndarray, max_search_dist: float,
    ) -> tuple[list, list]:
        """Merge HDBSCAN clusters split across a P-S pair from the same station.

        HDBSCAN may put the P pick of a station in one cluster and its S pick
        in another because the S is temporally closer to a different cluster.
        This mirrors Leiden's PS-boost: union-find any pair of clusters where
        one contains a P and the other the S of the same station, provided:
          - t_S > t_P  (causal ordering)
          - pseudo_tt[i_p, i_s] <= max_search_dist  (physically compatible)

        Returns updated (clusters, clusters_stability).
        """
        n = len(clusters)
        if n <= 1:
            return clusters, clusters_stability

        phase_to_idx = {id(p): i for i, p in enumerate(phases)}

        parent = list(range(n))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        # Map (network, station) → list of (cluster_idx, pick) for P picks
        p_index: dict = {}
        for ci, cluster in enumerate(clusters):
            for pick in cluster:
                if pick.is_p():
                    key = (pick.network, pick.station)
                    p_index.setdefault(key, []).append((ci, pick))

        # Process clusters smallest-first so small coherent clusters are merged
        # before larger ones can absorb their picks.  Once unioned, the union-find
        # prevents a pick from being moved again.
        order = sorted(range(n), key=lambda ci: len(clusters[ci]))

        merges = 0
        for ci in order:
            cluster = clusters[ci]
            for s_pick in cluster:
                if not s_pick.is_s():
                    continue
                i_s = phase_to_idx.get(id(s_pick))
                if i_s is None:
                    continue
                key = (s_pick.network, s_pick.station)
                for p_ci, p_pick in p_index.get(key, []):
                    if p_ci == ci:
                        continue
                    # Guard: P cluster must not be much larger than S cluster
                    # (avoids absorbing small events into a large noisy cluster)
                    if len(clusters[p_ci]) > 3 * len(cluster):
                        continue
                    if s_pick.time <= p_pick.time:
                        continue
                    i_p = phase_to_idx.get(id(p_pick))
                    if i_p is None:
                        continue
                    if pseudo_tt[i_p, i_s] > max_search_dist:
                        continue
                    if find(p_ci) != find(ci):
                        union(p_ci, ci)
                        merges += 1

        if merges == 0:
            return clusters, clusters_stability

        root_to_members: dict = {}
        for ci in range(n):
            root_to_members.setdefault(find(ci), []).append(ci)

        new_clusters = []
        new_stability = []
        for members in root_to_members.values():
            merged = []
            for ci in members:
                merged.extend(clusters[ci])
            new_clusters.append(merged)
            # Size-weighted average for consistency with cluster_merge_based_on_eventid()
            total = sum(len(clusters[ci]) for ci in members)
            new_stability.append(
                sum(clusters_stability[ci] * len(clusters[ci]) for ci in members) / total
                if total > 0 else 1.0
            )

        logger.info(
            "HDBSCAN PS-merge: %d union(s), %d → %d clusters.",
            merges, n, len(new_clusters),
        )
        return new_clusters, new_stability

    @staticmethod
    def _absorb_noise_s_picks(
        clusters: list, noise: list,
        phases: list, pseudo_tt: np.ndarray, max_search_dist: float,
    ) -> tuple[list, list]:
        """Absorb noise S picks whose same-station P partner is in a cluster.

        HDBSCAN may classify an S pick as noise while keeping its P in a cluster,
        degrading ps_ratio and station_score.  This mirrors the Leiden union-find
        P-S enforcement and restores coherent P-S pairs, subject to the same
        guards as _merge_ps_split_clusters():
          - t_S > t_P  (causal ordering)
          - pseudo_tt[i_p, i_s] <= max_search_dist  (physically compatible)

        Returns updated (clusters, noise).
        """
        phase_to_idx = {id(p): i for i, p in enumerate(phases)}
        recovered = 0
        remaining_noise = []
        for s_pick in noise:
            if not s_pick.is_s():
                remaining_noise.append(s_pick)
                continue
            i_s = phase_to_idx.get(id(s_pick))
            absorbed = False
            for cluster in clusters:
                for p_pick in cluster:
                    if not (
                        p_pick.is_p()
                        and p_pick.network == s_pick.network
                        and p_pick.station == s_pick.station
                    ):
                        continue
                    if s_pick.time <= p_pick.time:
                        continue
                    i_p = phase_to_idx.get(id(p_pick))
                    if i_s is None or i_p is None:
                        continue
                    if pseudo_tt[i_p, i_s] > max_search_dist:
                        continue
                    cluster.append(s_pick)
                    recovered += 1
                    absorbed = True
                    break
                if absorbed:
                    break
            if not absorbed:
                remaining_noise.append(s_pick)
        if recovered:
            logger.info(
                "HDBSCAN P-S recovery: absorbed %d noise S pick(s) "
                "into clusters with matching P partner.", recovered
            )
        return clusters, remaining_noise

    @staticmethod
    def get_clusters(
        phases,
        pseudo_tt,
        max_search_dist,
        min_cluster_size,
        average_velocity=5.0,
        cluster_selection_method="eom",
        allow_single_cluster=True,
        clustering_method="hdbscan",
        leiden_resolution=0.05,
        leiden_edge_weight_scale=None,
        leiden_ps_boost_factor=100.0,
        leiden_min_edge_weight=0.0,
        leiden_vp=6.0,
        leiden_vs=3.5,
        mega_cluster_fallback_leiden=False,
        mega_cluster_threshold=0.8,
        mega_cluster_min_size=150,
        mega_cluster_leiden_resolution=0.1,
        leiden_hdbscan_fallback=False,      # run HDBSCAN on Leiden noise + unstable clusters
        leiden_min_stability=0.0,           # clusters below this stability are re-tried with HDBSCAN
    ):
        # pseudo_tt is a square travel-time distance matrix (metric="precomputed")

        if clustering_method == "leiden":
            from dbclust.leiden import leiden_cluster
            clusters, stabilities, noise = leiden_cluster(
                phases, pseudo_tt, max_search_dist, min_cluster_size,
                resolution=leiden_resolution,
                edge_weight_scale=leiden_edge_weight_scale,
                ps_boost_factor=leiden_ps_boost_factor,
                min_edge_weight=leiden_min_edge_weight,
                vp=leiden_vp,
                vs=leiden_vs,
            )
            if leiden_hdbscan_fallback:
                # Separate stable clusters from unstable ones
                stable_clusters, stable_stab = [], []
                hdbscan_pool = list(noise)  # start with Leiden noise
                for cluster, stab in zip(clusters, stabilities):
                    if stab >= leiden_min_stability:
                        stable_clusters.append(cluster)
                        stable_stab.append(stab)
                    else:
                        hdbscan_pool.extend(cluster)

                n_unstable = len(clusters) - len(stable_clusters)
                if n_unstable:
                    logger.info(
                        "Leiden+HDBSCAN fallback: %d unstable cluster(s) (stability < %.3f) "
                        "+ %d noise picks → HDBSCAN pool (%d picks total).",
                        n_unstable, leiden_min_stability, len(noise), len(hdbscan_pool),
                    )

                if len(hdbscan_pool) >= min_cluster_size:
                    phase_to_idx = {id(p): i for i, p in enumerate(phases)}
                    pool_indices = [phase_to_idx[id(p)] for p in hdbscan_pool if id(p) in phase_to_idx]
                    sub_tt = pseudo_tt[np.ix_(pool_indices, pool_indices)]
                    hdb_clusters, hdb_stab, hdb_noise = Clusterize.get_clusters(
                        hdbscan_pool, sub_tt, max_search_dist, min_cluster_size,
                        cluster_selection_method=cluster_selection_method,
                        allow_single_cluster=allow_single_cluster,
                        clustering_method="hdbscan",
                    )
                    if hdb_clusters:
                        logger.info(
                            "Leiden+HDBSCAN fallback: %d extra cluster(s) recovered.",
                            len(hdb_clusters),
                        )
                    clusters = stable_clusters + hdb_clusters
                    stabilities = stable_stab + hdb_stab
                    noise = hdb_noise
                else:
                    clusters = stable_clusters
                    stabilities = stable_stab
                    noise = hdbscan_pool
            return clusters, stabilities, noise

        db = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,  # default 5
            min_samples=1,  # default None
            allow_single_cluster=allow_single_cluster,
            cluster_selection_epsilon=max_search_dist,  # default 0.0,
            metric="precomputed",
            cluster_selection_method=cluster_selection_method,
            n_jobs=-1,
        ).fit(pseudo_tt)

        labels = db.labels_

        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise_ = list(labels).count(-1)

        # No cluster found is the routine/uninteresting outcome (happens for most
        # time-windows) -> keep it at debug level regardless of leftover noise picks,
        # to avoid flooding the log. Only log at info when at least one cluster was found.
        log_fn = logger.debug if n_clusters_ == 0 else logger.info
        log_fn("Number of clusters: %d" % n_clusters_)
        log_fn("Number of noise points: %d" % n_noise_)

        # only for hdbscan
        # kind of cluster stability measurement [0, 1]
        if hasattr(db, "cluster_persistence_"):
            raw_stability = db.cluster_persistence_
        else:
            raw_stability = [1] * n_clusters_

        # Feed picks into clusters, building in sorted label order so that
        # clusters[i] aligns correctly with cluster_persistence_[i].
        label_to_cluster: dict = {}
        noise = []
        for p, label in zip(phases, labels):
            if label == -1:
                noise.append(p)
            else:
                label_to_cluster.setdefault(label, []).append(p)

        sorted_labels = sorted(label_to_cluster)
        clusters = [label_to_cluster[lbl] for lbl in sorted_labels]

        if hasattr(raw_stability, "__len__") and len(raw_stability) > 0 and len(sorted_labels) > 0:
            # Guard against HDBSCAN returning a non-empty cluster_persistence_ even when all
            # labels are -1 (0 clusters): skip the mismatch check in that degenerate case.
            # Safety check: ensure stability array matches number of clusters
            if len(raw_stability) != len(sorted_labels):
                logger.warning(
                    f"Stability array length mismatch: {len(raw_stability)} "
                    f"stability values for {len(sorted_labels)} clusters. "
                    f"Using default stability=1.0 for all clusters."
                )
                clusters_stability = [1.0] * len(clusters)
            else:
                clusters_stability = [raw_stability[lbl] for lbl in sorted_labels]
        else:
            clusters_stability = [1] * len(clusters)

        noise_event_ids = Counter(
            p.event_id.split("/")[-1] for p in noise if p.event_id
        )
        logger.debug(f"noise: {len(noise)} phases, event_ids: {dict(noise_event_ids)}")
        for lbl, cluster in zip(sorted_labels, clusters):
            event_id_counts = Counter(
                p.event_id.split("/")[-1] for p in cluster if p.event_id
            )
            logger.debug(
                f"cluster[{lbl}]: {len(cluster)} phases, event_ids: {dict(event_id_counts)}"
            )

        if mega_cluster_fallback_leiden:
            from dbclust.leiden import find_mega_cluster, recluster_mega_with_leiden
            mega_idx = find_mega_cluster(
                clusters, len(phases), mega_cluster_threshold, mega_cluster_min_size
            )
            if mega_idx >= 0:
                clusters, clusters_stability, noise = recluster_mega_with_leiden(
                    mega_idx, clusters, clusters_stability, noise,
                    phases, pseudo_tt, max_search_dist, min_cluster_size,
                    mega_cluster_leiden_resolution, leiden_edge_weight_scale,
                    leiden_ps_boost_factor, leiden_min_edge_weight,
                    vp=leiden_vp, vs=leiden_vs,
                )

        clusters, clusters_stability = Clusterize._merge_ps_split_clusters(
            clusters, clusters_stability, phases, pseudo_tt, max_search_dist
        )
        clusters, noise = Clusterize._absorb_noise_s_picks(
            clusters, noise, phases, pseudo_tt, max_search_dist
        )

        return clusters, clusters_stability, noise

    def cluster_merge_based_on_eventid(self) -> None:
        """Merge clusters containing picks from the same eventid (if provided)."""

        # logger.error(f"n_clusters={self.n_clusters}, lens(clusters) = {len(self.clusters)}")
        # assert self.n_clusters == len(self.clusters)

        if self.n_clusters <= 1:
            return

        logger.info(
            f"cluster_merge_based_on_eventid(): merging clusters sharing same EventId: {self.n_clusters} clusters to handle."
        )

        # Keep self.preloc (positionally aligned on self.clusters by PyOcto,
        # see dbclust2pyocto.py) in lockstep with the pop/merge below, mirroring
        # cluster_merge_one_pass() in dbclust2pyocto.py. Without this,
        # generate_nllobs() reads self.preloc[i] for the wrong event as soon as
        # a merge happens (see BUGS_COHERENCE_REVIEW_2026-09-17.md, A4).
        track_preloc = bool(self.preloc) and len(self.preloc) == len(self.clusters)
        # Event-ID merge guard (needs prelocations aligned on the clusters).
        from dbclust.dbclust2pyocto import eventid_merge_guard_from
        guard = eventid_merge_guard_from(self) if track_preloc else None
        stale_preloc = bool(self.preloc) and not track_preloc
        preloc: list = list(self.preloc) if track_preloc and self.preloc else []

        # Iterative merge until convergence to handle transitive chains:
        # if c1 shares event_id A with c2, and c2 shares event_id B with c3,
        # a single pass misses c1-c3. We repeat until no new merges occur.
        stabs = list(
            self.clusters_stability
            if len(self.clusters_stability) == len(self.clusters)
            else [1.0] * len(self.clusters)
        )
        changed = True
        while changed:
            changed = False
            final_cluster_list = []
            final_stabs = []
            final_preloc: list = []
            while self.clusters:
                c1 = self.clusters.pop(0)
                s1 = stabs.pop(0)
                clusters_to_merge = [c1]
                stabs_to_merge = [s1]
                preloc_to_merge = [preloc.pop(0)] if track_preloc else []
                indices_to_remove = []
                for i, c2 in enumerate(self.clusters):
                    if cluster_share_eventid(
                        c1,
                        c2,
                        shared_threshold=self.min_com_phases,
                        min_picks_per_cluster=self.eventid_shared_min_picks_per_cluster,
                        min_distinct_event_ids=self.eventid_shared_min_distinct_ids,
                    ):
                        # Same guard as cluster_merge_one_pass(): this is a second
                        # event_id-only merge point, it must not undo a refusal.
                        # The candidate is checked against every cluster already
                        # in the group.
                        if guard is not None:
                            refusal = None
                            for member, pl in zip(clusters_to_merge, preloc_to_merge):
                                allowed, why = guard.decide(member, c2, pl, preloc[i])
                                if not allowed:
                                    refusal = why
                                    break
                            if refusal is not None:
                                logger.info(
                                    f"Refused event-ID merge (cluster_merge_based_on_eventid): {refusal}"
                                )
                                continue
                        indices_to_remove.append(i)
                        clusters_to_merge.append(c2)
                        stabs_to_merge.append(stabs[i])
                        if track_preloc:
                            preloc_to_merge.append(preloc[i])

                if indices_to_remove:
                    changed = True
                for i in reversed(indices_to_remove):
                    self.clusters.pop(i)
                    stabs.pop(i)
                    if track_preloc:
                        preloc.pop(i)

                # Deduplicate by physical pick identity (network, station, phase, time),
                # preferring picks with an event_id over those without. This mirrors
                # merge_cluster_with_common_phases(): two clusters merged here because
                # they share an event_id may *also* share physical picks (e.g. HDBSCAN
                # fragments of the same event), which would otherwise be duplicated in
                # the merged cluster and sent twice to NLL.
                seen: dict = {}
                for p in chain(*clusters_to_merge):
                    key = (p.network, p.station, p.phase[0].upper(), p.time.datetime)
                    if key not in seen or (
                        seen[key].event_id is None and p.event_id is not None
                    ):
                        seen[key] = p
                new_cluster = list(seen.values())
                # Stability = size-weighted average of merged clusters (weights use
                # pre-dedup sizes, consistent with merge_cluster_with_common_phases()).
                total = sum(len(c) for c in clusters_to_merge)
                new_stab = (
                    sum(s * len(c) for s, c in zip(stabs_to_merge, clusters_to_merge))
                    / total
                    if total > 0
                    else 1.0
                )
                final_cluster_list.append(new_cluster)
                final_stabs.append(new_stab)
                if track_preloc:
                    # Keep the preloc of the largest pre-merge cluster, consistent
                    # with cluster_merge_one_pass() in dbclust2pyocto.py.
                    dominant_idx = max(
                        range(len(clusters_to_merge)),
                        key=lambda k: len(clusters_to_merge[k]),
                    )
                    final_preloc.append(preloc_to_merge[dominant_idx])

            self.clusters = final_cluster_list
            stabs = final_stabs
            preloc = final_preloc

        self.n_clusters = len(self.clusters)
        self.clusters_stability = np.array(stabs, dtype=float)
        if track_preloc:
            self.preloc = preloc
        elif stale_preloc:
            # self.preloc existed but didn't match self.clusters' length going in
            # (already inconsistent) — drop it rather than leave a stale/misaligned
            # list for generate_nllobs() to read positionally.
            logger.warning(
                "cluster_merge_based_on_eventid(): self.preloc length didn't match "
                "self.clusters before merging; discarding preloc instead of risking "
                "a positional mismatch."
            )
            self.preloc = None
        # Verify consistency after merge
        if len(self.clusters) != len(self.clusters_stability):
            logger.error(
                f"Cluster/stability length mismatch after merge: {len(self.clusters)} "
                f"clusters vs {len(self.clusters_stability)} stability values. "
                f"Attempting to recover..."
            )
            # Attempt recovery by truncating or padding stability array
            if len(self.clusters_stability) > len(self.clusters):
                self.clusters_stability = self.clusters_stability[:len(self.clusters)]
            else:
                # Pad with default stability value (1.0)
                padding = len(self.clusters) - len(self.clusters_stability)
                self.clusters_stability = np.concatenate([
                    self.clusters_stability,
                    np.ones(padding, dtype=float)
                ])
            logger.warning("Recovery successful - please investigate root cause")
        logger.debug(f"EventId merge completed: {self.n_clusters} clusters with stabilities {self.clusters_stability}")
        
        # Log stability statistics after merge
        if len(self.clusters) > 0:
            min_stab = float(np.min(self.clusters_stability))
            max_stab = float(np.max(self.clusters_stability))
            logger.info(
                f"EventId merge leads to {self.n_clusters} clusters. "
                f"Stability range: {min_stab:.3f}-{max_stab:.3f}"
            )

    def log_stability_summary(self):
        """Log a summary of cluster stability statistics."""
        if not self.clusters:
            logger.info("No clusters to analyze for stability")
            return

        stabilities = self.clusters_stability
        min_stab = float(np.min(stabilities))
        max_stab = float(np.max(stabilities))
        avg_stab = float(np.mean(stabilities))
        med_stab = float(np.median(stabilities))

        logger.info("Cluster stability summary:")
        logger.info(f"  Total clusters: {len(self.clusters)}")
        logger.info(f"  Min stability: {min_stab:.3f}")
        logger.info(f"  Max stability: {max_stab:.3f}")
        logger.info(f"  Average stability: {avg_stab:.3f}")
        logger.info(f"  Median stability: {med_stab:.3f}")

    def _compute_station_metrics(self, cluster):
        """Compute per-station phase metrics for a cluster in a single pass.

        Returns (station_phase_sets, stations_with_both, station_score) where:
          station_phase_sets  dict mapping 'net.sta' -> {'P', 'S'}
          stations_with_both  number of stations having both P and S
          station_score       weighted sum: P+S=2.0, P-only=1.0, S-only=0.5
        """
        station_phase_sets = defaultdict(set)
        for p in cluster:
            if not p.phase:
                continue
            station_code = f"{p.network}.{p.station}"
            if p.is_p():
                station_phase_sets[station_code].add("P")
            elif p.is_s():
                station_phase_sets[station_code].add("S")

        stations_with_both = sum(
            1 for phases in station_phase_sets.values()
            if "P" in phases and "S" in phases
        )
        station_score = sum(
            2.0 if ("P" in phases and "S" in phases) else (1.0 if "P" in phases else 0.5)
            for phases in station_phase_sets.values()
        )
        return station_phase_sets, stations_with_both, station_score

    def _stability_at(self, cluster_idx: int) -> float:
        """Bounds-safe accessor for self.clusters_stability[cluster_idx].

        clusters_stability can become momentarily shorter than self.clusters if
        some code path mutates one without the other (already guarded against in
        cluster_merge_based_on_eventid()/merge(), but this is a last line of
        defense to avoid an IndexError crashing NLL export over a stale value).
        """
        if cluster_idx < len(self.clusters_stability):
            return float(self.clusters_stability[cluster_idx])
        logger.warning(
            f"clusters_stability desync: no stability for cluster {cluster_idx} "
            f"(len={len(self.clusters_stability)}) — defaulting to 1.0"
        )
        return 1.0

    def _has_reserve_s_candidates(self, cluster, station_phase_sets) -> bool:
        """Existence check only — never an acceptance decision.

        Returns True when the reserve pool holds at least one S pick at one
        of the cluster's P-only stations, inside a generous window after that
        station's P pick. Used to bypass the pre-NLL min_station_with_P_and_S
        gate when reserve consolidation is enabled, so the event gets the
        chance to be enriched with S picks by NllLoc (the definitive
        decision is the post-NLL weighted gate).

        The window is anchored on the station's own P pick: an S can't
        arrive before it, and the S-P delay is bounded by the homogeneous
        model at the maximum station distance (km),
        reserve_max_station_dist_km * (1/reserve_vs - 1/reserve_vp).
        """
        if not self.reserve_pool:
            return False

        p_only_stations = {
            sta
            for sta, phases in station_phase_sets.items()
            if "P" in phases and "S" not in phases
        }
        if not p_only_stations:
            return False

        # P times of the P-only stations (several P per station are possible)
        p_times = defaultdict(list)
        for p in cluster:
            station_code = f"{p.network}.{p.station}"
            if station_code in p_only_stations and p.is_p():
                p_times[station_code].append(p.time)

        max_ps_delay = self.reserve_max_station_dist_km * (
            1.0 / self.reserve_vs - 1.0 / self.reserve_vp
        )
        tol = self.reserve_tt_tolerance_s

        for ph in self.reserve_pool:
            if not ph.is_s():
                continue
            for t_p in p_times.get(f"{ph.network}.{ph.station}", ()):
                if t_p - tol <= ph.time <= t_p + max_ps_delay + tol:
                    return True
        return False

    def _check_pre_nll_filter(
        self, label, criterion_failed, cluster_idx, event_id_counts,
        override_stability=0.0, override_score=0.0, station_score=None,
    ):
        """Decide whether a pre-NLL filter rejects or keeps a cluster.

        Returns (keep: bool, forced: bool).
        Logs the outcome at the appropriate level.
        """
        if not criterion_failed:
            return True, False

        cluster_stability = self._stability_at(cluster_idx)
        ids_str = f" [event_ids: {dict(event_id_counts)}]" if event_id_counts else ""

        if self.force_keep_catalog_events and event_id_counts:
            logger.warning(
                f"Kept | cluster {cluster_idx}{self._context_tag} failed {label} "
                f"but force_keep_catalog_events=True{ids_str} — keeping anyway "
                f"(next filters in this function still apply)"
            )
            return True, True

        if override_stability > 0.0 and cluster_stability >= override_stability:
            logger.warning(
                f"Kept | cluster {cluster_idx}{self._context_tag} failed {label} "
                f"but stability override applies "
                f"(stability={cluster_stability:.3f} > {override_stability}){ids_str} "
                f"(next filters in this function still apply)"
            )
            return True, False

        if override_score > 0.0 and station_score is not None and station_score >= override_score:
            logger.warning(
                f"Kept | cluster {cluster_idx}{self._context_tag} failed {label} "
                f"but score override applies "
                f"(station_score={station_score:.1f} >= {override_score}){ids_str} "
                f"(next filters in this function still apply)"
            )
            return True, False

        logger.warning(
            f"Rejected | cluster {cluster_idx}{self._context_tag}, stability:{cluster_stability:.3f} | "
            f"reason: {label} check failed{ids_str}"
        )
        return False, False

    def generate_nllobs(self, OBS_PATH, context_tag: str = ""):
        """
        export to obspy/NLL
        only 1 event/catalog (for NLL),
        no duplicated pick !

        context_tag: optional label (e.g. "job=0 win=3/12") identifying the
        time window this call processes, included in per-cluster log lines
        so that "cluster N" can be told apart across windows/jobs that each
        restart numbering from 0.
        """
        logger.info(f"Starting generate_nllobs()")
        self._context_tag = f" [{context_tag}]" if context_tag else ""

        # Log stability statistics before processing
        self.log_stability_summary()

        picks_bundles = []
        rejected_event_ids: set = set()
        accepted_event_ids: set = set()
        for i, cluster in enumerate(self.clusters):
            cat = Catalog()
            event = Event()
            # count the number of stations
            stations_list = set([p.station for p in cluster])

            # Count the number of picks associated to a given event ID (needed for all filters)
            event_id_counts = Counter(
                [p.event_id.split("/")[-1] for p in cluster if p.event_id]
            )

            logger.info(f"--- cluster {i}{self._context_tag} ---")
            logger.info(
                f"Generating nllobs for cluster {i}{self._context_tag} ({len(stations_list)} stations / {len(cluster)} picks, "
                f"stability={self._stability_at(i):.3f})"
                + (f" [event_ids: {dict(event_id_counts)}]" if event_id_counts else "")
            )

            # Compute metrics once for all filters
            station_phase_sets, stations_with_both, station_score = (
                self._compute_station_metrics(cluster)
            )

            forced_catalog_event = False

            # Pre-NLL filter: min_station_count
            if self.min_station_count:
                keep, forced = self._check_pre_nll_filter(
                    f"min_station_count ({len(stations_list)}/{self.min_station_count})",
                    len(stations_list) < self.min_station_count,
                    i, event_id_counts,
                )
                if not keep:
                    rejected_event_ids.update(event_id_counts.keys())
                    continue
                forced_catalog_event = forced_catalog_event or forced

            # Pre-NLL filter: station_score
            if self.min_station_score is not None:
                keep, forced = self._check_pre_nll_filter(
                    f"min_station_score ({station_score:.1f}/{self.min_station_score})",
                    station_score < self.min_station_score,
                    i, event_id_counts,
                )
                if not keep:
                    rejected_event_ids.update(event_id_counts.keys())
                    continue
                forced_catalog_event = forced_catalog_event or forced

            # Pre-NLL filter: min_station_with_P_and_S (with stability and score overrides)
            if self.min_station_with_P_and_S:
                # Reserve bypass: when reserve consolidation is enabled and
                # the cluster has a plausible reserve S candidate, let it
                # through NLL without tranching here — the post-NLL
                # weighted gate (NllLoc) makes the definitive decision.
                # This is the only way for a "many P, all S below import
                # threshold" cluster to ever be localized.
                if (
                    self.reserve_enabled
                    and stations_with_both < self.min_station_with_P_and_S
                    and self._has_reserve_s_candidates(cluster, station_phase_sets)
                ):
                    logger.warning(
                        f"Kept | cluster {i}{self._context_tag} failed "
                        f"min_station_with_P_and_S "
                        f"({stations_with_both}/{self.min_station_with_P_and_S}) "
                        f"but reserve S candidates exist — deferring the "
                        f"decision to the post-NLL weighted gate"
                    )
                    keep, forced = True, False
                else:
                    keep, forced = self._check_pre_nll_filter(
                        f"min_station_with_P_and_S ({stations_with_both}/{self.min_station_with_P_and_S})",
                        stations_with_both < self.min_station_with_P_and_S,
                        i, event_id_counts,
                        override_stability=self.min_station_with_P_and_S_stability_override,
                        override_score=self.min_station_with_P_and_S_score_override,
                        station_score=station_score,
                    )
                if not keep:
                    rejected_event_ids.update(event_id_counts.keys())
                    continue
                forced_catalog_event = forced_catalog_event or forced

            n_stations = len(stations_list)
            ids_str = f" [event_ids: {dict(event_id_counts)}]" if event_id_counts else ""
            logger.info(
                f"Cluster {i}{self._context_tag} → sent to NLL "
                f"(score={station_score:.1f}, ps={stations_with_both}/{n_stations}, "
                f"stability={self._stability_at(i):.3f}){ids_str} "
                f"stations={sorted(stations_list)}"
            )

            for p in cluster:
                pick = p.to_pick()
                event.picks.append(pick)
            event = deduplicate_picks(event)
            if forced_catalog_event:
                event.comments.append(Comment(text='{"force_kept": true}'))
            accepted_event_ids.update(event_id_counts.keys())

            # to be returned !
            picks_bundles.append(event.picks)

            cat.append(event)
            os.makedirs(OBS_PATH, exist_ok=True)
            obs_file = os.path.join(OBS_PATH, f"cluster-{i}.obs")
            cluster_stab = self._stability_at(i)
            logger.debug(
                f"Cluster {i}, writing {obs_file}, stability:{cluster_stab}, n_stations:{len(stations_list)})"
            )
            cat.write(obs_file, format="NLLOC_OBS")
            meta_file = obs_file.replace(".obs", ".meta")
            with open(meta_file, "w") as _mf:
                json.dump({"cluster_stability": round(cluster_stab, 4)}, _mf)

            # use pyocto pre-localization to select velocity model to be used
            # create vel_file with required information
            if self.preloc and len(self.preloc) > i and self.preloc[i]:
                hypo = self.preloc[i]
                logger.info(
                    f"Prelocalization is time={hypo['time']}, lat={hypo['latitude']}, "
                    f"lon={hypo['longitude']}, depth_m={hypo['depth_m']}"
                )

                if not self.zones.polygons.empty:
                    zone, min_dist_km = self.zones.find_zone(
                        latitude=hypo["latitude"],
                        longitude=hypo["longitude"],
                    )

                    # If preloc is too close from an polygon edge
                    # use the national velocity model
                    # if min_dist and min_dist_km < 100:
                    #     logger.info(f"Preloc is close ({min_dist_km} km)"
                    #                 f" to '{zone['name']}' polygon edge ! Using 'world' zone")
                    #     zone = self.zones.get_zone_from_name("world")
                    #     min_dist_km = None

                    if not zone.empty:
                        logger.info(
                            f"Using zone:\n"
                            f"\tname: '{zone['name']}'\n"
                            f"\twith velocity profile: '{zone['velocity_profile']}'\n"
                            f"\ttemplate: '{zone['template']}'\n"
                            f"\tmin_dist_km: {min_dist_km}"
                        )

                        vel_file = os.path.join(OBS_PATH, f"cluster-{i}.vel")
                        logger.debug(f"writing to file {vel_file}: {zone['template']}")
                        with open(vel_file, "w", encoding="utf-8") as vel:
                            vel.write(zone["velocity_profile"] + "\n")
                            vel.write(zone["template"] + "\n")
                            vel.write(f"{hypo['time']}\n")
                            vel.write(f"{hypo['latitude']}\n")
                            vel.write(f"{hypo['longitude']}\n")
                            vel.write(f"{hypo['depth_m']}\n")
                            vel.write(f"{hypo['phase_count']}\n")
                            vel.write(f"{hypo['model_name_used']}\n")

                        picks_file = os.path.join(OBS_PATH, f"cluster-{i}-picks.csv")
                        logger.debug(f"writing file {picks_file}")
                        with open(picks_file, "w", encoding="utf-8") as picks:
                            header = ",".join(hypo["picks_col_names"])
                            picks.write(f"{header}\n")
                            for fields in hypo["phases"]:
                                line = ",".join(map(str, fields))
                                picks.write(f"{line}\n")

                        sta_file = os.path.join(OBS_PATH, f"cluster-{i}-sta.csv")
                        logger.debug(f"writing file {sta_file}")
                        with open(sta_file, "w", encoding="utf-8") as sta:
                            header = ",".join(hypo["coords_col_names"])
                            sta.write(f"{header}\n")
                            for fields in hypo["coords"]:
                                line = ",".join(map(str, fields))
                                sta.write(f"{line}\n")

                    else:
                        logger.warning(
                            f"Can't find a matched zone for (lat={hypo['latitude']},lon={hypo['longitude']})!"
                        )

        # Summary: known event_ids rejected before NLL (not in any accepted cluster)
        lost = rejected_event_ids - accepted_event_ids
        if lost:
            logger.warning(
                f"Known event_id(s) rejected before NLL (all clusters failed pre-filters): {sorted(lost)}"
            )

        return picks_bundles

    def merge(self, clusters2):
        logger.info(
            f"Merging clusters list: {len(self.clusters)} clusters from list1 and {len(clusters2.clusters)} from list2"
        )

        self.clusters += clusters2.clusters
        self.n_clusters = len(self.clusters)
        self.noise += clusters2.noise
        self.n_noise = len(self.noise)

        # Verify clusters2 consistency before merging
        if len(clusters2.clusters_stability) != len(clusters2.clusters):
            logger.warning(
                f"Inconsistent clusters2: {len(clusters2.clusters)} clusters "
                f"but {len(clusters2.clusters_stability)} stability values. "
                f"Using default stability=1.0 for additional clusters."
            )
            # Pad with 1.0 values if needed
            required_length = len(clusters2.clusters)
            current_length = len(clusters2.clusters_stability)
            if current_length < required_length:
                clusters2.clusters_stability = np.concatenate([
                    clusters2.clusters_stability,
                    np.ones(required_length - current_length, dtype=float)
                ])
        
        self.clusters_stability = np.concatenate(
            [self.clusters_stability, clusters2.clusters_stability]
        )
        # self.show_clusters()

        # Consume clusters2: its cluster/stability/noise lists are now owned by
        # self (appended above, same underlying list objects). Clearing them here
        # prevents a caller from accidentally re-submitting clusters2's content a
        # second time if it forgets to discard/replace clusters2 after merging.
        clusters2.clusters = []
        clusters2.n_clusters = 0
        clusters2.noise = []
        clusters2.n_noise = 0
        clusters2.clusters_stability = np.ones(0, dtype=float)

    def show_clusters(self):
        print(f"Clusters: number of clusters = {self.n_clusters}")
        for i, cluster in enumerate(self.clusters):
            stations_list = set([p.station for p in cluster])
            evtids = set([p.event_id for p in cluster if p.event_id])
            print(
                f"\tcluster {i}: stability=%.2f, %d picks / %d stations, eventids: %s"
                % (
                    self._stability_at(i),
                    len(self.clusters[i]),
                    len(stations_list),
                    evtids,
                )
            )
            for p in sorted(cluster, key=lambda p: p.time):
                print(p)
            print("\n")

    def show_noise(self):
        print(f"Noise: {self.n_noise} picks")
        for i in self.noise:
            print(i)
        print("\n")


def _test():
    average_velocity = 5.0  # km/s

    picks_file = "../samples/renass.csv"
    logger.info(f"Opening {picks_file} file.")
    try:
        df = pd.read_csv(picks_file, parse_dates=["phase_time"])
    except Exception as e:
        logger.error(e)
        sys.exit()

    logger.info(f"Read {len(df)} phases.")
    phases = import_phases(
        df,
        P_proba_threshold=0.3,
        S_proba_threshold=0.3,
        info_sta="http://10.0.1.36:8080",
        # info_sta="http://ws.resif.fr",
    )

    myclusters = Clusterize(
        phases=phases,
        max_search_dist=60,
        min_station_with_P_and_S=1,
        min_cluster_size=3,
        average_velocity=average_velocity,
    )
    myclusters.generate_nllobs("../test/obs")
    myclusters.show_clusters()


if __name__ == "__main__":
    logger.setLevel(logging.DEBUG)
    _test()
