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
from tqdm import tqdm

from dbclust.phase import import_phases
from dbclust.phase import Phase
from dbclust.quakeml import deduplicate_picks
# import dask.bag as db

# default logger (uses hierarchical name for selective level control)
logger = logging.getLogger("dbclust.clusterize")


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
    for event in cat:
        for pick in event.picks:
            for p in set(chain(*clusters)):
                # We don't check phase_hint because after relabelling
                # (ex: P -> Pg), pick.phase_hint is modified but not p.phase
                if (
                    pick.waveform_id["station_code"] == p.station
                    and pick.time == p.time
                    # and pick.phase_hint == p.phase
                ):
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
            key = (p.station, p.time.datetime)
            pick_to_cluster[key] = cluster_event_ids

    for event in cat:
        o = event.preferred_origin()
        event_ids = []
        for a in o.arrivals:
            if a.time_weight is None or a.time_residual is None:
                continue
            pick = next((p for p in event.picks if p.resource_id == a.pick_id), None)
            if pick is None:
                continue
            key = (pick.waveform_id["station_code"], pick.time.datetime)
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
            pick_to_cluster_idx[(p.station, p.time.datetime)] = idx

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
            key = (pick.waveform_id["station_code"], pick.time.datetime)
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
                c1[:] = list(seen.values())  # Update in place
                if (
                    len(clusters1.clusters_stability) > i
                    and len(clusters2.clusters_stability) > j
                ):
                    # Use size-weighted average for consistency with cluster_merge_based_on_eventid()
                    c1_size = len(c1)
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
        tt_matrix_fname="tt_matrix.npy",
        tt_matrix_load=False,
        tt_matrix_save=False,
        zones=None,
        use_umap=False,  # if True, apply UMAP on TT matrix before HDBSCAN
        umap_clip_seconds=0.0,  # explicit clip ceiling (0 = use p75 of TT matrix)
        umap_aot_tt_blend_alpha=0.3,  # weight of TT in blended dist matrix (0=pure AOT)
        apparent_vp=6.0,  # apparent P-wave velocity for AOT (km/s)
        apparent_vs=3.5,  # apparent S-wave velocity for AOT (km/s)
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

        # tt_matrix load/save parameters
        self.tt_matrix_fname = tt_matrix_fname
        self.tt_matrix_load = tt_matrix_load
        self.use_umap = use_umap
        self.umap_clip_seconds = umap_clip_seconds
        self.umap_aot_tt_blend_alpha = umap_aot_tt_blend_alpha
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
                sys.exit()
        else:
            # sequential computation
            # don't forget to activate lru_cache for compute_tt()
            # pseudo_tt = self.compute_tt_matrix(phases, average_velocity)
            # pseudo_tt = self.numpy_compute_tt_matrix_seq(phases, average_velocity)

            # use the fact that the matrix is diagonal and symmetrical
            # running time is quite similar to sequential computation + lru_cache
            # pseudo_tt = self.numpy_compute_tt_matrix(phases, average_velocity)

            # vectorized haversine: ~20-100x faster than per-pair gps2dist_azimuth
            pseudo_tt = self.numpy_compute_tt_matrix_vectorized(
                phases, average_velocity,
                vp=6.0 if self.clustering_method != "leiden" else None,
                vs=3.5 if self.clustering_method != "leiden" else None,
            )
            # Optional UMAP dimensionality reduction: embed the TT distance matrix
            # into a low-dimensional Euclidean space before HDBSCAN. This separates
            # geographically incoherent pick pools (backward injection) that form
            # mega-clusters in raw TT space but are well-separated in UMAP space.
            if tt_clip_seconds > 0:
                pseudo_tt = np.clip(pseudo_tt, 0.0, tt_clip_seconds)
                logger.info(f"TT matrix clipped to {tt_clip_seconds}s.")

            if use_umap:
                pseudo_tt, max_search_dist = self.build_umap_embedding(
                    phases,
                    pseudo_tt,
                    min_cluster_size=min_cluster_size,
                    umap_clip_seconds=umap_clip_seconds,
                    aot_tt_blend_alpha=umap_aot_tt_blend_alpha,
                    vp=self.apparent_vp,
                    vs=self.apparent_vs,
                )

            # // computation using dask bag: slower for small cluster
            # pseudo_tt = self.dask_compute_tt_matrix(phases, average_velocity)
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
            metric="euclidean" if use_umap else "precomputed",
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
        self.max_search_dist = max_search_dist  # persist post-UMAP value for absorb_deferred_cluster

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
        deferred_keys = {(p.station, p.time.datetime, p.phase) for p in deferred_phases}

        # Remove matching picks (by key) from all existing clusters so the deferred
        # cluster becomes the sole owner of these picks and is not re-localized later
        # via the mega-cluster that originally contained them.
        n_extracted = 0
        for cluster in self.clusters:
            to_remove = [
                p
                for p in cluster
                if (p.station, p.time.datetime, p.phase) in deferred_keys
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
            if (p.station, p.time.datetime, p.phase) in deferred_keys:
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
            for p in to_move:
                cluster.remove(p)
                self.clusters[cluster_idx].append(p)
                n_from_clusters += 1

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
                for p in to_move:
                    cluster.remove(p)
                    self.clusters[cluster_idx].append(p)
                    n_from_event_id += 1
            # Pull from noise: catalog picks landing in noise also belong to
            # the same physical event and would otherwise re-appear as a
            # duplicate in the next window.
            to_move = [p for p in self.noise if p.event_id in known_event_ids]
            for p in to_move:
                self.noise.remove(p)
                self.clusters[cluster_idx].append(p)
                n_from_event_id += 1
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
                all_phases, self.average_velocity, vp=6.0 if self.clustering_method != "leiden" else None, vs=3.5 if self.clustering_method != "leiden" else None
            )
            self.clusters, stab, self.noise = self.get_clusters(
                all_phases, pseudo_tt, self.max_search_dist,
                self.min_cluster_size, average_velocity=self.average_velocity,
                metric="precomputed",
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

    def absorb_backward_picks(self, backward_phases):
        """Superseded by build_clusters_from_backward — raises if called."""
        raise NotImplementedError(
            "absorb_backward_picks is superseded by build_clusters_from_backward"
        )

    @staticmethod
    def compute_tt_matrix(phases, vmean):
        # optimization : matrix is symmetrical -> use lru_cache
        tt_matrix = []
        for p1 in tqdm(phases):
            line = []
            for p2 in phases:
                # line.append(compute_tt(p1, p2, vmean))
                line.append(compute_tt(*sorted((p1, p2)), vmean))
            tt_matrix.append(line)
        return tt_matrix

    @staticmethod
    def numpy_compute_tt_matrix_seq(phases, vmean):
        # optimization : matrix is symmetrical -> use lru_cache
        nb_phases = len(phases)
        tt_matrix = np.empty([nb_phases, nb_phases], dtype=float)
        for i in range(0, nb_phases):
            p1 = phases[i]
            for j in range(0, nb_phases):
                p2 = phases[j]
                tt_matrix[i, j] = compute_tt(*sorted((p1, p2)), vmean)
        return tt_matrix

    @staticmethod
    def numpy_compute_tt_matrix(phases, vmean):
        # optimization : matrix is diagonal and symmetrical
        nb_phases = len(phases)
        elements = []
        for i in range(0, nb_phases):
            p1 = phases[i]
            for j in range(i, nb_phases):
                p2 = phases[j]
                elements.append(compute_tt(*sorted((p1, p2)), vmean))

        matrix_upper1 = np.zeros((nb_phases, nb_phases))
        row, col = np.triu_indices(nb_phases)
        matrix_upper1[row, col] = elements

        matrix_upper2 = np.copy(matrix_upper1)
        np.fill_diagonal(matrix_upper2, 0)
        tt_matrix = matrix_upper1 + matrix_upper2.T
        return tt_matrix

    @staticmethod
    def build_umap_embedding(
        phases,
        pseudo_tt,
        *,
        min_cluster_size,
        umap_clip_seconds=0.0,
        aot_tt_blend_alpha=0.3,
        vp=6.0,
        vs=3.5,
    ):
        """Thin wrapper around :func:`dbclust.umap_embedding.build_umap_embedding`.

        Extracts numpy arrays from the Phase list and delegates to the
        standalone function so the same logic can be reused by scripts.
        """
        from dbclust.umap_embedding import build_umap_embedding as _build

        try:
            return _build(
                lats_deg=[p.coord["latitude"] for p in phases],
                lons_deg=[p.coord["longitude"] for p in phases],
                times=[float(p.time) for p in phases],
                phase_types=[p.phase for p in phases],
                pseudo_tt=pseudo_tt,
                min_cluster_size=min_cluster_size,
                umap_clip_seconds=umap_clip_seconds,
                aot_tt_blend_alpha=aot_tt_blend_alpha,
                vp=vp,
                vs=vs,
            )
        except ImportError as exc:
            logger.warning("%s — falling back to standard HDBSCAN on TT matrix.", exc)
            return pseudo_tt, 0

    @staticmethod
    def numpy_compute_tt_matrix_vectorized(phases, vmean, vp=None, vs=None):
        """Vectorized TT matrix using haversine formula.

        Replaces per-pair gps2dist_azimuth calls with a single NumPy broadcast.
        Haversine error < 0.5% for distances < 2000 km — sufficient for clustering.

        When vp and vs are provided, same-station P-S pairs whose observed S-P
        delay exceeds the maximum physically plausible value given the inter-station
        distance (dt_SP_max = dist_km * (1/vs - 1/vp)) are capped.  This prevents
        spurious links between P and S picks from different events that share a
        station code.  Enabled for HDBSCAN; disabled for Leiden (Leiden uses PS-boost
        instead, and the cap alters cluster composition adversely).
        """
        R = 6371.0  # Earth radius in km
        lats = np.radians([p.coord["latitude"] for p in phases])
        lons = np.radians([p.coord["longitude"] for p in phases])
        times = np.array([float(p.time) for p in phases])
        stations = np.array([f"{p.network}.{p.station}" for p in phases])
        is_p = np.array([p.is_p() for p in phases])
        is_s = np.array([p.is_s() for p in phases])

        dlat = lats[:, None] - lats[None, :]
        dlon = lons[:, None] - lons[None, :]
        a = (
            np.sin(dlat / 2) ** 2
            + np.cos(lats[:, None]) * np.cos(lats[None, :]) * np.sin(dlon / 2) ** 2
        )
        dist_km = 2 * R * np.arcsin(np.sqrt(a))

        dd = dist_km / vmean
        dt = times[:, None] - times[None, :]

        if vp is not None and vs is not None:
            # Cap |dt| for same-station P-S pairs at the maximum plausible S-P delay.
            # dt_SP_max = dist_km * (1/vs - 1/vp): beyond this, the implied source
            # distance is incompatible with the inter-station geometry.
            sp_inv = 1.0 / vs - 1.0 / vp
            ps_pair = (
                (stations[:, None] == stations[None, :])
                & ((is_p[:, None] & is_s[None, :]) | (is_s[:, None] & is_p[None, :]))
            )
            dt_SP_max = dist_km * sp_inv
            dt_capped = np.where(
                ps_pair & (np.abs(dt) > dt_SP_max),
                np.sign(dt) * dt_SP_max,
                dt,
            )
            n_capped = int(np.sum(np.triu(ps_pair & (np.abs(dt) > dt_SP_max), k=1)))
            if n_capped:
                logger.debug(
                    "TT matrix: %d same-station P-S pair(s) dt capped to dt_SP_max.",
                    n_capped,
                )
        else:
            dt_capped = dt

        return np.sqrt(dt_capped**2 + dd**2)

    @staticmethod
    def numpy_compute_proba_weighted_tt_matrix(phases, vmean, alpha):
        """Compute TT matrix with pick probability weighting on the temporal component.

        Low-probability picks get a larger effective time tolerance, making them
        easier to cluster with their high-probability neighbours. The spatial
        component (dd = dist/vmean) is left untouched so that two distinct events
        at the same time but different locations remain separated by their geography.

        alpha controls the strength of the effect. The tolerance is driven by the
        weakest pick in each pair via min(p_i, p_j): a strong pick (0.9) paired
        with a weak one (0.3) gets the same stretch as two weak picks (0.3, 0.3)
        — the uncertain pick sets the tolerance, not the average.

            w(p_i, p_j) = 1 / (alpha + (1-alpha) * min(p_i, p_j))    [w >= 1]

            alpha=1.0 → w=1 always — identical to numpy_compute_tt_matrix_vectorized (no effect)
            alpha=0.5 → picks (0.9,0.9): w≈1.04 (+4%)  ; picks (0.9,0.3) or (0.3,0.3): w≈1.23 (+23%)
            alpha=0.5 → picks (0.9,0.9): w≈1.10 (+10%) ; picks (0.9,0.3) or (0.3,0.3): w≈1.54 (+54%)
            alpha=0.0 → w = 1/min_proba — unbounded for near-zero picks, use with caution
        """
        R = 6371.0
        lats = np.radians([p.coord["latitude"] for p in phases])
        lons = np.radians([p.coord["longitude"] for p in phases])
        times = np.array([float(p.time) for p in phases])
        probas = np.clip([p.proba for p in phases], 0.0, 1.0)

        dlat = lats[:, None] - lats[None, :]
        dlon = lons[:, None] - lons[None, :]
        a = (
            np.sin(dlat / 2) ** 2
            + np.cos(lats[:, None]) * np.cos(lats[None, :]) * np.sin(dlon / 2) ** 2
        )
        dd = 2 * R * np.arcsin(np.sqrt(a)) / vmean  # same dd as in numpy_compute_tt_matrix_vectorized

        min_p = np.minimum(probas[:, None], probas[None, :])
        w = 1.0 / (alpha + (1.0 - alpha) * min_p)
        dt = (times[:, None] - times[None, :]) * w

        return np.sqrt(dt**2 + dd**2)

    # @staticmethod
    # def dask_compute_tt_matrix(phases, vmean):
    #     """Optimization to compute tt_matrix in //"""
    #     # data = [sorted((p1, p2)) for p1 in phases for p2 in phases]
    #     data = product(phases, repeat=2)
    #     b = db.from_sequence(data)
    #     tt_matrix_tmp = b.map(lambda x: compute_tt(*x, vmean)).compute()
    #     tt_matrix = np.array(tt_matrix_tmp).reshape((len(phases), len(phases)))
    #     return tt_matrix

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
            new_stability.append(
                sum(clusters_stability[ci] for ci in members) / len(members)
            )

        logger.info(
            "HDBSCAN PS-merge: %d union(s), %d → %d clusters.",
            merges, n, len(new_clusters),
        )
        return new_clusters, new_stability

    @staticmethod
    def _absorb_noise_s_picks(
        clusters: list, noise: list
    ) -> tuple[list, list]:
        """Absorb noise S picks whose same-station P partner is in a cluster.

        HDBSCAN may classify an S pick as noise while keeping its P in a cluster,
        degrading ps_ratio and station_score.  This mirrors the Leiden union-find
        P-S enforcement and restores coherent P-S pairs.

        Returns updated (clusters, noise).
        """
        recovered = 0
        remaining_noise = []
        for s_pick in noise:
            if not s_pick.is_s():
                remaining_noise.append(s_pick)
                continue
            absorbed = False
            for cluster in clusters:
                for p_pick in cluster:
                    if (
                        p_pick.is_p()
                        and p_pick.network == s_pick.network
                        and p_pick.station == s_pick.station
                    ):
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
        metric="precomputed",
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
        # metric is "precomputed" ==> X is assumed to be a distance matrix and must be square
        # metric is "euclidean" when pseudo_tt is a UMAP 2D embedding

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
                        metric="precomputed",
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

        # n_jobs is not supported by the KDTree-based algorithm used for euclidean metric
        hdbscan_kwargs = dict(
            min_cluster_size=min_cluster_size,  # default 5
            min_samples=1,  # default None
            allow_single_cluster=allow_single_cluster,
            cluster_selection_epsilon=max_search_dist,  # default 0.0,
            metric=metric,
            cluster_selection_method=cluster_selection_method,
        )
        if metric == "precomputed":
            hdbscan_kwargs["n_jobs"] = -1

        db = hdbscan.HDBSCAN(**hdbscan_kwargs).fit(pseudo_tt)

        labels = db.labels_

        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise_ = list(labels).count(-1)

        logger.info("Number of clusters: %d" % n_clusters_)
        logger.info("Number of noise points: %d" % n_noise_)

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
                )

        clusters, clusters_stability = Clusterize._merge_ps_split_clusters(
            clusters, clusters_stability, phases, pseudo_tt, max_search_dist
        )
        clusters, noise = Clusterize._absorb_noise_s_picks(clusters, noise)

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
            while self.clusters:
                c1 = self.clusters.pop(0)
                s1 = stabs.pop(0)
                clusters_to_merge = [c1]
                stabs_to_merge = [s1]
                indices_to_remove = []
                for i, c2 in enumerate(self.clusters):
                    if cluster_share_eventid(
                        c1,
                        c2,
                        shared_threshold=self.min_com_phases,
                        min_picks_per_cluster=self.eventid_shared_min_picks_per_cluster,
                        min_distinct_event_ids=self.eventid_shared_min_distinct_ids,
                    ):
                        indices_to_remove.append(i)
                        clusters_to_merge.append(c2)
                        stabs_to_merge.append(stabs[i])

                if indices_to_remove:
                    changed = True
                for i in reversed(indices_to_remove):
                    self.clusters.pop(i)
                    stabs.pop(i)

                new_cluster = list(chain(*clusters_to_merge))
                # Stability = size-weighted average of merged clusters
                total = sum(len(c) for c in clusters_to_merge)
                new_stab = (
                    sum(s * len(c) for s, c in zip(stabs_to_merge, clusters_to_merge))
                    / total
                    if total > 0
                    else 1.0
                )
                final_cluster_list.append(new_cluster)
                final_stabs.append(new_stab)

            self.clusters = final_cluster_list
            stabs = final_stabs

        self.n_clusters = len(self.clusters)
        self.clusters_stability = np.array(stabs, dtype=float)
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

    def generate_nllobs(self, OBS_PATH):
        """
        export to obspy/NLL
        only 1 event/catalog (for NLL),
        no duplicated pick !
        """
        logger.info(f"Starting generate_nllobs()")
        
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

            logger.info(
                f"Generating nllobs for cluster {i} ({len(stations_list)} stations / {len(cluster)} picks, "
                f"stability={self.clusters_stability[i]:.3f})"
                + (f" [event_ids: {dict(event_id_counts)}]" if event_id_counts else "")
            )

            forced_catalog_event = False
            if self.min_station_count:
                if len(stations_list) < self.min_station_count:
                    if self.force_keep_catalog_events and event_id_counts:
                        logger.warning(
                            f"Cluster {i} failed min_station_count ({len(stations_list)}/{self.min_station_count}) "
                            f"but force_keep_catalog_events=True [event_ids: {dict(event_id_counts)}] — keeping anyway"
                        )
                        forced_catalog_event = True
                    else:
                        logger.info(
                            f"Cluster {i}, stability:{self.clusters_stability[i]} ignored ... "
                            f"not enough stations ({len(stations_list)}/{self.min_station_count})"
                            + (
                                f" [event_ids: {dict(event_id_counts)}]"
                                if event_id_counts
                                else ""
                            )
                        )
                        rejected_event_ids.update(event_id_counts.keys())
                        continue

            # Compute per-station phase sets (P:1.0, S:0.5, P+S:2.0) in a single pass
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
                1
                for phases in station_phase_sets.values()
                if "P" in phases and "S" in phases
            )
            total_stations_ps = len(station_phase_sets)

            # Pre-NLL filter: station_score (P:1.0, S:0.5, P+S:2.0)
            if self.min_station_score is not None:
                station_score = sum(
                    2.0
                    if ("P" in phases and "S" in phases)
                    else (1.0 if "P" in phases else 0.5)
                    for phases in station_phase_sets.values()
                )
                if station_score < self.min_station_score:
                    if self.force_keep_catalog_events and event_id_counts:
                        logger.warning(
                            f"Cluster {i} failed min_station_score ({station_score:.1f}/{self.min_station_score}) "
                            f"but force_keep_catalog_events=True [event_ids: {dict(event_id_counts)}] — keeping anyway"
                        )
                        forced_catalog_event = True
                    else:
                        log_fn = logger.warning if event_id_counts else logger.info
                        log_fn(
                            f"Cluster {i}, stability:{self.clusters_stability[i]} ignored before NLL: "
                            f"station_score {station_score:.1f} < {self.min_station_score}"
                            + (
                                f" [event_ids: {dict(event_id_counts)}]"
                                if event_id_counts
                                else ""
                            )
                        )
                        rejected_event_ids.update(event_id_counts.keys())
                        continue

            # Pre-NLL filter: min_station_with_P_and_S (only when station_score not used)
            elif self.min_station_with_P_and_S:
                if stations_with_both < self.min_station_with_P_and_S:
                    if self.force_keep_catalog_events and event_id_counts:
                        logger.warning(
                            f"Cluster {i} failed min_station_with_P_and_S ({stations_with_both}/{self.min_station_with_P_and_S}) "
                            f"but force_keep_catalog_events=True [event_ids: {dict(event_id_counts)}] — keeping anyway"
                        )
                        forced_catalog_event = True
                    else:
                        log_fn = logger.warning if event_id_counts else logger.info
                        log_fn(
                            f"Cluster {i}, stability:{self.clusters_stability[i]} ignored ... "
                            f"not enough stations with both P and S ({stations_with_both}/{self.min_station_with_P_and_S})"
                            + (
                                f" [event_ids: {dict(event_id_counts)}]"
                                if event_id_counts
                                else ""
                            )
                        )
                        rejected_event_ids.update(event_id_counts.keys())
                        continue


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
            logger.debug(
                f"Cluster {i}, writing {obs_file}, stability:{self.clusters_stability[i]}, n_stations:{len(stations_list)})"
            )
            cat.write(obs_file, format="NLLOC_OBS")

            # use pyocto pre-localization to select velocity model to be used
            # create vel_file with required information
            if self.preloc and self.preloc[i]:
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

    def show_clusters(self):
        print(f"Clusters: number of clusters = {self.n_clusters}")
        for i, cluster in enumerate(self.clusters):
            stations_list = set([p.station for p in cluster])
            evtids = set([p.event_id for p in cluster if p.event_id])
            print(
                f"\tcluster {i}: stability=%.2f, %d picks / %d stations, eventids: %s"
                % (
                    self.clusters_stability[i],
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
