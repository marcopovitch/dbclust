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

    for j, c2 in enumerate(clusters2.clusters):
        merged = False
        c2_times = sorted(p.time for p in c2)
        c2_t0 = c2_times[0] if c2_times else None
        c2_t1 = c2_times[-1] if c2_times else None
        c2_eids = [p.event_id for p in c2 if p.event_id]
        logger.debug(
            f"merge_cluster_with_common_phases: c2 cluster [{c2_t0} .. {c2_t1}] "
            f"{len(c2)} picks, {len(c2_eids)} with event_id"
        )
        for i, c1 in enumerate(clusters1.clusters):
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
                    clusters1.clusters_stability[i] = max(
                        clusters1.clusters_stability[i],
                        clusters2.clusters_stability[j],
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
        umap_vp=6.0,  # apparent P-wave velocity for AOT (km/s)
        umap_vs=3.5,  # apparent S-wave velocity for AOT (km/s)
        cluster_selection_method="eom",  # HDBSCAN: "eom" (default) or "leaf"
        tt_clip_seconds=0.0,  # clip TT matrix to this value in seconds (0 = no clip)
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
        self.umap_vp = umap_vp
        self.umap_vs = umap_vs
        self.cluster_selection_method = cluster_selection_method
        self.tt_clip_seconds = tt_clip_seconds

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
                phases, average_velocity
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
                    vp=self.umap_vp,
                    vs=self.umap_vs,
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
            metric="euclidean" if use_umap else "precomputed",
            cluster_selection_method=cluster_selection_method,
        )
        self.n_clusters = len(self.clusters)
        self.n_noise = len(self.noise)
        self.max_search_dist = max_search_dist  # persist: used by absorb_deferred_cluster

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
            # For each agency, keep only the event_id with the most picks.
            # Two event_ids from the same agency in the same cluster means the
            # cluster spans distinct physical events — only the dominant one
            # should drive the pull.
            from collections import Counter
            event_id_counts: Counter = Counter(
                p.event_id
                for p in self.clusters[cluster_idx]
                if p.event_id
            )
            agency_event_map: dict = {}
            for p in self.clusters[cluster_idx]:
                if not p.event_id:
                    continue
                if p.agency not in agency_event_map:
                    agency_event_map[p.agency] = set()
                agency_event_map[p.agency].add(p.event_id)
            for agency, eids in agency_event_map.items():
                if len(eids) > 1:
                    dominant = max(eids, key=lambda e: event_id_counts[e])
                    discarded = eids - {dominant}
                    logger.warning(
                        f"[deferred] Agency {agency} has multiple event_ids"
                        f" {eids}: keeping dominant {dominant},"
                        f" discarding {discarded} from pull set."
                    )
                    known_event_ids -= discarded

            # Pull from other clusters
            for ci, cluster in enumerate(self.clusters):
                if ci == cluster_idx:
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

    def absorb_backward_picks(self, backward_phases):
        """Post-clustering absorption of backward overlap picks.

        Called after the main HDBSCAN run on the forward window picks.
        All backward picks (catalog and DL) are pooled with the main-pass
        noise and re-clustered via a fresh HDBSCAN pass (raw TT matrix,
        no UMAP). HDBSCAN naturally groups them by physical event. After
        clustering, cluster_merge_based_on_eventid is applied exclusively
        on the new backward clusters (never mixing with forward clusters)
        to fuse per-agency fragments of the same physical event.

        Parameters
        ----------
        backward_phases : list[Phase]
            Picks from the backward overlap zone [window_start-overlap, window_start].
        """
        if not backward_phases:
            return

        # All backward picks go to HDBSCAN — no injection into forward clusters.
        pool = backward_phases + list(self.noise)
        if len(pool) < self.min_cluster_size:
            logger.info(
                f"[backward] pool too small ({len(pool)} < {self.min_cluster_size}),"
                f" skipping."
            )
            return

        logger.info(
            f"[backward] HDBSCAN on {len(pool)} picks"
            f" ({len(backward_phases)} backward + {len(self.noise)} noise)."
        )
        pseudo_tt2 = self.numpy_compute_tt_matrix_vectorized(
            pool, self.average_velocity
        )
        new_clusters, new_stabilities, new_noise = self.get_clusters(
            pool,
            pseudo_tt2,
            self.max_search_dist,
            self.min_cluster_size,
            metric="precomputed",
        )

        if not new_clusters:
            logger.info("[backward] no new clusters found.")
            self.noise = new_noise
            self.n_noise = len(new_noise)
            return

        logger.info(f"[backward] {len(new_clusters)} new cluster(s) discovered.")

        # Merge backward clusters that share event_ids (same physical event,
        # different agencies) — strictly isolated from the forward clusters.
        # We do this by temporarily creating a mini Clusterize-like object.
        first_new_idx = len(self.clusters)
        self.clusters += new_clusters
        self.n_clusters = len(self.clusters)
        self.clusters_stability = np.concatenate(
            [
                np.atleast_1d(np.array(self.clusters_stability, dtype=float)),
                np.array(new_stabilities, dtype=float),
            ]
        )
        self.noise = new_noise
        self.n_noise = len(new_noise)

        # cluster_merge_based_on_eventid operates on self.clusters in-place.
        # To restrict it to backward clusters only, temporarily swap out the
        # forward clusters, merge, then restore.
        forward_clusters = self.clusters[:first_new_idx]
        forward_stabilities = self.clusters_stability[:first_new_idx]
        self.clusters = self.clusters[first_new_idx:]
        self.clusters_stability = self.clusters_stability[first_new_idx:]
        self.n_clusters = len(self.clusters)

        self.cluster_merge_based_on_eventid()

        # Restore: forward clusters first, merged backward clusters after.
        self.clusters = forward_clusters + self.clusters
        self.clusters_stability = np.concatenate(
            [
                np.atleast_1d(np.array(forward_stabilities, dtype=float)),
                np.atleast_1d(np.array(self.clusters_stability, dtype=float)),
            ]
        )
        self.n_clusters = len(self.clusters)

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
    def numpy_compute_tt_matrix_vectorized(phases, vmean):
        """Vectorized TT matrix using haversine formula.

        Replaces per-pair gps2dist_azimuth calls with a single NumPy broadcast.
        Haversine error < 0.5% for distances < 2000 km — sufficient for clustering.
        """
        R = 6371.0  # Earth radius in km
        lats = np.radians([p.coord["latitude"] for p in phases])  # (n,)
        lons = np.radians([p.coord["longitude"] for p in phases])  # (n,)
        times = np.array([float(p.time) for p in phases])  # (n,)

        dlat = lats[:, None] - lats[None, :]  # (n, n)
        dlon = lons[:, None] - lons[None, :]  # (n, n)
        a = (
            np.sin(dlat / 2) ** 2
            + np.cos(lats[:, None]) * np.cos(lats[None, :]) * np.sin(dlon / 2) ** 2
        )
        dist_km = 2 * R * np.arcsin(np.sqrt(a))  # (n, n)

        dd = dist_km / vmean  # (n, n)
        dt = times[:, None] - times[None, :]  # (n, n)
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
    def get_clusters(
        phases,
        pseudo_tt,
        max_search_dist,
        min_cluster_size,
        metric="precomputed",
        cluster_selection_method="eom",
    ):
        # metric is "precomputed" ==> X is assumed to be a distance matrix and must be square
        # metric is "euclidean" when pseudo_tt is a UMAP 2D embedding

        # n_jobs is not supported by the KDTree-based algorithm used for euclidean metric
        hdbscan_kwargs = dict(
            min_cluster_size=min_cluster_size,  # default 5
            min_samples=None,  # default None
            allow_single_cluster=True,
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

        cluster_ids = set(labels)

        # only for hdbscan
        # kind of cluster stability measurement [0, 1]
        if hasattr(db, "cluster_persistence_"):
            clusters_stability = db.cluster_persistence_
        else:
            clusters_stability = [1] * n_clusters_

        # feed picks to associated clusters.
        clusters = []
        noise = []
        for c_id in cluster_ids:
            cluster = []
            for p, l in zip(phases, labels):
                if c_id == l:
                    cluster.append(p)
                    # if duplicated picks, rely on NonLinLoc
                    # to keep the relevant picks at localization level
                    # or use the pick probability

            if c_id == -1:
                noise = cluster.copy()
                noise_event_ids = Counter(
                    p.event_id.split("/")[-1] for p in noise if p.event_id
                )
                logger.debug(
                    f"noise: {len(noise)} phases, event_ids: {dict(noise_event_ids)}"
                )
            else:
                clusters.append(cluster)
                event_id_counts = Counter(
                    p.event_id.split("/")[-1] for p in cluster if p.event_id
                )
                logger.debug(
                    f"cluster[{c_id}]: {len(cluster)} phases, event_ids: {dict(event_id_counts)}"
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

        final_cluster_list = []
        while self.clusters:
            clusters_to_merge = []
            c1 = self.clusters.pop(0)
            clusters_to_merge.append(c1)
            indices_to_remove = []
            logger.debug("Working on cluster %s with %d phases" % (c1, len(c1)))
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
                #     logger.info("Eventid shared.")
                # else:
                #     logger.info("No eventid shared.")

            for i in reversed(indices_to_remove):
                self.clusters.pop(i)

            # merge clusters
            new_cluster = list(chain(*clusters_to_merge))
            final_cluster_list.append(new_cluster)

        # Sanity check: self.clusters should be empty
        assert not len(self.clusters)
        self.clusters = final_cluster_list
        self.n_clusters = len(self.clusters)
        self.clusters_stability = np.full(self.n_clusters, 1.0)
        logger.info(f"EventId merge leads to {self.n_clusters} clusters.")

    def generate_nllobs(self, OBS_PATH):
        """
        export to obspy/NLL
        only 1 event/catalog (for NLL),
        no duplicated pick !
        """
        logger.info(f"Starting generate_nllobs()")
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
                f"Generating nllobs for cluster {i} ({len(stations_list)} stations / {len(cluster)} picks)"
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
                if "P" in p.phase.upper():
                    station_phase_sets[station_code].add("P")
                elif "S" in p.phase.upper():
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

            # Pre-NLL filter: min_ps_ratio (stations with both P and S / total stations)
            if self.min_ps_ratio is not None:
                ps_ratio = (
                    stations_with_both / total_stations_ps
                    if total_stations_ps > 0
                    else 0.0
                )
                if ps_ratio < self.min_ps_ratio:
                    if self.force_keep_catalog_events and event_id_counts:
                        logger.warning(
                            f"Cluster {i} failed min_ps_ratio ({stations_with_both}/{total_stations_ps}={ps_ratio:.2f}/{self.min_ps_ratio}) "
                            f"but force_keep_catalog_events=True [event_ids: {dict(event_id_counts)}] — keeping anyway"
                        )
                        forced_catalog_event = True
                    else:
                        log_fn = logger.warning if event_id_counts else logger.info
                        log_fn(
                            f"Cluster {i}, stability:{self.clusters_stability[i]} ignored before NLL: "
                            f"ps_ratio {stations_with_both}/{total_stations_ps} = {ps_ratio:.2f} < {self.min_ps_ratio}"
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
