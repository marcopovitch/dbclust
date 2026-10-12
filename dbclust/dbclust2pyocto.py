#!/usr/bin/env python
import copy
import datetime
import logging
import math
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import chain
from itertools import combinations
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import pandas as pd
import pyocto
import pyocto._core as _pyocto_backend
import pyproj
import pyproj.exceptions
from dbclust.constants import KM_PER_DEGREE
from dbclust.clusterize import cluster_share_eventid
from dbclust.clusterize import preloc_pseudo_tt
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


class _SuppressSecondPassWarning(logging.Filter):
    def filter(self, record):
        return "Did not associate more events after" not in record.getMessage()


logging.getLogger("pyocto").addFilter(_SuppressSecondPassWarning())


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


def _tolerance_candidates(original_tolerance, tolerance_steps, min_tolerance):
    """Build the descending list of tolerances tried by the linear decay,
    highest (original_tolerance) first, down to min_tolerance."""
    candidates = []
    tolerance = original_tolerance
    while tolerance >= min_tolerance:
        candidates.append(round(tolerance, 2))
        step = next((s for t, s in tolerance_steps.items() if tolerance > t), 0.5)
        tolerance -= step
    return candidates


def adjust_associator_tolerance(
    myclust,
    cfg,
    tolerance_steps={1: 0.5, 0: 0.1},
    min_tolerance=0.5,
    include_noise_in_aggregation=False,
    log_level=logging.INFO,
):
    """
    Adjust associator.pick_match_tolerance using a binary search over the same
    descending tolerance scale the previous linear decay used.

    Higher tolerance is tried first because it is cheap and the common case;
    once a candidate fails with MultipleEventIDsWithSameAgencyError, success is
    assumed monotonic in the remaining (lower) tolerances, so the search jumps
    straight to the midpoint of the remaining candidates instead of walking
    them one by one. This finds the same highest-tolerance success as the
    linear decay in O(log N) attempts instead of O(N), which matters because
    each attempt re-runs the full (expensive) pyocto association from scratch
    and its cost grows sharply as tolerance decreases. If every candidate the
    search actually probes fails, it falls back to a full linear scan of the
    rest so the "best partial result" still matches what the exhaustive
    linear decay would have found, rather than being limited to whichever
    few candidates the search happened to visit.

    Each attempt runs on its own shallow copy of ``myclust`` (fresh
    ``clusters``/``noise``/``preloc`` lists, same underlying Phase objects —
    not a deepcopy, which would needlessly duplicate each Phase's possibly
    tens-of-MB obspy Inventory): dbclust2pyocto() mutates its argument in
    place and the search may call it more than once, including after a
    success (to probe for an even higher one), so attempts must not corrupt
    each other's or the original's cluster/pick lists.

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
    original_tolerance = associator.pick_match_tolerance
    candidates = _tolerance_candidates(original_tolerance, tolerance_steps, min_tolerance)
    best_result = None
    best_n_clusters = 0

    def try_tolerance(tolerance):
        nonlocal best_result, best_n_clusters
        logger.info(f"Trying pick_match_tolerance: {tolerance:.2f}")
        associator.pick_match_tolerance = tolerance
        try:
            # See function docstring for why each attempt needs its own
            # shallow copy with fresh clusters/noise/preloc lists.
            # (_deferred_cluster_indices is only ever mutated by Clusterize's
            # own methods, never by dbclust2pyocto(), so it doesn't need the
            # same treatment.)
            myclust_copy = copy.copy(myclust)
            myclust_copy.clusters = [list(c) for c in myclust.clusters]
            myclust_copy.noise = list(myclust.noise)
            if myclust.preloc is not None:
                myclust_copy.preloc = list(myclust.preloc)
            result_myclust = dbclust2pyocto(
                myclust_copy,
                cfg.pyocto.default_model_name,
                associator,
                cfg.pyocto.velocity_model,
                cfg.cluster.min_picks_common,
                delegate_dbclust=cfg.pyocto.delegate_dbclust,
                include_noise_in_aggregation=include_noise_in_aggregation,
                log_level=log_level,
            )
            logger.info(
                f"Success with pick_match_tolerance: {tolerance:.2f} "
                f"({result_myclust.n_clusters} clusters, all known event_id conflicts resolved)"
            )
            return result_myclust
        except MultipleEventIDsWithSameAgencyError as e:
            logger.warning(f"Unsuccessful with pick_match_tolerance: {tolerance:.2f}.")
            logger.warning(f"{e}")
            # dbclust2pyocto() may attach myclust_copy, mutated further, to
            # the exception as partial_result. Keep a reference (in
            # best_result) only if it's the new best; otherwise let it go
            # out of scope with this function's other locals on return.
            partial_result = e.partial_result
            e.partial_result = None
            if partial_result is not None:
                n = partial_result.n_clusters
                if n > best_n_clusters:
                    best_n_clusters = n
                    best_result = partial_result
                    logger.info(
                        f"New best partial result: {n} clusters at tolerance {tolerance:.2f}"
                    )
            return None

    try:
        logger.info(
            f"Starting tolerance search over {len(candidates)} candidates: {candidates}"
        )

        # candidates is sorted highest-tolerance-first; success is assumed
        # monotonic from some index onward (low tolerance => success). The
        # highest tolerance (candidates[0]) is the common case (it
        # used to be probed at the binary search's first midpoint instead of
        # first, costing extra full-cost attempts on the very case this
        # search is meant to be cheap for) so it is always tried first;
        # binary search over the remaining, lower candidates only kicks in if
        # it fails.
        success_idx = None
        success_result = None
        attempts = 0
        try:
            result_myclust = try_tolerance(candidates[0])
            attempts += 1
            if result_myclust is not None:
                success_idx = 0
                success_result = result_myclust

            if success_idx is None:
                lo, hi = 1, len(candidates) - 1
                while lo <= hi:
                    mid = (lo + hi) // 2
                    attempts += 1
                    result_myclust = try_tolerance(candidates[mid])
                    if result_myclust is not None:
                        success_idx = mid
                        success_result = result_myclust
                        hi = mid - 1  # look for an even higher tolerance that succeeds
                    else:
                        lo = mid + 1  # this tolerance (and higher) fails, go lower
        except pyproj.exceptions.ProjError as e:
            logger.error(f"Projection error, skipping dbclust2pyocto() processing: {e}")
            raise

        if success_idx is not None:
            logger.info(
                f"Tolerance search found success at tolerance={candidates[success_idx]:.2f} "
                f"after {attempts}/{len(candidates)} attempts "
                f"(linear decay would have taken up to {success_idx + 1} attempts)."
            )
            return success_result

        # every candidate the binary search actually probed above
        # failed, but it only probes O(log N) of the N candidates (it prunes
        # ranges based on the monotonicity assumption, which is not
        # guaranteed to hold). Fall back to a full linear scan of the
        # remaining, still-untried candidates so both (a) best_result matches
        # what the old exhaustive linear decay would have found instead of
        # being silently limited to whichever few candidates the binary
        # search happened to visit, and (b) a real success skipped by a
        # monotonicity violation is still found. This only runs in the
        # already-slow all-failed case, so it doesn't affect the common-case
        # cost. try_tolerance() is idempotent per tolerance value, so
        # re-probing any candidate the binary search already tried is
        # wasted but not incorrect; skipping that bookkeeping keeps this
        # fallback simple.
        for tolerance in candidates[1:]:
            attempts += 1
            result_myclust = try_tolerance(tolerance)
            if result_myclust is not None:
                logger.warning(
                    f"Monotonicity assumption violated: tolerance={tolerance:.2f} "
                    f"succeeded after being skipped by the binary search."
                )
                logger.info(
                    f"Tolerance search found success at tolerance={tolerance:.2f} "
                    f"after {attempts}/{len(candidates)} attempts (fallback linear scan)."
                )
                return result_myclust

        logger.error(f"Exhausted all {len(candidates)} tolerances.")
        if best_result is not None:
            logger.warning(
                f"Returning best partial result with {best_n_clusters} clusters "
                f"despite unresolved agency conflict."
            )
            return best_result
        logger.error("No partial result available. Skipping pyocto processing.")
        return None
    finally:
        # Never leak a decayed tolerance into subsequent time windows: this
        # associator object is shared for the whole run (cfg.pyocto.current_model).
        associator.pick_match_tolerance = original_tolerance


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

    Per-cluster min_node_size is auto-scaled from the cluster's actual search
    volume (station lat/lon span x zlim depth), instead of using
    associator_cfg.min_node_size as-is: PyOcto's octree falls back to slower
    depth-first search once its node queue exceeds
    queue_memory_protection_dfs_size (500000 by default), which a cluster
    spanning hundreds of km (e.g. several real, distant events lumped
    together before PyOcto splits them) can approach or exceed, turning a
    seconds-long search into minutes. The scaling solves for the node size
    that puts the cluster near a fixed target leaf-cell count, symmetric in
    both directions (a small/compact cluster gets a finer resolution for
    free instead of only ever being coarsened), bounded below by
    min_node_size_location (no point going finer than the location-
    refinement resolution) and above by a fixed cap (do not blur events
    closer than that).
    """
    logger.info(
        f"Using pyocto to process clusters ({sum(len(c) for c in myclust.clusters)} picks)"
    )

    noise_picks = list(myclust.noise) if include_noise_in_aggregation else []
    if noise_picks:
        logger.info(
            f"Including {len(noise_picks)} noise picks in aggregation pool"
        )
    all_picks_list = list(chain(*myclust.clusters)) + noise_picks
    pyocto_clusters, pyocto_preloc = [], []
    # Seed with failed clusters carried over from the previous window
    clusters_to_process = list(myclust.clusters)
    if include_noise_in_aggregation and myclust.noise:
        noise_event_ids = set(p.event_id for p in myclust.noise if p.event_id)
        logger.info(
            f"Adding noise ({len(myclust.noise)} picks, "
            f"event_ids: {[e.split('/')[-1] for e in noise_event_ids]}) as additional cluster for PyOcto"
        )
        clusters_to_process.append(myclust.noise)
    n_first_pass_clusters = len(clusters_to_process)  # only first-pass clusters generate a second pass

    for i, cluster in enumerate(clusters_to_process):
        # Extract station and pick data for the cluster
        stations = get_stations_from_cluster(cluster)
        picks = get_picks_from_cluster(cluster)
        picks["_cluster_idx"] = range(len(cluster))

        # Detect multi-event clusters: if multiple distinct event_ids are present,
        # reduce min_pick_fraction to allow PyOcto to find smaller events.
        #
        # min_pick_fraction is a distance-based coverage test: a candidate origin
        # is only accepted if at least this fraction of the stations closer than
        # its farthest contributing station also have a pick for it. A cluster
        # with N genuinely distinct event_ids sharing the same station pool
        # (e.g. a dense swarm) splits that coverage N ways, so each individual
        # event structurally has less station coverage than a single isolated
        # event would — the more event_ids are packed in, the weaker any one
        # of them can look. Scaling the fraction down by event count (not just
        # by pick-quality via median_proba) lets PyOcto still accept those
        # thinner-coverage origins instead of leaving most of the cluster's
        # picks unassigned and falling back to an unresolved multi-event blob.
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
            # 1/n^exponent rather than 1/n: a sub-0.5 exponent keeps median_proba's
            # influence visible up to higher event counts (n=8-10 with the default
            # 0.3) instead of saturating at the floor as soon as ~4 events share
            # the cluster (as 1/sqrt(n) does). A lower exponent recovers more
            # thinly-covered events in dense swarms but also raises the risk of
            # PyOcto accepting a low-coverage origin that merges picks from
            # unrelated, distant events — tune min_pick_fraction_event_count_exponent
            # empirically rather than lowering it further than needed.
            exponent = associator_cfg.min_pick_fraction_event_count_exponent
            event_count_factor = 1.0 / (len(cluster_event_ids) ** exponent)
            effective_min_pick_fraction = max(
                associator_cfg.min_pick_fraction_floor,
                associator_cfg.min_pick_fraction * median_proba * event_count_factor,
            )
            logger.info(
                f"Cluster#{i}: {len(cluster_event_ids)} distinct event_ids detected, "
                f"median DL proba={median_proba:.3f} ({len(dl_probas)} DL picks), "
                f"reducing min_pick_fraction: {associator_cfg.min_pick_fraction} -> {effective_min_pick_fraction:.3f} "
                f"(event_count_factor=1/{len(cluster_event_ids)}^{exponent}, floor={associator_cfg.min_pick_fraction_floor})"
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

        logger.debug(f"range lat: {lat_range}, lon: {lon_range}")
        lat_span_km = (lat_range[1] - lat_range[0]) * KM_PER_DEGREE
        lon_span_km = (lon_range[1] - lon_range[0]) * KM_PER_DEGREE * math.cos(
            math.radians((lat_range[0] + lat_range[1]) / 2)
        )
        z_span_km = associator_cfg.zlim[1] - associator_cfg.zlim[0]

        # See dbclust2pyocto() docstring for the rationale. target_leaf_cells
        # keeps a 10x margin below the 5e5 DFS fallback threshold;
        # min_node_size_cap is the "do not blur events closer than this" cap.
        target_leaf_cells = 5.0e4
        min_node_size_cap = 30.0  # km
        search_volume_km3 = lat_span_km * lon_span_km * z_span_km
        if search_volume_km3 <= 0:
            # A degenerate/misconfigured zlim or lat/lon range (e.g.
            # zlim[1] <= zlim[0]) makes this non-positive; a fractional power
            # of a non-positive base silently returns a complex number in
            # Python, which would crash the min()/max() below with a
            # TypeError. Fall back to the configured min_node_size_location.
            logger.warning(
                f"Cluster#{i} has a non-positive search volume "
                f"({lat_span_km:.0f}km x {lon_span_km:.0f}km x {z_span_km:.0f}km) — "
                f"check associator zlim/station coordinates. Falling back to "
                f"min_node_size_location={associator_cfg.min_node_size_location}km."
            )
            effective_min_node_size = associator_cfg.min_node_size_location
        else:
            min_node_size_for_target = (search_volume_km3 / target_leaf_cells) ** (1 / 3)
            effective_min_node_size = min(
                max(min_node_size_for_target, associator_cfg.min_node_size_location),
                min_node_size_cap,
            )
        # min_node_size_location is a floor here, not a ceiling: never finer
        # than the configured resolution, but still coarsened toward
        # effective_min_node_size for big clusters.
        effective_min_node_size_location = max(
            associator_cfg.min_node_size_location,
            min(effective_min_node_size * 0.9, effective_min_node_size),
        )
        approx_leaf_cells = (
            (lat_span_km / effective_min_node_size)
            * (lon_span_km / effective_min_node_size)
            * (z_span_km / effective_min_node_size)
        )
        logger.info(
            f"Cluster#{i} search volume: {lat_span_km:.0f}km x {lon_span_km:.0f}km x "
            f"{z_span_km:.0f}km, {len(stations)} stations, min_node_size="
            f"{associator_cfg.min_node_size}km -> effective {effective_min_node_size:.1f}km "
            f"=> approx {approx_leaf_cells:.2e} leaf cells at effective resolution "
            f"(queue_memory_protection_dfs_size=500000 default)"
        )

        try:
            associator = pyocto.OctoAssociator.from_area(
                lat=lat_range,
                lon=lon_range,
                zlim=associator_cfg.zlim,
                time_before=associator_cfg.time_before,  # should be greater than dbclust time_window parameter
                max_pick_overlap=associator_cfg.max_pick_overlap,
                min_pick_fraction=effective_min_pick_fraction,
                min_node_size=effective_min_node_size,
                min_node_size_location=effective_min_node_size_location,
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

        if len(assignments):
            assigned_df_indices = set(assignments["pick_idx"].to_list())
            assigned_cluster_ids = set(
                picks.loc[picks.index.isin(assigned_df_indices), "_cluster_idx"].tolist()
            )
        else:
            assigned_cluster_ids = set()
        pre_filter_unassigned = [
            p for j, p in enumerate(cluster) if j not in assigned_cluster_ids
        ]

        # Drop same-station P/S picks whose local S-P delay is geometrically
        # incompatible with this event's location (see filter_incoherent_ps_pairs
        # docstring): a coincidental travel-time match can let a foreign pick
        # (from another, closer event) slip into a distant origin. Picks left
        # unassigned by PyOcto are checked too, in case only one of a
        # same-station P/S pair was assigned here.
        ps_incoherent_dropped_ids = set()
        event_rows = list(events.iterrows())
        assigned_clusters = get_clusters_from_assignment(cluster, picks, events, assignments)
        cleaned_clusters = []
        for (_, event_row), c in zip(event_rows, assigned_clusters):
            kept, dropped = filter_incoherent_ps_pairs(
                c, event_row["latitude"], event_row["longitude"],
                associator_cfg.zlim,
                vp=myclust.apparent_vp, vs=myclust.apparent_vs,
                epi_err_km=associator_cfg.ps_coherence_epi_err_km,
                vel_err=associator_cfg.ps_coherence_vel_err,
                pick_err_hard=associator_cfg.ps_coherence_pick_err_hard,
                pick_err_soft=associator_cfg.ps_coherence_pick_err_soft,
                external_candidates=pre_filter_unassigned,
            )
            cleaned_clusters.append(kept)
            ps_incoherent_dropped_ids.update(id(p) for p in dropped)

        if associator_cfg.min_ps_ratio is not None:
            filtered_clusters = []
            for c in cleaned_clusters:
                station_phases = defaultdict(set)
                for p in c:
                    if not p.phase:
                        continue
                    station_code = f"{p.network}.{p.station}"
                    if p.is_p():
                        station_phases[station_code].add("P")
                    elif p.is_s():
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
            pyocto_clusters.extend(cleaned_clusters)

        if ps_incoherent_dropped_ids:
            # Picks dropped by the P-S coherence filter go back to the
            # unassigned pool instead of staying counted as assigned.
            dropped_cluster_ids = {
                j for j, p in enumerate(cluster) if id(p) in ps_incoherent_dropped_ids
            }
            assigned_cluster_ids -= dropped_cluster_ids

        n_unassigned = len(cluster) - len(assigned_cluster_ids)
        log_fn = logger.info if len(events) > 0 else logger.debug
        log_fn(
            f"\t[tolerance={associator_cfg.pick_match_tolerance:.2f}, input_cluster#{i}] "
            f"{len(events)} events found with {len(cluster)} picks"
            f" ({n_unassigned} unassigned by PyOcto)"
        )

        if n_unassigned >= myclust.min_cluster_size and i < n_first_pass_clusters:
            unassigned = [p for j, p in enumerate(cluster) if j not in assigned_cluster_ids]
            pseudo_tt2 = myclust.numpy_compute_tt_matrix_vectorized(
                unassigned, myclust.average_velocity,
                # Same-station impossible P-S pairs pushed apart: enabled for
                # HDBSCAN, disabled for Leiden (Leiden uses PS-boost instead —
                # see numpy_compute_tt_matrix_vectorized and clusterize.py).
                vp=myclust.apparent_vp if myclust.clustering_method != "leiden" else None,
                vs=myclust.apparent_vs if myclust.clustering_method != "leiden" else None,
                max_search_dist=myclust.max_search_dist,
            )
            sp_epsilon = (
                myclust.max_search_dist
                if myclust.clustering_method == "leiden"
                else 30
            )
            # For small residual pools, use HDBSCAN — its density criterion
            # naturally separates small isolated events better than Leiden
            # on sparse picks. For larger pools, keep the configured method.
            sp_clustering_method = (
                "hdbscan"
                if n_unassigned <= 20 and myclust.clustering_method == "leiden"
                else myclust.clustering_method
            )
            sp_clusters, _sp_stab, _sp_noise = myclust.get_clusters(
                unassigned, pseudo_tt2,
                max_search_dist=sp_epsilon,
                min_cluster_size=myclust.min_cluster_size,
                average_velocity=myclust.average_velocity,
                clustering_method=sp_clustering_method,
                leiden_resolution=myclust.leiden_resolution,
                leiden_edge_weight_scale=myclust.leiden_edge_weight_scale,
                leiden_ps_boost_factor=myclust.leiden_ps_boost_factor,
                leiden_min_edge_weight=myclust.leiden_min_edge_weight,
                leiden_vp=myclust.leiden_vp,
                leiden_vs=myclust.leiden_vs,


                mega_cluster_fallback_leiden=myclust.mega_cluster_fallback_leiden,
                mega_cluster_threshold=myclust.mega_cluster_threshold,
                mega_cluster_min_size=myclust.mega_cluster_min_size,
                mega_cluster_leiden_resolution=myclust.mega_cluster_leiden_resolution,
                leiden_hdbscan_fallback=myclust.leiden_hdbscan_fallback,
                leiden_min_stability=myclust.leiden_min_stability,
            )
            for sp_cluster in sp_clusters:
                clusters_to_process.append(sp_cluster)

    # Merge clusters with common picks or event IDs
    pyocto_clusters, pyocto_preloc = cluster_merge(
        pyocto_clusters,
        pyocto_preloc,
        min_com_phases,
        eventid_shared_min_picks_per_cluster=myclust.eventid_shared_min_picks_per_cluster,
        eventid_shared_min_distinct_ids=myclust.eventid_shared_min_distinct_ids,
        guard=eventid_merge_guard_from(myclust),
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
            # this early return used to skip the moveout re-fusion
            # entirely, even though myclust.clusters (the original, pre-PyOcto
            # dbclust/Leiden clusters) can carry the exact same moveout
            # fragmentation the re-fusion elsewhere in this function exists to
            # undo. Apply it here too, on myclust.clusters/myclust.preloc,
            # before returning — myclust.preloc may be None (no PyOcto prelocs
            # were ever produced on this path), in which case there is nothing
            # for the fusion to keep aligned, so skip it.
            if _apply_moveout_refusion(myclust, myclust.clusters, myclust.preloc):
                myclust.n_clusters = len(myclust.clusters)
            return myclust
        return None

    # Shallow-copy the Clusterize object: all mutable attributes are immediately
    # overwritten below, so a deepcopy of the original clusters is not needed.
    newclust = copy.copy(myclust)
    newclust.clusters = pyocto_clusters
    newclust.n_clusters = len(newclust.clusters)
    newclust.clusters_stability = [1] * newclust.n_clusters  # placeholder, recomputed below
    newclust.preloc = pyocto_preloc  # used to choose NLL velocity model

    # Aggregate picks into clusters with shared event IDs
    # If multiple event_ids share the same agency, attach the partial result to the
    # exception so that the caller can use it as a fallback.
    try:
        pyocto_clusters = aggregate_pick_to_cluster_with_common_event_id(
            pyocto_clusters,
            all_picks_list,
            min_com_phases,
            pick_match_tolerance=associator_cfg.pick_match_tolerance,
        )
    except MultipleEventIDsWithSameAgencyError as e:
        # Re-fuse moveout fragments on the partial result too. The window that
        # produces a fragment is precisely the one that exhausts all tolerances
        # (an unresolvable same-agency conflict), so without this the fix is dead
        # on the very path the pipeline actually returns. pyocto_clusters is the
        # same list object as newclust.clusters (assigned above; aggregate mutates
        # it in place and never changes the cluster count), so fusing it and
        # refreshing n_clusters yields a valid, fragment-free partial.
        n_fused = _apply_moveout_refusion(myclust, pyocto_clusters, pyocto_preloc)
        if n_fused:
            # a fusion shrinks newclust.clusters but
            # clusters_stability was only ever set to the pre-fusion
            # placeholder [1] * n_clusters (above) and clusters_pyocto_ops
            # is never set at all on this exception path (unlike the
            # success path below, which recomputes both). Re-align both
            # to the post-fusion cluster count so a future consumer that
            # zips clusters with clusters_stability/clusters_pyocto_ops by
            # index does not silently read mismatched data.
            newclust.n_clusters = len(newclust.clusters)
            newclust.clusters_stability = np.array(
                [1.0] * newclust.n_clusters, dtype=float
            )
            newclust.clusters_pyocto_ops = [
                {"op": "passthrough", "parent_stabilities": []}
                for _ in range(newclust.n_clusters)
            ]
        e.partial_result = newclust
        raise

    # Re-fuse moveout fragments: the Leiden moveout edge filter can split the
    # picks of one physical event across two PyOcto events (near stations vs
    # far stations). Runs before the cluster count is taken so a fusion is
    # reflected in newclust.n_clusters. Only meaningful when the moveout
    # filter was active.
    _apply_moveout_refusion(myclust, pyocto_clusters, pyocto_preloc)

    newclust.clusters = pyocto_clusters
    newclust.n_clusters = len(newclust.clusters)

    # Inherit HDBSCAN stability into PyOcto clusters and annotate the operation
    # (split / merge / passthrough) by tracking which HDBSCAN parent(s) contributed
    # picks to each PyOcto cluster.
    hdbscan_stabs = list(myclust.clusters_stability)
    pick_key_to_hdbscan: dict = {}
    for hi, hc in enumerate(myclust.clusters):
        stab = float(hdbscan_stabs[hi]) if hi < len(hdbscan_stabs) else 1.0
        for p in hc:
            pick_key_to_hdbscan[(p.station, p.time.datetime, p.phase)] = (hi, stab)

    # Count how many PyOcto clusters each HDBSCAN parent contributed to (for split detection)
    hdbscan_parent_usage: dict = defaultdict(int)
    for pc in newclust.clusters:
        seen_parents: set = set()
        for p in pc:
            k = (p.station, p.time.datetime, p.phase)
            if k in pick_key_to_hdbscan:
                seen_parents.add(pick_key_to_hdbscan[k][0])
        for hi in seen_parents:
            hdbscan_parent_usage[hi] += 1

    inherited_stabs = []
    cluster_ops = []
    for pc in newclust.clusters:
        parent_info: dict = {}  # hi → stab, deduplicated by parent index
        for p in pc:
            k = (p.station, p.time.datetime, p.phase)
            if k in pick_key_to_hdbscan:
                hi, stab = pick_key_to_hdbscan[k]
                parent_info[hi] = stab
        parent_stabs = list(parent_info.values())
        avg_stab = float(np.mean(parent_stabs)) if parent_stabs else 1.0
        inherited_stabs.append(avg_stab)

        parents = set(parent_info.keys())
        if len(parents) > 1:
            op = "merge"
        elif len(parents) == 1 and hdbscan_parent_usage[next(iter(parents))] > 1:
            op = "split"
        else:
            op = "passthrough"
        cluster_ops.append({"op": op, "parent_stabilities": parent_stabs})

    newclust.clusters_stability = np.array(inherited_stabs, dtype=float)
    newclust.clusters_pyocto_ops = cluster_ops

    # Clean up the original cluster object only after successful processing
    for attr in ["clusters", "clusters_stability", "noise", "zones", "preloc"]:
        if hasattr(myclust, attr):
            delattr(myclust, attr)

    return newclust


def _apply_moveout_refusion(
    myclust, clusters: List[List[Phase]], prelocs: Optional[List[dict]]
) -> int:
    """Re-fuse Leiden moveout fragments in ``clusters``, if the moveout filter
    was active for ``myclust``. Returns the number of fusions performed (0 if
    the filter was not active, or if ``prelocs`` is None).

    this wraps the "check _moveout_fusion_params, then call
    _refuse_fragment_events" pattern that used to be duplicated verbatim at
    three call sites (the delegate_dbclust early return, the
    MultipleEventIDsWithSameAgencyError exception path, and the normal
    success path), so a future change to how/when re-fusion runs only needs
    to be made once.
    """
    if prelocs is None:
        return 0
    moveout_vps = _moveout_fusion_params(myclust)
    if moveout_vps is None:
        return 0
    return _refuse_fragment_events(
        clusters,
        prelocs,
        *moveout_vps,
        min_station_count=myclust.min_station_count,
        max_search_dist=myclust.max_search_dist,
    )


def _moveout_fusion_params(myclust) -> Optional[Tuple[float, float]]:
    """Return (vp, vs) of the Leiden moveout filter if it is active, else None.

    The moveout filter (leiden.py) drops edges whose P-P or S-S travel-time
    difference exceeds dist/vp or dist/vs, which can split the picks of one
    physical event across several PyOcto clusters. The re-fusion below is only
    meaningful when that filter was the thing that caused the split.

    Clusterize always sets leiden_vp/leiden_vs to non-zero
    defaults (6.0/3.5) regardless of clustering_method, so a plain
    getattr(..., None) check cannot tell "the filter ran" from "these
    attributes just hold their defaults". The vp > 0 and vs > 0 check below
    is not a fallback for that — it is the actual activation condition:
    leiden.py's moveout filter is gated by that exact same "vp > 0 and
    vs > 0" test (see leiden_cluster(), dist_km_matrix computation), so this
    mirrors it precisely rather than approximating it. If that gating
    condition in leiden.py ever changes, this must be updated to match.
    """
    if myclust.clustering_method != "leiden":
        return None
    vp = getattr(myclust, "leiden_vp", None)
    vs = getattr(myclust, "leiden_vs", None)
    if not vp or not vs or vp <= 0 or vs <= 0:
        return None
    return (float(vp), float(vs))


def _refuse_fragment_events(
    clusters: List[List[Phase]],
    prelocs: List[dict],
    vp: float,
    vs: float,
    min_station_count: int,
    max_search_dist: float,
    station_ratio_max: float = 0.35,
    min_pairs_checked: int = 3,
) -> int:
    """Fuse small PyOcto events back into a compatible larger one after the
    Leiden moveout filter has split them apart.

    The moveout edge filter can remove the pick-to-pick edges that link the
    near stations of one event to its far stations, leaving the near-station
    picks as their own small PyOcto event next to the larger, correct one.
    A pick in such a fragment violates the moveout bound with almost none of
    the main event's picks by construction, so a compatibility test on the
    cross pick-pairs is a safe way to tell "fragment of the same event" from
    "a genuinely separate nearby event": two distinct events that are close
    enough in time to be confused still have first-arrival offsets between
    their stations that exceed the moveout bounds.

    In place: fuses ``clusters[i]`` into ``clusters[j]`` (and vice-versa) by
    concatenating and de-duplicating the picks; the larger event's preloc is
    kept and the smaller one dropped, so ``prelocs`` stays aligned. Returns the
    number of fusions performed.

    Args:
        min_station_count: the pipeline's configured minimum station count
            for a credible event (``Clusterize.min_station_count``). Used to
            derive the fragment/target size gates below instead of
            introducing independent constants — see "size gates" comment.
        max_search_dist: the pseudo-travel-time edge threshold (seconds) the
            clustering step itself uses to decide whether two picks could
            plausibly belong to the same event (``Clusterize.max_search_dist``).
            Reused here, instead of a separate hardcoded value, as the origin
            time compatibility bound: two events are only merge-candidates if
            their preloc origin times are at least as close as that same
            threshold, since anything farther apart wouldn't have been
            considered the same event by the clustering step either.

    Design note (all remaining tunables were previously independent magic
    numbers; each is now either derived from an existing config value above,
    a structural bound with no free parameter, or reduced to the one ratio
    below that is genuinely an empirical choice, not a symptom of one):
        - frag_max_st / target_min_st (used to be separate hardcoded
          constants 6 and 10): replaced by min_station_count + 1 and
          2 * min_station_count respectively, so they track the pipeline's
          own definition of "enough stations for a credible event" instead
          of drifting independently from it.
        - max_origin_diff_s (used to be a hardcoded 10.0s): replaced by
          max_search_dist directly (see Args above) instead of an
          unrelated fixed number of seconds.
        - min_compat_fraction (used to be a configurable-looking float
          always called with 1.0): removed. cross_moveout_ok() now always
          requires every checked cross pair to be compatible — the
          "fraction" machinery existed for a flexibility nothing ever used.
        - max_rounds (used to be a hardcoded 5): removed from the public
          signature. It was never a real tuning knob — the fusion loop
          already stops as soon as a round performs no fusion — so it is
          now derived locally as len(clusters), the true structural upper
          bound (each round fuses at most one pair, removing one cluster).
    """
    frag_max_st = min_station_count + 1
    target_min_st = 2 * min_station_count
    max_origin_diff_s = max_search_dist
    max_rounds = len(clusters)

    def stations(cluster: List[Phase]) -> set:
        seen = set()
        for p in cluster:
            if p.coord is None:
                continue
            seen.add((p.network, p.station))
        return seen

    def cross_moveout_ok(a: List[Phase], b: List[Phase]) -> Tuple[bool, int, int]:
        """(compatible, n_ok, n_checked) over P-P / S-S cross pairs.

        P-S pairs are exempt (the moveout filter also leaves them unbounded).
        STRICT: every checked pair must satisfy the moveout bound — no slack,
        no fraction. (see design note above: min_compat_fraction used to
        make this a configurable threshold, but every caller passed 1.0, so
        the fraction bookkeeping was complexity with no exercised behavior;
        this now short-circuits on the first incompatible pair instead of
        exhaustively scoring all pairs, which is both simpler and cheaper.)

        Also requires at least min_pairs_checked cross pairs before deciding
        compatibility. With very few pairs (e.g. a single shared station),
        one coincidentally-compatible pair would otherwise count as 100%
        evidence and fuse two genuinely distinct nearby events, while a
        single noisy pick could otherwise veto a legitimate fusion.
        """
        n_checked = 0
        n_ok = 0
        for pa in a:
            if pa.coord is None:
                continue
            la1, lo1 = pa.coord["latitude"], pa.coord["longitude"]
            for pb in b:
                if pb.coord is None:
                    continue
                la2, lo2 = pb.coord["latitude"], pb.coord["longitude"]
                if pa.is_p() and pb.is_p():
                    v = vp
                elif pa.is_s() and pb.is_s():
                    v = vs
                else:
                    continue
                d = _haversine_km(la1, lo1, la2, lo2)
                dt = abs(float(pa.time) - float(pb.time))
                n_checked += 1
                if dt <= d / v:
                    n_ok += 1
                else:
                    return False, n_ok, n_checked
        if n_checked < min_pairs_checked:
            return False, n_ok, n_checked
        return True, n_ok, n_checked

    n_fused = 0
    for _ in range(max_rounds):
        fused = False
        n_clusters = len(clusters)
        for i in range(n_clusters):
            if fused or len(clusters[i]) < 3:
                continue
            small_stations = stations(clusters[i])
            if len(small_stations) > frag_max_st:
                continue
            small_preloc = prelocs[i]
            if not small_preloc or not small_preloc.get("time"):
                continue
            # Safety: a fragment is made of detector picks; if any pick already
            # belongs to a catalog event, it is not a moveout fragment and is
            # left alone.
            if any(p.event_id for p in clusters[i]):
                continue
            t_i = small_preloc["time"].timestamp()
            for j in range(n_clusters):
                if j == i or len(clusters[j]) < 2 * len(clusters[i]):
                    continue
                large_stations = stations(clusters[j])
                n_large = len(large_stations)
                if n_large < target_min_st:
                    continue
                if len(small_stations) > station_ratio_max * n_large:
                    continue
                large_preloc = prelocs[j]
                if not large_preloc or not large_preloc.get("time"):
                    continue
                if abs(t_i - large_preloc["time"].timestamp()) > max_origin_diff_s:
                    continue
                ok, n_ok, n_pairs = cross_moveout_ok(clusters[i], clusters[j])
                if not ok:
                    continue
                logger.info(
                    f"moveout fragment re-fusion: fusing {len(clusters[i])} picks "
                    f"({len(small_stations)} stations) into {len(clusters[j])} picks "
                    f"({n_large} stations); cross-pair moveout fraction={n_ok / n_pairs:.2f} "
                    f"({n_ok}/{n_pairs}); "
                    f"origin dt={abs(t_i - large_preloc['time'].timestamp()):.2f}s"
                )
                clusters[j] = _dedup_picks_prefer_event_id(clusters[j] + clusters[i])
                clusters.pop(i)
                prelocs.pop(i)
                n_fused += 1
                fused = True
                break
        if not fused:
            break
    if n_fused:
        logger.info(f"moveout fragment re-fusion: {n_fused} fusion(s) applied")
    return n_fused


def _dedup_picks_prefer_event_id(picks) -> List[Phase]:
    """De-duplicate picks by physical identity (network, station, phase
    family, time), preferring a pick that carries an event_id over one that
    doesn't when both share that identity. Order of ``picks`` sets which
    duplicate is kept as the non-preferred candidate (first occurrence),
    consistent across every caller.

    Do NOT use set(picks): even though Phase.__hash__ intentionally ignores
    channel/location (see Phase.__hash__ docstring), set() picks an
    arbitrary survivor per hash bucket instead of preferring the pick with
    an event_id, and there is no guarantee the dropped pick was truly
    redundant across e.g. differing agency provenance.
    """
    seen: dict = {}
    for p in picks:
        key = (p.network, p.station, p.phase[0].upper(), p.time.datetime)
        cur = seen.get(key)
        if cur is None or (cur.event_id is None and p.event_id is not None):
            seen[key] = p
    return list(seen.values())


# Upper bound on the picks compared per cluster and phase family by the
# moveout check (stride subsample by time): only bounds the cost.
EVENTID_MERGE_MOVEOUT_MAX_PICKS = 200

MOVEOUT_COMPATIBLE = "compatible"
MOVEOUT_INCOMPATIBLE = "incompatible"
MOVEOUT_UNDETERMINED = "undetermined"


def clusters_moveout_fraction(
    c1: List[Phase], c2: List[Phase], vp: float, vs: float
) -> Tuple[Optional[float], int, int, int]:
    """Moveout compatibility of two clusters' picks.

    For each cross pair of picks of the same family (P-P, S-S) recorded at
    stations d km apart, one wavefront cannot be seen with a delay larger than
    the travel time between the stations: |dt| <= d / v. Picks of a single
    earthquake satisfy it; two earthquakes with different origin times violate
    it for many pairs. P-S pairs and same-station pairs (< 1 km) are not
    checked. Each cluster is subsampled (stride, by time) to
    EVENTID_MERGE_MOVEOUT_MAX_PICKS picks per family.

    Returns:
        (fraction of compatible pairs or None if no pair, number of pairs,
        picks of c1 in at least one checked pair, same for c2)
    """
    def sample(cluster, family):
        picks = sorted(
            (
                p for p in cluster
                if p.coord is not None and (p.is_p() if family == "P" else p.is_s())
            ),
            key=lambda p: float(p.time),
        )
        step = max(1, math.ceil(len(picks) / EVENTID_MERGE_MOVEOUT_MAX_PICKS))
        return picks[::step]

    n_pairs = n_ok = n_used1 = n_used2 = 0
    for family, v in (("P", vp), ("S", vs)):
        a, b = sample(c1, family), sample(c2, family)
        if not a or not b:
            continue
        lat_a = np.array([p.coord["latitude"] for p in a])[:, None]
        lon_a = np.array([p.coord["longitude"] for p in a])[:, None]
        lat_b = np.array([p.coord["latitude"] for p in b])[None, :]
        lon_b = np.array([p.coord["longitude"] for p in b])[None, :]
        dist = _haversine_km(lat_a, lon_a, lat_b, lon_b)
        dt = np.abs(
            np.array([float(p.time) for p in a])[:, None]
            - np.array([float(p.time) for p in b])[None, :]
        )
        checked = dist >= 1.0
        n_pairs += int(checked.sum())
        n_ok += int((checked & (dt <= dist / v)).sum())
        n_used1 += int(checked.any(axis=1).sum())
        n_used2 += int(checked.any(axis=0).sum())
    return (n_ok / n_pairs if n_pairs else None), n_pairs, n_used1, n_used2


@dataclass
class EventIdMergeGuard:
    """Guard on PyOcto cluster merges driven by a shared event_id only.

    Decision (see ClusterConfig.eventid_merge_* for the parameters):
      - prelocations close (pseudo travel time <= max_tt_s) or missing: merge;
      - prelocations far: merge only if the picks are moveout-compatible
        (verdict MOVEOUT_COMPATIBLE), refuse otherwise. The moveout never
        vetoes a merge whose prelocations are close.
    """

    max_tt_s: float
    average_velocity: float
    vp: float = 6.0
    vs: float = 3.5
    min_fraction: float = 0.95
    min_picks: int = 3

    def moveout_verdict(self, c1: List[Phase], c2: List[Phase]) -> Tuple[str, str]:
        """(verdict, details for the log)."""
        frac, n_pairs, n1, n2 = clusters_moveout_fraction(c1, c2, self.vp, self.vs)
        details = (
            f"moveout {('n/a' if frac is None else f'{frac:.2f}')} on {n_pairs} pairs, "
            f"{n1}/{n2} checkable picks"
        )
        if frac is None or min(n1, n2) < self.min_picks:
            return MOVEOUT_UNDETERMINED, details
        if frac >= self.min_fraction:
            return MOVEOUT_COMPATIBLE, details
        return MOVEOUT_INCOMPATIBLE, details

    def decide(self, c1, c2, preloc1, preloc2) -> Tuple[bool, str]:
        """(merge allowed, details for the log)."""
        sep = preloc_pseudo_tt(preloc1, preloc2, self.average_velocity)
        if sep is None:
            return True, "no prelocation"
        dist_km, dt_s, tt_s = sep
        details = f"preloc dist: {dist_km:.1f} km, dt: {dt_s:.1f} s, tt: {tt_s:.1f} s"
        if tt_s <= self.max_tt_s:
            return True, details
        verdict, moveout = self.moveout_verdict(c1, c2)
        details += f" (max {self.max_tt_s} s); {moveout}: {verdict}"
        return verdict == MOVEOUT_COMPATIBLE, details


def eventid_merge_guard_from(myclust) -> Optional[EventIdMergeGuard]:
    """EventIdMergeGuard configured from a Clusterize, None when disabled."""
    max_tt_s = getattr(myclust, "eventid_merge_max_tt_s", None)
    if max_tt_s is None:
        return None
    return EventIdMergeGuard(
        max_tt_s=max_tt_s,
        average_velocity=myclust.average_velocity,
        vp=myclust.eventid_merge_moveout_vp,
        vs=myclust.eventid_merge_moveout_vs,
        min_fraction=myclust.eventid_merge_moveout_min_fraction,
        min_picks=myclust.eventid_merge_moveout_min_picks,
    )


def cluster_merge(
    clusters: List[List[Phase]],
    preloc,
    min_com_phases: int,
    eventid_shared_min_picks_per_cluster: Optional[int] = None,
    eventid_shared_min_distinct_ids: int = 2,
    guard: Optional[EventIdMergeGuard] = None,
) -> Tuple[List[List[Phase]], List]:
    """
    Iteratively merges clusters with shared phases or event IDs
    until no more merges are possible.

    Args:
        clusters (List[List[Phase]]): List of clusters to be merged.
        preloc: Preliminary localization data associated with clusters.
        min_com_phases (int): Minimum number of common phases required for merging clusters.
        guard: EventIdMergeGuard applied to merges driven by a shared event_id
            only (None = disabled, see cluster_merge_one_pass).

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
            guard=guard,
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
    guard: Optional[EventIdMergeGuard] = None,
) -> Tuple[List[List[Phase]], List, int]:
    """
    Perform one pass of cluster merging based on shared picks or event IDs.

    Args:
        clusters: List of clusters (each cluster is a list of Phase objects).
        preloc: List of prelocation data corresponding to clusters.
        min_com_phases: Minimum number of shared phases for merging.
        guard: EventIdMergeGuard deciding merges driven by a shared event_id
            only (fewer than min_com_phases common picks): refused when the two
            PyOcto prelocations are far apart, unless the picks are
            moveout-compatible. None = disabled (always merge).

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
            # A merge driven by a shared event_id alone can join two distinct
            # earthquakes (one mis-associated catalog pick is enough).
            details = ""
            if common_count < min_com_phases and guard is not None:
                allowed, why = guard.decide(c1, c2, preloc[c1_idx], preloc[c2_idx])
                if not allowed:
                    logger.info(f"Refused event-ID merge: picks shared: {common_count}, {why}")
                    continue
                details = f", {why}"
            logger.info(
                f"Merging clusters: picks shared: {common_count}, event ID shared: {eventid_shared}"
                f"{details}"
            )
            to_be_merged.append((c1_idx, c2_idx))

    # Merge identified clusters
    merged_indices = set()
    for c1_idx, c2_idx in to_be_merged:
        if c1_idx in merged_indices or c2_idx in merged_indices:
            continue

        # Merge the clusters
        c1, c2 = clusters[c1_idx], clusters[c2_idx]
        clusters[c1_idx] = _dedup_picks_prefer_event_id(chain(c1, c2))

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
    clusters: List[List[Phase]],
    picks: List[Phase],
    pick_count_threshold: int = 3,
    pick_match_tolerance: Optional[float] = None,
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
    tol_label = (
        f"tolerance={pick_match_tolerance:.2f}"
        if pick_match_tolerance is not None
        else "tolerance=?"
    )
    logger.info(
        f"aggregate_pick_to_cluster_with_common_event_id() [{tol_label}]: "
        f"{len(clusters)} clusters"
    )
    for cluster_idx, cluster in enumerate(clusters):
        # Note: cluster_idx here indexes `clusters` (PyOcto's output, aggregated
        # picks already merged in) — a different indexing than the input_cluster#i
        # used in dbclust2pyocto()'s "events found" log (PyOcto's input, before
        # splitting/merging). Do not assume the two indices refer to the same cluster.
        tag = f"[{tol_label}, output_cluster#{cluster_idx}]"
        # Count the occurrences of event_id in the cluster
        event_id_counts = Counter([p.event_id for p in cluster if p.event_id])
        if event_id_counts:
            counts_str = ", ".join(
                f"{eid.split('/')[-1]}({count} picks {'> threshold, will aggregate' if count > pick_count_threshold else f'<= threshold({pick_count_threshold}), skipped'})"
                for eid, count in event_id_counts.most_common()
            )
            logger.info(f"{tag} Cluster has picks from known event(s): {counts_str}")

        # count the number of agency in each event_id in event_id_counts
        # (only for event_ids that actually exceed the threshold and are
        # therefore candidates for aggregation — otherwise a benign trace of
        # picks from an unrelated event_id below the threshold would wrongly
        # trigger a MultipleEventIDsWithSameAgencyError)
        event_id_agency = {}
        for p in picks:
            if p.event_id in event_id_counts and event_id_counts[p.event_id] > pick_count_threshold:
                if p.event_id not in event_id_agency:
                    event_id_agency[p.event_id] = set()
                event_id_agency[p.event_id].add(p.agency)

        # Invert the mapping to find agencies associated with multiple event_ids
        agency_event_map = {}
        for event_id, agencies in event_id_agency.items():
            for agency in agencies:
                if agency not in agency_event_map:
                    agency_event_map[agency] = set()
                agency_event_map[agency].add(event_id)
        if agency_event_map:
            logger.info(f"{tag} agency_event_map: {agency_event_map}")

        # Detect agencies associated with multiple event_ids
        duplicate_agency_event_ids = {
            agency: event_ids
            for agency, event_ids in agency_event_map.items()
            if len(event_ids) > 1
        }

        if duplicate_agency_event_ids:
            logger.warning(
                f"{tag} UNRESOLVED: {sum(len(v) for v in duplicate_agency_event_ids.values())} "
                f"event_id(s) from the same agency still share this cluster after "
                f"aggregation — {duplicate_agency_event_ids}. This tolerance attempt "
                f"will be rejected for the whole window."
            )
            raise MultipleEventIDsWithSameAgencyError(duplicate_agency_event_ids)
        elif event_id_agency:
            logger.info(
                f"{tag} RESOLVED: {len(event_id_agency)} known event_id(s) aggregated "
                f"into this cluster with no same-agency conflict."
            )

        # Check if any event_id has a count > pick_count_threshold
        if not any(count > pick_count_threshold for count in event_id_counts.values()):
            continue

        # Partition picks into those added to cluster and those remaining.
        # Exclude picks already in the cluster up front so extend() below
        # cannot introduce duplicates (picks_to_add is drawn from the full
        # `picks` pool, which includes picks already assigned to clusters).
        existing_ids = set(id(p) for p in cluster)
        picks_to_add = [
            p
            for p in picks
            if p.event_id
            and p.event_id in event_id_counts
            and event_id_counts[p.event_id] > pick_count_threshold
            and id(p) not in existing_ids
        ]
        cluster.extend(picks_to_add)
        clusters[cluster_idx] = cluster
        added = set(id(p) for p in picks_to_add)
        picks = [p for p in picks if id(p) not in added]

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
    cluster: List[Phase], picks: pd.DataFrame, events: pd.DataFrame, assignments: pd.DataFrame
) -> List[List[Phase]]:
    """
    Returns a list of clusters (Phase objects) from PyOcto assignment results.

    Args:
        cluster: Original list of Phase objects for this cluster.
        picks: Filtered picks DataFrame (may have fewer rows than cluster if stations
               were filtered by max_lat_range/max_lon_range). Must have a '_cluster_idx'
               column mapping each row back to its position in cluster.
        events: DataFrame of events found by PyOcto.
        assignments: DataFrame mapping event indices to pick indices (into picks).
    """
    result = []
    for _, row in events.iterrows():
        event_idx = row["idx"]
        df_indices = assignments[assignments["event_idx"] == event_idx]["pick_idx"].to_list()
        cluster_indices = picks.loc[df_indices, "_cluster_idx"].tolist()
        result.append([cluster[j] for j in cluster_indices])
    return result


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km between two lat/lon points."""
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    # Clip before sqrt/arcsin: floating-point rounding can push `a` a hair
    # above 1.0 for near-antipodal or identical points, which would make
    # arcsin raise.
    return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def _ps_pair_confidence(p_pick: Phase, s_pick: Phase) -> Optional[str]:
    """Classify a same-station P/S pairing as "hard", "soft", or None (not a pair).

    See filter_incoherent_ps_pairs docstring for the decision matrix.
    """
    same_agency = p_pick.agency is not None and p_pick.agency == s_pick.agency
    if p_pick.event_id and s_pick.event_id:
        if p_pick.event_id == s_pick.event_id:
            return "hard"
        if same_agency:
            # Same agency split them into distinct events: not a pair.
            return None
        return "soft"
    return "soft"


def _ps_geometry_mismatch(
    p_pick: Phase, s_pick: Phase, event_lat: float, event_lon: float,
    zlim: Tuple[float, float], sp_inv: float,
    epi_err_km: float, vel_err: float, pick_err_s: float,
) -> Optional[Tuple[float, float, float, float]]:
    """Return (sp_observed, sp_expected_min, sp_expected_max, epi_km) if the
    pair is geometrically incompatible with an epicenter at
    (event_lat, event_lon), else None.

    PyOcto depths are poorly constrained, so the origin depth is not trusted:
    the expected S-P delay is the interval spanned by every source depth in
    `zlim` (km). That interval is then widened by an explicit error budget,
    each term covering one error source:
      - `epi_err_km`: PyOcto epicenter error, shifts the epicentral distance;
      - `vel_err`: relative error on the S-P slowness (Vp/Vs uncertainty);
      - `pick_err_s`: absolute S-P picking error, dominant in the near field
        where the S-P delay is short.
    The pair is only flagged when no admissible source can explain the
    observed delay.
    """
    epi_km = _haversine_km(
        p_pick.coord["latitude"], p_pick.coord["longitude"], event_lat, event_lon,
    )
    # Station elevation (m, above sea level) adds to the vertical path.
    elevation_km = (p_pick.coord.get("elevation") or 0.0) / 1000.0
    z_shallow = max(min(zlim) + elevation_km, 0.0)
    z_deep = max(max(zlim) + elevation_km, 0.0)
    dist_min = float(np.hypot(max(epi_km - epi_err_km, 0.0), z_shallow))
    dist_max = float(np.hypot(epi_km + epi_err_km, z_deep))
    sp_expected_min = max(dist_min * sp_inv * (1 - vel_err) - pick_err_s, 0.0)
    sp_expected_max = dist_max * sp_inv * (1 + vel_err) + pick_err_s
    sp_observed = s_pick.time - p_pick.time

    mismatch = sp_observed > sp_expected_max or sp_observed < sp_expected_min
    return (
        (sp_observed, sp_expected_min, sp_expected_max, epi_km) if mismatch else None
    )


def filter_incoherent_ps_pairs(
    cluster_phases: List[Phase],
    event_lat: float,
    event_lon: float,
    zlim: Tuple[float, float],
    vp: float,
    vs: float,
    epi_err_km: float = 10.0,
    vel_err: float = 0.15,
    pick_err_hard: float = 0.5,
    pick_err_soft: float = 1.0,
    external_candidates: Optional[List[Phase]] = None,
) -> Tuple[List[Phase], List[Phase]]:
    """Drop same-station P/S picks whose local S-P delay is incompatible with
    the station's distance to a PyOcto event origin.

    A same-station P and S pair observed close together in time constrains the
    source distance via the S-P delay (dist ~= (t_S - t_P) / (1/vs - 1/vp)).
    If PyOcto assigns both picks to an origin far enough that the implied
    travel-time difference cannot match the observed S-P delay, at least one
    of the two picks does not belong to this origin. Only the origin's
    epicenter is used: its depth is replaced by the admissible `zlim` (km)
    range, since PyOcto depths are unreliable (see _ps_geometry_mismatch).

    The expected S-P interval is widened by an error budget (`epi_err_km`,
    `vel_err`, and a picking error chosen by pairing confidence).

    Confidence in the pairing depends on event_id/agency provenance:
      - same non-null event_id on both picks: catalog-confirmed pair -> hard
        rule, picking error `pick_err_hard`.
      - different non-null event_id from the *same* agency: the agency itself
        distinguished two events -> not a real pair, each pick is judged on
        its own (no S-P coherence check applied).
      - different non-null event_id from different agencies, or either pick
        missing an event_id (automatic picks): unconfirmed pairing -> soft
        rule, larger picking error `pick_err_soft`.
    On a mismatch both picks are dropped: which of the two is the foreign
    one cannot be determined from the pair alone.

    `external_candidates` covers the case where PyOcto assigned only one of
    a same-station P/S pair to this origin (the other stayed unassigned or
    went to a different origin) — the exact failure mode that let a foreign
    P slip into a distant origin while its incompatible S was left out. This
    check is restricted to the hard-confidence case (shared catalog event_id):
    an external candidate was never grouped with the assigned pick by
    Leiden/PyOcto, so "same station, opposite phase" alone is too weak a
    signal in a large pool — an unrelated pick from another event can sit at
    that station by coincidence. Only a shared event_id proves the two picks
    are really a pair; on a mismatch the *assigned* pick alone is dropped
    (the external candidate was never claimed by this origin, so there is
    nothing else to drop from it here).

    Returns (kept, dropped) — `kept`/`dropped` only ever contain picks from
    `cluster_phases`, never from `external_candidates`.
    """
    if vp <= 0 or vs <= 0:
        return cluster_phases, []

    sp_inv = 1.0 / vs - 1.0 / vp
    by_station: Dict[str, List[Phase]] = defaultdict(list)
    for p in cluster_phases:
        by_station[f"{p.network}.{p.station}"].append(p)

    dropped_ids = set()

    # --- Pass 1: pairs where both P and S were assigned to this origin ---
    for station_code, station_phases in by_station.items():
        p_picks = [p for p in station_phases if p.is_p()]
        s_picks = [p for p in station_phases if p.is_s()]
        if not p_picks or not s_picks:
            continue

        for p_pick in p_picks:
            for s_pick in s_picks:
                if s_pick.time <= p_pick.time:
                    continue

                confidence = _ps_pair_confidence(p_pick, s_pick)
                if confidence is None:
                    continue

                mismatch = _ps_geometry_mismatch(
                    p_pick, s_pick, event_lat, event_lon, zlim,
                    sp_inv, epi_err_km, vel_err,
                    pick_err_hard if confidence == "hard" else pick_err_soft,
                )
                if mismatch is None:
                    continue
                sp_observed, sp_min, sp_max, epi_km = mismatch
                logger.info(
                    f"P-S coherence ({confidence}): {station_code} S-P observed="
                    f"{sp_observed:.2f}s, expected {sp_min:.2f}-{sp_max:.2f}s at "
                    f"epi {epi_km:.0f}km (event_id={p_pick.event_id}) -> "
                    "dropping both picks."
                )
                dropped_ids.add(id(p_pick))
                dropped_ids.add(id(s_pick))

    # --- Pass 2: picks assigned alone, checked against an external same-station
    # candidate of the opposite phase (unassigned, or assigned elsewhere).
    #
    # Restricted to the "hard" case (same catalog event_id) only: unlike Pass 1,
    # an external candidate was never grouped with this pick by Leiden/PyOcto,
    # so "same station, opposite phase" alone is a much weaker signal — in a
    # large cluster pool, an unrelated S from another event can coincidentally
    # sit near a valid P at that station. Only a shared event_id is strong
    # enough evidence that the two picks are really a pair; without it we
    # cannot tell a real match from a coincidental neighbor, so no pick is
    # dropped on the "soft" confidence level here.
    if external_candidates:
        ext_by_station: Dict[str, List[Phase]] = defaultdict(list)
        for p in external_candidates:
            ext_by_station[f"{p.network}.{p.station}"].append(p)

        for station_code, station_phases in by_station.items():
            candidates = ext_by_station.get(station_code)
            if not candidates:
                continue
            for pick in station_phases:
                if id(pick) in dropped_ids or not pick.event_id:
                    continue
                for ext_pick in candidates:
                    if ext_pick.event_id != pick.event_id:
                        continue
                    if pick.is_p() and ext_pick.is_s():
                        p_pick, s_pick = pick, ext_pick
                    elif pick.is_s() and ext_pick.is_p():
                        p_pick, s_pick = ext_pick, pick
                    else:
                        continue
                    if s_pick.time <= p_pick.time:
                        continue

                    mismatch = _ps_geometry_mismatch(
                        p_pick, s_pick, event_lat, event_lon, zlim,
                        sp_inv, epi_err_km, vel_err, pick_err_hard,
                    )
                    if mismatch is None:
                        continue
                    sp_observed, sp_min, sp_max, epi_km = mismatch
                    logger.info(
                        f"P-S coherence (hard, external): {station_code} "
                        f"S-P observed={sp_observed:.2f}s, expected "
                        f"{sp_min:.2f}-{sp_max:.2f}s at epi {epi_km:.0f}km "
                        f"(event_id={pick.event_id}) -> "
                        "dropping assigned pick."
                    )
                    dropped_ids.add(id(pick))
                    break

    kept, dropped = [], []
    for p in cluster_phases:
        if id(p) in dropped_ids:
            dropped.append(p)
        else:
            kept.append(p)
    return kept, dropped


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
