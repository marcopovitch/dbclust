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
    initial_tolerance = associator.pick_match_tolerance
    tolerance = initial_tolerance

    best_result = None
    best_n_clusters = 0

    logger.info(
        f"--- Single-phase: tolerance decay [{min_tolerance:.2f}..{tolerance:.2f}] "
        f"on {sum(len(c) for c in myclust.clusters)} picks ---"
    )
    while tolerance >= min_tolerance:
        logger.info(f"  Trying pick_match_tolerance: {tolerance:.2f}")
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
            logger.info(f"--- Single-phase: SUCCESS at tolerance={tolerance:.2f} ---")
            associator.pick_match_tolerance = initial_tolerance
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

    logger.error("--- Single-phase: FAILED (exhausted all tolerances) ---")
    associator.pick_match_tolerance = initial_tolerance
    if best_result is not None:
        logger.warning(
            f"Returning best partial result with {best_n_clusters} clusters "
            f"despite unresolved agency conflict."
        )
        return best_result
    logger.error("No partial result available. Skipping pyocto processing.")
    return None


def _clusters_share_stations(c1, c2, min_common=3):
    """Return True if two pick-lists share at least min_common unique stations.

    Used to suppress Phase 2b DL-only clusters that are near-duplicates of
    a Phase 1 catalogued cluster: if they share enough stations they represent
    the same physical earthquake, regardless of origin-time or location.
    """
    s1 = {(p.network, p.station) for p in c1}
    s2 = {(p.network, p.station) for p in c2}
    return len(s1 & s2) >= min_common


def adjust_associator_tolerance_two_phase(
    myclust,
    cfg,
    tolerance_steps={1: 0.5, 0: 0.1},
    min_tolerance=0.5,
    include_noise_in_aggregation=False,
    log_level=logging.INFO,
):
    """Two-phase PyOcto tolerance calibration.

    Phase 1: linear decay on catalogued picks only (picks with event_id) —
    fast convergence on small volume.
    Phase 2: single final pass on all picks with the calibrated tolerance.

    Falls back to single-phase behaviour if no catalogued picks are available.
    """
    associator = cfg.pyocto.current_model.associator

    # Build catalogued-only clusters for phase 1 calibration.
    # Include noise picks with event_id as an extra cluster so that distant
    # catalogued stations (placed in noise by HDBSCAN due to max_search_dist)
    # are also visible to PyOcto during calibration.
    catalogued_clusters = [[p for p in c if p.event_id] for c in myclust.clusters]
    catalogued_clusters = [c for c in catalogued_clusters if c]
    noise_cat = [p for p in getattr(myclust, 'noise', []) if p.event_id]
    if noise_cat:
        catalogued_clusters.append(noise_cat)

    if not catalogued_clusters:
        logger.info("No catalogued picks found, falling back to single-phase decay.")
        return adjust_associator_tolerance(
            myclust, cfg, tolerance_steps, min_tolerance,
            include_noise_in_aggregation, log_level,
        )

    n_cat = sum(len(c) for c in catalogued_clusters)
    n_total = sum(len(c) for c in myclust.clusters)
    logger.info(
        f"--- Two-phase calibration: {n_cat} catalogued / {n_total} total picks ---"
    )

    # Phase 1: decay on catalogued picks to find best tolerance
    myclust_cat = copy.copy(myclust)
    myclust_cat.clusters = catalogued_clusters

    initial_tolerance = associator.pick_match_tolerance  # Save initial tolerance
    best_tolerance = initial_tolerance
    tolerance = best_tolerance
    best_result_cat = None
    best_n_clusters = 0
    phase1_converged = False  # Track if phase 1 converged successfully

    logger.info(
        f"--- Phase 1: tolerance decay [{min_tolerance:.2f}..{tolerance:.2f}] "
        f"on {n_cat} catalogued picks ---"
    )
    result_cat = None
    while tolerance >= min_tolerance:
        logger.info(f"  Trying pick_match_tolerance: {tolerance:.2f} (catalogued only)")
        associator.pick_match_tolerance = tolerance
        try:
            result_cat = dbclust2pyocto(
                myclust_cat,
                cfg.pyocto.default_model_name,
                associator,
                cfg.pyocto.velocity_model,
                cfg.cluster.min_picks_common,
                delegate_dbclust=False,
                include_noise_in_aggregation=False,
                log_level=log_level,
            )
            best_tolerance = tolerance
            phase1_converged = True
            logger.info(f"--- Phase 1: SUCCESS, calibrated tolerance={best_tolerance:.2f} ---")
            break
        except pyproj.exceptions.ProjError:
            raise
        except MultipleEventIDsWithSameAgencyError as e:
            logger.warning(
                f"Unsuccessful with pick_match_tolerance: {tolerance:.2f} (catalogued)."
            )
            if e.partial_result is not None:
                n = e.partial_result.n_clusters
                if n > best_n_clusters:
                    best_n_clusters = n
                    best_result_cat = e.partial_result
                    best_tolerance = tolerance
            step = next((s for t, s in tolerance_steps.items() if tolerance > t), 0.5)
            tolerance -= step
    else:
        logger.warning(
            f"--- Phase 1: FAILED (exhausted tolerances), "
            f"using best partial tolerance={best_tolerance:.2f} ---"
        )
        result_cat = best_result_cat

    if result_cat is None or result_cat.n_clusters == 0:
        # No catalogued clusters to use as seeds — fall back to single-phase on all picks
        logger.warning("Phase 1 produced no clusters, falling back to single-phase on all picks.")
        associator.pick_match_tolerance = initial_tolerance
        return adjust_associator_tolerance(
            myclust, cfg, tolerance_steps, min_tolerance,
            include_noise_in_aggregation, log_level,
        )

    # Phase 2a: for each phase-1 cluster (seed), run PyOcto with initial_tolerance
    # on seed + all remaining DL picks.  Keep only the cluster that contains the
    # seed's catalogued picks; the other clusters produced by PyOcto are put back
    # into the DL residual pool for phase 2b.
    dl_picks = [p for c in myclust.clusters for p in c if not p.event_id]
    if include_noise_in_aggregation:
        dl_picks += [p for p in getattr(myclust, 'noise', []) if not p.event_id]
    dl_pick_ids = {id(p) for p in dl_picks}

    logger.info(
        f"--- Phase 2a: enriching {result_cat.n_clusters} phase-1 cluster(s) "
        f"with {len(dl_picks)} DL picks at tolerance={initial_tolerance:.2f} ---"
    )

    associator.pick_match_tolerance = initial_tolerance
    enriched_clusters = []
    enriched_preloc = []

    # Process largest clusters first
    cat_order = sorted(range(result_cat.n_clusters), key=lambda i: len(result_cat.clusters[i]), reverse=True)

    # Noise pool maintained across iterations (catalogued + DL noise not yet consumed)
    remaining_noise = list(getattr(myclust, 'noise', []))

    for i in cat_order:
        cat_cluster = result_cat.clusters[i]
        cat_pick_ids = {id(p) for p in cat_cluster}

        myclust_single = copy.copy(myclust)
        myclust_single.clusters = [cat_cluster + dl_picks]
        myclust_single.n_clusters = 1
        myclust_single.clusters_stability = [1.0]
        myclust_single.noise = remaining_noise
        myclust_single.preloc = [result_cat.preloc[i]] if result_cat.preloc else []

        try:
            r = dbclust2pyocto(
                myclust_single,
                cfg.pyocto.default_model_name,
                associator,
                cfg.pyocto.velocity_model,
                cfg.cluster.min_picks_common,
                delegate_dbclust=True,
                include_noise_in_aggregation=include_noise_in_aggregation,
                skip_aggregation=True,
                log_level=log_level,
            )
        except (MultipleEventIDsWithSameAgencyError, pyproj.exceptions.ProjError):
            r = None

        if r is not None and r.n_clusters > 0:
            # Find the cluster that contains the seed's catalogued picks
            best_cluster, best_idx = max(
                ((c, idx) for idx, c in enumerate(r.clusters)),
                key=lambda ci: sum(1 for p in ci[0] if id(p) in cat_pick_ids)
            )
            best_preloc = r.preloc[best_idx] if r.preloc and best_idx < len(r.preloc) else None

            # Re-inject catalogued picks (with event_id) from the Phase 2a seed that
            # PyOcto rejected (unassigned). Only picks whose event_id is already
            # represented in best_cluster are re-injected — this ensures they belong
            # to the same physical event. NLL will judge them via residuals.
            best_event_ids = {p.event_id for p in best_cluster if p.event_id}
            best_keys = {
                (p.network, p.station, p.phase[0].upper(), p.time.datetime)
                for p in best_cluster
            }
            assigned_pick_ids = {id(p) for c in r.clusters for p in c}
            rescued = [
                p for p in cat_cluster
                if id(p) not in assigned_pick_ids
                and p.event_id in best_event_ids
                and (p.network, p.station, p.phase[0].upper(), p.time.datetime) not in best_keys
            ]
            if rescued:
                logger.info(
                    f"  Phase-1 cluster {i}: re-injecting {len(rescued)} catalogued pick(s) "
                    f"rejected by PyOcto but matching known event_ids."
                )
                best_cluster = best_cluster + rescued

            enriched_clusters.append(best_cluster)
            enriched_preloc.append(best_preloc)

            # DL picks from spurious clusters and unused picks go to the residual pool
            all_r_pick_ids = {id(p) for c in r.clusters for p in c}
            other_dl = [
                p for idx, c in enumerate(r.clusters) if idx != best_idx
                for p in c if id(p) in dl_pick_ids
            ]
            unused_dl = [p for p in dl_picks if id(p) not in all_r_pick_ids]
            dl_picks = other_dl + unused_dl
            dl_pick_ids = {id(p) for p in dl_picks}
            # Update remaining noise pool (remove noise picks consumed by best_cluster)
            best_pick_ids = {id(p) for p in best_cluster}
            remaining_noise = [p for p in remaining_noise if id(p) not in best_pick_ids]
            n_spurious = r.n_clusters - 1
            logger.info(
                f"  Phase-1 cluster {i}: enriched to {len(best_cluster)} picks, "
                f"{n_spurious} spurious cluster(s) returned to DL pool, "
                f"{len(dl_picks)} DL picks remaining"
            )
        else:
            enriched_clusters.append(cat_cluster)
            enriched_preloc.append(result_cat.preloc[i] if result_cat.preloc else None)
            logger.info(f"  Phase-1 cluster {i}: kept as-is (PyOcto found nothing)")

    # Phase 2b: run PyOcto on residual DL picks to find DL-only events
    if dl_picks:
        logger.info(
            f"--- Phase 2b: searching {len(dl_picks)} residual DL picks "
            f"for DL-only events at tolerance={initial_tolerance:.2f} ---"
        )
        myclust_dl = copy.copy(myclust)
        myclust_dl.clusters = [dl_picks]
        myclust_dl.n_clusters = 1
        myclust_dl.clusters_stability = [1.0]
        myclust_dl.noise = []
        myclust_dl.preloc = []
        try:
            r_dl = dbclust2pyocto(
                myclust_dl,
                cfg.pyocto.default_model_name,
                associator,
                cfg.pyocto.velocity_model,
                cfg.cluster.min_picks_common,
                delegate_dbclust=False,
                include_noise_in_aggregation=False,
                log_level=log_level,
            )
            if r_dl is not None and r_dl.n_clusters > 0:
                dl_prelocs = r_dl.preloc if r_dl.preloc else [None] * r_dl.n_clusters
                filtered_clusters, filtered_preloc = [], []
                for dl_clust, dl_pre in zip(r_dl.clusters, dl_prelocs):
                    if any(_clusters_share_stations(dl_clust, ec) for ec in enriched_clusters):
                        n_shared = max(
                            len(
                                {(p.network, p.station) for p in dl_clust}
                                & {(p.network, p.station) for p in ec}
                            )
                            for ec in enriched_clusters
                        )
                        logger.info(
                            f"--- Phase 2b: suppressing DL-only cluster"
                            f" ({len(dl_clust)} picks, {n_shared} stations shared with Phase 1)"
                            f" — duplicate of a catalogued cluster ---"
                        )
                    else:
                        filtered_clusters.append(dl_clust)
                        filtered_preloc.append(dl_pre)
                if filtered_clusters:
                    logger.info(f"--- Phase 2b: found {len(filtered_clusters)} DL-only cluster(s) ---")
                    enriched_clusters += filtered_clusters
                    enriched_preloc += filtered_preloc
                else:
                    logger.info("--- Phase 2b: no DL-only events found (all suppressed as duplicates) ---")
            else:
                logger.info("--- Phase 2b: no DL-only events found ---")
        except (MultipleEventIDsWithSameAgencyError, pyproj.exceptions.ProjError):
            logger.warning("--- Phase 2b: PyOcto failed on residual DL picks, skipping ---")

    if not enriched_clusters:
        logger.warning("Phase 2 produced no clusters.")
        associator.pick_match_tolerance = initial_tolerance
        return result_cat

    associator.pick_match_tolerance = initial_tolerance
    newclust = copy.copy(myclust)
    newclust.clusters = enriched_clusters
    newclust.n_clusters = len(enriched_clusters)
    newclust.clusters_stability = [1.0] * len(enriched_clusters)
    newclust.preloc = enriched_preloc
    logger.info(f"--- Phase 2: SUCCESS, {newclust.n_clusters} cluster(s) total ---")
    return newclust


# def create_velocity_model(velocity_cfg: dict, model_path: str) -> None:
#     """
#     Create a 1D velocity model and save it to the specified path.

#     Parameters:
#         velocity_cfg (dict): Configuration dictionary containing the following keys:
#             - "depth" (list or array-like): Depth values for the model.
#             - "vp" (list or array-like): P-wave velocities for a given depths list.
#             - "vs" (list or array-like): S-wave velocities for a given  depths list.
#             - "grid_spacing_km" (float): Grid spacing in kilometers.
#             - "max_horizontal_dist_km" (float): Maximum distance in the horizontal direction in kilometers.
#             - "max_vertical_dist_km" (float): Maximum distance in the vertical direction in kilometers.
#         model_path (str): Path where the velocity model will be saved.

#     Returns:
#         None
#     """
#     model = pd.DataFrame(
#         {
#             "depth": velocity_cfg["depth"],
#             "vp": velocity_cfg["vp"],
#             "vs": velocity_cfg["vs"],
#         }
#     )

#     pyocto.VelocityModel1D.create_model(
#         model,
#         velocity_cfg["grid_spacing_km"],  # Grid spacing in kilometer
#         velocity_cfg[
#             "max_horizontal_dist_km"
#         ],  # Maximum distance in horizontal direction in km
#         velocity_cfg[
#             "max_vertical_dist_km"
#         ],  # Maximum distance in vertical direction in km
#         model_path,
#     )


def get_effective_min_pick_fraction(
    cluster: list,
    associator_cfg,
    cluster_idx: int = 0,
) -> float:
    """Compute the effective min_pick_fraction for a cluster.

    When adaptive_min_pick_fraction is enabled and multiple distinct event_ids
    are present (multi-event cluster), the fraction is scaled by median DL
    probability and floored at min_pick_fraction_floor, allowing PyOcto to
    find smaller events within a densely populated cluster.

    Returns associator_cfg.min_pick_fraction unchanged in all other cases.
    """
    if not associator_cfg.adaptive_min_pick_fraction:
        return associator_cfg.min_pick_fraction

    dl_method_ids = {m.upper() for m in associator_cfg.dl_method_ids}
    dl_probas = [
        p.proba for p in cluster
        if p.method is not None
        and isinstance(p.method, str)
        and p.method.upper() in dl_method_ids
    ]
    median_proba = statistics.median(dl_probas) if dl_probas else 1.0
    cluster_event_ids = {p.event_id for p in cluster if p.event_id}

    if len(cluster_event_ids) > 1:
        # Multiple catalogued events in cluster: scale fraction by DL quality
        effective = max(
            associator_cfg.min_pick_fraction_floor,
            associator_cfg.min_pick_fraction * median_proba,
        )
        logger.info(
            f"Cluster#{cluster_idx}: {len(cluster_event_ids)} distinct event_ids, "
            f"median DL proba={median_proba:.3f} ({len(dl_probas)} DL picks), "
            f"min_pick_fraction: {associator_cfg.min_pick_fraction} -> {effective:.3f} "
            f"(floor={associator_cfg.min_pick_fraction_floor})"
        )
        return effective

    return associator_cfg.min_pick_fraction


def dbclust2pyocto(
    myclust: Clusterize,
    model_name: str,
    associator_cfg: Associator,
    velocity_model: pyocto.VelocityModel1D,
    min_com_phases: int,
    delegate_dbclust: bool = False,
    include_noise_in_aggregation: bool = False,
    skip_aggregation: bool = False,
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
        skip_aggregation (bool, optional): If True, skip aggregate_pick_to_cluster_with_common_event_id().
            Used in Phase 2a where the seed already contains the catalogued picks. Defaults to False.
        log_level (int, optional): Logging level. Defaults to logging.INFO.

    Returns:
        Clusterize: A new Clusterize object with processed clusters.
    """
    logger.info(
        f"Using pyocto to process clusters ({sum(len(c) for c in myclust.clusters)} picks)"
    )

    noise_picks = list(myclust.noise) if include_noise_in_aggregation else []
    if noise_picks:
        logger.info(f"Including {len(noise_picks)} HDBSCAN noise picks in aggregation pool")
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

        effective_min_pick_fraction = get_effective_min_pick_fraction(
            cluster, associator_cfg, cluster_idx=i
        )

        # Step 1: pre-filter stations to max_lat_range/max_lon_range BEFORE computing
        # the range, so that extreme outliers do not corrupt min/max calculations.
        if associator_cfg.max_lat_range is not None:
            mask_out_lat = (
                stations["latitude"] < associator_cfg.max_lat_range[0]
            ) | (
                stations["latitude"] > associator_cfg.max_lat_range[1]
            )
            for row in stations.loc[mask_out_lat].itertuples():
                logger.warning(
                    f"Station {row.id} at lat: {row.latitude}, lon: {row.longitude} "
                    "is outside max_lat_range and will be excluded."
                )
            stations = stations[~mask_out_lat].reset_index(drop=True)

        if associator_cfg.max_lon_range is not None:
            mask_out_lon = (
                stations["longitude"] < associator_cfg.max_lon_range[0]
            ) | (
                stations["longitude"] > associator_cfg.max_lon_range[1]
            )
            for row in stations.loc[mask_out_lon].itertuples():
                logger.warning(
                    f"Station {row.id} at lat: {row.latitude}, lon: {row.longitude} "
                    "is outside max_lon_range and will be excluded."
                )
            stations = stations[~mask_out_lon].reset_index(drop=True)

        if stations.empty:
            logger.warning(f"Cluster#{i}: no stations left after range filtering, skipping.")
            continue

        # Filter picks to only keep those whose station is still in the stations df
        valid_station_ids = set(stations["id"])
        picks = picks[picks["station"].isin(valid_station_ids)].reset_index(drop=True)
        if picks.empty:
            logger.warning(f"Cluster#{i}: no picks left after station range filtering, skipping.")
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
                    1 for phases in station_phases.values()
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

        assigned_pick_ids = set(assignments["pick_idx"].to_list()) if len(assignments) else set()
        n_unassigned = len(cluster) - len(assigned_pick_ids)
        logger.info(
            f"\t{len(events)} events found in cluster#{i} with {len(cluster)} picks"
            f" ({n_unassigned} unassigned by PyOcto)"
        )

    # Merge clusters with common picks or event IDs
    pyocto_clusters, pyocto_preloc = cluster_merge(
        pyocto_clusters, pyocto_preloc, min_com_phases
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

    # Only re-inject picks that PyOcto itself validated (same Phase object identity).
    # all_picks_list may contain catalog picks from distant stations (e.g., national
    # networks) that PyOcto correctly excluded via its velocity-model geographic range.
    # Re-injecting those bypasses PyOcto's seismological validation and causes NLLoc
    # to receive incoherent picks → NaN uncertainty on pass 0 → event silently dropped.
    pyocto_selected_ids = {id(p) for c in pyocto_clusters for p in c}
    picks_for_aggregation = [p for p in all_picks_list if id(p) in pyocto_selected_ids]

    # Aggregate picks into clusters with shared event IDs.
    # Skipped in Phase 2a (skip_aggregation=True) because the seed already contains
    # the catalogued picks — re-aggregating would inject duplicates into best_cluster.
    if not skip_aggregation:
        try:
            pyocto_clusters = aggregate_pick_to_cluster_with_common_event_id(
                pyocto_clusters, picks_for_aggregation, min_com_phases
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

        # Check if clusters share at least 1 event_id that appears >= 2 times in BOTH clusters.
        # PyOcto may split a HDBSCAN cluster into sub-clusters that each inherit picks from the
        # same catalogued events. Requiring >= 2 picks in each cluster for a shared event_id
        # prevents spurious merges from a single contaminated pick, while still catching genuine
        # splits even when only 1 agency sees the event (1 shared event_id with enough picks).
        c1_counts = Counter(p.event_id for p in c1 if p.event_id)
        c2_counts = Counter(p.event_id for p in c2 if p.event_id)
        c1_event_ids = set(c1_counts.keys())
        c2_event_ids = set(c2_counts.keys())
        significant_shared = {
            eid for eid in (c1_event_ids & c2_event_ids)
            if c1_counts[eid] >= 2 and c2_counts[eid] >= 2
        }
        identical_event_ids = len(significant_shared) >= 1

        if common_count >= min_com_phases or eventid_shared or identical_event_ids:
            logger.info(
                f"Merging clusters: picks shared: {common_count}, event ID shared: {eventid_shared}, identical event_ids: {identical_event_ids}"
            )
            to_be_merged.append((c1_idx, c2_idx))

    # Merge identified clusters
    merged_indices = set()
    for c1_idx, c2_idx in to_be_merged:
        if c1_idx in merged_indices or c2_idx in merged_indices:
            continue

        # Merge the clusters, deduplicating by physical pick identity (network/station/phase/time),
        # preferring picks with event_id over those without
        c1, c2 = clusters[c1_idx], clusters[c2_idx]
        seen = {}
        for p in c1 + c2:
            key = (p.network, p.station, p.phase[0].upper(), p.time.datetime)
            if key not in seen or (seen[key].event_id is None and p.event_id is not None):
                seen[key] = p
        merged_cluster = list(seen.values())
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
    # Pre-compute event_id counts per cluster to assign each event_id
    # to the cluster that already has the most picks for it.
    cluster_event_counts = [
        Counter(p.event_id for p in cluster if p.event_id)
        for cluster in clusters
    ]

    # For each event_id, find the cluster index with the highest count
    event_id_best_cluster = {}
    for cluster_idx, counts in enumerate(cluster_event_counts):
        for eid, count in counts.items():
            if count > pick_count_threshold:
                if eid not in event_id_best_cluster or count > event_id_best_cluster[eid][1]:
                    event_id_best_cluster[eid] = (cluster_idx, count)

    already_aggregated_event_ids = set()
    for cluster_idx, cluster in enumerate(clusters):
        event_id_counts = cluster_event_counts[cluster_idx]
        if event_id_counts:
            counts_str = ", ".join(
                f"{eid.split('/')[-1]}({count} picks {'> threshold, will aggregate' if count > pick_count_threshold else f'<= threshold({pick_count_threshold}), skipped'})"
                for eid, count in event_id_counts.most_common()
            )
            logger.info(f"Cluster has picks from known event(s): {counts_str}")

        # Count agencies per event_id, but only for event_ids eligible for aggregation
        # (strictly above pick_count_threshold). Below-threshold event_ids are minor
        # contamination that will be skipped anyway, so they must not trigger a conflict.
        event_id_agency = {}
        for p in cluster:
            if p.event_id and event_id_counts.get(p.event_id, 0) > pick_count_threshold:
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
            logger.info(f"agency_event_map: {agency_event_map}")

        # Detect agencies associated with multiple event_ids
        duplicate_agency_event_ids = {
            agency: event_ids
            for agency, event_ids in agency_event_map.items()
            if len(event_ids) > 1
        }

        if duplicate_agency_event_ids:
            raise MultipleEventIDsWithSameAgencyError(duplicate_agency_event_ids)

        # Only aggregate event_ids for which this cluster is the best match
        # and that have not already been aggregated into another cluster.
        eligible_event_ids = {
            eid for eid, count in event_id_counts.items()
            if count > pick_count_threshold
            and eid not in already_aggregated_event_ids
            and event_id_best_cluster.get(eid, (None,))[0] == cluster_idx
        }
        if not eligible_event_ids:
            continue

        # Partition picks into those added to cluster and those remaining
        picks_to_add = [
            p for p in picks
            if p.event_id and p.event_id in eligible_event_ids
        ]
        cluster.extend(picks_to_add)
        added = set(id(p) for p in picks_to_add)
        picks = [p for p in picks if id(p) not in added]
        already_aggregated_event_ids.update(eligible_event_ids)

        # Remove duplicates by physical pick identity (network, station, phase, time),
        # preferring picks with an event_id over those without (catalog over DL picks).
        seen: dict = {}
        for p in cluster:
            key = (p.network, p.station, p.phase[0].upper(), p.time.datetime)
            if key not in seen or (seen[key].event_id is None and p.event_id is not None):
                seen[key] = p
        cluster = list(seen.values())

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
