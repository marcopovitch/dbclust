"""Graph-based seismic event clustering via the Leiden algorithm.

Nodes  = picks (Phase objects)
Edges  = pairs with d_tt < max_search_dist
Weights= exp(-d_tt / sigma) * p_i * p_j          for cross-station pairs
         p_i * p_j * ps_boost_factor                for same-station P-S pairs

The Leiden CPM objective finds communities that maximise internal edge density
minus resolution_parameter × (expected density under a random model).  A higher
resolution produces more, smaller communities.

Same-station P-S pairs receive a distance-free weight because the S-P time is a
physical property of the event (not a dissimilarity measure): for dd=0, the TT
component is purely the S-P delay, which should not penalise the pairing.
The boost weight p_P*p_S*ps_boost_factor makes it very costly for CPM to put
P and S from the same station into different communities.  With ps_boost_factor
large enough (>> 1/resolution), separating the pair never yields a positive CPM
gain.  A post-partition union-find step further enforces this constraint in case
Leiden still splits a pair.

Communities with fewer than min_cluster_size picks are reclassified as noise
(label −1), preserving the same return contract as get_clusters().

Pick probabilities enter as multiplicative edge weights: two high-confidence
picks form a strong edge; a low-confidence pick anchors to a cluster only if
the geometry (TT distance) is compelling enough.

Requires: leidenalg, python-igraph  (uv pip install leidenalg python-igraph)
"""

import logging
from collections import defaultdict

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Edge construction
# ---------------------------------------------------------------------------

def _build_edges(
    pseudo_tt: np.ndarray,
    probas: np.ndarray,
    stations: np.ndarray,
    is_p: np.ndarray,
    is_s: np.ndarray,
    times: np.ndarray,
    max_search_dist: float,
    sigma: float,
    resolution: float,
    ps_boost_factor: float = 100.0,
    min_edge_weight: float = 0.0,
    dist_km: np.ndarray = None,
    vp: float = 6.0,
    vs: float = 3.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return edge arrays for the TT graph with P-S boost applied.

    Edges with final weight < min_edge_weight are removed after boost.
    Boosted P-S same-station edges are never removed regardless of min_edge_weight.

    Returns
    -------
    rows, cols : upper-triangle edge indices
    weights : boosted edge weights (list-ready)
    weights_no_boost : pre-boost weights used for stability computation
    ps_same : boolean mask — True where the edge is a boosted P-S pair
    """
    mask = np.triu(pseudo_tt < max_search_dist, k=1)
    rows, cols = np.where(mask)

    # Phase-aware moveout compatibility filter.
    # For a pair (i, j), the observed Δt must be compatible with at least one
    # physical source location.  By the triangle inequality on travel times:
    #   PP: |t_i - t_j| ≤ dist(s_i, s_j) / vp
    #   SS: |t_i - t_j| ≤ dist(s_i, s_j) / vs
    #   PS/SP: no moveout bound — the valid Δt range depends on source position
    #     relative to both stations, making a station-pair bound too restrictive.
    #     Same-station PS is gated by max_search_dist; cross-station PS by the
    #     PS-boost causal guard (t_S > t_P) and ps_dt_max on same-station pairs.
    # Edges that violate these bounds cannot come from the same event.
    if dist_km is not None:
        abs_dt = np.abs(times[rows] - times[cols])
        d = dist_km[rows, cols]
        pp = is_p[rows] & is_p[cols]
        ss = is_s[rows] & is_s[cols]
        # PP: |Δt| ≤ d/vp — SS: |Δt| ≤ d/vs — PS: no bound (geometry too complex)
        dt_max = np.where(pp, d / vp, np.where(ss, d / vs, np.inf))
        incompatible = abs_dt > dt_max
        n_incompatible = int(np.sum(incompatible))
        if n_incompatible:
            keep = ~incompatible
            rows, cols = rows[keep], cols[keep]
            logger.info(
                "Leiden moveout filter: removed %d incompatible edge(s) "
                "(PP>d/vp, SS>d/vs, or PS-cross>d/vs).", n_incompatible,
            )

    edge_w = probas[rows] * probas[cols]

    weights = np.exp(-pseudo_tt[rows, cols] / sigma) * edge_w
    weights_no_boost = weights.copy()

    # Same-station P-S pairs: replace weight with p_P * p_S * ps_boost_factor.
    # A large ps_boost_factor (>> 1/resolution) makes it unprofitable for CPM
    # to separate P and S from the same station into different communities.
    # PS-boost conditions:
    #   1. same station
    #   2. causal ordering: t_S > t_P
    ps_same = (
        (stations[rows] == stations[cols])
        & (
            (is_p[rows] & is_s[cols] & (times[cols] > times[rows]))
            | (is_s[rows] & is_p[cols] & (times[rows] > times[cols]))
        )
    )
    weights[ps_same] = edge_w[ps_same] * ps_boost_factor

    logger.info(
        "Leiden PS boost: %d same-station P-S edges (out of %d total edges), factor=%.1f.",
        int(np.sum(ps_same)), len(rows), ps_boost_factor,
    )

    # Drop weak edges, but always keep boosted P-S same-station edges.
    if min_edge_weight > 0.0:
        keep = (weights >= min_edge_weight) | ps_same
        n_dropped = int(np.sum(~keep))
        if n_dropped:
            logger.info(
                "Leiden min_edge_weight=%.4f: dropped %d/%d weak edges.",
                min_edge_weight, n_dropped, len(rows),
            )
        rows, cols = rows[keep], cols[keep]
        weights, weights_no_boost = weights[keep], weights_no_boost[keep]
        ps_same = ps_same[keep]

    return rows, cols, weights, weights_no_boost, ps_same



# ---------------------------------------------------------------------------
# Post-partition P-S enforcement (union-find)
# ---------------------------------------------------------------------------

def _merge_split_ps_pairs(
    raw_labels: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    ps_same: np.ndarray,
) -> np.ndarray:
    """Merge communities that Leiden split across a boosted P-S edge.

    Despite the high boost weight, CPM may still separate a P-S pair if the
    global objective benefits from it.  This union-find pass enforces the hard
    seismic constraint: P and S at the same station must be in the same cluster.

    Returns merged_labels (same shape as raw_labels, community ids remapped).
    """
    n_raw = int(raw_labels.max()) + 1 if len(raw_labels) > 0 else 0
    parent = list(range(n_raw))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        parent[find(x)] = find(y)

    n_merged = 0
    for edge_idx in np.where(ps_same)[0]:
        r, c = int(rows[edge_idx]), int(cols[edge_idx])
        cr, cc = raw_labels[r], raw_labels[c]
        if find(cr) != find(cc):
            union(cr, cc)
            n_merged += 1

    if n_merged:
        logger.info(
            "Leiden post-merge: %d split P-S pair(s) forced into same community.",
            n_merged,
        )

    # Remap raw community ids through union-find to get contiguous labels
    root_to_label: dict[int, int] = {}
    merged_labels = np.empty_like(raw_labels)
    for node, raw in enumerate(raw_labels):
        root = find(int(raw))
        if root not in root_to_label:
            root_to_label[root] = len(root_to_label)
        merged_labels[node] = root_to_label[root]

    return merged_labels


# ---------------------------------------------------------------------------
# Noise S-pick diagnostics
# ---------------------------------------------------------------------------

def _log_noise_s_diagnostics(
    phases: list,
    labels: np.ndarray,
    is_s: np.ndarray,
    is_p: np.ndarray,
    stations: np.ndarray,
    pseudo_tt: np.ndarray,
    ps_same: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    max_search_dist: float,
) -> None:
    """Log a breakdown of why S picks ended up as noise."""
    ps_boosted_nodes = set(rows[ps_same].tolist()) | set(cols[ps_same].tolist())
    noise_s = [i for i, (lbl, s) in enumerate(zip(labels, is_s)) if lbl == -1 and s]
    if not noise_s:
        return

    with_boost = sum(1 for i in noise_s if i in ps_boosted_nodes)
    no_p, mismatch, tt_too_large = 0, 0, 0
    no_p_names: list[str] = []

    for s_idx in noise_s:
        if s_idx in ps_boosted_nodes:
            continue
        net_stn = stations[s_idx]
        p_same = np.where((stations == net_stn) & is_p)[0]
        if len(p_same) == 0:
            no_p += 1
            no_p_names.append(f"{net_stn}@{phases[s_idx].time.strftime('%H:%M:%S')}")
            continue
        min_tt = min(pseudo_tt[min(s_idx, p), max(s_idx, p)] for p in p_same)
        if min_tt < max_search_dist:
            mismatch += 1
        else:
            tt_too_large += 1

    logger.info(
        "Leiden noise S picks: %d total, %d boost ineffective, "
        "%d no-P-at-station [%s], %d network.station mismatch, %d TT > max_search_dist.",
        len(noise_s), with_boost,
        no_p, ", ".join(no_p_names),
        mismatch, tt_too_large,
    )


# ---------------------------------------------------------------------------
# Stability computation
# ---------------------------------------------------------------------------

def _compute_stabilities(
    valid_communities: list[np.ndarray],
    rows: np.ndarray,
    cols: np.ndarray,
    weights_no_boost: np.ndarray,
) -> list[float]:
    """Internal weighted edge density per community (stability proxy).

    Uses original (non-boosted) weights so that P-S same-station edges do not
    inflate the score.  Ranges in [0, 1].
    """
    stabilities = []
    for members in valid_communities:
        n_m = len(members)
        in_comm = np.isin(rows, members) & np.isin(cols, members)
        max_possible = n_m * (n_m - 1) // 2
        w_internal = float(np.sum(weights_no_boost[in_comm]))
        stabilities.append(w_internal / max_possible)
    return stabilities


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def leiden_cluster(
    phases,
    pseudo_tt: np.ndarray,
    max_search_dist: float,
    min_cluster_size: int,
    resolution: float = 0.05,
    edge_weight_scale: float = None,
    ps_boost_factor: float = 100.0,
    min_edge_weight: float = 0.0,
    vp: float = 6.0,   # P-wave velocity (km/s) for moveout compatibility filter
    vs: float = 3.5,   # S-wave velocity (km/s) for moveout compatibility filter
    ps_dt_max: float = 0.0,  # max S-P delay (s) for same-station boost; 0 = disabled
    sigma_km: float = 0.0,   # spatial decay (km) in factored weight; 0 = disabled
) -> tuple:
    """Cluster picks using Leiden community detection on the TT graph.

    Parameters
    ----------
    phases : list[Phase]
        Input picks (Phase objects with .proba, .time, .coord attributes).
    pseudo_tt : np.ndarray (n, n)
        Precomputed travel-time distance matrix (seconds).
    max_search_dist : float
        Edge threshold (s): pairs with d_tt >= max_search_dist have no edge.
        Equivalent to HDBSCAN cluster_selection_epsilon.
    min_cluster_size : int
        Communities smaller than this are noise (label -1).
    resolution : float
        CPM resolution parameter γ.  Higher values → more, smaller communities.
        Typical range: 0.01 – 0.2.  Default 0.05.
    edge_weight_scale : float or None
        σ in exp(-d / σ) edge weight.  None → max_search_dist / 2, so that
        at d = max_search_dist the weight decays to exp(-2) ≈ 0.14.
    ps_boost_factor : float
        Multiplicative boost applied to same-station P-S edge weights.
        Must be >> 1/resolution to guarantee the pair stays in the same
        community.  Default 100.0 (safe for resolution in 0.01–0.2).
    min_edge_weight : float
        Edges with weight below this threshold are removed after boost.
        Boosted P-S same-station edges are always kept.
        0.0 (default) disables the filter.  A value around
        exp(-max_search_dist/sigma)*p_min**2 cuts the weakest cross-station
        edges and prevents marginal picks from being pulled into a community.

    Returns
    -------
    clusters : list[list[Phase]]
        Clustered picks, sorted by community index.
    stabilities : list[float]
        Mean internal edge weight per community (proxy for persistence).
        Uses original weights exp(-d/σ)*p_i*p_j, not the PS-boosted ones.
    noise : list[Phase]
        Picks not assigned to any valid community (community size < min_cluster_size
        or isolated nodes).
    """
    logger.info(
        "leiden_cluster called: ps_boost_factor=%.1f, min_edge_weight=%.4f",
        ps_boost_factor, min_edge_weight,
    )
    try:
        import igraph as ig
        import leidenalg
    except ImportError as exc:
        raise ImportError(
            "leidenalg and python-igraph are required for Leiden clustering. "
            "Install with: uv pip install leidenalg python-igraph"
        ) from exc

    n = len(phases)
    if n == 0:
        return [], [], []

    sigma = (max_search_dist / 2.0) if edge_weight_scale is None else edge_weight_scale
    sigma = max(sigma, 1e-6)

    probas = np.clip(np.array([p.proba for p in phases], dtype=float), 0.01, 1.0)
    stations = np.array([f"{p.network}.{p.station}" for p in phases])
    is_p = np.array([p.is_p() for p in phases])
    is_s = np.array([p.is_s() for p in phases])
    times = np.array([float(p.time) for p in phases])

    # --- Compute inter-station distances for moveout compatibility filter ---
    # Disabled when vp=0 or vs=0.
    dist_km_matrix = None
    if vp > 0 and vs > 0:
        R = 6371.0
        lats = np.radians([p.coord["latitude"] for p in phases])
        lons = np.radians([p.coord["longitude"] for p in phases])
        dlat = lats[:, None] - lats[None, :]
        dlon = lons[:, None] - lons[None, :]
        a = np.sin(dlat / 2) ** 2 + np.cos(lats[:, None]) * np.cos(lats[None, :]) * np.sin(dlon / 2) ** 2
        dist_km_matrix = 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

    # --- Build graph ---
    rows, cols, weights, weights_no_boost, ps_same = _build_edges(
        pseudo_tt, probas, stations, is_p, is_s, times, max_search_dist, sigma, resolution,
        ps_boost_factor=ps_boost_factor,
        min_edge_weight=min_edge_weight,
        dist_km=dist_km_matrix,
        vp=vp,
        vs=vs,
        ps_dt_max=ps_dt_max,
        sigma_km=sigma_km,
    )
    if len(rows) == 0:
        logger.info("Leiden: no edges within max_search_dist — all picks are noise.")
        return [], [], list(phases)

    G = ig.Graph(n=n, edges=list(zip(rows.tolist(), cols.tolist())))
    G.es["weight"] = weights.tolist()

    # --- Partition ---
    partition = leidenalg.find_partition(
        G,
        leidenalg.CPMVertexPartition,
        weights="weight",
        resolution_parameter=resolution,
        seed=42,
    )

    # --- Enforce P-S cohesion ---
    merged_labels = _merge_split_ps_pairs(
        np.array(partition.membership), rows, cols, ps_same
    )

    # --- Assign community labels, discard small communities as noise ---
    comm_members: dict[int, list[int]] = defaultdict(list)
    for node, lbl in enumerate(merged_labels):
        comm_members[lbl].append(node)

    labels = np.full(n, -1, dtype=int)
    valid_communities: list[np.ndarray] = []
    community_idx = 0
    for members_list in comm_members.values():
        if len(members_list) >= min_cluster_size:
            for m in members_list:
                labels[m] = community_idx
            valid_communities.append(np.array(members_list))
            community_idx += 1

    # --- Diagnostics and output ---
    _log_noise_s_diagnostics(
        phases, labels, is_s, is_p, stations, pseudo_tt, ps_same, rows, cols,
        max_search_dist,
    )

    stabilities = _compute_stabilities(valid_communities, rows, cols, weights_no_boost)

    label_to_cluster: dict[int, list] = {}
    noise: list = []
    for p, lbl in zip(phases, labels):
        if lbl == -1:
            noise.append(p)
        else:
            label_to_cluster.setdefault(int(lbl), []).append(p)

    clusters = [label_to_cluster[lbl] for lbl in sorted(label_to_cluster)]

    logger.info(
        "Leiden: %d communities, %d noise points (resolution=%.3f, sigma=%.1fs).",
        len(clusters), len(noise), resolution, sigma,
    )
    return clusters, stabilities, noise


def find_mega_cluster(
    clusters: list, n_total: int,
    threshold: float = 0.8, min_size: int = 150,
) -> int:
    """Return index of the cluster absorbing > threshold of all picks, or -1.

    Only triggers when the cluster also exceeds min_size picks, to avoid
    re-clustering small pools from PyOcto sub-clustering.
    """
    if n_total == 0:
        return -1
    for i, c in enumerate(clusters):
        if len(c) >= min_size and len(c) / n_total > threshold:
            return i
    return -1


def recluster_mega_with_leiden(
    mega_idx: int,
    clusters: list, clusters_stability: list, noise: list,
    phases: list, pseudo_tt, max_search_dist: float,
    min_cluster_size: int,
    leiden_resolution: float, leiden_edge_weight_scale,
    leiden_ps_boost_factor: float, leiden_min_edge_weight: float,
) -> tuple[list, list, list]:
    """Replace a mega-cluster with Leiden sub-communities.

    The mega-cluster picks are re-clustered with Leiden (which handles
    geographically mixed pools better via PS-boost).  The resulting
    communities replace the mega-cluster in the cluster list; Leiden
    noise picks are added to the global noise pool.
    """
    import numpy as np

    mega_picks = list(clusters[mega_idx])
    phase_to_idx = {id(p): i for i, p in enumerate(phases)}
    mega_indices = [phase_to_idx[id(p)] for p in mega_picks if id(p) in phase_to_idx]
    sub_tt = pseudo_tt[np.ix_(mega_indices, mega_indices)]

    leiden_clusters, leiden_stab, leiden_noise = leiden_cluster(
        mega_picks, sub_tt, max_search_dist, min_cluster_size,
        resolution=leiden_resolution,
        edge_weight_scale=leiden_edge_weight_scale,
        ps_boost_factor=leiden_ps_boost_factor,
        min_edge_weight=leiden_min_edge_weight,
    )

    logger.info(
        "Mega-cluster (%d picks) re-clustered with Leiden: %d communities, %d noise.",
        len(mega_picks), len(leiden_clusters), len(leiden_noise),
    )

    new_clusters = [c for i, c in enumerate(clusters) if i != mega_idx]
    new_stability = [s for i, s in enumerate(clusters_stability) if i != mega_idx]
    new_clusters.extend(leiden_clusters)
    new_stability.extend(leiden_stab)

    return new_clusters, new_stability, noise + leiden_noise
