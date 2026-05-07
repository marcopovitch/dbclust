#!/usr/bin/env python
"""
compare_metrics.py — Compare HDBSCAN clustering results between
numpy_compute_tt_matrix_vectorized and numpy_compute_tt_matrix_vectorized_physical
using real picks from DuckDB.

Usage:
    python compare_metrics.py --config /path/to/config.yml \
                               --start "2014-07-09T22:00:00" \
                               --end   "2014-07-09T22:15:00"
"""

import argparse
import logging
import sys
from copy import deepcopy

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- dbclust imports ---
from dbclust.config import DBClustConfig
from dbclust.db import duckdb_init
from dbclust.phase import import_phases
from dbclust.preprocessing_picks import deduplicate_picks_by_time
from dbclust.clusterize import Clusterize

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("compare_metrics")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PALETTE = [
    "#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00",
    "#a65628", "#f781bf", "#999999", "#66c2a5", "#fc8d62",
    "#8da0cb", "#e78ac3", "#a6d854", "#ffd92f", "#e5c494",
    "#b3b3b3", "#1b9e77", "#d95f02", "#7570b3", "#e7298a",
]

NOISE_COLOR = "lightgrey"


def label_color(label: int) -> str:
    if label == -1:
        return NOISE_COLOR
    return PALETTE[label % len(PALETTE)]


def fetch_phases(cfg: DBClustConfig, t_start: str, t_end: str):
    """Fetch and pre-process phases from DuckDB for the given time window."""
    con = duckdb_init(cfg.pick.filenames, cfg.pick.type)

    rqt = f"""
        SELECT * FROM PICKS
        WHERE phase_time BETWEEN '{t_start}' AND '{t_end}'
        AND phase_type IN ('P', 'Pg', 'Pn', 'S', 'Sg', 'Sn')
    """
    df = con.sql(rqt).fetchdf()
    con.close()

    if df.empty:
        logger.error("No picks found in the requested time window.")
        sys.exit(1)

    # Timezone
    if df["phase_time"].dt.tz is None:
        df["phase_time"] = df["phase_time"].dt.tz_localize("UTC")
    else:
        df["phase_time"] = df["phase_time"].dt.tz_convert("UTC")

    # Blacklist
    if cfg.station.blacklist:
        for b in cfg.station.blacklist:
            df = df[~df["station_id"].str.contains(b, regex=True)]

    # Deduplication
    df = deduplicate_picks_by_time(
        df,
        cfg.pick.P_proximity_threshold,
        cfg.pick.S_proximity_threshold,
    )

    phases = import_phases(
        df,
        cfg.pick.P_proba_threshold,
        cfg.pick.S_proba_threshold,
        cfg.pick.P_uncertainty,
        cfg.pick.S_uncertainty,
        cfg.station.info_sta,
        cfg.station.fallback_df,
    )
    logger.info(f"Loaded {len(phases)} phases.")
    return phases


def run_clustering(phases, matrix_fn, cfg: DBClustConfig):
    """Build distance matrix with matrix_fn and run HDBSCAN."""
    vp = cfg.cluster.average_velocity
    min_cluster_size = cfg.cluster.min_cluster_size
    max_search_dist = cfg.cluster.max_search_dist or 0

    pseudo_tt = matrix_fn(phases, vp)
    clusters, stability, noise = Clusterize.get_clusters(
        phases, pseudo_tt, max_search_dist, min_cluster_size
    )

    # Build label array (same order as phases)
    labels = np.full(len(phases), -1, dtype=int)
    for c_id, cluster in enumerate(clusters):
        for p in cluster:
            idx = phases.index(p)
            labels[idx] = c_id

    return clusters, noise, labels, pseudo_tt


def build_scatter_trace(phases, labels, name_suffix):
    """Build one scatter trace per cluster + one for noise."""
    lons = [p.coord["longitude"] for p in phases]
    lats = [p.coord["latitude"] for p in phases]
    times = [str(p.time)[:19] for p in phases]
    phase_types = [p.phase for p in phases]
    stations = [f"{p.network}.{p.station}" for p in phases]
    event_ids = [p.event_id.split("/")[-1] if p.event_id else "" for p in phases]
    colors = [label_color(l) for l in labels]

    max_label = max(labels) if len(labels) else -1
    n_clusters = max_label + 1
    n_noise = int(np.sum(labels == -1))

    traces = []
    # One trace per cluster for legend
    for c_id in range(-1, n_clusters):
        mask = np.array(labels) == c_id
        if not np.any(mask):
            continue
        cidx = np.where(mask)[0]
        clabel = "noise" if c_id == -1 else f"C{c_id}"
        color = NOISE_COLOR if c_id == -1 else PALETTE[c_id % len(PALETTE)]
        symbol = "circle-open" if c_id == -1 else "circle"
        custom = [
            f"station={stations[i]}<br>phase={phase_types[i]}<br>"
            f"time={times[i]}<br>event_id={event_ids[i]}<br>cluster={clabel}"
            for i in cidx
        ]
        traces.append(
            go.Scattergeo(
                lon=[lons[i] for i in cidx],
                lat=[lats[i] for i in cidx],
                mode="markers",
                marker=dict(
                    size=6,
                    color=color,
                    symbol=symbol,
                    opacity=0.8 if c_id != -1 else 0.4,
                ),
                name=f"{name_suffix} {clabel}",
                legendgroup=f"{name_suffix}_{clabel}",
                hovertemplate="%{customdata}<extra></extra>",
                customdata=custom,
            )
        )

    return traces, n_clusters, n_noise


def build_heatmap_trace(pseudo_tt, title):
    """Heatmap of the distance matrix (capped for readability)."""
    cap = float(np.percentile(pseudo_tt, 99)) or 1.0
    return go.Heatmap(
        z=np.clip(pseudo_tt, 0, cap),
        colorscale="Viridis",
        reversescale=False,
        showscale=True,
        colorbar=dict(title=title, len=0.45),
        zmin=0,
        zmax=cap,
    )


def build_time_scatter(phases, labels, name_suffix):
    """Scatter of pick time vs. latitude, coloured by cluster."""
    times = [float(p.time) for p in phases]
    t0 = min(times)
    times_rel = [t - t0 for t in times]
    lats = [p.coord["latitude"] for p in phases]
    phase_types = [p.phase for p in phases]
    stations = [f"{p.network}.{p.station}" for p in phases]

    traces = []
    max_label = max(labels) if len(labels) else -1
    n_clusters = max_label + 1

    for c_id in range(-1, n_clusters):
        mask = np.array(labels) == c_id
        if not np.any(mask):
            continue
        cidx = np.where(mask)[0]
        clabel = "noise" if c_id == -1 else f"C{c_id}"
        color = NOISE_COLOR if c_id == -1 else PALETTE[c_id % len(PALETTE)]
        symbol = "circle-open" if c_id == -1 else "circle"
        custom = [
            f"station={stations[i]}<br>phase={phase_types[i]}<br>t+{times_rel[i]:.1f}s"
            for i in cidx
        ]
        traces.append(
            go.Scatter(
                x=[times_rel[i] for i in cidx],
                y=[lats[i] for i in cidx],
                mode="markers",
                marker=dict(size=5, color=color, symbol=symbol, opacity=0.7),
                name=f"{name_suffix} {clabel}",
                legendgroup=f"{name_suffix}_{clabel}",
                showlegend=False,
                hovertemplate="%{customdata}<extra></extra>",
                customdata=custom,
            )
        )
    return traces


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Compare HDBSCAN clustering metrics")
    parser.add_argument("--config", required=True, help="Path to dbclust YAML config")
    parser.add_argument("--start", required=True, help="Start time ISO8601")
    parser.add_argument("--end",   required=True, help="End time ISO8601")
    parser.add_argument("--output", default="compare_metrics.html",
                        help="Output HTML file (default: compare_metrics.html)")
    args = parser.parse_args()

    # --- Config ---
    cfg = DBClustConfig(args.config)

    # --- Picks ---
    print(f"Fetching picks [{args.start} .. {args.end}] ...", flush=True)
    phases = fetch_phases(cfg, args.start, args.end)
    print(f"  {len(phases)} phases loaded.")

    if len(phases) < cfg.cluster.min_cluster_size:
        print("Not enough phases to cluster.", file=sys.stderr)
        sys.exit(1)

    vp = cfg.cluster.average_velocity

    # --- Matrix A: vectorized (classic) ---
    print("Computing distance matrix A (vectorized) ...", flush=True)
    mat_A = Clusterize.numpy_compute_tt_matrix_vectorized(phases, vp)

    # --- Matrix B: physical ---
    print("Computing distance matrix B (physical) ...", flush=True)
    mat_B = Clusterize.numpy_compute_tt_matrix_vectorized_physical(phases, vp)

    # --- Cluster ---
    min_cs = cfg.cluster.min_cluster_size
    max_sd = cfg.cluster.max_search_dist or 0

    print("Clustering A ...", flush=True)
    clusters_A, noise_A, labels_A, _ = run_clustering(phases, Clusterize.numpy_compute_tt_matrix_vectorized, cfg)

    print("Clustering B ...", flush=True)
    clusters_B, noise_B, labels_B, _ = run_clustering(phases, Clusterize.numpy_compute_tt_matrix_vectorized_physical, cfg)

    nA, nNoiseA = len(clusters_A), int(np.sum(labels_A == -1))
    nB, nNoiseB = len(clusters_B), int(np.sum(labels_B == -1))
    print(f"  A: {nA} clusters, {nNoiseA} noise points")
    print(f"  B: {nB} clusters, {nNoiseB} noise points")

    # -----------------------------------------------------------------------
    # Layout: 3 rows
    #   Row 1: geo map A | geo map B
    #   Row 2: time-lat A | time-lat B
    #   Row 3: heatmap A  | heatmap B
    # -----------------------------------------------------------------------
    fig = make_subplots(
        rows=3, cols=2,
        specs=[
            [{"type": "geo"}, {"type": "geo"}],
            [{"type": "xy"}, {"type": "xy"}],
            [{"type": "xy"}, {"type": "xy"}],
        ],
        subplot_titles=[
            f"A - vectorized  ({nA} clusters, {nNoiseA} noise)",
            f"B - physical    ({nB} clusters, {nNoiseB} noise)",
            "A - time vs latitude",
            "B - time vs latitude",
            "A - distance matrix (capped p99)",
            "B - distance matrix (capped p99)",
        ],
        vertical_spacing=0.08,
        horizontal_spacing=0.05,
    )

    # --- Row 1: geo maps ---
    lons_all = [p.coord["longitude"] for p in phases]
    lats_all = [p.coord["latitude"] for p in phases]
    lon_c = (min(lons_all) + max(lons_all)) / 2
    lat_c = (min(lats_all) + max(lats_all)) / 2

    traces_A, _, _ = build_scatter_trace(phases, labels_A, "A")
    traces_B, _, _ = build_scatter_trace(phases, labels_B, "B")

    for t in traces_A:
        fig.add_trace(t, row=1, col=1)
    for t in traces_B:
        fig.add_trace(t, row=1, col=2)

    geo_common = dict(
        scope="europe",
        showland=True, landcolor="rgb(240,240,240)",
        showcoastlines=True, coastlinecolor="rgb(150,150,150)",
        showframe=True,
        projection_type="mercator",
        center=dict(lon=lon_c, lat=lat_c),
        lataxis_range=[min(lats_all) - 1, max(lats_all) + 1],
        lonaxis_range=[min(lons_all) - 1, max(lons_all) + 1],
    )
    fig.update_geos(geo_common, row=1, col=1)
    fig.update_geos(geo_common, row=1, col=2)

    # --- Row 2: time-lat ---
    for t in build_time_scatter(phases, labels_A, "A"):
        fig.add_trace(t, row=2, col=1)
    for t in build_time_scatter(phases, labels_B, "B"):
        fig.add_trace(t, row=2, col=2)

    fig.update_xaxes(title_text="time relative to first pick (s)", row=2, col=1)
    fig.update_xaxes(title_text="time relative to first pick (s)", row=2, col=2)
    fig.update_yaxes(title_text="latitude", row=2, col=1)
    fig.update_yaxes(title_text="latitude", row=2, col=2)

    # --- Row 3: heatmaps ---
    fig.add_trace(build_heatmap_trace(mat_A, "dist A"), row=3, col=1)
    fig.add_trace(build_heatmap_trace(mat_B, "dist B"), row=3, col=2)

    fig.update_xaxes(title_text="pick index", row=3, col=1)
    fig.update_xaxes(title_text="pick index", row=3, col=2)
    fig.update_yaxes(title_text="pick index", row=3, col=1)
    fig.update_yaxes(title_text="pick index", row=3, col=2)

    # --- Matrix stats ---
    diff = mat_B - mat_A
    stats = (
        f"n_phases={len(phases)}  |  "
        f"vP={vp} km/s  |  "
        f"min_cluster_size={min_cs}  |  "
        f"Δmatrix: mean={diff.mean():.3f}  std={diff.std():.3f}  "
        f"max_abs={np.abs(diff).max():.2f}"
    )

    fig.update_layout(
        title=dict(
            text=(
                f"<b>Metric comparison</b>  [{args.start} → {args.end}]<br>"
                f"<sup>{stats}</sup>"
            ),
            x=0.5,
        ),
        height=1600,
        legend=dict(tracegroupgap=2, font_size=10),
        margin=dict(l=30, r=30, t=100, b=30),
    )

    fig.write_html(args.output)
    print(f"Saved → {args.output}")


if __name__ == "__main__":
    main()
