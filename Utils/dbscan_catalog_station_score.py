#!/usr/bin/env python
import argparse
import sys
from os.path import exists

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN


EARTH_RADIUS_KM = 6371.0088


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Filter a seismicity catalog on station_score, run DBSCAN using a spatial distance in km, "
            "then keep only clusters above a size threshold and write an output CSV with CLUSTER_ID/CLUSTER_SIZE."
        )
    )
    parser.add_argument("-i", "--input-file", required=True, type=str, help="Input catalog CSV")
    parser.add_argument("-o", "--output-file", required=True, type=str, help="Output CSV")

    parser.add_argument(
        "--score-threshold",
        required=True,
        type=float,
        help="Keep only events with station_score > this threshold",
    )
    parser.add_argument(
        "--eps-km",
        required=True,
        type=float,
        help="DBSCAN epsilon (maximum neighborhood distance) in kilometers",
    )
    parser.add_argument(
        "--min-samples",
        default=2,
        type=int,
        help="DBSCAN min_samples (default: 2)",
    )
    parser.add_argument(
        "--min-cluster-size",
        default=5,
        type=int,
        help="Keep only clusters with at least this many events (default: 5)",
    )

    parser.add_argument(
        "--lat-column",
        default="latitude",
        type=str,
        help="Latitude column name (default: latitude)",
    )
    parser.add_argument(
        "--lon-column",
        default="longitude",
        type=str,
        help="Longitude column name (default: longitude)",
    )
    parser.add_argument(
        "--score-column",
        default="station_score",
        type=str,
        help="Station score column name (default: station_score)",
    )

    return parser.parse_args()


def _require_columns(df: pd.DataFrame, columns):
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _dbscan_haversine_km(lat_deg: np.ndarray, lon_deg: np.ndarray, eps_km: float, min_samples: int) -> np.ndarray:
    coords_rad = np.radians(np.column_stack([lat_deg, lon_deg]))
    eps_rad = float(eps_km) / EARTH_RADIUS_KM
    model = DBSCAN(eps=eps_rad, min_samples=int(min_samples), metric="haversine")
    labels = model.fit_predict(coords_rad)
    return labels


def main():
    args = parse_arguments()

    if exists(args.output_file):
        print(f"Error: output file {args.output_file} already exists.")
        sys.exit(1)

    df = pd.read_csv(args.input_file)

    try:
        _require_columns(df, [args.lat_column, args.lon_column, args.score_column])
    except ValueError as exc:
        print(f"Error: {exc}")
        sys.exit(1)

    df = df.copy()

    df = df.dropna(subset=[args.lat_column, args.lon_column, args.score_column])
    before_score = len(df)
    df = df[df[args.score_column] >= float(args.score_threshold)].copy()
    after_score = len(df)
    print(f"Filtering on {args.score_column} >= {args.score_threshold}: kept {after_score}/{before_score} events")

    if len(df) == 0:
        print("No events left after score filtering.")
        sys.exit(1)

    labels = _dbscan_haversine_km(
        lat_deg=df[args.lat_column].to_numpy(dtype=float),
        lon_deg=df[args.lon_column].to_numpy(dtype=float),
        eps_km=float(args.eps_km),
        min_samples=int(args.min_samples),
    )

    df["_DBSCAN_LABEL"] = labels

    labeled = df[df["_DBSCAN_LABEL"] >= 0].copy()
    if len(labeled) == 0:
        print("DBSCAN produced no clusters (all points are noise).")
        sys.exit(1)

    cluster_sizes = labeled["_DBSCAN_LABEL"].value_counts().sort_values(ascending=False)

    print("Clusters (dbscan_label -> n_events):")
    for cluster_label, size in cluster_sizes.items():
        print(f"  {int(cluster_label)} -> {int(size)}")

    keep_labels = cluster_sizes[cluster_sizes >= int(args.min_cluster_size)].index.to_list()
    print(f"Keeping clusters with size >= {int(args.min_cluster_size)}: {len(keep_labels)} clusters")

    kept = labeled[labeled["_DBSCAN_LABEL"].isin(keep_labels)].copy()
    if len(kept) == 0:
        print("No events left after cluster size filtering.")
        sys.exit(1)

    kept_cluster_sizes = kept["_DBSCAN_LABEL"].value_counts().to_dict()

    sorted_keep_labels = (
        cluster_sizes[cluster_sizes.index.isin(keep_labels)]
        .sort_values(ascending=False)
        .index.to_list()
    )
    relabel_map = {int(old): int(new) for new, old in enumerate(sorted_keep_labels)}

    kept["CLUSTER_ID"] = kept["_DBSCAN_LABEL"].map(lambda x: relabel_map[int(x)]).astype(int)
    kept["CLUSTER_SIZE"] = kept["_DBSCAN_LABEL"].map(lambda x: int(kept_cluster_sizes[int(x)])).astype(int)

    kept = kept.drop(columns=["_DBSCAN_LABEL"])

    kept.to_csv(args.output_file, index=False)
    print(f"Wrote {len(kept)} events to {args.output_file}")


if __name__ == "__main__":
    main()
