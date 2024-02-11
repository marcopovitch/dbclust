#!/usr/bin/env python
import argparse
import logging
import sys
from math import pow
from math import sqrt
from os.path import exists

import numpy as np
import pandas as pd
from obspy.geodetics import gps2dist_azimuth
from sklearn.cluster import DBSCAN

# default logger
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger("match")
logger.setLevel(logging.INFO)


class Point(object):
    def __init__(
        self,
        lat=None,
        lon=None,
        event_id=None,
        T0=None,
    ):
        self.lat = lat
        self.lon = lon
        self.event_id = event_id
        self.T0 = T0


def compute_pseudo_dist(p1, p2, vmean):
    distance, az, baz = gps2dist_azimuth(p1.lat, p2.lon, p2.lat, p2.lon)
    # distance in meters, convert it to km
    distance = distance / 1000.0
    dd = distance / vmean
    dt = p1.T0 - p2.T0
    pseudo_dist = sqrt(pow(dt.total_seconds(), 2) + pow(dd, 2))
    return pseudo_dist


def numpy_compute_pseudo_dist_matrix(points, vmean):
    # optimization : matrix is diagonal and symmetrical
    # computes only the upper part, and copy it to the lower part
    nb_points = len(points)
    elements = []
    for i in range(0, nb_points):
        p1 = points[i]
        for j in range(i, nb_points):
            p2 = points[j]
            elements.append(compute_pseudo_dist(p1, p2, vmean))

    matrix_upper1 = np.zeros((nb_points, nb_points))
    row, col = np.triu_indices(nb_points)
    matrix_upper1[row, col] = elements

    matrix_upper2 = np.copy(matrix_upper1)
    np.fill_diagonal(matrix_upper2, 0)
    matrix = matrix_upper1 + matrix_upper2.T
    return matrix


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input-file",
        action='append',
        nargs='?',
        default=None,
        dest="input_files",
        help="csv catalog files (multiple csv files is allowed)",
        type=str,
    )
    parser.add_argument(
        "-o",
        "--output-file",
        default=None,
        dest="output_file",
        help="matched events csv file",
        type=str,
    )
    parser.add_argument(
        "--mean-velocity",
        default=5,
        dest="vmean",
        help="mean velocity",
        type=float,
    )
    parser.add_argument(
        "-d",
        "--dist-max",
        default=5,
        dest="dist_max",
        help="max distance (km) allowed between event locations",
        type=float,
    )
    parser.add_argument(
        "-t",
        "--time-max",
        default=2,
        dest="dt_max",
        help="max time differences allowed between origin times",
        type=float,
    )
    args = parser.parse_args()

    if not args.input_files or not args.output_file:
        parser.print_help()
        sys.exit(255)

    if args.output_file and exists(args.output_file):
        print(f"File {args.output_file} already exists !")
        sys.exit(255)

    logger.info(f"Config: vmean={args.vmean}, dist_max={args.dist_max}, dt_max={args.dt_max}")
    max_distance = sqrt(pow(args.dist_max / args.vmean, 2) + pow(args.dt_max, 2))

    columns_of_interest = ["event_id", "time", "latitude", "longitude"]
    df = pd.DataFrame(columns=columns_of_interest)
    #df["time"] = pd.to_datetime(df["time"], utc=True)

    for f in args.input_files:
        logger.info(f"Reading {f} ...")
        tmp_df = pd.read_csv(f, usecols=columns_of_interest)
        if not len(df):
            df = tmp_df
        else:
            df = pd.concat([df, tmp_df], ignore_index=True)

    df.sort_values(by="time", inplace=True)

    points = []
    for index, row in df.iterrows():
        lat = float(row["latitude"])
        lon = float(row["longitude"])
        event_id = row["event_id"]
        T0 = pd.to_datetime(row["time"], utc=True)
        point = Point(lat=lat, lon=lon, event_id=event_id, T0=T0)
        points.append(point)

    logger.info(f"Computing matrix ...")
    matrix = numpy_compute_pseudo_dist_matrix(points, args.vmean)

    # dbscan
    logger.info(f"Starting DBscan ...")
    db = dbscan = DBSCAN(
        eps=max_distance,
        min_samples=1,
        metric="precomputed",
        n_jobs=-1,
    ).fit(matrix)

    logger.info(f"{db}")
    df["cluster_id"] = db.labels_

    gdf = df.groupby(["cluster_id"], as_index=False).agg(
        {
            "time": "first",
            "latitude": "first",
            "longitude": "first",
            "event_id": ",".join,
        }
    )
    gdf["count"] = df.groupby(["cluster_id"], as_index=False)["cluster_id"].count()
    gdf.to_csv(args.output_file, index=False)
    logger.info(f"Writing to {args.output_file}")
