#!/usr/bin/env python
import sys
import os
import argparse
import logging
import pandas as pd
from datetime import timedelta
from sklearn.cluster import DBSCAN


def unload_too_close_picks_clustering(
    csv_file_in, csv_file_out, P_delta_time, S_delta_time
):
    print(f"Reading from {csv_file_out}.")
    df = pd.read_csv(csv_file_in)
    df.rename(
        columns={
            "seedid": "station_id",
            "phasename": "phase_type",
            "time": "phase_time",
            "probability": "phase_score",
        },
        inplace=True,
    )
    # Keeps only network_code.station_code
    df["station_id"] = df["station_id"].map(lambda x: ".".join(x.split(".")[:2]))
    df["phase_time"] = pd.to_datetime(df["phase_time"], utc=True)

    df = df.sort_values(by=["station_id", "phase_type", "phase_time"])

    results = pd.DataFrame(
        columns=["station_id", "phase_type", "phase_time", "phase_score"]
    )
    # run separately by phase type
    for phase in ("P", "S"):
        print(f"Working on {phase}.")

        # dbscan configuration
        if "P" in phase:
            max_distance = P_delta_time  # secondes
        else:
            max_distance = S_delta_time  # secondes

        min_samples = 1
        dbscan = DBSCAN(eps=max_distance, min_samples=min_samples, metric="euclidean")

        phase_df = df[df["phase_type"].str.contains(phase)].copy()
        phase_df["phase_time"] = pd.to_datetime(phase_df["phase_time"], utc=True)
        # get min time from df
        min_timestamp = phase_df["phase_time"].min()
        # create numeric_time for time distance computation
        phase_df["numeric_time"] = (
            phase_df["phase_time"] - pd.to_datetime(min_timestamp, utc=True)
        ).dt.total_seconds()

        # loop over station_id
        for station_id in phase_df["station_id"].drop_duplicates():
            print(f"Working on {phase}/{station_id}")
            tmp_df = phase_df.loc[phase_df["station_id"] == station_id].copy()
            before = len(tmp_df)
            # clusterize by station_id
            tmp_df["cluster"] = dbscan.fit_predict(tmp_df[["numeric_time"]])
            # keeps only the pick with the higher score
            idx = tmp_df.groupby(["cluster"])["phase_score"].idxmax()
            tmp_df = tmp_df.loc[idx]
            tmp_df = tmp_df.drop(columns=["numeric_time", "cluster"])
            after = len(tmp_df)
            print(f"length before: {before}, after: {after}")
            if len(results):
                results = pd.concat([results, tmp_df], ignore_index=True)
            else:
                results = tmp_df.copy()

    results["phase_time"] = pd.to_datetime(results["phase_time"]).dt.strftime(
        "%Y-%m-%dT%H:%M:%S.%fZ"
    )
    print(results)
    print(f"Writing to {csv_file_out}.")
    results.to_csv(csv_file_out, index=False)


if __name__ == "__main__":
    # default logger
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    logger = logging.getLogger("dbclust")
    logger.setLevel(logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input",
        default=None,
        dest="inputfile",
        help="qml input file",
        type=str,
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        dest="outputfile",
        help="output file",
        type=str,
    )
    parser.add_argument(
        "--delta-p",
        default=0.1,
        dest="P_delta_time",
        help="P phase delta time",
        type=float,
    )
    parser.add_argument(
        "--delta-s",
        default=0.2,
        dest="S_delta_time",
        help="S phase delta time",
        type=float,
    )
    parser.add_argument(
        "-l",
        "--loglevel",
        default="INFO",
        dest="loglevel",
        help="loglevel (debug,warning,info,error)",
        type=str,
    )
    args = parser.parse_args()

    if not args.inputfile or not args.outputfile:
        parser.print_help()
        sys.exit(255)

    if os.path.isfile(args.outputfile):
        logger.error(f"outputfile {args.outputfile} already exists !")
        sys.exit(255)

    numeric_level = getattr(logging, args.loglevel.upper(), None)
    if not numeric_level:
        logger.error("Invalid loglevel '%s' !", args.loglevel.upper())
        logger.error("loglevel should be: debug,warning,info,error.")
        sys.exit(255)
    logger.setLevel(numeric_level)

    unload_too_close_picks_clustering(
        args.inputfile, args.outputfile, args.P_delta_time, args.S_delta_time
    )
