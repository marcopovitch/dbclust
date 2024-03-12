#!/usr/bin/env python
import argparse
import logging
import os
import sys
from datetime import timedelta

import pandas as pd
from icecream import ic
from pandas.core.groupby import GroupBy
from sklearn.cluster import DBSCAN


def get_index(group: GroupBy, debug=False) -> int:
    """Returns index of the max (only if one max value exists),
    else returns the index of the median picks (manual pick are privileged)

    Args:
        group (GroupBy): picks cluster related to one station and phase (P|S)

    Returns:
        int: _description_
    """
    max_value = group["phase_score"].max()
    max_value_indices = group[group["phase_score"] == max_value].index
    if len(max_value_indices) == 1:
        # only one absolute max
        max_index = max_value_indices[0]
        max_value = group.loc[max_index, "phase_score"]
        if debug:
            print(f"cluster nb picks {len(group)}, proba max {len(group)}: max_index={max_index} max_value={max_value}")
        return max_index
    else:
        # multiple picks with max proba set
        manual_df = group[group["phase_evaluation"] == "manual"]
        if len(manual_df) >= 1:
            # use only manual picks to get the preferred pick (based on median index)
            median_index = (
                manual_df["phase_score"].sort_values().index[len(manual_df) // 2]
            )
            median_value = manual_df.loc[median_index, "phase_score"]
            if debug:
                print(
                    f"cluster nb picks {len(group)}, median manual {len(manual_df)}: median_index={median_index} median_value={median_value}"
                )
            return median_index
        else:
            median_index = group["phase_score"].sort_values().index[len(manual_df) // 2]
            median_value = group.loc[median_index, "phase_score"]
            if debug:
                print(
                    f"cluster nb picks {len(group)}, median other {len(group)}: median_index={median_index} median_value={median_value}"
                )
            return median_index


def unload_too_close_picks_clustering(
    csv_file_in, csv_file_out, P_delta_time, S_delta_time
):

    col_types = {
        "station_id": "string",
        "channel": "string",
        "phase_type": "string",
        "phase_time": "string",
        "phase_score": "float64",
        "phase_evaluation": "string",
        "phase_method": "string",
        "event_id": "string",
        "agency": "string",
    }

    print(f"Reading from {csv_file_in}.")
    df = pd.read_csv(csv_file_in, dtype=col_types)

    # Use only network_code.station_code
    df["channel"] = df["station_id"].map(lambda x: ".".join(x.split(".")[2:4]))
    df["station_id"] = df["station_id"].map(lambda x: ".".join(x.split(".")[:2]))
    df["phase_time"] = pd.to_datetime(df["phase_time"], utc=True)

    if "phase_evaluation" not in df.columns:
        df["phase_evaluation"] = None

    df = df.sort_values(by=["station_id", "phase_type", "phase_time"])

    # empty dataframe
    results = pd.DataFrame(
        columns=[
            "station_id",
            "channel",
            "phase_type",
            "phase_time",
            "phase_score",
            "phase_evaluation",
            "phase_method",
            "event_id",
            "agency",
        ]
    )

    # run separately by phase type
    for phase in ("P", "S"):
        print(f"Working on {phase}.")
        phase_df = df[df["phase_type"].str.contains(phase)].copy()
        phase_df["phase_time"] = pd.to_datetime(phase_df["phase_time"], utc=True)

        # create numeric_time for time distance computation
        min_timestamp = phase_df["phase_time"].min()
        phase_df["numeric_time"] = (
            phase_df["phase_time"] - pd.to_datetime(min_timestamp, utc=True)
        ).dt.total_seconds()

        # dbscan configuration
        if "P" in phase:
            max_distance = P_delta_time  # secondes
        else:
            max_distance = S_delta_time  # secondes
        min_samples = 1
        dbscan = DBSCAN(eps=max_distance, min_samples=min_samples, metric="euclidean")

        # loop over station_id
        for station_id in phase_df["station_id"].drop_duplicates():
            print(f"Working on {phase}/{station_id}")
            # filter by station and sort by phase, as the order is kept by groupby
            # The order should be Pn, Pg, P.
            tmp_df = (
                phase_df.loc[phase_df["station_id"] == station_id]
                .sort_values(by="phase_type", ascending=False)
                .copy()
            )
            before = len(tmp_df)

            # clusterize by station_id
            tmp_df["cluster"] = dbscan.fit_predict(tmp_df[["numeric_time"]])

            # keeps only the pick with the higher score
            # idx_tmp = tmp_df.groupby(["cluster"])["phase_score"].idxmax()

            # keeps the pick with the higher score or the median one.
            # result = df.groupby('group').agg(index=('value', get_index))
            # idx = tmp_df.groupby(["cluster"])["phase_score", "phase_evaluation"].apply(get_index)
            # idx = tmp_df.groupby(["cluster"]).apply(lambda group: get_index(group[['phase_score', 'phase_evaluation']]))
            idx = tmp_df.groupby(["cluster"])[
                ["phase_score", "phase_evaluation"]
            ].apply(get_index)

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

    results.sort_values(by=["phase_time", "station_id"], inplace=True)
    print(f"Writing to {csv_file_out}.")
    results.to_csv(csv_file_out, index=False)


if __name__ == "__main__":
    # default logger
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    logger = logging.getLogger("pick_preproc")
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
