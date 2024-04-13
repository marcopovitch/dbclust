#!/usr/bin/env python
import argparse
import logging
import os
import sys

import dask
dask.config.set({'dataframe.query-planning-warning': False})
dask.config.set({'dataframe.query-planning': False})
import dask.dataframe as dd

from obspy import UTCDateTime
from icecream import ic

def convert_to_utc_datetime(series):
    return series.apply(lambda x: UTCDateTime(x))


if __name__ == "__main__":
    # default logger
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    logger = logging.getLogger("phasenet2dbclust")
    logger.setLevel(logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input",
        default=None,
        dest="inputfile",
        help="input file",
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
    args = parser.parse_args()

    if not args.inputfile or not args.outputfile:
        parser.print_help()
        sys.exit(255)

    if os.path.isfile(args.outputfile):
        logger.error(f"outputfile {args.outputfile} already exists !")
        sys.exit(255)

    logger.info(f"Reading {args.inputfile} ...")
    df = dd.read_csv(args.inputfile)

    df = df.repartition(npartitions=os.cpu_count())

    logger.info("Starting processing ...")
    df = df.rename(
        columns={
            "seedid": "station_id",
            "phasename": "phase_type",
            "time": "phase_time",
            "probability": "phase_score",
        },
    )
    # needed to have the right time format when exporting to csv
    df["phase_time"] = df.map_partitions(
        lambda partition: convert_to_utc_datetime(partition["phase_time"]),
        meta=("phase_time", object),
    )

    df = df.sort_values(by=["phase_time"], npartitions=6)

    if "phase_index" in df.columns:
        # seems to be fake picks
        df = df[df["phase_index"] != 1]

    df = df.drop(
        columns=["begin_time", "phase_index", "file_name"], errors="ignore", axis=1
    )

    df["phase_evaluation"] = "automatic"
    df["phase_method"] = "PHASENET"
    df["event_id"] = ""
    df["agency"] = "RENASS"
    df["channel"] = ""

    col_order = [
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

    df.to_csv(args.outputfile, single_file=True, index=False, columns=col_order)
