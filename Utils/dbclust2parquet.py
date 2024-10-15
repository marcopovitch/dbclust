#!/usr/bin/env python
import argparse
import concurrent.futures
import logging
import os
import sys

import dask
import pandas as pd

dask.config.set({"dataframe.query-planning-warning": False})
dask.config.set({"dataframe.query-planning": False})
import dask.dataframe as dd
import glob
from icecream import ic

from obspy import UTCDateTime


"""
Processes PhaseNet seismic phase data from CSV files and converts it into a standardized format.
The processed data is then saved to an output CSV file.

Usage:
    phasenet2dbclust.py -i <inputfile_pattern> -o <outputfile>

Arguments:
    -i, --input: Input file pattern (supports glob patterns) to read CSV files.
    -o, --output: Output file path to save the processed CSV.

Functions:
    convert_to_utc_datetime(series):
        Converts a Pandas Series of time strings to UTCDateTime objects.

For ultimate performance, use DucDB to repartition the parquet files:

    COPY (SELECT * FROM read_parquet('data.pq/**/*.parquet'))
        TO 'toto.pq'
        (FORMAT 'parquet', PARTITION_BY (year, month));
"""


def convert_to_utc_datetime(series):
    return series.apply(lambda x: UTCDateTime(x))


def read_and_modify(file: str) -> dd.DataFrame:
    """
    Read a CSV file and modify the columns to match the standardized format.
    Args:
        file (str): The CSV file to read.
    Returns:
        dd.DataFrame: The modified Dask DataFrame.
    """

    # Initialize an empty Dask DataFrame with specified columns
    df = dd.from_pandas(
        pd.DataFrame(
            columns=["seedid", "phasename", "time", "probability"],
        ),
        npartitions=1,
    )

    if not os.path.exists(file):
        logger.error(f"File {file} does not exist!")
        return df
    else:
        # check if the file is empty
        if os.stat(file).st_size == 0:
            logger.warning(f"File {file} is empty, skipping ...")
            return df

    # Read the CSV file
    df = dd.read_csv(file)
    if len(df) == 0:
        logger.warning(f"No data found in file {file}, skipping ...")
        return df

    # clean up the dataframe
    if "phase_index" in df.columns:
        # seems to be fake picks
        df = df[df["phase_index"] != 1]

    df = df.drop(
        columns=["begin_time", "phase_index", "file_name"], errors="ignore", axis=1
    )

    # Rename the columns to match the standardized format
    df = df.rename(
        columns={
            "seedid": "station_id",
            "phasename": "phase_type",
            "time": "phase_time",
            "probability": "phase_score",
        },
    )

    # Add missing columns
    df["phase_evaluation"] = "automatic"
    df["phase_method"] = "PHASENET"
    df["event_id"] = ""
    df["agency"] = "RENASS"
    df["channel"] = ""

    # Convert the time column to UTCDateTime objects
    df["phase_time"] = dd.to_datetime(df["phase_time"], errors="coerce")

    # limits to 10^-4 seconds same as NLL
    # (needed by dbclust to unload some picks)
    df["phase_time"] = df["phase_time"].dt.round("0.0001s")

    df["channel"] = df["station_id"].apply(
        lambda x: ".".join(x.split(".")[2:4]), meta=("channel", "str")
    )

    df["station_id"] = df["station_id"].apply(
        lambda x: ".".join(x.split(".")[0:2]), meta=("station_id", "str")
    )

    # Add the column types
    # col_types = {
    #     "station_id": "string",
    #     "channel": "string",
    #     "phase_type": "string",
    #     "phase_time": "string",
    #     "phase_score": "float64",
    #     "phase_evaluation": "string",
    #     "phase_method": "string",
    #     "event_id": "string",
    #     "agency": "string",
    # }
    # df = df.astype(col_types)

    return df


def export_to_csv(df: dd.DataFrame, outputfile: str) -> None:
    """
    Export the processed Dask DataFrame to a CSV file.
    Args:
        df (dd.DataFrame): The processed Dask DataFrame.
        outputfile (str): The output CSV file path.
    """

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

    df.to_csv(outputfile, single_file=True, index=False, columns=col_order)


def convert_phasenet_to_dbclust_format(phasenet_file: str, dbclust_file: str) -> None:
    """
    Convert PhaseNet seismic phase data from CSV files to a standardized format and save it to a CSV file.
    Args:
        phasenet_file (str): The input PhaseNet CSV file.
        dbclust_file (str): The output CSV file to save the processed data.
    """

    # Read the PhaseNet CSV file and modify the columns
    df = read_and_modify(phasenet_file)

    # Export the processed Dask DataFrame to a CSV file
    export_to_csv(df, dbclust_file)


def export_phasenet_to_parquet(
    inputfile: str,
    outputfile: str,
) -> None:
    """
    Export the processed Dask DataFrame to a Parquet file.
    Args:
        inputfile (str): The input CSV file.
        outputfile (str): The output Parquet file path.
    """

    # Read the PhaseNet CSV file and modify the columns
    df = read_and_modify(inputfile)
    if len(df) == 0:
        logger.warning(f"No data found in file {inputfile}, skipping ...")
        return

    # Convert 'phase_time' to datetime and remove timezone
    # limits to 10^-4 seconds same as NLL
    # (needed by dbclust to unload some picks)
    df["phase_time"] = dd.to_datetime(df["phase_time"], errors="coerce").dt.tz_localize(
        None
    )
    df["phase_time"] = df["phase_time"].dt.round("0.0001s")

    # Ensure 'phase_time' / 'idxtime' is the index
    df["idxtime"] = df["phase_time"]
    df = df.set_index("idxtime", sorted=True)

    # Check if the index is a datetime type
    if not pd.api.types.is_datetime64_any_dtype(df.index):
        raise ValueError("Index is not a datetime type.")

    # handle the partition
    df["year"] = df["phase_time"].dt.year
    df["month"] = df["phase_time"].dt.month
    df["day"] = df["phase_time"].dt.day

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
    col_order += ["year", "month", "day"]
    df = df[col_order]

    # Export the partitioned Dask DataFrame to a Parquet file
    df.to_parquet(
        outputfile,
        engine="pyarrow",
        compression="snappy",
        append=True,
        write_index=False,
        partition_on=["year", "month", "day"],
    )


def export_dbclust_to_parquet(
    inputfile: str,
    outputfile: str,
) -> None:
    """
    Export the processed Dask DataFrame to a Parquet file.
    Args:
        inputfile (str): The input CSV file.
        outputfile (str): The output Parquet file path.
    """
    # Read the CSV file
    df = dd.read_csv(inputfile)
    if len(df) == 0:
        logger.warning(f"No data found in file {inputfile}, skipping ...")
        return df

    # Convert 'phase_time' to datetime and remove timezone
    # limits to 10^-4 seconds same as NLL
    # (needed by dbclust to unload some picks)
    df["phase_time"] = dd.to_datetime(df["phase_time"], errors="coerce").dt.tz_localize(
        None
    )
    df["phase_time"] = df["phase_time"].dt.round("0.0001s")

    # Ensure 'phase_time' / 'idxtime' is the index
    df["idxtime"] = df["phase_time"]
    df = df.set_index("idxtime", sorted=True)

    # Check if the index is a datetime type
    if not pd.api.types.is_datetime64_any_dtype(df.index):
        raise ValueError("Index is not a datetime type.")

    # handle the partition
    df["year"] = df["phase_time"].dt.year
    df["month"] = df["phase_time"].dt.month
    df["day"] = df["phase_time"].dt.day

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
    col_order += ["year", "month", "day"]
    df = df[col_order]

    # Export the partitioned Dask DataFrame to a Parquet file
    df.to_parquet(
        outputfile,
        engine="pyarrow",
        compression="snappy",
        append=True,
        write_index=False,
        partition_on=["year", "month", "day"],
    )


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

    # sort the files by date in increasing lexicographic order
    # eg: RA.ECH.00.HN-2024.275.picks.csv
    csv_files = glob.glob(args.inputfile)

    try:
        sorted_files = sorted(csv_files, key=lambda x: x.split("-")[1])
    except IndexError:
        sorted_files = sorted(csv_files)

    # for f in sorted_files:
    #     logger.info(f"Reading PhaseNet file {f}")
    #     df = read_and_modify(f)
    #     export_to_csv(df, args.outputfile)
    #
    # df = dd.concat([read_and_modify(f) for f in sorted_files])
    # export_to_csv(df, args.outputfile)

    # Get the maximum number of CPUs
    # max_cpus = os.cpu_count()

    # # Use ProcessPoolExecutor with the max number of CPUs
    # with concurrent.futures.ProcessPoolExecutor(max_workers=max_cpus) as executor:
    #     results = list(executor.map(export_to_parquet, sorted_files))

    for i, f in enumerate(sorted_files):
        logger.info(f"Reading PhaseNet file {f}")

        # check the file type looking for the header
        with open(f) as file:
            header = file.readline()
            if "seedid" in header:
                print("Found PHASENET header")
                export_phasenet_to_parquet(f, args.outputfile)
            elif "event_id" in header:
                print("Found DBClust header")
                export_dbclust_to_parquet(f, args.outputfile)
            else:
                logger.error(f"Unknown file type for file {f}")
                sys.exit(1)

    logger.info("Done!")
