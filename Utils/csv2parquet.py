#!/usr/bin/env python
import sys
import argparse
import dask.dataframe as dd
from dask.distributed import Client, LocalCluster
import pandas as pd
from datetime import datetime
import numpy as np
from icecream import ic


def is_valid_freq(freq):
    try:
        pd.tseries.frequencies.to_offset(freq)
        return True
    except ValueError:
        return False


def convert_csv_to_parquet(
    csv_file: str,
    parquet_file: str,
    time_name: str,
    partition_duration: str,
) -> None:
    cluster = LocalCluster(n_workers=4)
    client = Client(cluster)

    ddf = dd.read_csv(
        csv_file,
        dtype={
            "station_id": "string",
            "phase_type": "string",
            "phase_time": "string",
            "phase_score": "float64",
            "eventid": "string",
            "agency": "string",
            "channel": "string",
        },
    )
    ddf[time_name] = dd.to_datetime(ddf[time_name])

    # remove what seems to be fake picks
    if "phase_index" in ddf.columns:
        ddf = ddf[ddf["phase_index"] != 1]

    # limits to 10^-4 seconds same as NLL (needed by dbclust to unload some picks)
    ddf["phase_time"] = ddf["phase_time"].dt.round("0.0001s")

    ddf = ddf.dropna(subset=["station_id"])

    # get rid off nan value when importing phases without eventid
    ddf = ddf.replace({np.nan: ""})

    # Needed for time partition
    ddf["idxtime"] = ddf[time_name]
    ddf = ddf.set_index("idxtime")

    ddf_partitioned = ddf.repartition(freq=partition_duration)

    print("Writing parquet file")

    ddf_partitioned.to_parquet(
        parquet_file,
        engine="pyarrow",
        compression="snappy",
        write_index=False,
    )

    client.close()
    cluster.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert CSV to Parquet file")
    parser.add_argument("-i", "--input", required=True, help="CSV input file")
    parser.add_argument("-o", "--output", required=True, help="Parquet output file")
    parser.add_argument(
        "-t", "--time_name", required=True, help="Name of the time column"
    )
    parser.add_argument(
        "--part_duration", required=True, type=str, help="Partition duration"
    )
    args = parser.parse_args()

    if not is_valid_freq(args.part_duration):
        raise ValueError("This is not a legit pandas frequency !")

    convert_csv_to_parquet(
        args.input,
        args.output,
        args.time_name,
        args.part_duration,
    )
