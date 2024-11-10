#!/usr/bin/env python
import argparse
import os
import shutil
import sys
from datetime import datetime
from typing import List

import dask.dataframe as dd
import duckdb
import numpy as np
import pandas as pd
import tqdm
from dask.distributed import Client
from dask.distributed import LocalCluster


def convert_csv_to_parquet(
    csv_files: List[str],
    parquet_file: str,
) -> None:
    """
    Convert CSV files to Parquet file

    Args:
        csv_files (List[str]): List of CSV files
        parquet_file (str): Parquet output file
    """
    # print(f"Writing to {parquet_file} parquet file")
    nb_procs = 1
    # nb_procs = os.cpu_count()
    # cluster = LocalCluster(n_workers=nb_procs)
    # client = Client(cluster)

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

    # Chargement des CSV avec Dask
    ddf = dd.read_csv(csv_files, dtype=col_types)

    # Conversion en datetime sans fuseau horaire pour 'phase_time' et arrondi
    ddf["phase_time"] = dd.to_datetime(
        ddf["phase_time"], errors="coerce"
    ).dt.tz_localize(None)
    ddf["phase_time"] = ddf["phase_time"].dt.round("1ms")
    ddf = ddf.sort_values(by="phase_time").compute()

    # Définir l'index sans forcer de tri immédiat
    ddf["idxtime"] = ddf["phase_time"]
    ddf = ddf.set_index("idxtime")

    # handle the partition
    ddf["year"] = ddf["phase_time"].dt.year
    ddf["month"] = ddf["phase_time"].dt.month

    ddf = dd.from_pandas(ddf, npartitions=nb_procs)

    # print("Writing parquet file")
    ddf.to_parquet(
        parquet_file,
        partition_on=["year", "month"],
        compression="snappy",
        engine="pyarrow",
        write_index=False,
        append=True,
    )

    # Fermer le client et le cluster Dask
    # client.close()
    # cluster.close()


def repartition_parquet(parquet_file_in: str, parquet_file_out: str) -> None:
    """
    Repartition a Parquet file using duckdb

    Args:
        parquet_file_in (str): Input Parquet file
        parquet_file_out (str): Output Parquet file
    """
    print(f"Repartitioning {parquet_file_in} to {parquet_file_out}")

    sql = f"""
        COPY (SELECT * FROM read_parquet('{parquet_file_in}/**/*.parquet'))
        TO '{parquet_file_out}'
        (FORMAT 'parquet', PARTITION_BY (year, month));
    """

    conn = duckdb.connect()
    try:
        conn.execute(sql)
    finally:
        conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert CSV to Parquet file")
    parser.add_argument(
        "-i",
        "--input",
        nargs="+",
        help="CSV input files (multiple files allowed)",
    )
    parser.add_argument("-o", "--output", required=True, help="Parquet output")
    parser.add_argument(
        "-d",
        "--directory",
        type=str,
        help="Input directory containing CSV files",
    )
    # add batch size
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=100,
        help="Batch size for processing the input files",
    )
    args = parser.parse_args()

    if args.directory and args.input:
        print("Cannot specify both input directory and input files")
        sys.exit(1)

    if args.output and os.path.exists(args.output):
        print(f"Output directory {args.output} already exists")
        sys.exit(1)

    # overwrite input files with files from the directory
    if args.directory:
        args.input = []
        for root, _, files in os.walk(args.directory):
            for file in files:
                if file.endswith(".csv"):
                    args.input.append(os.path.join(root, file))
        print(f"Input files: {len(args.input)}")

    # check if the output file already exists only if input was specified
    if not args.directory:
        for f in args.input:
            if not os.path.exists(f):
                print(f"File {f} does not exist !")
                sys.exit(1)

    # process the input files by batch
    tmp_parquet = ".".join([args.output, "tmp.parquet"])
    for i in tqdm.tqdm(range(0, len(args.input), args.batch_size)):
        # print(f"Processing files {i} to {i+args.batch_size}")
        convert_csv_to_parquet(
            args.input[i : i + args.batch_size],
            tmp_parquet,
        )

    repartition_parquet(tmp_parquet, args.output)

    # remove the temporary parquet file
    shutil.rmtree(tmp_parquet)
