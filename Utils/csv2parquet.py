#!/usr/bin/env python
import argparse
import os
import shutil
import sys
from typing import List

import duckdb
import tqdm


def convert_csv_to_parquet_duckdb(
    csv_files: List[str],
    parquet_dir: str,
    batch_id: int,
) -> None:
    """
    Convert CSV files to Parquet file using DuckDB (fast)

    Args:
        csv_files (List[str]): List of CSV files
        parquet_dir (str): Parquet output directory
        batch_id (int): Batch identifier for unique filenames
    """
    conn = duckdb.connect()
    try:
        files_list = ", ".join([f"'{f}'" for f in csv_files])

        # write each batch to a separate parquet file (no partitioning here)
        sql = f"""
            COPY (
                SELECT
                    station_id,
                    channel,
                    phase_type,
                    time_bucket(INTERVAL '1 millisecond', phase_time::TIMESTAMP) as phase_time,
                    phase_score,
                    phase_evaluation,
                    phase_method,
                    event_id,
                    agency
                FROM read_csv([{files_list}],
                    columns = {{
                        'station_id': 'VARCHAR',
                        'channel': 'VARCHAR',
                        'phase_type': 'VARCHAR',
                        'phase_time': 'VARCHAR',
                        'phase_score': 'DOUBLE',
                        'phase_evaluation': 'VARCHAR',
                        'phase_method': 'VARCHAR',
                        'event_id': 'VARCHAR',
                        'agency': 'VARCHAR'
                    }},
                    header = true
                )
            )
            TO '{parquet_dir}/batch_{batch_id:06d}.parquet'
            (FORMAT 'parquet', COMPRESSION 'snappy');
        """
        conn.execute(sql)
    finally:
        conn.close()


def merge_parquet_partitions(parquet_file_in: str, parquet_file_out: str) -> None:
    """
    Merge parquet files into a single partitioned dataset using DuckDB

    Args:
        parquet_file_in (str): Input Parquet directory containing batch files
        parquet_file_out (str): Output Parquet directory
    """
    print(f"Merging parquet files from {parquet_file_in} to {parquet_file_out}")

    conn = duckdb.connect()
    try:
        sql = f"""
            COPY (
                SELECT
                    station_id,
                    channel,
                    phase_type,
                    phase_time,
                    phase_score,
                    phase_evaluation,
                    phase_method,
                    event_id,
                    agency,
                    year(phase_time) as year,
                    month(phase_time) as month
                FROM read_parquet('{parquet_file_in}/*.parquet')
                ORDER BY phase_time
            )
            TO '{parquet_file_out}'
            (FORMAT 'parquet', PARTITION_BY (year, month), COMPRESSION 'snappy');
        """
        conn.execute(sql)
    finally:
        conn.close()


def main():
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
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=500,
        help="Batch size for processing the input files (default: 500)",
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

    if not args.input:
        print("No input files specified")
        sys.exit(1)

    # check if the output file already exists only if input was specified
    if not args.directory:
        for f in args.input:
            if not os.path.exists(f):
                print(f"File {f} does not exist !")
                sys.exit(1)

    # process the input files by batch
    tmp_parquet = ".".join([args.output, "tmp.parquet"])
    os.makedirs(tmp_parquet, exist_ok=True)
    batch_id = 0
    for i in tqdm.tqdm(range(0, len(args.input), args.batch_size)):
        convert_csv_to_parquet_duckdb(
            args.input[i : i + args.batch_size],
            tmp_parquet,
            batch_id,
        )
        batch_id += 1

    merge_parquet_partitions(tmp_parquet, args.output)

    # remove the temporary parquet file
    shutil.rmtree(tmp_parquet)


if __name__ == "__main__":
    main()
