#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import os
import platform
import subprocess
import warnings
from multiprocessing import Pool

from obspy import Catalog
from obspy import read_events
from obspy import UTCDateTime


def process_month(cat: Catalog, year: int, month: int, output_dir: str) -> None:
    starttime = UTCDateTime(year, month, 1)
    if month == 12:
        endtime = UTCDateTime(year + 1, 1, 1)
    else:
        endtime = UTCDateTime(year, month + 1, 1)
    month_cat = cat.filter(f"time >= {starttime}", f"time < {endtime}")
    if len(month_cat) == 0:
        return

    # split filename
    split_file = f"{output_dir}/{year}_{month}.sc3ml"
    print(split_file)
    # suppress the warning about the version of QuakeML
    month_cat.write(split_file, format="SC3ML")
    modify_sc3ml_version(split_file)


def split(filename: str, output_dir: str, n_jobs: int = 4) -> None:
    # read quakeml
    print(f"Reading {filename} ...")
    cat = read_events(filename)

    # split catalog by year and month using cat.filter on time attribute
    # get the min and max years from the events in catalog
    time_list = [ev.origins[0].time for ev in cat]
    min_year = min(time_list).year
    max_year = max(time_list).year

    warnings.filterwarnings("ignore", category=UserWarning)

    with Pool(n_jobs) as pool:
        pool.starmap(
            process_month,
            [
                (cat, year, month, output_dir)
                for year in range(min_year, max_year + 1)
                for month in range(1, 13)
            ],
        )


def modify_sc3ml_version(filename: str):
    # Substitute in filename using sed command
    # String to convert:
    # xmlns="http://geofon.gfz-potsdam.de/ns/seiscomp3-schema/0.12 by xmlns="http://geofon.gfz-potsdam.de/ns/seiscomp3-schema/0.11
    # version="0.12" by version="0.11
    sed_cmd = [
        "sed",
        "-i",
        "-e",
        's/xmlns="http:\/\/geofon.gfz-potsdam.de\/ns\/seiscomp3-schema\/0.12/xmlns="http:\/\/geofon.gfz-potsdam.de\/ns\/seiscomp3-schema\/0.11/g',
        "-e",
        's/version="0.12"/version="0.11"/g',
        filename,
    ]

    # On macOS, we need to add an empty argument for the -i option
    if platform.system() == "Darwin":
        sed_cmd.insert(2, "")  # insert empty string at index 2
    subprocess.run(sed_cmd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split QuakeML catalog by year and month"
    )
    parser.add_argument(
        "-f", "--catalog", type=str, help="Path to QuakeML catalog file"
    )
    parser.add_argument(
        "-d", "--directory", type=str, help="Output directory for split files"
    )
    parser.add_argument(
        "-j", "--n_jobs", type=int, default=4, help="Number of parallel jobs"
    )
    args = parser.parse_args()

    # Verify if catalog file exists
    if not os.path.isfile(args.catalog):
        raise FileNotFoundError(f"Catalog file '{args.catalog}' does not exist.")

    # Verify if directory does not exist
    if os.path.exists(args.directory):
        raise FileExistsError(f"Output directory '{args.directory}' already exists.")
    else:
        os.makedirs(args.directory)

    split(args.catalog, args.directory, args.n_jobs)
