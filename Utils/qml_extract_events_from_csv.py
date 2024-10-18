#!/usr/bin/env python
import argparse
import os
import platform
import subprocess
import sys

import pandas as pd
from obspy import Catalog
from obspy import read_events


def modify_sc3ml_version(filename: str) -> None:
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

    parser = argparse.ArgumentParser(description="Process CSV and QML files")
    parser.add_argument(
        "--csv",
        "-c",
        required=True,
        dest="csvfile",
        help="CSV file containing event IDs",
    )
    parser.add_argument(
        "--qml",
        "-q",
        required=True,
        dest="qmlfile",
        help="QML file containing seismic events",
    )
    parser.add_argument(
        "--output", "-o", dest="outputfile", required=True, help="Output file name"
    )
    parser.add_argument(
        "--eventid_col",
        default="event_id",
        help="Name of the column in the CSV file containing event IDs",
    )

    args = parser.parse_args()

    if not os.path.exists(args.csvfile) or not os.path.isfile(args.csvfile):
        print(f"'{args.csvfile}' does not exist, or is not a file")
        sys.exit(1)

    if not os.path.exists(args.qmlfile) or not os.path.isfile(args.qmlfile):
        print(f"'{args.qmlfile}' does not exist, or is not a file")
        sys.exit(1)

    _, qml_ext = os.path.splitext(args.outputfile)
    if qml_ext != ".qml":
        print(f"'{args.qmlfile}' should have '.qml' extension !")
        sys.exit(1)

    if os.path.exists(args.outputfile):
        print(f"'{args.outputfile}' already exits !")
        sys.exit(1)

    df = pd.read_csv(args.csvfile)
    cat = read_events(args.qmlfile)

    cat_keep = Catalog()
    to_keep = []

    for e in cat.events:
        if e.resource_id.id in df[args.eventid_col].values:
            to_keep.append(e)

    cat_keep.events = to_keep

    # Write the output file to QML format
    cat_keep.write(args.outputfile, format="QUAKEML")
    output_sc3ml = args.outputfile.replace(".qml", ".sc3ml")

    # Write the output file to SC3ML format
    cat_keep.write(output_sc3ml, format="SC3ML")
    modify_sc3ml_version(output_sc3ml)
