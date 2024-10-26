#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import json
import os
import sys
from math import isclose
from pathlib import Path

import pandas as pd
from icecream import ic
from obspy import Catalog
from obspy import read_events

# set time_weight tolerance
time_weight_tolerance = 0.01


# export phase relabeling info to dataframe
def export_phase_relabeling(cat: Catalog = None) -> pd.DataFrame:
    """
    Export phase relabeling information from a given catalog.

    This function processes a catalog of seismic events and extracts phase relabeling
    information, compiling it into a pandas DataFrame.

    Parameters:
    cat (Catalog, optional):
                A catalog of seismic events. If not provided, an empty DataFrame
                with the appropriate columns is returned.
    Returns:
    pd.DataFrame: A DataFrame containing phase relabeling information for each event in the catalog.
                The columns of the DataFrame are:
                - "event_id": The ID of the event.
                - "station": The station name.
                - "prev_phase": The previous phase before relabeling.
                - "action": The relabeling action taken.
                - "eval_score": The evaluation score of the relabeling.
                - "relabeled_phase": The new phase after relabeling.
                - "relabeled_score": The score of the relabeled phase.
                - "scores": The scores used for relabeling.
    """

    # create a dataframe
    df = pd.DataFrame(
        columns=[
            "event_id",
            "station",
            "prev_phase",
            "action",
            "eval_score",
            "relabeled_phase",
            "relabeled_score",
            "scores",
        ]
    )

    if not cat:
        return df

    for event in cat:
        origin = event.preferred_origin()

        for arrival in origin.arrivals:
            if hasattr(arrival, "time_weight") and isclose(
                arrival.time_weight, 0, abs_tol=time_weight_tolerance
            ):
                used = False
            else:
                used = True

            pick = next(
                (p for p in event.picks if p.resource_id == arrival.pick_id), None
            )
            if pick is None:
                continue

            wfid = pick.waveform_id
            station_name = f"{wfid.network_code}.{wfid.station_code}"
            phase_name = arrival.phase

            for c in arrival.comments:
                try:
                    info = json.loads(c.text)
                except:
                    continue

                if "relabel" in info.keys():
                    scores = info["relabel"]["scores"]
                    # get best score and phase fromn scores
                    best_score = max(scores.values())
                    best_phase = max(scores, key=scores.get)

                    new_row = {
                        "event_id": event.resource_id.id,
                        "station": station_name,
                        "prev_phase": info["relabel"]["prev_phase"],
                        "action": info["relabel"]["action"],
                        "eval_score": info["relabel"]["eval_score"],
                        "relabeled_phase": best_phase,
                        "relabeled_score": best_score,
                        "scores": info["relabel"]["scores"],
                    }
                    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    return df


def find_qml_files(directory: str) -> list:
    return list(Path(directory).rglob("*.qml"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Read ")
    parser.add_argument("-d", "--directory", help="Directory containing event files")
    # add -i to read one event file
    parser.add_argument("-i", "--input", help="Input qml file")
    parser.add_argument("-o", "--output", help="Output csv file")
    args = parser.parse_args()

    if args.output is None:
        print("Please provide the output file")
        sys.exit(1)

    if args.directory is None and args.input is None:
        print("Please provide either the events directory or the event file")
        sys.exit(1)

    if args.directory is not None and args.input is not None:
        print("Please provide either the events directory or the event file, not both")
        sys.exit(1)

    if args.input is not None:
        # check if the input file exists
        if not os.path.exists(args.input):
            print(f"File {args.input} does not exist")
            sys.exit(1)
        cat = read_events(args.input)
        df = export_phase_relabeling(cat)
        df.to_csv(args.output, index=False)
        sys.exit(0)

    if not os.path.exists(args.directory):
        print(f"Directory {args.directory} does not exist")
        sys.exit(1)

    df = pd.DataFrame(
        columns=[
            "event_id",
            "station",
            "prev_phase",
            "action",
            "eval_score",
            "relabeled_phase",
            "relabeled_score",
            "scores",
        ]
    )
    # Get event recursively from the directory
    for event in find_qml_files(args.directory):
        cat = read_events(event)
        ic(cat.events[0].resource_id.id)
        tmp = export_phase_relabeling(cat)
        df = pd.concat([df, tmp], ignore_index=True)

    df.to_csv(args.output, index=False)
