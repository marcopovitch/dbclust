#!/usr/bin/env python
"""
This script exports seismic picks from a SpatiaLite database to a CSV file
in a format compatible with DBClust.

This is the database equivalent of qml2dbclust.py - instead of reading from
QuakeML files, it reads from a SQLite database created by inject_spatialite.py.

Usage:
    python db2dbclust.py -i database.db -o outputfile.csv [options]

Options:
    -i, --input: Input SQLite database file (required)
    -o, --output: Output CSV file (required)
    -p, --probability: Set probability (default: 1)
    -e, --evaluation: Override pick evaluation mode (automatic|manual)
    -m, --method: Override pick method_id (AIC|PHASENET|...)
    -a, --agency: Agency name
    --from: Select start time window
    --to: Select end time window
    -k, --keep-manual: Keep picks with manual evaluation only
    -d, --keep-disabled: Keep disabled picks (time_weight = 0)
    -l, --loglevel: Set log level (debug, warning, info, error)
    --progress: Show progress bar

Example:
    python db2dbclust.py -i catalog.db -o picks.csv -p 0.9 -e manual -m PHASENET -a RENASS --from 2023-01-01T00:00:00 --to 2023-12-31T23:59:59 -l info
"""

import argparse
import logging
import os
import sqlite3
import sys
from typing import Any, Dict, List

import pandas as pd
from tqdm import tqdm

from dbclust.inject_spatialite import create_safe_connection, load_spatialite


# Default logger
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger("db2dbclust")
logger.setLevel(logging.INFO)


def filter_LDG_P_S(lines: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    For each station, filter out all lines with phase_type P or S
    if other lines contain Pn, Pg or Sn, Sg.

    Args:
        lines (List[Dict[str, Any]]): A list of dictionaries containing
                                        pick information.

    Returns:
        List[Dict[str, Any]]: the filtered list.
    """
    filtered_lines = []
    for line in lines:
        station_id = line["station_id"]
        phase_type = line["phase_type"]
        if phase_type == "P":
            contains_Pn = any(
                l["phase_type"] == "Pn" for l in lines if l["station_id"] == station_id
            )
            contains_Pg = any(
                l["phase_type"] == "Pg" for l in lines if l["station_id"] == station_id
            )
            if not (contains_Pn or contains_Pg):
                filtered_lines.append(line)
        elif phase_type == "S":
            contains_Sn = any(
                l["phase_type"] == "Sn" for l in lines if l["station_id"] == station_id
            )
            contains_Sg = any(
                l["phase_type"] == "Sg" for l in lines if l["station_id"] == station_id
            )
            if not (contains_Sn or contains_Sg):
                filtered_lines.append(line)
        else:
            filtered_lines.append(line)
    return filtered_lines


def export_picks_from_database(
    conn: sqlite3.Connection,
    from_time: str = None,
    to_time: str = None,
    probability: float = 1,
    evaluation: str = None,
    method: str = None,
    agency: str = None,
    keep_manual_evaluation_only: bool = False,
    keep_disabled_picks: bool = False,
    show_progress: bool = False,
) -> List[Dict[str, Any]]:
    """
    Export picks from the database in DBClust format.

    Args:
        conn: SQLite database connection
        from_time: Start time filter (ISO format)
        to_time: End time filter (ISO format)
        probability: Default probability score
        evaluation: Override evaluation mode
        method: Override method_id
        agency: Agency name
        keep_manual_evaluation_only: Keep only manual picks
        keep_disabled_picks: Include picks with time_weight = 0
        show_progress: Show progress bar with tqdm

    Returns:
        List of dictionaries with pick information in DBClust format.
    """
    cursor = conn.cursor()

    # Build the query: start from arrivals of preferred origins,
    # then join picks. This ensures we only get picks that are
    # actually associated with the preferred origin.
    query = """
        SELECT
            p.station_name,
            p.id as pick_id,
            p.pick_time,
            p.evaluation_mode,
            p.phase_hint,
            p.probability,
            a.name as arrival_phase,
            a.time_weight,
            e.event_id,
            o.time as origin_time,
            p.location_code,
            p.channel_code
        FROM origins o
        JOIN events e ON e.event_id = o.event_id
        JOIN arrivals a ON a.origin_id = o.id
        JOIN picks p ON p.id = a.pick_id
        WHERE o.preferred = 1
    """

    params = []

    # Add time filters based on origin time
    if from_time:
        query += " AND o.time >= ?"
        params.append(from_time)
    if to_time:
        query += " AND o.time < ?"
        params.append(to_time)

    # Filter disabled picks unless explicitly requested
    if not keep_disabled_picks:
        query += " AND a.time_weight != 0"

    query += " ORDER BY o.time, p.pick_time"

    logger.debug(f"Executing query: {query}")
    logger.debug(f"With params: {params}")

    cursor.execute(query, params)
    rows = cursor.fetchall()

    logger.info(f"Found {len(rows)} picks in database")

    lines = []
    row_iterator = tqdm(rows, desc="Exporting picks", disable=not show_progress)
    for row in row_iterator:
        station_name = row[0]  # Format: NET.STA
        pick_id = row[1]
        pick_time = row[2]
        eval_mode = row[3]
        phase_hint = row[4]
        pick_probability = row[5]
        arrival_phase = row[6]
        time_weight = row[7]
        event_id = row[8]
        origin_time = row[9]
        location_code = row[10] or ""
        channel_code = row[11] or ""

        # Skip if keeping manual only and pick is not manual
        if keep_manual_evaluation_only and eval_mode != "manual":
            continue

        # Use arrival phase if available, otherwise use phase_hint
        phase_type = arrival_phase if arrival_phase else phase_hint

        if not phase_type:
            logger.warning(f"No phase type for pick {pick_id}, skipping")
            continue

        # Parse station_name to get network.station
        parts = station_name.split(".") if station_name else []
        if len(parts) >= 2:
            station_id = f"{parts[0]}.{parts[1]}"
        else:
            station_id = station_name

        channel = f"{location_code}.{channel_code}"

        line = {
            "station_id": station_id,
            "channel": channel,
            "phase_type": phase_type,
            "phase_time": pick_time,
            "phase_score": pick_probability if pick_probability else probability,
            "phase_evaluation": eval_mode,
            "phase_method": None,  # method_id not stored in picks table
            "event_id": event_id,
        }

        # Override from command line args
        if evaluation:
            line["phase_evaluation"] = evaluation
        if method:
            line["phase_method"] = method
        if agency:
            line["agency"] = agency

        lines.append(line)

    # Apply LDG filter if agency is LDG
    if agency == "LDG":
        logger.info("Filtering out P and S phases for LDG")
        lines = filter_LDG_P_S(lines)

    return lines


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export picks from SQLite database to DBClust CSV format"
    )
    parser.add_argument(
        "-i",
        "--input",
        default=None,
        dest="inputfile",
        help="input SQLite database file",
        type=str,
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        dest="outputfile",
        help="output csv file",
        type=str,
    )
    parser.add_argument(
        "-p",
        "--probability",
        default=1,
        dest="probability",
        help="set probability",
        type=float,
    )
    parser.add_argument(
        "-e",
        "--evaluation",
        default=None,
        dest="evaluation",
        help="override pick evaluation mode (automatic|manual)",
        type=str,
    )
    parser.add_argument(
        "-k",
        "--keep-manual",
        default=False,
        dest="keep_manual_evaluation_only",
        help="keep picks with manual evaluation only",
        action="store_true",
    )
    parser.add_argument(
        "-d",
        "--keep-disabled",
        default=False,
        dest="keep_disabled_picks",
        help="keep disabled picks",
        action="store_true",
    )
    parser.add_argument(
        "-m",
        "--method",
        default=None,
        dest="method",
        help="override pick method_id (AIC|PHASENET|...)",
        type=str,
    )
    parser.add_argument(
        "-a",
        "--agency",
        default=None,
        dest="agency",
        help="agency name",
        type=str,
    )
    parser.add_argument(
        "--from",
        default=None,
        dest="from_time",
        help="select start time window",
        type=str,
    )
    parser.add_argument(
        "--to",
        default=None,
        dest="to_time",
        help="select end time window",
        type=str,
    )
    parser.add_argument(
        "-l",
        "--loglevel",
        default="INFO",
        dest="loglevel",
        help="loglevel (debug,warning,info,error)",
        type=str,
    )
    parser.add_argument(
        "--progress",
        default=False,
        dest="show_progress",
        help="show progress bar",
        action="store_true",
    )
    args = parser.parse_args()

    if not args.inputfile or not args.outputfile:
        parser.print_help()
        sys.exit(255)

    if not os.path.isfile(args.inputfile):
        logger.error(f"Input database {args.inputfile} does not exist!")
        sys.exit(255)

    if os.path.isfile(args.outputfile):
        logger.error(f"Output file {args.outputfile} already exists!")
        sys.exit(255)

    numeric_level = getattr(logging, args.loglevel.upper(), None)
    if not numeric_level:
        logger.error("Invalid loglevel '%s'!", args.loglevel.upper())
        logger.error("loglevel should be: debug,warning,info,error.")
        sys.exit(255)
    logger.setLevel(numeric_level)

    # Connect to database
    logger.info(f"Connecting to database {args.inputfile}")
    conn = create_safe_connection(args.inputfile, logger=logger)
    load_spatialite(conn, logger)

    # Export picks
    picks_list = export_picks_from_database(
        conn,
        from_time=args.from_time,
        to_time=args.to_time,
        probability=args.probability,
        evaluation=args.evaluation,
        method=args.method,
        agency=args.agency,
        keep_manual_evaluation_only=args.keep_manual_evaluation_only,
        keep_disabled_picks=args.keep_disabled_picks,
        show_progress=args.show_progress,
    )

    conn.close()

    # Create DataFrame with expected columns
    df = pd.DataFrame(
        picks_list,
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
        ],
    )

    logger.info(f"Exporting {len(df)} picks to {args.outputfile}")
    df.to_csv(args.outputfile, index=False)
    logger.info("Done.")
