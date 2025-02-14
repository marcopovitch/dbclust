#!/usr/bin/env python3
import argparse
import csv
import logging
import os
import sqlite3
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from icecream import ic
from obspy.core.event import Arrival
from obspy.core.event import Catalog
from obspy.core.event import Event
from obspy.core.event import Pick
from obspy.core.event import read_events
from obspy.core.utcdatetime import UTCDateTime
from rich.console import Console
from rich.table import Table
from scipy.stats import linregress
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def show_dataframes(df: pd.DataFrame, title: str) -> None:
    """Show the content of a DataFrame using the Rich library."""
    console = Console()
    table = Table(show_header=True, header_style="bold magenta", title=title)

    for col in df.columns:
        table.add_column(col, justify="center")
    for row in df.itertuples(index=False):
        #table.add_row(*row)
        table.add_row(*(f"{cell:.2f}" if isinstance(cell, float) else str(cell) for cell in row))
    console.print(table)


def read_polygons_from_yaml(filename: str):
    """
    Reads a YAML file containing a list of polygons with a title and coordinates.

    Args:
        filename (str): Path to the YAML file.

    Returns:
        list: List of polygons as dictionaries {"title": str, "coordinates": list}
    """
    with open(filename, "r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    return data.get("polygons", [])


def generate_wkt_from_polygon(coord_list: list) -> str:
    """
    Generate a polygon WKT (Well-Known Text) string from a list of coordinates.

    Args:
        coord_list (list): List of (longitude, latitude) tuples.

    Returns:
        str: WKT string representing the polygon.
    """
    if coord_list:
        coordinates = ", ".join(f"{lon} {lat}" for lon, lat in coord_list)
    else:
        coordinates = ""
    polygon_wkt = f"POLYGON(({coordinates}))"

    return polygon_wkt


def get_polygon_index(polygons: list, title: str) -> int:
    """
    Retrieve the index of a polygon in a list of polygons.

    Args:
        polygons (list): List of polygons as dictionaries.
        title (str): Title of the polygon to search for.

    Returns:
        int: Index of the polygon in the list, or -1 if not found.
    """
    for i, p in enumerate(polygons):
        if p.get("title") == title:
            return i
    return -1


def get_pick_from_arrival(event: Event, arrival: Arrival) -> Pick:
    """
    Retrieve a Pick object from given Arrival in a given Event.
    This function searches through the picks associated with the given event
    and returns the pick that matches the resource ID specified in the arrival.
    Args:
        event (Event): The event containing a list of picks.
        arrival (Arrival): The arrival containing the pick ID to search for.
    Returns:
        Pick: The pick that matches the arrival's pick ID, or None if no match is found.
    """
    pick = next((p for p in event.picks if p.resource_id == arrival.pick_id), None)
    return pick


def get_station_name_from_pick(pick: Pick) -> str:
    """
    Retrieve the station name from a given Pick object.
    Args:
        pick (Pick): The pick to retrieve the station name from.
    Returns:
        str: The station name associated with the pick.
    """
    station_name = f"{pick.waveform_id.network_code}.{pick.waveform_id.station_code}"
    return station_name


def wadati_catalog(catalog: Catalog) -> tuple:
    """
    Compute the Vp/Vs ratio using the Wadati method for a given catalog.

    Args:
        catalog (Catalog): The catalog of events to process.

    Returns:
        tuple: (T_P, T_S, e_ids, station_names).
    """
    # Initialize lists to store arrival times
    T_P = []
    T_S = []
    e_ids = []
    station_names = []

    # Extract arrival times for each event in the catalog
    for event in catalog:
        if event.event_type not in ["earthquake", "quarry blast", "explosion"]:
            continue
        origin = event.preferred_origin()
        if origin is None:
            logger.warning(f"No preferred origin found for {event.resource_id}.")
            continue

        T0 = origin.time  # Origin time in seconds

        # Use defaultdict to store arrivals per station
        stations_arrivals = defaultdict(dict)

        for a in origin.arrivals:
            if a.time_weight == 0:
                continue

            pick = get_pick_from_arrival(event, a)
            if pick is None:
                continue

            station_name = get_station_name_from_pick(pick)

            # Store arrival times according to the phase
            if a.phase == "Pg":
                stations_arrivals[station_name]["P"] = pick.time
            elif a.phase == "Sg":
                stations_arrivals[station_name]["S"] = pick.time

        # Calculate T_P and T_S times for each station
        for arrivals in stations_arrivals.values():
            P_arrival = arrivals.get("P")
            S_arrival = arrivals.get("S")

            if P_arrival is not None and S_arrival is not None:
                T_P.append(P_arrival - T0)
                T_S.append(S_arrival - T0)
                e_ids.append(event.resource_id.id)
                station_names.append(station_name)

    # Convert lists to numpy arrays
    T_P = np.array(T_P)
    T_S = np.array(T_S)
    e_ids = np.array(e_ids)
    station_names = np.array(station_names)

    # Check if there are enough data points for regression
    if len(T_P) < 2:
        logger.warning("Not enough data to estimate Vp/Vs ratio.")
        return [], [], [], []

    return T_P, T_S, e_ids, station_names


def get_wadati_times_from_db(conn: sqlite3.Connection, event_id: str) -> tuple:
    """
    Retrieve the arrival times of P and S waves for a given event,
    considering only the preferred origin.

    Args:
        conn: SQLite connection
        event_id (str): ID of the event.

    Returns:
        tuple:
            Two lists containing the P and S arrival times relative to the origin time,
            the event ID and station names.
    """

    cursor = conn.cursor()

    # Get the preferred origin
    cursor.execute(
        """
        SELECT id, time
        FROM origins
        WHERE event_id = ? AND preferred = 1
        """,
        (event_id,),
    )
    result = cursor.fetchone()
    if result is None:
        logger.warning(f"No preferred origin found for {event_id}.")
        return [], [], [], []

    preferred_origin_id, T0 = result
    T0 = UTCDateTime(str(T0))  # Convert to UTCDateTime

    # Get arrivals and picks in a single query
    cursor.execute(
        """
        SELECT p.id, p.station_name, p.pick_time, a.name
        FROM arrivals AS a
        JOIN picks AS p ON a.pick_id = p.id
        WHERE a.origin_id = ? AND a.time_weight > 0
        """,
        (preferred_origin_id,),
    )

    arrivals = defaultdict(dict)

    # Fill in P and S phases
    for pick_id, station_name, pick_time, arrival_name in cursor.fetchall():
        pick_time = UTCDateTime(str(pick_time))

        if arrival_name == "Pg":
            arrivals[station_name]["P"] = pick_time
        elif arrival_name == "Sg":
            arrivals[station_name]["S"] = pick_time

    # Calculate relative times
    T_P, T_S, evt_ids, sta_names = [], [], [], []
    for station, phases in arrivals.items():
        P_arrival = phases.get("P")
        S_arrival = phases.get("S")
        if P_arrival is not None and S_arrival is not None:
            T_P.append(P_arrival - T0)
            T_S.append(S_arrival - T0)
            evt_ids.append(event_id)
            sta_names.append(station)

    return T_P, T_S, evt_ids, sta_names


def wadati_db(db_filename: str, polygon_wkt: str, use_tqdm: bool = True) -> tuple:
    """
    Compute the Vp/Vs ratio using the Wadati method for events
    located within a given polygon.

    Args:
        db_filename (str): Path to the SQLite/SpatiaLite database.
        polygon_wkt (str): Polygon in WKT (Well-Known Text) format to filter events.

    Returns:
        tuple: (T_P, T_S, evt_ids, station_names).
    """
    conn = sqlite3.connect(db_filename)
    cursor = conn.cursor()

    # Ensure SpatiaLite is enabled
    conn.enable_load_extension(True)
    cursor.execute("SELECT load_extension('mod_spatialite');")

    # Retrieve event_ids located within the polygon
    cursor.execute(
        """
        SELECT DISTINCT event_id
        FROM event_coordinates AS e
        WHERE e.event_type IN ('earthquake', 'quarry blast', 'explosion')
            AND ST_Contains(GeomFromText(?), geometry)
        """,
        (polygon_wkt,),
    )
    event_ids = [row[0] for row in cursor.fetchall()]

    if not event_ids:
        logger.warning("No events found in the specified area.")
        return [], [], [], []

    T_P = []
    T_S = []
    e_ids = []
    station_names = []

    for event_id in tqdm(event_ids, desc="Processing events", disable=not use_tqdm):
        tp, ts, evt_ids, sta_names = get_wadati_times_from_db(conn, event_id)
        T_P.extend(tp)
        T_S.extend(ts)
        e_ids.extend(evt_ids)
        station_names.extend(sta_names)

    # Convert to numpy arrays
    T_P = np.array(T_P)
    T_S = np.array(T_S)
    e_ids = np.array(e_ids)
    station_names = np.array(station_names)

    if len(T_P) < 2:
        logger.warning("Not enough data to estimate the Vp/Vs ratio.")
        return [], [], [], []

    return T_P, T_S, e_ids, station_names


def filter_basic_outliers(
    T_P: np.ndarray,
    T_S: np.ndarray,
    event_ids: np.ndarray,
    station_names: np.ndarray,
    t=10,
) -> tuple:
    """
    Filter out outliers from the P and S arrival times.

    Args:
        T_P (np.ndarray): Array of P-wave arrival times.
        T_S (np.ndarray): Array of S-wave arrival times.

    Returns:
        tuple: Filtered arrays of P and S arrival times.
    """

    # filter data to remove outliers
    df = pd.DataFrame(
        {"T_P": T_P, "T_S": T_S, "event_id": event_ids, "station": station_names}
    ).astype({"T_P": float, "T_S": float, "event_id": str, "station": str})

    df_outliers = df[
        ~((df["T_P"] <= 200) & (df["T_S"] <= 200) & (df["T_P"] > 0) & (df["T_S"] > 0))
    ]
    show_dataframes(df_outliers, "Basic Outliers")

    df = df[(df["T_P"] <= 200) & (df["T_S"] <= 200)]
    df = df[(df["T_P"] > 0) & (df["T_S"] > 0)]

    T_P = df["T_P"].values
    T_S = df["T_S"].values
    event_ids = df["event_id"].values
    station_names = df["station"].values

    return T_P, T_S, event_ids, station_names


def get_outliers(
    T_P: np.ndarray,
    T_S: np.ndarray,
    event_ids: np.ndarray,
    station_names: np.ndarray,
    slope: float,
    intercept: float,
) -> tuple:
    """
    Get the outliers from the residuals of the linear regression.
    """
    df = pd.DataFrame(
        {"T_P": T_P, "T_S": T_S, "event_id": event_ids, "station": station_names}
    ).astype({"T_P": float, "T_S": float, "event_id": str, "station": str})

    # Calculate predicted values and residuals
    df["T_S_pred"] = slope * df["T_P"] + intercept
    df["residuals"] = df["T_S"] - df["T_S_pred"]

    # Detect outliers using IQR
    Q1 = df["residuals"].quantile(0.05)
    Q3 = df["residuals"].quantile(0.95)
    IQR = Q3 - Q1

    # Define bounds for outliers (1.5 * IQR)
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Filter outliers
    outliers = df[(df["residuals"] < lower_bound) | (df["residuals"] > upper_bound)]
    show_dataframes(outliers, "Outliers based on residuals")

    return outliers


# get regressed Vp/Vs ratio from the Wadati diagram
def get_regressed_vp_vs(T_P: np.ndarray, T_S: np.ndarray) -> tuple:
    """
    Get the regressed Vp/Vs ratio from the Wadati diagram.

    Args:
        T_P (np.ndarray): Array of P-wave arrival times.
        T_S (np.ndarray): Array of S-wave arrival times.

    Returns:
        tuple: (slope, intercept, r_value).
    """
    # Linear regression to estimate the Vp/Vs ratio
    slope, intercept, r_value, _, _ = linregress(T_P, T_S)
    return slope, intercept, r_value


def wadati_plot(
    T_P: np.ndarray,
    T_S: np.ndarray,
    slope: float = None,
    intercept: float = None,
    output: str = None,
    title: str = "Wadati diagram",
    outliers: pd.DataFrame = None,
) -> None:
    """
    Plot a Wadati diagram from given P and S arrival times.

    Args:
        T_P (np.ndarray): Array of P-wave arrival times relative to origin time.
        T_S (np.ndarray): Array of S-wave arrival times relative to origin time.
        slope (float): Estimated Vp/Vs ratio (optional).
        r_value (float): Coefficient of determination (optional).
        output (str): Path to the output image file (optional).
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(T_P, T_S, label="Observations", color="blue")
    if outliers is not None:
        plt.scatter(outliers["T_P"], outliers["T_S"], label="Observations", color="red")
    plt.plot(
        T_P,
        slope * T_P + intercept,
        color="red",
        label=f"Linear regression (Vp/Vs = {slope:.2f})",
    )
    plt.xlabel("T_P - T0 (s)")
    plt.ylabel("T_S - T0 (s)")
    plt.title(title)
    plt.legend()
    plt.grid()

    if output is not None:
        plt.savefig(output)
    else:
        plt.show()


def process_database(args):
    """Process SQLite database and extract Wadati parameters."""
    polygons = read_polygons_from_yaml(args.config)
    polygon_name = args.polygon if args.polygon else ""
    polygon_wkt = "POLYGON(())"
    use_tqdm = args.use_tqdm

    if args.polygon:
        i = get_polygon_index(polygons, args.polygon)
        if i == -1:
            logger.error(f"Polygon '{args.polygon}' not found in polygons.yaml.")
            sys.exit(1)
        polygon_center = polygons[i]["center"]
        polygon_wkt = generate_wkt_from_polygon(polygons[i]["coordinates"])
    else:
        polygon_center = (None, None)

    return (
        wadati_db(args.database, polygon_wkt=polygon_wkt, use_tqdm=use_tqdm),
        polygon_name,
        polygon_center,
    )


def process_quakeml(args):
    """Process QuakeML file and extract Wadati parameters."""
    logger.info(f"Reading QuakeML file: {args.input}")
    try:
        catalog = read_events(args.input)
    except FileNotFoundError:
        logger.error(f"File not found: {args.input}")
        sys.exit(1)

    logger.info(f"Catalog contains {len(catalog)} events.")
    return wadati_catalog(catalog), "", (None, None)


def write_results_to_csv(
    polygon_name, polygon_center, slope, r_value, nb_events, num_points
):
    """Write Wadati results to a CSV file."""
    csv_output_filename = f"{polygon_name}.csv"
    if os.path.exists(csv_output_filename):
        logger.error(f"File {csv_output_filename} already exists.")
        return

    with open(csv_output_filename, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "Longitude",
                "Latitude",
                "Vp/Vs Ratio",
                "R^2",
                "Number of Events",
                "Number of Points",
            ]
        )
        writer.writerow(
            [
                polygon_center[0],
                polygon_center[1],
                f"{slope:.2f}",
                f"{r_value**2:.3f}",
                nb_events,
                num_points,
            ]
        )


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Plot a Wadati diagram from a QuakeML file or SQLite3."
    )
    parser.add_argument(
        "-c",
        "--config",
        required=True,
        help="Path to the YAML configuration file containing polygons",
    )
    parser.add_argument("-i", "--input", help="Path to the QuakeML catalog file")
    parser.add_argument("-d", "--database", help="Path to the SQLite3 database file")
    parser.add_argument("-o", "--output", help="Path to the output image file")
    parser.add_argument("-p", "--polygon", help="Title of the polygon in polygons.yaml")
    parser.add_argument(
        "--use-tqdm",
        action="store_true",
        help="Use tqdm for progress bar (default: False)",
    )
    return parser.parse_args()


def validate_arguments(args):
    """Validate command-line arguments and return the chosen input source."""
    if not args.database and not args.input:
        logger.error("Please provide a QuakeML file or an SQLite database.")
        polygons = read_polygons_from_yaml(args.config)
        for p in polygons:
            print(p.get("title"))
        sys.exit(1)

    if args.database and args.input:
        logger.error(
            "Please provide either a QuakeML file or an SQLite database, not both."
        )
        sys.exit(1)

    return "database" if args.database else "quakeML"


def main():
    """Main execution function."""
    args = parse_arguments()
    input_type = validate_arguments(args)

    if input_type == "database":
        (
            (T_P, T_S, evt_ids, sta_names),
            polygon_name,
            polygon_center,
        ) = process_database(args)
    else:
        (
            (T_P, T_S, evt_ids, sta_names),
            polygon_name,
            polygon_center,
        ) = process_quakeml(args)

    T_P, T_S, evt_ids, sta_names = filter_basic_outliers(T_P, T_S, evt_ids, sta_names)

    if T_P is None:
        logger.warning("Not enough data to estimate Vp/Vs ratio.")
        sys.exit(1)

    slope, intercept, r_value = get_regressed_vp_vs(T_P, T_S)
    outliers = get_outliers(T_P, T_S, evt_ids, sta_names, slope, intercept)

    logger.info(
        f"Estimated Vp/Vs ratio: {slope:.2f} (R²={r_value**2:.3f}), #events={len(set(evt_ids))}, #points={len(T_P)}"
    )

    write_results_to_csv(
        polygon_name, polygon_center, slope, r_value, len(set(evt_ids)), len(T_P)
    )

    wadati_plot(
        T_P,
        T_S,
        slope=slope,
        intercept=intercept,
        output=args.output,
        title=f"Wadati diagram [{polygon_name} / {len(set(evt_ids))} events]",
        outliers=outliers,
    )


if __name__ == "__main__":
    main()
