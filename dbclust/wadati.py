#!/usr/bin/env python3
import argparse
import logging
import sqlite3
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from icecream import ic
from obspy.core.event import Arrival
from obspy.core.event import Catalog
from obspy.core.event import Event
from obspy.core.event import Pick
from obspy.core.event import read_events
from obspy.core.utcdatetime import UTCDateTime
from scipy.stats import linregress
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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


def wadati(catalog: Catalog):
    # Initialize lists to store arrival times
    T_P = []
    T_S = []

    # Extract arrival times for each event in the catalog
    for event in catalog:
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

    # Convert lists to numpy arrays
    T_P = np.array(T_P)
    T_S = np.array(T_S)

    # Check if there are enough data points for regression
    if len(T_P) < 2:
        logger.warning("Not enough data to estimate Vp/Vs ratio.")
        return None, None, None, None, None

    # Perform linear regression to estimate the slope (Vp/Vs)
    slope, intercept, r_value, _, _ = linregress(T_P, T_S)

    return T_P, T_S, slope, intercept, r_value


def get_wadati_times_from_db(conn, event_id: str):
    """
    Retrieve the arrival times of P and S waves for a given event,
    considering only the preferred origin.

    Args:
        conn: Open SQLite connection to avoid repeated openings.
        event_id (str): ID of the event.

    Returns:
        tuple: Two lists containing the P and S arrival times relative to the origin time.
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
        return [], []

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
    T_P, T_S = [], []
    for station, phases in arrivals.items():
        P_arrival = phases.get("P")
        S_arrival = phases.get("S")
        if P_arrival is not None and S_arrival is not None:
            T_P.append(P_arrival - T0)
            T_S.append(S_arrival - T0)

    return T_P, T_S


def wadati_db(db_filename: str):
    """
    Perform a Wadati analysis on all events.

    Args:
        db_filename (str): Path to the SQLite database file.

    Returns:
        tuple: Data (T_P, T_S, slope, intercept, r_value) for the Vp/Vs ratio.
    """

    conn = sqlite3.connect(db_filename)

    # ⚡ Enable SQLite optimizations
    conn.execute("PRAGMA synchronous = OFF")
    conn.execute("PRAGMA journal_mode = MEMORY")
    conn.execute("PRAGMA temp_store = MEMORY")

    # Retrieve events with preferred origin
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT event_id FROM origins WHERE preferred = 1")
    event_ids = [row[0] for row in cursor.fetchall()]

    logger.info(f"Catalog contains {len(event_ids)} events.")

    T_P, T_S = [], []

    # Process all events
    for event_id in tqdm(event_ids, desc="Processing events"):
        tp, ts = get_wadati_times_from_db(conn, event_id)
        T_P.extend(tp)
        T_S.extend(ts)

    conn.close()

    # ⚡ NumPy only at the end
    T_P = np.array(T_P, dtype=np.float64)
    T_S = np.array(T_S, dtype=np.float64)

    if len(T_P) < 2:
        logger.warning("Not enough data to estimate Vp/Vs.")
        return None, None, None, None, None

    # ⚡ Optimized linear regression
    slope, intercept, r_value, _, _ = linregress(T_P, T_S)

    return T_P, T_S, slope, intercept, r_value


def wadati_plot(T_P, T_S, slope=None, intercept=None, output: str = None):
    """
    Plot a Wadati diagram from given P and S arrival times.

    Args:
        T_P (np.ndarray): Array of P-wave arrival times relative to origin time.
        T_S (np.ndarray): Array of S-wave arrival times relative to origin time.
        slope (float): Estimated Vp/Vs ratio (optional).
        r_value (float): Coefficient of determination (optional).
        output (str): Path to the output image file (optional).
    """

    # Check if we have enough points
    if len(T_P) < 2:
        print("Not enough data to plot a Wadati diagram.")
        exit()

    # Linear regression to estimate the slope (Vp/Vs)
    if slope is None or intercept is None:
        slope, intercept, r_value, _, _ = linregress(T_P, T_S)

    # Plot the Wadati diagram
    plt.figure(figsize=(8, 6))
    plt.scatter(T_P, T_S, label="Observations", color="blue")
    plt.plot(
        T_P,
        slope * T_P + intercept,
        color="red",
        label=f"Linear regression (Vp/Vs = {slope:.2f})",
    )
    plt.xlabel("T_P - T0 (s)")
    plt.ylabel("T_S - T0 (s)")
    plt.title("Wadati Diagram")
    plt.legend()
    plt.grid()

    if output is not None:
        plt.savefig(output)
    else:
        plt.show()


if __name__ == "__main__":
    # Argument parser configuration
    parser = argparse.ArgumentParser(
        description="Plot a Wadati diagram from a QuakeML file."
    )
    parser.add_argument(
        "-i", "--input", help="Path to the QuakeML catalog file (optional)"
    )
    parser.add_argument(
        "-d", "--database", help="Path to the SQLite database file (optional)"
    )

    parser.add_argument(
        "-o", "--output", help="Path to the output image file (optional)"
    )
    args = parser.parse_args()

    if not args.database and not args.input:
        logger.error("Please provide a QuakeML file or a SQLite database.")
        exit()

    if args.database and args.input:
        logger.error("Please provide either a QuakeML file or a SQLite database.")
        exit()

    if args.database is not None:
        T_P, T_S, slope, intercept, r_value = wadati_db(args.database)
    else:
        logger.info(f"Reading QuakeML file: {args.input}")
        try:
            catalog = read_events(args.input)
        except FileNotFoundError:
            logger.error(f"File not found: {args.input}")
            exit()
        logger.info(f"Catalog contains {len(catalog)} events.")
        # Extract P and S arrival times and estimate Vp/Vs ratio
        T_P, T_S, slope, intercept, r_value = wadati(catalog)

    if T_P is None:
        logger.warning("Not enough data to estimate Vp/Vs ratio.")
        exit()

    logger.info(
        f"Estimated Vp/Vs ratio: {slope:.2f} (R²={r_value**2:.3f}), len(T_P)={len(T_P)}"
    )

    # Plot the Wadati diagram
    wadati_plot(T_P, T_S, slope=slope, intercept=intercept, output=args.output)
