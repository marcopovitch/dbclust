#!/usr/bin/env python
"""
Processes QuakeML files and stores the data in a SpatiaLite-enabled SQLite database.
"""
import argparse
import csv
import json
import logging
import math
import os
import re
import sqlite3
import sys
import time
import warnings
import xml.etree.ElementTree as ET
import zlib
from datetime import datetime
from io import BytesIO
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
import pandas as pd
from icecream import ic
from obspy import Catalog
from obspy import read_events
from obspy import UTCDateTime
from obspy.core.event import Arrival
from obspy.core.event import Event
from obspy.core.event import Magnitude
from obspy.core.event import Origin
from tqdm import tqdm

from dbclust.gt5 import compute_gt5_score
from dbclust.localization_quality import classify_Michele_mod

# Suppress UserWarnings in ObsPy
warnings.filterwarnings("ignore", category=UserWarning, module="obspy")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("inject_spatialite")
logger.setLevel(logging.INFO)

# Define ANSI escape code constants for colors
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
MAGENTA = "\033[95m"
CYAN = "\033[96m"
WHITE = "\033[97m"
RESET = "\033[0m"  # Reset to default color


# SQL for creating the event coordinates view
EVENT_COORDINATES_VIEW = """
    CREATE VIEW IF NOT EXISTS event_coordinates AS
    SELECT
        e.event_id,
        o.time,
        o.latitude, o.longitude,
        o.depth / 1000.0 AS depth_km,
        o.rms,
        o.erh AS erh_km,
        o.erz AS erz_km,
        o.er_method,
        o.method_id AS location_method_id,
        o.earth_model_id,
        e.nb_origins,
        e.nb_magnitudes,
        m.magnitude,
        m.magnitude_type,
        m.uncertainty AS magnitude_uncertainty,
        o.used_station_count, o.used_phase_count, o.P_count, o.S_count,
        o.minimum_distance AS minimum_distance_deg,
        o.maximum_distance AS maximum_distance_deg,
        o.median_distance AS median_distance_deg,
        o.azimuthal_gap, o.secondary_azimuthal_gap,
        o.expectation_latitude, o.expectation_longitude,
        o.expectation_depth / 1000.0 AS expectation_depth_km,
        o.scatter_volume,
        e.dist_km_from_preloc AS dist_from_preloc_km,
        e.nb_agencies, e.agencies_list, e.agency_names, e.multiple_same_agencies,
        o.evaluation_mode,
        e.event_type,
        e.discrimination_probability,
        e.discrimination_station_count,
        e.discrimination_certainty,
        o.quality, o.quality_factor,
        o.gt5_status, o.delta_U, o.num_stations_10km, o.num_stations_30km, o.num_stations_150km,
        o.geometry
    FROM
        events AS e
    JOIN
        origins AS o
        ON e.event_id = o.event_id AND o.preferred = 1  -- INNER JOIN because a preferred origin always exists
    LEFT JOIN
        magnitudes AS m
        ON e.event_id = m.event_id AND m.preferred = 1
    WHERE
        COALESCE(e.event_type, '') NOT IN ('not existing', 'not locatable');
"""

RELABELING_VIEW = """
    CREATE VIEW IF NOT EXISTS relabeling AS
    SELECT
        a.id AS arrival_id,
        a.origin_id,
        a.pick_id,
        p.station_name,
        a.name as station,
        a.time_weight,
        a.time_residual,
        p.evaluation_mode,
        p.probability,
        a.distance,
        a.relabel_action,
        a.relabel_previous_phase,
        a.relabel_evaluation_score,
        a.relabel_scores
    FROM
        arrivals a
    JOIN
        origins o ON a.origin_id = o.id
    JOIN
        picks p ON a.pick_id = p.id
    WHERE
        o.preferred = TRUE;
"""


def get_event_agencies_ids(event: Event) -> list:
    """
    Extracts and returns a list of event agency IDs from the comments of a given event.
    Args:
        event (Event): An event object containing comments with potential agency IDs.
    Returns:
        list: A list of agency IDs extracted from the event's comments.
    """
    ids = list()
    for comment in event.comments:
        try:
            info = json.loads(comment.text)
        except:
            continue

        if "event_ids" in info.keys():
            ids.extend(info["event_ids"])
    return ids


def get_distance_km_info(event: Event) -> float:
    """
    Extracts from the event's comment the distance from preloc to preferred origin in kilometers.
    Args:
        event (Event): An event object containing comments with potential distance information.
    Returns:
        float or None: The distance in kilometers if found, otherwise None.
    """
    dist_km = None
    for comment in event.comments:
        try:
            info = json.loads(comment.text)
        except:
            continue
        if "preloc_distance_km" in info.keys():
            dist_km = info["preloc_distance_km"]
            break
    return dist_km


def get_scatter_volume(origin: Origin) -> float:
    """
    Extracts from the origin's comment the scatter volume.

    Args:
        origin (Origin): An origin object containing comments with potential scatter volume information.
    Returns:
        float or None: The scatter volume if found, otherwise None.
    """
    scatter_volume = None
    for comment in origin.comments:
        try:
            info = json.loads(comment.text)
        except:
            continue
        if "scatter_volume" in info.keys():
            scatter_volume = info["scatter_volume"]
            break
    return scatter_volume


def get_expectation_localization(origin: Origin) -> Tuple[float, float, float]:
    """
    Extracts from the origin's comment the expectation localization.

    Args:
        origin (Origin): An origin object containing comments with potential expectation localization information.
    Returns:
        Tuple[float, float, float] or None: The expectation localization if found, otherwise None.
    """
    expectation_latitude = None
    expectation_longitude = None
    expectation_depth = None
    for comment in origin.comments:
        try:
            info = json.loads(comment.text)
        except:
            continue

        data = info.get("expectation")
        if data:
            expectation_latitude = data.get("latitude")
            expectation_longitude = data.get("longitude")
            expectation_depth = data.get("depth") * 1000.0
            break
    return expectation_latitude, expectation_longitude, expectation_depth


def get_pick_probability(pick):
    """
    Extracts the probability of the pick from the pick's comment.

    Args:
        pick (Pick): A pick object containing comments with potential probability information.

    Returns:
        float or None: The probability if found, otherwise None.
    """
    probability = None
    for comment in pick.comments:
        try:
            info = json.loads(comment.text)
        except:
            continue
        if "probability" in info.keys():
            probability = info["probability"]["value"]
            break
    return probability


def get_erh_erz(origin: Origin) -> Tuple[float, float, str]:
    """
    Calculate the values of erh (horizontal uncertainty) and erz (vertical uncertainty) in km.

    Parameters:
        origin (Origin): The origin.

    Returns:
        Tuple[float, float, str]: The values of erh and erz, and the method used to compute them.
    """
    for comment in origin.comments:
        text = comment.text
        match = re.search(r"CovXX (\d+\.\d+) .* YY (\d+\.\d+) .* ZZ (\d+\.\d+)", text)
        if text and match:
            CovXX = float(match.group(1))
            CovYY = float(match.group(2))
            ZZ = float(match.group(3))

            erz = np.sqrt(ZZ)
            erh = np.sqrt(CovXX + CovYY)
            method = "covariance"
            return erh, erz, method

    # compute erh and erz from origin errors
    earth_radius = 6371.0
    deg_latitude_km = earth_radius * math.pi / 180.0
    deg_longitude_km = (
        earth_radius * math.pi / 180.0 * math.cos(math.radians(origin.latitude))
    )

    try:
        erh = np.sqrt(
            (origin.latitude_errors.uncertainty * deg_latitude_km) ** 2
            + (origin.longitude_errors.uncertainty * deg_longitude_km) ** 2
        )
        method = "origin_errors"
    except:
        try:
            erh = origin.origin_uncertainty.horizontal_uncertainty / 1000.0
            method = "origin_uncertainty"
        except:
            erh = None
            method = "unknown"

    try:
        erz = origin.depth_errors.uncertainty / 1000.0
        method = "origin_errors"
    except:
        try:
            erz = origin.origin_uncertainty.depth_uncertainty / 1000.0
            method = "origin_uncertainty"
        except:
            erz = None
            method = "unknown"

    return erh, erz, method


def get_relabel_info(
    arrival: Arrival,
) -> Tuple[Optional[str], Optional[str], Optional[float], Dict[str, float]]:
    """Extract relabeling information from an Arrival object.

    Args:
        arrival: An Arrival object containing the information.

    Returns:
        A tuple (action, previous_phase, evaluation_score, scores)

    Examples:
        >>> arrival = Arrival()
        >>> arrival.comments = [Comment(text='{"relabel": {"prev_phase": "P", "action": "score too low", "eval_score": "1.0024", "scores": {"Pg": "0.5006", "Sg": "0.4994"}}}')]
        >>> get_relabel_info(arrival)
        ('score too low', 'P', 1.0024, {'Pg': 0.5006, 'Sg': 0.4994})
    """
    if not arrival or not hasattr(arrival, "comments"):
        return None, None, None, {}

    for comment in arrival.comments:
        try:
            relabel_data = json.loads(comment.text)
            if isinstance(relabel_data, dict) and "relabel" in relabel_data:
                relabel_info = relabel_data["relabel"]

                action = relabel_info.get("action")
                previous_phase = relabel_info.get("prev_phase")
                evaluation_score = (
                    float(relabel_info["eval_score"])
                    if "eval_score" in relabel_info
                    else None
                )
                scores = {
                    k: float(v) for k, v in relabel_info.get("scores", {}).items()
                }
                return action, previous_phase, evaluation_score, scores

        except (json.JSONDecodeError, TypeError, ValueError) as e:
            continue

    return None, None, None, {}


def phase_count(event: Event, origin: Origin, phase_type: str) -> int:
    """
    Count the number of phase of a given type in an origin.

    Parameters:
        origin (Origin): The origin.
        phase_type (str): The phase type "P" or "S".

    Returns:
        int: The number of phase of the given type.
    """
    count = 0
    for arrival in origin.arrivals:
        pick = next((p for p in event.picks if p.resource_id == arrival.pick_id), None)

        if phase_type in pick.phase_hint.upper():
            count += 1

    return count


def compress_quakeml_data(catalog: Catalog, format: str = "QUAKEML"):
    """
    Compresses a QuakeML catalog using zlib compression.

    Parameters:
        catalog (Catalog): The QuakeML catalog to be compressed.
        format (str): The format of the catalog, default is "QUAKEML".

    Returns:
        bytes: The compressed catalog data.
    """
    buffer = BytesIO()
    catalog.write(buffer, format=format)
    raw_quakeml = buffer.getvalue()
    buffer.close()
    compressed_data = zlib.compress(raw_quakeml)
    return compressed_data


def to_datetime(utc_datetime: UTCDateTime) -> datetime:
    """
    Convert a UTCDateTime object to a datetime object.

    Parameters:
        utc_datetime (UTCDateTime or datetime): The input object to be converted.

    Returns:
        datetime:
            converted datetime object if the input is a UTCDateTime object,
            otherwise returns the input as is.
    """
    return (
        utc_datetime.datetime if isinstance(utc_datetime, UTCDateTime) else utc_datetime
    )


def execute_with_retry(conn, operation, retries=5, delay=0.1):
    """Execute an operation with retry logic in case of database lock."""
    while retries > 0:
        try:
            operation()
            return
        except sqlite3.OperationalError as e:
            if "locked" in str(e):
                retries -= 1
                time.sleep(delay)
            else:
                raise
    raise sqlite3.OperationalError("Database is locked after multiple attempts")


def inject_event(conn: sqlite3.Connection, event: Event, quakeml: str) -> None:
    """
    Injects an earthquake event and its related data into a SpatiaLite-enabled SQLite database.

    Parameters:
        conn (sqlite3.Connection): The SQLite database connection object.
        event (Event): The earthquake event containing event details.
        quakeml (str): The QuakeML XML string representing the event.

    Raises:
        Exception: If any database operation fails, the transaction is rolled back and the exception is raised.
    """
    conn.enable_load_extension(True)
    conn.load_extension("mod_spatialite")

    def insert_event_data():
        with conn:
            logger.debug(f"Inserting event {event.resource_id.id}.")

            # Insert full QuakeML data
            logger.debug(f"Inserting QuakeML data for event {event.resource_id.id}.")

            conn.execute(
                """
                INSERT INTO quakeml (event_id, data)
                VALUES (?, ?)
                """,
                (event.resource_id.id, quakeml),
            )

            # Insert event metadata
            all_agencies_ids = get_event_agencies_ids(event)
            agencies_list_str = json.dumps(all_agencies_ids)
            logger.debug(f"Agencies list: {agencies_list_str}")

            conn.execute(
                """
                INSERT INTO events (event_id, event_type, dist_km_from_preloc, nb_agencies, agencies_list, nb_origins, nb_magnitudes)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.resource_id.id,
                    event.event_type,
                    get_distance_km_info(event),
                    len(all_agencies_ids),
                    agencies_list_str,
                    len(event.origins),
                    len(event.magnitudes),
                ),
            )

            # Insert origins
            logger.debug(f"Inserting origins for event {event.resource_id.id}.")
            for origin in event.origins:
                insert_origin(conn, origin, event)

            # Insert picks
            logger.debug(f"Inserting picks for event {event.resource_id.id}.")
            cursor = conn.cursor()
            for pick in event.picks:
                agency_id = (
                    pick.creation_info.agency_id
                    if pick.creation_info and hasattr(pick.creation_info, "agency_id")
                    else None
                )
                probability = get_pick_probability(pick)

                # Check if the pick ID already exists, and to which event it belongs
                cursor.execute(
                    "SELECT id FROM picks WHERE id = ?", (pick.resource_id.id,)
                )
                if cursor.fetchone():
                    # Pick ID already exists in the database related to another event
                    # it is something possible in the case of close events in QuakeML-RT.
                    # This is something that should be handled in the future.

                    # get event_id  of the pick
                    cursor.execute(
                        "SELECT event_id FROM picks WHERE id = ?",
                        (pick.resource_id.id,),
                    )
                    event_id = cursor.fetchone()[0]
                    logger.warning(
                        f"[{event.resource_id.id}] ID '{pick.resource_id.id}' already exists in event {event_id}. Pick will not be inserted."
                    )
                    continue

                conn.execute(
                    """
                    INSERT INTO picks (
                        id, event_id, station_name, pick_time, uncertainty,
                        evaluation_mode, phase_hint, agency_id, probability)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        pick.resource_id.id,
                        event.resource_id.id,
                        f"{pick.waveform_id.network_code}.{pick.waveform_id.station_code}",
                        to_datetime(pick.time),
                        pick.time_errors.uncertainty,
                        pick.evaluation_mode,
                        pick.phase_hint,
                        agency_id,
                        probability,
                    ),
                )

            # Insert arrivals
            logger.debug(f"Inserting arrivals for event {event.resource_id.id}.")
            for origin in event.origins:
                insert_arrivals(conn, origin)

            # Insert magnitudes
            logger.debug(
                f"Inserting magnitude for event {event.resource_id.id} with origin_id {origin.resource_id.id}."
            )
            insert_magnitudes(conn, event)
            insert_station_magnitudes(conn, event)

            logger.debug(f"Event {event.resource_id.id} successfully inserted.")

    try:
        execute_with_retry(conn, insert_event_data)
    except Exception as e:
        logger.error(f"Failed to insert event {event.resource_id.id}: {e}")
        raise


def insert_origin(conn: sqlite3.Connection, origin: Origin, event: Event) -> None:
    """Inserts an origin into the database."""
    q = origin.quality
    rms = getattr(q, "standard_error", None)
    P_count = phase_count(event, origin, "P")
    S_count = phase_count(event, origin, "S")
    erz, erh, err_method = get_erh_erz(origin)
    num_stations_10km = sum(
        1
        for arrival in origin.arrivals
        if arrival.time_weight and arrival.distance * 111.11 <= 10
    )
    num_stations_30km = sum(
        1
        for arrival in origin.arrivals
        if arrival.time_weight and arrival.distance * 111.11 <= 30
    )
    num_stations_150km = sum(
        1
        for arrival in origin.arrivals
        if arrival.time_weight and arrival.distance * 111.11 <= 150
    )

    scatter_volume = get_scatter_volume(origin)
    expectation_latitude, expectation_longitude, expectation_depth = (
        get_expectation_localization(origin)
    )

    # Get only the relevant info from the origin method ID
    origin_method_id = getattr(getattr(origin, "method_id", None), "id", "unknown")
    if isinstance(origin_method_id, str) and "/" in origin_method_id:
        origin_method_id = origin_method_id.rsplit("/", 1)[-1]

    earth_model_id = getattr(getattr(origin, "earth_model_id", None), "id", "unknown")
    if isinstance(earth_model_id, str) and "/" in earth_model_id:
        earth_model_id = earth_model_id.rsplit("/", 1)[-1]

    # compute GT5 score
    try:
        gt5_status, gt5_details = compute_gt5_score(origin)
        delta_U = gt5_details.get("delta_U", None)
    except Exception as e:
        logger.debug(f"Error computing GT5 score: {e}")
        gt5_status = None
        delta_U = None
        gt5_details = None

    # fix azimuthal gap
    if (q.azimuthal_gap is None or q.azimuthal_gap == 0) and gt5_details:
        azimuthal_gap = gt5_details.get("primary_gap", None)
    else:
        azimuthal_gap = q.azimuthal_gap

    if (
        q.secondary_azimuthal_gap is None or q.secondary_azimuthal_gap == 0
    ) and gt5_details:
        secondary_azimuthal_gap = gt5_details.get("secondary_gap", None)
    else:
        secondary_azimuthal_gap = q.secondary_azimuthal_gap

    # compute Michele et al. quality factor
    try:
        quality_factor, quality = classify_Michele_mod(
            rms,
            erh,
            erz,
            q.used_phase_count,
            q.minimum_distance,
            q.median_distance,
            azimuthal_gap,
            secondary_azimuthal_gap,
            scatter_volume,
        )
    except Exception as e:
        logger.debug(f"Error classifying Michele mod: {e}")
        quality_factor = None
        quality = None

    conn.execute(
        """
        INSERT INTO origins (
            id, event_id, time, time_errors,
            latitude, longitude, depth, depth_type,
            rms, erh, erz, er_method,
            method_id,
            earth_model_id,
            used_station_count, used_phase_count, P_count, S_count,
            minimum_distance, maximum_distance, median_distance,
            azimuthal_gap, secondary_azimuthal_gap,
            expectation_latitude, expectation_longitude, expectation_depth,
            scatter_volume,
            quality, quality_factor,
            num_stations_10km,
            num_stations_30km,
            num_stations_150km,
            delta_U,
            gt5_status,
            evaluation_mode, preferred, geometry
        )
        VALUES (
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?, ?,
            ST_GeomFromText(?, 4326)
        )
        """,
        (
            origin.resource_id.id,
            event.resource_id.id,
            to_datetime(origin.time),
            origin.time_errors.uncertainty,
            origin.latitude,
            origin.longitude,
            origin.depth,
            origin.depth_type,
            rms,
            erh,
            erz,
            err_method,
            origin_method_id,
            earth_model_id,
            q.used_station_count,
            q.used_phase_count,
            P_count,
            S_count,
            q.minimum_distance,
            q.maximum_distance,
            q.median_distance,
            azimuthal_gap,
            secondary_azimuthal_gap,
            expectation_latitude,
            expectation_longitude,
            expectation_depth,
            scatter_volume,
            quality,
            quality_factor,
            num_stations_10km,
            num_stations_30km,
            num_stations_150km,
            delta_U,
            1 if gt5_status else 0,
            origin.evaluation_mode,
            1 if origin.resource_id == event.preferred_origin().resource_id else 0,
            f"POINT({origin.longitude} {origin.latitude})",
        ),
    )
    logger.debug(f"Origin {origin.resource_id.id} inserted.")


def insert_magnitudes(conn: sqlite3.Connection, event: Event) -> None:
    """Inserts all magnitudes into the database."""
    for magnitude in event.magnitudes:
        conn.execute(
            """
            INSERT INTO magnitudes (id, origin_id, event_id, magnitude, uncertainty, station_count, magnitude_type, evaluation_mode, method_id, preferred)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                magnitude.resource_id.id,
                magnitude.origin_id.id if magnitude.origin_id else None,
                event.resource_id.id,
                magnitude.mag,
                magnitude.mag_errors.uncertainty,
                magnitude.station_count,
                magnitude.magnitude_type,
                magnitude.evaluation_mode,
                magnitude.method_id.id,
                (
                    1
                    if event.preferred_magnitude()
                    and magnitude.resource_id == event.preferred_magnitude().resource_id
                    else 0
                ),
            ),
        )
        insert_station_magnitude_contributions(conn, magnitude)
        logger.debug(f"Magnitude {magnitude.resource_id.id} inserted.")


def insert_station_magnitudes(conn: sqlite3.Connection, event: Event) -> None:
    """Inserts all station magnitudes into the database."""
    for station_magnitude in event.station_magnitudes:
        conn.execute(
            """
            INSERT INTO station_magnitudes (id, origin_id, magnitude, uncertainty, magnitude_type, method_id, waveform_id)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                station_magnitude.resource_id.id,
                station_magnitude.origin_id.id,
                station_magnitude.mag,
                station_magnitude.mag_errors.uncertainty,
                station_magnitude.station_magnitude_type,
                station_magnitude.method_id.id,
                station_magnitude.waveform_id.get_seed_string(),
            ),
        )
        logger.debug(f"Station Magnitude {station_magnitude.resource_id.id} inserted.")


def insert_station_magnitude_contributions(
    conn: sqlite3.Connection, magnitude: Magnitude
) -> None:
    """Inserts all station magnitude contribution for a given magnitude into the database."""

    for station_magnitude_contribution in magnitude.station_magnitude_contributions:
        conn.execute(
            """
            INSERT INTO station_magnitude_contributions (id, residual, weight)
            VALUES (?, ?, ?)
            """,
            (
                station_magnitude_contribution.station_magnitude_id.id,
                station_magnitude_contribution.residual,
                station_magnitude_contribution.weight,
            ),
        )
        logger.debug(
            f"Station Magnitude Contribution {station_magnitude_contribution.station_magnitude_id.id} inserted."
        )


def insert_arrivals(conn: sqlite3.Connection, origin: Origin) -> None:
    """Inserts arrivals associated with an origin into the database."""
    for arrival in origin.arrivals:
        pick_id = arrival.pick_id.id if arrival.pick_id else None
        action, previous_phase, evaluation_score, scores = get_relabel_info(arrival)
        conn.execute(
            """
            INSERT INTO arrivals (id, origin_id, pick_id, name, time_weight, time_residual,
                takeoff_angle, azimuth, distance,
                relabel_action, relabel_previous_phase, relabel_evaluation_score, relabel_scores)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                arrival.resource_id.id,
                origin.resource_id.id,
                pick_id,
                arrival.phase,
                arrival.time_weight,
                arrival.time_residual,
                arrival.takeoff_angle,
                arrival.azimuth,
                arrival.distance,
                action,
                previous_phase,
                evaluation_score,
                json.dumps(scores),
            ),
        )
        logger.debug(f"Arrival {arrival.resource_id.id} inserted.")


def export_sqlite_to_quakeml(
    db_path: str,
    output_file: str,
    event_ids: List[str] = None,
    start_time: str = None,
    end_time: str = None,
) -> None:
    """
    Concatenate multiple QuakeML streams stored in a database into a single XML file,
    minimizing memory usage by writing to the file incrementally.

    Args:
        db_path (str): Path to the SQLite database.
        output_file (str): Path to the output QuakeML file.
        event_ids (List[str], optional): List of event IDs to export. Defaults to None.
        start_time (str, optional): Start time for filtering events. Defaults to None.
        end_time (str, optional): End time for filtering events. Defaults to None.
    """
    # Connect to the SQLite database
    conn = sqlite3.connect(db_path)
    if not conn:
        raise Exception(f"Failed to connect to the database at {db_path}")
    cursor = conn.cursor()

    # Open the output file for writing
    with open(output_file, "wb") as f:
        # Write the XML declaration and root element for QuakeML
        f.write(b'<?xml version="1.0" encoding="utf-8"?>\n')
        f.write(
            b'<q:quakeml xmlns="http://quakeml.org/xmlns/bed/1.2" xmlns:q="http://quakeml.org/xmlns/quakeml/1.2">\n'
        )
        f.write(b"  <eventParameters>\n")

        # Define the namespace used in the QuakeML XML
        namespaces = {
            "": "http://quakeml.org/xmlns/bed/1.2",  # Default namespace for "quakeml"
        }

        # Iterate over the rows of QuakeML data in the database
        if event_ids:
            # Iterate through event IDs and query the database for each one
            for event_id in event_ids:
                query = "SELECT data FROM quakeml WHERE event_id = ?;"
                cursor.execute(query, (event_id,))
                rows = cursor.fetchall()
                if not rows:
                    print(f"No data found for event_id: {event_id}")
                for row in rows:
                    process_quakeml_row(row, f, namespaces)
        else:
            # If no event_ids are provided, process all data in the table
            # count the number of events
            count_query = (
                """
                SELECT COUNT(*) FROM event_coordinates as e WHERE e.time >= ? AND e.time < ?;
                """
                # """
                # SELECT COUNT(*) FROM event_coordinates as e WHERE e.time >= ? AND e.time < ? AND e.quality IN ('A', 'B', 'C');
                # """
                if start_time and end_time
                else "SELECT COUNT(*) FROM quakeml;"
            )
            cursor.execute(
                count_query, (start_time, end_time) if start_time and end_time else ()
            )
            count = cursor.fetchone()[0]
            print(f"Processing {count} events from {start_time} to {end_time}")

            query = (
                # """
                # SELECT q.data FROM quakeml q JOIN event_coordinates e ON q.event_id = e.event_id
                # WHERE e.time >= ? AND e.time < ? and e.quality IN ('A', 'B', 'C');
                # """
                """
                SELECT q.data FROM quakeml q JOIN event_coordinates e ON q.event_id = e.event_id
                WHERE e.time >= ? AND e.time < ?;
                """
                if start_time and end_time
                else "SELECT q.data FROM quakeml q;"
            )

            for row in cursor.execute(
                query, (start_time, end_time) if start_time and end_time else ()
            ):
                process_quakeml_row(row, f, namespaces)

        # Close the eventParameters tag and the root tag
        f.write(b"  </eventParameters>\n")
        f.write(b"</q:quakeml>\n")

    # Close the database connection
    conn.close()
    print(f"Concatenated QuakeML written to {output_file}")


def process_quakeml_row(row, output_file, namespaces):
    """
    Process a single row of QuakeML data and write events to the output file.

    Args:
        row (tuple): Row containing compressed QuakeML data.
        output_file (file): Open file object to write the events.
        namespaces (dict): XML namespaces to handle in the QuakeML data.
    """
    # Each row contains a compressed QuakeML
    compressed_quakeml_data = row[0]
    if compressed_quakeml_data is None:
        return

    # Decompress the QuakeML data
    quakeml_data = zlib.decompress(compressed_quakeml_data).decode("utf-8")

    try:
        # Parse the decompressed QuakeML data
        root = ET.fromstring(quakeml_data)

        # Find all <event> elements and process them
        for event in root.findall(".//event", namespaces):
            # Remove any namespace from the event element
            for elem in event.iter():
                if isinstance(elem.tag, str) and elem.tag.startswith("{"):
                    elem.tag = elem.tag.split("}", 1)[1]  # Remove namespace part

            # Write the event directly to the output file
            output_file.write(ET.tostring(event, encoding="utf-8"))

    except ET.ParseError as e:
        print(f"Error parsing QuakeML: {e}")


def create_schema(db_path: str) -> sqlite3.Connection:
    """
    Create the database schema for a SpatiaLite-enabled SQLite database.

    Args:
        db_path (str): The file path to the SQLite database.

    Returns:
        sqlite3.Connection: The connection object to the SQLite database.
    """
    try:
        conn = sqlite3.connect(db_path)
        if not conn:
            raise Exception(f"Failed to connect to the database at {db_path}")

        conn.execute(
            "PRAGMA journal_mode=WAL;"
        )  # Enable WAL mode for concurrent read/write
        conn.execute("PRAGMA foreign_keys = ON;")  # Enable foreign key constraints
        conn.enable_load_extension(True)

        try:
            conn.load_extension("mod_spatialite")
        except sqlite3.OperationalError as e:
            logger.error(f"Failed to load SpatiaLite extension: {e}")
            raise

        cursor = conn.cursor()

        # Initialize SpatiaLite metadata if not already initialized
        cursor.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='spatial_ref_sys';"
        )
        if cursor.fetchone()[0] == 0:
            logger.info(f"Initializing SpatiaLite metadata : {db_path}")
            cursor.execute("SELECT InitSpatialMetadata();")

        # Create tables
        create_tables(cursor)

        # Ensure 'geometry' column exists in 'origins' table
        cursor.execute("PRAGMA table_info(origins);")
        columns = [row[1] for row in cursor.fetchall()]
        if "geometry" not in columns:
            logger.info("Adding 'geometry' column to 'origins' table...")
            cursor.execute(
                "SELECT AddGeometryColumn('origins', 'geometry', 4326, 'POINT', 'XY');"
            )

        # Create spatial index if not exists
        cursor.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='idx_origins_geometry';"
        )
        if cursor.fetchone()[0] == 0:
            try:
                logger.info(
                    "Creating spatial index for 'geometry' column in 'origins' table..."
                )
                cursor.execute("SELECT CreateSpatialIndex('origins', 'geometry');")
            except sqlite3.OperationalError as e:
                logger.error(f"Error creating spatial index: {e}")

        # Create the event coordinates view
        cursor.execute(EVENT_COORDINATES_VIEW)
        logger.info("Database SpatiaLite schema created successfully.")

        return conn

    except Exception as e:
        logger.error(f"Error creating schema: {e}")
        raise


def create_tables(cursor: sqlite3.Cursor) -> None:
    """
    Create the required tables in the database.

    Args:
        cursor (sqlite3.Cursor): The database cursor.
    """
    logger.info("Creating database tables...")

    try:
        # Begin a transaction to speed up execution
        cursor.execute("BEGIN TRANSACTION;")

        # Table creation SQL
        tables_sql = [
            """
            CREATE TABLE IF NOT EXISTS quakeml (
                event_id TEXT PRIMARY KEY,
                data BLOB
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS events (
                event_id TEXT PRIMARY KEY REFERENCES quakeml(event_id) ON DELETE CASCADE,
                event_type TEXT,
                dist_km_from_preloc DOUBLE,
                discrimination_probability DOUBLE,
                discrimination_station_count INTEGER,
                discrimination_certainty DOUBLE,
                nb_agencies INTEGER,
                agencies_list JSON,
                agency_names TEXT,
                multiple_same_agencies BOOLEAN,
                nb_origins INTEGER,
                nb_magnitudes INTEGER
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS picks (
                id TEXT PRIMARY KEY,
                event_id TEXT REFERENCES events(event_id) ON DELETE CASCADE,
                station_name TEXT,
                pick_time TIMESTAMP,
                evaluation_mode TEXT,
                uncertainty DOUBLE,
                phase_hint TEXT,
                agency_id TEXT,
                probability DOUBLE
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS origins (
                id TEXT PRIMARY KEY,
                event_id TEXT REFERENCES events(event_id) ON DELETE CASCADE,
                time TIMESTAMP,
                time_errors DOUBLE,
                latitude DOUBLE,
                longitude DOUBLE,
                depth DOUBLE,
                depth_type TEXT,
                rms DOUBLE,
                erh DOUBLE,
                erz DOUBLE,
                er_method TEXT,
                method_id TEXT,
                earth_model_id TEXT,
                used_station_count INTEGER,
                used_phase_count INTEGER,
                P_count INTEGER,
                S_count INTEGER,
                minimum_distance DOUBLE,
                maximum_distance DOUBLE,
                median_distance DOUBLE,
                azimuthal_gap DOUBLE,
                secondary_azimuthal_gap DOUBLE,
                expectation_latitude DOUBLE,
                expectation_longitude DOUBLE,
                expectation_depth DOUBLE,
                scatter_volume DOUBLE,
                quality TEXT,
                quality_factor DOUBLE,
                num_stations_10km INTEGER,
                num_stations_30km INTEGER,
                num_stations_150km INTEGER,
                delta_U DOUBLE,
                gt5_status BOOLEAN,
                evaluation_mode TEXT,
                preferred BOOLEAN
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS arrivals (
                id TEXT PRIMARY KEY,
                origin_id TEXT REFERENCES origins(id) ON DELETE CASCADE,
                pick_id TEXT REFERENCES picks(id) ON DELETE CASCADE,
                name TEXT,
                time_weight DOUBLE,
                time_residual DOUBLE,
                takeoff_angle DOUBLE,
                azimuth DOUBLE,
                distance DOUBLE,
                relabel_action TEXT,
                relabel_previous_phase TEXT,
                relabel_evaluation_score DOUBLE,
                relabel_scores TEXT
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS magnitudes (
                id TEXT PRIMARY KEY,
                origin_id TEXT REFERENCES origins(id) ON DELETE CASCADE,
                event_id TEXT REFERENCES events(event_id) ON DELETE CASCADE,
                magnitude DOUBLE,
                uncertainty DOUBLE,
                station_count INTEGER,
                magnitude_type TEXT,
                evaluation_mode TEXT,
                method_id TEXT,
                preferred BOOLEAN
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS station_magnitudes (
                id TEXT PRIMARY KEY,
                origin_id TEXT REFERENCES origins(id) ON DELETE CASCADE,
                magnitude DOUBLE,
                uncertainty DOUBLE,
                magnitude_type TEXT,
                method_id TEXT,
                waveform_id TEXT
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS station_magnitude_contributions (
                id TEXT PRIMARY KEY,
                residual DOUBLE,
                weight DOUBLE
            );
            """,
        ]

        for sql in tables_sql:
            cursor.execute(sql)

        # Create indexes to speed up queries
        indexes_sql = [
            #
            "CREATE INDEX IF NOT EXISTS idx_events_event_id ON events(event_id);",
            #
            "CREATE INDEX IF NOT EXISTS idx_origins_id ON origins(id);",
            "CREATE INDEX IF NOT EXISTS idx_origins_event_id ON origins(event_id);",
            "CREATE INDEX IF NOT EXISTS idx_origins_preferred ON origins(preferred);",
            "CREATE INDEX IF NOT EXISTS idx_origins_event_preferred ON origins(event_id, preferred);",
            #
            "CREATE INDEX IF NOT EXISTS idx_magnitudes_event_id ON magnitudes(event_id);",
            "CREATE INDEX IF NOT EXISTS idx_magnitudes_preferred ON magnitudes(preferred);",
            "CREATE INDEX IF NOT EXISTS idx_magnitudes_event_preferred ON magnitudes(event_id, preferred);",
            "CREATE INDEX IF NOT EXISTS idx_magnitudes_origin_id ON magnitudes(origin_id);",
            "CREATE INDEX IF NOT EXISTS idx_station_magnitudes_origin_id ON station_magnitudes(origin_id);",
            #
            "CREATE INDEX IF NOT EXISTS idx_arrivals_pick_id ON arrivals(pick_id);",
            "CREATE INDEX IF NOT EXISTS idx_arrivals_origin_id ON arrivals(origin_id);",
            "CREATE INDEX IF NOT EXISTS idx_arrivals_time_weight ON arrivals(time_weight);",
            "CREATE INDEX IF NOT EXISTS idx_arrivals_origin_time_weight ON arrivals(origin_id, time_weight);",
            #
            "CREATE INDEX IF NOT EXISTS idx_picks_id ON picks(id);",
            "CREATE INDEX IF NOT EXISTS idx_picks_station_name ON picks(station_name);",
            "CREATE INDEX IF NOT EXISTS idx_picks_event_id ON picks(event_id);",
            "CREATE INDEX IF NOT EXISTS idx_picks_evaluation_mode ON picks(evaluation_mode);",
            "CREATE INDEX IF NOT EXISTS idx_picks_phase_hint ON picks(phase_hint);",
            "CREATE INDEX IF NOT EXISTS idx_picks_agency_id ON picks(agency_id);",
            "CREATE INDEX IF NOT EXISTS idx_picks_probability ON picks(probability);",
            "CREATE INDEX IF NOT EXISTS idx_arrivals_relabel_action ON arrivals(relabel_action);",
            "CREATE INDEX IF NOT EXISTS idx_arrivals_relabel_previous_phase ON arrivals(relabel_previous_phase);",
            "CREATE INDEX IF NOT EXISTS idx_arrivals_relabel_evaluation_score ON arrivals(relabel_evaluation_score);",
            "CREATE INDEX IF NOT EXISTS idx_arrivals_relabel_scores ON arrivals(relabel_scores);",
        ]

        for sql in indexes_sql:
            cursor.execute(sql)

        cursor.execute("COMMIT;")
        logger.info("Database tables created successfully.")

    except sqlite3.Error as e:
        logger.error(f"Error creating tables: {e}")
        cursor.execute("ROLLBACK;")
        raise


def refresh_event_coordinates_view(conn: sqlite3.Connection):
    """
    Refreshes event_coordinates view in SQLite by dropping and recreating it,
    and registers the geometry column in geometry_columns.

    Args:
        conn: SQLite connection object.
    """
    cursor = conn.cursor()

    # Drop the view if it exists
    cursor.execute("DROP VIEW IF EXISTS event_coordinates;")
    conn.commit()

    # Recreate the view
    cursor.execute(EVENT_COORDINATES_VIEW)
    conn.commit()

    # Register the geometry column
    register_geometry_for_view(
        conn=conn,
        view_name="event_coordinates",
        geometry_column="geometry",
        srid=4326,
        geom_type=1,  # POINT
        coord_dim=2,  # XY
    )


def register_geometry_for_view(
    conn, view_name, geometry_column, srid=4326, geom_type=1, coord_dim=2
):
    """
    Automatically register a geometry column for a view in SpatiaLite.

    Args:
        conn (sqlite3.Connection): Active connection to the SQLite database.
        view_name (str): Name of the view.
        geometry_column (str): Name of the geometry column in the view.
        srid (int): Spatial reference ID (default: 4326).
        geom_type (int): Geometry type (default: 1 for POINT).
        coord_dim (int): Coordinate dimension (default: 2 for XY).
    """
    cursor = conn.cursor()

    # Check if the view exists
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='view' AND name=?;", (view_name,)
    )
    if not cursor.fetchone():
        raise ValueError(f"The view '{view_name}' does not exist.")

    # Check if the geometry is already registered
    cursor.execute(
        "SELECT * FROM geometry_columns WHERE f_table_name=? AND f_geometry_column=?;",
        (view_name, geometry_column),
    )
    if cursor.fetchone():
        print(
            f"Geometry column '{geometry_column}' is already registered for view '{view_name}' ... removing it."
        )
        # Remove the existing registration manually
        cursor.execute(
            """
            DELETE FROM geometry_columns
            WHERE f_table_name=? AND f_geometry_column=?;
            """,
            (view_name, geometry_column),
        )

    # Register the geometry column
    print(f"Registering geometry column '{geometry_column}' for view '{view_name}'...")
    cursor.execute(
        """
        INSERT INTO geometry_columns (
            f_table_name, f_geometry_column, geometry_type, coord_dimension, srid, spatial_index_enabled
        ) VALUES (?, ?, ?, ?, ?, 0);
        """,
        (view_name, geometry_column, geom_type, coord_dim, srid),
    )
    conn.commit()
    print("Geometry column registered successfully.")


def import_catalog_object_to_sqlite_from_file(
    db_path: str,
    catalog: Catalog,
    enable_quakeml: bool = False,
    retries: int = 5,
    delay: int = 1,
    disable_tqdm: bool = False,
    backoff: str = "linear",  # "linear" or "exponential"
):
    """
    Import a catalog of seismic events into a SQLite database with retry logic and configurable backoff.

    Args:
        db_path (str): Path to the SQLite database file.
        catalog (Catalog): ObsPy Catalog object containing seismic events.
        enable_quakeml (bool, optional): If True, serialize and compress QuakeML for each event. Defaults to False.
        retries (int, optional): Maximum number of retry attempts in case of database lock. Defaults to 5.
        delay (int, optional): Base delay (in seconds) for retry attempts. Defaults to 1.
        disable_tqdm (bool, optional): If True, disables progress bars. Defaults to False.
        backoff (str, optional): Type of delay increase strategy: "linear" or "exponential". Defaults to "linear".
    """
    for attempt in range(1, retries + 1):
        conn = None
        try:
            conn = sqlite3.connect(db_path)
            if not conn:
                raise Exception(f"Failed to connect to the database at {db_path}")
            logger.info("Connected to the database successfully.")

            import_catalog_to_sqlite(conn, catalog, enable_quakeml, disable_tqdm)

            conn.commit()
            logger.info("Catalog imported successfully.")
            return

        except sqlite3.OperationalError as e:
            logger.warning(
                f"[Attempt {attempt}/{retries}] Database is locked or unavailable: {e}"
            )

            if backoff == "exponential":
                wait_time = delay * (2 ** (attempt - 1))
            else:  # linear fallback
                wait_time = delay * attempt

            logger.warning(f"Waiting {wait_time} second(s) before retrying...")
            time.sleep(wait_time)

        except Exception as e:
            logger.exception("Unexpected error during catalog import.")
            raise
        finally:
            if conn:
                conn.close()
                logger.debug("Database connection closed.")

    logger.error(f"Failed to import catalog after {retries} attempts.")
    raise sqlite3.OperationalError(f"Unable to access the database after {retries} retries.")


def import_catalog_to_sqlite_from_file(
    db_path: str, catalog_file: str, enable_quakeml: bool = False
):
    """
    Import a catalog of seismic events from a file to a SQLite database.

    Args:
        db_path (str): Path to the SQLite database file.
        catalog_file (str): Path to the file containing the catalog of seismic events.
        enable_quakeml (bool, optional): If True, serialize and compress QuakeML content for each event. Defaults to False.
    """

    # Create the database schema
    try:
        conn = create_schema(db_path)
    except Exception as e:
        logger.error(f"Error creating schema: {e}")
        raise e

    # # Read QuakeML file
    logger.info(f"Reading catalog from file '{catalog_file}'...")
    catalog = read_events(catalog_file)

    # Import the catalog into the SQLite database
    import_catalog_to_sqlite(conn, catalog, enable_quakeml)

    # extract agency names and stats to event table
    add_agency_names(conn)

    # Register the geometry column for the 'event_coordinates' view
    register_geometry_for_view(conn, "event_coordinates", "geometry")
    conn.close()


def import_catalog_to_sqlite(
    conn: sqlite3.Connection, catalog: Catalog, enable_quakeml: bool = False, disable_tqdm: bool = False
) -> None:
    """
    Export a catalog of seismic events to an SQLite database.
    This function creates the necessary schema in the SQLite database,
    processes each event in the catalog, and inserts the event data into
    the database. Optionally, it can serialize and compress QuakeML content
    for each event.

    Args:
        database (str): Path to the SQLite database file.
        catalog (Catalog): A catalog of seismic events to be exported.
        enable_quakeml (bool, optional): If True, serialize and compress
            QuakeML content for each event. Defaults to False.

    Raises:
        Exception: If there is an error processing an event, it will be caught
            and printed, but the function will continue processing the remaining
            events.

    Returns:
        None
    """

    # Process events and insert into SQLite
    # tqdm is used to display a progress bar
    for event in tqdm(catalog, desc="Importing events to SQLite", disable=disable_tqdm):
        # Serialize QuakeML content using format and compress it
        if enable_quakeml:
            quakeml_data = compress_quakeml_data(event, format="QUAKEML")
        else:
            quakeml_data = None

        try:
            inject_event(conn, event, quakeml_data)
        except sqlite3.IntegrityError as e:
            # not unique exception
            logger.warning(f"event {event.resource_id.id}: {e}")
            continue
        except Exception as e:
            logger.error(f"event {event.resource_id.id}: {e}")
            raise


def export_view_to_csv_exclude_geometry(db_path: str, view_name: str, output_csv: str):
    """
    Export a SQLite view to a CSV file, excluding the 'geometry' column. Format the 'time'
    column using UTCDateTime from ObsPy and apply rounding on specific numeric columns.

    Args:
        db_path (str): Path to the SQLite database.
        view_name (str): Name of the view to export.
        output_csv (str): Path to the output CSV file.
    """
    print(f"Exporting view '{view_name}' to '{output_csv}' ...")

    conn = sqlite3.connect(db_path)
    if not conn:
        raise Exception(f"Failed to connect to the database at {db_path}")

    # Get column names from the view
    cursor = conn.cursor()
    cursor.execute(f"SELECT * FROM {view_name} WHERE 1=0;")
    column_names = [desc[0] for desc in cursor.description if desc[0] != "geometry"]

    # Query the view excluding the geometry column
    selected_columns = ", ".join(column_names)
    cursor.execute(f"SELECT {selected_columns} FROM {view_name} ORDER BY time;")

    rows = cursor.fetchall()

    # Write to CSV
    with open(output_csv, mode="w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(column_names)  # Write the header

        # use tqdm to display a progress bar
        for row in tqdm(rows, desc="Exporting rows to CSV"):
            row_dict = dict(zip(column_names, row))

            # Format the 'time' column using UTCDateTime
            if "time" in row_dict and row_dict["time"]:
                row_dict["time"] = UTCDateTime(row_dict["time"]).isoformat(sep=" ")

            # Apply rounding to specific columns
            for col, precision in [
                ("time_errors", 2),
                ("depth", 1),
                ("depth_km", 1),
                ("quality_factor", 2),
                ("scatter_volume", 2),
                ("azimuthal_gap", 2),
                ("secondary_azimuthal_gap", 2),
                ("minimum_distance", 2),
                ("maximum_distance", 2),
                ("median_distance", 2),
                ("minimum_distance_deg", 2),
                ("maximum_distance_deg", 2),
                ("median_distance_deg", 2),
                ("rms", 2),
                ("erh", 2),
                ("erz", 2),
                ("erh_km", 2),
                ("erz_km", 2),
                ("expectation_depth", 1),
                ("expectation_depth_km", 1),
                ("magnitude", 2),
                ("magnitude_uncertainty", 2),
                ("uncertainty", 2),
                ("dist_km_from_preloc", 2),
                ("dist_from_preloc_km", 2),
                ("discrimination_probability", 2),
                ("discrimination_certainty", 2),
                ("delta_U", 2),
            ]:
                if col in row_dict and row_dict[col] is not None:
                    row_dict[col] = round(row_dict[col], precision)

            # Write the row to CSV
            writer.writerow([row_dict.get(col, "") for col in column_names])

    print(
        f"View '{view_name}' exported successfully to '{output_csv}' without 'geometry'."
    )
    conn.close()


def add_agency_names(conn: sqlite3.Connection) -> None:
    """
    Add a column to the 'events' table to store the names of the agencies
    associated with each event based on the 'agencies_list' column.

    Args:
        conn (sqlite3.Connection): Active connection to the SQLite database.
    """

    # Define the agency patterns to match
    agency_patterns = [
        {"name": "Renass", "pattern": "smi:franceseisme.fr"},
        {"name": "ISTerre", "pattern": "/ISTerre"},
        {"name": "CEA", "pattern": "cea.ldg"},
        {"name": "OCA", "pattern": "geofon/oca"},
        {"name": "OMP", "pattern": "event/omp"},
        {"name": "Renass/PhaseNet", "pattern": "PhaseNet"},
    ]

    # Define the mapping function
    def map_agencies_to_json(agencies_list_json):
        try:
            # Load the list of agency IDs from the JSON string
            event_ids = json.loads(agencies_list_json)
            # Find the matching agencies based on the event IDs
            matched_agencies = [
                pattern["name"]
                for pattern in agency_patterns
                if any(pattern["pattern"] in event_id for event_id in event_ids)
            ]
            # Return the matched agencies as a JSON string
            return json.dumps(matched_agencies)
        except json.JSONDecodeError:
            return json.dumps([])

    # Register the mapping function in SQLite
    conn.create_function("map_agencies_to_json", 1, map_agencies_to_json)

    # Add a column to the 'events' table to store the agency names
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(events);")
    columns = [row[1] for row in cursor.fetchall()]
    if "agency_names" not in columns:
        cursor.execute("ALTER TABLE events ADD COLUMN agency_names JSON;")
        logger.info("Added 'agency_names' column to the 'events' table.")

    # Update the 'agency_names' column based on the 'agencies_list' column
    cursor.execute(
        """
        UPDATE events
        SET agency_names = map_agencies_to_json(agencies_list)
        WHERE agencies_list IS NOT NULL;
        """
    )

    # Detect if any agency appears multiple times in the agency_names list
    def has_duplicates(json_array):
        """
        Detect if there are duplicates in a JSON array.
        Args:
            json_array (str): JSON array as a string.
        Returns:
            bool: True if duplicates exist, False otherwise.
        """
        try:
            items = json.loads(json_array)
            return len(items) > len(set(items))
        except (json.JSONDecodeError, TypeError):
            logger.error(f"Error parsing JSON array: {json_array}")
            return False

    conn.create_function("HAS_DUPLICATES", 1, has_duplicates)

    if "multiple_same_agencies" not in columns:
        cursor.execute("ALTER TABLE events ADD COLUMN multiple_same_agencies BOOLEAN;")

    cursor.execute(
        """
        UPDATE events
        SET multiple_same_agencies = HAS_DUPLICATES(agency_names);
        """
    )

    conn.commit()


def add_discrimination_info(conn: sqlite3.Connection, csv_file: str) -> None:
    """
    Add discrimination info to the event table from a CSV file.

    Args:
        db_path (str): Path to the SQLite database.
        csv_file (str): Path to the CSV file containing discrimination info.
    """
    cursor = conn.cursor()

    try:
        discrimination_df = pd.read_csv(csv_file)
    except Exception as e:
        print(f"Error reading CSV file '{csv_file}': {e}")
        return

    # check if the columns exist, and print the missing columns
    if not all(
        col in discrimination_df.columns
        for col in [
            "event_id",  # event_id
            "predhdq50",  # event_type
            "EqProbaPred hdq50",  # discrimination_probability
            "proba_count",  # discrimination_station_count
            "hdq50mad",  # discrimination_certainty
        ]
    ):
        # print the missing columns
        print(f"CSV file '{csv_file}' is missing required columns.")
        print(
            f"{RED}Required columns: 'event_id', 'predhdq50', 'EqProbaPred hdq50', 'proba_count', 'hdq50mad'{RESET}"
        )
        return

    print(f"Adding discrimination info from '{csv_file}' ...")

    # Update the event table with discrimination info
    for index, row in discrimination_df.iterrows():
        event_id = row["event_id"]
        # hdq50mad is the median absolute deviation of hdq50,
        # used as a measure of certainty,
        # the lower the value the more certain the prediction
        certainty = row["hdq50mad"]
        probability = (
            row["EqProbaPred hdq50"]
            if row["EqProbaPred hdq50"] > 0.5
            else 1 - row["EqProbaPred hdq50"]
        )
        station_count = row["proba_count"]
        predhdq50 = row["predhdq50"]

        # Determine the event type based on the probability
        if predhdq50 == 0:
            event_type = "earthquake"
        elif predhdq50 == 1:
            event_type = "quarry blast"
        else:
            # TODO: fix this in spectrocnn when station_count is very low
            event_type = "unknown"

        print(
            f"event_id: {event_id}, event_type: {event_type}, probability: {probability}, station_count: {station_count}, certainty: {certainty}"
        )

        cursor.execute(
            """
            UPDATE events
            SET event_type = ?,
                discrimination_probability = ?,
                discrimination_station_count = ?,
                discrimination_certainty = ?
            WHERE event_id = ?;
            """,
            (event_type, probability, station_count, certainty, event_id),
        )

    conn.commit()


def add_compute_localization_quality(conn: sqlite3.Connection) -> None:
    """
    Compute localization quality info and add it to the event table.

    Args:
        conn (sqlite3.Connection): Active connection to the SQLite database.
    """
    cursor = conn.cursor()

    # Check if the `quality` and `quality_factor` columns exist in the `origins` table
    cursor.execute("PRAGMA table_info(origins);")
    columns = [col[1] for col in cursor.fetchall()]

    if "quality" not in columns:
        cursor.execute("ALTER TABLE origins ADD COLUMN quality TEXT;")
    if "quality_factor" not in columns:
        cursor.execute("ALTER TABLE origins ADD COLUMN quality_factor DOUBLE;")
    conn.commit()

    # Fetch data for quality computation
    cursor.execute(
        """
        SELECT
            o.id,
            o.rms,
            o.erh,
            o.erz,
            o.used_phase_count,
            o.minimum_distance,
            o.median_distance,
            o.azimuthal_gap,
            o.secondary_azimuthal_gap,
            o.scatter_volume
        FROM
            origins AS o;
        """
    )
    rows = cursor.fetchall()

    # Process each row and update localization quality and quality_factor
    for row in rows:
        (
            origin_id,
            rms,
            erh,
            erz,
            num_phases,
            min_distance,
            median_distance,
            azimuthal_gap,
            secondary_azimuthal_gap,
            scatter_volume,
        ) = row

        if None in row:
            continue

        quality_factor, quality = classify_Michele_mod(
            rms,
            erh,
            erz,
            num_phases,
            min_distance,
            median_distance,
            azimuthal_gap,
            secondary_azimuthal_gap,
            scatter_volume,
        )

        # Update the database with the computed quality and quality_factor
        cursor.execute(
            """
            UPDATE origins
            SET quality = ?, quality_factor = ?
            WHERE id = ?;
            """,
            (quality, quality_factor, origin_id),
        )
    conn.commit()


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Process QuakeML files and store in SQLite."
    )
    parser.add_argument(
        "-d",
        "--database",
        default="seismic_data.sqlite",
        help="Path to the SQLite database.",
    )

    ############################
    # Import catalog to sqlite #
    ############################
    parser.add_argument(
        "-i", "--input", nargs="+", default=None, help="Input QuakeML files."
    )
    parser.add_argument(
        "-q",
        "--enable-quakeml",
        action="store_true",
        default=False,
        help="import full quakeml in the database.",
    )

    #####################
    # Add export to csv #
    #####################
    parser.add_argument(
        "-c",
        "--csv-output",
        default=None,
        help="export the view to a csv file.",
    )

    #################################
    # Export QuakeML data to a file #
    #################################
    parser.add_argument(
        "--export-quakeml", required=False, help="Path to the output QuakeML file."
    )

    # add -e to export quakeml only for a list of specific events (event_id)
    parser.add_argument(
        "-e", "--event-id", nargs="+", default=None, help="List of event_id to export."
    )

    # use csv file to use event_id list
    parser.add_argument(
        "--event-id-csv", default=None, help="CSV file containing event_id list."
    )

    ###########################
    # export quakeml by month #
    ###########################
    parser.add_argument(
        "--start-time", default=None, help="Start time for the export (YYYY-MM-DD)."
    )
    parser.add_argument(
        "--end-time", default=None, help="End time for the export (YYYY-MM-DD)."
    )

    ###########################
    # Add discrimination info #
    ###########################
    parser.add_argument(
        "--add-discrimination",
        default=None,
        help="Add discrimination info from csv file to the event table.",
    )

    ############################
    # Add localization quality #
    ############################
    parser.add_argument(
        "--add-localization-quality",
        action="store_true",
        default=False,
        help="Compute localization quality info to the event table.",
    )

    ####################
    # Add agency names #
    ####################
    parser.add_argument(
        "--add-agency-names",
        action="store_true",
        default=False,
        help="Add agency names to the event table.",
    )

    #########################
    # GT5 score computation #
    #########################
    parser.add_argument(
        "--gt5",
        action="store_true",
        default=False,
        help="Compute GT5 score.",
    )

    ######################################
    # Refresh the event_coordinates view #
    ######################################
    parser.add_argument(
        "--refresh-view",
        action="store_true",
        default=False,
        help="Refresh the event_coordinates view.",
    )

    args = parser.parse_args()

    print(args)

    # check if -i is given
    if args.input:
        # import quakeml file to sqlite
        for files in args.input:
            import_catalog_to_sqlite_from_file(
                args.database, files, args.enable_quakeml
            )
    elif args.csv_output:
        if os.path.exists(args.database) is False:
            print(f"Database '{args.database}' does not exist.")
            sys.exit(1)
        # output file should not already exist
        if os.path.exists(args.csv_output):
            print(f"Output file '{args.csv_output}' already exists.")
            sys.exit(1)
        # Export the view to a CSV file
        export_view_to_csv_exclude_geometry(
            args.database, "event_coordinates", args.csv_output
        )
    elif args.event_id_csv:
        if os.path.exists(args.database) is False:
            print(f"Database '{args.database}' does not exist.")
            sys.exit(1)
        # read event_id from csv file using pandas
        print(f"Reading event_id from '{args.event_id_csv}'")
        event_ids = pd.read_csv(args.event_id_csv)["event_id"].tolist()
        export_sqlite_to_quakeml(args.database, args.export_quakeml, event_ids)
    elif args.event_id:
        if os.path.exists(args.database) is False:
            print(f"Database '{args.database}' does not exist.")
            sys.exit(1)
        # export quakeml only for a list of specific events (event_id)
        print(f"Exporting QuakeML for event_id: {args.event_id}")
        export_sqlite_to_quakeml(args.database, args.export_quakeml, args.event_id)
    elif args.export_quakeml:
        if os.path.exists(args.database) is False:
            print(f"Database '{args.database}' does not exist.")
            sys.exit(1)
        # Export QuakeML data to a files by year and month
        print(f"Exporting QuakeML by year and month to {args.export_quakeml}")

        # Ensure the database view is up-to-date
        conn = sqlite3.connect(args.database)
        if not conn:
            raise Exception(f"Failed to connect to the database at {args.database}")
        refresh_event_coordinates_view(conn)
        conn.close()

        # Determine start_time and end_time
        if not args.start_time:
            conn = sqlite3.connect(args.database)
            if not conn:
                raise Exception(f"Failed to connect to the database at {args.database}")
            cursor = conn.cursor()
            cursor.execute("SELECT MIN(time) FROM event_coordinates;")
            start_time = cursor.fetchone()[0]
            conn.close()
            # Convert start_time to datetime, handling possible time components
            start_time = datetime.strptime(start_time.split(" ")[0], "%Y-%m-%d")
            ic(start_time, type(start_time))
        else:
            start_time = datetime.strptime(args.start_time, "%Y-%m-%d")

        if not args.end_time:
            conn = sqlite3.connect(args.database)
            if not conn:
                raise Exception(f"Failed to connect to the database at {args.database}")
            cursor = conn.cursor()
            cursor.execute("SELECT MAX(time) FROM event_coordinates;")
            end_time = cursor.fetchone()[0]
            conn.close()
            # Convert end_time to datetime and set it to the end of the month
            end_time = pd.to_datetime(end_time.split(" ")[0]) + pd.offsets.MonthBegin(1)
            end_time = end_time.to_pydatetime()
        else:
            # Adjust args.end_time to the end of the month if provided
            end_time = pd.to_datetime(
                args.end_time.split(" ")[0]
            ) + pd.offsets.MonthBegin(1)
            end_time = end_time.to_pydatetime()

        # Loop over months: each month starts on the 1st at 00:00:00
        months = pd.date_range(start=start_time, end=end_time, freq="MS")
        ic(months)

        for i in range(len(months) - 1):  # Exclude the last interval
            month_start = months[i]
            month_end = months[i + 1]  # Start of the next month

            # Ensure export directory exists
            os.makedirs(args.export_quakeml, exist_ok=True)

            # Define export file path
            export_path = os.path.join(
                args.export_quakeml,
                f"{month_start:%Y-%m}.qml",
            )

            # Skip if the file already exists
            if os.path.exists(export_path):
                print(f"File '{export_path}' already exists.")
                continue

            # Log and export QuakeML for the month
            ic(export_path, month_start, month_end)
            export_sqlite_to_quakeml(
                args.database,
                export_path,
                start_time=month_start.isoformat(),
                end_time=month_end.isoformat(),
            )

    elif args.add_discrimination:
        if os.path.exists(args.database) is False:
            print(f"Database '{args.database}' does not exist.")
            sys.exit(1)
        # Add discrimination info to the event table
        conn = sqlite3.connect(args.database)
        if not conn:
            raise Exception(f"Failed to connect to the database at {args.database}")
        add_discrimination_info(conn, args.add_discrimination)
        refresh_event_coordinates_view(conn)
        conn.close()
    elif args.add_localization_quality:
        if os.path.exists(args.database) is False:
            print(f"Database '{args.database}' does not exist.")
            sys.exit(1)
        # Add localisation quality info to the event table
        conn = sqlite3.connect(args.database)
        if not conn:
            raise Exception(f"Failed to connect to the database at {args.database}")
        add_compute_localization_quality(conn)
        refresh_event_coordinates_view(conn)
        conn.close()
    elif args.add_agency_names:
        if os.path.exists(args.database) is False:
            print(f"Database '{args.database}' does not exist.")
            sys.exit(1)
        # Add agency names to the event table
        conn = sqlite3.connect(args.database)
        if not conn:
            raise Exception(f"Failed to connect to the database at {args.database}")
        add_agency_names(conn)
        refresh_event_coordinates_view(conn)
        conn.close()
    elif args.gt5:
        if os.path.exists(args.database) is False:
            print(f"Database '{args.database}' does not exist.")
            sys.exit(1)
        # Compute GT5 score
        conn = sqlite3.connect(args.database)
        if not conn:
            raise Exception(f"Failed to connect to the database at {args.database}")
        add_gt5_score(conn)
        refresh_event_coordinates_view(conn)
        conn.close()
    elif args.refresh_view:
        if os.path.exists(args.database) is False:
            print(f"Database '{args.database}' does not exist.")
            sys.exit(1)
        # Refresh the event_coordinates view
        conn = sqlite3.connect(args.database)
        if not conn:
            raise Exception(f"Failed to connect to the database at {args.database}")
        logger.info("Refreshing event_coordinates view ...")
        # drop and recreate the view
        query = "DROP VIEW IF EXISTS event_coordinates;"
        conn.execute(query)
        conn.commit()
        logger.info("Dropped event_coordinates view.")
        query = EVENT_COORDINATES_VIEW
        conn.execute(query)
        conn.commit()
        logger.info("Created event_coordinates view.")
        conn.close()
    else:
        # Create the database schema
        try:
            conn = create_schema(args.database)
        except Exception as e:
            logger.error(f"Error creating schema: {e}")
        sys.exit(1)
