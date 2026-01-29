#!/usr/bin/env python
"""
Processes QuakeML files and stores the data in a SpatiaLite-enabled SQLite database.

Module organization:
    1. Imports
    2. Constants & Configuration
    3. ObsPy Patches
    4. Database Connection & Utilities
    5. QuakeML Data Extraction
    6. Database Schema Management
    7. Data Insertion (QuakeML -> SQLite)
    8. Data Import
    9. Data Export
    10. Database Enhancements
    11. CLI & Main
"""

# =============================================================================
# IMPORTS
# =============================================================================

import argparse
import csv
import json
import logging
import math
import os
import random
import re
import sqlite3
import sys
import time
import warnings
import xml.etree.ElementTree as ET
import zlib
from collections import defaultdict
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

from dbclust.db import validate_sql_identifier
from dbclust.gt5 import compute_gt5_score
from dbclust.localization_quality import classify_Michele_mod2
from dbclust.localization_quality import haversine_distance

# =============================================================================
# CONSTANTS & CONFIGURATION
# =============================================================================

# Suppress UserWarnings in ObsPy
warnings.filterwarnings("ignore", category=UserWarning, module="obspy")

# Logger (hierarchical name for selective level control)
logger = logging.getLogger("dbclust.inject_spatialite")

# ANSI escape codes for terminal colors
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
MAGENTA = "\033[95m"
CYAN = "\033[96m"
WHITE = "\033[97m"
RESET = "\033[0m"

# Non-standard QuakeML event types used by some agencies
NON_STANDARD_EVENT_TYPES = [
    "outside of network interest",
    "not locatable",
    "duplicate",
]

# =============================================================================
# SQL VIEW DEFINITIONS
# =============================================================================

# SQL for creating the event coordinates view
EVENT_COORDINATES_VIEW = """
    CREATE VIEW IF NOT EXISTS event_coordinates AS
    SELECT
        e.event_id,
        o.id AS origin_id,
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
        COALESCE(o.station_score, 0.0) AS station_score,
        COALESCE(o.avg_prob_p, 0.0) AS avg_prob_p,
        COALESCE(o.avg_prob_s, 0.0) AS avg_prob_s,
        COALESCE(o.avg_prob_total, 0.0) AS avg_prob_total,
        o.geometry
    FROM
        events AS e
        JOIN origins AS o ON e.event_id = o.event_id AND o.preferred = 1
        LEFT JOIN magnitudes AS m ON o.id = m.origin_id AND m.preferred = 1
    WHERE
        COALESCE(e.event_type, '') NOT IN ('not existing', 'not locatable');
"""

RELABELING_VIEW = """
    CREATE VIEW IF NOT EXISTS relabeling AS
    SELECT
        e.event_id,
        o.time,
        o.latitude, o.longitude,
        o.depth / 1000.0 AS depth_km,
        o.rms,
        o.erh AS erh_km,
        o.erz AS erz_km,
        o.quality,
        o.quality_factor,
        p.station_name,
        a.name AS phase_name,
        p.pick_time,
        a.time_weight,
        a.time_residual,
        a.distance AS distance_deg,
        a.relabel_action,
        a.relabel_previous_phase,
        a.relabel_evaluation_score,
        a.relabel_scores
    FROM
        events e
    JOIN
        origins o ON o.event_id = e.event_id
    JOIN
        arrivals a ON a.origin_id = o.id
    JOIN
        picks p ON a.pick_id = p.id
    WHERE
        o.preferred = 1
        AND a.time_weight != 0;
"""

# =============================================================================
# OBSPY PATCHES
# =============================================================================


def _patch_obspy_event_types():
    """Add non-standard event types to ObsPy's EventType enum."""
    try:
        from obspy.core.event.header import EventType

        for event_type in NON_STANDARD_EVENT_TYPES:
            if event_type not in EventType.keys():
                # Access internal OrderedDict to add new types
                EventType._Enum__enums[event_type.lower()] = event_type
    except Exception:
        pass  # Silently fail if ObsPy structure changes


_patch_obspy_event_types()


# =============================================================================
# DATABASE CONNECTION & UTILITIES
# =============================================================================


def load_spatialite(conn, logger=None):
    """
    Load SpatiaLite with better error handling and bus error prevention.
    """
    # Check if already loaded using a more robust method
    try:
        # Test if SpatiaLite functions are available
        conn.execute("SELECT InitSpatialMetaData(1)")
        if logger:
            logger.debug("SpatiaLite already initialized on this connection")
        return True
    except sqlite3.OperationalError:
        # SpatiaLite not loaded yet, continue
        pass
    except Exception as e:
        if logger:
            logger.debug(f"SpatiaLite check failed: {e}")

    try:
        # Enable loading extensions with better error handling
        conn.enable_load_extension(True)

        # Try loading SpatiaLite with specific error handling for macOS
        spatialite_paths = [
            "mod_spatialite",  # Try system path first (safest)
            "/opt/homebrew/lib/mod_spatialite.dylib",  # Apple Silicon Mac
            "/usr/local/lib/mod_spatialite.dylib",  # Intel Mac
            "/usr/lib/libspatialite.so.7",  # Linux newer
            "/usr/lib/libspatialite.so",  # Linux
        ]

        for i, path in enumerate(spatialite_paths):
            try:
                if logger:
                    logger.debug(f"Attempting to load SpatiaLite from: {path}")

                # Create a test connection to avoid corrupting the main one
                if i > 0:  # For file paths, check if they exist
                    if not os.path.exists(path):
                        continue

                conn.load_extension(path)

                # Verify the extension loaded correctly
                conn.execute("SELECT sqlite_version(), spatialite_version()")

                if logger:
                    logger.info(f"SpatiaLite loaded successfully from {path}")
                return True

            except sqlite3.OperationalError as e:
                if "cannot open shared object file" in str(
                    e
                ) or "image not found" in str(e):
                    continue  # Try next path
                elif "already loaded" in str(e):
                    if logger:
                        logger.info("SpatiaLite already loaded")
                    return True
                else:
                    if logger:
                        logger.debug(f"Failed to load from {path}: {e}")
                    continue
            except Exception as e:
                if logger:
                    logger.warning(f"Unexpected error loading from {path}: {e}")
                continue

        if logger:
            logger.warning("Could not load SpatiaLite from any known path")
        return False

    except Exception as e:
        if logger:
            logger.error(f"Critical error loading SpatiaLite: {e}")
        return False
    finally:
        try:
            conn.enable_load_extension(False)
        except (sqlite3.Error, AttributeError):
            pass


def create_safe_connection(db_path: str, uri=False, logger=None):
    """
    Create a SQLite connection with safe settings to prevent bus errors.
    """
    conn = None
    try:
        # Use connection WITHOUT autocommit mode to avoid transaction conflicts
        if uri:
            conn = sqlite3.connect(
                db_path, timeout=30, check_same_thread=False, uri=True
            )
        else:
            conn = sqlite3.connect(db_path, timeout=30, check_same_thread=False)

        # Set PRAGMA settings BEFORE any transactions start
        # These must be set outside of transactions
        pragmas_outside_transaction = [
            ("journal_mode", "WAL"),  # Better concurrency
            ("synchronous", "NORMAL"),  # Good balance of safety/performance
            ("busy_timeout", "60000"),  # 60s timeout for busy databases
        ]

        # These can be set anytime
        pragmas_anytime = [
            ("cache_size", "-2000"),  # Memory cache
            ("temp_store", "MEMORY"),  # Store temp tables in memory
            ("foreign_keys", "ON"),  # Enable foreign key constraints
        ]

        # Set critical pragmas first (outside transaction)
        for pragma, value in pragmas_outside_transaction:
            try:
                conn.execute(f"PRAGMA {pragma}={value};")
                conn.commit()  # Ensure pragma is applied
            except Exception as e:
                if logger:
                    logger.warning(f"Could not set PRAGMA {pragma}: {e}")

        # Set other pragmas
        for pragma, value in pragmas_anytime:
            try:
                conn.execute(f"PRAGMA {pragma}={value};")
            except Exception as e:
                if logger:
                    logger.warning(f"Could not set PRAGMA {pragma}: {e}")

        return conn

    except Exception as e:
        if conn:
            try:
                conn.close()
            except sqlite3.Error:
                pass
        if logger:
            logger.error(f"Failed to create safe connection: {e}")
        raise


# =============================================================================
# QUAKEML DATA EXTRACTION
# =============================================================================


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
        except (json.JSONDecodeError, TypeError):
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
        except (json.JSONDecodeError, TypeError):
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
        except (json.JSONDecodeError, TypeError):
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
        except (json.JSONDecodeError, TypeError):
            continue

        data = info.get("expectation")
        if data:
            expectation_latitude = data.get("latitude")
            expectation_longitude = data.get("longitude")
            depth = data.get("depth")
            expectation_depth = depth * 1000.0 if depth is not None else None
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
        except (json.JSONDecodeError, TypeError):
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
    except (AttributeError, TypeError):
        try:
            erh = origin.origin_uncertainty.horizontal_uncertainty / 1000.0
            method = "origin_uncertainty"
        except (AttributeError, TypeError):
            erh = None
            method = "unknown"

    try:
        erz = origin.depth_errors.uncertainty / 1000.0
        method = "origin_errors"
    except (AttributeError, TypeError):
        try:
            erz = origin.origin_uncertainty.depth_uncertainty / 1000.0
            method = "origin_uncertainty"
        except (AttributeError, TypeError):
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
        if pick is None:
            logger.error(
                f"Pick not found for arrival {arrival.resource_id.id} in event {event.resource_id.id}"
            )
            ic(event)
            ic(origin)
            ic(arrival)
            return None

        if pick.phase_hint and phase_type in pick.phase_hint.upper():
            count += 1

    return count


# -----------------------------------------------------------------------------
# Data Utilities
# -----------------------------------------------------------------------------


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


def execute_with_retry(conn, operation, retries=10, delay=0.5):
    """Execute an operation with retry logic in case of database lock."""
    attempts = 0
    while attempts < retries:
        try:
            operation()
            return
        except sqlite3.OperationalError as e:
            if "locked" in str(e):
                attempts += 1
                # Add random jitter to avoid synchronized retries
                jitter = random.uniform(0, delay)
                time.sleep(delay + jitter)
            else:
                raise
    raise sqlite3.OperationalError("Database is locked after multiple attempts")


# =============================================================================
# DATA INSERTION (QuakeML -> SQLite)
# =============================================================================


def inject_event(conn: sqlite3.Connection, event: Event, quakeml: str) -> None:
    """
    Injects an earthquake event and its related data into a SpatiaLite-enabled SQLite database.

    Parameters:
        conn (sqlite3.Connection): The SQLite database connection object.
        event (Event): The earthquake event containing event details.
        quakeml (str): The QuakeML XML string representing the event.

    Raises:
        Exception: If any database operation fails.
    """

    def insert_event_data():
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

            conn.execute(
                """
                INSERT OR IGNORE INTO picks (
                    id, event_id, station_name, pick_time, uncertainty,
                    evaluation_mode, phase_hint, agency_id, probability)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    pick.resource_id.id,
                    event.resource_id.id,
                    f"{pick.waveform_id.network_code}.{pick.waveform_id.station_code}",
                    to_datetime(pick.time),
                    pick.time_errors.uncertainty if pick.time_errors else None,
                    pick.evaluation_mode,
                    pick.phase_hint,
                    agency_id,
                    probability,
                ),
            )

        # Insert arrivals and collect phase information for station score calculation
        logger.debug(f"Inserting arrivals for event {event.resource_id.id}.")
        for origin in event.origins:
            insert_arrivals(conn, origin)

            # Calculate station score for this origin
            station_score = 0.0
            station_phases = defaultdict(set)

            # Group phases by station
            for arrival in origin.arrivals:
                if arrival.time_weight is None or arrival.time_weight == 0:
                    continue

                pick_id = arrival.pick_id
                pick = next((p for p in event.picks if p.resource_id == pick_id), None)
                if pick is None or pick.waveform_id is None:
                    continue

                net = pick.waveform_id.network_code
                sta = pick.waveform_id.station_code
                station_code = f"{net}.{sta}"

                # Add phase type to the station's set of phases
                phase = arrival.phase.lower() if arrival.phase else ""
                if phase.startswith("p"):
                    station_phases[station_code].add("P")
                elif phase.startswith("s"):
                    station_phases[station_code].add("S")

            # Calculate score based on phase combinations
            for phases in station_phases.values():
                if "P" in phases and "S" in phases:
                    station_score += 2.0
                elif "P" in phases:
                    station_score += 1.0
                elif "S" in phases:
                    station_score += 0.5

            # Update the origin with the calculated station score
            cursor.execute(
                "UPDATE origins SET station_score = ? WHERE id = ?",
                (station_score, origin.resource_id.id),
            )
            logger.debug(
                f"Updated station score for origin {origin.resource_id.id}: {station_score}"
            )

        # Insert magnitudes
        logger.debug(f"Inserting magnitudes for event {event.resource_id.id}.")
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
    quality = origin.quality
    if quality is None:
        logger.error(
            f"Origin '{origin.resource_id.id}' in event '{event.resource_id.id}' "
            f"has no quality information (origin.quality is None). "
            f"This origin will be skipped."
        )
        return
    rms = getattr(quality, "standard_error", None)
    P_count = phase_count(event, origin, "P")
    S_count = phase_count(event, origin, "S")
    erz, erh, err_method = get_erh_erz(origin)

    # Some origins may not have arrivals fully populated
    valid_arrivals = [
        arrival
        for arrival in origin.arrivals
        if arrival.time_weight
        and hasattr(arrival, "distance")
        and arrival.distance is not None
    ]

    if valid_arrivals:
        num_stations_10km = sum(1 for a in valid_arrivals if a.distance * 111.11 <= 10)
        num_stations_30km = sum(1 for a in valid_arrivals if a.distance * 111.11 <= 30)
        num_stations_150km = sum(
            1 for a in valid_arrivals if a.distance * 111.11 <= 150
        )
    else:
        num_stations_10km = None
        num_stations_30km = None
        num_stations_150km = None

    scatter_volume = get_scatter_volume(origin)
    expectation_latitude, expectation_longitude, expectation_depth = (
        get_expectation_localization(origin)
    )

    # distance in km between the horizontal coordinates of the origin and
    # the expectation coordinates
    if expectation_latitude is not None and expectation_longitude is not None:
        dloch = haversine_distance(
            origin.latitude,
            origin.longitude,
            expectation_latitude,
            expectation_longitude,
        )
    else:
        dloch = None

    # distance in km between the depth of the origin and the expectation depth
    dz = (
        abs(origin.depth - expectation_depth) / 1000.0
        if expectation_depth is not None
        else None
    )

    # Get only the relevant info from the origin method ID
    origin_method_id = getattr(getattr(origin, "method_id", None), "id", "unknown")
    if isinstance(origin_method_id, str) and "/" in origin_method_id:
        origin_method_id = origin_method_id.rsplit("/", 1)[-1]

    earth_model_id = getattr(getattr(origin, "earth_model_id", None), "id", "unknown")
    if isinstance(earth_model_id, str) and "/" in earth_model_id:
        earth_model_id = earth_model_id.rsplit("/", 1)[-1]

    # GT5 score will be computed by the compute_gt5_score function when called with --gt5
    gt5_status = None
    delta_U = None

    # Use azimuthal gaps from quality object
    azimuthal_gap = quality.azimuthal_gap
    secondary_azimuthal_gap = quality.secondary_azimuthal_gap

    # compute Michele et al. quality factor
    try:
        quality_factor, quality_class = classify_Michele_mod2(
            rms,
            erh,
            erz,
            quality.used_phase_count,
            quality.minimum_distance,
            quality.median_distance,
            azimuthal_gap,
            secondary_azimuthal_gap,
            scatter_volume,
            dloch,
            dz,
        )
    except Exception as e:
        logger.debug(f"Error classifying Michele mod: {e}")
        quality_factor = None
        quality_class = None

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
            origin.time_errors.uncertainty if origin.time_errors else None,
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
            quality.used_station_count,
            quality.used_phase_count,
            P_count,
            S_count,
            quality.minimum_distance,
            quality.maximum_distance,
            quality.median_distance,
            azimuthal_gap,
            secondary_azimuthal_gap,
            expectation_latitude,
            expectation_longitude,
            expectation_depth,
            scatter_volume,
            quality_class,
            quality_factor,
            num_stations_10km,
            num_stations_30km,
            num_stations_150km,
            delta_U,
            1 if gt5_status else 0,
            origin.evaluation_mode,
            (
                1
                if event.preferred_origin()
                and origin.resource_id.id == event.preferred_origin().resource_id.id
                else 0
            ),
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
                magnitude.method_id.id if magnitude.method_id else None,
                (
                    1
                    if event.preferred_magnitude()
                    and magnitude.resource_id.id
                    == event.preferred_magnitude().resource_id.id
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
            INSERT OR IGNORE INTO station_magnitudes (id, origin_id, magnitude, uncertainty, magnitude_type, method_id, waveform_id)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                station_magnitude.resource_id.id,
                station_magnitude.origin_id.id if station_magnitude.origin_id else None,
                station_magnitude.mag,
                (
                    station_magnitude.mag_errors.uncertainty
                    if station_magnitude.mag_errors
                    else None
                ),
                station_magnitude.station_magnitude_type,
                station_magnitude.method_id.id if station_magnitude.method_id else None,
                (
                    station_magnitude.waveform_id.get_seed_string()
                    if station_magnitude.waveform_id
                    else None
                ),
            ),
        )
        logger.debug(f"Station Magnitude {station_magnitude.resource_id.id} inserted.")


def insert_station_magnitude_contributions(
    conn: sqlite3.Connection, magnitude: Magnitude
) -> None:
    """Inserts all station magnitude contribution for a given magnitude into the database."""

    magnitude_id = magnitude.resource_id.id
    for station_magnitude_contribution in magnitude.station_magnitude_contributions:
        # StationMagnitudeContribution in ObsPy has station_magnitude_id, not resource_id
        if (
            not hasattr(station_magnitude_contribution, "station_magnitude_id")
            or station_magnitude_contribution.station_magnitude_id is None
        ):
            logger.warning(
                f"Station magnitude contribution for magnitude {magnitude_id} has no station_magnitude_id, skipping"
            )
            continue

        # Use station_magnitude_id as the unique identifier
        contribution_id = station_magnitude_contribution.station_magnitude_id.id

        conn.execute(
            """
            INSERT INTO station_magnitude_contributions (id, magnitude_id, residual, weight)
            VALUES (?, ?, ?, ?)
            """,
            (
                contribution_id,
                magnitude_id,
                station_magnitude_contribution.residual,
                station_magnitude_contribution.weight,
            ),
        )
        logger.debug(
            f"Station Magnitude Contribution {contribution_id} "
            f"for magnitude {magnitude_id} inserted."
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


# -----------------------------------------------------------------------------
# QuakeML Export Helpers (used by export functions in DATA EXPORT section)
# -----------------------------------------------------------------------------


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
    conn = create_safe_connection(db_path, logger=logger)
    spatialite_loaded = load_spatialite(conn, logger)
    if not spatialite_loaded:
        raise Exception("Failed to load SpatiaLite")

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
                if start_time and end_time
                else "SELECT COUNT(*) FROM quakeml;"
            )
            cursor.execute(
                count_query, (start_time, end_time) if start_time and end_time else ()
            )
            row = cursor.fetchone()
            count = row[0] if row else 0
            logger.info(f"Processing {count} events from {start_time} to {end_time}")

            query = (
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
    logger.info(f"Concatenated QuakeML written to {output_file}")


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
        logger.error(f"Error parsing QuakeML: {e}")


# =============================================================================
# DATABASE SCHEMA MANAGEMENT
# =============================================================================


def create_schema(db_path: str) -> sqlite3.Connection:
    """
    Create the database schema for a SpatiaLite-enabled SQLite database.

    Args:
        db_path (str): The file path to the SQLite database.

    Returns:
        sqlite3.Connection: The connection object to the SQLite database.
    """
    try:
        conn = create_safe_connection(db_path, logger=logger)
        spatialite_loaded = load_spatialite(conn, logger)
        if not spatialite_loaded:
            raise Exception("Failed to load SpatiaLite")

        cursor = conn.cursor()

        # Initialize SpatiaLite metadata if not already initialized
        cursor.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='spatial_ref_sys';"
        )
        row = cursor.fetchone()
        if row is None or row[0] == 0:
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
        row = cursor.fetchone()
        if row is None or row[0] == 0:
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


def create_tables(cursor: sqlite3.Cursor, create_indexes: bool = False) -> None:
    """
    Create the required tables in the database.

    Args:
        cursor (sqlite3.Cursor): The database cursor.
        create_indexes (bool): If True, also create indexes. Default False for faster bulk inserts.
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
                preferred BOOLEAN,
                station_score DOUBLE,
                avg_prob_p DOUBLE,
                avg_prob_s DOUBLE,
                avg_prob_total DOUBLE
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
                id TEXT NOT NULL,
                magnitude_id TEXT NOT NULL,
                residual DOUBLE,
                weight DOUBLE,
                PRIMARY KEY (id, magnitude_id)
            );
            """,
        ]

        for sql in tables_sql:
            cursor.execute(sql)

        # Only create indexes if explicitly requested (defer for bulk imports)
        if create_indexes:
            create_indexes_sql(cursor)

        cursor.execute("COMMIT;")
        logger.info("Database tables created successfully.")

    except sqlite3.Error as e:
        logger.error(f"Error creating tables: {e}")
        cursor.execute("ROLLBACK;")
        raise


def create_indexes_sql(cursor: sqlite3.Cursor) -> None:
    """
    Create indexes on database tables. Should be called AFTER bulk data insertion.

    Args:
        cursor (sqlite3.Cursor): The database cursor.
    """
    logger.info("Creating database indexes...")

    try:
        cursor.execute("BEGIN TRANSACTION;")

        indexes_sql = [
            #
            "CREATE INDEX IF NOT EXISTS idx_events_event_id ON events(event_id);",
            #
            "CREATE INDEX IF NOT EXISTS idx_origins_time ON origins(time);",
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
            "CREATE INDEX IF NOT EXISTS idx_picks_station_name_id ON picks(station_name, id);",
            #
            "CREATE INDEX IF NOT EXISTS idx_arrivals_relabel_action ON arrivals(relabel_action);",
            "CREATE INDEX IF NOT EXISTS idx_arrivals_relabel_previous_phase ON arrivals(relabel_previous_phase);",
            "CREATE INDEX IF NOT EXISTS idx_arrivals_relabel_evaluation_score ON arrivals(relabel_evaluation_score);",
            "CREATE INDEX IF NOT EXISTS idx_arrivals_relabel_scores ON arrivals(relabel_scores);",
        ]

        for sql in indexes_sql:
            cursor.execute(sql)

        cursor.execute("COMMIT;")
        logger.info("Database indexes created successfully.")

    except sqlite3.Error as e:
        logger.error(f"Error creating indexes: {e}")
        cursor.execute("ROLLBACK;")
        raise


def print_view_statistics(conn: sqlite3.Connection):
    """
    Print statistics about the event_coordinates view and underlying tables.

    Args:
        conn: SQLite connection object.
    """
    cursor = conn.cursor()

    # Count events and origins
    cursor.execute("SELECT COUNT(*) FROM events;")
    total_events = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM origins;")
    total_origins = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM origins WHERE preferred = 1;")
    preferred_origins = cursor.fetchone()[0]

    # Count rows in the view
    cursor.execute("SELECT COUNT(*) FROM event_coordinates;")
    view_count = cursor.fetchone()[0]

    # Count excluded events (not existing, not locatable)
    cursor.execute(
        "SELECT COUNT(*) FROM events WHERE event_type IN "
        "('not existing', 'not locatable', 'outside of network interest');"
    )
    excluded_events = cursor.fetchone()[0]

    # Print statistics
    print(f"  Statistics:")
    print(f"    - Total events in database: {total_events}")
    print(f"    - Total origins in database: {total_origins}")
    print(f"    - Preferred origins: {preferred_origins}")
    print(
        f"    - Excluded events (not existing/not locatable/outside of network interest): {excluded_events}"
    )
    print(f"    - Rows in event_coordinates view: {view_count}")

    # Warn if view is empty but there are events
    if view_count == 0:
        if total_events == 0:
            print("  Warning: No events in database. Import data first.")
        elif preferred_origins == 0:
            print(
                "  Warning: No preferred origins found. "
                "Ensure origins have 'preferred = 1' set."
            )
        else:
            print(
                "  Warning: View is empty despite having data. "
                "Check event_type values or origin-event links."
            )


def refresh_event_coordinates_view(conn: sqlite3.Connection):
    """
    Refreshes event_coordinates view in SQLite by dropping and recreating it,
    and registers the geometry column in geometry_columns.

    Args:
        conn: SQLite connection object.
    """
    with conn:
        cursor = conn.cursor()

        # Ensure all required columns exist
        ensure_required_columns_exist(conn)

        # Drop the view if it exists
        cursor.execute("DROP VIEW IF EXISTS event_coordinates;")

        # Recreate the view
        cursor.execute(EVENT_COORDINATES_VIEW)

        # Register the geometry column
        register_geometry_for_view(
            conn=conn,
            view_name="event_coordinates",
            geometry_column="geometry",
            srid=4326,
            geom_type=1,  # POINT
            coord_dim=2,  # XY
        )

        # Print statistics about the refreshed view
        print_view_statistics(conn)


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

    # Clean previous registrations (if any) in both tables to avoid duplicates
    try:
        cursor.execute(
            "DELETE FROM views_geometry_columns WHERE view_name = ?;",
            (view_name,),
        )
    except Exception as e:
        logger.debug(f"No views_geometry_columns cleanup needed: {e}")

    try:
        # In case older code incorrectly registered the view in geometry_columns
        cursor.execute(
            "DELETE FROM geometry_columns WHERE f_table_name = ?;",
            (view_name,),
        )
    except Exception as e:
        logger.debug(f"No geometry_columns cleanup needed: {e}")

    # Register the view in views_geometry_columns
    # Map the view's geometry to the base table 'origins'.
    base_table = "origins"
    base_geom_col = "geometry"
    view_rowid_col = "origin_id"  # provided by EVENT_COORDINATES_VIEW

    logger.info(
        f"Registering view geometry: view='{view_name}', geom='{geometry_column}', base='{base_table}.{base_geom_col}', rowid='{view_rowid_col}'"
    )

    # Detect schema of views_geometry_columns and insert accordingly
    cursor.execute("PRAGMA table_info(views_geometry_columns);")
    vg_cols = [row[1] for row in cursor.fetchall()]

    if {"geometry_type", "coord_dimension", "srid"}.issubset(set(vg_cols)):
        # Newer schema with explicit geometry metadata
        cursor.execute(
            """
            INSERT INTO views_geometry_columns (
                view_name, view_geometry, view_rowid,
                f_table_name, f_geometry_column,
                read_only, geometry_type, coord_dimension, srid
            ) VALUES (?, ?, ?, ?, ?, 1, ?, ?, ?);
            """,
            (
                view_name,
                geometry_column,
                view_rowid_col,
                base_table,
                base_geom_col,
                geom_type,
                coord_dim,
                srid,
            ),
        )
    else:
        # Older schema without geometry_type/coord_dimension/srid
        cursor.execute(
            """
            INSERT INTO views_geometry_columns (
                view_name, view_geometry, view_rowid,
                f_table_name, f_geometry_column, read_only
            ) VALUES (?, ?, ?, ?, ?, 1);
            """,
            (
                view_name,
                geometry_column,
                view_rowid_col,
                base_table,
                base_geom_col,
            ),
        )
    conn.commit()
    logger.info("View geometry registered successfully in views_geometry_columns.")


# =============================================================================
# DATA IMPORT
# =============================================================================


def import_catalog_object_to_sqlite_from_file(
    db_path: str,
    catalog,
    enable_quakeml: bool = False,
    retries: int = 3,
    delay: int = 2,
    disable_tqdm: bool = False,
    backoff: str = "exponential",
):
    """
    Import a catalog of seismic events into a SQLite database.
    Exit immediately if connection or SpatiaLite loading fails.
    """

    # Ensure the database file exists
    if not os.path.exists(db_path):
        try:
            with open(db_path, "w"):
                pass  # Create empty file
        except Exception as e:
            logging.error(f"Cannot create or modify database file {db_path}: {e}")
            raise

    last_exception = None

    for attempt in range(1, retries + 1):
        conn = None
        try:
            # Create connection
            conn = create_safe_connection(db_path, logger=logger)
            logger.info("Connected to the database successfully.")

            # Load SpatiaLite -> must succeed
            if not load_spatialite(conn, logger):
                raise RuntimeError("Failed to load SpatiaLite extension")

            # Import catalog
            import_catalog_to_sqlite(conn, catalog, enable_quakeml, disable_tqdm)
            logger.info("Catalog imported successfully.")
            return

        except sqlite3.OperationalError as e:
            last_exception = e
            logging.warning(f"[Attempt {attempt}/{retries}] Database error: {e}")

            if attempt < retries:
                wait_time = (
                    delay * (2 ** (attempt - 1))
                    if backoff == "exponential"
                    else delay * attempt
                )
                logging.warning(f"Waiting {wait_time} second(s) before retrying...")
                time.sleep(wait_time)

        except Exception as e:
            last_exception = e
            logging.error(f"Fatal error during catalog import: {e}")
            break

        finally:
            if conn:
                try:
                    conn.close()
                except Exception as e:
                    logging.warning(f"Error closing connection: {e}")

    # All attempts failed
    logging.error(f"Failed to import catalog after {retries} attempts.")
    raise last_exception or sqlite3.OperationalError(
        f"Unable to access the database after {retries} retries."
    )


def import_catalog_to_sqlite_from_file(
    conn: sqlite3.Connection, catalog_file: str, enable_quakeml: bool = False
):
    """
    Import a catalog of seismic events from a file to a SQLite database.

    Args:
        db_path (str): Path to the SQLite database file.
        catalog_file (str): Path to the file containing the catalog of seismic events.
        enable_quakeml (bool, optional): If True, serialize and compress QuakeML content for each event. Defaults to False.
    """

    # # Read QuakeML file
    logger.info(f"Reading catalog from file '{catalog_file}'...")
    catalog = read_events(catalog_file)

    # Import the catalog into the SQLite database
    import_catalog_to_sqlite(conn, catalog, enable_quakeml)

    # extract agency names and stats to event table
    add_agency_names(conn)


def import_catalog_to_sqlite(conn, catalog, enable_quakeml=False, disable_tqdm=False):

    # batch transactions for performance
    BATCH_SIZE = 100  # Commit every 100 events

    success_count = 0
    error_count = 0
    batch_count = 0

    conn.execute("BEGIN IMMEDIATE;")

    try:
        for i, event in enumerate(catalog):
            try:
                if enable_quakeml:
                    quakeml_data = compress_quakeml_data(event, format="QUAKEML")
                else:
                    quakeml_data = None

                inject_event(conn, event, quakeml_data)
                success_count += 1
                batch_count += 1

                # Periodic commit to avoid excessively long transactions
                if batch_count >= BATCH_SIZE:
                    conn.commit()
                    logging.info(f"Committed batch of {batch_count} events")
                    batch_count = 0
                    if i < len(catalog) - 1:  # not the last iteration
                        conn.execute("BEGIN IMMEDIATE;")

            except sqlite3.IntegrityError as e:
                error_count += 1
                logging.warning(f"Event {event.resource_id.id} skipped: {e}")
                continue

        # Final commit if there are remaining events in the batch
        if batch_count > 0:
            conn.commit()
            logging.info(f"Final commit of {batch_count} events")

        logging.info(
            f"Import completed. Success: {success_count}, Errors: {error_count}"
        )

    except Exception as e:
        conn.rollback()
        logging.error(f"Fatal error, rolling back current batch: {e}")
        import traceback

        logging.error(traceback.format_exc())
        raise


# =============================================================================
# DATA EXPORT
# =============================================================================


def export_view_to_csv_exclude_geometry(
    db_path: str, view_name: str, output_csv: str, batch_size: int = 10000
):
    """
    Export a SQLite view to a CSV file, excluding the 'geometry' column. Format the 'time'
    column using UTCDateTime from ObsPy and apply rounding on specific numeric columns.

    Args:
        db_path (str): Path to the SQLite database.
        view_name (str): Name of the view to export.
        output_csv (str): Path to the output CSV file.
        batch_size (int): Number of rows to process at a time. Default is 10000.
    """
    logger.info(f"Exporting view '{view_name}' to '{output_csv}' ...")

    # Validate view name to prevent SQL injection
    view_name = validate_sql_identifier(view_name)

    try:
        conn = create_safe_connection(db_path, logger=logger)
        spatialite_loaded = load_spatialite(conn, logger)
        if not spatialite_loaded:
            raise Exception("Failed to load SpatiaLite")

        cursor = conn.cursor()

        # Get column names from the view
        cursor.execute(f"SELECT * FROM {view_name} LIMIT 0;")
        column_names = [
            desc[0] for desc in cursor.description if desc[0].lower() != "geometry"
        ]

        # Columns to round and their precision
        round_columns = {
            "time_errors": 2,
            "depth": 1,
            "depth_km": 1,
            "quality_factor": 2,
            "scatter_volume": 2,
            "azimuthal_gap": 2,
            "secondary_azimuthal_gap": 2,
            "minimum_distance": 2,
            "maximum_distance": 2,
            "median_distance": 2,
            "minimum_distance_deg": 2,
            "maximum_distance_deg": 2,
            "median_distance_deg": 2,
            "rms": 2,
            "erh": 2,
            "erz": 2,
            "erh_km": 2,
            "erz_km": 2,
            "expectation_depth": 1,
            "expectation_depth_km": 1,
            "avg_prob_p": 2,
            "avg_prob_s": 2,
            "avg_prob_total": 2,
            "magnitude": 2,
            "magnitude_uncertainty": 2,
            "uncertainty": 2,
            "dist_km_from_preloc": 2,
            "dist_from_preloc_km": 2,
            "discrimination_probability": 2,
            "discrimination_certainty": 2,
            "delta_U": 2,
        }

        # Write to CSV in chunks
        with open(output_csv, mode="w", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(column_names)  # Write the header

            offset = 0
            total_processed = 0

            while True:
                # Fetch rows in batches
                query = f"""
                SELECT {", ".join(f'"{col}"' for col in column_names)} 
                FROM {view_name} 
                ORDER BY time
                LIMIT {batch_size} OFFSET {offset};
                """

                cursor.execute(query)
                rows = cursor.fetchall()

                if not rows:
                    break  # No more rows to process

                # Process batch
                for row in rows:
                    processed_row = list(row)

                    # Process time column
                    time_idx = (
                        column_names.index("time") if "time" in column_names else -1
                    )
                    if time_idx >= 0 and processed_row[time_idx]:
                        try:
                            processed_row[time_idx] = UTCDateTime(
                                processed_row[time_idx]
                            ).isoformat(sep=" ")
                        except (ValueError, TypeError):
                            pass  # Keep original value if conversion fails

                    # Process numeric columns
                    for col, precision in round_columns.items():
                        if col in column_names:
                            col_idx = column_names.index(col)
                            if processed_row[col_idx] is not None and isinstance(
                                processed_row[col_idx], (int, float)
                            ):
                                try:
                                    processed_row[col_idx] = round(
                                        float(processed_row[col_idx]), precision
                                    )
                                except (ValueError, TypeError):
                                    pass  # Keep original value if conversion fails

                    writer.writerow(processed_row)

                total_processed += len(rows)
                offset += batch_size
                logger.info(f"Processed {total_processed} rows...")

                # Commit changes to the database to free memory
                conn.commit()

        logger.info(f"Successfully exported {total_processed} rows to '{output_csv}'")

    except Exception as e:
        logger.error(f"Error exporting view to CSV: {str(e)}")
        raise
    finally:
        if "conn" in locals():
            conn.close()


# =============================================================================
# DATABASE COLUMN ADDITIONS
# =============================================================================


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
        conn: SQLite database connection
        csv_file (str): Path to the CSV file containing discrimination info.
    """

    try:
        discrimination_df = pd.read_csv(csv_file)
    except Exception as e:
        logger.error(f"Error reading CSV file '{csv_file}': {e}")
        return

    cursor = conn.cursor()
    try:
        # Start transaction
        cursor.execute("BEGIN TRANSACTION")

        # Check if the columns exist
        required_columns = [
            "event_id",  # event_id
            "predhdq50",  # event_type
            "EqProbaPred hdq50",  # discrimination_probability
            "proba_count",  # discrimination_station_count
            "hdq50mad",  # discrimination_certainty
        ]

        missing_columns = [
            col for col in required_columns if col not in discrimination_df.columns
        ]
        if missing_columns:
            logger.error(
                f"CSV file '{csv_file}' is missing required columns: {', '.join(missing_columns)}"
            )
            logger.error(f"{RED}Required columns: {', '.join(required_columns)}{RESET}")
            cursor.execute("ROLLBACK")
            return

        logger.info(f"Adding discrimination info from '{csv_file}'...")

        # Prepare data for batch update
        updates = []
        for _, row in discrimination_df.iterrows():
            try:
                event_id = row["event_id"]
                certainty = row["hdq50mad"]
                probability = (
                    row["EqProbaPred hdq50"]
                    if row["EqProbaPred hdq50"] > 0.5
                    else 1 - row["EqProbaPred hdq50"]
                )
                station_count = row["proba_count"]
                predhdq50 = row["predhdq50"]

                # Determine event type
                event_type = "unknown"
                if predhdq50 == 0:
                    event_type = "earthquake"
                elif predhdq50 == 1:
                    event_type = "quarry blast"

                updates.append(
                    (event_type, probability, station_count, certainty, event_id)
                )

            except Exception as e:
                logger.warning(f"Error processing row {_}: {e}")
                continue

        # Perform batch update
        if updates:
            try:
                cursor.executemany(
                    """
                    UPDATE events
                    SET event_type = ?,
                        discrimination_probability = ?,
                        discrimination_station_count = ?,
                        discrimination_certainty = ?
                    WHERE event_id = ?
                    """,
                    updates,
                )
                logger.info(f"Updated {len(updates)} events with discrimination info")
                conn.commit()

            except Exception as e:
                conn.rollback()
                logger.error(f"Error updating events: {e}")
                raise
        else:
            logger.warning("No valid updates to process")
            conn.rollback()

    except Exception as e:
        logger.error(f"Unexpected error in add_discrimination_info: {e}")
        try:
            conn.rollback()
        except Exception as rollback_error:
            logger.error(f"Error during rollback: {rollback_error}")
        raise


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
            o.scatter_volume,
            o.latitude,
            o.longitude,
            o.depth,
            o.expectation_latitude,
            o.expectation_longitude,
            o.expectation_depth
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
            origin_latitude,
            origin_longitude,
            origin_depth,
            expectation_latitude,
            expectation_longitude,
            expectation_depth,
        ) = row

        if None in row:
            continue

        dloch = haversine_distance(
            origin_longitude,
            origin_latitude,
            expectation_longitude,
            expectation_latitude,
        )

        dz = abs(origin_depth - expectation_depth) / 1000.0

        quality_factor, quality = classify_Michele_mod2(
            rms,
            erh,
            erz,
            num_phases,
            min_distance,
            median_distance,
            azimuthal_gap,
            secondary_azimuthal_gap,
            scatter_volume,
            dloch,
            dz,
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


# =============================================================================
# COMPUTED METRICS
# =============================================================================


def compute_origin_station_score(conn: sqlite3.Connection) -> None:
    """
    Compute and update station scores for all origins in the database.

    The score is calculated based on the number of P and S phases per station:
    - 2.0 points for stations with both P and S phases
    - 1.0 point for stations with only P phase
    - 0.5 points for stations with only S phase

    The total score is stored in the 'station_score' column of the origins table.
    """
    logger.info("Computing station scores for all origins...")
    cursor = conn.cursor()

    # Add station_score column if it doesn't exist
    cursor.execute("PRAGMA table_info(origins)")
    columns = [col[1] for col in cursor.fetchall()]

    if "station_score" not in columns:
        cursor.execute("ALTER TABLE origins ADD COLUMN station_score FLOAT DEFAULT 0.0")

    # Get only preferred origins
    cursor.execute("SELECT id FROM origins WHERE preferred = 1")
    origins = cursor.fetchall()

    for (origin_id,) in origins:
        # Get all valid arrivals for this origin with their phase information
        cursor.execute(
            """
            SELECT
                p.station_name,
                LOWER(a.name) as phase
            FROM arrivals a
            JOIN picks p ON a.pick_id = p.id
            WHERE a.origin_id = ?
            AND a.time_weight > 0
            AND p.station_name IS NOT NULL
        """,
            (origin_id,),
        )

        # Group phases by station
        station_phases = {}
        for station_name, phase in cursor.fetchall():
            if station_name not in station_phases:
                station_phases[station_name] = set()
            if phase.startswith("p"):
                station_phases[station_name].add("P")
            elif phase.startswith("s"):
                station_phases[station_name].add("S")

        # Calculate score
        score = 0.0
        for phases in station_phases.values():
            if "P" in phases and "S" in phases:
                score += 2.0
            elif "P" in phases:
                score += 1.0
            elif "S" in phases:
                score += 0.5

        # Update the origin with the computed score
        cursor.execute(
            "UPDATE origins SET station_score = ? WHERE id = ?", (score, origin_id)
        )

    conn.commit()
    logger.info("Station scores computation completed")


def compute_average_probabilities(conn):
    """Compute and store average probabilities for P, S and all picks."""
    logger.info("Computing average probabilities for all origins...")

    cursor = conn.cursor()

    # Ensure required columns exist in origins table
    ensure_required_columns_exist(conn)

    # Get all origins
    cursor.execute("SELECT id FROM origins WHERE preferred = 1")
    origins = cursor.fetchall()

    for (origin_id,) in origins:
        # For manual picks, probability is 1.0
        cursor.execute(
            """
        WITH pick_probs AS (
            SELECT
                p.id,
                p.phase_hint,
                CASE
                    WHEN p.evaluation_mode = 'manual' THEN 1.0
                    ELSE COALESCE(p.probability, 0.0)
                END as prob
            FROM arrivals a
            JOIN picks p ON a.pick_id = p.id
            WHERE a.origin_id = ?
        )
        UPDATE origins
        SET
            avg_prob_p = (
                SELECT COALESCE(AVG(prob), 0.0)
                FROM pick_probs
                WHERE phase_hint LIKE 'P%'
            ),
            avg_prob_s = (
                SELECT COALESCE(AVG(prob), 0.0)
                FROM pick_probs
                WHERE phase_hint LIKE 'S%'
            ),
            avg_prob_total = (
                SELECT COALESCE(AVG(prob), 0.0)
                FROM pick_probs
            )
        WHERE id = ?
        """,
            (origin_id, origin_id),
        )

    conn.commit()
    logger.info("Average probabilities computation completed")


# =============================================================================
# DATABASE ENHANCEMENT WORKFLOW
# =============================================================================


def ensure_required_columns_exist(conn: sqlite3.Connection) -> None:
    """
    Ensure all required columns exist in the database tables.

    Args:
        conn: SQLite database connection
    """
    try:
        cursor = conn.cursor()

        # Check and add columns to origins
        cursor.execute("PRAGMA table_info(origins)")
        origin_columns = [col[1] for col in cursor.fetchall()]

        if "station_score" not in origin_columns:
            logger.info("Adding station_score column to origins table...")
            cursor.execute(
                "ALTER TABLE origins ADD COLUMN station_score DOUBLE DEFAULT 0.0"
            )

        if "avg_prob_p" not in origin_columns:
            logger.info("Adding avg_prob_p column to origins table...")
            cursor.execute(
                "ALTER TABLE origins ADD COLUMN avg_prob_p DOUBLE DEFAULT 0.0"
            )

        if "avg_prob_s" not in origin_columns:
            logger.info("Adding avg_prob_s column to origins table...")
            cursor.execute(
                "ALTER TABLE origins ADD COLUMN avg_prob_s DOUBLE DEFAULT 0.0"
            )

        if "avg_prob_total" not in origin_columns:
            logger.info("Adding avg_prob_total column to origins table...")
            cursor.execute(
                "ALTER TABLE origins ADD COLUMN avg_prob_total DOUBLE DEFAULT 0.0"
            )

        conn.commit()
        logger.info("All required columns verified/added successfully.")

    except Exception as e:
        logger.error(f"Error ensuring required columns exist: {e}")
        raise


def apply_database_enhancements(args) -> None:
    """Apply database enhancements based on the provided arguments."""
    print("Applying database enhancements...")
    conn = None
    try:
        if any(
            [
                args.compute_station_scores,
                args.add_discrimination,
                args.add_localization_quality,
                args.add_agency_names,
                args.gt5,
                args.compute_prob_avg,
                args.refresh_view,
            ]
        ):
            conn = create_schema(args.database)

            # Always ensure all supplemental columns exist
            ensure_required_columns_exist(conn)

            if args.compute_station_scores:
                print("Computing station scores...")
                compute_origin_station_score(conn)

            if args.add_discrimination:
                print("Adding discrimination info...")
                add_discrimination_info(conn, args.add_discrimination)

            if args.add_localization_quality:
                print("Computing localization quality...")
                add_compute_localization_quality(conn)

            if args.add_agency_names:
                print("Adding agency names...")
                add_agency_names(conn)

            if args.compute_prob_avg:
                print("Computing average probabilities...")
                compute_average_probabilities(conn)

            if args.gt5:
                print("Computing GT5 metrics...")
                compute_gt5_score(conn)

            if any(
                [
                    args.compute_station_scores,
                    args.add_discrimination,
                    args.add_localization_quality,
                    args.add_agency_names,
                    args.gt5,
                    args.compute_prob_avg,
                    args.refresh_view,
                ]
            ):
                print("Refreshing event coordinates view...")
                refresh_event_coordinates_view(conn)
    except Exception as e:
        print(f"Error applying database enhancements: {str(e)}", file=sys.stderr)
        raise
    finally:
        if conn:
            conn.close()


# -----------------------------------------------------------------------------
# QuakeML Export Functions (called from main/CLI)
# -----------------------------------------------------------------------------


def get_database_time_range(db_path: str) -> tuple[datetime | None, datetime | None]:
    """Get the minimum and maximum time range from the database."""
    conn = None
    try:
        conn = create_safe_connection(db_path, logger=logger)
        spatialite_loaded = load_spatialite(conn, logger)
        if not spatialite_loaded:
            raise Exception("Failed to load SpatiaLite")

        cursor = conn.cursor()
        cursor.execute("SELECT MIN(time), MAX(time) FROM event_coordinates;")
        row = cursor.fetchone()
        if row is None:
            return None, None
        min_time, max_time = row

        # Convert to datetime objects
        min_time = (
            datetime.strptime(min_time.split(" ")[0], "%Y-%m-%d") if min_time else None
        )
        max_time = (
            datetime.strptime(max_time.split(" ")[0], "%Y-%m-%d") if max_time else None
        )

        return min_time, max_time
    finally:
        if conn:
            conn.close()


def export_quakeml_monthly(
    database_path: str, export_dir: str, start_time: str = None, end_time: str = None
) -> None:
    """Export QuakeML data with one file per month."""
    print(f"Exporting QuakeML by year and month to {export_dir}")

    # Ensure the database view is up-to-date
    conn = create_schema(database_path)
    refresh_event_coordinates_view(conn)
    conn.close()

    # Determine time range
    db_min, db_max = get_database_time_range(database_path)

    # Convert start_time to datetime if it's a string
    if start_time:
        if isinstance(start_time, str):
            start_dt = datetime.strptime(start_time.split(" ")[0], "%Y-%m-%d")
        else:
            start_dt = start_time
    else:
        start_dt = db_min

    # Convert end_time to datetime and adjust to end of month if needed
    if end_time:
        if isinstance(end_time, str):
            end_dt = pd.to_datetime(end_time.split(" ")[0]) + pd.offsets.MonthBegin(1)
        else:
            end_dt = pd.to_datetime(end_time) + pd.offsets.MonthBegin(1)
        end_dt = end_dt.to_pydatetime()
    else:
        # If db_max is a string, parse it first
        max_time_str = (
            db_max if isinstance(db_max, str) else db_max.strftime("%Y-%m-%d")
        )
        end_dt = pd.to_datetime(max_time_str) + pd.offsets.MonthBegin(1)
        end_dt = end_dt.to_pydatetime()

    # Loop over months
    months = pd.date_range(start=start_dt, end=end_dt, freq="MS")
    print(f"Exporting months from {start_dt} to {end_dt}")

    for i in range(len(months) - 1):  # Exclude the last interval
        month_start = months[i]
        month_end = months[i + 1]  # Start of the next month

        # Ensure export directory exists
        os.makedirs(export_dir, exist_ok=True)

        # Define export file path
        export_path = os.path.join(export_dir, f"{month_start:%Y-%m}.qml")

        # Skip if the file already exists
        if os.path.exists(export_path):
            print(f"File '{export_path}' already exists. Skipping...")
            continue

        print(f"Exporting {export_path}...")
        export_sqlite_to_quakeml(
            database_path,
            export_path,
            start_time=month_start.isoformat(),
            end_time=month_end.isoformat(),
        )


def export_quakeml_single(
    database_path: str,
    output_file: str,
    event_ids: list = None,
    event_id_csv: str = None,
    start_time: str = None,
    end_time: str = None,
) -> None:
    """Export QuakeML data to a single file."""
    # Handle event IDs from CSV if provided
    if event_id_csv:
        print(f"Reading event IDs from {event_id_csv}...")
        event_ids = pd.read_csv(event_id_csv)["event_id"].tolist()

    print(f"Exporting to {output_file}...")
    export_sqlite_to_quakeml(
        database_path,
        output_file,
        event_ids=event_ids,
        start_time=start_time,
        end_time=end_time,
    )


def handle_quakeml_export(args) -> None:
    """Handle QuakeML export based on the provided arguments."""
    if not args.export_quakeml:
        return

    try:
        if os.path.isdir(args.export_quakeml) or args.export_quakeml.endswith(os.sep):
            export_quakeml_monthly(
                database_path=args.database,
                export_dir=args.export_quakeml,
                start_time=args.start_time,
                end_time=args.end_time,
            )
        else:
            export_quakeml_single(
                database_path=args.database,
                output_file=args.export_quakeml,
                event_ids=args.event_id,
                event_id_csv=args.event_id_csv,
                start_time=args.start_time,
                end_time=args.end_time,
            )
    except Exception as e:
        print(f"Error during QuakeML export: {str(e)}", file=sys.stderr)
        sys.exit(1)


# =============================================================================
# CLI & ARGUMENT PARSING
# =============================================================================


def validate_date(date_str: str) -> str:
    """Validate date string format (YYYY-MM-DD)."""
    try:
        datetime.strptime(date_str, "%Y-%m-%d")
        return date_str
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Invalid date format: {date_str}. Use YYYY-MM-DD"
        )


def validate_file_exists(file_path: str) -> str:
    """Validate that a file exists."""
    if not os.path.exists(file_path):
        raise argparse.ArgumentTypeError(f"File not found: {file_path}")
    return file_path


def validate_dir_exists(dir_path: str) -> str:
    """Validate that a directory exists and is writable."""
    dir_path = os.path.abspath(dir_path)
    if not os.path.isdir(dir_path):
        raise argparse.ArgumentTypeError(f"Directory not found: {dir_path}")
    if not os.access(dir_path, os.W_OK):
        raise argparse.ArgumentTypeError(f"Directory not writable: {dir_path}")
    return dir_path


def parse_arguments() -> argparse.Namespace:
    """Parse and validate command line arguments."""
    # Create the top-level parser
    parser = argparse.ArgumentParser(
        description="Process QuakeML files and manage seismic event data in a SpatiaLite database.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Add a custom action to track if any unknown arguments are encountered
    class CustomHelpAction(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            parser.print_help()
            parser.exit(status=2)

    parser.register("action", "help", CustomHelpAction)

    # Database configuration
    db_group = parser.add_argument_group("Database Configuration")
    db_group.add_argument(
        "-d",
        "--database",
        default="seismic_data.sqlite",
        help="Path to the SQLite database file.",
    )

    # Import options
    import_group = parser.add_argument_group("Data Import Options")
    import_group.add_argument(
        "-i",
        "--input",
        nargs="+",
        type=validate_file_exists,
        help="Input QuakeML file(s) to import.",
    )
    import_group.add_argument(
        "-q",
        "--enable-quakeml",
        action="store_true",
        help="Store full QuakeML data in the database (increases size).",
    )

    # Export options
    export_group = parser.add_argument_group("Data Export Options")
    export_group.add_argument(
        "-c",
        "--csv-output",
        help="Export event coordinates to a CSV file.",
    )

    # QuakeML export options
    export_group.add_argument(
        "--export-quakeml",
        help="Export events to a QuakeML file.",
    )
    export_group.add_argument(
        "-e",
        "--event-id",
        nargs="+",
        help="Export specific events by ID (use with --export-quakeml).",
    )
    export_group.add_argument(
        "--event-id-csv",
        type=validate_file_exists,
        help="CSV file containing event IDs to export (use with --export-quakeml).",
    )
    export_group.add_argument(
        "--start-time",
        type=validate_date,
        help="Start time for export (YYYY-MM-DD).",
    )
    export_group.add_argument(
        "--end-time",
        type=validate_date,
        help="End time for export (YYYY-MM-DD).",
    )

    # Database enhancements
    enhancement_group = parser.add_argument_group("Database Enhancements")
    enhancement_group.add_argument(
        "--compute-station-scores",
        action="store_true",
        help="Compute and store station scores for all origins.",
    )
    enhancement_group.add_argument(
        "--add-discrimination",
        type=validate_file_exists,
        help="Add discrimination info from CSV file to events.",
    )
    enhancement_group.add_argument(
        "--add-localization-quality",
        action="store_true",
        help="Compute and add localization quality metrics.",
    )
    enhancement_group.add_argument(
        "--add-agency-names",
        action="store_true",
        help="Add agency names to the event table.",
    )
    enhancement_group.add_argument(
        "--gt5",
        action="store_true",
        help="Compute GT5 quality metrics.",
    )
    enhancement_group.add_argument(
        "--compute-prob-avg",
        action="store_true",
        help="Compute average probabilities for P, S and total picks.",
    )
    enhancement_group.add_argument(
        "--refresh-view",
        action="store_true",
        help="Refresh the event_coordinates view.",
    )

    # Parse known arguments first to check for help
    args, remaining = parser.parse_known_args()

    # If there are remaining arguments, they are unknown
    if remaining:
        parser.error(f"Unrecognized arguments: {' '.join(remaining)}")

    # Validate argument combinations
    if args.event_id and not args.export_quakeml:
        parser.error("--event-id requires --export-quakeml")
    if args.event_id_csv and not args.export_quakeml:
        parser.error("--event-id-csv requires --export-quakeml")
    if (args.start_time or args.end_time) and not args.export_quakeml:
        parser.error("Time range requires --export-quakeml")

    # Validate at least one action is specified if no input files are provided
    if not args.input and not any(
        [
            args.csv_output,
            args.export_quakeml,
            args.add_discrimination,
            args.add_localization_quality,
            args.add_agency_names,
            args.gt5,
            args.compute_prob_avg,
            args.compute_station_scores,
            args.refresh_view,
        ]
    ):
        parser.error(
            "No action requested. Please specify at least one action (import, export, or enhancement option)"
        )

    return args


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================


def main():
    """Main entry point for the script."""
    import sys

    args = parse_arguments()
    conn = None

    try:
        # Ensure database exists for operations that require it
        db_operations = [
            args.csv_output,
            args.export_quakeml,
            args.add_discrimination,
            args.add_localization_quality,
            args.add_agency_names,
            args.gt5,
            args.compute_prob_avg,
            args.compute_station_scores,
            args.refresh_view,
        ]

        if any(db_operations) and not os.path.exists(args.database):
            print(f"Error: Database '{args.database}' does not exist.", file=sys.stderr)
            sys.exit(1)

        # Handle input files
        if args.input:
            # Create the database schema
            try:
                conn = create_schema(args.database)
            except Exception as e:
                logger.error(f"Error creating schema: {e}")
                raise

            # Process files sequentially to preserve preferred_origin_id and preferred_magnitude_id
            parsed_count = 0
            error_count = 0
            total_events = 0

            with tqdm(
                total=len(args.input), desc="Processing QuakeML files", unit="file"
            ) as pbar:
                for input_file in args.input:
                    try:
                        logger.info(f"Reading catalog from file '{input_file}'...")
                        catalog = read_events(input_file)

                        parsed_count += 1
                        num_events = len(catalog)
                        total_events += num_events
                        pbar.set_postfix(
                            {
                                "parsed": parsed_count,
                                "errors": error_count,
                                "events": total_events,
                            }
                        )

                        import_catalog_to_sqlite(conn, catalog, args.enable_quakeml)
                        pbar.update(1)

                    except Exception as e:
                        error_count += 1
                        pbar.set_postfix(
                            {"parsed": parsed_count, "errors": error_count}
                        )
                        logger.error(f"Failed to parse {input_file}: {e}")
                        pbar.update(1)
                        continue

            print(
                f"\nProcessing completed: {parsed_count} files imported ({total_events} events), {error_count} files failed"
            )

            # Extract agency names after all imports
            add_agency_names(conn)

            # Create indexes AFTER all data is inserted (much faster)
            print("Creating database indexes...")
            cursor = conn.cursor()
            create_indexes_sql(cursor)
            print("Indexes created successfully.")

            register_geometry_for_view(conn, "event_coordinates", "geometry")
            conn.close()

        # Handle CSV export
        if args.csv_output:
            if os.path.exists(args.csv_output):
                print(
                    f"Error: Output file '{args.csv_output}' already exists.",
                    file=sys.stderr,
                )
                sys.exit(1)

            # Ensure the view is up-to-date before exporting
            print("Refreshing event_coordinates view...")
            conn = None
            try:
                conn = create_schema(args.database)
                refresh_event_coordinates_view(conn)
                print("Successfully refreshed event_coordinates view")
            finally:
                if conn:
                    conn.close()

            print(f"Exporting to {args.csv_output}...")
            export_view_to_csv_exclude_geometry(
                args.database, "event_coordinates", args.csv_output
            )

        # Handle QuakeML export
        handle_quakeml_export(args)

        # Database enhancements
        # Apply database enhancements if any enhancement option is specified
        enhancement_options = {
            "add_discrimination": args.add_discrimination,
            "add_localization_quality": args.add_localization_quality,
            "add_agency_names": args.add_agency_names,
            "gt5": args.gt5,
            "compute_prob_avg": args.compute_prob_avg,
            "compute_station_scores": args.compute_station_scores,
            "refresh_view": args.refresh_view,
        }
        print(
            f"Enhancement options detected: {[k for k, v in enhancement_options.items() if v]}"
        )

        active_enhancements = [k for k, v in enhancement_options.items() if v]
        if active_enhancements:
            print(f"Applying database enhancements: {', '.join(active_enhancements)}")
            apply_database_enhancements(args)
            print("Database enhancements completed successfully")
        else:
            print("No database enhancement options specified")

    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)
    finally:
        if conn:
            conn.close()


if __name__ == "__main__":
    main()
