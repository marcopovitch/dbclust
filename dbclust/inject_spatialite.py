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
from obspy import Catalog
from obspy import read_events
from obspy import UTCDateTime
from obspy.core.event import Arrival
from obspy.core.event import Event
from obspy.core.event import Magnitude
from obspy.core.event import Origin
from obspy.core.event import OriginQuality
from tqdm import tqdm

from dbclust.db import validate_sql_identifier
from dbclust.gap import compute_azimuthal_gap
from dbclust.gap import compute_secondary_azimuthal_gap
from dbclust.gt5 import compute_cpq
from dbclust.gt5 import compute_gallacher_gt5_score_obspy
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
# IMPORTANT: o.rowid is required for QGIS to display the view correctly.
# QGIS needs an integer rowid to identify features in SpatiaLite views.
EVENT_COORDINATES_VIEW = """
    CREATE VIEW IF NOT EXISTS event_coordinates AS
    SELECT
        o.rowid AS rowid,
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
        e.nb_agencies, e.agencies_list, e.agency_names, e.agency_ai_contributors, e.multiple_same_agencies,
        o.evaluation_mode,
        e.event_type,
        e.discrimination_probability,
        e.discrimination_station_count,
        e.discrimination_certainty,
        o.quality, o.quality_factor,
        o.gt5_status, o.delta_U, o.num_stations_10km, o.num_stations_30km, o.num_stations_150km,
        o.cpq, o.gallacher_gt5_status,
        COALESCE(o.station_score, 0.0) AS station_score,
        o.ps_ratio,
        o.ps_station_count,
        COALESCE(o.median_prob_p, 0.0) AS median_prob_p,
        COALESCE(o.median_prob_s, 0.0) AS median_prob_s,
        COALESCE(o.median_prob_total, 0.0) AS median_prob_total,
        ss.score AS silence_score,
        o.geometry
    FROM
        events AS e
        JOIN origins AS o ON e.event_id = o.event_id AND o.preferred = 1
        LEFT JOIN magnitudes AS m ON o.id = m.origin_id AND m.preferred = 1
        LEFT JOIN silence_scores AS ss ON ss.origin_id = o.id
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
    # Check if already loaded using a side-effect-free query
    try:
        conn.execute("SELECT spatialite_version()")
        if logger:
            logger.debug("SpatiaLite already loaded on this connection")
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

        # SQLite's load_extension() automatically appends the platform suffix
        # (.so on Linux, .dylib on macOS) — do NOT include the extension in the path.
        spatialite_paths = [
            "mod_spatialite",  # system PATH
            "/opt/homebrew/lib/mod_spatialite",  # Apple Silicon Mac
            "/usr/local/lib/mod_spatialite",  # Intel Mac / custom build
            "/usr/lib64/mod_spatialite",  # Linux RHEL/CentOS/HPC
            "/usr/lib/mod_spatialite",  # Linux Debian/Ubuntu
        ]

        import platform as _platform
        _so_suffix = ".dylib" if _platform.system() == "Darwin" else ".so"

        for i, path in enumerate(spatialite_paths):
            try:
                if logger:
                    logger.debug(f"Attempting to load SpatiaLite from: {path}")

                # For absolute paths, verify the file exists before trying
                if i > 0:
                    if not os.path.exists(path + _so_suffix) and not os.path.exists(path):
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


class MedianAggregate:
    """SQLite aggregate function that computes the median of a set of values."""

    def __init__(self):
        self.values = []

    def step(self, value):
        if value is not None:
            self.values.append(value)

    def finalize(self):
        if not self.values:
            return None
        self.values.sort()
        n = len(self.values)
        mid = n // 2
        if n % 2 == 0:
            return (self.values[mid - 1] + self.values[mid]) / 2.0
        return self.values[mid]


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

        # Register custom aggregate functions
        conn.create_aggregate("MEDIAN", 1, MedianAggregate)

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
# DATABASE UTILITIES
# =============================================================================


def _get_table_columns(cursor: sqlite3.Cursor, table: str) -> list:
    """Return the list of column names for a given table."""
    cursor.execute(f"PRAGMA table_info({table});")
    return [row[1] for row in cursor.fetchall()]


def _ensure_column(
    cursor: sqlite3.Cursor, table: str, column: str, col_type: str
) -> None:
    """Add *column* to *table* if it does not already exist."""
    if column not in _get_table_columns(cursor, table):
        cursor.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type};")


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
        if not isinstance(info, dict):
            continue
        if "probability" in info:
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
    Count the number of used phases of a given type in an origin.

    Only counts phases with time_weight > 0 (i.e., phases actually used
    in the localization, not rejected ones).

    Parameters:
        event (Event): The event containing picks.
        origin (Origin): The origin.
        phase_type (str): The phase type "P" or "S".

    Returns:
        int: The number of used phases of the given type.
    """
    count = 0
    for arrival in origin.arrivals:
        # Skip phases not used in localization (weight = 0 or missing)
        if not hasattr(arrival, "time_weight") or arrival.time_weight == 0:
            continue

        pick = next((p for p in event.picks if p.resource_id == arrival.pick_id), None)
        if pick is None:
            logger.error(
                f"Pick not found for arrival {arrival.resource_id.id} in event {event.resource_id.id}"
            )
            logger.debug(f"  event: {event.resource_id.id}")
            logger.debug(f"  origin: {origin.resource_id.id}")
            logger.debug(f"  arrival: {arrival.resource_id.id}")
            return None

        if pick.phase_hint and phase_type in pick.phase_hint.upper():
            count += 1

    return count


def compute_used_phase_count(origin: Origin) -> int:
    """
    Count the number of phases actually used in localization.

    Only counts arrivals with time_weight > 0 (i.e., phases actually used
    in the localization, not rejected ones).

    Parameters:
        origin (Origin): The origin containing arrivals.

    Returns:
        int: The number of used phases.
    """
    return sum(
        1
        for arrival in origin.arrivals
        if arrival.time_weight is not None and arrival.time_weight != 0
    )


def compute_used_station_count(event: Event, origin: Origin) -> int:
    """
    Count the number of unique stations actually used in localization.

    Only counts stations from arrivals with time_weight > 0.

    Parameters:
        event (Event): The event containing picks.
        origin (Origin): The origin containing arrivals.

    Returns:
        int: The number of unique stations used.
    """
    weighted_stations = set()
    for arrival in origin.arrivals:
        if arrival.time_weight is None or arrival.time_weight == 0:
            continue
        if arrival.pick_id:
            pick = next(
                (p for p in event.picks if p.resource_id == arrival.pick_id), None
            )
            if pick and pick.waveform_id:
                station_code = (
                    f"{pick.waveform_id.network_code}.{pick.waveform_id.station_code}"
                )
                weighted_stations.add(station_code)
    return len(weighted_stations)


def compute_ps_ratio(event: Event, origin: Origin) -> float:
    """
    Compute the ratio of stations with both P and S used phases
    over the total number of stations with at least one used phase.

    Only considers arrivals with time_weight not None and != 0.

    Parameters:
        event (Event): The event containing picks.
        origin (Origin): The origin containing arrivals.

    Returns:
        float: The PS ratio (0.0 to 1.0), or None if no used station.
    """
    ps_ratio, _, _ = compute_ps_ratio_and_station_score(event, origin)
    return ps_ratio


def compute_ps_ratio_and_station_score(
    event: Event, origin: Origin
) -> Tuple[Optional[float], float, int]:
    """
    Compute ps_ratio, station_score, and ps_station_count in a single pass over arrivals.

    Returns:
        Tuple[Optional[float], float, int]: (ps_ratio, station_score, ps_station_count)
    """
    station_phases = defaultdict(set)
    for arrival in origin.arrivals:
        if arrival.time_weight is None or arrival.time_weight == 0:
            continue
        pick = next((p for p in event.picks if p.resource_id == arrival.pick_id), None)
        if pick is None or pick.waveform_id is None:
            continue
        station_code = (
            f"{pick.waveform_id.network_code}.{pick.waveform_id.station_code}"
        )
        phase = arrival.phase.lower() if arrival.phase else ""
        if phase.startswith("p"):
            station_phases[station_code].add("P")
        elif phase.startswith("s"):
            station_phases[station_code].add("S")

    total_stations = len(station_phases)

    # station_score: P+S=2.0, P only=1.0, S only=0.5
    station_score = 0.0
    stations_with_both = 0
    for phases in station_phases.values():
        has_p = "P" in phases
        has_s = "S" in phases
        if has_p and has_s:
            station_score += 2.0
            stations_with_both += 1
        elif has_p:
            station_score += 1.0
        elif has_s:
            station_score += 0.5

    ps_ratio = stations_with_both / total_stations if total_stations > 0 else None

    return ps_ratio, station_score, stations_with_both


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


def repair_origin_quality(origin: Origin, event: Event) -> None:
    """Reconstruct missing OriginQuality from arrival data.

    Args:
        origin: Origin with missing quality.
        event: Parent event (used for station count via picks).
    """
    arrivals = origin.arrivals
    if not arrivals:
        logger.warning(
            f"Cannot repair quality for origin '{origin.resource_id.id}': no arrivals."
        )
        origin.quality = OriginQuality()
        return

    # Distances
    distances = [a.distance for a in arrivals if a.distance is not None]
    azimuths = [a.azimuth for a in arrivals if a.azimuth is not None]
    residuals = [
        a.time_residual
        for a in arrivals
        if a.time_residual is not None and a.time_weight and a.time_weight > 0
    ]

    quality = OriginQuality()

    if distances:
        quality.minimum_distance = min(distances)
        quality.maximum_distance = max(distances)
        quality.median_distance = float(np.median(distances))

    if azimuths:
        quality.azimuthal_gap = compute_azimuthal_gap(azimuths)
        quality.secondary_azimuthal_gap = compute_secondary_azimuthal_gap(azimuths)

    if residuals:
        quality.standard_error = float(
            np.round(np.sqrt(np.mean(np.array(residuals) ** 2)), 3)
        )

    quality.used_phase_count = compute_used_phase_count(origin)
    quality.used_station_count = compute_used_station_count(event, origin)
    quality.associated_phase_count = len(arrivals)

    origin.quality = quality
    logger.warning(
        f"Repaired missing quality for origin '{origin.resource_id.id}' "
        f"in event '{event.resource_id.id}' from {len(arrivals)} arrivals."
    )


def inject_event(
    conn: sqlite3.Connection,
    event: Event,
    quakeml: str,
    fix_quality: bool = False,
    ignore_missing_picks: bool = False,
) -> None:
    """
    Injects an earthquake event and its related data into a SpatiaLite-enabled SQLite database.

    Parameters:
        conn (sqlite3.Connection): The SQLite database connection object.
        event (Event): The earthquake event containing event details.
        quakeml (str): The QuakeML XML string representing the event.
        fix_quality (bool): If True, repair missing origin quality from arrival data.
        ignore_missing_picks (bool): If True, remove arrivals with missing picks instead of rejecting.

    Raises:
        ValueError: If the event is malformed and cannot be fixed.
        Exception: If any database operation fails.
    """
    # Validate and optionally fix event before any insertion
    pick_ids = {p.resource_id.id for p in event.picks}
    for origin in event.origins:
        if origin.quality is None:
            if fix_quality:
                repair_origin_quality(origin, event)
            else:
                raise ValueError(
                    f"Malformed QuakeML: origin '{origin.resource_id.id}' "
                    f"has no quality information (origin.quality is None)"
                )

        bad_arrivals = [
            a for a in origin.arrivals if a.pick_id and a.pick_id.id not in pick_ids
        ]
        if bad_arrivals:
            if ignore_missing_picks:
                for a in bad_arrivals:
                    logger.warning(
                        f"Event '{event.resource_id.id}': removing arrival referencing "
                        f"missing pick '{a.pick_id.id}' in origin '{origin.resource_id.id}'."
                    )
                origin.arrivals = [
                    a
                    for a in origin.arrivals
                    if not a.pick_id or a.pick_id.id in pick_ids
                ]
            else:
                raise ValueError(
                    f"Malformed QuakeML: arrival in origin '{origin.resource_id.id}' "
                    f"references missing pick '{bad_arrivals[0].pick_id.id}'"
                )

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

        # Deduplicate origins by resource_id (some QuakeML files contain duplicate origins)
        seen_origin_ids = set()
        unique_origins = []
        for origin in event.origins:
            if origin.resource_id.id not in seen_origin_ids:
                seen_origin_ids.add(origin.resource_id.id)
                unique_origins.append(origin)
            else:
                logger.warning(
                    f"Duplicate origin {origin.resource_id.id} "
                    f"in event {event.resource_id.id}, skipping."
                )

        # Insert origins (track which ones were actually inserted)
        logger.debug(f"Inserting origins for event {event.resource_id.id}.")
        inserted_origins = []
        for origin in unique_origins:
            if insert_origin(conn, origin, event):
                inserted_origins.append(origin)

        # Insert picks
        logger.debug(f"Inserting picks for event {event.resource_id.id}.")
        cursor = conn.cursor()
        for pick in event.picks:
            agency_id = (
                pick.creation_info.agency_id
                if pick.creation_info and hasattr(pick.creation_info, "agency_id")
                else None
            )
            source_event_id = (
                pick.creation_info.author
                if pick.creation_info and hasattr(pick.creation_info, "author")
                else None
            )
            method_id = (
                pick.method_id.id
                if pick.method_id and hasattr(pick.method_id, "id")
                else None
            )
            # Keep only the last component of a URI method_id (e.g. "PHASENET")
            if method_id and "/" in method_id:
                method_id = method_id.rsplit("/", 1)[-1]
            probability = get_pick_probability(pick)

            conn.execute(
                """
                INSERT OR IGNORE INTO picks (
                    id, event_id, station_name, location_code, channel_code,
                    pick_time, uncertainty, evaluation_mode, phase_hint, agency_id,
                    source_event_id, probability, method_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    pick.resource_id.id,
                    event.resource_id.id,
                    f"{pick.waveform_id.network_code}.{pick.waveform_id.station_code}",
                    pick.waveform_id.location_code or "",
                    pick.waveform_id.channel_code or "",
                    to_datetime(pick.time),
                    pick.time_errors.uncertainty if pick.time_errors else None,
                    pick.evaluation_mode,
                    pick.phase_hint,
                    agency_id,
                    source_event_id,
                    probability,
                    method_id,
                ),
            )

        # Insert arrivals for each origin
        logger.debug(f"Inserting arrivals for event {event.resource_id.id}.")
        for origin in inserted_origins:
            insert_arrivals(conn, origin)

            # Compute median pick probabilities for this origin (in Python)
            probs_p = []
            probs_s = []
            probs_all = []
            for arrival in origin.arrivals:
                if not arrival.pick_id:
                    continue
                pick = next(
                    (p for p in event.picks if p.resource_id == arrival.pick_id), None
                )
                if pick is None:
                    continue
                prob = (
                    1.0
                    if pick.evaluation_mode == "manual"
                    else (
                        pick.time_errors.uncertainty
                        if pick.time_errors and pick.time_errors.uncertainty is not None
                        else 0.0
                    )
                )
                # Use pick probability if available, otherwise fallback
                pick_prob = get_pick_probability(pick)
                prob = (
                    1.0
                    if pick.evaluation_mode == "manual"
                    else (pick_prob if pick_prob is not None else 0.0)
                )
                probs_all.append(prob)
                phase_hint = pick.phase_hint or ""
                if phase_hint.upper().startswith("P"):
                    probs_p.append(prob)
                elif phase_hint.upper().startswith("S"):
                    probs_s.append(prob)

            median_prob_p = float(np.median(probs_p)) if probs_p else 0.0
            median_prob_s = float(np.median(probs_s)) if probs_s else 0.0
            median_prob_total = float(np.median(probs_all)) if probs_all else 0.0

            cursor.execute(
                "UPDATE origins SET median_prob_p = ?, median_prob_s = ?, median_prob_total = ? WHERE id = ?",
                (
                    median_prob_p,
                    median_prob_s,
                    median_prob_total,
                    origin.resource_id.id,
                ),
            )

        # Insert magnitudes
        logger.debug(f"Inserting magnitudes for event {event.resource_id.id}.")
        inserted_origin_ids = {o.resource_id.id for o in inserted_origins}
        insert_magnitudes(conn, event, inserted_origin_ids)
        insert_station_magnitudes(conn, event, inserted_origin_ids)

        logger.debug(f"Event {event.resource_id.id} successfully inserted.")

    try:
        execute_with_retry(conn, insert_event_data)
    except Exception as e:
        logger.error(f"Failed to insert event {event.resource_id.id}: {e}")
        raise


def insert_origin(conn: sqlite3.Connection, origin: Origin, event: Event) -> bool:
    """Inserts an origin into the database.

    Returns:
        bool: True if the origin was inserted, False if it was skipped.
    """
    quality = origin.quality
    if quality is None:
        logger.warning(
            f"Origin '{origin.resource_id.id}' in event '{event.resource_id.id}' "
            f"has no quality information (origin.quality is None). "
            f"This origin and its arrivals will be skipped."
        )
        return False
    rms = getattr(quality, "standard_error", None)
    P_count = phase_count(event, origin, "P")
    S_count = phase_count(event, origin, "S")
    erz, erh, err_method = get_erh_erz(origin)

    # Recalculate used_phase_count and used_station_count from arrivals with non-zero weight
    # This is more reliable than using quality.used_phase_count/used_station_count
    used_phase_count = compute_used_phase_count(origin)
    used_station_count = compute_used_station_count(event, origin)

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

    # Gallacher GT5 score (cpq + gallacher_gt5_status) computed at injection time
    cpq = None
    gallacher_gt5_status = None
    try:
        result = compute_gallacher_gt5_score_obspy(origin)
        if result and result is not False:
            gallacher_gt5_bool, gallacher_details = result
            cpq = gallacher_details.get("cpq")
            gallacher_gt5_status = gallacher_gt5_bool
    except Exception as e:
        logger.debug(f"Could not compute Gallacher GT5 score for origin {origin.resource_id.id}: {e}")

    # Ratio of stations with both P and S used phases over total used stations
    # Also compute station_score and ps_station_count in the same pass to avoid redundant iteration
    ps_ratio, station_score, ps_station_count = compute_ps_ratio_and_station_score(event, origin)

    # Use azimuthal gaps from quality object
    azimuthal_gap = quality.azimuthal_gap
    secondary_azimuthal_gap = quality.secondary_azimuthal_gap

    # compute Michele et al. quality factor
    try:
        quality_factor, quality_class = classify_Michele_mod2(
            rms,
            erh,
            erz,
            used_phase_count,
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
            cpq,
            gallacher_gt5_status,
            evaluation_mode, preferred, ps_ratio, ps_station_count, station_score, geometry
        )
        VALUES (
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
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
            used_station_count,
            used_phase_count,
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
            cpq,
            1 if gallacher_gt5_status else 0,
            origin.evaluation_mode,
            (
                1
                if event.preferred_origin()
                and origin.resource_id.id == event.preferred_origin().resource_id.id
                else 0
            ),
            ps_ratio,
            ps_station_count,
            station_score,
            f"POINT({origin.longitude} {origin.latitude})",
        ),
    )
    logger.debug(f"Origin {origin.resource_id.id} inserted.")
    return True


def insert_magnitudes(
    conn: sqlite3.Connection, event: Event, inserted_origin_ids: Optional[set] = None
) -> None:
    """Inserts all magnitudes into the database."""
    for magnitude in event.magnitudes:
        # Only reference origin_id if it was actually inserted in the database
        origin_id = magnitude.origin_id.id if magnitude.origin_id else None
        if origin_id and inserted_origin_ids and origin_id not in inserted_origin_ids:
            logger.warning(
                f"Magnitude {magnitude.resource_id.id} references unknown origin {origin_id}, "
                f"setting origin_id to None."
            )
            origin_id = None

        conn.execute(
            """
            INSERT INTO magnitudes (id, origin_id, event_id, magnitude, uncertainty, station_count, magnitude_type, evaluation_mode, method_id, preferred)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                magnitude.resource_id.id,
                origin_id,
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


def insert_station_magnitudes(
    conn: sqlite3.Connection, event: Event, inserted_origin_ids: Optional[set] = None
) -> None:
    """Inserts all station magnitudes into the database."""
    for station_magnitude in event.station_magnitudes:
        # Handle empty origin_id: use None instead of empty string to avoid FK constraint failure
        origin_id = None
        if station_magnitude.origin_id and station_magnitude.origin_id.id:
            origin_id = station_magnitude.origin_id.id

        # Only reference origin_id if it was actually inserted in the database
        if origin_id and inserted_origin_ids and origin_id not in inserted_origin_ids:
            logger.warning(
                f"Station magnitude {station_magnitude.resource_id.id} references unknown origin {origin_id}, "
                f"setting origin_id to None."
            )
            origin_id = None

        conn.execute(
            """
            INSERT OR IGNORE INTO station_magnitudes (id, origin_id, magnitude, uncertainty, magnitude_type, method_id, waveform_id)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                station_magnitude.resource_id.id,
                origin_id,
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


def create_schema(db_path: str, create_spatial_index: bool = False) -> sqlite3.Connection:
    """
    Create the database schema for a SpatiaLite-enabled SQLite database.

    Args:
        db_path (str): The file path to the SQLite database.
        create_spatial_index (bool): Whether to create a spatial index on the
            geometry column. Disabled by default to avoid segfaults on macOS ARM
            during temp DB creation. Enable only on the final merged database.

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
        if "geometry" not in _get_table_columns(cursor, "origins"):
            logger.info("Adding 'geometry' column to 'origins' table...")
            cursor.execute(
                "SELECT AddGeometryColumn('origins', 'geometry', 4326, 'POINT', 'XY');"
            )

        # Optionally create spatial index (skip for temp DBs — CreateSpatialIndex
        # can cause segfaults on macOS ARM when called in parallel subprocesses)
        if create_spatial_index:
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
        else:
            logger.debug("Skipping spatial index creation (create_spatial_index=False)")

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
                agency_ai_contributors JSON,
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
                location_code TEXT,
                channel_code TEXT,
                pick_time TIMESTAMP,
                evaluation_mode TEXT,
                uncertainty DOUBLE,
                phase_hint TEXT,
                agency_id TEXT,
                source_event_id TEXT,
                probability DOUBLE,
                method_id TEXT
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
                cpq DOUBLE,
                gallacher_gt5_status BOOLEAN,
                evaluation_mode TEXT,
                preferred BOOLEAN,
                station_score DOUBLE,
                ps_ratio DOUBLE,
                ps_station_count INTEGER,
                median_prob_p DOUBLE,
                median_prob_s DOUBLE,
                median_prob_total DOUBLE
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
            """
            CREATE TABLE IF NOT EXISTS silence_scores (
                origin_id TEXT PRIMARY KEY REFERENCES origins(id) ON DELETE CASCADE,
                event_id TEXT REFERENCES events(event_id) ON DELETE CASCADE,
                score DOUBLE,
                n_candidates INTEGER,
                n_excluded INTEGER,
                n_used INTEGER,
                n_used_outside_radius INTEGER,
                n_active INTEGER,
                n_active_missing INTEGER,
                expected_weight DOUBLE,
                missing_weight DOUBLE,
                effective_radius_km DOUBLE,
                reason TEXT,
                radius_km DOUBLE,
                threshold DOUBLE,
                decay_factor DOUBLE
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
    # IMPORTANT: view_rowid must be an INTEGER column for QGIS to work correctly.
    # Using o.rowid from the origins table (exposed as 'rowid' in the view).
    base_table = "origins"
    base_geom_col = "geometry"
    view_rowid_col = "rowid"  # INTEGER rowid from origins table, required for QGIS

    logger.info(
        f"Registering view geometry: view='{view_name}', geom='{geometry_column}', base='{base_table}.{base_geom_col}', rowid='{view_rowid_col}'"
    )

    # Detect schema of views_geometry_columns and insert accordingly
    vg_cols = _get_table_columns(cursor, "views_geometry_columns")

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


def import_catalog_to_sqlite(
    conn,
    catalog,
    enable_quakeml=False,
    disable_tqdm=False,
    fix_quality=False,
    ignore_missing_picks=False,
    sqlite_batch_size: int = 100,
):

    # batch transactions for performance
    batch_size = max(1, sqlite_batch_size)

    success_count = 0
    duplicate_count = 0
    malformed_count = 0
    batch_count = 0

    conn.execute("BEGIN IMMEDIATE;")

    try:
        for i, event in enumerate(catalog):
            conn.execute("SAVEPOINT sp_event;")
            try:
                if enable_quakeml:
                    quakeml_data = compress_quakeml_data(event, format="QUAKEML")
                else:
                    quakeml_data = None

                inject_event(
                    conn,
                    event,
                    quakeml_data,
                    fix_quality=fix_quality,
                    ignore_missing_picks=ignore_missing_picks,
                )
                conn.execute("RELEASE SAVEPOINT sp_event;")
                success_count += 1
                batch_count += 1

                # Periodic commit to avoid excessively long transactions
                if batch_count >= batch_size:
                    conn.commit()
                    logging.info(f"Committed batch of {batch_count} events")
                    batch_count = 0
                    if i < len(catalog) - 1:  # not the last iteration
                        conn.execute("BEGIN IMMEDIATE;")

            except sqlite3.IntegrityError as e:
                conn.execute("ROLLBACK TO SAVEPOINT sp_event;")
                conn.execute("RELEASE SAVEPOINT sp_event;")
                duplicate_count += 1
                logging.warning(
                    f"Event {event.resource_id.id} skipped (duplicate): {e}"
                )
                continue

            except ValueError as e:
                conn.execute("ROLLBACK TO SAVEPOINT sp_event;")
                conn.execute("RELEASE SAVEPOINT sp_event;")
                malformed_count += 1
                logging.warning(
                    f"Event {event.resource_id.id} skipped (malformed): {e}"
                )
                continue

        # Final commit if there are remaining events in the batch
        if batch_count > 0:
            conn.commit()
            logging.info(f"Final commit of {batch_count} events")

        logging.info(
            f"Import completed. Success: {success_count}, "
            f"Duplicates: {duplicate_count}, Malformed: {malformed_count}"
        )

    except Exception as e:
        conn.rollback()
        logging.error(f"Fatal error, rolling back current batch: {e}")
        import traceback

        logging.error(traceback.format_exc())
        raise

    return success_count, duplicate_count, malformed_count


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
            "ps_ratio": 2,
            "median_prob_p": 2,
            "median_prob_s": 2,
            "median_prob_total": 2,
            "magnitude": 2,
            "magnitude_uncertainty": 2,
            "uncertainty": 2,
            "dist_km_from_preloc": 2,
            "dist_from_preloc_km": 2,
            "discrimination_probability": 2,
            "discrimination_certainty": 2,
            "delta_U": 2,
            "cpq": 3,
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
    Populate agency columns in the 'events' table from picks:
      - agency_names: JSON array of distinct agency_id values that contributed a
        real localization (picks with a non-null source_event_id, i.e. picks that
        originate from an operator's catalogue event).
      - agency_ai_contributors: JSON array of distinct agency_id values that only
        contributed AI/automatic picks (source_event_id IS NULL) without being
        counted as a localizing agency.
      - agencies_list: JSON array of distinct source_event_id display labels for
        picks used in the preferred origin (real ids for operator picks, synthetic
        "AGENCY/METHOD" labels for AI picks).
      - nb_agencies: count of distinct agencies that provided a real localization.
      - multiple_same_agencies: True if any localizing agency contributed picks
        from more than one distinct real source_event_id (signals a merge bug).

    Args:
        conn (sqlite3.Connection): Active connection to the SQLite database.
    """
    cursor = conn.cursor()
    _ensure_column(cursor, "events", "agency_names", "JSON")
    _ensure_column(cursor, "events", "agency_ai_contributors", "JSON")
    _ensure_column(cursor, "events", "multiple_same_agencies", "BOOLEAN")

    # Only consider picks actually used in the preferred origin (time_weight > 0).
    #
    # real_source_event_id IS NOT NULL  →  pick comes from a real operator catalogue
    #                                       event: the agency is a localizing agency.
    # real_source_event_id IS NULL      →  pick is AI/automatic without a catalogue
    #                                       origin: the agency is an AI contributor only.
    #
    # agency_names        : agencies with at least one real (operator) pick.
    # agency_ai_contributors: agencies whose picks are exclusively AI/automatic.
    # nb_agencies         : count of localizing agencies only.
    # multiple_same_agencies: localizing agency with > 1 distinct real source_event_id
    #                         (merge/association bug — AI picks excluded intentionally).
    _ensure_column(cursor, "picks", "source_event_id", "TEXT")
    _ensure_column(cursor, "picks", "method_id", "TEXT")

    # Boost cache for this heavy operation: 128 MB instead of the default 2 MB.
    cursor.execute("PRAGMA cache_size = -128000;")

    # Step 1 – materialise used_picks into a temp table so it is scanned only once
    # instead of being re-evaluated for every downstream CTE reference.
    cursor.executescript(
        """
        DROP TABLE IF EXISTS temp.t_used_picks;
        CREATE TEMP TABLE t_used_picks AS
            SELECT p.event_id,
                   p.agency_id,
                   p.source_event_id AS real_source_event_id,
                   CASE
                       WHEN p.source_event_id IS NOT NULL THEN p.source_event_id
                       WHEN p.method_id IS NOT NULL       THEN p.agency_id || '/' || p.method_id
                       WHEN p.evaluation_mode = 'automatic' THEN p.agency_id || '/AI'
                       ELSE NULL
                   END AS source_event_id
            FROM picks p
            JOIN arrivals a ON a.pick_id = p.id
            JOIN origins o  ON o.id = a.origin_id
            WHERE o.preferred = 1
              AND a.time_weight > 0
              AND p.agency_id IS NOT NULL
              AND p.agency_id != '';

        CREATE INDEX temp.idx_tup_event_agency
            ON t_used_picks (event_id, agency_id);

        -- Step 2 – localizing agencies: those with at least one real operator pick.
        DROP TABLE IF EXISTS temp.t_localizing;
        CREATE TEMP TABLE t_localizing AS
            SELECT DISTINCT event_id, agency_id
            FROM t_used_picks
            WHERE real_source_event_id IS NOT NULL;

        CREATE INDEX temp.idx_tloc_event_agency
            ON t_localizing (event_id, agency_id);

        -- Step 3 – events where a localizing agency spans > 1 source_event_id (merge bug).
        DROP TABLE IF EXISTS temp.t_multi;
        CREATE TEMP TABLE t_multi AS
            SELECT DISTINCT event_id
            FROM t_used_picks
            WHERE real_source_event_id IS NOT NULL
            GROUP BY event_id, agency_id
            HAVING COUNT(DISTINCT real_source_event_id) > 1;

        CREATE INDEX temp.idx_tmulti_event ON t_multi (event_id);
        """
    )

    # Step 4 – aggregate and update in a single pass over the materialised tables.
    cursor.execute(
        """
        UPDATE events
        SET agency_names           = agg.agency_names,
            agency_ai_contributors = agg.agency_ai_contributors,
            agencies_list          = agg.agencies_list,
            nb_agencies            = agg.nb_agencies,
            multiple_same_agencies = agg.multiple_same_agencies
        FROM (
            SELECT
                up.event_id,
                json_group_array(DISTINCT la.agency_id)
                    FILTER (WHERE la.agency_id IS NOT NULL) AS agency_names,
                json_group_array(DISTINCT up.agency_id)
                    FILTER (WHERE la.agency_id IS NULL)     AS agency_ai_contributors,
                json_group_array(DISTINCT up.source_event_id) AS agencies_list,
                COUNT(DISTINCT la.agency_id)                  AS nb_agencies,
                MAX(CASE WHEN m.event_id IS NOT NULL THEN 1 ELSE 0 END)
                                                              AS multiple_same_agencies
            FROM t_used_picks up
            LEFT JOIN t_localizing la
                   ON la.event_id = up.event_id AND la.agency_id = up.agency_id
            LEFT JOIN t_multi m ON m.event_id = up.event_id
            GROUP BY up.event_id
        ) AS agg
        WHERE events.event_id = agg.event_id;
        """
    )

    conn.commit()

    # Cleanup temp tables
    cursor.executescript(
        """
        DROP TABLE IF EXISTS temp.t_used_picks;
        DROP TABLE IF EXISTS temp.t_localizing;
        DROP TABLE IF EXISTS temp.t_multi;
        """
    )


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
                updated_count = 0
                for row_args in updates:
                    cursor.execute(
                        """
                        UPDATE events
                        SET event_type = ?,
                            discrimination_probability = ?,
                            discrimination_station_count = ?,
                            discrimination_certainty = ?
                        WHERE event_id = ?
                        """,
                        row_args,
                    )
                    if cursor.rowcount == 0:
                        logger.warning(
                            f"Discrimination CSV: event_id '{row_args[-1]}' not found in DB, skipped"
                        )
                    else:
                        updated_count += 1
                logger.info(
                    f"Updated {updated_count}/{len(updates)} events with discrimination info"
                    + (
                        f" ({len(updates) - updated_count} not found in DB)"
                        if updated_count < len(updates)
                        else ""
                    )
                )
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


def add_silence_score(conn: sqlite3.Connection, csv_file: str) -> None:
    """
    Import silence scores from a CSV file into the silence_scores table.

    The CSV is the output of the silence-score tool. Each row is matched to an
    origin via the optional `origin_id` column; when absent, the preferred origin
    for the event is used.

    Args:
        conn: SQLite database connection
        csv_file: Path to the silence score CSV file
    """
    try:
        df = pd.read_csv(csv_file, low_memory=False)
    except Exception as e:
        logger.error(f"Error reading CSV file '{csv_file}': {e}")
        return

    required_columns = [
        "event_id",
        "score",
        "n_candidates",
        "n_excluded",
        "n_used",
        "n_used_outside_radius",
        "n_active",
        "n_active_missing",
        "expected_weight",
        "missing_weight",
        "effective_radius_km",
        "reason",
        "radius_km",
        "threshold",
        "decay_factor",
    ]
    missing_columns = [c for c in required_columns if c not in df.columns]
    if missing_columns:
        logger.error(
            f"CSV file '{csv_file}' is missing required columns: {', '.join(missing_columns)}"
        )
        return

    # Load preferred origins from DB into memory for fast lookup (event_id → origin_id)
    cursor = conn.cursor()
    has_origin_id_col = "origin_id" in df.columns

    if has_origin_id_col:
        # Build set of known origin ids for validation
        cursor.execute("SELECT id FROM origins")
        known_origins = {r[0] for r in cursor.fetchall()}
        df_matched = df[df["origin_id"].notna()].copy()
        skipped_no_origin = int((~df_matched["origin_id"].isin(known_origins)).sum())
        df_matched = df_matched[df_matched["origin_id"].isin(known_origins)]
        skipped_no_event = 0
    else:
        cursor.execute("SELECT event_id, id FROM origins WHERE preferred = 1")
        preferred = {r[0]: r[1] for r in cursor.fetchall()}
        df_matched = df.copy()
        df_matched["origin_id"] = df_matched["event_id"].map(preferred)
        skipped_no_event = int(df_matched["origin_id"].isna().sum())
        skipped_no_origin = 0
        df_matched = df_matched[df_matched["origin_id"].notna()]

    score_cols = [
        "score", "n_candidates", "n_excluded", "n_used", "n_used_outside_radius",
        "n_active", "n_active_missing", "expected_weight", "missing_weight",
        "effective_radius_km", "reason", "radius_km", "threshold", "decay_factor",
    ]
    # Replace NaN with None for SQLite
    df_matched = df_matched.where(pd.notna(df_matched), None)

    rows = [
        (
            row["origin_id"],
            row["event_id"],
            row["score"],
            row["n_candidates"],
            row["n_excluded"],
            row["n_used"],
            row["n_used_outside_radius"],
            row["n_active"],
            row["n_active_missing"],
            row["expected_weight"],
            row["missing_weight"],
            row["effective_radius_km"],
            row["reason"],
            row["radius_km"],
            row["threshold"],
            row["decay_factor"],
        )
        for _, row in df_matched[["origin_id", "event_id"] + score_cols].iterrows()
    ]

    try:
        cursor.execute("BEGIN TRANSACTION")
        cursor.executemany(
            """
            INSERT OR REPLACE INTO silence_scores (
                origin_id, event_id,
                score, n_candidates, n_excluded, n_used, n_used_outside_radius,
                n_active, n_active_missing, expected_weight, missing_weight,
                effective_radius_km, reason, radius_km, threshold, decay_factor
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        conn.commit()

        total = len(df)
        inserted_count = len(rows)
        msg = (
            f"Silence scores: {inserted_count}/{total} inserted"
            + (f", {skipped_no_event} event not found" if skipped_no_event else "")
            + (f", {skipped_no_origin} origin not found" if skipped_no_origin else "")
        )
        logger.info(msg)
        print(f"  {msg}")

    except Exception as e:
        logger.error(f"Unexpected error in add_silence_score: {e}")
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

    _ensure_column(cursor, "origins", "quality", "TEXT")
    _ensure_column(cursor, "origins", "quality_factor", "DOUBLE")
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
            origin_latitude,
            origin_longitude,
            expectation_latitude,
            expectation_longitude,
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


def _fetch_station_phases(cursor: sqlite3.Cursor, origin_id: str) -> dict:
    """Return a dict mapping station_name -> set({'P', 'S'}) for used arrivals.

    Shared helper for compute_origin_station_score() and recompute_ps_ratio().
    Only arrivals with time_weight > 0 are considered.
    """
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
    station_phases: dict = {}
    for station_name, phase in cursor.fetchall():
        if station_name not in station_phases:
            station_phases[station_name] = set()
        if phase.startswith("p"):
            station_phases[station_name].add("P")
        elif phase.startswith("s"):
            station_phases[station_name].add("S")
    return station_phases


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

    _ensure_column(cursor, "origins", "station_score", "FLOAT DEFAULT 0.0")

    cursor.execute("SELECT id FROM origins WHERE preferred = 1")
    origins = cursor.fetchall()

    for (origin_id,) in origins:
        station_phases = _fetch_station_phases(cursor, origin_id)

        score = 0.0
        for phases in station_phases.values():
            if "P" in phases and "S" in phases:
                score += 2.0
            elif "P" in phases:
                score += 1.0
            elif "S" in phases:
                score += 0.5

        cursor.execute(
            "UPDATE origins SET station_score = ? WHERE id = ?", (score, origin_id)
        )

    conn.commit()
    logger.info("Station scores computation completed")


def recompute_ps_ratio(conn: sqlite3.Connection) -> None:
    """
    Recompute ps_ratio and ps_station_count for all preferred origins in the database.

    ps_ratio is the ratio of stations with both P and S phases over total stations
    with at least one used phase (time_weight > 0).
    ps_station_count is the absolute number of stations with both P and S phases.
    """
    logger.info("Recomputing ps_ratio and ps_station_count for all preferred origins...")
    cursor = conn.cursor()

    _ensure_column(cursor, "origins", "ps_ratio", "DOUBLE")
    _ensure_column(cursor, "origins", "ps_station_count", "INTEGER")

    cursor.execute("SELECT id FROM origins WHERE preferred = 1")
    origins = cursor.fetchall()

    for (origin_id,) in origins:
        station_phases = _fetch_station_phases(cursor, origin_id)

        total_stations = len(station_phases)
        if total_stations == 0:
            ps_ratio = None
            ps_station_count = 0
        else:
            stations_with_both = sum(
                1
                for phases in station_phases.values()
                if "P" in phases and "S" in phases
            )
            ps_ratio = stations_with_both / total_stations
            ps_station_count = stations_with_both

        cursor.execute(
            "UPDATE origins SET ps_ratio = ?, ps_station_count = ? WHERE id = ?",
            (ps_ratio, ps_station_count, origin_id)
        )

    conn.commit()
    logger.info("ps_ratio and ps_station_count recomputation completed")


def compute_median_probabilities(conn):
    """Compute and store median probabilities for P, S and all picks."""
    logger.info("Computing median probabilities for all origins...")

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
            median_prob_p = (
                SELECT COALESCE(MEDIAN(prob), 0.0)
                FROM pick_probs
                WHERE phase_hint LIKE 'P%'
            ),
            median_prob_s = (
                SELECT COALESCE(MEDIAN(prob), 0.0)
                FROM pick_probs
                WHERE phase_hint LIKE 'S%'
            ),
            median_prob_total = (
                SELECT COALESCE(MEDIAN(prob), 0.0)
                FROM pick_probs
            )
        WHERE id = ?
        """,
            (origin_id, origin_id),
        )

    conn.commit()
    logger.info("Median probabilities computation completed")


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
        # Check and add columns to events
        _ensure_column(cursor, "events", "agency_names", "JSON")
        _ensure_column(cursor, "events", "agency_ai_contributors", "JSON")
        _ensure_column(cursor, "events", "multiple_same_agencies", "BOOLEAN")

        # Check and add columns to origins
        _ensure_column(cursor, "origins", "station_score", "DOUBLE DEFAULT 0.0")
        _ensure_column(cursor, "origins", "ps_ratio", "DOUBLE")

        # Warn about deprecated avg_prob_* columns from older databases
        deprecated_cols = {"avg_prob_p", "avg_prob_s", "avg_prob_total"}
        found_deprecated = deprecated_cols & set(_get_table_columns(cursor, "origins"))
        if found_deprecated:
            logger.warning(
                "Deprecated columns detected in origins table: %s. "
                "These columns are no longer used and have been replaced by "
                "median_prob_p, median_prob_s, median_prob_total.",
                ", ".join(sorted(found_deprecated)),
            )

        _ensure_column(cursor, "origins", "median_prob_p", "DOUBLE DEFAULT 0.0")
        _ensure_column(cursor, "origins", "median_prob_s", "DOUBLE DEFAULT 0.0")
        _ensure_column(cursor, "origins", "median_prob_total", "DOUBLE DEFAULT 0.0")
        _ensure_column(cursor, "origins", "cpq", "DOUBLE")
        _ensure_column(cursor, "origins", "gallacher_gt5_status", "BOOLEAN")

        conn.commit()
        logger.info("All required columns verified/added successfully.")

    except Exception as e:
        logger.error(f"Error ensuring required columns exist: {e}")
        raise


def compute_gallacher_gt5_score(conn: sqlite3.Connection) -> None:
    """
    Compute and update Gallacher et al. (2025) GT5 metrics (cpq, gallacher_gt5_status)
    for all preferred origins in the database.

    Criteria applied:
        1. Five or more stations within 150 km
        2. CPQ >= 0.4
        3. Secondary Azimuthal Gap <= 210°
        4. One or more stations within 10 km OR five or more stations with both P & S
        5. Max distance >= 2° (teleseismic constraint)
        6. erh <= 5 km (error ellipse proxy)
        7. Depth is resolved (not fixed)
        8. Depth <= 35 km

    Args:
        conn: SQLite database connection
    """
    logger.info("Computing Gallacher GT5 metrics for all events...")
    cursor = conn.cursor()

    # Get all preferred origins with relevant fields
    cursor.execute(
        """
        SELECT id, num_stations_10km, num_stations_150km,
               secondary_azimuthal_gap, maximum_distance, depth,
               depth_type, erh
        FROM origins WHERE preferred = 1
        """
    )
    origins = cursor.fetchall()

    for row in origins:
        (
            origin_id,
            num_stations_10km,
            num_stations_150km,
            secondary_gap,
            max_distance_deg,
            depth_m,
            depth_type,
            erh,
        ) = row

        # Get arrivals with azimuth, distance and phase info
        # station_name contains "NET.STA" as stored at injection time
        cursor.execute(
            """
            SELECT a.azimuth, a.distance, a.name, p.station_name
            FROM arrivals a
            JOIN picks p ON a.pick_id = p.id
            WHERE a.origin_id = ? AND a.time_weight > 0
              AND a.azimuth IS NOT NULL AND a.distance IS NOT NULL
            """,
            (origin_id,),
        )
        arrivals = cursor.fetchall()

        if len(arrivals) < 3:
            cursor.execute(
                "UPDATE origins SET cpq = NULL, gallacher_gt5_status = 0 WHERE id = ?",
                (origin_id,),
            )
            continue

        azimuths = [a[0] for a in arrivals]
        distances_km = [a[1] * 111.11 for a in arrivals]

        # Stations with both P and S (station_name is "NET.STA")
        stations_with_p = set()
        stations_with_s = set()
        for _, _, phase, station_name in arrivals:
            if phase and station_name:
                ph = phase.upper()
                if ph.startswith("P"):
                    stations_with_p.add(station_name)
                elif ph.startswith("S"):
                    stations_with_s.add(station_name)
        num_stations_both_ps = len(stations_with_p & stations_with_s)

        cpq = compute_cpq(azimuths)

        # Recompute station counts from arrivals if DB values are missing
        n10 = num_stations_10km if num_stations_10km is not None else sum(1 for d in distances_km if d <= 10)
        n150 = num_stations_150km if num_stations_150km is not None else sum(1 for d in distances_km if d <= 150)

        depth_km = depth_m / 1000.0 if depth_m is not None else None
        depth_is_fixed = False
        if depth_type:
            dt = str(depth_type).lower()
            depth_is_fixed = "operator" in dt or "fixed" in dt

        gallacher_gt5 = all([
            n150 >= 5,
            cpq is not None and cpq >= 0.4,
            secondary_gap is not None and secondary_gap <= 210,
            (n10 >= 1) or (num_stations_both_ps >= 5),
            max_distance_deg is not None and max_distance_deg >= 2.0,
            erh is not None and erh <= 5.0,
            depth_km is not None and not depth_is_fixed,
            depth_km is not None and depth_km <= 35.0,
        ])

        cursor.execute(
            "UPDATE origins SET cpq = ?, gallacher_gt5_status = ? WHERE id = ?",
            (cpq, 1 if gallacher_gt5 else 0, origin_id),
        )

    conn.commit()
    logger.info("Gallacher GT5 metrics computation completed")


def print_discrimination_stats(conn: sqlite3.Connection) -> None:
    """Print a breakdown of event types after discrimination info has been applied."""
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT
            COALESCE(event_type, 'NULL') AS event_type,
            COUNT(*) AS count
        FROM events
        GROUP BY event_type
        ORDER BY count DESC;
        """
    )
    rows = cursor.fetchall()
    total = sum(r[1] for r in rows)
    print(f"  Discrimination stats ({total} total events):")
    for event_type, count in rows:
        print(f"    {event_type}: {count} ({100.0 * count / total:.1f}%)")


def apply_database_enhancements(args) -> None:
    """Apply database enhancements based on the provided arguments."""
    print("Applying database enhancements...")
    conn = None
    try:
        if any(
            [
                args.compute_station_scores,
                args.compute_ps_ratio,
                args.add_discrimination,
                args.add_localization_quality,
                args.add_agency_names,
                args.gt5,
                args.gallacher_gt5,
                args.compute_prob_median,
                args.add_silence_score,
                args.refresh_view,
            ]
        ):
            conn = create_schema(args.database)

            # Always ensure all supplemental columns exist
            ensure_required_columns_exist(conn)

            if args.compute_station_scores:
                print("Computing station scores...")
                compute_origin_station_score(conn)

            if args.compute_ps_ratio:
                print("Recomputing ps_ratio...")
                recompute_ps_ratio(conn)

            if args.add_discrimination:
                print("Adding discrimination info...")
                add_discrimination_info(conn, args.add_discrimination)
                print_discrimination_stats(conn)

            if args.add_localization_quality:
                print("Computing localization quality...")
                add_compute_localization_quality(conn)

            if args.add_agency_names:
                print("Adding agency names...")
                add_agency_names(conn)

            if args.compute_prob_median:
                print("Computing median probabilities...")
                compute_median_probabilities(conn)

            if args.gt5:
                print("Computing GT5 metrics...")
                compute_gt5_score(conn)

            if args.gallacher_gt5:
                print("Computing Gallacher GT5 metrics (cpq + gallacher_gt5_status)...")
                compute_gallacher_gt5_score(conn)

            if args.add_silence_score:
                print("Adding silence scores...")
                add_silence_score(conn, args.add_silence_score)

            if any(
                [
                    args.compute_station_scores,
                    args.compute_ps_ratio,
                    args.add_discrimination,
                    args.add_localization_quality,
                    args.add_agency_names,
                    args.gt5,
                    args.gallacher_gt5,
                    args.compute_prob_median,
                    args.add_silence_score,
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
    """Validate date string format (YYYY-MM-DD) and normalize to start of day."""
    try:
        datetime.strptime(date_str, "%Y-%m-%d")
        return date_str + " 00:00:00"
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Invalid date format: {date_str}. Use YYYY-MM-DD"
        )


def validate_end_date(date_str: str) -> str:
    """Validate date string format (YYYY-MM-DD) and normalize to start of the *next* day.

    This ensures --end-time covers the full last day (exclusive upper bound).
    Example: --end-time 2024-01-15  →  '2024-01-16 00:00:00'
    SQL condition  e.time < '2024-01-16 00:00:00'  includes all events on 2024-01-15.
    """
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        next_day = dt + pd.Timedelta(days=1)
        return next_day.strftime("%Y-%m-%d 00:00:00")
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
        "--input-list",
        type=validate_file_exists,
        help="Text file containing QuakeML paths to import (one per line).",
    )
    import_group.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Number of events to accumulate before importing into the database.",
    )
    import_group.add_argument(
        "--sqlite-batch-size",
        type=int,
        default=100,
        help="Number of events per SQLite transaction commit during import.",
    )
    import_group.add_argument(
        "--fast-import",
        action="store_true",
        help="Enable fast import mode: disables foreign_keys and uses aggressive SQLite pragmas for bulk loading.",
    )
    import_group.add_argument(
        "-q",
        "--enable-quakeml",
        action="store_true",
        help="Store full QuakeML data in the database (increases size).",
    )
    import_group.add_argument(
        "--fix-quality",
        action="store_true",
        help="Repair missing origin quality by reconstructing from arrival data.",
    )
    import_group.add_argument(
        "--ignore-missing-picks",
        action="store_true",
        help="Ignore arrivals referencing missing picks instead of rejecting the event.",
    )
    import_group.add_argument(
        "--log-file",
        dest="log_file",
        type=str,
        default=None,
        help="Write import warnings and errors to a log file for debugging.",
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
        type=validate_end_date,
        help="End time for export (YYYY-MM-DD, inclusive — covers the full day).",
    )

    # Database enhancements
    enhancement_group = parser.add_argument_group("Database Enhancements")
    enhancement_group.add_argument(
        "--compute-station-scores",
        action="store_true",
        help="Compute and store station scores for all origins.",
    )
    enhancement_group.add_argument(
        "--compute-ps-ratio",
        action="store_true",
        help="Recompute ps_ratio for all preferred origins based on P/S phase distribution.",
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
        "--gallacher-gt5",
        action="store_true",
        help="Compute Gallacher et al. (2025) revised GT5 metrics (cpq, gallacher_gt5_status).",
    )
    enhancement_group.add_argument(
        "--compute-prob-median",
        action="store_true",
        help="Compute median probabilities for P, S and total picks.",
    )
    enhancement_group.add_argument(
        "--add-silence-score",
        type=validate_file_exists,
        help="Import silence scores from CSV file (output of silence-score tool).",
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

    if args.batch_size <= 0:
        parser.error("--batch-size must be a positive integer")

    if args.sqlite_batch_size <= 0:
        parser.error("--sqlite-batch-size must be a positive integer")

    # Warn about fast-import risks
    if args.fast_import:
        print(
            "WARNING: --fast-import disables foreign key constraints and reduces durability. "
            "Use only for initial bulk loading with reliable data.",
            file=sys.stderr,
        )

    # Validate at least one action is specified if no input files are provided
    if (
        not args.input
        and not args.input_list
        and not any(
            [
                args.csv_output,
                args.export_quakeml,
                args.add_discrimination,
                args.add_localization_quality,
                args.add_agency_names,
                args.gt5,
                args.gallacher_gt5,
                args.compute_prob_median,
                args.compute_station_scores,
                args.compute_ps_ratio,
                args.add_silence_score,
                args.refresh_view,
            ]
        )
    ):
        parser.error(
            "No action requested. Please specify at least one action (import, export, or enhancement option)"
        )

    return args


def collect_input_files(args: argparse.Namespace) -> List[str]:
    """Collect input file paths from CLI arguments and optional list file."""
    input_files: List[str] = []

    if args.input:
        input_files.extend(args.input)

    if args.input_list:
        listed_files: List[str] = []
        try:
            with open(args.input_list, "r", encoding="utf-8") as list_file:
                for line in list_file:
                    path = line.strip()
                    if not path or path.startswith("#"):
                        continue
                    expanded_path = os.path.expanduser(path)
                    if not os.path.exists(expanded_path):
                        raise FileNotFoundError(
                            f"File listed in {args.input_list} does not exist: {path}"
                        )
                    listed_files.append(expanded_path)
        except Exception as exc:
            logger.error(f"Error reading input list {args.input_list}: {exc}")
            raise

        input_files.extend(listed_files)
        print(f"Loaded {len(listed_files)} files from list {args.input_list}")

    return input_files


def process_quakeml_import(args: argparse.Namespace, input_files: List[str]) -> None:
    """Import QuakeML files with batching support for large datasets."""
    logger.info(f"Starting import of {len(input_files)} QuakeML files")
    logger.info(f"Batch size: {args.batch_size} events")
    logger.info(f"SQLite batch size: {args.sqlite_batch_size} events per commit")
    if args.fast_import:
        logger.info("Fast import mode enabled: using aggressive SQLite pragmas")

    conn = create_schema(args.database)
    file_handler: Optional[logging.Handler] = None

    try:
        if args.log_file:
            file_handler = logging.FileHandler(args.log_file, mode="w")
            file_handler.setLevel(logging.WARNING)
            file_handler.setFormatter(
                logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
            )
            logging.getLogger().addHandler(file_handler)
            print(f"Logging warnings/errors to: {args.log_file}")

        # Apply fast-import pragmas if enabled
        if args.fast_import:
            logger.info("Applying fast-import SQLite pragmas...")
            conn.execute("PRAGMA foreign_keys = OFF;")
            conn.execute("PRAGMA synchronous = OFF;")
            conn.execute("PRAGMA journal_mode = MEMORY;")
            conn.execute("PRAGMA cache_size = -64000;")  # 64 MB cache (vs 2 MB default)
            conn.commit()
            logger.info("Fast-import pragmas applied")

        parsed_count = 0
        parse_error_count = 0
        total_events = 0
        imported_count = 0
        duplicate_count = 0
        malformed_count = 0

        batch_catalog = Catalog()

        def flush_batch() -> None:
            nonlocal batch_catalog, imported_count, duplicate_count, malformed_count
            if len(batch_catalog) == 0:
                return
            success, dups, malformed = import_catalog_to_sqlite(
                conn,
                batch_catalog,
                args.enable_quakeml,
                fix_quality=args.fix_quality,
                ignore_missing_picks=args.ignore_missing_picks,
                sqlite_batch_size=args.sqlite_batch_size,
            )
            imported_count += success
            duplicate_count += dups
            malformed_count += malformed
            batch_catalog = Catalog()

        with tqdm(
            total=len(input_files), desc="Processing QuakeML files", unit="file"
        ) as pbar:
            for input_file in input_files:
                try:
                    logger.info(f"Reading catalog from file '{input_file}'...")
                    catalog = read_events(input_file)

                    parsed_count += 1
                    num_events = len(catalog)
                    total_events += num_events

                    for event in catalog:
                        batch_catalog.events.append(event)
                        if len(batch_catalog) >= args.batch_size:
                            flush_batch()

                    current_ok = imported_count
                    pbar.set_postfix(
                        {
                            "parsed": parsed_count,
                            "ok": current_ok,
                            "dup": duplicate_count,
                            "bad": malformed_count,
                        }
                    )
                    pbar.update(1)

                except Exception as exc:
                    parse_error_count += 1
                    logger.error(f"Failed to parse {input_file}: {exc}")
                    pbar.set_postfix(
                        {
                            "parsed": parsed_count,
                            "parse_errors": parse_error_count,
                        }
                    )
                    pbar.update(1)
                    continue

        # Flush any remaining events
        flush_batch()

        imported_count = total_events - duplicate_count - malformed_count

        print("\nProcessing completed:")
        print(f"  Files:  {parsed_count} parsed, {parse_error_count} unreadable")
        print(
            "  Events: "
            f"{total_events} found, {imported_count} imported, "
            f"{duplicate_count} duplicates, {malformed_count} malformed"
        )

        add_agency_names(conn)

        print("Creating database indexes...")
        cursor = conn.cursor()
        create_indexes_sql(cursor)
        print("Indexes created successfully.")

        register_geometry_for_view(conn, "event_coordinates", "geometry")

        # Restore normal SQLite pragmas if fast-import was used
        if args.fast_import:
            logger.info("Restoring normal SQLite pragmas...")
            conn.execute("PRAGMA foreign_keys = ON;")
            conn.execute("PRAGMA synchronous = NORMAL;")
            conn.execute("PRAGMA journal_mode = WAL;")
            conn.commit()
            logger.info("Normal pragmas restored")

    finally:
        if file_handler:
            logging.getLogger().removeHandler(file_handler)
            file_handler.close()
        if conn:
            conn.close()


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
            args.compute_prob_median,
            args.compute_station_scores,
            args.compute_ps_ratio,
            args.add_silence_score,
            args.refresh_view,
        ]

        if any(db_operations) and not os.path.exists(args.database):
            print(f"Error: Database '{args.database}' does not exist.", file=sys.stderr)
            sys.exit(1)

        # Handle input files (optionally via --input-list)
        input_files = collect_input_files(args)
        if input_files:
            process_quakeml_import(args, input_files)

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
            "compute_prob_median": args.compute_prob_median,
            "compute_station_scores": args.compute_station_scores,
            "compute_ps_ratio": args.compute_ps_ratio,
            "gallacher_gt5": args.gallacher_gt5,
            "add_silence_score": args.add_silence_score,
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
