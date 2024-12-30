#!/usr/bin/env python
"""
    Processes QuakeML files and stores the data in a SpatiaLite-enabled SQLite database.
"""
import argparse
import csv
import datetime
import json
import logging
import math
import os
import re
import sqlite3
import sys
import xml.etree.ElementTree as ET
import zlib
from collections import Counter
from io import BytesIO
from typing import List
from typing import Tuple

import numpy as np
import pandas as pd
from icecream import ic
from localization_quality import classify_Michele_mod
from obspy import Catalog
from obspy import read_events
from obspy import UTCDateTime
from obspy.core.event import Event
from obspy.core.event import Origin

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("inject_spatialite")
logger.setLevel(logging.INFO)


# SQL for creating the event coordinates view
EVENT_COORDINATES_VIEW = """
CREATE VIEW IF NOT EXISTS event_coordinates AS
SELECT
    e.event_id,
    o.time, o.latitude, o.longitude, o.depth,
    o.rms, o.erh, o.erz, o.er_method,
    o.used_station_count, o.used_phase_count, o.P_count, o.S_count,
    o.minimum_distance, o.maximum_distance, o.median_distance,
    o.azimuthal_gap, o.secondary_azimuthal_gap,
    o.scatter_volume, e.dist_km_from_preloc,
    e.nb_agencies, e.agencies_list, e.agency_names, e.multiple_same_agencies,
    o.evaluation_mode,
    e.event_type, e.discrimination_probability, e.discrimination_station_count, e.discrimination_certainty,
    o.quality, o.quality_factor,
    o.geometry
FROM
    events AS e
JOIN
    origins AS o ON e.event_id = o.event_id
WHERE
    o.preferred = 1;
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
    Calculate the values of erh (horizontal uncertainty) and erz (vertical uncertainty).

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


def to_datetime(utc_datetime: UTCDateTime) -> datetime.datetime:
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

    try:
        with conn:
            logger.info(f"Inserting event {event.resource_id.id}.")

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
                INSERT INTO events (event_id, event_type, dist_km_from_preloc, nb_agencies, agencies_list)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    event.resource_id.id,
                    event.event_type,
                    get_distance_km_info(event),
                    len(all_agencies_ids),
                    agencies_list_str,
                ),
            )

            # Insert origins
            logger.debug(f"Inserting origins for event {event.resource_id.id}.")
            for origin in event.origins:
                insert_origin(conn, origin, event)

            # Insert picks
            logger.debug(f"Inserting picks for event {event.resource_id.id}.")
            for pick in event.picks:
                agency_id = (
                    pick.creation_info.agency_id
                    if pick.creation_info and hasattr(pick.creation_info, "agency_id")
                    else None
                )
                probability = get_pick_probability(pick)
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
                        pick.waveform_id.station_code,
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

            logger.info(f"Event {event.resource_id.id} successfully inserted.")

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
    scatter_volume = get_scatter_volume(origin)

    try:
        quality_factor, quality = classify_Michele_mod(
            rms,
            erh,
            erz,
            q.used_phase_count,
            q.minimum_distance,
            q.median_distance,
            q.azimuthal_gap,
            q.secondary_azimuthal_gap,
            scatter_volume,
        )
    except Exception as e:
        quality_factor = None
        quality = None

    conn.execute(
        """
        INSERT INTO origins (
            id, event_id, time, latitude, longitude, depth, depth_type,
            rms, erh, erz, er_method,
            used_station_count, used_phase_count, P_count, S_count,
            minimum_distance, maximum_distance, median_distance,
            azimuthal_gap, secondary_azimuthal_gap,
            scatter_volume, quality, quality_factor,
            evaluation_mode, preferred, geometry
        )
        VALUES (
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?,
            ST_GeomFromText(?, 4326)
        )
        """,
        (
            origin.resource_id.id,
            event.resource_id.id,
            to_datetime(origin.time),
            origin.latitude,
            origin.longitude,
            origin.depth,
            origin.depth_type,
            rms,
            erh,
            erz,
            err_method,
            q.used_station_count,
            q.used_phase_count,
            P_count,
            S_count,
            q.minimum_distance,
            q.maximum_distance,
            q.median_distance,
            q.azimuthal_gap,
            q.secondary_azimuthal_gap,
            get_scatter_volume(origin),
            quality,
            quality_factor,
            origin.evaluation_mode,
            1 if origin.resource_id == event.preferred_origin().resource_id else 0,
            f"POINT({origin.longitude} {origin.latitude})",
        ),
    )
    logger.debug(f"Origin {origin.resource_id.id} inserted.")


def insert_arrivals(conn: sqlite3.Connection, origin: Origin) -> None:
    """Inserts arrivals associated with an origin into the database."""
    for arrival in origin.arrivals:
        pick_id = arrival.pick_id.id if arrival.pick_id else None
        conn.execute(
            """
            INSERT INTO arrivals (id, origin_id, pick_id, name, takeoff_angle, azimuth, distance)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                arrival.resource_id.id,
                origin.resource_id.id,
                pick_id,
                arrival.phase,
                arrival.takeoff_angle,
                arrival.azimuth,
                arrival.distance,
            ),
        )
        logger.debug(f"Arrival {arrival.resource_id.id} inserted.")


def export_sqlite_to_quakeml(
    db_path: str, output_file: str, event_ids: List[str] = None
) -> None:
    """
    Concatenate multiple QuakeML streams stored in a database into a single XML file,
    minimizing memory usage by writing to the file incrementally.

    Args:
        db_path (str): Path to the SQLite database.
        output_file (str): Path to the output QuakeML file.
    """
    # Connect to the SQLite database
    conn = sqlite3.connect(db_path)
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
            query = "SELECT data FROM quakeml;"
            for row in cursor.execute(query):
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
        conn.execute(
            "PRAGMA journal_mode=WAL;"
        )  # Enable WAL mode for concurrent read/write
        conn.enable_load_extension(True)

        try:
            conn.load_extension("mod_spatialite")
        except sqlite3.OperationalError as e:
            logger.error(f"Failed to load SpatiaLite extension: {e}")
            raise

        with conn:
            cursor = conn.cursor()

            # Initialize SpatiaLite metadata if not already initialized
            cursor.execute(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='spatial_ref_sys';"
            )
            if cursor.fetchone()[0] == 0:
                logger.info("Initializing SpatiaLite metadata...")
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
            logger.info("Database schema created successfully.")

        return conn

    except Exception as e:
        logger.error(f"Failed to create schema: {e}")
        raise


def create_tables(cursor: sqlite3.Cursor) -> None:
    """
    Create the required tables in the database.

    Args:
        cursor (sqlite3.Cursor): The database cursor.
    """
    logger.info("Creating database tables...")

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
            event_id TEXT PRIMARY KEY REFERENCES quakeml(event_id),
            event_type TEXT,
            dist_km_from_preloc DOUBLE,
            discrimination_probability DOUBLE,
            discrimination_station_count INTEGER,
            discrimination_certainty DOUBLE,
            nb_agencies INTEGER,
            agencies_list JSON,
            agency_names TEXT,
            multiple_same_agencies BOOLEAN
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS picks (
            id TEXT PRIMARY KEY,
            event_id TEXT REFERENCES events(event_id),
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
            event_id TEXT REFERENCES events(event_id),
            time TIMESTAMP,
            latitude DOUBLE,
            longitude DOUBLE,
            depth DOUBLE,
            depth_type TEXT,
            rms DOUBLE,
            erh DOUBLE,
            erz DOUBLE,
            er_method TEXT,
            used_station_count INTEGER,
            used_phase_count INTEGER,
            P_count INTEGER,
            S_count INTEGER,
            minimum_distance DOUBLE,
            maximum_distance DOUBLE,
            median_distance DOUBLE,
            azimuthal_gap DOUBLE,
            secondary_azimuthal_gap DOUBLE,
            scatter_volume DOUBLE,
            quality TEXT,
            quality_factor DOUBLE,
            evaluation_mode TEXT,
            preferred BOOLEAN
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS arrivals (
            id TEXT PRIMARY KEY,
            origin_id TEXT REFERENCES origins(id),
            pick_id TEXT REFERENCES picks(id),
            name TEXT,
            takeoff_angle DOUBLE,
            azimuth DOUBLE,
            distance DOUBLE
        );
        """,
    ]

    # Execute each table creation SQL
    for sql in tables_sql:
        cursor.execute(sql)


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
    db_path: str, catalog: Catalog, enable_quakeml: bool = False, retries: int = 5, delay: int = 1
):
    """
    Import a catalog of seismic events to a SQLite database with conflict handling.

    Args:
        db_path (str): Path to the SQLite database file.
        catalog (Catalog): A catalog of seismic events.
        enable_quakeml (bool, optional): If True, serialize and compress QuakeML content for each event. Defaults to False.
        retries (int, optional): Number of retry attempts in case of OperationalError. Defaults to 5.
        delay (int, optional): Delay (in seconds) between retry attempts. Defaults to 1.
    """
    attempt = 0
    while attempt < retries:
        try:
            # Connect to the database
            conn = sqlite3.connect(db_path)
            logger.info("Connected to the database successfully.")

            # Import the catalog into the SQLite database
            import_catalog_to_sqlite(conn, catalog, enable_quakeml)

            # Optional: Additional operations
            # add_agency_names(conn)
            # add_compute_localization_quality(conn)

            conn.close()
            logger.info("Catalog imported successfully.")
            return  # Exit on success

        except sqlite3.OperationalError as e:
            logger.warning(
                f"Database is locked or unavailable (attempt {attempt + 1}/{retries}): {e}"
            )
            attempt += 1
            time.sleep(delay)  # Wait before retrying
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            raise e  # Raise unexpected errors
        finally:
            if 'conn' in locals() and conn:
                conn.close()

    # If we exhausted retries
    logger.error("Failed to import catalog after multiple attempts.")
    raise sqlite3.OperationalError("Unable to access the database after several retries.")


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
    catalog = read_events(catalog_file)

    # Import the catalog into the SQLite database
    import_catalog_to_sqlite(conn, catalog, enable_quakeml)

    # extract agency names and stats to event table
    add_agency_names(conn)

    # Register the geometry column for the 'event_coordinates' view
    register_geometry_for_view(conn, "event_coordinates", "geometry")
    conn.close()


def import_catalog_to_sqlite(
    conn: sqlite3.Connection, catalog: Catalog, enable_quakeml: bool = False
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
    for event in catalog:
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
    Export a SQLite view to a CSV file, excluding the 'geometry' column.
    The 'time' column is formatted as UTC datetime.

    Args:
        db_path (str): Path to the SQLite database.
        view_name (str): Name of the view to export.
        output_csv (str): Path to the output CSV file.
    """
    print(f"Exporting view '{view_name}' to '{output_csv}' without 'geometry'...")

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        # Get column names from the view, excluding 'geometry'
        cursor.execute(f"SELECT * FROM {view_name} WHERE 1=0;")
        column_names = [desc[0] for desc in cursor.description if desc[0] != "geometry"]

        if not column_names:
            raise ValueError(f"The view '{view_name}' has no columns to export (or only 'geometry').")

        formatted_columns = [
            "strftime('%Y-%m-%dT%H:%M:%fZ', time) AS time" if col == "time" else
            f"ROUND({col}, 1) AS {col}" if col == "depth" else
            f"ROUND({col}, 2) AS {col}" if col in [
                "quality_factor", "scatter_volume", "azimuthal_gap", "secondary_azimuthal_gap",
                "minimum_distance", "maximum_distance", "median_distance", "rms",
                "erh", "erz", "uncertainty", "dist_km_from_preloc",
                "discrimination_probability", "discrimination_certainty"
            ] else
            col
            for col in column_names
        ]

        selected_columns = ", ".join(formatted_columns)

        # Prepare and execute the query
        query = f"SELECT {selected_columns} FROM {view_name} ORDER BY time;"
        cursor.execute(query)

        # Write the data to CSV
        with open(output_csv, mode="w", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(column_names)  # Write the header
            writer.writerows(cursor.fetchall())  # Write the rows

        print(f"View '{view_name}' exported successfully to '{output_csv}' without 'geometry'.")
    except sqlite3.OperationalError as e:
        print(f"Error: Unable to export view '{view_name}'. {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
    finally:
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

    # check if the columns exist
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
        print(f"CSV file '{csv_file}' is missing required columns.")
        return

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
        # output file should not already exist
        if os.path.exists(args.csv_output):
            print(f"Output file '{args.csv_output}' already exists.")
            sys.exit(1)
        # Export the view to a CSV file
        export_view_to_csv_exclude_geometry(
            args.database, "event_coordinates", args.csv_output
        )
    elif args.event_id_csv:
        # read event_id from csv file using pandas
        event_ids = pd.read_csv(args.event_id_csv)["event_id"].tolist()
        export_sqlite_to_quakeml(args.database, args.export_quakeml, event_ids)
    elif args.export_quakeml:
        # Export QuakeML data to a file
        export_sqlite_to_quakeml(args.database, args.export_quakeml, args.event_id)
    elif args.add_discrimination:
        # Add discrimination info to the event table
        conn = sqlite3.connect(args.database)
        add_discrimination_info(conn, args.add_discrimination)
        refresh_event_coordinates_view(conn)
        conn.close()
    elif args.add_localization_quality:
        # Add localisation quality info to the event table
        conn = sqlite3.connect(args.database)
        add_compute_localization_quality(conn)
        refresh_event_coordinates_view(conn)
        conn.close()
    elif args.add_agency_names:
        # Add agency names to the event table
        conn = sqlite3.connect(args.database)
        add_agency_names(conn)
        refresh_event_coordinates_view(conn)
        conn.close()
    else:
        parser.print_help()
        sys.exit(1)
