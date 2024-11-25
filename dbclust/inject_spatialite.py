#!/usr/bin/env python
import argparse
import datetime
import logging
import sqlite3
import xml.etree.ElementTree as ET
import zlib
from io import BytesIO

from icecream import ic
from obspy import Catalog
from obspy import read_events
from obspy import UTCDateTime
from obspy.core.event import Event

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("inject_spatialite")
logger.setLevel(logging.INFO)

"""
Processes QuakeML files and stores the data in a SpatiaLite-enabled SQLite database.
"""


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
        event (obspy.core.event.Event): The earthquake event object containing event details.
        quakeml (str): The QuakeML XML string representing the event.

    Raises:
        Exception: If any database operation fails, the transaction is rolled back and the exception is raised.
    """

    # Enable SpatiaLite
    conn.enable_load_extension(True)
    conn.load_extension("mod_spatialite")

    try:
        # Start a transaction
        conn.execute("BEGIN TRANSACTION;")

        # Insert full QuakeML data
        conn.execute(
            """
            INSERT INTO quakeml (event_id, data)
            VALUES (?, ?)
            """,
            (event.resource_id.id, quakeml),
        )

        # Insert event details
        preferred_origin = event.preferred_origin()
        conn.execute(
            """
            INSERT INTO events (id, time, latitude, longitude, depth, magnitude, geometry, event_type,
                                latitude_uncertainty, longitude_uncertainty,
                                depth_uncertainty, event_id)
            VALUES (?, ?, ?, ?, ?, ?, ST_GeomFromText(?, 4326), ?, ?, ?, ?, ?)
            """,
            (
                event.resource_id.id,
                to_datetime(preferred_origin.time),
                preferred_origin.latitude,
                preferred_origin.longitude,
                preferred_origin.depth,
                event.magnitudes[0].mag if event.magnitudes else None,
                f"POINT({preferred_origin.longitude} {preferred_origin.latitude})",  # Géométrie en WKT
                event.event_type if hasattr(event, "event_type") else None,
                (
                    preferred_origin.latitude_errors.uncertainty
                    if preferred_origin.latitude_errors
                    else None
                ),
                (
                    preferred_origin.longitude_errors.uncertainty
                    if preferred_origin.longitude_errors
                    else None
                ),
                (
                    preferred_origin.depth_errors.uncertainty
                    if preferred_origin.depth_errors
                    else None
                ),
                event.resource_id.id,
            ),
        )

        # Insert origins
        for origin in event.origins:
            conn.execute(
                """
                INSERT INTO origins (id, event_id, time, latitude, longitude, depth, depth_type, evaluation_mode, preferred)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    origin.resource_id.id,
                    event.resource_id.id,
                    to_datetime(origin.time),
                    origin.latitude,
                    origin.longitude,
                    origin.depth,
                    origin.depth_type,
                    origin.evaluation_mode,
                    (
                        1
                        if origin.resource_id == event.preferred_origin().resource_id
                        else 0
                    ),
                ),
            )

        # Insert picks
        for pick in event.picks:
            conn.execute(
                """
                INSERT INTO picks (id, event_id, station_name, pick_time, uncertainty)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    pick.resource_id.id,
                    event.resource_id.id,
                    pick.waveform_id.station_code,
                    to_datetime(pick.time),
                    pick.time_errors.uncertainty,
                ),
            )

        # Insert arrivals
        for origin in event.origins:
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

        # Commit the transaction
        conn.execute("COMMIT;")
        logger.info(f"Event {event.resource_id.id} inserted.")

    except Exception as e:
        # Rollback the transaction if an error occurs
        conn.execute("ROLLBACK;")
        raise e


def export_sqlite_to_quakeml(db_path: str, output_file: str):
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
        for row in cursor.execute("SELECT data FROM quakeml;"):
            # Each row contains a compressed QuakeML
            compressed_quakeml_data = row[0]
            if compressed_quakeml_data is None:
                continue

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
                            elem.tag = elem.tag.split("}", 1)[
                                1
                            ]  # Remove namespace part

                    # Write the event directly to the output file
                    f.write(ET.tostring(event, encoding="utf-8"))

            except ET.ParseError as e:
                print(f"Error parsing QuakeML: {e}")

        # Close the eventParameters tag and the root tag
        f.write(b"  </eventParameters>\n")
        f.write(b"</q:quakeml>\n")

    # Close the database connection
    conn.close()
    print(f"Concatenated QuakeML written to {output_file}")


def create_schema(db_path: str) -> sqlite3.Connection:
    """
    Create the database schema for a SpatiaLite-enabled SQLite database.
    This function initializes the SpatiaLite metadata, creates necessary tables,
    and adds a geometry column to the 'events' table if it does not already exist.

    Args:
        db_path (str): The file path to the SQLite database.
    Returns:
        sqlite3.Connection: The connection object to the SQLite database.
    """

    # SpatiaLite
    conn = sqlite3.connect(db_path)

    # Set WAL mode: to be able to read and write at the same time
    # https://www.sqlite.org/wal.html
    conn.execute("PRAGMA journal_mode=WAL;")

    #  SpatiaLite metadata initialization
    conn.enable_load_extension(True)
    conn.load_extension("mod_spatialite")
    cursor = conn.cursor()
    cursor.execute(
        "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='spatial_ref_sys';"
    )
    if cursor.fetchone()[0] == 0:
        logger.info("Initializing SpatiaLite metadata...")
        cursor.execute("SELECT InitSpatialMetadata();")

    # quakeml table
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS quakeml (
            event_id TEXT PRIMARY KEY,
            data BLOB
        );
        """
    )

    # events table
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS events (
            id TEXT PRIMARY KEY,
            time TIMESTAMP,
            latitude DOUBLE,
            longitude DOUBLE,
            depth DOUBLE,
            magnitude DOUBLE,
            event_type TEXT,
            latitude_uncertainty DOUBLE,
            longitude_uncertainty DOUBLE,
            depth_uncertainty DOUBLE,
            event_id TEXT REFERENCES quakeml(event_id)
        );
        """
    )

    # add geometry column to events table
    # only if it does not already exist
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(events);")
    columns = [row[1] for row in cursor.fetchall()]
    if "geometry" not in columns:
        logger.info("Adding geometry column to events table...")
        conn.execute(
            """
            SELECT AddGeometryColumn('events', 'geometry', 4326, 'POINT', 'XY');
        """
        )

    # picks table
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS picks (
            id TEXT PRIMARY KEY,
            event_id TEXT REFERENCES events(id),
            station_name TEXT,
            pick_time TIMESTAMP,
            uncertainty DOUBLE
        );
        """
    )

    # origins table
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS origins (
            id TEXT PRIMARY KEY,
            event_id TEXT REFERENCES events(id),
            time TIMESTAMP,
            latitude DOUBLE,
            longitude DOUBLE,
            depth DOUBLE,
            depth_type TEXT,
            evaluation_mode TEXT,
            preferred BOOLEAN
        );
        """
    )

    # arrivals table
    conn.execute(
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
        """
    )

    return conn


def export_catalog_to_sqlite(
    database: str, catalog: Catalog, enable_quakeml: bool = False
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

    # Create schema
    conn = create_schema(database)

    # Process events and insert into SQLite
    for event in catalog:
        # Serialize QuakeML content using format and compress it
        if enable_quakeml:
            quakeml_data = compress_quakeml_data(event, format="QUAKEML")
        else:
            quakeml_data = None

        try:
            inject_event(conn, event, quakeml_data)
        except Exception as e:
            logger.error(f"event {event.resource_id.id}: {e}")
    conn.close()


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Process QuakeML files and store in SQLite."
    )
    parser.add_argument(
        "-i", "--input", required=True, help="Path to the input QuakeML file."
    )
    parser.add_argument(
        "-d",
        "--database",
        default="seismic_data.sqlite",
        help="Path to the SQLite database.",
    )
    parser.add_argument(
        "-q",
        "--enable-quakeml",
        action="store_true",
        default=False,
        help="write full quakeml in the existing database file.",
    )
    args = parser.parse_args()

    # Read QuakeML file
    # catalog = read_events(args.input)
    # export_catalog_to_sqlite(args.database, catalog, args.enable_quakeml)

    # Export QuakeML data to a file
    export_sqlite_to_quakeml(args.database, "output_quakeml.xml")
