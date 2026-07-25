#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import os
import sqlite3

import geopandas as gpd
import pandas as pd
from shapely import wkb

from dbclust.db import validate_sql_identifier


def create_filtered_view(conn: sqlite3.Connection, shape_table: str) -> None:
    # Validate table name to prevent SQL injection
    shape_table = validate_sql_identifier(shape_table)
    query = f"""
    CREATE VIEW IF NOT EXISTS event_coordinates_filtered AS
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
        e.nb_agencies, e.agencies_list, e.agency_names, e.agency_ai_contributors, e.multiple_same_agencies,
        o.evaluation_mode,
        e.event_type,
        e.discrimination_probability,
        e.discrimination_station_count,
        e.discrimination_certainty,
        o.quality, o.quality_factor,
        o.gt5_status, o.delta_U, o.num_stations_10km, o.num_stations_30km, o.num_stations_150km
    FROM
        events AS e
    JOIN
        origins AS o
        ON e.event_id = o.event_id AND o.preferred = 1  -- INNER JOIN because a preferred origin always exists
    LEFT JOIN
        magnitudes AS m
        ON e.event_id = m.event_id AND m.preferred = 1
    JOIN
        {shape_table} s ON ST_Within(o.geometry, s.geometry)
    WHERE
        COALESCE(e.event_type, '') NOT IN ('not existing', 'not locatable');
    """

    conn.execute(query)
    conn.commit()
    print(f"View event_coordinates_filtered successfully created")


def get_spatialite_connection(db_path: str) -> sqlite3.Connection:
    """Establish a connection to the SQLite database with Spatialite extension."""

    conn = sqlite3.connect(db_path)
    conn.enable_load_extension(True)

    # Load the Spatialite extension (adjust the path if necessary)
    try:
        conn.execute("SELECT load_extension('mod_spatialite')")
    except sqlite3.OperationalError:
        try:
            # Alternative path on some systems
            conn.execute("SELECT load_extension('libspatialite')")
        except sqlite3.OperationalError:
            print("Unable to load the Spatialite extension. Check its installation.")
            return None

    # Initialize spatial configuration
    conn.execute("SELECT InitSpatialMetaData(1)")
    return conn


def import_shapefile_to_db(conn: sqlite3.Connection, shapefile_path: str, table_name: str) -> None:
    """Import a shapefile into the SQLite database with Spatialite."""
    # Validate table name to prevent SQL injection
    table_name = validate_sql_identifier(table_name)

    # Read the shapefile with geopandas
    gdf = gpd.read_file(shapefile_path)

    # Ensure the CRS is in WGS84 (EPSG:4326) if necessary
    if gdf.crs != "EPSG:4326":
        gdf = gdf.to_crs("EPSG:4326")

    # Drop the table if it already exists
    conn.execute(f"DROP TABLE IF EXISTS {table_name}")

    # Create the table with a geometry column
    conn.execute(
        f"""
        CREATE TABLE {table_name} (
            id INTEGER PRIMARY KEY,
            name TEXT
        )
    """
    )

    # Add the geometry column
    conn.execute(
        f"SELECT AddGeometryColumn('{table_name}', 'geometry', 4326, 'POLYGON', 'XY')"
    )

    # Insert the data
    for idx, row in gdf.iterrows():
        geom_wkb = row.geometry.wkb
        name = f"Zone {idx}" if "name" not in row else row["name"]
        conn.execute(
            f"""
            INSERT INTO {table_name} (id, name, geometry)
            VALUES (?, ?, ST_GeomFromWKB(?, 4326))
        """,
            (idx, name, geom_wkb),
        )

    conn.commit()
    print(f"Polygon imported into table {table_name}")


def get_filtered_events(conn: sqlite3.Connection) -> pd.DataFrame:
    """Retrieve filtered events from the database."""

    query = """
    SELECT *
    FROM event_coordinates_filtered
    ORDER BY time ASC;
    """

    df = pd.read_sql_query(query, conn)
    return df


def main(db_path: str, shapefile_path: str, shape_table: str) -> None:
    """Execute the workflow."""

    # Establish the database connection
    conn = get_spatialite_connection(db_path)
    if conn is None:
        return

    # Import the shapefile into the database
    import_shapefile_to_db(conn, shapefile_path, shape_table)

    # Create the filtered view
    create_filtered_view(conn, shape_table)

    # Execute the spatial query (if still necessary)
    results = get_filtered_events(conn)

    # Display the results
    print(f"Number of earthquakes found in the polygon: {len(results)}")
    print(results.head())

    # Optional: Export the results to a CSV
    results.to_csv("earthquakes_in_polygon.csv", index=False)

    # Close the connection
    conn.close()


if __name__ == "__main__":
    parse = argparse.ArgumentParser(description="Filter events by polygon")
    parse.add_argument(
        "-d", "--db_path", type=str, help="Path to the database", required=True
    )
    parse.add_argument(
        "-s", "--shapefile_path", type=str, help="Path to the shapefile", required=True
    )
    args = parse.parse_args()

    if not args.db_path:
        print("Database path is required.")
        exit(1)

    if not args.shapefile_path:
        print("Shapefile path is required.")
        exit(1)

    # Table name to store the polygon
    shape_table = "polygon_zone"

    # Call the main function
    main(args.db_path, args.shapefile_path, shape_table)
