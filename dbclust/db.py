#!/usr/bin/env python
import logging
import re
from typing import Dict
from typing import List

import duckdb
import pandas as pd

logger = logging.getLogger("dbclust.db")


def validate_sql_identifier(name: str) -> str:
    """Validate and sanitize SQL identifier (table/view/column name).

    Args:
        name: The SQL identifier to validate.

    Returns:
        The validated identifier (unchanged if valid).

    Raises:
        ValueError: If the name contains invalid characters.
    """
    if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", name):
        raise ValueError(f"Invalid SQL identifier: {name}")
    return name


def duckdb_init(filenames: List[str], file_type: str):
    """
    Initialize a DuckDB connection with a view depending on the file type (parquet or CSV).

    Parameters:
        filenames (List[str]): List of file paths to load.
        file_type (str): The file type ("parquet" or "csv").

    Returns:
        duckdb.DuckDBPyConnection: DuckDB connection with the view created.
    """
    if file_type == "parquet":
        return duckdb_init_parquet(filenames)
    elif file_type == "csv":
        return duckdb_init_csv(filenames)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")


def duckdb_init_parquet(parquet_filenames: List[str], threads: int = 4):
    """
    Initialize a DuckDB connection and create a view from parquet files.

    Parameters:
        parquet_filenames (List[str]): List of parquet file paths to load.
        threads (int): Number of threads for DuckDB operations (default is 4).

    Returns:
        duckdb.DuckDBPyConnection: DuckDB connection with the view created.
    """
    if not parquet_filenames:
        raise ValueError("The list of parquet filenames cannot be empty.")

    config = {"threads": threads}

    files_str = ""
    for f in parquet_filenames:
        files_str += f"'{f}', "

    files_str = "[" + files_str[:-2] + "]"
    rqt = f"""
        CREATE VIEW PICKS AS SELECT *
        FROM read_parquet({files_str});
    """

    # Initialize the DuckDB connection
    duckdb_con = duckdb.connect(
        database=":memory:",
        config=config,
    )
    try:
        duckdb_con.execute(rqt)
    except Exception as e:
        duckdb_con.close()  # Ensure the connection is closed on failure
        raise RuntimeError(f"Failed to execute the query: {e}") from e

    return duckdb_con


def duckdb_init_csv(csv_filenames: list[str], threads: int = 1):
    """
    Initializes a DuckDB in-memory database and creates a view combining the given CSV files.

    Parameters:
        csv_filenames (list[str]): List of CSV file paths to be loaded into the view.
        threads (int): Number of threads to use for DuckDB operations (default is 1).

    Returns:
        duckdb.DuckDBPyConnection: The DuckDB connection with the view created.
    """
    if not csv_filenames:
        raise ValueError("The list of CSV filenames cannot be empty.")

    config = {"threads": threads}

    # Generate a SQL query to combine CSV files using UNION ALL
    file_queries = [f"SELECT * FROM read_csv_auto('{f}')" for f in csv_filenames]
    combined_query = " UNION ALL ".join(file_queries)

    # Final SQL query
    rqt = f"CREATE VIEW PICKS AS {combined_query};"

    # Initialize the DuckDB connection
    duckdb_con = duckdb.connect(database=":memory:", config=config)
    try:
        duckdb_con.execute(rqt)
    except Exception as e:
        duckdb_con.close()  # Ensure the connection is closed on failure
        raise RuntimeError(f"Failed to execute the query: {e}") from e

    return duckdb_con


def filter_stations_by_bbox(
    station_coords: pd.DataFrame,
    bbox: Dict[str, float],
) -> pd.DataFrame:
    """Filter stations DataFrame by bounding box.

    Args:
        station_coords: DataFrame with columns [network, station, latitude, longitude].
        bbox: Bounding box with keys: min_lat, max_lat, min_lon, max_lon.

    Returns:
        Filtered DataFrame with only stations within the bounding box.
    """
    if station_coords.empty:
        logger.warning("No station coordinates provided, geographic filtering disabled")
        return station_coords

    mask = (
        (station_coords["latitude"] >= bbox["min_lat"])
        & (station_coords["latitude"] <= bbox["max_lat"])
        & (station_coords["longitude"] >= bbox["min_lon"])
        & (station_coords["longitude"] <= bbox["max_lon"])
    )

    filtered = station_coords[mask].copy()

    logger.info(
        f"Geographic filter (DataFrame): bbox "
        f"[{bbox['min_lat']:.2f}, {bbox['max_lat']:.2f}] x "
        f"[{bbox['min_lon']:.2f}, {bbox['max_lon']:.2f}] -> "
        f"{len(filtered)}/{len(station_coords)} stations"
    )

    return filtered


def filter_inventory_by_bbox(inventory, bbox: Dict[str, float]):
    """Filter ObsPy Inventory by bounding box.

    Args:
        inventory: ObsPy Inventory object.
        bbox: Bounding box with keys: min_lat, max_lat, min_lon, max_lon.

    Returns:
        Filtered Inventory with only stations within the bounding box.
    """
    if inventory is None:
        return None

    from obspy import Inventory

    filtered_networks = []
    total_stations = 0
    kept_stations = 0

    for network in inventory:
        filtered_stations = []
        for station in network:
            total_stations += 1
            lat = station.latitude
            lon = station.longitude
            if (
                bbox["min_lat"] <= lat <= bbox["max_lat"]
                and bbox["min_lon"] <= lon <= bbox["max_lon"]
            ):
                filtered_stations.append(station)
                kept_stations += 1

        if filtered_stations:
            # Create a copy of the network with only filtered stations
            new_network = network.copy()
            new_network.stations = filtered_stations
            filtered_networks.append(new_network)

    filtered_inventory = Inventory(networks=filtered_networks, source=inventory.source)

    logger.info(
        f"Geographic filter (Inventory): bbox "
        f"[{bbox['min_lat']:.2f}, {bbox['max_lat']:.2f}] x "
        f"[{bbox['min_lon']:.2f}, {bbox['max_lon']:.2f}] -> "
        f"{kept_stations}/{total_stations} stations"
    )

    return filtered_inventory
