#!/usr/bin/env python3
from typing import List, Union
import sqlite3
import logging
import numpy as np

from dbclust.gap import compute_azimuthal_gap
from dbclust.gap import compute_secondary_azimuthal_gap

logger = logging.getLogger(__name__)



def compute_delta_U(azimuths: List[float]) -> Union[float, None]:
    """Calculate the Network Quality Metric ΔU from Bondár & McLaughlin (2009)."""
    if len(azimuths) < 2:
        return None  # Impossible to calculate ΔU with only one station

    azimuths = np.sort(np.array(azimuths))  # Sort azimuths in ascending order
    N = len(azimuths)

    # Calculate uniform angles
    uniform_angles = np.linspace(0, 360, N, endpoint=False)

    # Correction factor b
    b = np.mean(azimuths) - np.mean(uniform_angles)

    # Calculate ΔU
    delta_U = (4 / (360 * N)) * np.sum(np.abs(azimuths - (uniform_angles + b)))

    return delta_U


def compute_gt5_score_obspy(origin: Origin) -> Union[bool, dict]:
    """Determine if an event meets the GT5 criteria."""

    arrivals = [a for a in origin.arrivals if a.time_weight > 0]
    azimuths = []
    distances = []

    for arrival in arrivals:
        pick = arrival.pick_id.get_referred_object()
        if pick and pick.waveform_id:
            distance_km = arrival.distance * 111.11  # Approximate conversion
            azimuths.append(arrival.azimuth)
            distances.append(distance_km)

    if len(azimuths) < 2:
        return False  # Not enough stations for an evaluation

    # GT5 criteria
    primary_gap = compute_azimuthal_gap(azimuths)
    secondary_gap = compute_secondary_azimuthal_gap(azimuths)
    delta_U = compute_delta_U(azimuths)

    num_stations_150km = sum(1 for d in distances if d <= 150)
    num_stations_10km = sum(1 for d in distances if d <= 10)

    gt5_criteria = (
        num_stations_150km >= 10
        and num_stations_10km >= 1
        and primary_gap <= 110
        and secondary_gap <= 160
        and delta_U <= 0.35
    )

    details = {
        "num_stations_10km": num_stations_10km,
        "num_stations_150km": num_stations_150km,
        "primary_gap": primary_gap,
        "secondary_gap": secondary_gap,
        "delta_U": delta_U,
    }

    return gt5_criteria, details


def compute_gt5_score(conn: sqlite3.Connection) -> None:
    """
    Compute and update GT5 metrics for all events in the database.

    Args:
        conn: SQLite database connection
    """
    logger.info("Computing GT5 metrics for all events...")
    cursor = conn.cursor()

    # Add GT5 columns if they don't exist
    cursor.execute("PRAGMA table_info(origins)")
    columns = [col[1] for col in cursor.fetchall()]

    if "gt5_status" not in columns:
        cursor.execute("ALTER TABLE origins ADD COLUMN gt5_status BOOLEAN")
    if "delta_U" not in columns:
        cursor.execute("ALTER TABLE origins ADD COLUMN delta_U FLOAT")
    if "num_stations_10km" not in columns:
        cursor.execute("ALTER TABLE origins ADD COLUMN num_stations_10km INTEGER")
    if "num_stations_150km" not in columns:
        cursor.execute("ALTER TABLE origins ADD COLUMN num_stations_150km INTEGER")

    # Get all preferred origins
    cursor.execute("SELECT id, event_id FROM origins WHERE preferred = 1")
    origins = cursor.fetchall()

    for origin_id, event_id in origins:
        # Get arrivals for this origin
        cursor.execute(
            """
            SELECT a.azimuth, a.distance, a.time_weight
            FROM arrivals a
            JOIN picks p ON a.pick_id = p.id
            WHERE a.origin_id = ? AND a.time_weight > 0
        """,
            (origin_id,),
        )

        arrivals = cursor.fetchall()

        if len(arrivals) < 2:
            # Not enough stations for evaluation
            cursor.execute(
                """
                UPDATE origins 
                SET gt5_status = 0,
                    delta_U = NULL,
                    num_stations_10km = ?,
                    num_stations_150km = ?
                WHERE id = ?
            """,
                (0, len(arrivals), origin_id),
            )
            continue

        # Process arrivals
        azimuths = []
        distances_km = []

        for azimuth, distance_deg, _ in arrivals:
            if azimuth is not None and distance_deg is not None:
                distance_km = distance_deg * 111.11  # Approximate conversion
                azimuths.append(azimuth)
                distances_km.append(distance_km)

        if len(azimuths) < 2:
            continue

        # Compute metrics
        primary_gap = compute_azimuthal_gap(azimuths)
        secondary_gap = compute_secondary_azimuthal_gap(azimuths)
        delta_U = compute_delta_U(azimuths)

        num_stations_150km = sum(1 for d in distances_km if d <= 150)
        num_stations_10km = sum(1 for d in distances_km if d <= 10)

        # GT5 criteria
        gt5_status = (
            num_stations_150km >= 10
            and num_stations_10km >= 1
            and primary_gap <= 110
            and secondary_gap <= 160
            and delta_U is not None
            and delta_U <= 0.35
        )

        # Update the origin with GT5 metrics
        cursor.execute(
            """
            UPDATE origins 
            SET gt5_status = ?,
                delta_U = ?,
                num_stations_10km = ?,
                num_stations_150km = ?,
                azimuthal_gap = COALESCE(azimuthal_gap, ?),
                secondary_azimuthal_gap = COALESCE(secondary_azimuthal_gap, ?)
            WHERE id = ?
        """,
            (
                1 if gt5_status else 0,
                delta_U,
                num_stations_10km,
                num_stations_150km,
                primary_gap,
                secondary_gap,
                origin_id,
            ),
        )

    conn.commit()
    logger.info("GT5 metrics computation completed")
