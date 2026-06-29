#!/usr/bin/env python3
from typing import List, Tuple, Union
import sqlite3
import logging
import numpy as np
from obspy.core.event import Origin

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


def compute_cpq(azimuths: List[float]) -> Union[float, None]:
    """
    Calculate the Cyclic Polygon Quotient (CPQ) for a list of event-station azimuths.

    The CPQ measures the azimuthal coverage quality of a seismic network around an event.
    It computes the area of a polygon formed by projecting station azimuths onto a unit
    circle, normalized by π. A CPQ of 1.0 indicates perfect azimuthal coverage.

    Reference:
        Gallacher et al. (2025) Revising the Seismic Ground Truth Reference Event
        Identification Criteria. Seismica.
        Code: https://github.com/Ryan-isc/cpq

    Args:
        azimuths: List of event-station azimuths in degrees (0-360).

    Returns:
        CPQ value (0 to 1), or None if fewer than 3 azimuths are provided.

    Example:
        >>> compute_cpq([2, 100, 150, 160, 170, 200, 250, 300, 359])
        0.8029686796937914
    """
    if len(azimuths) < 3:
        return None  # Minimum 3 azimuths required to form a polygon

    # Sort azimuths
    sorted_azimuths = np.sort(np.array(azimuths, dtype=float))

    # Validate range
    if np.max(sorted_azimuths) > 360 or np.min(sorted_azimuths) < 0:
        logger.warning("Azimuths must be between 0 and 360 degrees")
        return None

    # Convert azimuths to Cartesian coordinates on unit circle
    x = np.sin(np.radians(sorted_azimuths))
    y = np.cos(np.radians(sorted_azimuths))

    # Calculate polygon area using Surveyor's Area Formula (Shoelace formula)
    # Reference: Bart Braden, The College Mathematics Journal, 1986
    S1 = np.sum(x * np.roll(y, -1))
    S2 = np.sum(y * np.roll(x, -1))

    cpq = (0.5 * np.abs(S1 - S2)) / np.pi

    return cpq


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


def compute_gallacher_gt5_score_obspy(origin: Origin) -> Tuple[bool, dict]:
    """
    Determine if an event meets the revised GT5 criteria from Gallacher et al. (2025).

    Reference:
        Gallacher et al. (2025) Revising the Seismic Ground Truth Reference Event
        Identification Criteria. Seismica.

    Criteria:
        1. Five or more stations within 150 km of the event
        2. CPQ >= 0.4
        3. Secondary Azimuthal Gap <= 210°
        4. One or more stations within 10 km OR five or more stations with both P & S
        5. Recorded at distances >= 2° (teleseismic constraint)
        6. Semi-major axis of error ellipse <= 5 km
        7. Depth is resolved (not fixed to default)
        8. Resolved depth <= 35 km

    Args:
        origin: ObsPy Origin object with arrivals and quality information.

    Returns:
        Tuple of (bool, dict) where bool indicates if GT5 criteria are met,
        and dict contains detailed metrics.
    """
    arrivals = [a for a in origin.arrivals if a.time_weight and a.time_weight > 0]
    azimuths = []
    distances_km = []
    distances_deg = []
    stations_with_p = set()
    stations_with_s = set()

    for arrival in arrivals:
        pick = arrival.pick_id.get_referred_object()
        if pick and pick.waveform_id and arrival.distance is not None:
            distance_deg = arrival.distance
            distance_km = distance_deg * 111.11
            azimuths.append(arrival.azimuth)
            distances_km.append(distance_km)
            distances_deg.append(distance_deg)

            # Track stations with P and S phases
            station_id = f"{pick.waveform_id.network_code}.{pick.waveform_id.station_code}"
            if arrival.phase:
                phase_upper = arrival.phase.upper()
                if phase_upper.startswith("P"):
                    stations_with_p.add(station_id)
                elif phase_upper.startswith("S"):
                    stations_with_s.add(station_id)

    if len(azimuths) < 3:
        return False, {"error": "Not enough stations for evaluation"}

    # Compute metrics
    cpq = compute_cpq(azimuths)
    secondary_gap = compute_secondary_azimuthal_gap(azimuths)

    num_stations_150km = sum(1 for d in distances_km if d <= 150)
    num_stations_10km = sum(1 for d in distances_km if d <= 10)
    max_distance_deg = max(distances_deg) if distances_deg else 0

    # Stations with both P and S phases
    stations_with_both_ps = stations_with_p & stations_with_s
    num_stations_both_ps = len(stations_with_both_ps)

    # Error ellipse semi-major axis (from origin uncertainty)
    semi_major_axis_km = None
    if origin.origin_uncertainty:
        # Try to get semi-major axis from origin uncertainty
        if origin.origin_uncertainty.max_horizontal_uncertainty is not None:
            semi_major_axis_km = origin.origin_uncertainty.max_horizontal_uncertainty / 1000.0
        elif origin.origin_uncertainty.horizontal_uncertainty is not None:
            semi_major_axis_km = origin.origin_uncertainty.horizontal_uncertainty / 1000.0

    # Depth resolution check
    depth_km = origin.depth / 1000.0 if origin.depth is not None else None
    depth_is_fixed = False
    if origin.depth_type:
        depth_type_str = str(origin.depth_type).lower()
        depth_is_fixed = "operator" in depth_type_str or "fixed" in depth_type_str
    depth_is_resolved = depth_km is not None and not depth_is_fixed

    # Apply Gallacher et al. (2025) criteria
    criterion_1 = num_stations_150km >= 5
    criterion_2 = cpq is not None and cpq >= 0.4
    criterion_3 = secondary_gap <= 210
    criterion_4 = (num_stations_10km >= 1) or (num_stations_both_ps >= 5)
    criterion_5 = max_distance_deg >= 2.0
    criterion_6 = semi_major_axis_km is not None and semi_major_axis_km <= 5.0
    criterion_7 = depth_is_resolved
    criterion_8 = depth_km is not None and depth_km <= 35.0

    gt5_criteria = all([
        criterion_1,
        criterion_2,
        criterion_3,
        criterion_4,
        criterion_5,
        criterion_6,
        criterion_7,
        criterion_8,
    ])

    details = {
        "num_stations_150km": num_stations_150km,
        "num_stations_10km": num_stations_10km,
        "num_stations_both_ps": num_stations_both_ps,
        "cpq": cpq,
        "secondary_gap": secondary_gap,
        "max_distance_deg": max_distance_deg,
        "semi_major_axis_km": semi_major_axis_km,
        "depth_km": depth_km,
        "depth_is_resolved": depth_is_resolved,
        "criteria": {
            "1_stations_150km": criterion_1,
            "2_cpq": criterion_2,
            "3_secondary_gap": criterion_3,
            "4_local_coverage": criterion_4,
            "5_teleseismic": criterion_5,
            "6_error_ellipse": criterion_6,
            "7_depth_resolved": criterion_7,
            "8_shallow_depth": criterion_8,
        },
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
