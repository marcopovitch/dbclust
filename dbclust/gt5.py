#!/usr/bin/env python3
from typing import List
from typing import Union

import numpy as np
from icecream import ic
from obspy.core.event import Origin

from dbclust.gap import compute_azimuthal_gap
from dbclust.gap import compute_secondary_azimuthal_gap


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


def compute_gt5_score(origin: Origin) -> Union[bool, dict]:
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
