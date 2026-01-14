#!/usr/bin/env python
import logging
from typing import List
from typing import Union

import numpy as np
from obspy.core.event import Event

logger = logging.getLogger(__name__)


def get_arrival_with_distance_gap_greater_than(
    event: Event, dist_max_km: float, apply_to_evaluation_mode: list = ["automatic"]
) -> Union[float, None]:
    """Get arrival with distance greater than dist_max

    Args:
        event (Event): event to work on
        dist_max_km (float): max distance in km allowed
        apply_to_evaluation_mode (list, optional): list of evaluation mode to apply the selection.

    Returns:
        Union[float, None]: arrivals with distance greater than dist_max_km
    """
    origin = event.preferred_origin()

    if not origin:
        return []

    # sort arrival by distance
    sorted_arrivals = sorted(origin.arrivals, key=lambda x: x.distance)

    # compute distance (in degrees) between consecutive arrivals
    dist_list = [
        sorted_arrivals[i].distance - sorted_arrivals[i - 1].distance
        for i in range(1, len(sorted_arrivals))
    ]

    # find the first arrival with distance greater than dist_max_km
    # and return the corresponding arrival and all the following arrivals
    # if their evaluation mode is in apply_to_evaluation_mode
    # return None if no arrival with distance greater than dist_max_km.
    for i in range(len(dist_list)):
        if dist_list[i] >= dist_max_km / 111.1:
            # get the index of the first arrival with distance >= dist_max_km
            i_max = i + 1
            break
    else:
        return []  # no arrival with distance greater than dist_max_km

    arrivals_to_unset = []
    for i in range(i_max, len(origin.arrivals)):
        # find corresponding pick to arrival
        pick = next(
            (p for p in event.picks if p.resource_id == sorted_arrivals[i].pick_id),
            None,
        )
        if pick.evaluation_mode in apply_to_evaluation_mode:
            arrivals_to_unset.append(sorted_arrivals[i])

    return arrivals_to_unset


def compute_gap(azimuth_list: List[float]) -> Union[float, None]:
    """Compute gap from azimuth list in degree

    Args:
        azimuth_list (List[float]): azimuth list in degree

    Returns:
        float: the max gap in degree
    """

    if len(azimuth_list) <= 2:
        logger.warning("Not enough azimuths to compute gap")
        return None

    az_list_sorted = sorted(azimuth_list)
    az_list_sorted.append(az_list_sorted[0] + 360)
    gap_max = 0
    for i in range(1, len(az_list_sorted)):
        gap = abs(az_list_sorted[i] - az_list_sorted[i - 1])
        if gap > gap_max:
            gap_max = gap
    return gap_max


def compute_azimuthal_gap(azimuths: List[float]) -> Union[float, None]:
    """Calculate the largest angular gap (Azimuthal Gap)."""

    # exclude None values
    azimuths = [az for az in azimuths if az is not None]
    logger
    if len(azimuths) < 2:
        return None

    azimuths = np.sort(np.array(azimuths))
    azimuthal_gaps = np.diff(np.append(azimuths, azimuths[0] + 360))
    return np.max(azimuthal_gaps)


def compute_secondary_azimuthal_gap(azimuths: List[float]) -> Union[float, None]:
    """Calculate the Secondary Azimuthal Gap by removing one station at a time."""
    if len(azimuths) < 3:
        return None

    max_secondary_gap = 0
    for i in range(len(azimuths)):
        reduced_azimuths = np.delete(azimuths, i)  # Remove one station
        new_gap = compute_azimuthal_gap(reduced_azimuths)
        if not new_gap:
            continue
        max_secondary_gap = max(max_secondary_gap, new_gap)

    return max_secondary_gap


if __name__ == "__main__":
    az = [
        183.7569938580601,
        154.89095329554073,
        211.70697349452007,
        126.22567184667095,
        110.94526613548878,
        156.4765831306809,
        144.26245491761682,
        144.26245491761682,
        166.5878232569371,
        245.1692270809625,
        245.1692270809625,
        218.48579094382796,
        218.48579094382796,
        228.01854823758313,
        228.01854823758313,
        269.5097847454167,
        269.5097847454167,
        245.1692270809625,
        245.1692270809625,
        103.47634128284595,
        103.47634128284595,
        165.67508097759827,
        165.67508097759827,
        110.99225593297678,
        170.22264538550203,
        170.22264538550203,
        72.54989835048973,
        72.54989835048973,
        355.2361683158599,
        132.612854984889,
        194.50423691373513,
        194.50423691373513,
        322.6125368812272,
        269.5252018020378,
        269.5252018020378,
        247.60296736789797,
        247.60296736789797,
        226.93933555560378,
        226.93933555560378,
        120.43096867855603,
        120.43096867855603,
        43.48329435414466,
        247.55913555832421,
    ]

    gap = compute_gap(az)
    print(f"Gap = {gap} degrees")

    gap1 = compute_azimuthal_gap(az)
    gap2 = compute_secondary_azimuthal_gap(az)
    print(f"Azimuthal Gap = {gap1} degrees")
    print(f"Secondary Azimuthal Gap = {gap2} degrees")
