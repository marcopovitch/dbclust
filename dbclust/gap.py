#!/usr/bin/env python
from typing import List
from typing import Union

from obspy.core.event import Event


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
        return None

    # sort arrival by distance
    sorted_arrivals = sorted(origin.arrivals, key=lambda x: x.distance)

    # compute distance (in degrees) between consecutive arrivals
    dist_list = [
        sorted_arrivals[i].distance - sorted_arrivals[i - 1].distance
        for i in range(1, len(sorted_arrivals))
    ]
    # get arrival with corresponding dist_list greater than dist_max
    arrivals_to_unset = []
    for i in range(len(dist_list)):
        if dist_list[i] >= dist_max_km / 111.1:
            # find corresponding pick to arrival
            pick = next(
                (
                    p
                    for p in event.picks
                    if p.resource_id == sorted_arrivals[i + 1].pick_id
                ),
                None,
            )
            if pick.evaluation_mode in apply_to_evaluation_mode:
                arrivals_to_unset.append(sorted_arrivals[i + 1])

    return arrivals_to_unset


def compute_gap(azimuth_list: List[float]) -> Union[float, None]:
    """Compute gap from azimuth list in degree

    Args:
        azimuth_list (List[float]): azimuth list in degree

    Returns:
        float: the max gap in degree
    """

    if len(azimuth_list) <= 2:
        return None

    az_list_sorted = sorted(azimuth_list)
    az_list_sorted.append(az_list_sorted[0] + 360)
    gap_max = 0
    for i in range(1, len(az_list_sorted)):
        gap = abs(az_list_sorted[i] - az_list_sorted[i - 1])
        if gap > gap_max:
            gap_max = gap
    return gap_max


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
