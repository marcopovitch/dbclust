#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import Tuple

from icecream import ic
from localization_error import get_erh_erz
from obspy.core.event import Event
from obspy.core.event import Origin


def classify_event(
    event: Event, origin_id: str = None, debug: bool = False
) -> Tuple[str, str, str, str]:
    """
    Classify the quality of an event's origin.

    Parameters:
        event (Event): The event to classify.
        origin_id (str, optional): The ID of the origin to classify. If None, the preferred origin of the event is used.

    Returns:
        Tuple[str, str, str, str]: A tuple containing :
            - the overall quality,
            - epicentral quality,
            - depth quality,
            - and a textual representation of the classification.

    Raises:
        ValueError: If the specified origin_id is not found in the event.
    """

    if origin_id is None:
        origin = event.preferred_origin()
    else:
        for o in event.origins:
            if o.resource_id.id == origin_id:
                origin = o
                break
        else:
            raise ValueError(
                f"Origin {origin_id} not found in event {event.resource_id.id}"
            )

    erh, erz, error_method = get_erh_erz(origin)

    # TBD: compute minimal distance between stations and the event with the formula real coordinates
    # rather than the 111.1 km/deg approximation

    quality, qs, qd = classify(
        origin.quality.standard_error,  # in seconds
        erh,  # in km
        erz,  # in km
        origin.quality.used_station_count,
        origin.quality.azimuthal_gap,  # in degrees
        origin.quality.minimum_distance * 111.1,  # convert distance from degres to km
        origin.depth / 1000.0,  # depth in km
    )

    if debug:
        ic(
            origin.quality.standard_error,
            erh,
            erz,
            origin.quality.used_station_count,
            origin.quality.azimuthal_gap,
            origin.quality.minimum_distance * 111.1,
            origin.depth / 1000.0,
            quality,
            qs,
            qd,
            error_method,
        )

    return quality, qs, qd, get_classification_text(quality)


def classify(
    rms: float, erh: float, erz: float, no: float, gap: float, dmin: float, depth: float
) -> Tuple[str, str, str]:
    """
    Hypo71 Quality Classification
    from: https://www.usgs.gov/publications/hypo71-earthquake-location-program
    -----------------------------

    Input Attributes:
        - RMS (root mean square of residuals)
        - ERH (horizontal error)
        - ERZ (vertical error)
        - NO (number of stations used)
        - GAP (maximum azimuthal gap)
        - DMIN (minimum distance to the nearest station)

    Classify QS (Epicenter Quality):
        Using the following thresholds:
        - A:  RMS < 0.15 s, ERH <= 1.0 km, ERZ <= 2.0 km
        - B:  RMS < 0.30 s, ERH <= 2.5 km, ERZ <= 5.0 km
        - C:  RMS < 0.50 s, ERH <= 5.0 km
        - D: Other values

    Classify QD (Focal Depth Quality):
        Using the station distribution criteria:
        - A:  NO >= 6,  GAP < 90 ,  DMIN <= Depth or 5 km
        - B:  NO >= 6,  GAP < 135,  DMIN <= 2 x Depth or  10 km
        - C:  NO >= 6,  GAP < 180,  DMIN <= 50 km
        - D: Other values

    Calculate Overall Quality (Q):
        Q is the average of QS and QD. If the two are more than one grade apart
        (e.g., QS = A and QD = C), assign the lower quality (B in this case).
        If the difference is within one level, take the lower one (A and B yield B).

        Class A: Excellent (epicenter) / Good (depth)
        Class B: Good (epicenter) / Fair (depth)
        Class C: Fair (epicenter) / Poor (depth)
        Class D: Poor (epicenter and depth)

    Returns the overall quality Q, and the individual qualities QS and QD.
    """

    # Classify QS (Epicenter Quality)
    if rms < 0.15 and erh <= 1.0 and erz <= 2.0:
        qs = "A"
    elif rms < 0.30 and erh <= 2.5 and erz <= 5.0:
        qs = "B"
    elif rms < 0.50 and erh <= 5.0:
        qs = "C"
    else:
        qs = "D"

    # Classify QD (Focal Depth Quality)
    if no >= 6 and gap < 90 and dmin <= max(depth, 5):
        qd = "A"
    elif no >= 6 and gap < 135 and dmin <= max(2 * depth, 10):
        qd = "B"
    elif no >= 6 and gap < 180 and dmin <= 50:
        qd = "C"
    else:
        qd = "D"

    # Calculate Overall Quality Q
    levels = {"A": 1, "B": 2, "C": 3, "D": 4}

    if abs(levels[qs] - levels[qd]) == 1:
        # assign the lower quality if difference is within one level
        val = max(levels[qs], levels[qd])
    else:
        # assign the mean of qs and qd if difference > 1
        val = (levels[qs] + levels[qd]) / 2
        # get the nearest integer
        val = round(val)

    for k, v in levels.items():
        if v == val:
            q = k

    return q, qs, qd


def get_classification_text(classification: str) -> str:
    """
    Class A: Excellent (epicenter) / Good (depth)
    Class B: Good (epicenter) / Fair (depth)
    Class C: Fair (epicenter) / Poor (depth)
    Class D: Poor (epicenter and depth)
    """
    classification_text = {
        "A": "Excellent (epicenter) / Good (depth)",
        "B": "Good (epicenter) / Fair (depth)",
        "C": "Fair (epicenter) / Poor (depth)",
        "D": "Poor (epicenter and depth)",
    }

    return classification_text[classification]
