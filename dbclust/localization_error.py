#!/usr/bin/env python
import math
import re
from typing import Tuple

import numpy as np
from obspy.core.event import Origin

def get_erh_erz(origin: Origin) -> Tuple[float, float, str]:
    """
    Calculate the values of erh (horizontal uncertainty) and erz (vertical uncertainty).

    Parameters:
        origin (Origin): The origin.

    Returns:
        Tuple[float, float, str]: A tuple containing the values of erh and erz respectively in km
            and the method used to calculate them.
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
            erh = np.nan
            method = "unknown"

    try:
        erz = origin.depth_errors.uncertainty / 1000.0
        method = "origin_errors"
    except:
        try:
            erz = origin.origin_uncertainty.depth_uncertainty / 1000.0
            method = "origin_uncertainty"
        except:
            erz = np.nan
            method = "unknown"

    return erh, erz, method
