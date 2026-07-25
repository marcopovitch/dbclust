#!/usr/bin/env python
# -*- coding: utf-8 -*-
import json
import logging
import re
from typing import Optional, Tuple

import numpy as np
from geopy.distance import geodesic
from obspy.core.event import Event
from scipy.special import erf
from scipy.special import expit

from dbclust.localization_error import get_erh_erz

logger = logging.getLogger("dbclust")


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculates the distance in km between two points using the Haversine formula.

    Args:
        lat1, lon1: Latitude and longitude of the first point.
        lat2, lon2: Latitude and longitude of the second point.

    Returns:
        Distance in kilometers.
    """
    try:
        return geodesic((lat1, lon1), (lat2, lon2)).kilometers
    except Exception as e:
        logger.error(f"Error calculating distance: {str(e)}")
        return float("inf")


def chauvenet_filter(data) -> np.ndarray:
    """
    Filters out data points that do not meet Chauvenet's criterion.

    Args:
        data (array-like): Array of RMS values.

    Returns:
        filtered_data (numpy.ndarray): Array of data points that meet Chauvenet's criterion.
    """
    # Convert to a numpy array for easier computation
    data = np.array(data)

    # Calculate mean and standard deviation
    mean = np.mean(data)
    std_dev = np.std(data)
    N = len(data)

    # Handle edge case: if std_dev is 0, all points are identical
    if std_dev == 0:
        return data

    # Calculate the threshold probability
    threshold_prob = 1.0 / (2 * N)

    # Calculate Z-scores for each data point
    z_scores = np.abs(data - mean) / std_dev

    # Calculate the two-tailed probability for each Z-score
    probs = 1 - (erf(z_scores / np.sqrt(2)))

    # Apply Chauvenet's criterion
    filtered_data = data[probs >= threshold_prob]

    return filtered_data


def normalize_rms(
    rms, num_stations, avg_distance, azimuthal_gap, alpha=0.01, beta=0.01
):
    """
    Normalize the RMS to evaluate the localization quality.

    Parameters:
    - rms : float, the RMS of the localization.
    - num_stations : int, the number of seismic stations used.
    - avg_distance : float, the average or median distance of the stations to the epicenter (in km).
    - azimuthal_gap : float, the azimuthal gap (in degrees).
    - alpha : float, weighting factor for the distance.
    - beta : float, weighting factor for the azimuthal gap.

    Returns:
    - float, the normalized RMS.
    """
    return rms / (
        (num_stations ** (1 / 10))
        * (1 + alpha * avg_distance)
        * (1 + beta * azimuthal_gap)
    )


def classify_event(
    event: Event, origin_id: str = None, debug: bool = False
) -> Tuple[str, str, str, str]:
    """
    Classify (hypo71) the quality of an event's origin.

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
        ValueError: If the specified origin_id is not found in the event or if no origin is available.
    """

    if origin_id is None:
        origin = event.preferred_origin()
        if origin is None:
            raise ValueError(
                f"No preferred origin found for event {event.resource_id.id}"
            )
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

    # TBD: compute minimal distance between stations and the event with the real coordinates
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
        logger.debug(
            f"classify_event: rms={origin.quality.standard_error}, erh={erh}, erz={erz}, "
            f"stations={origin.quality.used_station_count}, gap={origin.quality.azimuthal_gap}, "
            f"dmin={origin.quality.minimum_distance * 111.1}, depth={origin.depth / 1000.0}, "
            f"quality={quality}, qs={qs}, qd={qd}, error_method={error_method}"
        )

    return quality, qs, qd, get_classification_text(quality)


def _extract_from_comments(
    comments: list, key: str, pattern: Optional[str] = None
) -> Optional[float]:
    """
    Extract a numeric value from origin comments.

    Searches comments for JSON format (e.g., {"scatter_volume": 123.4}) or
    key-value patterns (e.g., "scatvol=123.4" or "scatvol: 123.4").

    Args:
        comments: List of Comment objects from origin.comments.
        key: The key to search for in JSON comments.
        pattern: Optional regex pattern to match in non-JSON comments.

    Returns:
        The extracted float value, or None if not found.
    """
    if not comments:
        return None

    for comment in comments:
        if not comment.text:
            continue

        # Try JSON parsing first
        try:
            data = json.loads(comment.text)
            if isinstance(data, dict) and key in data:
                return float(data[key])
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

        # Try regex pattern matching for non-JSON formats
        text_lower = comment.text.lower()
        key_lower = key.lower()

        if key_lower in text_lower or (pattern and pattern.lower() in text_lower):
            # Try to extract numeric value after key
            match = re.search(
                rf"{key_lower}\s*[=:]\s*([-+]?\d*\.?\d+)", text_lower
            )
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    pass

            # Fallback: extract any number from the comment
            match = re.search(r"[-+]?\d*\.?\d+", comment.text)
            if match:
                try:
                    return float(match.group())
                except ValueError:
                    pass

    return None


def _extract_expectation(comments: list) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Extract expectation hypocenter from origin comments.

    Searches for JSON format: {"expectation": {"latitude": x, "longitude": y, "depth": z}}

    Args:
        comments: List of Comment objects from origin.comments.

    Returns:
        Tuple of (latitude, longitude, depth) or (None, None, None) if not found.
    """
    if not comments:
        return None, None, None

    for comment in comments:
        if not comment.text:
            continue

        try:
            data = json.loads(comment.text)
            if isinstance(data, dict) and "expectation" in data:
                exp = data["expectation"]
                return (
                    float(exp.get("latitude")) if exp.get("latitude") is not None else None,
                    float(exp.get("longitude")) if exp.get("longitude") is not None else None,
                    float(exp.get("depth")) if exp.get("depth") is not None else None,
                )
        except (json.JSONDecodeError, ValueError, TypeError, KeyError):
            pass

    return None, None, None


def classify_event_michele_mod2(
    event: Event, origin_id: str = None, debug: bool = False
) -> Tuple[Optional[float], Optional[str]]:
    """
    Classify the quality of an event's origin using the Michele_mod2 method.

    This function extracts all required parameters from the event/origin,
    including scatter_volume and expectation hypocenter from QuakeML comments,
    and computes the Michele_mod2 quality classification.

    Parameters:
        event (Event): The event to classify.
        origin_id (str, optional): The ID of the origin to classify.
            If None, the preferred origin of the event is used.
        debug (bool): If True, log debug information.

    Returns:
        Tuple[Optional[float], Optional[str]]: A tuple containing:
            - qf: Quality factor (float), or None if classification not possible.
            - q: Quality category ("A", "B", "C", "D", "E"), or None if not possible.

    Notes:
        Parameters extracted from origin.comments:
        - scatter_volume (scatvol): Scatter volume from NonLinLoc
        - expectation: Expected hypocenter for computing dloch and dz

        If scatter_volume or expectation is not found in comments, the function
        returns (None, None).
    """
    # Get the origin
    if origin_id is None:
        origin = event.preferred_origin()
        if origin is None:
            logger.warning(
                f"No preferred origin found for event {event.resource_id.id}"
            )
            return None, None
    else:
        origin = None
        for o in event.origins:
            if o.resource_id.id == origin_id:
                origin = o
                break
        if origin is None:
            logger.warning(
                f"Origin {origin_id} not found in event {event.resource_id.id}"
            )
            return None, None

    # Extract basic quality parameters
    try:
        rms = origin.quality.standard_error
        erh, erz, error_method = get_erh_erz(origin)
        nbpha = origin.quality.used_phase_count
        gap = origin.quality.azimuthal_gap
        gap2 = origin.quality.secondary_azimuthal_gap
        dmin_deg = origin.quality.minimum_distance
        dmed_deg = origin.quality.median_distance
        depth_km = origin.depth / 1000.0
        lat = origin.latitude
        lon = origin.longitude
    except (AttributeError, TypeError) as e:
        logger.warning(f"Missing quality parameters for Michele_mod2: {e}")
        return None, None

    # Check required parameters
    if any(v is None for v in [rms, erh, erz, nbpha, gap, dmin_deg]):
        logger.warning("Missing required parameters for Michele_mod2 classification")
        return None, None

    # Use gap as gap2 fallback
    if gap2 is None:
        gap2 = gap

    # Use dmin as dmed fallback
    if dmed_deg is None:
        dmed_deg = dmin_deg

    # Extract scatter_volume from comments
    scatvol = _extract_from_comments(origin.comments, "scatter_volume", "scatvol")
    if scatvol is None:
        logger.debug("scatter_volume not found in origin comments")
        return None, None

    # Extract expectation hypocenter from comments
    expect_lat, expect_lon, expect_depth = _extract_expectation(origin.comments)
    if expect_lat is None or expect_lon is None or expect_depth is None:
        logger.debug("expectation not found in origin comments")
        return None, None

    # Compute dloch (horizontal distance between location and expectation)
    dloch = haversine_distance(lat, lon, expect_lat, expect_lon)

    # Compute dz (depth difference)
    dz = depth_km - expect_depth

    # Call classify_Michele_mod2
    try:
        qf, q = classify_Michele_mod2(
            rms=rms,
            erh=erh,
            erz=erz,
            nbpha=nbpha,
            dmin=dmin_deg,
            dmed=dmed_deg,
            gap=gap,
            gap2=gap2,
            scatvol=scatvol,
            dloch=dloch,
            dz=dz,
        )

        if debug:
            logger.debug(
                f"classify_event_michele_mod2: rms={rms}, erh={erh}, erz={erz}, "
                f"nbpha={nbpha}, dmin={dmin_deg}, dmed={dmed_deg}, gap={gap}, gap2={gap2}, "
                f"scatvol={scatvol}, dloch={dloch:.2f}, dz={dz:.2f}, "
                f"qf={qf:.3f}, q={q}, error_method={error_method}"
            )

        return qf, q

    except Exception as e:
        logger.warning(f"Michele_mod2 classification failed: {e}")
        return None, None


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
    if (rms < 0.15) and (erh <= 1.0) and (erz <= 2.0):
        qs = "A"
    elif (rms < 0.30) and (erh <= 2.5) and (erz <= 5.0):
        qs = "B"
    elif (rms < 0.50) and (erh <= 5.0):
        qs = "C"
    else:
        qs = "D"

    # Classify QD (Focal Depth Quality)
    if (no >= 6) and (gap < 90) and (dmin <= max(depth, 5)):
        qd = "A"
    elif (no >= 6) and (gap < 135) and (dmin <= max(2 * depth, 10)):
        qd = "B"
    elif (no >= 6) and (gap < 180) and (dmin <= 50):
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

    # Reverse lookup: find quality letter for computed value
    reverse_levels = {v: k for k, v in levels.items()}
    q = reverse_levels.get(val, "D")

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


def classify_Michele_mod(
    rms: float,
    erh: float,
    erz: float,
    nbpha: int,
    dmin: float,
    dmed: float,
    gap: float,
    gap2: float,
    scatvol: float,
) -> Tuple[float, str]:
    """
    Classify seismic event quality using a modified version of Michele et al. (2019).

    This function computes a quality factor (qf) for seismic events based on normalized
    input parameters. The quality factor determines the event's classification into one
    of four quality categories: "A", "B", "C", or "D".

    Parameters:
    -----------
    rms : float
        Root mean square of residuals (seconds).
    erh : float
        Horizontal error (kilometers).
    erz : float
        Vertical error (kilometers).
    nbpha : int
        Number of phases used in the solution.
    dmin : float
        Minimum distance to the nearest station (degrees).
    dmed : float
        Median distance of the stations used (degrees).
    gap : float
        Maximum azimuthal gap (degrees).
    gap2 : float
        Secondary azimuthal gap (degrees).
    scatvol : float
        Scatter volume from NonLinLoc (unit ? mˆ3 ).

    Returns:
    --------
    Tuple[float, str]
        - `qf`: Quality factor, a float representing the normalized quality of the event.
        - `q`: Quality category as a string ("A", "B", "C", or "D"):
            - "A": High quality (qf <= 0.25).
            - "B": Good quality (0.25 < qf <= 0.5).
            - "C": Moderate quality (0.5 < qf <= 0.75).
            - "D": Poor quality (qf > 0.75 and qf <= 1).
            - "E": Very poor quality (qf > 1.0)"

    Notes:
    ------
    The quality factor is computed as:
        qf = sqrt( sum(params_norm**2) / len(params_norm) )
    where `params_norm` are the input parameters normalized using predefined thresholds.

    Parameters like `nbpha` (number of phases) are inverted during normalization to reflect
    their inverse contribution to quality (more phases = better quality).
    """
    params = {
        "rms": rms,
        "erh": erh,
        "erz": erz,
        "used_phase_count": nbpha,
        "min_dist": dmin,
        "med_dist": dmed,
        "azgap": gap,
        "azgap2": gap2,
        "scat_vol": scatvol,
    }
    params2 = ["used_phase_count"]

    # Normalized values (Chauvenet's) for quality parameters
    normvalschauv = {
        "rms": 1.8,
        "erh": 16.0,
        "erz": 9.0,
        "used_phase_count": 21.0,
        "min_dist": 0.8,
        "med_dist": 3.0,
        "azgap": 352.0,
        "azgap2": 359.0,
        "scat_vol": 13900.0,
    }

    # Handle division by zero for nbpha (used_phase_count)
    if nbpha == 0:
        return float("inf"), "E"

    qf = [
        params[key] / normvalschauv[key] for key in params.keys() if key not in params2
    ] + [normvalschauv[key] / params[key] for key in params2]
    qf = np.sqrt(np.sum(np.array(qf) ** 2) / len(params))

    if qf <= 0.25:
        q = "A"
    elif qf <= 0.5:
        q = "B"
    elif qf <= 0.75:
        q = "C"
    elif qf <= 1:
        q = "D"
    else:
        q = "E"

    return qf, q


def classify_Michele_mod2(
    rms: float,
    erh: float,
    erz: float,
    nbpha: int,
    dmin: float,
    dmed: float,
    gap: float,
    gap2: float,
    scatvol: float,
    dloch: float,
    dz: float,
) -> Tuple[float, str]:
    """
    Classification using a modified version of Michele et al. 2019
    -----------------------------

    Input Attributes:
        - RMS (root mean square of residuals)
        - ERH (horizontal error)
        - ERZ (vertical error)
        - NBPHA (number of phases used)
        - DMIN (minimum distance to the nearest station in degree)
        - DMED (median distance of stations used in degree)
        - GAP (maximum azimuthal gap)
        - GAP2 (seconday azimuthal gap)
        - SCATVOL (scatter volume from NonLinLoc)
        - DLOCH (difference of location horizontal in km between location and expected location)
        - DZ (differnce of depth in km between depth and expected depth)

    Calculate quality factor from a modified version of Michele et al. 2019:
    qf = sqrt( sum(params_norm**2)/len(params_norm) )

    return the quality factor and the associated quality
    """
    nbpha_sigmoid = 1 - expit(0.3 * (nbpha - 21))
    params = {
        "rms": rms,
        "erh": erh,
        "erz": erz,
        "nbpha_sigmoid": nbpha_sigmoid,
        "min_dist": dmin,
        "med_dist": dmed,
        "azgap": gap,
        "azgap2": gap2,
        "scat_vol": scatvol,
        "dloch": dloch,
        "dz": np.abs(dz),
    }
    # Keys in params2b are already bounded in [0,1] by construction (sigmoid)
    # and are used raw in the qf sum below, without dividing by normvals2mad.
    params2b = ["nbpha_sigmoid"]

    normvals2mad = {
        "rms": 1.0,
        "erh": 10.5,
        "erz": 9.5,
        "min_dist": 0.8,
        "med_dist": 2.4,
        "azgap": 360.0,
        "azgap2": 360.0,
        "scat_vol": 271.0,
        "dloch": 4.9,
        "dz": 9.5,
    }

    qf = [
        params[key] / normvals2mad[key] for key in params.keys() if key not in params2b
    ] + [params[key] for key in params2b]
    qf = np.sqrt(np.sum(np.array(qf) ** 2) / len(params))

    if qf <= 0.2:
        q = "A"
    elif qf <= 0.4:
        q = "B"
    elif qf <= 0.6:
        q = "C"
    elif qf <= 1.0:
        q = "D"
    else:
        q = "E"

    return qf, q
