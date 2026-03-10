#!/usr/bin/env python
# -*- coding: utf-8 -*-
import json
import logging
import sys
import warnings
from collections import OrderedDict
from functools import lru_cache
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import pandas as pd
from icecream import ic
from obspy.core.event import Arrival
from obspy.core.event import Comment
from obspy.core.event import Pick
from scipy.special import erfc
from scipy.stats import norm
from shapely.geometry import LineString
from shapely.geometry import Point
from shapely.geometry import Polygon

# default logger (uses hierarchical name for selective level control)
logger = logging.getLogger("dbclust.relabel")


def get_value_from_key_in_list_of_dict(
    key: str, array: List[Dict[str, float]]
) -> Optional[float]:
    """Retrieve a value from a list of dictionaries given a key."""
    return next((i[key] for i in array if key in i), None)


def format_floats(d: dict) -> None:
    """Format floats in a dictionary to 4 decimal"""
    for key, value in d.items():
        if isinstance(value, dict):
            format_floats(value)
        elif isinstance(value, float):
            d[key] = f"{value:.4f}"
        elif isinstance(value, list):
            for i, item in enumerate(value):
                if isinstance(item, float):
                    value[i] = f"{item:.4f}"
                elif isinstance(item, dict):
                    format_floats(item)


def relabel_phase_and_comment_arrival(
    arrival: Arrival,
    pick: Pick,
    key: str,
    evaluation_score: float,
    polygons_score: OrderedDict,
    force_status: Optional[str] = "relabel",
) -> Tuple[str, Comment]:
    """
    Relabel phase name and add comment to the arrival object.
    Parameters:
        arrival (Arrival): The arrival object to add the comment to.
        pick (Pick): The pick object associated with the arrival.
        key (str): The phase key.
        evaluation_score (float): The evaluation score.
        polygons_score (OrderedDict): The scores of the phase in polygons.
        force_status (str, optional): Force status. Defaults to None.
    Returns:
        Tuple[str, Comment]: A tuple containing the relabel key and the comment object.

    Example: json format of the comment:
        {
            "relabel": {
                "prev_phase": "Sg"
                "action": "set by user",
                "eval_score": "0.9970",
                "scores": {
                    "Sg": "0.9083",
                    "Sn": "0.0028",
                    "Pn": "0.0003"
                }
            }
        }

    """

    polygons_score = OrderedDict(
        sorted(polygons_score.items(), key=lambda x: x[1], reverse=True)
    )

    comment_dict = {
        "relabel": {
            "prev_phase": arrival.phase,
            "action": force_status,
            "eval_score": evaluation_score,
            "scores": polygons_score,
        }
    }
    format_floats(comment_dict)
    comment = Comment(text=json.dumps(comment_dict))
    arrival.comments.append(comment)

    if key:
        arrival.phase = key
        pick.phase_hint = key

    relabel_key = f"{pick.waveform_id.get_seed_string()}-{arrival.phase}-{pick.time}"
    return relabel_key, comment


def softmax(scores: dict, temperature: float = 1.0) -> dict:
    """
    Apply softmax to a dictionary of scores with a temperature parameter.

    Args:
        scores (dict): A dictionary where keys are polygon names and values are raw scores.
        temperature (float, optional): Temperature parameter for softmax.
            Lower values (<1) make probabilities sharper,
            higher values (>1) make them more uniform. Default is 1.0.

    Returns:
        dict: A dictionary with the same keys but normalized probabilities as values.
    """
    if not scores:
        return {}

    # Convert scores to numpy array
    score_values = np.array(list(scores.values()))

    # Apply temperature scaling (avoid division by zero)
    if temperature <= 0:
        raise ValueError("Temperature must be a positive value.")

    score_values /= temperature

    # Compute softmax
    exp_scores = np.exp(
        score_values - np.max(score_values)
    )  # Stability trick to avoid overflow
    softmax_probs = exp_scores / np.sum(exp_scores)

    # Return probabilities as a dictionary
    return {key: prob for key, prob in zip(scores.keys(), softmax_probs)}


def get_best_polygon_for_point(
    point: Point,
    phase_info: str,
    df_polygons: pd.DataFrame,
    sigma_list: List[Dict[str, float]],
    eval_threshold: float = 0.05,
) -> Tuple[Optional[str], Optional[float], OrderedDict, float]:
    """
    Finds the best polygon for a given point within a DataFrame of polygons using Bayesian probability.

    Args:
        point (Point): The point to find the best polygon for.
        phase (str): The phase name of the point ("P", "S", "Pn", "Sn", "Pg", "Sg")
        df_polygons (pd.DataFrame): The DataFrame of polygons to search within.
        sigma_list (List[Dict[str, float]]): The sigma values for each polygon.
        eval_threshold (float, optional): The threshold for the evaluation score. Defaults to 0.05.
    Returns:
        Tuple[Union[str, None], Union[float, None]]:
        A tuple containing the name of the best polygon and its probability, or None and None if the point is in a complex zone.
        polygon_score (dict): A dictionary containing the probability of each polygon.,
        evaluation_score (float): A float quantifying the difference between the two best probabilities.
    """
    mu = 0
    polygon_score = OrderedDict()
    for zone_id, zone_polygon in df_polygons.iterrows():
        if zone_polygon["geometry"].contains(point):
            distance_between_longest_edges, edges = get_distance_between_longest_edges(
                zone_polygon["geometry"], zone_polygon["name"]
            )
            # find the minimum distance between the point and the edges
            dist = min(point.distance(e) for e in edges)

            # Convert distance from the edge to distance to the "center"
            # fixme: take into account of Mu
            dist = 1 - dist / (distance_between_longest_edges / 2)

            # Get the sigma value for the polygon (default to 1 if not found)
            sigma = (
                get_value_from_key_in_list_of_dict(zone_polygon["name"], sigma_list)
                or 1
            )
            if sigma <= 0:
                logger.warning(
                    f"Sigma value for polygon {zone_polygon['name']} must be positive. Defaulting to 1."
                )
                sigma = 1

            proba = norm.sf(dist, mu, sigma)  # survival function (1 - cdf)
            #proba = erfc((dist-mu) / (sigma*np.sqrt(2))) / 2
            polygon_score[zone_polygon["name"]] = proba

    if not polygon_score:
        logger.debug(f"Point {point} is not in any zone.")
        return None, None, polygon_score, 0

    # --- Previous approach (kept for reference) ---
    # Softmax with temperature + difference of top-two scores.
    # Requires empirical temperature calibration; T=1.5 was chosen so that a
    # 2.5:1 score ratio stays below the 0.10 confidence threshold.
    # Needs tuning.
    #
    # polygon_score = softmax(polygon_score, temperature=1.5)
    # sorted_probs = sorted(polygon_score.values(), reverse=True)
    # if len(sorted_probs) < 2:
    #     confidence_ratio = 1.0
    # else:
    #     confidence_ratio = sorted_probs[0] - sorted_probs[1]

    # --- Current approach ---
    # Confidence = normalised score advantage of the best polygon over the second best.
    #
    #   confidence = (best - second) / (best + second)
    #
    # Properties:
    #   - 0   when scores are equal (full ambiguity, e.g. polygon overlap zone)
    #   - 1   when only one polygon contains the point
    #   - 0.5 when best score is 3× the second (natural "clear decision" boundary)
    #   - scale-invariant: depends only on the ratio, not on absolute score values
    #   - no temperature parameter to calibrate
    #
    # Typical threshold, minimum score ratio to accept a relabelling:
    #   0.30 => ratio ≥ 1.86:1
    #   0.45 => ratio ≥ 2.64:1  used as default (eval_threshold)
    #   0.50 => ratio ≥ 3.00:1
    sorted_scores = sorted(polygon_score.values(), reverse=True)

    if len(sorted_scores) < 2:
        confidence_ratio = 1.0  # single polygon: unambiguous
    else:
        best, second = sorted_scores[0], sorted_scores[1]
        confidence_ratio = (best - second) / (best + second)

    # Check if the confidence is sufficient
    if confidence_ratio < eval_threshold:
        logger.debug(
            f"{phase_info}: {point} is in a complex zone. Confidence ratio={confidence_ratio:.4f}."
        )
        return None, None, polygon_score, confidence_ratio

    proba_max = max(polygon_score.values())
    key_max = max(polygon_score, key=polygon_score.get)
    return key_max, proba_max, polygon_score, confidence_ratio


@lru_cache(maxsize=None)
def get_distance_between_longest_edges(
    p: Polygon, name: str = None
) -> Tuple[float, List[LineString]]:
    """
    Get the distance between the two longest edges of a polygon

    Args:
        p (Polygon): The polygon for which to calculate the distance between the longest edges.
        name (str, optional): A name for the polygon (default: None).
    Returns:
        Tuple[float, List[LineString]]: A tuple containing the distance between the longest edges and the longest edges.
    """
    # warnings.filterwarnings("error")
    # try:
    #     coords = list(p.minimum_rotated_rectangle.exterior.coords)
    # except Exception as e:
    #     ic(e, name, p, p.minimum_rotated_rectangle)
    # warnings.resetwarnings()

    # There is warning when computing the minimum rotated rectangle
    # but the result is correct regarding the distance between the longest edges
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        coords = list(p.minimum_rotated_rectangle.exterior.coords)

    edges = [
        (
            LineString([coords[i], coords[i + 1]]).length,
            LineString([coords[i], coords[i + 1]]),
        )
        for i in range(len(coords) - 1)
    ]

    edges.sort(reverse=True, key=lambda x: x[0])
    longest_edges = edges[:2]

    distance_between_longest_edges = longest_edges[0][1].distance(longest_edges[1][1])

    edges_only = [edge for _, edge in edges]

    return distance_between_longest_edges, edges_only
