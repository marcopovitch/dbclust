#!/usr/bin/env python
# -*- coding: utf-8 -*-
import json
import logging
import math
import sys
import warnings
from collections import OrderedDict
from functools import lru_cache
from typing import Dict
from typing import List
from typing import Tuple
from typing import Union

import pandas as pd
from icecream import ic
from obspy.core.event import Arrival
from obspy.core.event import Comment
from obspy.core.event import Pick
from scipy.stats import norm
from shapely.geometry import LineString
from shapely.geometry import Point
from shapely.geometry import Polygon

# default logger
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger("relabel")
logger.setLevel(logging.INFO)


def get_value_from_key_in_list_of_dict(
    key: str, array: List[Dict[str, float]]
) -> Union[str, None]:
    # ic(key, array)
    for i in array:
        if key in i:
            return i[key]
    return None


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


def add_relabel_comment_to_arrival(
    arrival: Arrival,
    pick: Pick,
    key: str,
    evaluation_score: float,
    polygons_score: OrderedDict,
    force_status: str = None,
) -> Union[str, str]:
    """
    Adds a relabel comment to the arrival object.
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
                "action": "set by user",
                "eval_score": "0.9970",
                "scores": {
                    "Pn": "0.0003",
                    "Sg": "0.9083",
                    "Sn": "0.0028"
                }
            }
        }
    """

    if not force_status:
        force_status = "relabel"

    # sort polygons_score by value
    polygons_score = OrderedDict(
        sorted(polygons_score.items(), key=lambda x: x[1], reverse=True)
    )

    # json format
    comment_dict = {
        "relabel": {
            "action": force_status,
            "eval_score": evaluation_score,
            "scores": polygons_score,
        }
    }
    format_floats(comment_dict)
    comment = Comment(text=json.dumps(comment_dict))
    arrival.comments.append(comment)
    # relabel only if key is not None
    if key:
        arrival.phase = key
        pick.phase_hint = key
    relabel_key = f"{pick.waveform_id.get_seed_string()}-{arrival.phase}-{pick.time}"

    return relabel_key, comment


def get_best_polygon_for_point(
    point: Point,
    phase_info: str,
    df_polygons: pd.DataFrame,
    sigma_list: List[Dict[str, float]],
    eval_threshold: float = 0.05,
) -> Tuple[Union[str, None], Union[float, None], OrderedDict, float]:
    """
    Finds the best polygon for a given point within a DataFrame of polygons.
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
            dist = None
            for e in edges:
                d = point.distance(e)
                if not dist:
                    dist = d
                    continue
                if d < dist:
                    dist = d

            # Convert distance from the edge to the "center" to probability
            dist = 1 - dist / (distance_between_longest_edges / 2)

            # Get the sigma value for the polygon
            sigma = get_value_from_key_in_list_of_dict(zone_polygon["name"], sigma_list)
            if not sigma:
                logger.warning(
                    f"Sigma value not found for polygon {zone_polygon['name']}. Using default value of 1."
                )
                sigma = 1
            # Normalized probability
            proba = norm.pdf(dist, mu, sigma) / norm.pdf(0, mu, sigma)
            # ic(dist, sigma, proba)
            polygon_score[zone_polygon["name"]] = proba

    if not polygon_score:
        logger.debug(f"Point {point} is not in any zone.")
        return None, None, polygon_score, 0

    # check key name for multiple polygons with the same phase Pg, Pn or Sg, Sn
    # if len([k for k in polygon_score.keys() if phase[0] in k]) > 1:
    #     logger.info(f"Point {point} is in zone inside multiple polygons. ")
    #     return None, None, polygon_score, 0

    # Do not allow multiple polygons with proba > proba_threshold
    # if len([p for p in polygon_score.values() if p >= proba_threshold]) > 1:
    #     ic(polygon_score)
    #     logger.info(
    #         f"Point {point} is in a complex zone. "
    #         f"There are at least two polygons with proba > {proba_threshold}."
    #     )
    #     return None, None, polygon_score

    # Quantify the difference (in %) between the two probabilities
    if len(polygon_score) > 1:
        proba_values = list(polygon_score.values())
        # Sort the probabilities in descending order
        proba_values.sort(reverse=True)

        # Get percentage difference between the two max probabilities
        evaluation_score = (proba_values[0] - proba_values[1]) / proba_values[0]

        if evaluation_score < eval_threshold:
            logger.debug(
                f"{phase_info}: {point} is in a complex zone. "
                f"The evaluation score between the two max probabilities is {evaluation_score:.4f} < {eval_threshold}."
            )
            return None, None, polygon_score, evaluation_score
    else:
        # Set evaluation score to 1 if there is only one polygon
        evaluation_score = 1

    # Return the polygon with the max proba
    proba_max = max(polygon_score.values())
    key_max = max(polygon_score, key=polygon_score.get)
    return key_max, proba_max, polygon_score, evaluation_score


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

    edges = []
    for i in range(len(coords) - 1):
        edge = LineString([coords[i], coords[i + 1]])
        length = edge.length
        edges.append((length, edge))

    edges.sort(reverse=True, key=lambda x: x[0])
    longest_edges = edges[:2]

    distance_between_longest_edges = longest_edges[0][1].distance(longest_edges[1][1])

    edges_only = [edge for length, edge in edges]

    return distance_between_longest_edges, edges_only
