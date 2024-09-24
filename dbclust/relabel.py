#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
import math
import sys
from collections import OrderedDict
from functools import lru_cache
from typing import List
from typing import Tuple
from typing import Union

import pandas as pd
from icecream import ic
from obspy.core.event import Arrival
from scipy.stats import norm
from shapely.geometry import LineString
from shapely.geometry import Point
from shapely.geometry import Polygon

# default logger
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger("relabel")
logger.setLevel(logging.INFO)


def get_best_polygon_for_point(
    point: Point, phase_info: str, df_polygons: pd.DataFrame, eval_threshold: float = 0.05
) -> Tuple[Union[str, None], Union[float, None], OrderedDict, float]:
    """
    Finds the best polygon for a given point within a DataFrame of polygons.
    Args:
        point (Point): The point to find the best polygon for.
        phase (str): The phase name of the point ("P", "S", "Pn", "Sn", "Pg", "Sg")
        df_polygons (pd.DataFrame): The DataFrame of polygons to search within.
        eval_threshold (float, optional): The threshold for the evaluation score. Defaults to 0.05.
    Returns:
        Tuple[Union[str, None], Union[float, None]]: A tuple containing the name of
        the best polygon and its probability, or None and None if the point is in a complex zone.
        polygon_score (dict): A dictionary containing the probability of each polygon.,
        evaluation_score (float): A float quantifying the difference between the two best probabilities.
    """
    mu = 0
    sigma = 1
    # half_gaussian_mean = sigma * math.sqrt(2 / math.pi)

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
            proba = norm.pdf(dist, mu, sigma) / norm.pdf(0, mu, sigma)
            polygon_score[zone_polygon["name"]] = proba

    if not polygon_score:
        logger.info(f"Point {point} is not in any zone.")
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
            logger.info(
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
    coords = list(p.minimum_rotated_rectangle.exterior.coords)
    # To be done: catch warning for minimum_rotated_rectangle

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
