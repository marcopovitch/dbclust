#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
import sys
from functools import lru_cache
from typing import List
from typing import Tuple
from typing import Union

import pandas as pd
from icecream import ic
from shapely.geometry import LineString
from shapely.geometry import Point
from shapely.geometry import Polygon

# default logger
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger("relabel")
logger.setLevel(logging.INFO)


def get_best_polygon_for_point_old(
    point: Point, df_polygons: pd.DataFrame, proba_threshold: float = 0.68
) -> Tuple[Union[str, None], Union[float, None]]:
    polygon_score = {}
    for zone_id, zone_polygon in df_polygons.iterrows():
        if zone_polygon["geometry"].contains(point):

            distance_between_longest_edges, _ = get_distance_between_longest_edges(
                zone_polygon["geometry"], zone_polygon["name"]
            )

            dist = point.distance(zone_polygon["geometry"].boundary)

            # convert to probability
            # proba = 1 - dist / (distance_between_longest_edges / 2)
            proba = dist / (distance_between_longest_edges / 2)

            if proba > proba_threshold:
                polygon_score[zone_polygon["name"]] = proba
            else:
                polygon_score[zone_polygon["name"]] = 0

    # get the polygon with the max proba
    if polygon_score:
        proba_max = max(polygon_score.values())
        key_max = max(polygon_score, key=polygon_score.get)
        return key_max, proba_max
    else:
        return None, None


def get_best_polygon_for_point(
    point: Point, df_polygons: pd.DataFrame, proba_threshold: float = 0.90
) -> Tuple[Union[str, None], Union[float, None]]:
    """
    Finds the best polygon for a given point within a DataFrame of polygons.
    Args:
        point (Point): The point to find the best polygon for.
        df_polygons (pd.DataFrame): The DataFrame of polygons to search within.
        proba_threshold (float, optional): The probability threshold. Defaults to 0.90.
    Returns:
        Tuple[Union[str, None], Union[float, None]]: A tuple containing the name of
        the best polygon and its probability, or None and None if the point is in a complex zone.
    """
    polygon_score = {}
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

            # convert to probability
            # proba = 1 - dist / (distance_between_longest_edges / 2)
            proba = dist / (distance_between_longest_edges / 2)
            polygon_score[zone_polygon["name"]] = proba

    # Get the polygon with the max proba
    # if there are at least two polygons with proba > proba_threshold
    # return None and None to indicate that the point is in a complex zone
    if (
        not polygon_score
        or len([p for p in polygon_score.values() if p > proba_threshold]) > 1
    ):
        logger.debug(
            f"Point {point} is in a complex zone. "
            f"There are at least two polygons with proba > {proba_threshold}."
        )
        return None, None

    # Return the polygon with the max proba
    proba_max = max(polygon_score.values())
    key_max = max(polygon_score, key=polygon_score.get)
    return key_max, proba_max


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
