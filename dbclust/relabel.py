#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import Tuple
from typing import Union

import pandas as pd
from shapely.geometry import Point


def get_best_polygon_for_point(
    point: Point, df_polygons: pd.DataFrame
) -> Tuple[Union[str, None], Union[float, None]]:
    polygon_score = {}
    for zone_id, zone_polygon in df_polygons.iterrows():
        if zone_polygon["geometry"].contains(point):
            polygon_score[zone_polygon["name"]] = point.distance(
                zone_polygon["geometry"].boundary
            )

    # get the polygon with the max score
    if polygon_score:
        score_min = max(polygon_score.values())
        key_min = max(polygon_score, key=polygon_score.get)
        return key_min, score_min
    else:
        return None, None
