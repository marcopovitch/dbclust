import sys
import os
from typing import List
import geopandas as gpd
from shapely.geometry import Polygon, Point
import logging

# default logger
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger("dbclust")
logger.setLevel(logging.INFO)


def load_zones(zones: List[dict], nll_conf: dict) -> gpd.GeoDataFrame:
    """Associate NLL models to polygones given zones definitions

    Args:
        zones (List[dict]): zones definitions
        nll_conf (dict): NonLinLoc models

    Returns:
        gpd.GeoDataFrame: geopandas dataframe containing all polygones
    """
    records = []
    for z in zones:
        name = z["name"]
        vmodel = z["velocity_profile"]

        # check if model exists
        found = False
        for vprofile in nll_conf["velocity_profile"]:
            if vprofile["name"] == vmodel:
                found = True
                break
        if not found:
            logger.error(f"Can not find velocity profile {vmodel}")
            return gpd.GeoDataFrame()

        polygon = Polygon(z["points"])
        records.append(
            {
                "name": name,
                "velocity_profile": vprofile["name"],
                "template": os.path.join(
                    nll_conf["nll_template_path"], vprofile["template"]
                ),
                "geometry": polygon,
            }
        )

    gdf = gpd.GeoDataFrame(records)
    return gdf


def find_zone(
    latitude: float = None, longitude: float = None, zones: gpd.GeoDataFrame = None
) -> gpd:
    """Find zone

    Args:
        latitude (float, optional): Defaults to None.
        longitude (float, optional): Defaults to None.
        zones (gpd.GeoDataFrame, optional): Defaults to None.

    Returns:
        gpd.GeoDataFrame: geodataframe requested or an empty one if request failed.
    """
    point_shapely = Point(longitude, latitude)
    for index, row in zones.iterrows():
        if row["geometry"].contains(point_shapely):
            return row
    return gpd.GeoDataFrame()
