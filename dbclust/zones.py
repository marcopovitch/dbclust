import sys
import os
import geopandas as gpd
from shapely.geometry import Polygon, Point
import logging

# default logger
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger("dbclust")
logger.setLevel(logging.INFO)


def load_zones(zones, nll_conf):
    """Returns a geodataframe containing:"
    - zone information
    - nll template information
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
                "template": os.path.join(nll_conf['nll_template_path'], vprofile["template"]),
                "geometry": polygon,
            }
        )

    gdf = gpd.GeoDataFrame(records)
    return gdf


def find_zone(latitude=None, longitude=None, zones=None):
    """Returns the geodataframe requested
    or an empty one if request failed.
    """
    point_shapely = Point(longitude, latitude)
    for index, row in zones.iterrows():
        if row["geometry"].contains(point_shapely):
            return row
    return gpd.GeoDataFrame()
