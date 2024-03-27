#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import os

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="Input CSV file path")
    parser.add_argument("-p", "--polygon", help="Polygon file path")
    parser.add_argument("-o", "--output", help="Output CSV file path")
    parser.add_argument(
        "-k", "--keep", help="Keep only points inside the polygon", action="store_true"
    )
    args = parser.parse_args()

    if not args.input or not args.polygon or not args.output:
        parser.print_help()
        exit(1)

    # Check if input file exists
    if not os.path.isfile(args.input):
        print(f"Input file '{args.input}' does not exist.")
        exit(1)

    # Check if polygon file exists
    if not os.path.isfile(args.polygon):
        print(f"Polygon file '{args.polygon}' does not exist.")
        exit(1)

    # Read CSV file
    df = pd.read_csv(args.input)

    # Convert coordinates to points with EPSG:4326 spatial reference
    geometry = [Point(xy) for xy in zip(df.longitude, df.latitude)]
    crs = "epsg:4326"
    gdf_points = gpd.GeoDataFrame(df, geometry=geometry, crs=crs)

    # Read the polygon file
    gdf_polygon = gpd.read_file(args.polygon)

    joined = gpd.sjoin(gdf_points, gdf_polygon, how="left", predicate="within")

    # The "index_right" column contains the indices of the polygon in which each point is located
    # If the point is inside the polygon, the value of "index_right" is the index of the polygon; otherwise, NaN
    df["inside_ZEE"] = ~joined["index_right"].isna()

    # drop columns that are not needed
    for col in ["time2", "event_type2", "time2_2"]:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    if args.keep:
        # Keep only points inside the polygon
        df = df[df["inside_ZEE"]]

    # Save the result to a new CSV file
    df.to_csv(args.output, index=False)
