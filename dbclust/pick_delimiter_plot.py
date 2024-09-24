#!/usr/bin/env python
import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from config import DBClustConfig
from config import Zone
from icecream import ic


def pick_delimiter_plot_all(conf):
    # Plotly figure
    fig = go.Figure()

    # iterate over all zones and plot all the polygons on the same figure
    for z in conf.zones.zones:
        zone = conf.zones.get_zone_from_name(z.name)
        # ic(zone)
        poly_df = zone.picks_delimiter
        # ic(poly_df)

        # Iterate over all polygons in the DataFrame
        for idx, row in poly_df.iterrows():
            polygon = row["geometry"]
            x, y = polygon.exterior.xy

            x_list = list(x)
            y_list = list(y)

            fig.add_trace(
                go.Scatter(
                    x=x_list, y=y_list, fill="toself", name=f'{z.name}-{row["name"]}'
                )
            )

    fig.update_layout(
        title=f"zone: all",
        xaxis_title="Distance (deg)",
        yaxis_title="Time (s)",
        showlegend=True,
        #width=800,
        #height=600,
    )

    #fig.write_image("pick_delimiter-all.png")
    fig.show()


def pick_delimiter_plot(conf, zone_name):
    zone = conf.zones.get_zone_from_name(zone_name)
    # ic(zone)
    poly_df = zone.picks_delimiter
    # ic(poly_df)

    # Plotly figure
    fig = go.Figure()

    # Iterate over all polygons in the DataFrame
    for idx, row in poly_df.iterrows():
        polygon = row["geometry"]
        x, y = polygon.exterior.xy

        x_list = list(x)
        y_list = list(y)

        fig.add_trace(go.Scatter(x=x_list, y=y_list, fill="toself", name=row["name"]))

    fig.update_layout(
        title=f"zone: {zone_name}",
        xaxis_title="Distance (deg)",
        yaxis_title="Time (s)",
        showlegend=True,
    )

    fig.write_image(f"pick_delimiter-{zone_name}.png")
    # fig.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--conf_file",
        default=None,
        dest="config_file",
        help="yaml configuration file.",
        type=str,
    )

    parser.add_argument(
        "-t",
        "--conf_type",
        default=None,
        dest="config_type",
        help="std|reloc configuration type.",
        type=str,
    )

    # add -z option for zone name
    parser.add_argument(
        "-z",
        "--zone",
        default=None,
        dest="zone_name",
        help="zone name",
        type=str,
    )

    args = parser.parse_args()
    if not args.config_file:
        parser.print_help()
        sys.exit(255)

    myconf = DBClustConfig(args.config_file, config_type=args.config_type)
    # myconf.show()

    if args.zone_name:
        pick_delimiter_plot(myconf, args.zone_name)
    else:
        # iterate over all zones
        #for zone in myconf.zones.zones:
        #    pick_delimiter_plot(myconf, zone.name)
        pick_delimiter_plot_all(myconf)
