#!/usr/bin/env python
import sys
import argparse
import pandas as pd
import numpy as np
from obspy import read_events
from obspy.geodetics import locations2degrees
import pygmt
import logging

# default logger
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger("scatter_plot")
logger.setLevel(logging.INFO)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-e",
        "--eventid",
        default=None,
        dest="event_id",
        help="event id",
        type=str,
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action='store_true',
        dest="verbose",
        help="verbose output",
    )
    args = parser.parse_args()
    if not args.event_id:
        parser.print_help()
        sys.exit(255)

    #variant = "World_Imagery"
    variant = "World_Street_Map"
    tiles_server = (
        # "https://{s}.tile.openstreetmap.fr/osmfr/{z}/{x}/{y}.png"
        "https://server.arcgisonline.com/ArcGIS/rest/services/%s/MapServer/tile/{z}/{y}/{x}" % variant

    )
    # tiles_server = "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"

    # parameters
    scat_file = f"{args.event_id}.scat"
    evt_file = f"{args.event_id}.sc3ml"

    # Loc from LocSAT event
    locsat = None
    cat = read_events(evt_file)
    e = cat[0]
    for o in e.origins:
        # FIXME: get the latest or the one with the max #pahases
        if "LOCSAT" in o.method_id.id:
            # and o.evaluation_mode == "manual":
            locsat_latitude = o.latitude
            locsat_longitude = o.longitude
            locsat_depth = -1 * o.depth / 1000.0
            locsat_vmodel = o.earth_model_id.id.split("/")[-1]
            locsat = True
            logger.info(
                f"Coord from *LocSAT*, lat: {locsat_latitude:.3f}, "
                f"lon: {locsat_longitude:.3f}, "
                f"depth: {locsat_depth:.1f} km, "
                f"vmodel: {locsat_vmodel}, "
                f"evaluation: {o.evaluation_mode}"
            )
            break

    # Loc fron event
    o = e.preferred_origin()
    vmodel = o.earth_model_id.id.split("/")[-1]
    logger.info(
        f"Coord from *event*, lat: {o.latitude:.3f}, "
        f"lon: {o.longitude:.3f},"
        f"depth: {o.depth/1000.0:.1f} km, "
        f"vmodel: {vmodel}, "
        f"evaluation: {o.evaluation_mode}"
    )

    # loc from scatter data
    df = pd.read_csv(
        scat_file,
        delim_whitespace=True,
        skiprows=3,
        header=None,
        names=("latitude", "longitude", "depth", "h1", "h2"),
    )
    df["longitude"] = df["longitude"].astype(np.float64)
    df["latitude"] = df["latitude"].astype(np.float64)
    df.sort_values(by=["h2"], inplace=True)
    df["depth"] = -1 * df["depth"]

    # get the max proba
    dfmax = df[df["h2"] == df["h2"].max()]
    if len(dfmax) > 1:
        logger.warning(f"len(dfmax): {len(dfmax)}")

    # get max proba barycenter to get the loc
    max_latitude = dfmax["latitude"].mean()
    max_longitude = dfmax["longitude"].mean()
    max_depth = dfmax["depth"].mean()
    logger.info(
        f"Coord from *scatter*, lat: {max_latitude:.3f}, "
        f"lon: {max_longitude:.3f},"
        f"depth: {max_depth:.1f} km"
    )

    # Plot parameters
    max_marker = "star"
    size_marker = 8
    transparency_marker = 70

    # Adapt region to the dataset
    df["distance_deg"] = df[["latitude", "longitude"]].apply(
        lambda row: locations2degrees(row[0], row[1], max_latitude, max_longitude),
        axis=1,
    )
    offset = df["distance_deg"].max() * 2  # Deg
    depth_offset = 1  # km
    region = [
        max_longitude - offset,
        max_longitude + offset,
        max_latitude - offset,
        max_latitude + offset,
    ]

    # Start plot
    fig = pygmt.Figure()
    pygmt.makecpt(cmap="viridis", series=[df["h2"].min(), df["h2"].max()])
    pygmt.config(FORMAT_GEO_MAP="ddd.xx")
    pygmt.config(MAP_FRAME_TYPE="plain")
    pygmt.config(FONT_HEADING="10p")
    if args.verbose:
        pygmt.config(GMT_VERBOSE="d")


    with fig.subplot(
        nrows=2,
        ncols=2,
        figsize=("18c", "18c"),
        frame=["afg"],
        margins=["0.6c", "0.2c"],
        title=f"{args.event_id}: lat: {max_latitude:.3f}, lon: {max_longitude:.3f}, depth: {max_depth:.1f} km, vmodel: {o.earth_model_id.id.split('/')[-1]}, evaluation: {o.evaluation_mode}",
    ):
        ############
        # lat, lon #
        ############
        with fig.set_panel(panel=[0, 1]):
            fig.tilemap(
                region=region,
                projection="X8/8",
                source=tiles_server,
                lonlat=True,
            )

            fig.plot(
                projection="X8/8",
                x=df["longitude"],
                y=df["latitude"],
                style="c0.2c",
                cmap=True,
                fill=df["h2"],
                transparency=transparency_marker,
            )

            # NLL loc all max PDF
            fig.plot(
                projection="X8/8",
                x=dfmax["longitude"],
                y=dfmax["latitude"],
                style="c0.2c",
                fill="red",
                cmap=False,
                transparency=transparency_marker,
            )
            fig.plot(
                projection="X8/8",
                x=max_longitude,
                y=max_latitude,
                style="a0.3c",
                fill="red",
                pen="black",
                cmap=False,
                transparency=0,
            )

            # locsat loc
            if locsat:
                fig.plot(
                    projection="X8/8",
                    x=locsat_longitude,
                    y=locsat_latitude,
                    style="a0.3c",
                    fill="black",
                    pen="black",
                    cmap=False,
                    transparency=0,
                )

        ##############
        # lon, depth #
        ##############
        with fig.set_panel(panel=[1, 1]):
            fig.basemap(
                region=[
                    max_longitude - offset,
                    max_longitude + offset,
                    df["depth"].min() - depth_offset,
                    0,
                ],
                projection="X8/8",
                frame=["afg0", "Wsne", "y+ldepth"],
            )

            fig.plot(
                projection="X8/8",
                x=df["longitude"],
                y=df["depth"],
                style="c0.2c",
                cmap=True,
                fill=df["h2"],
                transparency=transparency_marker,
            )

            # NLL loc
            fig.plot(
                projection="X8/8",
                x=dfmax["longitude"],
                y=dfmax["depth"],
                style="c0.2c",
                fill="red",
                cmap=False,
                transparency=transparency_marker,
            )
            fig.plot(
                projection="X8/8",
                x=max_longitude,
                y=max_depth,
                style="a0.3c",
                fill="red",
                pen="black",
                cmap=False,
                transparency=0,
            )

            # locsat loc
            if locsat:
                fig.plot(
                    projection="X8/8",
                    x=locsat_longitude,
                    y=locsat_depth,
                    style="a0.3c",
                    fill="black",
                    pen="black",
                    cmap=False,
                    transparency=0,
                )

        ##############
        # lat, depth #
        ##############
        with fig.set_panel(panel=[0, 0]):
            fig.basemap(
                region=[
                    df["depth"].min() - depth_offset,
                    0,
                    max_latitude - offset,
                    max_latitude + offset,
                ],
                projection="X8/8",
                frame=["afg0", "wSne", "x+ldepth"],
            )

            fig.plot(
                projection="X8/8",
                x=df["depth"],
                y=df["latitude"],
                style="c0.2c",
                cmap=True,
                fill=df["h2"],
                transparency=transparency_marker,
            )

            # NLL loc
            fig.plot(
                projection="X8/8",
                x=dfmax["depth"],
                y=dfmax["latitude"],
                style="c0.2c",
                fill="red",
                cmap=False,
                transparency=transparency_marker,
            )
            fig.plot(
                projection="X8/8",
                x=max_depth,
                y=max_latitude,
                style="a0.3c",
                fill="red",
                pen="black",
                cmap=False,
                transparency=0,
            )

            # locsat loc
            if locsat:
                fig.plot(
                    projection="X8/8",
                    x=locsat_depth,
                    y=locsat_latitude,
                    style="a0.3c",
                    fill="black",
                    pen="black",
                    cmap=False,
                    transparency=0,
                )

        ########
        # logo #
        ########
        with fig.set_panel(panel=[1, 0]):
            # fig.basemap(region=[0, 8, 0, 8], projection="X8/8", frame="a0f0g0")
            # fig.image(
            #     imagefile="/Users/marc/Projets/Lalaigne/logos/logo-bcsf-renass.png",
            #     position="g1.6/0.6+w3c+jCM",
            #     box=False,
            # )
            fig.coast(
                region=[-6, 10, 41, 52],
                # projection="X8/8",
                projection="A2/47/8/8c",
                land="gray",
                water="white",
                frame="g5",
                dcw=[
                    "GB+p0.5p,black",
                    "IT+p0.5p,black",
                    "ES+p0.5p,black",
                    "BE+p0.75p,black",
                    "DE+p0.5p,black",
                    "France+p0.5p,steelblue",
                    "CH+p0.2p,black",
                ],
            )
            # NLL loc
            fig.plot(
                # projection="X8/8",
                projection="A2/47/8/8c",
                x=max_longitude,
                y=max_latitude,
                style="a0.3c",
                fill="red",
                pen="black",
                cmap=False,
                transparency=0,
            )

    fig.savefig(
        f"{scat_file}.png", transparent=False, anti_alias=True, crop=True, show=True
    )
