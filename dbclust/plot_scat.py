#!/usr/bin/env python
import argparse
import glob
import logging
import os
import sys
import urllib.parse
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pygmt
from obspy import read_events
from obspy.geodetics import locations2degrees

from dbclust.localization_error import get_erh_erz
from dbclust.localization_quality import (
    classify,
    classify_Michele_mod,
    get_classification_text,
)

# default logger
logger = logging.getLogger("scatter_plot")

# Default topography resolution for profiles
TOPO_RESOLUTION = "03s"  # 3 arc-seconds (~90m)

# Path to logo file
LOGO_PATH = "/Users/marc/Documents/Geothermie/Vendenheim Fonroche/Reseaux_Sismicite/logo-bcsf-renass.png"



def _origin_sort_key(origin):
    """Return a comparable timestamp for sorting origins safely."""
    creation_info = getattr(origin, "creation_info", None)
    if creation_info is not None:
        for attr in ("creation_time", "time"):
            value = getattr(creation_info, attr, None)
            if value is not None:
                return value
    if getattr(origin, "time", None) is not None:
        return origin.time
    return datetime.min


def _draw_text_block(fig, *, x, start_y, lines, line_spacing=0.4, justify="LM"):
    """Render a vertical stack of (text, font) tuples with consistent spacing."""
    for idx, (text, font) in enumerate(lines):
        fig.text(
            x=x,
            y=start_y - idx * line_spacing,
            text=text,
            font=font,
            justify=justify,
            no_clip=True,
        )


def _get_topography_profile(lon_start, lon_end, lat_start, lat_end, resolution=None, n_points=200):
    """Extract topography profile between two points using GMT earth_relief data.

    Returns a DataFrame with columns: longitude, latitude, elevation_km.
    elevation_km uses the same convention as depth: negative = below sea level.
    """
    if resolution is None:
        resolution = TOPO_RESOLUTION

    # Build region with margin (need minimum extent for grid loading)
    lon_min, lon_max = min(lon_start, lon_end), max(lon_start, lon_end)
    lat_min, lat_max = min(lat_start, lat_end), max(lat_start, lat_end)
    # Ensure minimum region extent for grid loading
    margin = 0.05  # degrees
    if lon_max - lon_min < 0.01:
        lon_min -= margin
        lon_max += margin
    if lat_max - lat_min < 0.01:
        lat_min -= margin
        lat_max += margin
    region = [lon_min - 0.01, lon_max + 0.01, lat_min - 0.01, lat_max + 0.01]

    try:
        # Load earth relief data for the region
        logger.debug(f"Loading earth relief for region {region}")
        grid = pygmt.datasets.load_earth_relief(resolution=resolution, region=region)  # type: ignore[arg-type]

        # Create explicit sample points along the profile
        lons = np.linspace(lon_start, lon_end, n_points)
        lats = np.linspace(lat_start, lat_end, n_points)
        points_df = pd.DataFrame({"lon": lons, "lat": lats})

        # Extract elevation along profile using explicit points
        track_result = pygmt.grdtrack(points=points_df, grid=grid, newcolname="elevation")

        if track_result is not None and len(track_result) > 0:
            # grdtrack returns a DataFrame when given a DataFrame input
            track_df = track_result if isinstance(track_result, pd.DataFrame) else pd.DataFrame(track_result)
            # Rename columns for consistency
            if "lon" in track_df.columns:
                track_df = track_df.rename(columns={"lon": "longitude", "lat": "latitude"})
            # Convert elevation from meters to km
            # Keep the same sign: positive elevation (mountains) = positive km
            # negative elevation (below sea level) = negative km
            # This is consistent with the depth convention in plots where
            # negative = below sea level, positive = above sea level
            track_df["elevation_km"] = track_df["elevation"] / 1000.0
            logger.debug(f"Topography profile: {len(track_df)} points, "
                        f"elevation range: {track_df['elevation'].min():.0f} to "
                        f"{track_df['elevation'].max():.0f} m, "
                        f"elevation_km range: {track_df['elevation_km'].min():.2f} to "
                        f"{track_df['elevation_km'].max():.2f} km")
            return track_df
        else:
            logger.warning("grdtrack returned empty result")
    except Exception as e:
        logger.warning(f"Failed to extract topography profile: {e}")

    return None


def _resolve_input_file(description, explicit_path, pattern=None):
    """Return a validated absolute path for the requested input file."""
    if explicit_path:
        candidate = os.path.abspath(os.path.expanduser(explicit_path))
        if os.path.isfile(candidate):
            return candidate
        logger.error("%s '%s' does not exist", description, candidate)
        sys.exit(1)

    if not pattern:
        logger.error("Missing %s path and no pattern provided", description)
        sys.exit(1)

    matches = sorted(glob.glob(pattern))
    if not matches:
        logger.error("Could not find %s matching pattern %s", description, pattern)
        sys.exit(1)
    if len(matches) > 1:
        logger.warning(
            "Multiple %s candidates found, using %s", description, matches[0]
        )
    return os.path.abspath(matches[0])


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
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
        action="store_true",
        dest="verbose",
        help="verbose output",
    )
    parser.add_argument(
        "--scat-file",
        dest="scat_file",
        help="explicit path to the *.scat file (defaults to <event_id>.scat)",
    )
    parser.add_argument(
        "--evt-file",
        dest="evt_file",
        help="explicit path to the QuakeML event file (defaults to <event_id>.*ml glob)",
    )
    parser.add_argument(
        "-o",
        "--output",
        dest="output_pdf",
        help="output PDF path (defaults to <scat_file>.pdf)",
    )
    parser.add_argument(
        "--tile-server",
        action="append",
        dest="tile_servers",
        help="Optional custom tile server URL (can be provided multiple times)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        dest="show",
        help="open the generated figure interactively after export",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
        dest="log_level",
        help="set logging verbosity",
    )
    args = parser.parse_args()
    logging.getLogger().setLevel(getattr(logging, args.log_level.upper(), logging.INFO))
    if not args.event_id and not (args.scat_file and args.evt_file):
        parser.print_help()
        logger.error("Provide --eventid or both --scat-file and --evt-file overrides.")
        sys.exit(255)

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Tile servers to try (in order of preference)
    default_tile_servers = [
        "https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}",
        "https://a.tile.openstreetmap.fr/osmfr/{z}/{x}/{y}.png",
        "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
        "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        "http://tile.stamen.com/terrain/{z}/{x}/{y}.png",
    ]
    tile_servers = args.tile_servers if args.tile_servers else default_tile_servers

    scat_pattern = f"{args.event_id}.scat" if args.event_id else None
    evt_pattern = f"{args.event_id}.*ml" if args.event_id else None
    scat_file = _resolve_input_file("scatter file", args.scat_file, scat_pattern)
    evt_file = _resolve_input_file("event file", args.evt_file, evt_pattern)
    event_label = args.event_id or os.path.splitext(os.path.basename(evt_file))[0]
    logger.info("Using scatter: %s", scat_file)
    logger.info("Using event file: %s", evt_file)

    if args.output_pdf:
        output_pdf = os.path.abspath(os.path.expanduser(args.output_pdf))
    else:
        base, _ = os.path.splitext(scat_file)
        output_pdf = f"{base}.pdf"
    os.makedirs(os.path.dirname(output_pdf), exist_ok=True)
    logger.info("Output PDF: %s", output_pdf)

    # Loc from LocSAT event
    locsat = None
    cat = read_events(evt_file)
    e = cat[0]
    for o in sorted(e.origins, key=_origin_sort_key, reverse=True):
        # FIXME: get the latest or the one with the max #phases
        if "LOCSAT" in o.method_id.id:
            # and o.evaluation_mode == "manual":
            locsat_latitude = o.latitude
            locsat_longitude = o.longitude
            locsat_depth = -1 * o.depth / 1000.0
            locsat_vmodel = (
                o.earth_model_id.id.split("/")[-1] if o.earth_model_id else "N/A"
            )
            locsat = True
            logger.info(
                f"Coord from *LocSAT*, lat: {locsat_latitude:.3f}, "
                f"lon: {locsat_longitude:.3f}, "
                f"depth: {locsat_depth:.1f} km, "
                f"vmodel: {locsat_vmodel}, "
                f"evaluation: {o.evaluation_mode}"
            )
            break

    # Loc from event
    o = e.preferred_origin()
    vmodel = o.earth_model_id.id.split("/")[-1] if o.earth_model_id else "N/A"

    # Extract event info
    event_time_utc = o.time.strftime("%Y-%m-%d %H:%M:%S") if o.time else "N/A"
    # Local time (Europe/Paris)
    try:
        local_tz = ZoneInfo("Europe/Paris")
        event_time_local = (
            o.time.datetime.replace(tzinfo=ZoneInfo("UTC"))
            .astimezone(local_tz)
            .strftime("%Y-%m-%d %H:%M:%S")
            if o.time
            else "N/A"
        )
    except Exception:
        event_time_local = "N/A"
    event_lat = o.latitude
    event_lon = o.longitude
    # Normalize depth sign: other sources (LocSAT, scatter) use negative = depth
    event_depth = -1 * o.depth / 1000.0 if o.depth else 0

    # Quality info
    n_arrivals = len(o.arrivals) if o.arrivals else 0
    n_stations = (
        o.quality.used_station_count
        if o.quality and o.quality.used_station_count
        else 0
    )
    n_p_picks = sum(
        1 for arr in o.arrivals or [] if arr.phase and arr.phase.upper().startswith("P")
    )
    n_s_picks = sum(
        1 for arr in o.arrivals or [] if arr.phase and arr.phase.upper().startswith("S")
    )

    # RMS and azimuthal gap
    rms = o.quality.standard_error if o.quality and o.quality.standard_error else None
    azimuthal_gap = (
        o.quality.azimuthal_gap if o.quality and o.quality.azimuthal_gap else None
    )

    # Get error estimates and additional quality parameters for classification
    erh, erz, _ = get_erh_erz(o)
    dmin = (
        o.quality.minimum_distance * 111.1
        if o.quality and o.quality.minimum_distance
        else None
    )  # deg to km
    dmin_deg = (
        o.quality.minimum_distance if o.quality and o.quality.minimum_distance else None
    )
    dmed_deg = (
        o.quality.median_distance if o.quality and o.quality.median_distance else None
    )
    dmax_deg = (
        o.quality.maximum_distance if o.quality and o.quality.maximum_distance else None
    )
    gap2 = (
        o.quality.secondary_azimuthal_gap
        if o.quality and o.quality.secondary_azimuthal_gap
        else None
    )
    used_station_count = (
        o.quality.used_station_count
        if o.quality and o.quality.used_station_count
        else n_stations
    )

    # Scatter volume from NonLinLoc (stored in origin comments or extra parameters)
    scatvol = None
    for comment in o.comments or []:
        if "scatter" in comment.text.lower() or "scatvol" in comment.text.lower():
            try:
                # Try to extract numeric value
                import re

                match = re.search(r"[\d.]+", comment.text)
                if match:
                    scatvol = float(match.group())
            except Exception:
                pass

    # Hypo71 classification
    hypo71_quality = None
    hypo71_text = None
    if (
        rms is not None
        and erh is not None
        and erz is not None
        and used_station_count
        and azimuthal_gap
        and dmin is not None
    ):
        try:
            hypo71_quality, qs, qd = classify(
                rms, erh, erz, used_station_count, azimuthal_gap, dmin, event_depth
            )
            hypo71_text = get_classification_text(hypo71_quality)
        except Exception as e:
            logger.warning(f"Hypo71 classification failed: {e}")

    # Michele modified classification
    michele_qf = None
    michele_quality = None
    if (
        rms is not None
        and erh is not None
        and erz is not None
        and n_arrivals > 0
        and dmin_deg is not None
        and dmed_deg is not None
        and azimuthal_gap is not None
        and gap2 is not None
        and scatvol is not None
    ):
        try:
            michele_qf, michele_quality = classify_Michele_mod(
                rms,
                erh,
                erz,
                n_arrivals,
                dmin_deg,
                dmed_deg,
                azimuthal_gap,
                gap2,
                scatvol,
            )
        except Exception as e:
            logger.warning(f"Michele classification failed: {e}")

    logger.info(
        f"Coord from *event*, lat: {event_lat:.3f}, "
        f"lon: {event_lon:.3f}, "
        f"depth: {event_depth:.1f} km, "
        f"vmodel: {vmodel}, "
        f"evaluation: {o.evaluation_mode}"
    )

    # loc from scatter data
    df = pd.read_csv(
        scat_file,
        sep="\s+",
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
        f"lon: {max_longitude:.3f}, "
        f"depth: {max_depth:.1f} km"
    )

    # Plot parameters
    transparency_marker = 70

    # Adapt region to the dataset
    df["distance_deg"] = df[["latitude", "longitude"]].apply(
        lambda row: locations2degrees(
            row.iloc[0], row.iloc[1], max_latitude, max_longitude
        ),
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

    # Extract topography profiles for cross-sections
    # Latitude profile (N-S at max_longitude)
    logger.info(f"Extracting latitude topography profile at lon={max_longitude:.3f}")
    topo_lat_profile = _get_topography_profile(
        max_longitude, max_longitude,
        max_latitude - offset, max_latitude + offset
    )
    if topo_lat_profile is not None:
        logger.info(f"Latitude topo profile: {len(topo_lat_profile)} points")
    else:
        logger.warning("Failed to extract latitude topography profile")

    # Longitude profile (E-W at max_latitude)
    logger.info(f"Extracting longitude topography profile at lat={max_latitude:.3f}")
    topo_lon_profile = _get_topography_profile(
        max_longitude - offset, max_longitude + offset,
        max_latitude, max_latitude
    )
    if topo_lon_profile is not None:
        logger.info(f"Longitude topo profile: {len(topo_lon_profile)} points")
    else:
        logger.warning("Failed to extract longitude topography profile")

    # Compute depth range considering both scatter data and topography
    # Convention: negative depth = below sea level, positive = above
    depth_min = df["depth"].min() - depth_offset
    depth_max = 0  # Sea level by default

    # Topography elevation_km: positive = above sea level (mountains), negative = below (ocean)
    # We need the upper bound to include the highest point (max elevation_km)
    if topo_lat_profile is not None:
        topo_max_elev = topo_lat_profile["elevation_km"].max()
        if topo_max_elev > depth_max:
            depth_max = topo_max_elev + 0.5  # Add margin above highest point
    if topo_lon_profile is not None:
        topo_max_elev = topo_lon_profile["elevation_km"].max()
        if topo_max_elev > depth_max:
            depth_max = topo_max_elev + 0.5

    # Also consider if scatter points are above sea level
    if df["depth"].max() > depth_max:
        depth_max = df["depth"].max() + depth_offset

    logger.info(f"Depth range for plots: {depth_min:.1f} to {depth_max:.1f} km")

    # Start plot
    fig = pygmt.Figure()
    pygmt.makecpt(cmap="viridis", series=[df["h2"].min(), df["h2"].max()])
    pygmt.config(FORMAT_GEO_MAP="ddd.xx")
    pygmt.config(MAP_FRAME_TYPE="plain")
    pygmt.config(FONT_TITLE="14p,Helvetica-Bold")
    pygmt.config(FONT_HEADING="12p,Helvetica")
    pygmt.config(FONT_LABEL="10p")
    pygmt.config(PS_MEDIA="A4")
    pygmt.config(PS_PAGE_ORIENTATION="portrait")
    if args.verbose:
        pygmt.config(GMT_VERBOSE="d")

    # A4 page: 21cm x 29.7cm
    # Layout parameters (all in cm)
    page_margin = 1.5  # margin around the page
    panel_size = 8
    gap = 1.5  # gap between panels

    # Title
    event_name = urllib.parse.unquote(event_label).split("/")[-1]
    title_text = f"Event: {event_name}"

    # Start with offset from bottom-left corner for page margins
    # First panel starts at (page_margin, page_margin + panel_size + gap) to leave room for bottom row
    fig.shift_origin(
        xshift=f"{page_margin}c", yshift=f"{page_margin + panel_size + gap}c"
    )

    ##############
    # lat, depth # (top-left)
    ##############
    lat_depth_region = [
        depth_min,
        depth_max,
        max_latitude - offset,
        max_latitude + offset,
    ]
    fig.basemap(
        region=lat_depth_region,
        projection=f"X{panel_size}c/{panel_size}c",
        frame=["afg", "WSne", "x+ldepth (km)", "y+llatitude"],
    )

    # Plot topography profile (latitude vs elevation)
    if topo_lat_profile is not None and len(topo_lat_profile) > 0:
        # Subsample if too many points (PyGMT can struggle with very large polygons)
        topo_df = topo_lat_profile
        if len(topo_df) > 500:
            step = len(topo_df) // 500
            topo_df = topo_df.iloc[::step].copy()
            logger.debug(f"Subsampled lat topo profile to {len(topo_df)} points")

        topo_x = list(topo_df["elevation_km"])
        topo_y = list(topo_df["latitude"])

        # Fill area below topography (underground/rock) - create closed polygon
        # Go along topo profile, then close at the bottom (depth_min side)
        poly_x = topo_x + [depth_min, depth_min, topo_x[0]]
        poly_y = topo_y + [topo_y[-1], topo_y[0], topo_y[0]]
        fig.plot(
            x=poly_x,
            y=poly_y,
            fill="lightgray",
            transparency=50,
            close=True,
        )

        # Draw the topography line on top
        fig.plot(
            x=topo_x,
            y=topo_y,
            pen="1.5p,saddlebrown",
        )

    fig.plot(
        x=df["depth"],
        y=df["latitude"],
        style="c0.15c",
        cmap=True,
        fill=df["h2"],
        transparency=transparency_marker,
    )
    fig.plot(
        x=dfmax["depth"],
        y=dfmax["latitude"],
        style="c0.15c",
        fill="red",
        transparency=transparency_marker,
    )
    fig.plot(x=max_depth, y=max_latitude, style="a0.25c", fill="red", pen="0.5p,black")
    if locsat:
        fig.plot(
            x=locsat_depth,
            y=locsat_latitude,
            style="a0.25c",
            fill="black",
            pen="0.5p,black",
        )

    ############
    # lat, lon # (top-right) - main map with tilemap
    ############
    fig.shift_origin(xshift=f"{panel_size + gap}c")

    # Try tilemap with multiple servers, fallback to basemap if all fail
    tilemap_success = False
    for tile_server in tile_servers:
        try:
            logger.debug(f"Trying tilemap with {tile_server}")
            fig.tilemap(
                region=region,
                projection=f"X{panel_size}c/{panel_size}c",
                source=tile_server,
                lonlat=True,
                frame=["afg", "wSnE"],
            )
            tilemap_success = True
            logger.info(f"Tilemap loaded from {tile_server.split('/')[2]}")
            break
        except Exception as e:
            logger.debug(f"Tilemap failed with {tile_server}: {e}")
            continue

    if not tilemap_success:
        logger.warning("All tilemap servers failed, using basemap with coast")
        fig.basemap(
            region=region,
            projection=f"X{panel_size}c/{panel_size}c",
            frame=["afg", "wSnE"],
        )
        fig.coast(land="lightgray", water="lightblue", shorelines="0.5p,gray50")

    fig.plot(
        x=df["longitude"],
        y=df["latitude"],
        style="c0.15c",
        cmap=True,
        fill=df["h2"],
        transparency=transparency_marker,
    )
    fig.plot(
        x=dfmax["longitude"],
        y=dfmax["latitude"],
        style="c0.15c",
        fill="red",
        transparency=transparency_marker,
    )
    fig.plot(
        x=max_longitude,
        y=max_latitude,
        style="a0.25c",
        fill="red",
        pen="0.5p,black",
    )
    if locsat:
        fig.plot(
            x=locsat_longitude,
            y=locsat_latitude,
            style="a0.25c",
            fill="black",
            pen="0.5p,black",
        )

    # Manual legend for location markers
    legend_dx = region[1] - region[0]
    legend_dy = region[3] - region[2]
    legend_x = region[0] + 0.03 * legend_dx
    legend_y = region[3] - 0.04 * legend_dy
    legend_spacing = 0.045 * legend_dy
    legend_text_offset = 0.05 * legend_dx

    legend_entries = [
        ("a0.25c", "red", "0.5p,black", "NLL centroid"),
        ("c0.15c", "red", "0.3p,black", "NLL max proba"),
    ]
    if locsat:
        legend_entries.append(("a0.25c", "black", "0.5p,black", "LocSAT"))

    for idx, (style, fill, pen, label) in enumerate(legend_entries):
        y_pos = legend_y - idx * legend_spacing
        fig.plot(x=legend_x, y=y_pos, style=style, fill=fill, pen=pen)
        fig.text(
            x=legend_x + legend_text_offset,
            y=y_pos,
            text=label,
            font="9p,Helvetica",
            justify="LM",
            no_clip=True,
        )

    ########
    # Overview map # (bottom-left)
    ########
    fig.shift_origin(xshift=f"-{panel_size + gap}c", yshift=f"-{panel_size + gap}c")

    fig.coast(
        region=[-6, 10, 41, 52],
        projection=f"M{panel_size}c",
        land="lightgray",
        water="white",
        frame=["afg", "WSne"],
        borders="1/0.5p,gray50",
        shorelines="1/0.3p,gray50",
        dcw=["France+gwheat+p0.8p,steelblue"],
    )
    fig.plot(
        x=max_longitude, y=max_latitude, style="a0.4c", fill="red", pen="0.5p,black"
    )

    ##############
    # lon, depth # (bottom-right)
    ##############
    fig.shift_origin(xshift=f"{panel_size + gap}c")

    lon_depth_region = [
        max_longitude - offset,
        max_longitude + offset,
        depth_min,
        depth_max,
    ]
    fig.basemap(
        region=lon_depth_region,
        projection=f"X{panel_size}c/{panel_size}c",
        frame=["afg", "wSnE", "y+ldepth (km)", "x+llongitude"],
    )

    # Plot topography profile (longitude vs elevation)
    if topo_lon_profile is not None and len(topo_lon_profile) > 0:
        # Subsample if too many points
        topo_df = topo_lon_profile
        if len(topo_df) > 500:
            step = len(topo_df) // 500
            topo_df = topo_df.iloc[::step].copy()
            logger.debug(f"Subsampled lon topo profile to {len(topo_df)} points")

        topo_x = list(topo_df["longitude"])
        topo_y = list(topo_df["elevation_km"])

        # Fill area below topography (underground/rock) - create closed polygon
        poly_x = topo_x + [topo_x[-1], topo_x[0], topo_x[0]]
        poly_y = topo_y + [depth_min, depth_min, topo_y[0]]
        fig.plot(
            x=poly_x,
            y=poly_y,
            fill="lightgray",
            transparency=50,
            close=True,
        )

        # Draw the topography line on top
        fig.plot(
            x=topo_x,
            y=topo_y,
            pen="1.5p,saddlebrown",
        )

    fig.plot(
        x=df["longitude"],
        y=df["depth"],
        style="c0.15c",
        cmap=True,
        fill=df["h2"],
        transparency=transparency_marker,
    )
    fig.plot(
        x=dfmax["longitude"],
        y=dfmax["depth"],
        style="c0.15c",
        fill="red",
        transparency=transparency_marker,
    )
    fig.plot(x=max_longitude, y=max_depth, style="a0.25c", fill="red", pen="0.5p,black")
    if locsat:
        fig.plot(
            x=locsat_longitude,
            y=locsat_depth,
            style="a0.25c",
            fill="black",
            pen="0.5p,black",
        )

    # Add title and event info at the top
    fig.shift_origin(
        xshift=f"-{panel_size + gap}c", yshift=f"{panel_size + gap + panel_size + 0.8}c"
    )

    title_width = 2 * panel_size + gap
    title_height = 7.0

    # Add logo in top-left corner
    if os.path.exists(LOGO_PATH):
        fig.image(
            imagefile=LOGO_PATH,
            position="jTL+o0.3c/-0.3c+w5.0c",  # Top-Left, offset, width 5.0cm
        )

    # Main title
    fig.text(
        x=title_width / 2,
        y=6.8,
        text=title_text,
        font="16p,Helvetica-Bold",
        justify="TC",
        no_clip=True,
        region=[0, title_width, 0, title_height],
        projection=f"X{title_width}c/{title_height}c",
    )

    # Origin time below title (UTC and local)
    fig.text(
        x=title_width / 2,
        y=6.1,
        text=f"{event_time_utc} UTC  /  {event_time_local} (local)",
        font="11p,Helvetica",
        justify="TC",
        no_clip=True,
    )

    # Draw info box
    box_left = 1.0
    box_right = title_width - 1.0
    box_bottom = -0.3
    box_top = 5.1
    row_split = box_bottom + (box_top - box_bottom) * 0.42

    # Box outline
    fig.plot(
        x=[box_left, box_right, box_right, box_left, box_left],
        y=[box_bottom, box_bottom, box_top, box_top, box_bottom],
        pen="0.8p,gray30",
        no_clip=True,
    )

    # Horizontal separator line
    fig.plot(
        x=[box_left, box_right],
        y=[row_split, row_split],
        pen="0.5p,gray50",
        no_clip=True,
    )

    # Vertical separator line
    mid_x = (box_left + box_right) / 2
    fig.plot(
        x=[mid_x, mid_x],
        y=[box_bottom, box_top],
        pen="0.5p,gray50",
        no_clip=True,
    )

    # Text block helpers
    title_font = "10p,Helvetica-Bold,gray30"
    value_font = "9p,Helvetica"
    column_left_x = box_left + 0.6
    column_right_x = mid_x + 0.6
    top_section_y = box_top - 0.3
    bottom_section_y = row_split - 0.3

    # Top-left cell: Location
    erh_str = f"{erh:.1f} km" if erh is not None else "N/A"
    erz_str = f"{erz:.1f} km" if erz is not None else "N/A"
    # Display depth or elevation depending on sign
    # Convention: negative = below sea level (depth), positive = above sea level (elevation)
    if event_depth >= 0:
        depth_str = f"Elev: {event_depth:.1f} km (above sea level)"
    else:
        depth_str = f"Depth: {-event_depth:.1f} km (below sea level)"
    location_lines = [
        ("Localization", title_font),
        (f"Lat: {event_lat:.4f}@~\\260@~", value_font),
        (f"Lon: {event_lon:.4f}@~\\260@~", value_font),
        (depth_str, value_font),
        (f"ERH: {erh_str}   ERZ: {erz_str}", value_font),
    ]
    _draw_text_block(
        fig,
        x=column_left_x,
        start_y=top_section_y,
        lines=location_lines,
        line_spacing=0.5,
    )

    # Top-right cell: Quality metrics
    rms_str = f"{rms:.3f} s" if rms else "N/A"
    gap_str = f"{azimuthal_gap:.0f}@~\\260@~" if azimuthal_gap else "N/A"
    dmin_str = f"{dmin:.1f} km" if dmin is not None else "N/A"
    dmed_km = dmed_deg * 111.1 if dmed_deg is not None else None
    dmed_str = f"{dmed_km:.1f} km" if dmed_km is not None else "N/A"
    dmax_km = dmax_deg * 111.1 if dmax_deg is not None else None
    dmax_str = f"{dmax_km:.1f} km" if dmax_km is not None else "N/A"
    gap2_str = f"Gap2: {gap2:.0f}@~\\260@~" if gap2 is not None else "Gap2: N/A"
    quality_lines = [
        ("Quality metrics", title_font),
        (f"Phases: {n_arrivals}  (P: {n_p_picks}, S: {n_s_picks})", value_font),
        (f"Stations: {n_stations}", value_font),
        (f"RMS: {rms_str}", value_font),
        (f"Gaps: {gap_str} / {gap2_str}", value_font),
        (f"dmin: {dmin_str}   dmed: {dmed_str}   dmax: {dmax_str}", value_font),
    ]
    _draw_text_block(
        fig,
        x=column_right_x,
        start_y=top_section_y,
        lines=quality_lines,
        line_spacing=0.5,
    )

    # Bottom-left cell: Model and Evaluation
    eval_mode = o.evaluation_mode if o.evaluation_mode else "N/A"
    model_lines = [
        ("Velocity model / Evaluation mode", title_font),
        (vmodel, value_font),
        (f"{eval_mode}", value_font),
    ]
    _draw_text_block(
        fig,
        x=column_left_x,
        start_y=bottom_section_y,
        lines=model_lines,
        line_spacing=0.5,
    )

    # Bottom-right cell: Classification
    classification_lines = [("Localization quality", title_font)]
    if hypo71_quality:
        hypo71_label = f"Hypo71: {hypo71_quality}"
        if hypo71_text:
            hypo71_label += f" ({hypo71_text})"
        classification_lines.append((hypo71_label, value_font))
    else:
        classification_lines.append(("Hypo71: N/A", value_font))

    if michele_quality:
        classification_lines.append(
            (f"Michele mod.: {michele_quality} (qf={michele_qf:.2f})", value_font)
        )
    else:
        classification_lines.append(("Michele: N/A", value_font))

    _draw_text_block(
        fig,
        x=column_right_x,
        start_y=bottom_section_y,
        lines=classification_lines,
        line_spacing=0.5,
    )

    # Save as PDF
    fig.savefig(output_pdf, crop=False, show=args.show)
