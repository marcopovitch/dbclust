#!/usr/bin/env python
import argparse
import glob
import logging
import os
import sys
import traceback
import urllib.parse
from dataclasses import asdict
from shutil import copyfile

from icecream import ic
from obspy import read_events

from dbclust.config import DBClustConfig
from dbclust.localization import NllLoc
from dbclust.localization import reloc_fdsn_event
from dbclust.localization import show_bulletin
from dbclust.localization import show_event
from dbclust.runner import MyTemporaryDirectory

# Default logger
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger("reloc_fdsn_event")
logger.setLevel(logging.DEBUG)

def only_one(inputs):
    """
    Check if exactly one of the provided arguments is not None.

    Args:
        inputs (list): A list of input values to check.

    Returns:
        bool: True if exactly one input is not None, False otherwise.
    """
    non_none_count = sum(1 for item in inputs if item is not None)
    return non_none_count == 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--conf",
        default=None,
        dest="profile_conf_file",
        help="dbclust configuration file.",
        type=str,
    )
    parser.add_argument(
        "-d",
        "--dist-km-cutoff",
        default=None,
        dest="dist_km_cutoff",
        help="station cut off distance in km",
        type=float,
    )
    parser.add_argument(
        "-e",
        "--eventid",
        default=None,
        dest="event_id",
        help="event id to fetch from FDSN and to relocate",
        type=str,
    )
    parser.add_argument(
        "--event",
        default=None,
        dest="event",
        help="event file in QuakeML format",
        type=str,
    )
    parser.add_argument(
        "--dir",
        default=None,
        dest="dir",
        help="Directory containing QuakeML files to relocate",
        type=str,
    )
    parser.add_argument(
        "-f",
        "--fdsn-event-profile",
        default=None,
        dest="fdsn_event_profile",
        help="fdsn event profile name to use (see conf.yml file)",
        type=str,
    )
    parser.add_argument(
        "-l",
        "--loglevel",
        default="INFO",
        dest="loglevel",
        help="set loglevel (debug, warning, info, error)",
        type=str,
    )
    parser.add_argument(
        "-u",
        "--use-deactivated-arrivals",
        default=False,
        dest="use_deactivated_arrivals",
        help="force deactivated arrivals use",
        action="store_true",
    )
    parser.add_argument(
        "-t",
        "--min-score-threshold-pick-zone",
        default=None,
        dest="min_score_threshold_pick_zone",
        help="min score threshold pick zone",
        type=float,
    )
    parser.add_argument(
        "-r",
        "--relabel",
        default=False,
        dest="relabel",
        help="enable relabeling",
        action="store_true",
    )
    parser.add_argument(
        "-s",
        "--scat",
        default=False,
        dest="scat",
        help="get xyz scat file",
        action="store_true",
    )
    parser.add_argument(
        "--plot",
        default=False,
        dest="enable_plot",
        help="enable plot",
        action="store_true",
    )
    parser.add_argument(
        "--force-uncertainty",
        default=False,
        dest="force_uncertainty",
        help="force phase uncertainty (see conf.yml file)",
        action="store_true",
    )
    parser.add_argument(
        "--single-pass",
        default=False,
        dest="single_pass",
        help="Nonlinloc single or double pass",
        action="store_true",
    )
    parser.add_argument(
        "-z",
        "--zone",
        default=None,
        dest="zone_name",
        help="force zone name to use (default is autodetect from event lat/lon)",
        type=str,
    )
    parser.add_argument(
        "-o",
        "--output-format",
        default="QUAKEML",
        dest="output_format",
        help="output format for the event file",
        type=str,
    )

    args = parser.parse_args()
    if not args.profile_conf_file:
        logger.error("Please provide a profile configuration file")
        sys.exit()

    numeric_level = getattr(logging, args.loglevel.upper(), None)
    if not numeric_level:
        logger.error("Invalid loglevel '%s' !", args.loglevel.upper())
        logger.error("loglevel should be: debug, warning, info, error.")
        sys.exit(255)
    else:
        logger.setLevel(numeric_level)

    cfg = DBClustConfig(args.profile_conf_file)

    # Update configuration
    if args.dist_km_cutoff:
        cfg.relocation.dist_km_cutoff = args.dist_km_cutoff

    if args.use_deactivated_arrivals:
        cfg.relocation.use_deactivated_arrivals = args.use_deactivated_arrivals

    if args.force_uncertainty:
        cfg.relocation.force_uncertainty = args.force_uncertainty

    if args.single_pass:
        cfg.relocation.double_pass = not args.single_pass

    if args.scat:
        cfg.nll.enable_scatter = args.scat

    if not args.zone_name:
        cfg.quakeml.model_id = None

    enable_relabel = args.relabel

    if args.min_score_threshold_pick_zone:
        cfg.relocation.min_score_threshold_pick_zone = args.min_score_threshold_pick_zone

    # Check event source
    if not any([args.event_id, args.event, args.dir]):
        logger.error("Please provide an event source")
        sys.exit()

    if not only_one([args.event_id, args.event, args.dir]):
        logger.error("Please provide only one event source")
        sys.exit()

    if args.fdsn_event_profile:
        cfg.fdsnws_event.set_url_from_service_name(args.fdsn_event_profile)
        ic(cfg.fdsnws_event.get_url())
    elif args.dir and not os.path.exists(args.dir):
        logger.error("Please provide a valid directory")
        sys.exit()
    elif args.event and not os.path.exists(args.event):
        logger.error("Please provide a valid event file")
        sys.exit()


    with MyTemporaryDirectory(dir=cfg.file.tmp_path, delete=True) as tmp_path:
        locator = NllLoc(
            cfg.nll.nlloc_bin,
            cfg.nll.scat2latlon_bin,
            cfg.nll.time_path,
            tmpdir=tmp_path,
            double_pass=cfg.relocation.double_pass,
            gap_dist_max_km=cfg.relocation.gap_dist_max_km,
            P_time_residual_threshold=cfg.relocation.P_time_residual_threshold,
            S_time_residual_threshold=cfg.relocation.S_time_residual_threshold,
            dist_km_cutoff=cfg.relocation.dist_km_cutoff,
            use_deactivated_arrivals=cfg.relocation.use_deactivated_arrivals,
            keep_manual_picks=cfg.relocation.keep_manual_picks,
            nll_min_phase=cfg.nll.min_phase,
            min_station_with_P_and_S=cfg.cluster.min_station_with_P_and_S,
            quakeml_settings=asdict(cfg.quakeml),
            nll_verbose=cfg.nll.verbose,
            keep_scat=cfg.nll.enable_scatter,
            zones=cfg.zones,
            force_zone_name=args.zone_name,
            min_score_threshold_pick_zone=cfg.relocation.min_score_threshold_pick_zone,
            enable_relabel_pick_zone=enable_relabel,
            enable_cleanup_pick_zone=True,
            log_level=numeric_level,
        )

        def process_file(f):
            cat = read_events(f)
            if len(cat) == 0:
                logging.error(f"No event found in QuakeML file {f}")
                return
            elif len(cat) > 1:
                logging.error(f"More than one event found in QuakeML file {f}")
                return

            event = cat[0]
            o = event.preferred_origin() or event.origins[0]
            zone, _ = cfg.zones.find_zone(o.latitude, o.longitude)
            ic(zone["name"])

            cat = reloc_fdsn_event(locator, event=event, zone_name=zone["name"])
            if len(cat) == 0:
                logging.error("No relocated event found")
                return
            elif len(cat) > 1:
                logging.error("More than one relocated event found")
                return

            # merge relocated event with original event
            e = cat[0]
            e.origins.extend(event.origins)
            e.origins.sort(key=lambda x: x.creation_info.creation_time, reverse=True)
            e.picks.extend(event.picks)
            e.amplitudes.extend(event.amplitudes)
            e.magnitudes.extend(event.magnitudes)

            # show relocated event
            show_event(e, "****", header=True)
            show_bulletin(e, zones=cfg.zones, plot=args.enable_plot)

            event_id = cat[0].resource_id.id.split("/")[-1]
            file_extension = args.output_format.lower()
            cat.write(
                f"{urllib.parse.quote(event_id, safe='')}.{file_extension}",
                format=args.output_format,
            )

            if locator.scat_file:
                try:
                    copyfile(locator.scat_file, f"{urllib.parse.quote(event_id, safe='')}.scat")
                except Exception as e:
                    logging.error("Can't get nll scat file (%s)", e)

        if args.event:
            process_file(args.event)
        elif args.dir:
            for f in glob.glob(f"{args.dir}/*.qml"):
                process_file(f)
        else:
            # fetch directly from FDSNWS url
            filename = os.path.join(tmp_path, args.event_id + ".xml")
            options="includeallorigins=true&includeallmagnitudes=true&includearrivals=true&nodata=404"
            url = cfg.fdsnws_event.get_url() + f"/query?{options}&eventid={args.event_id}"
            try:
                urllib.request.urlretrieve(url, filename)
            except urllib.error.HTTPError as e:
                # get 404 error
                if e.code == 404:
                    logging.error(f"Event: {args.event_id} not found in FDSNWS")
                    #logging.error(f"URL: {url}")
                else:
                    logging.error(f"Error: {e}")
                sys.exit()
            except Exception as e:
                logging.error(f"Error: {e}")
                traceback.print_exc()
                sys.exit()
            process_file(filename)
