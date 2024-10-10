#!/usr/bin/env python
import argparse
import logging
import sys
import traceback
import urllib.parse
from dataclasses import asdict
from shutil import copyfile

from config import DBClustConfig
from icecream import ic
from localization import NllLoc
from localization import reloc_fdsn_event
from localization import show_bulletin
from localization import show_event
from obspy import read_events

from dbclust import MyTemporaryDirectory

# default logger
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger("reloc_fdsn_event")
logger.setLevel(logging.DEBUG)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--conf",
        default=None,
        dest="profile_conf_file",
        help="profile configuration file.",
        type=str,
    )
    parser.add_argument(
        "-d",
        "--dist-km-cutoff",
        default=None,
        dest="dist_km_cutoff",
        help="cut off distance in km",
        type=float,
    )
    parser.add_argument(
        "-e",
        "--eventid",
        default=None,
        dest="event_id",
        help="event id",
        type=str,
    )
    parser.add_argument(
        "--event",
        default=None,
        dest="event",
        help="event in QuakeML format",
        type=str,
    )
    parser.add_argument(
        "-f",
        "--fdsn-event-profile",
        default=None,
        dest="fdsn_event_profile",
        help="fdsn event profile",
        type=str,
    )
    parser.add_argument(
        "-l",
        "--loglevel",
        default="INFO",
        dest="loglevel",
        help="loglevel (debug,warning,info,error)",
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
    parser.add_argument("-s", "--scat", help="get xyz scat file", action="store_true")
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
        help="force zone name to use",
        type=str,
    )

    args = parser.parse_args()
    if not args.profile_conf_file:
        logger.error("Please provide a profile configuration file")
        sys.exit()

    numeric_level = getattr(logging, args.loglevel.upper(), None)
    if not numeric_level:
        logger.error("Invalid loglevel '%s' !", args.loglevel.upper())
        logger.error("loglevel should be: debug,warning,info,error.")
        sys.exit(255)
    else:
        logger.setLevel(numeric_level)

    cfg = DBClustConfig(args.profile_conf_file)

    # update configuration
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

    if args.relabel:
        enable_relabel = True
    else:
        enable_relabel = False

    if args.min_score_threshold_pick_zone:
        cfg.relocation.min_score_threshold_pick_zone = args.min_score_threshold_pick_zone

    if args.event_id and args.event:
        logger.error("Please provide only one event source")
        sys.exit()

    if args.fdsn_event_profile:
        cfg.fdsnws_event.set_url_from_service_name(args.fdsn_event_profile)
        ic(cfg.fdsnws_event.get_url())

    if args.event:
        cat = read_events(args.event)
        if len(cat) == 0:
            logger.error("No event found in QuakeML file")
            sys.exit()
        elif len(cat) > 1:
            logger.error("More than one event found in QuakeML file")
            sys.exit()
        event = cat.events[0]

    output_format = "QUAKEML"

    with MyTemporaryDirectory(dir=cfg.file.tmp_path, delete=True) as tmp_path:
        locator = NllLoc(
            cfg.nll.nlloc_bin,
            cfg.nll.scat2latlon_bin,
            cfg.nll.time_path,
            #
            tmpdir=tmp_path,
            double_pass=cfg.relocation.double_pass,
            #
            P_time_residual_threshold=cfg.relocation.P_time_residual_threshold,
            S_time_residual_threshold=cfg.relocation.S_time_residual_threshold,
            dist_km_cutoff=cfg.relocation.dist_km_cutoff,
            use_deactivated_arrivals=cfg.relocation.use_deactivated_arrivals,  # to be added in the configuration file
            #
            keep_manual_picks=cfg.relocation.keep_manual_picks,
            nll_min_phase=cfg.nll.min_phase,
            min_station_with_P_and_S=cfg.cluster.min_station_with_P_and_S,
            #
            quakeml_settings=asdict(cfg.quakeml),
            nll_verbose=cfg.nll.verbose,
            keep_scat=cfg.nll.enable_scatter,
            #
            zones=cfg.zones,
            force_zone_name=args.zone_name,
            min_score_threshold_pick_zone=cfg.relocation.min_score_threshold_pick_zone,
            enable_relabel_pick_zone=enable_relabel,
            enable_cleanup_pick_zone=True,
            #
            log_level=numeric_level,
        )

        try:
            if args.event:
                cat = reloc_fdsn_event(locator, event=event, zone_name=args.zone_name)
            else:
                cat = reloc_fdsn_event(
                    locator,
                    args.event_id,
                    cfg.fdsnws_event.get_url(),
                    zone_name=args.zone_name,
                )
        except Exception as e:
            logger.error(f"Error: {e}")
            traceback.print_exc()
            sys.exit()

        event_id = cat[0].resource_id.id.split("/")[-1]

        for e in cat:
            show_event(e, "****", header=True)
            show_bulletin(
                e,
                zones=cfg.zones,
                plot=args.enable_plot,
            )

        file_extension = output_format.lower()
        cat.write(
            f"{urllib.parse.quote(event_id, safe='')}.{file_extension}",
            format=output_format,
        )
        if locator.scat_file:
            try:
                copyfile(
                    locator.scat_file,
                    f"{urllib.parse.quote(event_id, safe='')}.scat",
                )
            except Exception as e:
                logger.error("Can't get nll scat file (%s)", e)
