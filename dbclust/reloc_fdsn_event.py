#!/usr/bin/env python
import argparse
import logging
import sys
import traceback
import urllib.parse
from dataclasses import asdict
from shutil import copyfile

import yaml
from config import DBClustConfig
from icecream import ic
from localization import NllLoc
from localization import reloc_fdsn_event
from localization import show_bulletin
from localization import show_event

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
        "-f",
        "--fdsn-profile",
        default=None,
        dest="fdsn_profile",
        help="fdsn profile",
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
    if not args.profile_conf_file or not args.event_id:
        parser.print_help()
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

    if args.fdsn_profile:
        cfg.station.fdsnws.set_url_from_service_name(args.fdsn_profile)
        cfg.station.info_sta = cfg.station.fdsnws.get_url()

    output_format = "QUAKEML"

    with MyTemporaryDirectory(dir=cfg.file.tmp_path, delete=True) as tmp_path:
        locator = NllLoc(
            cfg.nll.nlloc_bin,
            cfg.nll.scat2latlon_bin,
            cfg.nll.time_path,
            # cfg.nll.template_path,
            # "../nll_template/nll_haslach-0.2_template.conf",
            # "../nll_template/nll_rittershoffen_template.conf",
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
            relabel_pick_zone=enable_relabel,  # to be added in the configuration file
            cleanup_pick_zone=True,  # to be added in the configuration file
            #
            log_level=numeric_level,
        )

        try:
            cat = reloc_fdsn_event(
                locator, args.event_id, cfg.station.fdsnws.get_url(), args.zone_name
            )
        except Exception as e:
            logger.error(f"Error with {args.event_id}: {e}")
            traceback.print_exc()
            sys.exit()

        for e in cat:
            show_event(e, "****", header=True)
            show_bulletin(
                e,
                zones=cfg.zones,
                plot=args.enable_plot,
            )

        file_extension = output_format.lower()
        cat.write(
            f"{urllib.parse.quote(args.event_id, safe='')}.{file_extension}",
            format=output_format,
        )
        if locator.scat_file:
            try:
                copyfile(
                    locator.scat_file,
                    f"{urllib.parse.quote(args.event_id, safe='')}.scat",
                )
            except Exception as e:
                logger.error("Can't get nll scat file (%s)", e)
