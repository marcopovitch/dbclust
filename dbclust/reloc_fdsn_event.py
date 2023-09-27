#!/usr/bin/env python
import sys
import os
import tempfile
import logging
import urllib.parse
from obspy import read_events
from localization import reloc_fdsn_event, NllLoc, show_event
import argparse
import yaml
from shutil import copyfile

# default logger
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger("reloc_fdsn_event")
logger.setLevel(logging.DEBUG)


def load_config(conf_file):
    with open(conf_file, "r") as stream:
        try:
            conf = yaml.safe_load(stream)
        except yaml.YAMLError as e:
            logger.error(e)
            conf = None
    return conf


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
        "-p",
        "--profile",
        default=None,
        dest="velocity_profile_name",
        help="velocity profile name to use",
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
        "-d",
        "--dist-km-cutoff",
        default=None,
        dest="dist_km_cutoff",
        help="cut off distance in km",
        type=float,
    )
    parser.add_argument("-s", "--scat", help="get xyz scat file", action="store_true")
    parser.add_argument(
        "--single-pass",
        default=False,
        dest="single_pass",
        help="Nonlinloc single or double pass",
        action="store_true",
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
        "-l",
        "--loglevel",
        default="INFO",
        dest="loglevel",
        help="loglevel (debug,warning,info,error)",
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

    conf = load_config(args.profile_conf_file)
    if not conf:
        sys.exit()

    nll_conf = conf["nll"]
    parameters_conf = conf["parameters"]
    fdsnws_conf = conf["fdsnws"]
    velocity_profile_conf = conf["velocity_profile"]
    if hasattr(args, "velocity_profile_name") and args.velocity_profile_name:
        default_velocity_profile = args.velocity_profile_name
    else:
        default_velocity_profile = conf["default_velocity_profile"]
    logger.info(f"Using {default_velocity_profile} profile")
    quakeml_conf = conf["quakeml"]

    # force fdsn ws
    fdsnws_cfg = conf["fdsnws"]
    if not args.fdsn_profile:
        default_url_mapping = fdsnws_cfg["default_url_mapping"]
    else:
        default_url_mapping = args.fdsn_profile
    fdsn_debug = fdsnws_cfg["fdsn_debug"]
    url_mapping = fdsnws_cfg["url_mapping"]
    if default_url_mapping not in fdsnws_cfg["url_mapping"]:
        logger.error("unknown fdsn profile '%s'. Exiting !", default_url_mapping)
        sys.exit(255)
    ws_event_url = url_mapping[default_url_mapping]["ws_event_url"]

    verbose = conf["verbose"]
    tmpdir = conf["tmpdir"]
    output_format = conf["output"]["format"]

    # parameters
    if args.single_pass:
        double_pass = False
    else:
        double_pass = parameters_conf["double_pass"]
    force_uncertainty = parameters_conf["force_uncertainty"]
    P_uncertainty = parameters_conf["P_uncertainty"]
    S_uncertainty = parameters_conf["S_uncertainty"]
    P_time_residual_threshold = parameters_conf["P_time_residual_threshold"]
    S_time_residual_threshold = parameters_conf["S_time_residual_threshold"]
    if not args.dist_km_cutoff:
        dist_km_cutoff = parameters_conf["dist_km_cutoff"]
    else:
        dist_km_cutoff = args.dist_km_cutoff
    # use_deactivated_arrivals = parameters_conf["use_deactivated_arrivals"]

    # NonLinLoc
    nlloc_bin = nll_conf["bin"]
    scat2latlon_bin = nll_conf["scat2latlon_bin"]
    nlloc_times_path = nll_conf["times_path"]
    nlloc_template_path = nll_conf["template_path"]
    template = None
    for p in velocity_profile_conf:
        if p["name"] == default_velocity_profile:
            template = p["template"]
    if not template:
        logger.error(f"profile {default_velocity_profile} does not exist !")
        sys.exit()
    nlloc_template = os.path.join(nlloc_template_path, template)
    nlloc_verbose = nll_conf["verbose"]
    nlloc_min_phase = nll_conf["min_phase"]

    # quakeml
    quakeml_settings = {
        "agency_id": quakeml_conf["agency_id"],
        "author": quakeml_conf["author"],
        "evaluation_mode": quakeml_conf["evaluation_mode"],
        "method_id": quakeml_conf["method_id"],
        "model_id": default_velocity_profile,
    }

    with tempfile.TemporaryDirectory(dir=tmpdir) as tmp_path:
        locator = NllLoc(
            nlloc_bin,
            scat2latlon_bin,
            nlloc_times_path,
            nlloc_template,
            nll_min_phase=nlloc_min_phase,
            #
            tmpdir=tmp_path,
            #
            force_uncertainty=force_uncertainty,
            P_uncertainty=P_uncertainty,
            S_uncertainty=S_uncertainty,
            #
            double_pass=double_pass,
            #
            dist_km_cutoff=dist_km_cutoff,
            P_time_residual_threshold=P_time_residual_threshold,
            S_time_residual_threshold=S_time_residual_threshold,
            #
            quakeml_settings=quakeml_settings,
            nll_verbose=nlloc_verbose,
            keep_scat=args.scat,
            #
            log_level=numeric_level,
        )

        cat = reloc_fdsn_event(locator, args.event_id, ws_event_url)

        for e in cat:
            show_event(e, "****", header=True)

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
