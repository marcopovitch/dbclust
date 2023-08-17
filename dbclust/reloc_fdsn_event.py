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
        dest="profile_name",
        help="profile name to use",
        type=str,
    )
    parser.add_argument(
        "-e",
        "--eventid",
        default=None,
        dest="event_id",
        help="event id",
        type=str,
    )

    args = parser.parse_args()
    if not args.profile_conf_file or not args.event_id:
        parser.print_help()
        sys.exit()

    conf = load_config(args.profile_conf_file)
    if not conf:
        sys.exit()

    nll_conf = conf["nll"]
    parameters_conf = conf["parameters"]
    fdsnws_conf = conf["fdsnws"]
    profile_conf = conf["profile"]
    if hasattr(args, "profile_name") and args.profile_name:
        default_profile = args.profile_name
    else:
        default_profile = conf["default_profile"]
    logger.info(f"Using {default_profile} profile")
    quakeml_conf = conf["quakeml"]

    verbose = conf["verbose"]
    tmpdir = conf["tmpdir"]
    output_format = conf["output"]["format"]

    # parameters
    double_pass = parameters_conf["double_pass"]
    force_uncertainty = parameters_conf["force_uncertainty"]
    P_uncertainty = parameters_conf["P_uncertainty"]
    S_uncertainty = parameters_conf["S_uncertainty"]
    P_time_residual_threshold = parameters_conf["P_time_residual_threshold"]
    S_time_residual_threshold = parameters_conf["S_time_residual_threshold"]
    dist_km_cutoff  = parameters_conf["dist_km_cutoff"]
    # use_deactivated_arrivals = parameters_conf["use_deactivated_arrivals"]

    # NonLinLoc
    nlloc_bin = nll_conf["bin"]
    nlloc_times_path = nll_conf["times_path"]
    nlloc_template_path = nll_conf["template_path"]
    template = None
    for p in profile_conf:
        if p["name"] == default_profile:
            template = p["template"]
    if not template:
        logger.error(f"profile {default_profile} does not exist !")
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
        "model_id": default_profile,
    }

    fdsnws = fdsnws_conf["event"]

    if tmpdir is None:
        tmpdir = tempfile.TemporaryDirectory()

    locator = NllLoc(
        nlloc_bin,
        nlloc_times_path,
        nlloc_template,
        nll_min_phase=nlloc_min_phase,
        #
        tmpdir=tmpdir,
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
    )

    cat = reloc_fdsn_event(locator, args.event_id, fdsnws)
    if tmpdir is None:
        tmpdir.cleanup()

    for e in cat:
        show_event(e, "****", header=True)

    file_extension = output_format.lower()
    cat.write(
        f"{urllib.parse.quote(args.event_id, safe='')}.{file_extension}",
        format=output_format,
    )
