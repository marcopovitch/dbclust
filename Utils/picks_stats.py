#!/usr/bin/env python
import argparse
import json
import logging
import os
import sys

from obspy import Catalog
from obspy import read
from obspy import read_events
from obspy import read_inventory
from obspy import UTCDateTime


def get_pick_proba_info(outfile, cat, show_station_name=False):
    with open(outfile, "w") as f:
        # csv header
        if show_station_name:
            f.write("station_id,phase_name,probability,source\n")
        else:
            f.write("phase_name,probability,source\n")

        for event in cat:
            origin = event.preferred_origin()
            for arrival in origin.arrivals:
                if hasattr(arrival, "time_weight") and arrival.time_weight == 0:
                    continue
                pick = next(
                    (p for p in event.picks if p.resource_id == arrival.pick_id), None
                )
                if not pick:
                    continue
                for comment in pick.comments:
                    try:
                        info = json.loads(comment.text)
                        if "probability" in info.keys():
                            if show_station_name:
                                f.write(
                                    "%s,%s,%.2f,%s\n"
                                    % (
                                        pick.waveform_id.get_seed_string()[:-1],
                                        pick.phase_hint,
                                        info["probability"]["value"],
                                        info["probability"]["name"],
                                    )
                                )
                            else:
                                f.write(
                                    "%s,%.2f,%s\n"
                                    % (
                                        pick.phase_hint,
                                        info["probability"]["value"],
                                        info["probability"]["name"],
                                    )
                                )
                    except:
                        pass


def get_event_ids_info(outfile, cat):
    with open(outfile, "w") as f:
        f.write("time,event_id,nb_agencies,agencies_list\n")
        for event in sorted(cat.events, key=lambda e: e.preferred_origin().time):
            origin = event.preferred_origin()
            f.write(f"{origin.time},")
            ids = []
            for comment in event.comments:
                try:
                    info = json.loads(comment.text)
                except:
                    continue
                if "event_ids" in info.keys():
                    ids.extend(info["event_ids"])
            all_ids = " ".join(ids)
            f.write(f"{event.resource_id.id},{len(ids)},{all_ids}\n")


def get_distance_km_info(outfile, cat):
    with open(outfile, "w") as f:
        f.write("time,event_id,dist_km,event_type\n")
        for event in sorted(cat.events, key=lambda e: e.preferred_origin().time):
            origin = event.preferred_origin()
            f.write(f"{origin.time},")
            dist_km = None
            for comment in event.comments:
                try:
                    info = json.loads(comment.text)
                except:
                    continue
                if "preloc_distance_km" in info.keys():
                    dist_km = info["preloc_distance_km"]
                    break
            f.write(f"{event.resource_id.id},{dist_km},{event.event_type}\n")


if __name__ == "__main__":
    # default logger
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    logger = logging.getLogger("dbclust")
    logger.setLevel(logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input",
        default=None,
        dest="inputfile",
        help="qml input file",
        type=str,
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        dest="outputfile",
        help="output file",
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

    if not args.inputfile or not args.outputfile:
        parser.print_help()
        sys.exit(255)

    if os.path.isfile(args.outputfile):
        logger.error(f"outputfile {args.outputfile} already exists !")
        sys.exit(255)

    numeric_level = getattr(logging, args.loglevel.upper(), None)
    if not numeric_level:
        logger.error("Invalid loglevel '%s' !", args.loglevel.upper())
        logger.error("loglevel should be: debug,warning,info,error.")
        sys.exit(255)
    logger.setLevel(numeric_level)

    cat = read_events(args.inputfile)
    get_pick_proba_info(args.outputfile, cat, show_station_name=True)
    get_event_ids_info("events_merge_info.csv", cat)
    get_distance_km_info("preloc_distance.csv", cat)
