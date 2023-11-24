#!/usr/bin/env python
import sys
import os
import logging
import argparse
import pandas as pd
from obspy import read_events


def export_picks_to_phasenet_format(event, origin, agency=None):
    # seedid,phasename,time,probability
    # 1K.OFAS0.00.EH.D,P,2023-02-13T18:30:58.558999Z,0.366400
    lines = []
    for arrival in origin.arrivals:
        if arrival.time_weight and arrival.time_residual:
            pick = next(
                (p for p in event.picks if p.resource_id == arrival.pick_id), None
            )
            if pick:
                line = {
                    "seedid": pick.waveform_id.get_seed_string(),
                    "phasename": pick.phase_hint,
                    "time": pick.time,
                    "probability": 1,
                    "eventid": event.resource_id,
                }
                if agency:
                    line["agency"] = agency
                lines.append(line)
    return lines


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
        help="input file",
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
        "-a",
        "--agency",
        default=None,
        dest="agency",
        help="agency name",
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
    df = pd.DataFrame(columns=["seedid", "phasename", "time", "probability", "eventid", "agency"])
    for event in cat.events:
        origin = event.preferred_origin()
        picks_list = export_picks_to_phasenet_format(event, origin, agency=args.agency)
        df = pd.concat([df, pd.DataFrame(picks_list)], ignore_index=True)

    df.to_csv(args.outputfile, index=False)
