#!/usr/bin/env python
import argparse
import logging
import os
import sys

import pandas as pd
from obspy import read_events


def export_picks_to_phasenet_format(
    event, origin, probability=1, evaluation=None, method=None, agency=None
):
    # seedid,phasename,time,probability
    # 1K.OFAS0.00.EH.D,P,2023-02-13T18:30:58.558999Z,0.366400
    lines = []
    for arrival in origin.arrivals:
        if arrival.time_weight:
            # if arrival.time_weight and arrival.time_residual:
            pick = next(
                (p for p in event.picks if p.resource_id == arrival.pick_id), None
            )
            if pick:
                if arrival.phase != pick.phase_hint:
                    logger.warning(
                        f"[{event.resource_id.id}] {pick.waveform_id.get_seed_string()}: "
                        f"phase mismatch between arrival ({arrival.phase}) and pick ({pick.phase_hint}). "
                        "Using arrival phase."
                    )
                line = {
                    "station_id": pick.waveform_id.get_seed_string().rstrip(".."),
                    "phase_type": arrival.phase,
                    "phase_time": pick.time,
                    "phase_score": probability,
                    "phase_evaluation": pick.evaluation_mode,
                    "phase_method": pick.method_id,
                    # "eventid": event.resource_id.id.split("/")[-1],
                    "event_id": event.resource_id.id,
                }
                # override
                if evaluation:
                    line["phase_evaluation"] = evaluation
                if method:
                    line["phase_method"] = method
                if agency:
                    line["agency"] = agency
                lines.append(line)
    return lines


if __name__ == "__main__":
    # default logger
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    logger = logging.getLogger("qml2dbclust")
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
        "-p",
        "--probability",
        default=1,
        dest="probability",
        help="probability",
        type=float,
    )
    parser.add_argument(
        "-e",
        "--evaluation",
        default=None,
        dest="evaluation",
        help="override pick evaluation mode (automatic|manual)",
        type=str,
    )
    parser.add_argument(
        "-m",
        "--method",
        default=None,
        dest="method",
        help="override pick method_id (AIC|PHASENET|...)",
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
        "--from",
        default=None,
        dest="from_time",
        help="select time window start",
        type=str,
    )
    parser.add_argument(
        "--to",
        default=None,
        dest="to_time",
        help="select time window end",
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

    rqt = []
    if args.from_time:
        rqt_from = f"time >= {args.from_time}"
        rqt.append(rqt_from)

    if args.to_time:
        rqt_to = f"time < {args.to_time}"
        rqt.append(rqt_to)

    if len(rqt):
        logger.info(f"Filtering in time {rqt}")
        cat = cat.filter(*rqt)

    df = pd.DataFrame(
        columns=[
            "station_id",
            "channel",
            "phase_type",
            "phase_time",
            "phase_score",
            "phase_evaluation",
            "phase_method",
            "event_id",
            "agency",
        ]
    )
    for event in cat.events:
        origin = event.preferred_origin()
        picks_list = export_picks_to_phasenet_format(
            event, origin, probability=args.probability, agency=args.agency
        )
        df = pd.concat([df, pd.DataFrame(picks_list)], ignore_index=True)

    df.to_csv(args.outputfile, index=False)
