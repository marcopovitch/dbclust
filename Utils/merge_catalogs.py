#!/usr/bin/env python
import argparse
import logging
import os
import sys

import pandas as pd
import yaml
from obspy import Catalog
from obspy import read_events


def yml_read_config(filename):
    with open(filename, "r") as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
    return cfg


if __name__ == "__main__":
    # default logger
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    logger = logging.getLogger("dbclust")
    logger.setLevel(logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--conf",
        default=None,
        dest="configfile",
        help="yaml configuration file.",
        type=str,
    )

    args = parser.parse_args()
    if not args.configfile:
        parser.print_help()
        sys.exit(255)

    cfg = yml_read_config(args.configfile)

    ################ CONFIG ################

    event_merge_info_file = cfg["merge_info_file"]
    main_catalog = cfg["main_catalog"]
    files = cfg["contributing_catalogs"]

    for i in cfg["output"]:
        if os.path.exists(i["filename"]):
            logger.error(f"Output file {i['filename']} already exists !")
            sys.exit(255)

    print(f"Reading merge catalogs information: {event_merge_info_file}")
    df = pd.read_csv(event_merge_info_file)
    df["agencies_list"].fillna("", inplace=True)

    print(f"Reading merged catalog: {main_catalog}")
    mycat = read_events(main_catalog)

    cat = Catalog()
    for f in files:
        print(f"Reading contributing catalogs {f}")
        tmpcat = read_events(f)
        cat.extend(tmpcat)

    print("starting merge ...")
    for index, row in df.iterrows():
        myevent_id = row["event_id"]
        print(myevent_id)
        myevent = [e for e in mycat.events if e.resource_id.id == myevent_id].pop()
        agencies_event_list = row["agencies_list"].split(" ")

        events = [e for e in cat.events if e.resource_id.id in agencies_event_list]
        origins = [e.preferred_origin() for e in events]
        magnitudes = [
            e.preferred_magnitude() for e in events if e.preferred_magnitude()
        ]
        picks = [p for e in events for p in e.picks]

        myevent.origins.extend(origins)
        myevent.magnitudes.extend(magnitudes)
        myevent.picks.extend(picks)

    for i in cfg["output"]:
        print(f"Writing {i['name']}")
        mycat.write(i["filename"], format=i["format"])
