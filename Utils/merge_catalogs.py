#!/usr/bin/env python
import argparse
import logging
import os
import sys
import warnings

import pandas as pd
import yaml
from icecream import ic
from obspy import Catalog
from obspy import read_events


def yml_read_config(filename: str) -> dict:
    """
    Reads a YAML configuration file and returns its contents as a dictionary.

    :param filename: Path to the YAML configuration file.
    :return: Parsed configuration dictionary.
    """
    with open(filename, "r") as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
    return cfg


if __name__ == "__main__":
    # Setup default logger
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger("merge_catalog")

    # Silence invalid QuakeML URI warnings emitted by ObsPy writes
    warnings.filterwarnings(
        "ignore",
        message=".*is not a valid QuakeML URI.*",
        category=UserWarning,
    )

    # Argument parser
    parser = argparse.ArgumentParser(description="Merge seismic catalogs based on merge information.")
    parser.add_argument(
        "-c",
        "--conf",
        required=True,
        dest="configfile",
        help="YAML configuration file path.",
        type=str,
    )

    args = parser.parse_args()

    # Get configuration
    try:
        cfg = yml_read_config(args.configfile)
    except Exception as e:
        logger.error(f"Error reading configuration file: {e}")
        sys.exit(1)

    event_merge_info_file = cfg.get("merge_info_file")
    main_catalog = cfg.get("main_catalog")
    files = cfg.get("contributing_catalogs", [])

    if not event_merge_info_file or not main_catalog or not files:
        logger.error("Missing required configuration parameters. Check your YAML file.")
        sys.exit(1)

    # Check if output files already exist
    for i in cfg["output"]:
        if os.path.exists(i["filename"]):
            logger.error(f"Output file {i['filename']} already exists!")
            sys.exit(1)

    # Read merge catalogs information
    logger.info(f"Reading merge catalogs information from: {event_merge_info_file}")
    try:
        df = pd.read_csv(event_merge_info_file)
        df.fillna({"agencies_list": ""}, inplace=True)
    except Exception as e:
        logger.error(f"Error reading merge info file: {e}")
        sys.exit(1)

    # Read main catalog
    logger.info(f"Reading main catalog: {main_catalog}")
    try:
        mycat = read_events(main_catalog)
    except Exception as e:
        logger.error(f"Error reading main catalog: {e}")
        sys.exit(1)

    # Read all contributing catalogs
    contributing_cat = Catalog()
    for f in files:
        logger.info(f"Reading contributing catalog: {f}")
        try:
            contributing_cat.extend(read_events(f))
        except Exception as e:
            logger.warning(f"Failed to read contributing catalog {f}: {e}")

    # Merge contributing catalogs into main catalog
    logger.info("Starting merge process...")
    for index, row in df.iterrows():
        # Get the main event_id from the merge_info_file
        myevent_id = row["event_id"]

        # Get the main event from the main catalog
        try:
            myevent = [e for e in mycat.events if e.resource_id.id == myevent_id].pop()
        except IndexError:
            logger.warning(f"Event {myevent_id} not found in main catalog. Skipping.")
            continue

        # Get contributing events from the contributing catalogs
        agencies_event_list = row["agencies_list"].split()
        events = [e for e in contributing_cat.events if e.resource_id.id in agencies_event_list]

        if not events:
            logger.warning(f"No contributing events found for {myevent_id}. Skipping.")
            continue

        # Collect preferred origins, magnitudes, and picks
        origins = [e.preferred_origin() for e in events if e.preferred_origin()]
        magnitudes = [e.preferred_magnitude() for e in events if e.preferred_magnitude()]
        picks = [p for e in events for p in e.picks]

        # Merge into the main event
        myevent.origins.extend(origins)
        myevent.magnitudes.extend(magnitudes)
        myevent.picks.extend(picks)

    # Write output files
    for i in cfg["output"]:
        output_file = i["filename"]
        output_format = i["format"]
        logger.info(f"Writing merged catalog to {output_file} in format {output_format}")
        try:
            mycat.write(output_file, format=output_format)
        except Exception as e:
            logger.error(f"Failed to write output file {output_file}: {e}")
            sys.exit(1)

    logger.info("Merge process completed successfully.")
