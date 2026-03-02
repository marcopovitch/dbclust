#!/usr/bin/env python
import argparse
import concurrent.futures
import glob
import logging
import os
import sys
import traceback
import urllib.request
from dataclasses import asdict
from datetime import datetime
from datetime import timedelta
from shutil import copyfile
from icecream import ic
import pandas as pd
from obspy import read_events

from dbclust.config import DBClustConfig
from dbclust.localization import LocalizationError
from dbclust.localization import NllLoc
from dbclust.localization import reloc_fdsn_event
from dbclust.localization import show_bulletin
from dbclust.localization import show_event
from dbclust.core import MyTemporaryDirectory

# Default logger (uses hierarchical name for selective level control)
logger = logging.getLogger("dbclust.reprocess")


def round_to_centisecond(dt: datetime) -> datetime:
    """
    Round a datetime object to the nearest centisecond (10 ms).
    Args:
        dt (datetime): The datetime object to round.
    Returns:
        datetime: The rounded datetime object.
    """
    us = dt.microsecond
    rounded_us = round(us / 10000) * 10000
    if rounded_us == 1000000:
        return dt.replace(microsecond=0) + timedelta(seconds=1)
    return dt.replace(microsecond=rounded_us)


def process_file(
    f: str,
    cfg: DBClustConfig,
    args,
    output_format: str = "QUAKEML",
    verbose: bool = True,
) -> str:
    """
    Process a QuakeML file, relocate the event, and save the result.

    Args:
        f (str): Path to the QuakeML file.
        cfg (DBClustConfig): Configuration object.
        args: Command line arguments.
        output_format (str): Output format for the event file.

    Returns:
        str: Status message.
    """
    if not os.path.exists(f):
        err_msg = f"File {f} does not exist"
        logging.error(err_msg)
        return err_msg

    # Log localization method being used
    loc_method = cfg.nll.loc_method if hasattr(cfg.nll, "loc_method") else "EDT_OT_WT"
    logger.info(f"Relocating {f} using {loc_method} localization method")

    try:
        cat = read_events(f)

        if len(cat) == 0:
            err_msg = f"No event found in QuakeML file {f}"
            logging.error(err_msg)
            return err_msg
        elif len(cat) > 1:
            err_msg = f"More than one event found in QuakeML file {f}"
            logging.error(err_msg)
            return err_msg

        event = cat[0]
        for p in event.picks:
            # round the time to centisecond precision
            # needed to avoid issues pick matching using obspy/NonLinLoc
            p.time = round_to_centisecond(p.time)

        o = event.preferred_origin() or event.origins[0]
        zone, _ = cfg.zones.find_zone(o.latitude, o.longitude)
        logger.debug("Relocation zone: %s", zone["name"])

        with MyTemporaryDirectory(dir=cfg.file.tmp_path, delete=True) as tmp_path:
            locator = NllLoc(
                cfg.nll.nlloc_bin,
                cfg.nll.scat2latlon_bin,
                cfg.nll.time_path,
                tmpdir=tmp_path,
                loc_method=loc_method,
                #
                force_uncertainty=cfg.relocation.force_uncertainty,
                P_uncertainty=cfg.relocation.P_uncertainty,
                S_uncertainty=cfg.relocation.S_uncertainty,
                #
                double_pass=cfg.relocation.double_pass,
                gap_dist_max_km=cfg.relocation.gap_dist_max_km,
                closest_station_dist_km=cfg.relocation.closest_station_dist_km,
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
                enable_relabel_pick_zone=args.relabel,
                enable_cleanup_pick_zone=True,
            )

            try:
                cat = reloc_fdsn_event(locator, event=event, zone_name=args.zone_name)
            except LocalizationError as e:
                err_msg = f"[{f}] Error during relocation. {e}"
                logging.error(err_msg)
                return err_msg

            if len(cat) == 0:
                err_msg = "No relocated event found"
                logging.error(err_msg)
                return err_msg
            elif len(cat) > 1:
                err_msg = "More than one relocated event found"
                logging.error(err_msg)
                return err_msg

            # merge relocated event with original event (avoiding duplicates)
            e = cat[0]
            existing_origin_ids = {o.resource_id.id for o in e.origins}
            existing_pick_ids = {p.resource_id.id for p in e.picks}
            existing_amplitude_ids = {a.resource_id.id for a in e.amplitudes}
            existing_magnitude_ids = {m.resource_id.id for m in e.magnitudes}

            e.origins.extend(
                o for o in event.origins if o.resource_id.id not in existing_origin_ids
            )
            e.origins.sort(
                key=lambda x: x.creation_info.creation_time or 0, reverse=True
            )
            e.picks.extend(
                p for p in event.picks if p.resource_id.id not in existing_pick_ids
            )
            e.amplitudes.extend(
                a for a in event.amplitudes if a.resource_id.id not in existing_amplitude_ids
            )
            e.magnitudes.extend(
                m for m in event.magnitudes if m.resource_id.id not in existing_magnitude_ids
            )

            # show relocated event
            if verbose:
                show_event(e, "****", header=True)
                show_bulletin(e, zones=cfg.zones, plot=False)

            # Create output directory based on year and month
            event_time = e.preferred_origin().time
            year = str(event_time.year)
            month = str(event_time.month).zfill(
                2
            )  # Ensure month is 2 digits (e.g., "01" for January)

            if args.output_name:
                # Use the provided output name and directory
                output_path = args.output_name
                output_dir = os.path.dirname(output_path)
                if output_dir:
                    os.makedirs(output_dir, exist_ok=True)
            else:
                # Extract the basename from the input file (without path and extension)
                input_basename = os.path.basename(f)
                basename = os.path.splitext(input_basename)[0]

                # Create directory structure
                output_dir = os.path.join(year, month)
                os.makedirs(output_dir, exist_ok=True)
                logger.info(f"Created output directory: {output_dir}")
                file_extension = output_format.lower()
                output_filename = f"{basename}.{file_extension}"
                output_path = os.path.join(output_dir, output_filename)

            # Write the catalog to file
            logger.info(f"Writing output to: {output_path}")
            cat.write(output_path, format=output_format)

            # Handle the scatter file if available
            if locator.scat_file:
                try:
                    scat_filename = f"{basename}.scat"
                    scat_path = os.path.join(output_dir, scat_filename)
                    logger.info(f"Copying scatter file to: {scat_path}")
                    copyfile(locator.scat_file, scat_path)
                except (IOError, OSError) as e:
                    logging.error(f"Can't get nll scat file: {e}")

            return "OK"
    except Exception as e:
        err_msg = f"Unexpected error processing file {f}: {str(e)}"
        logging.error(err_msg)
        logging.error(traceback.format_exc())
        return err_msg


def process_directory(
    directory: str,
    cfg: DBClustConfig,
    args,
    output_format: str = "QUAKEML",
    max_workers: int = 4,
) -> pd.DataFrame:
    """
    Process all QuakeML files in a directory using parallel execution.

    Args:
        directory (str): Directory containing QuakeML files.
        cfg (DBClustConfig): Configuration object.
        args: Command line arguments.
        output_format (str): Output format for the event file.
        max_workers (int): Maximum number of parallel workers.

    Returns:
        pd.DataFrame: DataFrame with processing results.
    """
    files = glob.glob(f"{directory}/**/*.qml", recursive=True)
    # files = glob.glob(f"{directory}/2021/11/*.qml")
    verbose = False

    results = []

    # Using ThreadPoolExecutor for parallel processing
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_file, f, cfg, args, output_format, verbose): f
            for f in files
        }

        for future in concurrent.futures.as_completed(futures):
            f = futures[future]
            try:
                status = future.result()
                results.append({"file": f, "status": status})
            except Exception as exc:
                results.append({"file": f, "status": f"Generated exception: {exc}"})

    return pd.DataFrame(results)


def fetch_event_from_fdsn(event_id: str, fdsnws_url: str, output_file: str) -> None:
    """
    Fetch an event from FDSNWS server.

    Args:
        event_id (str): Event ID to fetch.
        fdsnws_url (str): FDSNWS server URL.
        output_file (str): Path to save the event file.

    Raises:
        urllib.error.HTTPError: If HTTP request fails.
        Exception: For other errors.
    """
    options = "includeallorigins=true&includeallmagnitudes=true&includearrivals=true&nodata=404"
    url = f"{fdsnws_url}/query?{options}&eventid={event_id}"

    try:
        logging.info(f"Fetching event from {url}")
        urllib.request.urlretrieve(url, output_file)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            raise FileNotFoundError(f"Event {event_id} not found in FDSNWS")
        else:
            raise


def setup_logging(loglevel: str) -> int:
    """
    Setup logging with the specified level.

    Args:
        loglevel (str): Log level string (debug, info, warning, error).

    Returns:
        int: Numeric log level.

    Raises:
        ValueError: If log level is invalid.
    """
    numeric_level = getattr(logging, loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {loglevel}")

    logging.basicConfig(stream=sys.stdout, level=numeric_level, force=True)
    # Set level on the dbclust parent logger so all child loggers inherit it
    logging.getLogger("dbclust").setLevel(numeric_level)
    return numeric_level


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Relocate seismic events using NonLinLoc."
    )

    # Input source group - mutually exclusive
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "-e", "--eventid", dest="event_id", help="Event ID to fetch from FDSN", type=str
    )
    input_group.add_argument(
        "--event", dest="event", help="Event file in QuakeML format", type=str
    )
    input_group.add_argument(
        "--dir",
        dest="dir",
        help="Directory containing QuakeML files to relocate",
        type=str,
    )

    # Configuration
    parser.add_argument(
        "-c",
        "--conf",
        required=True,
        dest="profile_conf_file",
        help="dbclust configuration file",
        type=str,
    )
    parser.add_argument(
        "-f",
        "--fdsn-event-profile",
        dest="fdsn_event_profile",
        help="FDSN event profile name to use (see conf.yml file)",
        type=str,
    )

    # Relocation parameters
    reloc_group = parser.add_argument_group("Relocation parameters")
    reloc_group.add_argument(
        "-d",
        "--dist-km-cutoff",
        dest="dist_km_cutoff",
        help="Station cut off distance in km",
        type=float,
    )
    reloc_group.add_argument(
        "-u",
        "--use-deactivated-arrivals",
        dest="use_deactivated_arrivals",
        help="Force deactivated arrivals use",
        action="store_true",
    )
    reloc_group.add_argument(
        "-t",
        "--min-score-threshold-pick-zone",
        dest="min_score_threshold_pick_zone",
        help="Minimum score threshold pick zone",
        type=float,
    )
    reloc_group.add_argument(
        "-r", "--relabel", dest="relabel", help="Enable relabeling", action="store_true"
    )
    reloc_group.add_argument(
        "--force-uncertainty",
        dest="force_uncertainty",
        nargs=2,
        metavar=("P_uncertainty", "S_uncertainty"),
        type=float,
        help="Force phase uncertainty: provide P_uncertainty and S_uncertainty",
        default=None,
    )
    reloc_group.add_argument(
        "--single-pass",
        dest="single_pass",
        help="NonLinLoc single pass (disables double pass)",
        action="store_true",
    )
    reloc_group.add_argument(
        "-z",
        "--zone",
        dest="zone_name",
        help="Force zone name to use (default is autodetect from event lat/lon)",
        type=str,
    )

    # Output options
    output_group = parser.add_argument_group("Output options")
    output_group.add_argument(
        "-s", "--scat", dest="scat", help="Get xyz scat file", action="store_true"
    )
    output_group.add_argument(
        "-o",
        "--output",
        dest="output_name",
        default=None,
        help="Output name for the event file",
        type=str,
    )

    output_group.add_argument(
        "--format",
        dest="output_format",
        default="QUAKEML",
        help="Output format for the event file",
        type=str,
    )

    # Parallel processing
    parser.add_argument(
        "--max-workers",
        dest="max_workers",
        default=4,
        help="Maximum number of parallel workers for directory processing",
        type=int,
    )

    # Logging
    parser.add_argument(
        "-l",
        "--loglevel",
        dest="loglevel",
        default="INFO",
        help="Set loglevel (debug, warning, info, error)",
        type=str,
    )

    return parser.parse_args()


def main():
    """Main function to run the relocation process."""
    try:
        args = parse_arguments()

        try:
            numeric_level = setup_logging(args.loglevel)
        except ValueError as e:
            logger.error(str(e))
            logger.error("Log level should be: debug, warning, info, error.")
            sys.exit(255)

        # Load configuration
        try:
            cfg = DBClustConfig(args.profile_conf_file, config_type="reloc")
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            # show traceback
            logger.error(traceback.format_exc())
            sys.exit(1)

        if args.output_name and os.path.isfile(args.output_name):
            logger.error(
                f"Error: file {args.output_name} already exists. Please remove it first."
            )
            sys.exit(1)

        # Update configuration from command line arguments
        if args.dist_km_cutoff:
            cfg.relocation.dist_km_cutoff = args.dist_km_cutoff

        if args.use_deactivated_arrivals:
            cfg.relocation.use_deactivated_arrivals = args.use_deactivated_arrivals

        if args.force_uncertainty:
            cfg.relocation.force_uncertainty = True
            cfg.relocation.P_uncertainty = args.force_uncertainty[0]
            cfg.relocation.S_uncertainty = args.force_uncertainty[1]
        else:
            cfg.relocation.force_uncertainty = False
            cfg.relocation.P_uncertainty = None
            cfg.relocation.S_uncertainty = None

        if args.single_pass:
            cfg.relocation.double_pass = not args.single_pass

        if args.scat:
            cfg.nll.enable_scatter = args.scat

        if not args.zone_name:
            cfg.quakeml.model_id = None

        if args.min_score_threshold_pick_zone:
            cfg.relocation.min_score_threshold_pick_zone = (
                args.min_score_threshold_pick_zone
            )

        # Setup FDSNWS URL if needed
        if args.fdsn_event_profile:
            cfg.fdsnws_event.set_url_from_service_name(args.fdsn_event_profile)
            ic(cfg.fdsnws_event.get_url())

        # Validate input sources
        if args.dir and not os.path.exists(args.dir):
            logger.error("Please provide a valid directory")
            sys.exit(1)
        elif args.event and not os.path.exists(args.event):
            logger.error("Please provide a valid event file")
            sys.exit(1)

        # Process based on input source
        if args.event:
            with MyTemporaryDirectory(dir=cfg.file.tmp_path, delete=True) as tmp_path:
                process_file(
                    args.event,
                    cfg,
                    args,
                    args.output_format,
                )
        elif args.dir:
            results_df = process_directory(
                args.dir,
                cfg,
                args,
                args.output_format,
                args.max_workers,
            )
            # Save results to CSV
            csv_file = "reprocess_event_status.csv"
            results_df.to_csv(csv_file, index=False)
            logger.info(f"Results saved to {csv_file}")
        else:  # args.event_id
            # Create a temporary directory for the event
            with MyTemporaryDirectory(dir=cfg.file.tmp_path, delete=True) as tmp_path:
                filename = os.path.join(tmp_path, f"{args.event_id}.xml")
                try:
                    fetch_event_from_fdsn(
                        args.event_id, cfg.fdsnws_event.get_url(), filename
                    )
                    process_file(
                        filename,
                        cfg,
                        args,
                        args.output_format,
                    )
                except FileNotFoundError as e:
                    logger.error(str(e))
                    sys.exit(1)
                except urllib.error.HTTPError as e:
                    logger.error(
                        f"Error fetching event {args.event_id} from FDSNWS: {e}"
                    )
                    sys.exit(1)
                except Exception as e:
                    logger.error(
                        f"Error fetching event {args.event_id} from FDSNWS: {e}"
                    )
                    logger.error(traceback.format_exc())
                    sys.exit(1)

    except Exception as e:
        logger.error(f"Unhandled error: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
