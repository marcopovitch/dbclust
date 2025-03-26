#!/usr/bin/env python
import argparse
import base64
import logging
import os
import sys
import warnings
from datetime import datetime
from itertools import combinations
from typing import Dict
from typing import List

import alphabetic_timestamp as ats
from icecream import ic
from obspy import Catalog
from obspy import read_events
from obspy import UTCDateTime
from obspy.core.event import Comment
from obspy.core.event import CreationInfo
from obspy.core.event import Event
from obspy.core.event import Origin
from obspy.core.event import ResourceIdentifier
from obspy.core.event.base import WaveformStreamID
from obspy.core.event.origin import Pick
from obspy.geodetics import gps2dist_azimuth

warnings.filterwarnings("ignore", category=UserWarning, module="obspy")

# default logger
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger("quakeml")
logger.setLevel(logging.INFO)


def safe_creation_time(origin):
    if origin.creation_info and origin.creation_info.creation_time:
        return origin.creation_info.creation_time.timestamp
    return 0


def datetime_to_base64_timestamp(dt, precision="microsecond"):
    # Convert the datetime object to a timestamp in seconds
    timestamp = dt.timestamp()  # Returns the seconds with decimals
    if precision == "microsecond":
        fractional_part = int(dt.microsecond)
    elif precision == "millisecond":
        fractional_part = int(dt.microsecond / 1000)
    elif precision == "centisecond":
        fractional_part = int(dt.microsecond / 10000)
    elif precision == "decisecond":
        fractional_part = int(dt.microsecond / 100000)
    else:
        raise ValueError(
            "Invalid precision. Use 'microsecond', 'millisecond', 'centisecond', or 'decisecond'."
        )

    # Create a combined integer from the integer and fractional parts
    combined = int(timestamp) * (10**6) + fractional_part

    # Encode in base64
    base64_encoded = base64.urlsafe_b64encode(
        combined.to_bytes((combined.bit_length() + 7) // 8, "big")
    ).decode("utf-8")

    # Return the timestamp
    return base64_encoded.rstrip("=")  # Remove the '=' padding characters


def make_event_id(time: UTCDateTime, prefix: str, smi_base: str) -> ResourceIdentifier:
    """
    Generate a unique event identifier based on the provided time, prefix, and SMI base.

    Args:
        time (UTCDateTime): The time of the event.
        prefix (str): A prefix to be added to the event ID.
        smi_base (str): The base URL for the SMI (Seismological Metadata Identifier).

    Returns:
        ResourceIdentifier: A unique resource identifier for the event.
    """
    dt = time.datetime
    year = time.year
    # alphatime = ats.base36.from_datetime(dt, time_unit=ats.TimeUnit.milliseconds)
    alphatime = datetime_to_base64_timestamp(dt, precision="microsecond")
    # ic(dt, alphatime)
    event_id = f"{prefix}{year}{alphatime}"
    event_resource_id = ResourceIdentifier("/".join([smi_base, "event", event_id]))
    return event_resource_id


def make_origin_id(event: Event) -> ResourceIdentifier:
    """
    Generate a unique origin ID for an event.
    This function creates a unique origin ID for an event by iterating through
    existing origin IDs and appending a number to the event's resource ID until
    a unique ID is found.

    Args:
        event (Event): The event object containing origins and a resource ID.
    Returns:
        ResourceIdentifier: A unique resource identifier for the origin.
    """

    origin_id_list = {o.resource_id.id for o in event.origins}
    n_origins = 0
    while True:
        origin_id = f"{event.resource_id.id}/origin/{n_origins}"
        if origin_id not in origin_id_list:
            break
        n_origins += 1
    return ResourceIdentifier(origin_id)


def make_pick_id(event: Event) -> ResourceIdentifier:
    """
    Generate a unique pick ID for an event.
    This function creates a unique pick ID for an event by iterating through
    existing pick IDs and appending a number to the event's resource ID until
    a unique ID is found.

    Args:
        event (Event): The event object containing picks and a resource ID.
    Returns:
        ResourceIdentifier: A unique resource identifier for the pick.
    """
    pick_id_list = {p.resource_id.id for p in event.picks}
    n_picks = 0
    while True:
        pick_id = f"{event.resource_id.id}/pick/{n_picks}"
        if pick_id not in pick_id_list:
            break
        n_picks += 1
    return ResourceIdentifier(pick_id)


def make_comment_id(parent) -> ResourceIdentifier:
    """
    Generate a unique comment ID for a parent object.
    This function creates a unique comment ID for a parent object by iterating
    through existing comment IDs and appending a number to the parent's resource
    ID until a unique ID is found.

    Args:
        parent: The parent object containing comments and a resource ID.
    Returns:
        ResourceIdentifier: A unique resource identifier for the comment.
    """
    comment_list = [c.resource_id.id for c in parent.comments if c.resource_id]
    n_comments = 0
    while True:
        comment_id = f"{parent.resource_id.id}/comment/{n_comments}"
        if comment_id not in comment_list:
            break
        n_comments += 1
    return ResourceIdentifier(comment_id)


def make_arrival_id(origin: Origin) -> ResourceIdentifier:
    """
    Generate a unique arrival ID for an origin.
    This function creates a unique arrival ID for an origin by iterating through
    existing arrival IDs and appending a number to the origin's resource ID until
    a unique ID is found.

    Args:
        origin (Origin): The origin object containing arrivals and a resource ID.
    Returns:
        ResourceIdentifier: A unique resource identifier for the arrival.
    """
    arrival_id_list = {a.resource_id.id for a in origin.arrivals}
    n_arrival = 0
    while True:
        arrival_id = f"{origin.resource_id.id}/arrival/{n_arrival}"
        if arrival_id in arrival_id_list:
            n_arrival += 1
        else:
            break
    return ResourceIdentifier(arrival_id)


def make_readable_id(cat: Catalog, prefix: str, smi_base: str) -> Catalog:
    """
    Make the IDs of the given catalog readable by replacing
    the existing IDs with human-readable IDs based on the provided prefix and SMI base.

    Args:
        cat (Catalog): The catalog object to update.
        prefix (str): A prefix to be added to the event ID.
        smi_base (str): The base URL for the SMI (Seismological Metadata Identifier).
    Returns:
        Catalog: The catalog object with human-readable IDs.
    """
    # Generate a readable catalog ID
    alphatime = ats.base36.from_datetime(
        datetime.now(), time_unit=ats.TimeUnit.milliseconds
    )
    catalog_id = "/".join([smi_base, "catalog", alphatime])
    cat.resource_id = ResourceIdentifier(catalog_id)

    for e in cat.events:
        # Generate a new ID for the event
        o = e.preferred_origin()
        if o is None:
            raise ValueError(f"Event {e.resource_id} has no preferred origin.")
        event_id = make_event_id(o.time, prefix, smi_base)
        e.resource_id = event_id

        # Generate readable IDs for associated comments
        for c in e.comments:
            comment_id = make_comment_id(e)
            c.resource_id = comment_id

        # Create a lookup table for pick IDs
        pick_lookup_table: Dict[str, str] = {}
        for p in sorted(e.picks, key=lambda p: p.time):
            old_pick_id = p.resource_id.id
            pick_id = make_pick_id(e)
            p.resource_id = pick_id
            pick_lookup_table[old_pick_id] = pick_id.id

            for c in p.comments:
                comment_id = make_comment_id(p)
                c.resource_id = comment_id

        # Generate readable IDs for origins
        for o in sorted(e.origins, key=safe_creation_time):
            origin_id = make_origin_id(e)
            if o.resource_id.id == e.preferred_origin_id.id:
                e.preferred_origin_id = origin_id
            o.resource_id = origin_id

            # Generate readable IDs for arrivals
            for a in o.arrivals:
                arrival_id = make_arrival_id(o)
                a.resource_id = arrival_id

                # Link the pick ID if available in the lookup table
                if a.pick_id.id in pick_lookup_table:
                    a.pick_id = ResourceIdentifier(pick_lookup_table[a.pick_id.id])
                else:
                    logger.warning(
                        f"Arrival {a.resource_id} references a missing pick {a.pick_id.id if a.pick_id else 'None'}."
                    )

    return cat


def deduplicate_picks_one_pass(event: Event) -> bool:
    """
    Deduplicate picks from the given event by identifying and removing duplicate picks
    based on waveform ID, time, and phase hint.
    This function performs a single pass through the list of picks.
    Args:
        event (Event): The event object containing picks.
    Returns:
        bool: True if duplicates were removed, False otherwise.
    """
    if not event.picks:
        return False  # No picks to process

    pick_map = {}  # Map removed pick IDs to their replacements
    to_remove = set()  # List of picks to remove
    unique_picks = {}

    for pick in event.picks:
        key = (
            pick.waveform_id.get_seed_string() if pick.waveform_id else None,
            round(pick.time.timestamp, 6),  # Tolerance on time
            pick.phase_hint,
        )

        if key in unique_picks:
            ref_pick = unique_picks[key]

            # Use ref_pick.resource_id.id as the replacement value
            pick_map[pick.resource_id.id] = ref_pick.resource_id.id
            to_remove.add(pick.resource_id.id)

            logger.info(
                f"Duplicate found: {pick.resource_id.id} -> {ref_pick.resource_id.id}"
            )
        else:
            unique_picks[key] = pick  # Add a new unique pick

    if not to_remove:
        return False  # No duplicates found

    # Update references in arrivals
    for origin in event.origins:
        for arrival in origin.arrivals:
            if arrival.pick_id.id in pick_map:
                arrival.pick_id = pick_map[arrival.pick_id.id]

    # Remove duplicate picks
    event.picks = [p for p in event.picks if p.resource_id.id not in to_remove]

    logger.debug(
        f"Removed {len(to_remove)} duplicate picks, {len(event.picks)} remaining."
    )
    return True


def deduplicate_picks(event: Event) -> Event:
    """
    Deduplicate picks from the given event iteratively until no duplicates remain.

    Args:
        event (Event): The event object containing picks.

    Returns:
        Event: The event object with deduplicated picks.
    """
    while deduplicate_picks_one_pass(event):
        pass

    return event


def feed_distance_from_preloc_to_pref_origin(cat: Catalog) -> Catalog:
    """
    Add a comment to each event in the catalog with the distance (in km) between the
    preferred origin and a prelocation origin (if available).

    Args:
        cat (Catalog): The catalog containing events.

    Returns:
        Catalog: The updated catalog with distance comments added to events.
    """
    for event in cat:
        # Get the preferred origin
        pref_origin = event.preferred_origin()
        if not pref_origin:
            continue

        # Find a prelocation origin and calculate the distance
        distance = None
        for origin in event.origins:
            if (
                origin.resource_id != pref_origin.resource_id
                and "PyOcto" in origin.method_id.id
            ):
                distance, _, _ = gps2dist_azimuth(
                    origin.latitude,
                    origin.longitude,
                    pref_origin.latitude,
                    pref_origin.longitude,
                )
                # Convert distance from meters to kilometers
                distance = distance / 1000.0
                break

        # Append the distance as a comment to the event
        if distance is not None:
            event.comments.append(
                Comment(text=f'{{"preloc_distance_km": {distance:.2f}}}')
            )

    return cat


def remove_duplicate_picks(picks: List[Pick]) -> List[Pick]:
    """
    Removes duplicate picks based on resource ID, ensuring that time, phase hint, and
    waveform ID are identical before removal. If conflicting values exist for the same
    resource ID, logs an error.

    Args:
        picks (List[Pick]): A list of picks to deduplicate.

    Returns:
        List[Pick]: A list of unique picks.
    """
    seen_picks = {}
    unique_picks = []

    for pick in picks:
        if not pick.resource_id:
            logger.error("Pick without resource_id found")
            continue

        pick_id = pick.resource_id.id
        pick_values = (
            round(pick.time.timestamp, 6),  # Rounded to avoid floating-point errors
            pick.phase_hint,
            pick.waveform_id.get_seed_string() if pick.waveform_id else None,
        )

        if pick_id in seen_picks:
            # Check if the values are consistent with those already recorded
            if seen_picks[pick_id] != pick_values:
                logger.error(
                    f"Conflict for pick {pick_id}: inconsistent time, phase, or station"
                )
            else:
                logger.warning(f"Pathological duplicate pick ignored: {pick_id}")
        else:
            seen_picks[pick_id] = pick_values
            unique_picks.append(pick)  # Add to the final list

    # Stats, total number of picks and number of duplicates, remaining picks
    logger.info(
        f"Total number of picks: {len(picks)}, "
        f"number of pathological duplicates: {len(picks) - len(unique_picks)}, "
        f"number of remaining picks: {len(unique_picks)}"
    )
    return unique_picks


# function to deduplicate picks and make readable ids for a catalog
def deduplicate_picks_and_make_readable_ids(
    cat: Catalog, prefix: str, smi_base: str
) -> Catalog:
    """
    Deduplicate picks from the given catalog and make the IDs readable by replacing
    the existing IDs with human-readable IDs based on the provided prefix and SMI base.

    Args:
        cat (Catalog): The catalog object to update.
        prefix (str): A prefix to be added to the event ID.
        smi_base (str): The base URL for the SMI (Seismological Metadata Identifier).

    Returns:
        Catalog: The catalog object with deduplicated picks and human-readable IDs.
    """

    # Deduplicate picks in each event
    for e in cat.events:
        # remove picks with same id. It should not happen but it happens ...
        e.picks = remove_duplicate_picks(e.picks)
        e = deduplicate_picks(e)

    # Make the IDs readable
    cat = make_readable_id(cat, prefix, smi_base)

    return cat


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make readable quakeml IDs")
    parser.add_argument(
        "-i",
        "--input",
        default=None,
        dest="inputfile",
        help="qml input file",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        dest="outputfile",
        help="quakeml output file",
        type=str,
        required=True,
    )
    args = parser.parse_args()

    if not os.path.isfile(args.inputfile):
        print(f"File {args.inputfile} doesn't exist !")
        exit(1)

    if os.path.isfile(args.outputfile):
        print(f"File {args.outputfile} already exist !")
        exit(1)

    logger.info("Reading catalog ...")
    cat = read_events(args.inputfile)
    print(cat)
    logger.info("Pick deduplication and make readable ids ...")
    cat = deduplicate_picks_and_make_readable_ids(
        cat, "sihex", "quakeml:franceseisme.fr"
    )
    print(cat)
    logger.info("Writing catalog ...")
    cat.write(args.outputfile, format="QUAKEML")
