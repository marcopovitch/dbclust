#!/usr/bin/env python
import argparse
import base64
import logging
import os
import warnings
from datetime import datetime
from datetime import timezone
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import alphabetic_timestamp as ats
from obspy import Catalog
from obspy import read_events
from obspy import UTCDateTime
from obspy.core.event import Comment
from obspy.core.event import Event
from obspy.core.event import Magnitude
from obspy.core.event import Origin
from obspy.core.event import ResourceIdentifier
from obspy.core.event.origin import Pick
from obspy.geodetics import gps2dist_azimuth

warnings.filterwarnings("ignore", category=UserWarning, module="obspy")

# default logger (uses hierarchical name for selective level control)
logger = logging.getLogger("dbclust.quakeml")


def safe_creation_time(origin):
    if origin.creation_info and origin.creation_info.creation_time:
        return origin.creation_info.creation_time.timestamp
    return 0


def datetime_to_base64_timestamp(dt, precision="microsecond"):
    # dt is a naive datetime whose fields are already in UTC (from
    # UTCDateTime.datetime); .timestamp() on a naive datetime interprets it
    # as local time, so force UTC explicitly to avoid machine-timezone-
    # dependent event IDs.
    timestamp = dt.replace(tzinfo=timezone.utc).timestamp()
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


def make_magnitude_id(event: Event) -> ResourceIdentifier:
    """
    Generate a unique magnitude ID for an event.
    This function creates a unique magnitude ID for an event by iterating through
    existing magnitude IDs and appending a number to the event's resource ID until
    a unique ID is found.

    Args:
        event (Event): The event object containing magnitudes and a resource ID.
    Returns:
        ResourceIdentifier: A unique resource identifier for the magnitude.
    """
    magnitude_id_list = {m.resource_id.id for m in event.magnitudes}
    n_magnitudes = 0
    while True:
        magnitude_id = f"{event.resource_id.id}/magnitude/{n_magnitudes}"
        if magnitude_id not in magnitude_id_list:
            break
        n_magnitudes += 1
    return ResourceIdentifier(magnitude_id)


def make_station_magnitude_contribution_id(magnitude: Magnitude) -> ResourceIdentifier:
    """
    Generate a unique station magnitude ID for a magnitude.
    This function creates a unique station magnitude ID for a magnitude by iterating through
    existing station magnitude IDs and appending a number to the magnitude's resource ID until
    a unique ID is found.

    Args:
        magnitude (Magnitude): The magnitude object containing station magnitudes and a resource ID.
    Returns:
        ResourceIdentifier: A unique resource identifier for the station magnitude.
    """
    station_magnitude_id_list = {m.resource_id.id for m in magnitude.station_magnitude_contributions}
    n_station_magnitudes = 0
    while True:
        station_magnitude_id = f"{magnitude.resource_id.id}/station_magnitude_contribution/{n_station_magnitudes}"
        if station_magnitude_id not in station_magnitude_id_list:
            break
        n_station_magnitudes += 1
    return ResourceIdentifier(station_magnitude_id)


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
        logger.debug(f"Event {e.resource_id.id} has {len(e.origins)} origins.")

        dedup_pick_map = getattr(e, "_dedup_pick_map", {})

        # check every arrival has a pick_id and rewire from dedup map if needed
        valid_pick_ids = {p.resource_id.id for p in e.picks}
        for o in e.origins:
            filtered_arrivals = []
            for a in o.arrivals:
                if not a.pick_id:
                    logger.warning(f"Arrival {a.resource_id} has no pick_id.")
                    continue
                if a.pick_id.id in dedup_pick_map:
                    mapped_id = _resolve_pick_id(a.pick_id.id, dedup_pick_map)
                    logger.debug(
                        "Arrival %s remapped pick %s -> %s via dedup map",
                        a.resource_id.id if a.resource_id else "unknown",
                        a.pick_id.id,
                        mapped_id,
                    )
                    a.pick_id = ResourceIdentifier(mapped_id)
                if a.pick_id.id not in valid_pick_ids:
                    logger.warning(
                        "Dropping arrival %s referencing missing pick %s",
                        a.resource_id.id if a.resource_id else "unknown",
                        a.pick_id.id,
                    )
                    continue
                filtered_arrivals.append(a)
            if len(filtered_arrivals) != len(o.arrivals):
                logger.debug(
                    "Origin %s: pruned %d arrivals referencing missing picks",
                    o.resource_id.id if o.resource_id else "unknown",
                    len(o.arrivals) - len(filtered_arrivals),
                )

            # Deduplicate arrivals that point to the same pick_id with the same phase
            # This can happen when picks on different channels were deduplicated but
            # their arrivals were kept separately
            seen_arrival_keys = {}
            deduplicated_arrivals = []
            for a in filtered_arrivals:
                # Key: (pick_id, phase) - two arrivals pointing to same pick with same phase are duplicates
                arrival_key = (a.pick_id.id if a.pick_id else None, str(a.phase))
                if arrival_key in seen_arrival_keys:
                    logger.debug(
                        "Removing duplicate arrival for pick %s phase %s",
                        a.pick_id.id if a.pick_id else "unknown",
                        a.phase,
                    )
                    continue
                seen_arrival_keys[arrival_key] = a
                deduplicated_arrivals.append(a)

            if len(deduplicated_arrivals) != len(filtered_arrivals):
                logger.debug(
                    "Origin %s: deduplicated %d arrivals pointing to same pick/phase",
                    o.resource_id.id if o.resource_id else "unknown",
                    len(filtered_arrivals) - len(deduplicated_arrivals),
                )
            o.arrivals = deduplicated_arrivals

        logger.debug(f"Event {e.resource_id.id} has {sum(len(o.arrivals) for o in e.origins)} arrivals from all origins.")

        # Generate a new ID for the event
        o = e.preferred_origin()
        if o is None:
            raise ValueError(f"Event {e.resource_id.id} has no preferred origin.")

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

        logger.debug(f"Event {e.resource_id.id} has {len(e.picks)} picks.")
        logger.debug(f"Event {e.resource_id.id} has {len(pick_lookup_table)} pick lookup table entries.")

        # Generate readable IDs for origins
        origin_map = {}
        for o in sorted(e.origins, key=safe_creation_time):
            origin_id = make_origin_id(e)
            origin_map[o.resource_id.id] = origin_id
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

        # Clear dedup map once processed
        if hasattr(e, "_dedup_pick_map"):
            delattr(e, "_dedup_pick_map")

        # Generate readable IDs for magnitude origins
        for m in e.magnitudes:
            magnitude_id = make_magnitude_id(e)
            if e.preferred_magnitude_id and m.resource_id.id == e.preferred_magnitude_id.id:
                e.preferred_magnitude_id = magnitude_id
            m.resource_id = magnitude_id

            if m.origin_id.id in origin_map:
                m.origin_id = origin_map[m.origin_id.id]
            else:
                logger.warning(f"Magnitude {m.resource_id.id} references a missing origin {m.origin_id.id}.")
                m.origin_id = None

            # Generate readable IDs for station magnitude contribution
            for sm in m.station_magnitude_contributions:
                station_magnitude_contribution_id = make_station_magnitude_contribution_id(m)
                sm.resource_id = station_magnitude_contribution_id


    return cat


def _remap_arrivals_from_map(event: Event, pick_map: Dict[str, str]) -> None:
    """Update event arrivals according to provided pick mapping."""
    if not pick_map:
        return
    for origin in event.origins:
        for arrival in origin.arrivals:
            if arrival.pick_id:
                resolved_id = _resolve_pick_id(arrival.pick_id.id, pick_map)
                if resolved_id != arrival.pick_id.id:
                    arrival.pick_id = ResourceIdentifier(resolved_id)


def _record_dedup_pick_map(event: Event, pick_map: Dict[str, str]) -> None:
    """Persist mapping on the event for later stages (e.g. readable IDs)."""
    if not pick_map:
        return
    existing_map = getattr(event, "_dedup_pick_map", {})
    combined_map = dict(existing_map)
    combined_map.update(pick_map)
    setattr(event, "_dedup_pick_map", combined_map)


def _resolve_pick_id(pick_id: str, pick_map: Dict[str, str]) -> str:
    """Follow pick_map transitively to find the surviving pick id."""
    visited = set()
    current = pick_id
    while current in pick_map and current not in visited:
        visited.add(current)
        new_id = pick_map[current]
        if new_id == current:
            break
        current = new_id
    return current


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
    unique_picks = {}
    new_pick_list: List[Pick] = []
    removed_picks: List[Pick] = []

    logger.debug(
        "Starting deduplicate_picks_one_pass for event %s with %d picks",
        event.resource_id.id if event.resource_id else "unknown",
        len(event.picks),
    )

    for pick in event.picks:
        # Deduplicate by network + station + time + phase, ignoring channel
        # This ensures picks on different channels (HH vs BH) for the same
        # station/time/phase are deduplicated
        key = (
            pick.waveform_id.network_code if pick.waveform_id else None,
            pick.waveform_id.station_code if pick.waveform_id else None,
            round(pick.time.timestamp, 6),  # Tolerance on time
            pick.phase_hint,
        )
        # Original key using full waveform_id (including channel):
        # key = (
        #     pick.waveform_id.get_seed_string() if pick.waveform_id else None,
        #     round(pick.time.timestamp, 6),  # Tolerance on time
        #     pick.phase_hint,
        # )

        if key in unique_picks:
            ref_pick = unique_picks[key]

            if pick.resource_id and ref_pick.resource_id:
                # Always memoize, even if ids are equal, so later passes know the survivor
                pick_map[pick.resource_id.id] = ref_pick.resource_id.id

            removed_picks.append(pick)

            logger.debug(
                f"Duplicate found: {pick.resource_id.id} -> {ref_pick.resource_id.id}"
            )
        else:
            unique_picks[key] = pick  # Add a new unique pick
            new_pick_list.append(pick)
    # count the number of unique picks and duplicates
    logger.debug(
        f"Picks deduplication: number of unique picks: {len(unique_picks)}, "
        f"number of duplicates: {len(removed_picks)}"
    )

    if not removed_picks:
        return False  # No duplicates found

    # Update references in arrivals and persist mapping for later stages
    _remap_arrivals_from_map(event, pick_map)
    _record_dedup_pick_map(event, pick_map)

    # Remove duplicate picks
    event.picks = new_pick_list

    logger.debug(
        "Removed %d duplicate picks from event %s, %d remaining. Removed ids=%s",
        len(removed_picks),
        event.resource_id.id if event.resource_id else "unknown",
        len(event.picks),
        sorted([p.resource_id.id for p in removed_picks if p.resource_id]),
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
    iteration = 0
    before = len(event.picks)
    while deduplicate_picks_one_pass(event):
        iteration += 1
        logger.debug(
            "deduplicate_picks iteration %d for event %s -> %d picks remaining",
            iteration,
            event.resource_id.id if event.resource_id else "unknown",
            len(event.picks),
        )

    if iteration:
        logger.info(
            "deduplicate_picks removed %d picks from event %s (initial=%d, final=%d)",
            before - len(event.picks),
            event.resource_id.id if event.resource_id else "unknown",
            before,
            len(event.picks),
        )

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


def remove_duplicate_picks(event: Event) -> None:
    """
    Removes duplicate picks (same resource ID) from an event while recording mapping
    so arrivals keep pointing to surviving picks.
    """
    seen_ids: Dict[str, Tuple[float, Optional[str]]] = {}
    unique_picks: List[Pick] = []
    pick_map: Dict[str, str] = {}
    duplicates: List[str] = []

    for pick in event.picks:
        if not pick.resource_id:
            logger.error("Pick without resource_id found")
            continue

        pick_id = pick.resource_id.id
        pick_values = (
            round(pick.time.timestamp, 6),
            pick.waveform_id.get_seed_string() if pick.waveform_id else None,
        )

        if pick_id in seen_ids:
            if seen_ids[pick_id] != pick_values:
                logger.error(
                    "Conflict for pick %s: inconsistent time/station for identical id",
                    pick_id,
                )
            else:
                logger.debug("Pathological duplicate pick ignored: %s", pick_id)
            duplicates.append(pick_id)
            pick_map[pick_id] = pick_id  # arrival should keep pointing to survivor
        else:
            seen_ids[pick_id] = pick_values
            unique_picks.append(pick)

    logger.debug(
        "Total number of picks: %d, number of pathological duplicates: %d, "
        "number of remaining picks: %d",
        len(event.picks),
        len(event.picks) - len(unique_picks),
        len(unique_picks),
    )
    if duplicates:
        logger.debug(
            "remove_duplicate_picks removed %d duplicate ids: %s",
            len(duplicates),
            sorted(duplicates),
        )

    if pick_map:
        _remap_arrivals_from_map(event, pick_map)
        _record_dedup_pick_map(event, pick_map)

    event.picks = unique_picks


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
        event_id = e.resource_id.id if e.resource_id else "unknown"
        before_total = len(e.picks)
        # remove picks with same id. It should not happen but it happens (picks info are the same)
        remove_duplicate_picks(e)
        after_remove = len(e.picks)
        logger.debug(
            "Event %s: remove_duplicate_picks -> %d picks (from %d)",
            event_id,
            after_remove,
            before_total,
        )
        e = deduplicate_picks(e)
        logger.debug(
            "Event %s: deduplicate_picks final count %d",
            event_id,
            len(e.picks),
        )

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
