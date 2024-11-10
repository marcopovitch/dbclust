#!/usr/bin/env python
import argparse
import logging
import os
import sys
from datetime import datetime
from itertools import combinations

import alphabetic_timestamp as ats
from icecream import ic
from obspy import Catalog
from obspy import read_events
from obspy.core.event import Comment
from obspy.core.event import CreationInfo
from obspy.core.event import Event
from obspy.core.event import Origin
from obspy.core.event import ResourceIdentifier
from obspy.core.event.base import WaveformStreamID
from obspy.core.event.origin import Pick
from obspy.geodetics import gps2dist_azimuth


# default logger
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger("quakeml")
logger.setLevel(logging.INFO)


def make_event_id(time, prefix, smi_base):
    # set and readable event id
    dt = time.datetime
    year = time.year
    alphatime = ats.base36.from_datetime(dt, time_unit=ats.TimeUnit.milliseconds)
    event_id = f"{prefix}{year}{alphatime}"
    event_resource_id = ResourceIdentifier("/".join([smi_base, "event", event_id]))
    return event_resource_id


def make_origin_id(event):
    # create origin_id
    origin_id_list = [o.resource_id.id for o in event.origins]
    n_origins = 0
    while True:
        origin_id = f"{event.resource_id.id}/origin/{n_origins}"
        if origin_id not in origin_id_list:
            break
        n_origins += 1
    return ResourceIdentifier(origin_id)


def make_pick_id(event):
    pick_id_list = [p.resource_id.id for p in event.picks]
    n_picks = 0
    while True:
        pick_id = f"{event.resource_id.id}/pick/{n_picks}"
        if pick_id not in pick_id_list:
            break
        n_picks += 1
    return ResourceIdentifier(pick_id)


def make_comment_id(parent):
    """
    Replace comments id from parent (event, pick)
    """
    comment_list = [c.resource_id.id for c in parent.comments if c.resource_id]
    n_comments = 0
    while True:
        comment_id = f"{parent.resource_id.id}/comment/{n_comments}"
        if comment_id not in comment_list:
            break
        n_comments += 1
    return ResourceIdentifier(comment_id)


def make_arrival_id(origin):
    arrival_id_list = [a.resource_id.id for a in origin.arrivals]
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
    make object id more readable
    """
    alphatime = ats.base36.from_datetime(
        datetime.now(), time_unit=ats.TimeUnit.milliseconds
    )
    catalog_id = "/".join([smi_base, "catalog", alphatime])
    cat.resource_id = ResourceIdentifier(catalog_id)
    for e in cat.events:
        o = e.preferred_origin()
        event_id = make_event_id(o.time, prefix, smi_base)
        e.resource_id.id = event_id.id

        for c in e.comments:
            comment_id = make_comment_id(e)
            c.resource_id = comment_id

        # forge readable pick_id
        pick_lookup_table = {}
        for p in sorted(e.picks, key=lambda p: p.time):
            old_pick_id = p.resource_id.id
            pick_id = make_pick_id(e)
            p.resource_id = pick_id
            pick_lookup_table[old_pick_id] = pick_id.id
            for c in p.comments:
                comment_id = make_comment_id(p)
                c.resource_id = comment_id

        # ic(pick_lookup_table)
        # forge readable origin_id
        for o in sorted(e.origins, key=lambda o: o.creation_info.version):
            origin_id = make_origin_id(e)
            if o.resource_id == e.preferred_origin_id:
                e.preferred_origin_id = origin_id.id
            o.resource_id = origin_id

            # forge readable arrival_id
            for a in o.arrivals:
                old_arrival_id = a.resource_id
                arrival_id = make_arrival_id(o)
                a.resource_id = arrival_id
                if a.pick_id.id in pick_lookup_table.keys():
                    a.pick_id.id = pick_lookup_table[a.pick_id.id]
                # else:
                #     logger.error(
                #         f"make_readable_id: arrival old:{old_arrival_id}/new:{a.resource_id} with unreferenced pick {a.pick_id.id}!!!"
                #     )
    return cat


def deduplicate_picks(event: Event) -> Event:
    """
    Deduplicate picks from the given event.

    Args:
        event (Event): The event object containing picks.

    Returns:
        Event: The event object with deduplicated picks.
    """
    picks = event.picks
    to_be_removed = []
    match_pick_id = {}
    for p1, p2 in combinations(picks, 2):
        if (
            p1.waveform_id.get_seed_string() == p2.waveform_id.get_seed_string()
            and p1.time == p2.time
            and p1.phase_hint == p2.phase_hint
        ):
            if p1 in to_be_removed or p2 in to_be_removed:
                continue

            # set the id mapping
            match_pick_id[p2.resource_id] = p1.resource_id
            to_be_removed.append(p2)

    for origin in event.origins:
        for arrival in origin.arrivals:
            if arrival.pick_id in match_pick_id.keys():
                arrival.pick_id = match_pick_id[arrival.pick_id]

    logger.debug(f"Deduplicate picks: to remove:{len(to_be_removed)}, remaining:{len(picks)}.")
    for p in to_be_removed:
        picks.remove(p)
    return event


def feed_distance_from_preloc_to_pref_origin(cat):
    for e in cat:
        pref_o = e.preferred_origin()
        distance = None
        for o in e.origins:
            if o.resource_id != pref_o.resource_id and "PyOcto" in o.method_id.id:
                distance, az, baz = gps2dist_azimuth(
                    o.latitude,
                    o.longitude,
                    pref_o.latitude,
                    pref_o.longitude,
                )
                # distance in meters, convert it to km
                distance = distance / 1000.0
                break
        if distance != None:
            e.comments.append(Comment(text='{"preloc_distance_km": %.2f}' % (distance)))


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

    # logger.info("Pick deduplication ...")
    # new_cat = Catalog()
    # for e in cat.events:
    #     new_e = deduplicate_picks(e)
    #     new_cat.events.append(new_e)

    new_cat = cat

    logger.info("Make readable ids ...")
    new_cat = make_readable_id(new_cat, "sihex", "quakeml:franceseisme.fr")
    print(new_cat)

    logger.info("Writing catalog ...")
    new_cat.write(args.outputfile, format="QUAKEML")
