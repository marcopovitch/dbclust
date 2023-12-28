#!/usr/bin/env python
import sys
import logging
from datetime import datetime
from itertools import combinations
import alphabetic_timestamp as ats
from obspy.core.event import ResourceIdentifier, Comment
from obspy.geodetics import gps2dist_azimuth

# default logger
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger("clusterize")
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


def make_readable_id(cat, prefix, smi_base):
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
        for p in e.picks:
            old_pick_id = p.resource_id.id
            pick_id = make_pick_id(e)
            p.resource_id = pick_id
            pick_lookup_table[old_pick_id] = pick_id
            for c in p.comments:
                comment_id = make_comment_id(p)
                c.resource_id = comment_id

        # forge readable origin_id
        for o in e.origins:
            origin_id = make_origin_id(e)
            if o.resource_id == e.preferred_origin_id:
                e.preferred_origin_id = origin_id.id
            o.resource_id = origin_id

            # forge readable arrival_id
            for a in o.arrivals:
                arrival_id = make_arrival_id(o)
                a.resource_id = arrival_id
                a.pick_id = pick_lookup_table[a.pick_id.id]
    return cat


def remove_duplicated_picks(event):
    picks = event.picks
    to_be_removed = []
    match_pick_id = {}
    for p1, p2 in combinations(picks, 2):
        if p1.resource_id == p2.resource_id or (
            p1.waveform_id.get_seed_string() == p2.waveform_id.get_seed_string()
            and p1.time == p2.time
            and p1.phase_hint == p2.phase_hint
        ):
            match_pick_id[p2.resource_id] = p1.resource_id
            to_be_removed.append(p2)

    logger.info(f"Removed {len(to_be_removed)}/{len(picks)} duplicated picks.")
    for p in to_be_removed:
        picks.remove(p)
    logger.info(f"Remaining {len(picks)} picks.")

    for origin in event.origins:
        for arrival in origin.arrivals:
            if arrival.pick_id in match_pick_id.keys():
                arrival.pick_id = match_pick_id[arrival.pick_id]
    return event


def feed_distance_from_preloc_to_pref_origin(cat):
    for e in cat:
        pref_o = e.preferred_origin()
        distance = None
        for o in e.origins:
            if o.resource_id != pref_o.resource_id and "pyocto" in o.method_id.id:
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
