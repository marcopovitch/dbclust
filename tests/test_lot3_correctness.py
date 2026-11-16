#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Regression tests for BUGS_COHERENCE_REVIEW_2026-10-09.md, batch 3.

Covers:
- H12: import_phases() applies the probability thresholds to the whole phase
  family (Pg/Pn, Sg/Sn), like the SQL pick queries in core.py
- H7: make_readable_id() accepts a magnitude without origin
- H13: deduplicate_picks() keeps the manual pick of a manual/automatic pair
"""
import logging

import pandas as pd
import pytest
from obspy import UTCDateTime
from obspy.core.event import Arrival
from obspy.core.event import Catalog
from obspy.core.event import Event
from obspy.core.event import Magnitude
from obspy.core.event import Origin
from obspy.core.event import Pick
from obspy.core.event import ResourceIdentifier
from obspy.core.event import WaveformStreamID

from dbclust.phase import import_phases
from dbclust.quakeml import deduplicate_picks
from dbclust.quakeml import make_readable_id

logging.disable(logging.WARNING)

T0 = UTCDateTime(2020, 6, 1)


# --- H12 -------------------------------------------------------------------------


def test_import_phases_filters_whole_phase_family():
    fallback = pd.DataFrame(
        [
            {
                "network": "FR",
                "station": "STR",
                "location": "00",
                "channel": "HHZ",
                "latitude": 48.5,
                "longitude": 7.7,
                "elevation": 150.0,
                "starttime": UTCDateTime(2010, 1, 1),
                "endtime": UTCDateTime(2030, 1, 1),
            }
        ]
    )
    rows = [
        ("P", 0.9),
        ("Pg", 0.0),  # below P threshold: dropped
        ("Pn", 0.9),
        ("S", 0.1),  # below S threshold: dropped
        ("Sg", 0.0),  # below S threshold: dropped
        ("Sn", 0.9),
    ]
    df = pd.DataFrame(
        {
            "station_id": ["FR.STR.00.HHZ"] * len(rows),
            "phase_type": [r[0] for r in rows],
            "phase_time": [T0 + i for i in range(len(rows))],
            "phase_score": [r[1] for r in rows],
        }
    )

    phases = import_phases(
        df, P_proba_threshold=0.5, S_proba_threshold=0.5, info_sta=None, fallback_df=fallback
    )

    assert sorted(p.phase for p in phases) == ["P", "Pn", "Sn"]


# --- H7 --------------------------------------------------------------------------


def test_make_readable_id_magnitude_without_origin():
    origin = Origin(time=T0, latitude=48.5, longitude=7.7)
    magnitude = Magnitude(mag=2.0)  # no origin_id
    event = Event(
        resource_id=ResourceIdentifier("smi:test/event/1"),
        origins=[origin],
        magnitudes=[magnitude],
        preferred_origin_id=origin.resource_id,
    )

    cat = make_readable_id(Catalog(events=[event]), "test", "smi:test")

    assert cat.events[0].magnitudes[0].origin_id is None


# --- H13 -------------------------------------------------------------------------


def make_pick(rid, mode, channel="HHZ"):
    return Pick(
        resource_id=ResourceIdentifier(rid),
        time=T0,
        phase_hint="P",
        evaluation_mode=mode,
        waveform_id=WaveformStreamID("FR", "STR", "00", channel),
    )


@pytest.mark.parametrize("order", [("auto", "manual"), ("manual", "auto")])
def test_deduplicate_picks_keeps_manual_pick(order):
    picks = {
        "auto": make_pick("smi:test/pick/auto", "automatic", "HHZ"),
        "manual": make_pick("smi:test/pick/manual", "manual", "HHN"),
    }
    arrivals = [Arrival(pick_id=picks[k].resource_id, phase="P") for k in order]
    event = Event(
        picks=[picks[k] for k in order],
        origins=[Origin(time=T0, arrivals=arrivals)],
    )

    deduplicate_picks(event)

    assert [p.resource_id.id for p in event.picks] == ["smi:test/pick/manual"]
    assert {a.pick_id.id for a in event.origins[0].arrivals} == {"smi:test/pick/manual"}
