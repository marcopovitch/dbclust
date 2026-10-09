#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Same-station P-S pairs in Clusterize.numpy_compute_tt_matrix_vectorized.

With vp/vs/max_search_dist (HDBSCAN), a P-S pair of the same station keeps its
raw distance |S-P| when the delay is physically possible, and is pushed apart
(TT_FAR) when S precedes P or S-P exceeds
max_search_dist * vmean * (1/vs - 1/vp). Formerly every such pair was capped to
distance 0, linking P and S picks of distinct events (see
BUGS_COHERENCE_REVIEW_2026-10-09.md, TT1).
"""
import logging

import pytest
from obspy import UTCDateTime

from dbclust.clusterize import TT_FAR
from dbclust.clusterize import Clusterize
from dbclust.phase import Phase

logging.disable(logging.WARNING)

T0 = UTCDateTime(2020, 1, 1)
VMEAN, VP, VS, MAX_SEARCH_DIST = 5.0, 5.5, 3.2, 70
SP_MAX = MAX_SEARCH_DIST * VMEAN * (1 / VS - 1 / VP)  # ~45.7 s


def phase(ph, dt, station="S0"):
    return Phase(
        network="FR", station=station, location="", channel="HHZ", phase=ph,
        time=T0 + dt, time_uncertainty=0.1, proba=0.9, info_sta=None,
        coord={"latitude": 48.0, "longitude": 7.0, "elevation": 0.0},
    )


def tt(phases, **kw):
    return Clusterize.numpy_compute_tt_matrix_vectorized(phases, VMEAN, **kw)


CAP = dict(vp=VP, vs=VS, max_search_dist=MAX_SEARCH_DIST)


@pytest.mark.parametrize(
    "s_minus_p, expected",
    [
        (20.0, 20.0),  # plausible delay: raw distance, not 0
        (600.0, TT_FAR),  # beyond sp_max: distinct events
        (-5.0, TT_FAR),  # S before P
    ],
)
def test_same_station_ps_pair(s_minus_p, expected):
    m = tt([phase("P", 0.0), phase("S", s_minus_p)], **CAP)
    assert m[0, 1] == pytest.approx(expected)
    assert m[1, 0] == pytest.approx(expected)


def test_bound_derived_from_max_search_dist():
    assert SP_MAX == pytest.approx(45.7, abs=0.1)
    inside = tt([phase("P", 0.0), phase("S", SP_MAX - 1)], **CAP)
    outside = tt([phase("P", 0.0), phase("S", SP_MAX + 1)], **CAP)
    assert inside[0, 1] == pytest.approx(SP_MAX - 1)
    assert outside[0, 1] == TT_FAR


def test_other_pairs_untouched():
    phases = [phase("P", 0.0), phase("P", 600.0), phase("S", 600.0, station="S1")]
    assert (tt(phases, **CAP) == tt(phases)).all()


def test_no_cap_without_velocities():
    # Leiden: vp/vs not passed, raw distance
    m = tt([phase("P", 0.0), phase("S", 600.0)])
    assert m[0, 1] == pytest.approx(600.0)
