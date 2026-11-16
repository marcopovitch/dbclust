"""Gap detection on events mixing manual and automatic picks.

Regression: pass-2 gap enforcement was skipped as soon as one pick was manual,
so an automatic pick ~985 km away survived in a swarm event (see
BUG_REPORT_gap_dist_bypass_pass3_2026-11-16.md).
"""
from obspy import UTCDateTime
from obspy.core.event import Arrival
from obspy.core.event import Event
from obspy.core.event import Origin
from obspy.core.event import Pick
from obspy.core.event import WaveformStreamID

from dbclust.constants import KM_PER_DEGREE
from dbclust.gap import get_arrival_with_distance_gap_greater_than
from dbclust.gap import get_station_count_before_distance_gap


def _event(specs):
    """specs: list of (station, distance_km, evaluation_mode)."""
    ev = Event()
    origin = Origin(time=UTCDateTime(2014, 7, 9, 11, 14, 21), latitude=42.85, longitude=-0.3)
    for sta, dist_km, mode in specs:
        pick = Pick(
            time=UTCDateTime(2014, 7, 9, 11, 14, 30),
            waveform_id=WaveformStreamID(network_code="XX", station_code=sta),
            phase_hint="P",
            evaluation_mode=mode,
        )
        ev.picks.append(pick)
        origin.arrivals.append(
            Arrival(pick_id=pick.resource_id, phase="P", distance=dist_km / KM_PER_DEGREE)
        )
    ev.origins.append(origin)
    ev.preferred_origin_id = origin.resource_id
    return ev


SPECS = [
    ("A", 20, "manual"),
    ("B", 60, "manual"),
    ("C", 120, "automatic"),
    ("D", 180, "manual"),
    ("FAR", 985, "automatic"),  # > 120 km after the last station
]


def test_automatic_arrival_beyond_gap_is_returned_with_manual_picks():
    ev = _event(SPECS)
    beyond = get_arrival_with_distance_gap_greater_than(ev, 120)
    assert [round(a.distance * KM_PER_DEGREE) for a in beyond] == [985]


def test_manual_arrival_beyond_gap_is_kept():
    ev = _event(SPECS[:-1] + [("FAR", 985, "manual")])
    assert get_arrival_with_distance_gap_greater_than(ev, 120) == []


def test_station_count_still_skipped_for_mixed_events():
    # the "reject if < 5 stations" rule keeps applying to fully automatic events only
    assert get_station_count_before_distance_gap(_event(SPECS), 120) is None
