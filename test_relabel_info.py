"""Relabel comments and deactivation reach the final (pass 3) origin (review B10j)."""
import json

from obspy import UTCDateTime
from obspy.core.event import Arrival, Comment, Event, Origin, Pick, ResourceIdentifier, WaveformStreamID

from dbclust.localization import NllLoc


def _event():
    t = UTCDateTime("2014-07-09T10:00:00")
    picks, arrivals = [], []
    for i, sta in enumerate(("A", "B", "C")):
        p = Pick(resource_id=ResourceIdentifier(f"p{i}"), time=t + i,
                 waveform_id=WaveformStreamID("FR", sta, "00", "HHZ"), phase_hint="P")
        picks.append(p)
        arrivals.append(Arrival(pick_id=p.resource_id, phase="P", time_weight=1.0))
    origin = Origin(arrivals=arrivals, time=t, latitude=0, longitude=0)
    return Event(picks=picks, origins=[origin]), origin, t


def _comment(action):
    return Comment(text=json.dumps({"relabel": {"action": action, "prev_phase": "Pg"}}))


def test_relabel_comments_and_deactivation_applied():
    event, origin, t = _event()
    relabel = {
        f"FR.A.00.HHZ-P-{t}": _comment("relabel"),
        f"FR.B.00.HHZ-P-{t + 1}": _comment("removed: already set"),
        f"FR.C.00.HHZ-P-{t + 2}": _comment("ignored: already set"),
    }
    locator = NllLoc.__new__(NllLoc)
    locator.enable_cleanup_pick_zone = True
    locator._apply_relabel_info(origin, event, relabel)

    a, b, c = origin.arrivals
    assert [len(x.comments) for x in (a, b, c)] == [1, 1, 1]
    assert (a.time_weight, b.time_weight, c.time_weight) == (1.0, 0, 1.0)
    # a copy, not the shared object
    assert a.comments[0] is not relabel[f"FR.A.00.HHZ-P-{t}"]


def test_relabel_info_noop_when_disabled_or_empty():
    event, origin, t = _event()
    locator = NllLoc.__new__(NllLoc)
    locator.enable_cleanup_pick_zone = False
    locator._apply_relabel_info(origin, event, {f"FR.A.00.HHZ-P-{t}": _comment("removed")})
    locator.enable_cleanup_pick_zone = True
    locator._apply_relabel_info(origin, event, {})
    assert all(not a.comments and a.time_weight == 1.0 for a in origin.arrivals)
