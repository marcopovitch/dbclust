"""Regression tests for the event-ID merge guard and the event_id owner rule
(dbclust/dbclust2pyocto.py: EventIdMergeGuard, clusters_moveout_fraction,
cluster_merge, aggregate_pick_to_cluster_with_common_event_id;
dbclust/clusterize.py: preloc_pseudo_tt, Clusterize.cluster_merge_based_on_eventid;
dbclust/config.py: validate_eventid_merge_config).

Same convention as test_refuse_fragment_events.py: lightweight fakes, plain
assertions, runnable with `python test_eventid_merge_guard.py` or pytest.
"""
import datetime
from types import SimpleNamespace

import numpy as np

from dbclust.clusterize import Clusterize, preloc_pseudo_tt
from dbclust.config import validate_eventid_merge_config
from dbclust.dbclust2pyocto import (
    MOVEOUT_COMPATIBLE,
    MOVEOUT_INCOMPATIBLE,
    MOVEOUT_UNDETERMINED,
    EventIdMergeGuard,
    aggregate_pick_to_cluster_with_common_event_id,
    cluster_merge,
    clusters_moveout_fraction,
)


class FakeTime(float):
    """UTCDateTime-like: float() and .datetime."""

    @property
    def datetime(self):
        return datetime.datetime.fromtimestamp(float(self))


class FakePhase:
    def __init__(self, sta, phase, t, lat, lon, event_id=None, agency=None, net="XX"):
        self.network = net
        self.station = sta
        self.phase = phase
        self.time = FakeTime(t)
        self.coord = {"latitude": lat, "longitude": lon}
        self.event_id = event_id
        self.agency = agency

    def is_p(self):
        return self.phase[0].upper() == "P"

    def is_s(self):
        return self.phase[0].upper() == "S"

    def __repr__(self):
        return f"<{self.station}/{self.phase}/eid={self.event_id}>"


T0 = 1700000000.0
VP = 6.0


def _event(n, lat0, lon0, t0, event_id=None, agency=None, start=0):
    """n P picks of one earthquake at (lat0, lon0, t0): stations on a north
    line, arrival = t0 + distance / VP (always moveout-compatible)."""
    picks = []
    for i in range(n):
        lat = lat0 + 0.2 * (i + 1)
        dist_km = 0.2 * (i + 1) * 111.19
        picks.append(FakePhase(f"S{start + i}", "P", t0 + dist_km / VP, lat, lon0, event_id, agency))
    return picks


def _preloc(lat, lon, t):
    return {"latitude": lat, "longitude": lon, "time": datetime.datetime.fromtimestamp(t)}


FAR = (_preloc(47.44, 4.03, T0), _preloc(46.32, 7.18, T0 + 40))  # ~270 km, 40 s
CLOSE = (_preloc(47.44, 4.03, T0), _preloc(47.46, 4.05, T0 + 2))


def _guard(**kw):
    params = dict(max_tt_s=30.0, average_velocity=5.0, vp=VP, vs=3.5, min_fraction=0.9, min_picks=3)
    params.update(kw)
    return EventIdMergeGuard(**params)


# --- preloc_pseudo_tt -------------------------------------------------------

def test_preloc_pseudo_tt_burgundy_valais():
    dist_km, dt_s, tt_s = preloc_pseudo_tt(*FAR, 5.0)
    assert abs(dist_km - 270.4) < 1.0, dist_km
    assert dt_s == 40.0
    assert abs(tt_s - np.hypot(40.0, dist_km / 5.0)) < 1e-9
    assert 66 < tt_s < 69, tt_s


def test_preloc_pseudo_tt_missing_preloc():
    assert preloc_pseudo_tt(FAR[0], None, 5.0) is None
    assert preloc_pseudo_tt(FAR[0], {"latitude": 1.0}, 5.0) is None


# --- moveout verdict --------------------------------------------------------

def test_moveout_same_earthquake_is_compatible():
    a = _event(6, 45.0, 6.0, T0)
    b = _event(4, 45.0, 6.5, T0, start=10)
    frac, n_pairs, n1, n2 = clusters_moveout_fraction(a, b, VP, 3.5)
    assert frac == 1.0 and n_pairs == 24 and (n1, n2) == (6, 4)
    assert _guard().moveout_verdict(a, b)[0] == MOVEOUT_COMPATIBLE


def test_moveout_60s_offset_is_incompatible():
    a = _event(6, 45.0, 6.0, T0)
    b = _event(4, 45.0, 6.1, T0 + 60, start=10)
    assert _guard().moveout_verdict(a, b)[0] == MOVEOUT_INCOMPATIBLE


def test_moveout_two_picks_is_undetermined():
    """Pairs are a product: 8 x 2 = 16 pairs, but only 2 independent picks."""
    a = _event(8, 45.0, 6.0, T0)
    b = _event(2, 45.0, 6.5, T0, start=10)
    verdict, details = _guard().moveout_verdict(a, b)
    assert verdict == MOVEOUT_UNDETERMINED, details


def test_moveout_s_against_p_is_undetermined():
    a = _event(6, 45.0, 6.0, T0)
    b = [FakePhase(f"T{i}", "S", T0 + 30 + i, 45.5 + 0.1 * i, 6.5) for i in range(5)]
    frac, n_pairs, _, _ = clusters_moveout_fraction(a, b, VP, 3.5)
    assert frac is None and n_pairs == 0
    assert _guard().moveout_verdict(a, b)[0] == MOVEOUT_UNDETERMINED


# --- decision table ---------------------------------------------------------

def test_decision_close_preloc_merges_even_if_incompatible():
    a = _event(6, 45.0, 6.0, T0)
    b = _event(4, 45.0, 6.1, T0 + 60, start=10)
    allowed, _ = _guard().decide(a, b, *CLOSE)
    assert allowed  # the moveout never vetoes a close prelocation


def test_decision_missing_preloc_merges():
    a, b = _event(6, 45.0, 6.0, T0), _event(4, 45.0, 6.1, T0 + 60, start=10)
    assert _guard().decide(a, b, FAR[0], None)[0]


def test_decision_far_preloc_compatible_merges():
    a, b = _event(6, 45.0, 6.0, T0), _event(4, 45.0, 6.5, T0, start=10)
    allowed, details = _guard().decide(a, b, *FAR)
    assert allowed and MOVEOUT_COMPATIBLE in details


def test_decision_far_preloc_incompatible_refuses():
    a, b = _event(6, 45.0, 6.0, T0), _event(4, 45.0, 6.1, T0 + 60, start=10)
    assert not _guard().decide(a, b, *FAR)[0]


def test_decision_far_preloc_undetermined_refuses():
    a, b = _event(8, 45.0, 6.0, T0), _event(2, 45.0, 6.5, T0, start=10)
    assert not _guard().decide(a, b, *FAR)[0]


# --- both merge points ------------------------------------------------------

def _two_clusters_sharing_an_event_id(offset_s):
    big = _event(6, 45.0, 6.0, T0, event_id="E1", agency="SED")
    small = _event(4, 45.0, 6.1, T0 + offset_s, event_id="E1", agency="SED", start=10)
    return big, small


def test_cluster_merge_far_and_incompatible_is_refused():
    big, small = _two_clusters_sharing_an_event_id(60)
    merged, _ = cluster_merge([list(big), list(small)], list(FAR), 2,
                              eventid_shared_min_distinct_ids=1, guard=_guard())
    assert len(merged) == 2
    merged, _ = cluster_merge([list(big), list(small)], list(FAR), 2,
                              eventid_shared_min_distinct_ids=1, guard=None)
    assert len(merged) == 1  # guard disabled: previous behaviour


def test_cluster_merge_far_but_compatible_is_kept():
    big, small = _two_clusters_sharing_an_event_id(0)
    merged, _ = cluster_merge([list(big), list(small)], list(FAR), 2,
                              eventid_shared_min_distinct_ids=1, guard=_guard())
    assert len(merged) == 1


class _BareClusterize(Clusterize):
    def __init__(self):  # skip clustering: only cluster_merge_based_on_eventid is tested
        pass


def _clusterize(clusters, prelocs, max_tt_s):
    c = _BareClusterize()
    c.clusters = [list(x) for x in clusters]
    c.n_clusters = len(c.clusters)
    c.clusters_stability = [1.0] * c.n_clusters
    c.preloc = list(prelocs)
    c.min_com_phases = 2
    c.eventid_shared_min_picks_per_cluster = None
    c.eventid_shared_min_distinct_ids = 1
    c.average_velocity = 5.0
    c.eventid_merge_max_tt_s = max_tt_s
    c.eventid_merge_moveout_min_fraction = 0.9
    c.eventid_merge_moveout_min_picks = 3
    c.eventid_merge_moveout_vp = VP
    c.eventid_merge_moveout_vs = 3.5
    return c


def test_merge_based_on_eventid_applies_the_same_guard():
    big, small = _two_clusters_sharing_an_event_id(60)
    c = _clusterize([big, small], FAR, 30.0)
    c.cluster_merge_based_on_eventid()
    assert c.n_clusters == 2 and len(c.preloc) == 2

    c = _clusterize([big, small], FAR, None)
    c.cluster_merge_based_on_eventid()
    assert c.n_clusters == 1 and len(c.preloc) == 1

    big, small = _two_clusters_sharing_an_event_id(0)
    c = _clusterize([big, small], FAR, 30.0)
    c.cluster_merge_based_on_eventid()
    assert c.n_clusters == 1


# --- event_id owner rule ----------------------------------------------------

def _owner_case():
    big = _event(44, 43.0, 0.0, T0, event_id="LDG1", agency="LDG")
    big_other = _event(6, 43.0, 0.5, T0, event_id="IGN1", agency="IGN", start=100)
    small = _event(5, 44.0, 2.0, T0, event_id="LDG1", agency="LDG", start=200)
    small_other = _event(4, 44.0, 2.5, T0, start=300)
    noise = _event(3, 42.0, 0.0, T0, event_id="LDG1", agency="LDG", start=400)
    pool = big + big_other + small + small_other + noise
    return [big + big_other, small + small_other], pool


def test_owner_rule_no_pick_in_two_clusters():
    clusters, pool = _owner_case()
    out = aggregate_pick_to_cluster_with_common_event_id(clusters, pool, pick_count_threshold=3)
    ids = [{id(p) for p in c} for c in out]
    assert not ids[0] & ids[1]
    ldg = [sum(p.event_id == "LDG1" for p in c) for c in out]
    assert ldg == [52, 0], ldg  # all picks of the event follow its owner


def test_owner_rule_independent_of_cluster_order():
    clusters, pool = _owner_case()
    out = aggregate_pick_to_cluster_with_common_event_id(clusters[::-1], pool, pick_count_threshold=3)
    assert [sum(p.event_id == "LDG1" for p in c) for c in out] == [0, 52]
    assert not {id(p) for p in out[0]} & {id(p) for p in out[1]}


# --- config validation ------------------------------------------------------

def _cluster_cfg(**kw):
    params = dict(eventid_merge_max_tt_s=30.0, eventid_merge_moveout_min_fraction=0.9,
                  eventid_merge_moveout_min_picks=3, eventid_merge_moveout_vp=6.0,
                  eventid_merge_moveout_vs=3.5)
    params.update(kw)
    return SimpleNamespace(**params)


def test_config_validation():
    validate_eventid_merge_config(_cluster_cfg())
    validate_eventid_merge_config(_cluster_cfg(eventid_merge_max_tt_s=None))
    validate_eventid_merge_config(None)
    for bad in (
        dict(eventid_merge_max_tt_s=0),
        dict(eventid_merge_moveout_min_fraction=0),
        dict(eventid_merge_moveout_min_fraction=1.5),
        dict(eventid_merge_moveout_min_picks=0),
        dict(eventid_merge_moveout_vs=6.0),
    ):
        try:
            validate_eventid_merge_config(_cluster_cfg(**bad))
        except ValueError:
            continue
        raise AssertionError(f"accepted invalid config {bad}")


if __name__ == "__main__":
    tests = [obj for name, obj in list(globals().items()) if name.startswith("test_")]
    failures = []
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except AssertionError as e:
            failures.append(t.__name__)
            print(f"FAIL {t.__name__}: {e}")
    print(f"\n{len(tests) - len(failures)}/{len(tests)} passed")
    if failures:
        raise SystemExit(1)
