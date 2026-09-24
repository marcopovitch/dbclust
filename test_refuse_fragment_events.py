"""Regression tests for the moveout fragment re-fusion logic in
dbclust/dbclust2pyocto.py (_refuse_fragment_events, _dedup_picks_prefer_event_id,
_haversine_km, _tolerance_candidates).

These use lightweight fakes instead of real Phase/Clusterize objects so they
run without ObsPy/PyOcto/NonLinLoc — matches the rest of this repo's
test_*.py convention (plain assertions, runnable directly with `python
test_refuse_fragment_events.py`; also discoverable by pytest if introduced
later, since every check lives in a test_* function).
"""
import datetime

from dbclust.dbclust2pyocto import (
    _dedup_picks_prefer_event_id,
    _haversine_km,
    _refuse_fragment_events,
    _tolerance_candidates,
)


class FakePhase:
    def __init__(self, net, sta, phase, t, lat, lon, event_id=None):
        self.network = net
        self.station = sta
        self.phase = phase
        self.time = t
        self.coord = {"latitude": lat, "longitude": lon}
        self.event_id = event_id

    def is_p(self):
        return self.phase[0].upper() == "P"

    def is_s(self):
        return self.phase[0].upper() == "S"

    def __repr__(self):
        return f"<{self.station}/{self.phase}/eid={self.event_id}>"


class FakeTime(float):
    """Minimal stand-in for dbclust's Phase.time (UTCDateTime-like): supports
    float() and .datetime, which is all _refuse_fragment_events/dedup need.
    """

    @property
    def datetime(self):
        return datetime.datetime.fromtimestamp(float(self))


T0 = 1700000000.0


def _station_ring(n, t0, start_idx=0):
    """n picks on n distinct stations, all consistent with a single source
    near (0, 0) at time t0 (tiny lat/lon/time offsets, well within any
    reasonable vp/vs moveout bound)."""
    return [
        FakePhase("XX", f"ST{start_idx + i}", "P", FakeTime(t0 + i * 0.01), 0.01 * i, 0.01 * i)
        for i in range(n)
    ]


def test_fragment_fuses_into_compatible_larger_event():
    """A small, moveout-compatible fragment must be fused into the larger
    event it split off from."""
    large = _station_ring(10, T0)
    small = _station_ring(3, T0, start_idx=10)
    clusters = [small, large]
    prelocs = [
        {"time": datetime.datetime.fromtimestamp(T0 + 0.05)},
        {"time": datetime.datetime.fromtimestamp(T0)},
    ]

    n_fused = _refuse_fragment_events(
        clusters, prelocs, vp=6.0, vs=3.5, min_station_count=5, max_search_dist=8.0
    )

    assert n_fused == 1
    assert len(clusters) == 1
    assert len(clusters[0]) == 13


def test_distinct_nearby_event_is_not_fused():
    """A genuinely separate event, close in time but geometrically
    incompatible with the moveout bound, must not be merged in."""
    large = _station_ring(10, T0)
    distinct = [
        FakePhase(
            "XX", f"ST{10 + i}", "P", FakeTime(T0 + 50 + i * 0.01), 5.0 + 0.01 * i, 5.0 + 0.01 * i
        )
        for i in range(3)
    ]
    clusters = [distinct, large]
    prelocs = [
        {"time": datetime.datetime.fromtimestamp(T0 + 50)},
        {"time": datetime.datetime.fromtimestamp(T0)},
    ]

    n_fused = _refuse_fragment_events(
        clusters, prelocs, vp=6.0, vs=3.5, min_station_count=5, max_search_dist=8.0
    )

    assert n_fused == 0
    assert len(clusters) == 2


def test_fragment_larger_than_frag_max_st_is_not_fused():
    """frag_max_st (derived as min_station_count + 1) must still gate: a
    "fragment" as large as the target event is not a fragment."""
    large = _station_ring(10, T0)
    not_a_fragment = _station_ring(10, T0, start_idx=10)
    clusters = [not_a_fragment, large]
    prelocs = [
        {"time": datetime.datetime.fromtimestamp(T0)},
        {"time": datetime.datetime.fromtimestamp(T0)},
    ]

    n_fused = _refuse_fragment_events(
        clusters, prelocs, vp=6.0, vs=3.5, min_station_count=5, max_search_dist=8.0
    )

    assert n_fused == 0
    assert len(clusters) == 2


def test_too_few_cross_pairs_blocks_fusion():
    """min_pairs_checked guards against deciding compatibility from a single
    coincidentally-compatible pair."""
    large = _station_ring(10, T0)
    # Single-pick fragment: at most 1 cross pair, below the default
    # min_pairs_checked=3 threshold.
    single = [FakePhase("XX", "ST99", "P", FakeTime(T0), 0.0, 0.0)]
    clusters = [single, large]
    prelocs = [
        {"time": datetime.datetime.fromtimestamp(T0)},
        {"time": datetime.datetime.fromtimestamp(T0)},
    ]

    n_fused = _refuse_fragment_events(
        clusters, prelocs, vp=6.0, vs=3.5, min_station_count=5, max_search_dist=8.0
    )

    assert n_fused == 0
    assert len(clusters) == 2


def test_empty_clusters_is_a_noop():
    n_fused = _refuse_fragment_events(
        [], [], vp=6.0, vs=3.5, min_station_count=5, max_search_dist=8.0
    )
    assert n_fused == 0


def test_dedup_prefers_pick_with_event_id_regardless_of_order():
    t = FakeTime(T0)
    no_id = FakePhase("XX", "ST1", "P", t, 0.0, 0.0, event_id=None)
    with_id = FakePhase("XX", "ST1", "P", t, 0.0, 0.0, event_id="evt123")

    result_a = _dedup_picks_prefer_event_id([no_id, with_id])
    result_b = _dedup_picks_prefer_event_id([with_id, no_id])

    assert len(result_a) == 1 and result_a[0].event_id == "evt123"
    assert len(result_b) == 1 and result_b[0].event_id == "evt123"


def test_dedup_keeps_distinct_picks():
    t = FakeTime(T0)
    a = FakePhase("XX", "ST1", "P", t, 0.0, 0.0)
    b = FakePhase("XX", "ST2", "P", t, 0.0, 0.0)  # different station -> distinct

    result = _dedup_picks_prefer_event_id([a, b])

    assert len(result) == 2


def test_haversine_zero_distance():
    assert abs(_haversine_km(45.0, 5.0, 45.0, 5.0)) < 1e-9


def test_haversine_known_distance():
    # Strasbourg -> Paris, ~400 km great-circle.
    d = _haversine_km(48.5734, 7.7521, 48.8566, 2.3522)
    assert 390.0 < d < 410.0


def test_haversine_handles_identical_antipodal_edge_case():
    # Same point at the pole: must not raise even with float rounding.
    assert _haversine_km(90.0, 0.0, 90.0, 0.0) == 0.0


def test_tolerance_candidates_descend_from_original_to_min():
    candidates = _tolerance_candidates(2.0, {1: 0.5, 0: 0.1}, 0.5)
    assert candidates == [2.0, 1.5, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
    assert candidates == sorted(candidates, reverse=True)
    assert candidates[-1] >= 0.5


def test_tolerance_candidates_single_step():
    candidates = _tolerance_candidates(1.0, {0: 0.5}, 0.5)
    assert candidates == [1.0, 0.5]


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
