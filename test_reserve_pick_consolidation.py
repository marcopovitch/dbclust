#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Unit tests for the reserve S-pick consolidation (plan v6).

Covers:
- config validation (validate_reserve_pick_config)
- NllLoc._consolidate_reserve_s_picks: acceptance, uniqueness rule,
  S-P coherence, no P completion, deactivated-S guard, single use
- weighted ps counting (_compute_ps_ratio, get_origin_station_score)
- Clusterize._has_reserve_s_candidates (pre-NLL bypass existence check)
- pick tag through the remap, final arrival tag, weighted post-NLL gate
- pool ingestion (component dedup, event_id stripping, noisy stations,
  cross-window dedup) and release/commit of consumed candidates
- inject_spatialite weighted counts
"""
import logging
import os

import pytest
from obspy import UTCDateTime
from obspy.core.event import Arrival
from obspy.core.event import Event
from obspy.core.event import Origin
from obspy.core.event import Pick
from obspy.core.event import WaveformStreamID

from dbclust.clusterize import Clusterize
from dbclust.config import ReservePickConfig
from dbclust.config import validate_reserve_pick_config
from dbclust.localization import NllLoc
from dbclust.localization import _reserve_pick_comment
from dbclust.phase import Phase

logging.disable(logging.WARNING)

T0 = UTCDateTime(2020, 1, 2, 3, 4, 5)
# Station ~5.5 km epicentral from the origin, origin depth 10 km
STATION_COORD = {"latitude": 45.05, "longitude": 5.0, "elevation": 500.0}
ORIGIN_LAT, ORIGIN_LON, ORIGIN_DEPTH_M = 45.0, 5.0, 10_000.0


def make_locator(**kwargs) -> NllLoc:
    params = dict(
        nll_bin="x",
        scat2latlon_bin="y",
        nll_times_path="/tmp",
        reserve_enabled=True,
        reserve_vp=6.0,
        reserve_vs=3.5,
        reserve_tt_tolerance_s=1.0,
        reserve_ps_diff_tolerance_s=0.5,
        reserve_ps_count_weight=0.5,
    )
    params.update(kwargs)
    return NllLoc(**params)


def make_pick(station: str, phase: str, time: UTCDateTime) -> Pick:
    net, sta = station.split(".")
    return Pick(
        time=time,
        waveform_id=WaveformStreamID(
            network_code=net, station_code=sta, channel_code="HHZ"
        ),
        phase_hint=phase,
    )


def make_event_with_origin(picks_with_weight) -> tuple:
    """Build (event, origin) from [(pick, phase, time_weight), ...]."""
    event = Event(picks=[p for p, _, _ in picks_with_weight])
    arrivals = []
    for pick, phase, weight in picks_with_weight:
        arrival = Arrival(
            pick_id=pick.resource_id,
            phase=phase,
            time_weight=weight,
        )
        arrivals.append(arrival)
    origin = Origin(
        time=T0,
        latitude=ORIGIN_LAT,
        longitude=ORIGIN_LON,
        depth=ORIGIN_DEPTH_M,
        arrivals=arrivals,
    )
    event.origins.append(origin)
    event.preferred_origin_id = origin.resource_id
    return event, origin


def make_reserve_phase(station: str, time: UTCDateTime, proba: float = 0.4) -> Phase:
    net, sta = station.split(".")
    return Phase(
        network=net,
        station=sta,
        location="",
        channel="HH",
        phase="S",
        time=time,
        time_uncertainty=0.2,
        proba=proba,
        info_sta=None,
        coord=dict(STATION_COORD),
    )


@pytest.fixture
def locator() -> NllLoc:
    return make_locator()


def predicted_times(locator: NllLoc) -> tuple:
    """(d_km, t_pred_S, t_pred_P) for the fixture geometry."""
    from dbclust.localization import gps2dist_azimuth
    from math import sqrt

    epi_m, _, _ = gps2dist_azimuth(
        ORIGIN_LAT, ORIGIN_LON, STATION_COORD["latitude"], STATION_COORD["longitude"]
    )
    d = sqrt((epi_m / 1000.0) ** 2 + (ORIGIN_DEPTH_M / 1000.0) ** 2)
    return d, T0 + d / locator.reserve_vs, T0 + d / locator.reserve_vp


# --------------------------------------------------------------------------
# Config validation
# --------------------------------------------------------------------------


def test_config_ok_when_disabled():
    # disabled: no cross-checks, whatever the other values
    validate_reserve_pick_config(
        ReservePickConfig(enable=False, S_proba_threshold_low=0.9), 0.5, False
    )


def test_config_reloc_mode_disables_with_warning(caplog):
    rp = ReservePickConfig(enable=True, S_proba_threshold_low=0.3)
    validate_reserve_pick_config(rp, 0.5, True, config_type="reloc")
    assert rp.enable is False


def test_config_requires_double_pass():
    with pytest.raises(ValueError, match="double_pass"):
        validate_reserve_pick_config(
            ReservePickConfig(enable=True, S_proba_threshold_low=0.3),
            0.5,
            double_pass=False,
        )


def test_config_band_below_threshold():
    with pytest.raises(ValueError, match="strictly lower"):
        validate_reserve_pick_config(
            ReservePickConfig(enable=True, S_proba_threshold_low=0.5),
            0.5,
            True,
        )


def test_config_ps_count_weight_range():
    with pytest.raises(ValueError, match="ps_count_weight"):
        validate_reserve_pick_config(
            ReservePickConfig(enable=True, S_proba_threshold_low=0.3, ps_count_weight=1.5),
            0.5,
            True,
        )


def test_config_tolerances_positive():
    with pytest.raises(ValueError, match="tolerance"):
        validate_reserve_pick_config(
            ReservePickConfig(enable=True, S_proba_threshold_low=0.3, tt_tolerance_s=0.0),
            0.5,
            True,
        )


# --------------------------------------------------------------------------
# Consolidation
# --------------------------------------------------------------------------


def test_consolidation_accepts_unique_candidate(locator):
    d, t_s, t_p = predicted_times(locator)
    p_pick = make_pick("FR.AAA", "P", t_p)
    event, origin = make_event_with_origin([(p_pick, "P", 1.0)])
    pool = [make_reserve_phase("FR.AAA", t_s - 0.2)]

    event = locator._consolidate_reserve_s_picks(event, origin, pool)

    assert len(event.picks) == 2
    s_pick = event.picks[-1]
    assert s_pick.phase_hint.upper().startswith("S")
    # tagged on the PICK (survives the remap deepcopy), not on an arrival
    info = _reserve_pick_comment(s_pick)
    assert info is not None and info["reserve_consolidated"] is True
    assert "predicted_tt" in info and "residual_at_accept" in info
    # hypocentral distance: ~5.56 km epicentral, 10 km deep -> ~11.44 km
    assert d == pytest.approx(11.44, abs=0.05)
    assert info["predicted_tt"] == pytest.approx(d / locator.reserve_vs, abs=1e-3)
    assert info["residual_at_accept"] == pytest.approx(-0.2, abs=1e-3)
    assert info["ps_count_weight"] == locator.reserve_ps_count_weight
    # no event_id contamination (author carries the event_id)
    assert not (s_pick.creation_info and s_pick.creation_info.author)
    # single use: consumed from the pool
    assert pool == []


def test_consolidation_uniqueness_zero_or_multiple(locator):
    d, t_s, t_p = predicted_times(locator)
    p_pick = make_pick("FR.AAA", "P", t_p)
    event, origin = make_event_with_origin([(p_pick, "P", 1.0)])

    # zero candidate in window
    pool = [make_reserve_phase("FR.AAA", t_s - 30.0)]
    event = locator._consolidate_reserve_s_picks(event, origin, pool)
    assert len(event.picks) == 1
    assert len(pool) == 1  # not consumed

    # two candidates in window -> ambiguous, skip (no "best" selection)
    pool = [make_reserve_phase("FR.AAA", t_s - 0.2), make_reserve_phase("FR.AAA", t_s + 0.2)]
    event = locator._consolidate_reserve_s_picks(event, origin, pool)
    assert len(event.picks) == 1
    assert len(pool) == 2


def test_consolidation_ps_diff_coherence(locator):
    d, t_s, t_p = predicted_times(locator)
    p_pick = make_pick("FR.AAA", "P", t_p)
    event, origin = make_event_with_origin([(p_pick, "P", 1.0)])

    # candidate matching the TT window but inconsistent with the observed
    # S-P difference (P moved far away from its predicted time)
    bad_candidate_time = t_s  # right window
    p_late = make_pick("FR.AAA", "P", t_p + 5.0)
    event_late, origin_late = make_event_with_origin([(p_late, "P", 1.0)])
    pool = [make_reserve_phase("FR.AAA", bad_candidate_time)]
    event_late = locator._consolidate_reserve_s_picks(event_late, origin_late, pool)
    assert len(event_late.picks) == 1
    assert len(pool) == 1


def test_consolidation_never_completes_P(locator):
    # station with an active S and no P must stay untouched
    s_pick = make_pick("FR.AAA", "S", T0 + 3.0)
    event, origin = make_event_with_origin([(s_pick, "S", 1.0)])
    # reserve pool would hold a P... but the pool only ever contains S
    # (ingestion guarantees it); the consolidation method must not attempt
    # anything on S-only stations regardless.
    pool = [make_reserve_phase("FR.AAA", T0 + 3.1)]
    event = locator._consolidate_reserve_s_picks(event, origin, pool)
    assert len(event.picks) == 1


def test_consolidation_skips_station_with_deactivated_S(locator):
    d, t_s, t_p = predicted_times(locator)
    p_pick = make_pick("FR.AAA", "P", t_p)
    s_deactivated = make_pick("FR.AAA", "S", t_s - 0.3)
    event, origin = make_event_with_origin(
        [(p_pick, "P", 1.0), (s_deactivated, "S", 0.0)]
    )
    pool = [make_reserve_phase("FR.AAA", t_s - 0.2)]
    event = locator._consolidate_reserve_s_picks(event, origin, pool)
    # no double S at the station: the deactivated S pick stays the only one
    assert len([p for p in event.picks if p.phase_hint.upper().startswith("S")]) == 1
    assert len(pool) == 1


def test_consolidation_single_use_across_events(locator):
    d, t_s, t_p = predicted_times(locator)
    pool = [make_reserve_phase("FR.AAA", t_s - 0.2)]

    p1 = make_pick("FR.AAA", "P", t_p)
    e1, o1 = make_event_with_origin([(p1, "P", 1.0)])
    e1 = locator._consolidate_reserve_s_picks(e1, o1, pool)

    # a second event at the same place/time cannot reuse the consumed pick
    p2 = make_pick("FR.AAA", "P", t_p)
    e2, o2 = make_event_with_origin([(p2, "P", 1.0)])
    e2 = locator._consolidate_reserve_s_picks(e2, o2, pool)
    assert len(e1.picks) == 2
    assert len(e2.picks) == 1


def test_consolidation_noop_when_disabled(locator):
    d, t_s, t_p = predicted_times(locator)
    p_pick = make_pick("FR.AAA", "P", t_p)
    event, origin = make_event_with_origin([(p_pick, "P", 1.0)])
    pool = [make_reserve_phase("FR.AAA", t_s)]
    disabled = make_locator(reserve_enabled=False)
    event = disabled._consolidate_reserve_s_picks(event, origin, pool)
    assert len(event.picks) == 1


# --------------------------------------------------------------------------
# Weighted ps counting
# --------------------------------------------------------------------------


def test_compute_ps_ratio_weighted(locator):
    d, t_s, t_p = predicted_times(locator)
    p1 = make_pick("FR.AAA", "P", t_p)
    s1 = make_pick("FR.AAA", "S", t_s)
    p2 = make_pick("FR.BBB", "P", t_p)
    s2 = make_pick("FR.BBB", "S", t_s)

    # native P+S station + reserve-completed P+S station
    import json
    from obspy.core.event import Comment

    s2.comments.append(
        Comment(text=json.dumps({"reserve_consolidated": True, "predicted_tt": 1.0, "residual_at_accept": 0.0}))
    )

    event, origin = make_event_with_origin(
        [(p1, "P", 1.0), (s1, "S", 1.0), (p2, "P", 1.0), (s2, "S", 1.0)]
    )
    ps_with_both, total, ratio, n_reserve = locator._compute_ps_ratio(event, origin)
    assert ps_with_both == pytest.approx(1.0 + locator.reserve_ps_count_weight)
    assert total == 2
    assert n_reserve == 1
    assert ratio == pytest.approx(ps_with_both / 2)


def test_compute_ps_ratio_zero_when_no_ps_station(locator):
    p1 = make_pick("FR.AAA", "P", T0 + 1.0)
    p2 = make_pick("FR.BBB", "P", T0 + 1.0)
    event, origin = make_event_with_origin([(p1, "P", 1.0), (p2, "P", 1.0)])
    ps_with_both, total, ratio, n_reserve = locator._compute_ps_ratio(event, origin)
    assert ps_with_both == 0
    assert n_reserve == 0


def test_station_score_counts_reserve_consistently(locator):
    d, t_s, t_p = predicted_times(locator)
    p1 = make_pick("FR.AAA", "P", t_p)
    s1 = make_pick("FR.AAA", "S", t_s)
    import json
    from obspy.core.event import Comment

    s1.comments.append(Comment(text=json.dumps({"reserve_consolidated": True})))
    event, origin = make_event_with_origin([(p1, "P", 1.0), (s1, "S", 1.0)])
    score = locator.get_origin_station_score(event, origin)
    # P full (1.0) + S weighted (ps_count_weight), NOT a full 2.0
    assert score == pytest.approx(1.0 + locator.reserve_ps_count_weight)


# --------------------------------------------------------------------------
# Pre-NLL bypass existence check (Clusterize)
# --------------------------------------------------------------------------


def make_clusterize(**kwargs) -> Clusterize:
    params = dict(
        phases=None,
        min_cluster_size=2,
        min_station_count=1,
        min_station_with_P_and_S=2,
        reserve_enabled=True,
        reserve_tt_tolerance_s=1.0,
        leiden_vp=6.0,
        leiden_vs=3.5,
        reserve_max_station_dist_km=100.0,
    )
    params.update(kwargs)
    return Clusterize(**params)


def test_has_reserve_s_candidates_in_window():
    clust = make_clusterize()
    p = Phase(
        network="FR",
        station="AAA",
        location="",
        channel="HH",
        phase="P",
        time=T0,
        time_uncertainty=0.1,
        proba=0.9,
        info_sta=None,
        coord=dict(STATION_COORD),
    )
    station_phase_sets = {"FR.AAA": {"P"}}
    # S-P delay bound for 100 km: 100*(1/3.5 - 1/6) ≈ 11.9 s
    pool = [make_reserve_phase("FR.AAA", T0 + 5.0)]
    clust.reserve_pool = pool
    assert clust._has_reserve_s_candidates([p], station_phase_sets) is True

    # out of the generous window -> no bypass
    clust.reserve_pool = [make_reserve_phase("FR.AAA", T0 + 60.0)]
    assert clust._has_reserve_s_candidates([p], station_phase_sets) is False

    # station already has an S -> not a P-only station
    station_phase_sets_ps = {"FR.AAA": {"P", "S"}}
    clust.reserve_pool = pool
    assert clust._has_reserve_s_candidates([p], station_phase_sets_ps) is False

    # empty pool
    clust.reserve_pool = []
    assert clust._has_reserve_s_candidates([p], station_phase_sets) is False


def make_p_phase(station: str, time: UTCDateTime) -> Phase:
    net, sta = station.split(".")
    return Phase(
        network=net,
        station=sta,
        location="",
        channel="HH",
        phase="P",
        time=time,
        time_uncertainty=0.1,
        proba=0.9,
        info_sta=None,
        coord=dict(STATION_COORD),
    )


def test_has_reserve_s_candidates_window_anchored_on_station_P():
    # S-P bound for 100 km ≈ 11.9 s, anchored on the station's OWN P —
    # not on the last pick of the cluster.
    clust = make_clusterize()
    p_aaa = make_p_phase("FR.AAA", T0)
    p_bbb = make_p_phase("FR.BBB", T0 + 30.0)  # far station, late P
    sets = {"FR.AAA": {"P"}, "FR.BBB": {"P"}}

    # 20 s after AAA's P: beyond the AAA S-P bound even though it is before
    # the last cluster pick
    clust.reserve_pool = [make_reserve_phase("FR.AAA", T0 + 20.0)]
    assert clust._has_reserve_s_candidates([p_aaa, p_bbb], sets) is False

    # S well before the station's P: physically impossible
    clust.reserve_pool = [make_reserve_phase("FR.BBB", T0 + 10.0)]
    assert clust._has_reserve_s_candidates([p_aaa, p_bbb], sets) is False

    clust.reserve_pool = [make_reserve_phase("FR.BBB", T0 + 35.0)]
    assert clust._has_reserve_s_candidates([p_aaa, p_bbb], sets) is True


def test_bypass_distance_in_km_not_pseudo_tt(monkeypatch):
    # max_search_dist is a pseudo travel time (s): the km bound is
    # max_search_dist * average_velocity unless set explicitly.
    from types import SimpleNamespace

    import dbclust.core as core

    class Anything(SimpleNamespace):
        """Config section stub: unset attributes read as None."""

        def __getattr__(self, name):
            return None

    captured = {}
    monkeypatch.setattr(core, "Clusterize", lambda **kw: captured.update(kw))
    cfg = SimpleNamespace(
        cluster=Anything(average_velocity=5, max_search_dist=20, use_umap=False),
        pick=Anything(),
        time=Anything(overlap_window=0),
        zones=None,
        reserve_pick=ReservePickConfig(enable=True),
    )
    core.get_clusterize_from_config(cfg)
    assert captured["reserve_max_station_dist_km"] == pytest.approx(100.0)

    cfg.reserve_pick = ReservePickConfig(enable=True, max_station_dist_km=42.0)
    core.get_clusterize_from_config(cfg)
    assert captured["reserve_max_station_dist_km"] == pytest.approx(42.0)


# --------------------------------------------------------------------------
# Consolidation uses the post-cleanup origin (active P only)
# --------------------------------------------------------------------------


def test_consolidation_skips_station_with_deactivated_P(locator):
    # gap_dist_max_km sets time_weight=0 on event2's arrivals; with
    # use_deactivated_arrivals=True the P pick stays in event2.picks.
    d, t_s, t_p = predicted_times(locator)
    p_pick = make_pick("FR.AAA", "P", t_p)
    event, origin = make_event_with_origin([(p_pick, "P", 0.0)])
    pool = [make_reserve_phase("FR.AAA", t_s - 0.2)]
    event = locator._consolidate_reserve_s_picks(event, origin, pool)
    assert len(event.picks) == 1
    assert len(pool) == 1


def test_consolidation_ps_anchor_is_active_P(locator):
    # a deactivated P far from the prediction must not be the S-P anchor
    d, t_s, t_p = predicted_times(locator)
    p_bad = make_pick("FR.AAA", "P", t_p - 5.0)
    p_good = make_pick("FR.AAA", "P", t_p)
    event, origin = make_event_with_origin([(p_bad, "P", 0.0), (p_good, "P", 1.0)])
    pool = [make_reserve_phase("FR.AAA", t_s - 0.2)]
    event = locator._consolidate_reserve_s_picks(event, origin, pool)
    assert len(event.picks) == 3
    assert pool == []


def test_consolidated_pick_carries_probability(locator):
    d, t_s, t_p = predicted_times(locator)
    p_pick = make_pick("FR.AAA", "P", t_p)
    event, origin = make_event_with_origin([(p_pick, "P", 1.0)])
    pool = [make_reserve_phase("FR.AAA", t_s - 0.2, proba=0.37)]
    event = locator._consolidate_reserve_s_picks(event, origin, pool)

    from dbclust.inject_spatialite import get_pick_probability

    assert get_pick_probability(event.picks[-1]) == pytest.approx(0.37)


# --------------------------------------------------------------------------
# Tag survives the remap; arrival tag on the final origin; event_id
# --------------------------------------------------------------------------


def make_quality_origin(event: Event, origin: Origin, rms: float = 0.2) -> None:
    from obspy.core.event import OriginQuality

    origin.quality = OriginQuality(standard_error=rms)


def test_reserve_tag_survives_remap(locator):
    d, t_s, t_p = predicted_times(locator)
    p_pick = make_pick("FR.AAA", "P", t_p)
    event2, origin2 = make_event_with_origin([(p_pick, "P", 1.0)])
    pool = [make_reserve_phase("FR.AAA", t_s - 0.2)]
    event2 = locator._consolidate_reserve_s_picks(event2, origin2, pool)
    reserve_pick = event2.picks[-1]

    # pass-2 origin: NLL builds new arrivals referencing event2's picks
    orig_pass2 = Origin(
        time=T0, latitude=ORIGIN_LAT, longitude=ORIGIN_LON, depth=ORIGIN_DEPTH_M,
        arrivals=[
            Arrival(pick_id=p_pick.resource_id, phase="P", time_weight=1.0),
            Arrival(pick_id=reserve_pick.resource_id, phase="S", time_weight=1.0),
        ],
    )
    # e = pass-1 event: it does NOT hold the reserve pick
    e = Event(picks=[make_pick("FR.AAA", "P", t_p)])
    NllLoc._remap_arrivals_to_existing_picks(orig_pass2, event2, e)

    cloned = [p for p in e.picks if _reserve_pick_comment(p) is not None]
    assert len(cloned) == 1
    assert orig_pass2.arrivals[1].pick_id == cloned[0].resource_id


def build_final_event(locator, native_ps: int, reserve_ps: int, p_only: int = 0):
    """Event whose preferred origin has native_ps native P+S stations,
    reserve_ps reserve-completed P+S stations and p_only P-only stations."""
    import json
    from obspy.core.event import Comment

    rows = []
    idx = 0
    for kind, n in (("native", native_ps), ("reserve", reserve_ps), ("p", p_only)):
        for _ in range(n):
            sta = f"FR.S{idx:02d}"
            idx += 1
            rows.append((make_pick(sta, "P", T0 + 2.0), "P", 1.0))
            if kind != "p":
                s = make_pick(sta, "S", T0 + 3.5)
                if kind == "reserve":
                    s.comments.append(Comment(text=json.dumps(
                        {"reserve_consolidated": True, "ps_count_weight": 0.5}
                    )))
                rows.append((s, "S", 1.0))
    event, origin = make_event_with_origin(rows)
    make_quality_origin(event, origin)
    return event, origin


def gate_locator(**kwargs):
    params = dict(min_station_score=None, min_station_with_P_and_S=2, p_only_max_rms=0.1)
    params.update(kwargs)
    return make_locator(**params)


def test_final_origin_arrivals_tagged_and_event_comment():
    import json
    from obspy import Catalog

    loc = gate_locator(min_station_with_P_and_S=1)
    event, origin = build_final_event(loc, native_ps=1, reserve_ps=1)
    final = loc.get_catalog_from_results([Catalog(events=[event])])
    assert len(final) == 1
    tagged = [
        a for a in origin.arrivals
        if any("reserve_consolidated" in c.text for c in a.comments)
    ]
    assert len(tagged) == 1 and tagged[0].phase == "S"
    decomposition = [
        json.loads(c.text)["reserve_ps"] for c in event.comments if "reserve_ps" in c.text
    ]
    assert decomposition == [{"native": 1, "reserve": 1, "weighted": 1.5}]


def test_weighted_gate_rejects_1_native_plus_1_reserve():
    from obspy import Catalog

    loc = gate_locator(min_station_with_P_and_S=2)
    event, _ = build_final_event(loc, native_ps=1, reserve_ps=1)
    assert len(loc.get_catalog_from_results([Catalog(events=[event])])) == 0

    event, _ = build_final_event(loc, native_ps=2, reserve_ps=0)
    assert len(loc.get_catalog_from_results([Catalog(events=[event])])) == 1


def test_sentinel_reserve_only_station_is_not_no_ps(caplog):
    from obspy import Catalog

    loc = gate_locator(min_station_with_P_and_S=2)
    # 0 native + 1 reserve (0.5): NOT "no P+S station at all" -> the
    # low-RMS P-only override (p_only_max_rms) must not apply
    event, origin = build_final_event(loc, native_ps=0, reserve_ps=1, p_only=3)
    origin.quality.standard_error = 0.05  # < p_only_max_rms
    logging.disable(logging.NOTSET)
    try:
        with caplog.at_level(logging.WARNING, logger="dbclust.localization"):
            final = loc.get_catalog_from_results([Catalog(events=[event])])
    finally:
        logging.disable(logging.WARNING)
    assert len(final) == 0
    assert "no P+S station at all" not in caplog.text

    # 0 P+S station at all + low RMS -> p_only_max_rms override
    event, origin = build_final_event(loc, native_ps=0, reserve_ps=0, p_only=4)
    origin.quality.standard_error = 0.05
    assert len(loc.get_catalog_from_results([Catalog(events=[event])])) == 1


def test_reserve_pick_has_no_event_id():
    import pandas as pd

    from dbclust.phase import import_phases

    df = pd.DataFrame({
        "station_id": ["FR.AAA.00"],
        "channel": ["HHE"],
        "phase_type": ["S"],
        "phase_time": [pd.Timestamp("2020-01-02T03:04:10", tz="UTC")],
        "phase_score": [0.4],
        "phase_evaluation": ["automatic"],
        "phase_method": ["PHASENET"],
        "event_id": ["smi:catalog/event/123"],
        "agency": ["RENASS"],
    })
    df_prep = _prepare(df)
    assert df_prep["event_id"].isna().all()
    phases = import_phases(df_prep, 0.0, 0.3, 0.1, 0.2, None, None)
    assert all(ph.event_id is None for ph in phases)


# --------------------------------------------------------------------------
# Pool ingestion: dedup, noisy stations, cross-window merge
# --------------------------------------------------------------------------


def _prepare(df, noisy=None, frequency_threshold=None):
    from types import SimpleNamespace

    import pandas as pd

    from dbclust.core import _prepare_reserve_df

    cfg = SimpleNamespace(
        station=SimpleNamespace(
            blacklist=None, frequency_threshold=frequency_threshold, rename=None
        ),
        pick=SimpleNamespace(P_proximity_threshold=0.1, S_proximity_threshold=0.2),
    )
    begin = pd.Timestamp("2020-01-02T03:00:00", tz="UTC")
    end = begin + pd.Timedelta("10min")
    return _prepare_reserve_df(df, cfg, begin, end, [], noisy)


def _reserve_df(rows):
    import pandas as pd

    return pd.DataFrame([
        {
            "station_id": sta,
            "channel": chan,
            "phase_type": "S",
            "phase_time": pd.Timestamp(t, tz="UTC"),
            "phase_score": score,
            "phase_evaluation": "automatic",
            "phase_method": "PHASENET",
            "event_id": None,
            "agency": "RENASS",
        }
        for sta, chan, t, score in rows
    ])


def test_pool_dedup_merges_components():
    df = _reserve_df([
        ("FR.AAA.00", "HHE", "2020-01-02T03:04:10.00", 0.40),
        ("FR.AAA.00", "HHN", "2020-01-02T03:04:10.05", 0.45),
        ("FR.AAA.00", "HHN", "2020-01-02T03:05:10.00", 0.40),
    ])
    out = _prepare(df)
    assert len(out) == 2


def test_pool_drops_stations_noisy_in_active_picks():
    df = _reserve_df([
        ("FR.AAA.00", "HHE", "2020-01-02T03:04:10", 0.4),
        ("FR.BBB.00", "HHE", "2020-01-02T03:04:10", 0.4),
    ])
    out = _prepare(df, noisy={"FR.BBB.00"})
    assert list(out["station_id"]) == ["FR.AAA.00"]


def test_merge_reserve_pool_dedups_overlapping_windows():
    from dbclust.core import merge_reserve_pool

    t = T0 + 10.0
    pool = [make_reserve_phase("FR.AAA", t)]
    # overlap zone re-imported by the next window (fresh Phase objects)
    new = [make_reserve_phase("FR.AAA", t), make_reserve_phase("FR.AAA", t + 30.0)]
    added = merge_reserve_pool(pool, new, consumed=None, time_tolerance=0.2)
    assert added == 1
    assert len(pool) == 2

    # uniqueness rule still works on the merged pool
    loc = make_locator()
    d, t_s, t_p = predicted_times(loc)
    pool = [make_reserve_phase("FR.AAA", t_s)]
    merge_reserve_pool(pool, [make_reserve_phase("FR.AAA", t_s)], None, 0.2)
    event, origin = make_event_with_origin([(make_pick("FR.AAA", "P", t_p), "P", 1.0)])
    event = loc._consolidate_reserve_s_picks(event, origin, pool)
    assert len(event.picks) == 2


def test_merge_reserve_pool_skips_consumed():
    from dbclust.core import merge_reserve_pool

    t = T0 + 10.0
    pool = []
    consumed = {"FR.AAA": [float(t.timestamp)]}
    added = merge_reserve_pool(pool, [make_reserve_phase("FR.AAA", t)], consumed, 0.2)
    assert added == 0 and pool == []


def test_prune_reserve_consumed(locator):
    locator.reserve_consumed["FR.AAA"].extend([float(T0), float(T0 + 100)])
    locator.prune_reserve_consumed(T0 + 50)
    assert locator.reserve_consumed["FR.AAA"] == [float(T0 + 100)]
    locator.prune_reserve_consumed(T0 + 200)
    assert "FR.AAA" not in locator.reserve_consumed


# --------------------------------------------------------------------------
# Release of unused / rejected reserve picks
# --------------------------------------------------------------------------


def test_release_when_no_event(locator):
    ph = make_reserve_phase("FR.AAA", T0 + 5)
    pool = []
    kept = locator._release_unused_reserve_picks(None, [ph], pool)
    assert kept == [] and pool == [ph]


def test_release_when_final_origin_drops_reserve_pick(locator):
    # e.g. pass2_fallback: preferred origin reverted to pass 1, pass-2
    # origin removed -> the reserve pick is referenced by no arrival
    from obspy import Catalog

    d, t_s, t_p = predicted_times(locator)
    p_pick = make_pick("FR.AAA", "P", t_p)
    event, origin = make_event_with_origin([(p_pick, "P", 1.0)])
    pool = [make_reserve_phase("FR.AAA", t_s - 0.2)]
    consumed = list(pool)
    event = locator._consolidate_reserve_s_picks(event, origin, pool)
    assert pool == []

    kept = locator._release_unused_reserve_picks(Catalog(events=[event]), consumed, pool)
    assert kept == []
    assert pool == consumed
    # the orphan reserve pick is removed from the event
    assert all(_reserve_pick_comment(p) is None for p in event.picks)


def test_kept_when_referenced_even_with_zero_weight(locator):
    from obspy import Catalog

    d, t_s, t_p = predicted_times(locator)
    p_pick = make_pick("FR.AAA", "P", t_p)
    event, origin = make_event_with_origin([(p_pick, "P", 1.0)])
    pool = [make_reserve_phase("FR.AAA", t_s - 0.2)]
    consumed = list(pool)
    event = locator._consolidate_reserve_s_picks(event, origin, pool)
    reserve_pick = event.picks[-1]
    origin.arrivals.append(
        Arrival(pick_id=reserve_pick.resource_id, phase="S", time_weight=0.0)
    )
    kept = locator._release_unused_reserve_picks(Catalog(events=[event]), consumed, pool)
    assert kept == consumed and pool == []


def test_localisations_commit_accepted_release_rejected(tmp_path):
    """End-to-end bookkeeping of get_localisations_from_nllobs_dir with a
    fake nll_localisation: accepted event -> committed to reserve_consumed,
    rejected event -> candidate back in the pool."""
    from obspy import Catalog

    # min_station_with_P_and_S disabled: the gate decision is driven by the
    # patched _ps_ratio_ok below
    loc = gate_locator(min_station_with_P_and_S=0)
    d, t_s, t_p = predicted_times(loc)
    for k in range(2):
        (tmp_path / f"cluster-{k}.obs").write_text("")

    stations = ["FR.AAA", "FR.BBB"]
    pool = [make_reserve_phase(sta, t_s - 0.2) for sta in stations]

    def fake_nll_localisation(nll_obs_file, picks=None, double_pass=False,
                              reserve_picks=None, **kwargs):
        k = int(os.path.basename(nll_obs_file).split("-")[1].split(".")[0])
        sta = stations[k]
        p_pick = make_pick(sta, "P", t_p)
        event, origin = make_event_with_origin([(p_pick, "P", 1.0)])
        event = loc._consolidate_reserve_s_picks(event, origin, reserve_picks)
        s_pick = event.picks[-1]
        origin.arrivals.append(Arrival(pick_id=s_pick.resource_id, phase="S", time_weight=1.0))
        make_quality_origin(event, origin)
        return Catalog(events=[event])

    loc.nll_localisation = fake_nll_localisation
    # post-NLL gate: accept the first event only
    calls = {"n": 0}

    def ps_ratio_ok(ps_with_both, ps_total):
        calls["n"] += 1
        return calls["n"] == 1

    loc._ps_ratio_ok = ps_ratio_ok
    catalog = loc.get_localisations_from_nllobs_dir(
        str(tmp_path), picks=None, append=False, reserve_picks=pool
    )

    assert len(catalog) == 1
    assert list(loc.reserve_consumed) == ["FR.AAA"]
    assert [f"{ph.network}.{ph.station}" for ph in pool] == ["FR.BBB"]


# --------------------------------------------------------------------------
# inject_spatialite: weighted counts coherent with the pipeline gate
# --------------------------------------------------------------------------


def test_inject_spatialite_weighted_counts():
    from dbclust.inject_spatialite import compute_ps_ratio_and_station_score
    from dbclust.inject_spatialite import get_reserve_pick_weight

    loc = gate_locator()
    event, origin = build_final_event(loc, native_ps=2, reserve_ps=1, p_only=1)
    ps_ratio, score, ps_count = compute_ps_ratio_and_station_score(event, origin)
    ps_with_both, total, ratio, _ = loc._compute_ps_ratio(event, origin)
    assert ps_count == pytest.approx(ps_with_both) == pytest.approx(2.5)
    assert ps_ratio == pytest.approx(ratio)
    assert score == pytest.approx(loc.get_origin_station_score(event, origin))
    assert get_reserve_pick_weight(make_pick("FR.X", "S", T0)) is None


def test_inject_spatialite_db_recompute_weighted(tmp_path):
    import sqlite3

    from dbclust.inject_spatialite import _fetch_station_phases
    from dbclust.inject_spatialite import _score_station_phases

    conn = sqlite3.connect(tmp_path / "t.db")
    cur = conn.cursor()
    cur.execute("CREATE TABLE picks (id TEXT, station_name TEXT)")  # old schema
    cur.execute("CREATE TABLE arrivals (origin_id TEXT, pick_id TEXT, name TEXT, time_weight REAL)")
    cur.executemany("INSERT INTO picks (id, station_name) VALUES (?, ?)", [
        ("p1", "FR.AAA"), ("s1", "FR.AAA"), ("p2", "FR.BBB"), ("s2", "FR.BBB"),
    ])
    cur.executemany("INSERT INTO arrivals VALUES ('o', ?, ?, 1.0)", [
        ("p1", "P"), ("s1", "S"), ("p2", "P"), ("s2", "S"),
    ])
    phases, weights = _fetch_station_phases(cur, "o")  # migrates the column
    assert weights == {}
    cur.execute("UPDATE picks SET reserve_weight = 0.5 WHERE id = 's2'")
    phases, weights = _fetch_station_phases(cur, "o")
    ps_ratio, score, ps_count = _score_station_phases(phases, weights)
    assert ps_count == pytest.approx(1.5)
    assert score == pytest.approx(2.0 + 1.5)
    conn.close()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
