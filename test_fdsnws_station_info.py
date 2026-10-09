#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Regression tests for BUGS_COHERENCE_REVIEW_2026-10-09.md, batch 2 (FDSNWS).

Covers:
- H6: the FDSNWS request is made with a timeout
- H4: a failed request (timeout, server error) is not cached by lru_cache,
  only remembered for FDSNWS_RETRY_AFTER_S; an unknown station (204/404) is
- C6: with info_sta=None (fetch_method: None) no FDSNWS request is attempted,
  the fallback CSV is used directly
"""
import logging
from types import SimpleNamespace

import pandas as pd
import pytest
import requests
from obspy import UTCDateTime

from dbclust import phase as phase_mod
from dbclust.phase import FdsnwsError
from dbclust.phase import Phase
from dbclust.phase import get_station_info_from_fdsnws

logging.disable(logging.WARNING)

URL = "http://fdsnws.test"
STATION_TEXT = (
    "#Network|Station|Location|Channel|Latitude|Longitude|Elevation|Depth|"
    "Azimuth|Dip|SensorDescription|Scale|ScaleFreq|ScaleUnits|SampleRate|"
    "StartTime|EndTime\n"
    "FR|STR||HHZ|48.5|7.7|150.0|0|0|-90|sensor|1|1|M/S|100|2010-01-01|\n"
)


@pytest.fixture(autouse=True)
def clean_caches():
    get_station_info_from_fdsnws.cache_clear()
    phase_mod._fdsnws_failures.clear()
    yield
    get_station_info_from_fdsnws.cache_clear()
    phase_mod._fdsnws_failures.clear()


@pytest.fixture
def fake_get(monkeypatch):
    """requests.get replacement returning the queued responses in order."""
    calls = []
    responses = []

    def get(url, timeout=None):
        calls.append(timeout)
        r = responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r

    monkeypatch.setattr(phase_mod.requests, "get", get)
    return SimpleNamespace(calls=calls, responses=responses)


def response(status, text=""):
    r = requests.Response()
    r.status_code = status
    r._content = text.encode()
    return r


def query(station="STR"):
    return get_station_info_from_fdsnws("FR", station, "2020-01-01", "2021-01-01", URL)


def test_success_is_cached_and_uses_timeout(fake_get):
    fake_get.responses.append(response(200, STATION_TEXT))

    lat, lon, *_ = query()
    assert (lat, lon) == pytest.approx((48.5, 7.7))
    query()

    assert fake_get.calls == [phase_mod.FDSNWS_TIMEOUT_S]


def test_unknown_station_is_cached(fake_get):
    fake_get.responses.append(response(204))

    assert query() == (None, None, None, None, None)
    query()

    assert len(fake_get.calls) == 1


def test_failure_not_cached_and_retried_after_delay(fake_get, monkeypatch):
    fake_get.responses += [requests.Timeout("slow"), response(200, STATION_TEXT)]
    now = [1000.0]
    monkeypatch.setattr(phase_mod.time, "monotonic", lambda: now[0])

    with pytest.raises(FdsnwsError):
        query()
    # within the retry delay: no new request
    with pytest.raises(FdsnwsError):
        query()
    assert len(fake_get.calls) == 1

    now[0] += phase_mod.FDSNWS_RETRY_AFTER_S + 1
    lat, *_ = query()
    assert lat == pytest.approx(48.5)
    assert len(fake_get.calls) == 2


def test_server_error_raises(fake_get):
    fake_get.responses.append(response(503, "unavailable"))
    with pytest.raises(FdsnwsError):
        query()


def test_info_sta_none_uses_fallback_without_request(monkeypatch):
    def no_request(*args, **kwargs):
        raise AssertionError("no FDSNWS request expected")

    monkeypatch.setattr(phase_mod.requests, "get", no_request)
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

    p = Phase(
        network="FR",
        station="STR",
        location="",
        channel="HHZ",
        phase="P",
        time=UTCDateTime(2020, 6, 1),
        time_uncertainty=0.1,
        proba=0.9,
        info_sta=None,
        fallback_df=fallback,
    )

    assert p.coord["latitude"] == pytest.approx(48.5)
    assert p.provenance == "fallback"
