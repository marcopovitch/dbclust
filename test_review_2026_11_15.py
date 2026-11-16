"""Regression tests for the fixes of BUGS_COHERENCE_REVIEW_2026-11-15.md."""
import os

import numpy as np
import pandas as pd
import pytest
from obspy import UTCDateTime

from dbclust.config import RenameConfig
from dbclust.config import Zone
from dbclust.config import Zones
from dbclust.core import unload_picks_list
from dbclust.rename import apply_regex_to_row
from dbclust.rename import rename_waveform_id


T0 = pd.Timestamp("2014-07-09T10:00:00.123456Z")


def _picks_df(rows):
    return pd.DataFrame(
        rows, columns=["station_id", "channel", "phase_type", "phase_time"]
    )


def test_unload_matches_phase_family_after_relabel():
    """B1: a pick relabelled Pg -> Pn in the event must still be unloaded."""
    df = _picks_df([["FR.OGMO", "00.HH", "Pg", T0], ["FR.URF", "00.HH", "S", T0]])
    picks = [["FR.OGMO.00.HHZ", "Pn", UTCDateTime(T0)]]
    out = unload_picks_list(df, picks)
    assert out["station_id"].tolist() == ["FR.URF"]
    assert list(out.columns) == list(df.columns)


def test_unload_does_not_mix_p_and_s():
    df = _picks_df([["FR.URF", "00.HH", "P", T0], ["FR.URF", "00.HH", "S", T0]])
    out = unload_picks_list(df, [["FR.URF.00.HHZ", "Sg", UTCDateTime(T0)]])
    assert out["phase_type"].tolist() == ["P"]


def test_unload_after_rename_removes_renamed_station():
    """B1: picks are renamed before the unload, so the renamed event pick matches."""
    rename = RenameConfig(before=[{r"^CH\.(VANN)\.([^.]*)\.(.*)": r"CH.VANNI.\2.\3"}])
    df = _picks_df([["CH.VANN", ".HH", "P", T0], ["FR.URF", "00.HH", "P", T0]])
    df = rename_waveform_id(df.copy(), rename)
    out = unload_picks_list(df, [["CH.VANNI..HHZ", "P", UTCDateTime(T0)]])
    assert out["station_id"].tolist() == ["FR.URF"]


def test_rename_matches_row_wise_reference():
    """P2: the per-distinct-pair implementation gives the row-wise result."""
    rename = RenameConfig(
        before=[
            {r"^(FR\..*?\..*?)\.XXZ": r"\1.SH"},
            {r"^CH\.(VANN)\.([^.]*)\.(.*)": r"CH.VANNI.\2.\3"},
        ],
        time_windows=[
            {
                "time_window": "1970-01-01/2016-01-01",
                "regex": [{r"FR\.LBL\.00\.SH.*": "FR.LBL..SH"}],
            },
            {
                "time_window": None,
                "regex": [
                    {r"FR\.ECHE\.00\.[SE]H.*": "FR.ECH..SH"},
                    {r"FR\.ECHZ\.00\.[SE]H.*": "FR.ECH..SH"},
                ],
            },
        ],
        after=[{r"^(FR\.ECH)\.\.SH$": r"\1..SHZ"}],
    )
    rng = np.random.default_rng(1)
    stations = ["FR.LBL", "FR.ECHE", "FR.ECHZ", "CH.VANN", "FR.URF", "FR.AGO"]
    channels = ["00.XXZ", "00.SHZ", "00.EHZ", ".HH", "00.HH"]
    n = 300
    df = pd.DataFrame(
        {
            "station_id": rng.choice(stations, n),
            "channel": rng.choice(channels, n),
            "phase_type": rng.choice(["P", "S"], n),
            "phase_time": [
                pd.Timestamp("2014-07-09T00:00:00Z") + pd.Timedelta(days=int(d))
                for d in rng.integers(0, 1500, n)
            ],
        }
    )

    # Row-wise reference: before, time windows, after — first match per step.
    expected = []
    for row in df.to_dict("records"):
        cur = {"station_id": row["station_id"], "channel": row["channel"]}
        steps = [rename.before]
        for tw in rename.time_windows:
            if tw["time_window"]:
                start, end = [pd.Timestamp(t, tz="UTC") for t in tw["time_window"].split("/")]
                if not (start <= row["phase_time"] <= end):
                    continue
            steps.append(tw["regex"])
        steps.append(rename.after)
        for patterns in steps:
            cur["station_id"], cur["channel"] = apply_regex_to_row(cur, patterns)
        expected.append((cur["station_id"], cur["channel"]))

    out = rename_waveform_id(df.copy(), rename)
    assert list(zip(out["station_id"], out["channel"])) == expected
    assert out.index.equals(df.index)


def test_find_zone_edge_distance_uses_lon_lat_order():
    """B10i: distance to the polygon edge, with (lon, lat) passed in that order."""
    zones = Zones(zones=[Zone(name="z", velocity_profile="p", polygon=[])])
    import geopandas as gpd
    from shapely.geometry import Polygon

    polygon = Polygon([[7.50, 48.40], [8.15, 48.40], [8.15, 49.00], [7.50, 49.00]])
    zones.polygons = gpd.GeoDataFrame([{"name": "z", "geometry": polygon}])
    # 0.10 degree of longitude east of the west edge at 48.7N: about 7.4 km
    _, distance_km = zones.find_zone(latitude=48.70, longitude=7.60)
    assert distance_km == pytest.approx(7.36, abs=0.2)


def test_suspicious_duplicates_sliding_window():
    """P7: close-in-time, close-in-space events with auto/manual conflicts are flagged."""
    import sqlite3

    from dbclust.parallel_import import detect_suspicious_duplicates

    conn = sqlite3.connect(":memory:")
    conn.executescript(
        """
        CREATE TABLE picks (event_id, station_name, phase_hint, pick_time, evaluation_mode);
        CREATE TABLE origins (event_id, time, latitude, longitude, used_phase_count, preferred);
        """
    )
    origins = [
        ("evB", "2014-07-09T10:00:02.000000Z", 48.01, 7.01, 8, 1),
        ("evA", "2014-07-09T10:00:00.000000Z", 48.00, 7.00, 9, 1),
        ("evC", "2014-07-09T10:00:30.000000Z", 48.00, 7.00, 9, 1),  # too late
        ("evD", "2014-07-09T10:00:01.000000Z", 45.00, 3.00, 9, 1),  # too far
    ]
    conn.executemany("INSERT INTO origins VALUES (?,?,?,?,?,?)", origins)
    picks = []
    for sta in ("S1", "S2", "S3"):
        picks.append(("evA", sta, "P", "2014-07-09T10:00:05.000000Z", "automatic"))
        picks.append(("evB", sta, "Pg", "2014-07-09T10:00:05.100000Z", "manual"))
        picks.append(("evC", sta, "P", "2014-07-09T10:00:05.100000Z", "manual"))
        picks.append(("evD", sta, "P", "2014-07-09T10:00:05.100000Z", "manual"))
    conn.executemany("INSERT INTO picks VALUES (?,?,?,?,?)", picks)

    suspicious = detect_suspicious_duplicates(conn)
    assert [(s["event_id_1"], s["event_id_2"]) for s in suspicious] == [("evA", "evB")]
    assert suspicious[0]["n_conflict_stations"] == 3
    assert suspicious[0]["dt_s"] == 2.0


def test_travel_time_grid_written_atomically(tmp_path):
    """B8: the PyOcto grid is renamed into place, no temp file is left behind."""
    from dbclust.config import PyoctoConfig
    from dbclust.config import VelocityModel

    vmodel = VelocityModel(
        depth=[0.0, 20.0], vp=[6.0, 8.0], vs=[3.5, 4.5], tolerance=2.0,
        grid_spacing_km=5.0, max_horizontal_dist_km=50.0, max_vertical_dist_km=20.0,
    )
    filename = os.path.join(tmp_path, "model")
    PyoctoConfig.create_travel_time_grid_file(None, vmodel, filename)
    assert os.path.exists(filename)
    assert os.listdir(tmp_path) == ["model"]


def test_tolerance_search_does_not_mutate_shared_associator(monkeypatch):
    """B6: the associator in cfg is shared between tasks and must stay untouched."""
    from types import SimpleNamespace

    import dbclust.dbclust2pyocto as d2p

    shared = SimpleNamespace(pick_match_tolerance=1.5)
    cfg = SimpleNamespace(
        pyocto=SimpleNamespace(
            current_model=SimpleNamespace(associator=shared),
            default_model_name="m",
            velocity_model=None,
            delegate_dbclust=False,
        ),
        cluster=SimpleNamespace(min_picks_common=3),
    )
    myclust = SimpleNamespace(clusters=[], noise=[], preloc=None, n_clusters=0)
    seen = []

    def fake_dbclust2pyocto(myclust_copy, model_name, associator, *args, **kwargs):
        # what another thread would read on the shared object during the search
        seen.append((shared.pick_match_tolerance, associator.pick_match_tolerance))
        if associator.pick_match_tolerance > 0.6:
            raise d2p.MultipleEventIDsWithSameAgencyError("conflict")
        return myclust_copy

    monkeypatch.setattr(d2p, "dbclust2pyocto", fake_dbclust2pyocto)
    result = d2p.adjust_associator_tolerance(
        myclust, cfg, tolerance_steps={1: 1, 0.5: 0.1, 0: 0.05}, min_tolerance=0.1
    )

    assert result is not None
    assert len(seen) > 1  # the search did lower the tolerance...
    assert all(shared_value == 1.5 for shared_value, _ in seen)  # ...on its copy only
    assert shared.pick_match_tolerance == 1.5


def test_finalize_sqlite_rerun_of_merged_run(tmp_path):
    """B10g: re-running an already merged run is not a failure."""
    from types import SimpleNamespace

    from dbclust.runner import finalize_sqlite

    final_db = tmp_path / "final.db"
    cfg = SimpleNamespace(
        catalog=SimpleNamespace(
            enable_sqlite=True,
            temp_db_dir=str(tmp_path),
            sqlite_db_path=str(tmp_path),
            sqlite_db_fullpath=str(final_db),
        )
    )
    assert finalize_sqlite(cfg, completed_job_indices={0}) is False  # nothing at all
    final_db.write_bytes(b"")
    assert finalize_sqlite(cfg, completed_job_indices={0}) is True


def test_job_picks_materialized_once(tmp_path):
    """P3: window queries give the same rows on the files and on the job table."""
    import duckdb

    from dbclust.core import _materialize_job_picks
    from dbclust.core import get_cross_partition_picks

    base = pd.Timestamp("2014-07-09T09:50:00")
    rows = []
    for i in range(400):
        t = base + pd.Timedelta(seconds=6 * i)
        rows.append((f"FR.S{i % 7}", "00.HH", "P" if i % 2 else "S", t, 0.4 + (i % 6) / 10,
                     "automatic", "PHASENET", None, "X", t.month, t.year))
    df = pd.DataFrame(rows, columns=["station_id", "channel", "phase_type", "phase_time",
                                     "phase_score", "phase_evaluation", "phase_method",
                                     "event_id", "agency", "month", "year"])
    path = str(tmp_path / "picks.parquet")
    df.to_parquet(path)

    start = pd.Timestamp("2014-07-09T10:00:00")
    overlap = pd.Timedelta(250, "s")

    def queries(con):
        backward = get_cross_partition_picks(con, start, overlap, base, 0.1, 0.2, 0.6, 0.5)
        window = con.sql(
            f"SELECT DISTINCT station_id, phase_type, phase_time, phase_score FROM PICKS "
            f"WHERE year = 2014 AND month = 7 AND phase_time BETWEEN '{start}' AND "
            f"'{start + pd.Timedelta(minutes=10) + overlap}' ORDER BY ALL"
        ).fetchdf()
        return backward.reset_index(drop=True), window

    def connect():
        con = duckdb.connect(":memory:")
        con.execute(f"CREATE VIEW PICKS AS SELECT * FROM read_parquet('{path}')")
        return con

    con = connect()
    expected = queries(con)
    con.close()

    con = connect()
    _materialize_job_picks(con, "parquet", start - overlap,
                           start + pd.Timedelta(minutes=10) + overlap)
    n_all = len(df)
    n_job = con.execute("SELECT COUNT(*) FROM PICKS").fetchone()[0]
    got = queries(con)
    con.close()

    assert 0 < n_job < n_all
    assert len(expected[0]) > 0 and len(expected[1]) > 0
    for a, b in zip(expected, got):
        pd.testing.assert_frame_equal(a, b)


def test_station_info_cache_follows_epochs():
    """P5: the cached lookup still returns the epoch active at the pick time."""
    from obspy.core.inventory import Channel, Inventory, Network, Station

    from dbclust.phase import _get_station_info_from_inventory_uncached
    from dbclust.phase import get_station_info_from_inventory

    def channel(start, end):
        return Channel(code="HHZ", location_code="00", latitude=0, longitude=0,
                       elevation=0, depth=0, sample_rate=100,
                       start_date=UTCDateTime(start), end_date=UTCDateTime(end) if end else None)

    old = Station(code="STA", latitude=48.0, longitude=7.0, elevation=100.0,
                  start_date=UTCDateTime("2000-01-01"), end_date=UTCDateTime("2012-01-01"),
                  channels=[channel("2000-01-01", "2012-01-01")])
    new = Station(code="STA", latitude=48.5, longitude=7.5, elevation=200.0,
                  start_date=UTCDateTime("2012-01-02"), channels=[channel("2012-01-02", None)])
    inv = Inventory(networks=[Network(code="FR", stations=[old, new])], source="test")

    for t in ("2005-06-01T00:00:00", "2012-01-01T00:00:00", "2012-01-01T00:00:01",
              "2014-07-09T10:00:00", "2005-06-01T00:00:00"):
        expected = _get_station_info_from_inventory_uncached("FR", "STA", t, inv, "00", "HH")
        got = get_station_info_from_inventory("FR", "STA", t, t, inv, loc="00", chan="HH")
        assert repr(got) == repr(expected), t

    lat_2005 = get_station_info_from_inventory(
        "FR", "STA", "2005-06-01T00:00:00", "", inv, loc="00", chan="HH")[0]
    lat_2014 = get_station_info_from_inventory(
        "FR", "STA", "2014-07-09T10:00:00", "", inv, loc="00", chan="HH")[0]
    assert float(lat_2005) == 48.0 and float(lat_2014) == 48.5
