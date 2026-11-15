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
