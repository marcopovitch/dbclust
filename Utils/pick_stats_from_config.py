#!/usr/bin/env python3
"""Compute pick statistics (manual/automatic counts, station count) from the
parquet pick files defined in a DBClust YAML configuration, applying the same
P/S probability thresholds, station blacklist, and rename rules used by the pipeline.

Usage:
    python Utils/pick_stats_from_config.py -c /path/to/dbclust-config.yml
    python Utils/pick_stats_from_config.py -c /path/to/dbclust-config.yml \\
        --stations-csv stations.csv \\
        --missing-stations-csv missing.csv \\
        --suggest-rename-yaml suggested_renames.yml
"""

import argparse
import re
from pathlib import Path

import duckdb
import pandas as pd

from dbclust.config import DBClustConfig
from dbclust.phase import extract_station_coords
from dbclust.rename import rename_waveform_id


def main():
    parser = argparse.ArgumentParser(description="Pick statistics from a DBClust YAML config")
    parser.add_argument("-c", "--config", required=True, help="Path to DBClust YAML config")
    parser.add_argument(
        "--stations-csv",
        metavar="FILE",
        default=None,
        help="Export unique stations with coordinates to this CSV file",
    )
    parser.add_argument(
        "--missing-stations-csv",
        metavar="FILE",
        default=None,
        help="Export stations found in picks but missing from the inventory/fallback",
    )
    parser.add_argument(
        "--suggest-rename-yaml",
        metavar="FILE",
        default=None,
        help="Write suggested rename rules (for missing stations found under another network) to this YAML file",
    )
    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve()

    full_cfg = DBClustConfig(str(config_path))

    pick_cfg = full_cfg.pick
    station_cfg = full_cfg.station

    # filenames already contain glob patterns like /path/to/picks.pq/**/*.parquet
    glob_patterns = pick_cfg.filenames
    p_thr = pick_cfg.P_proba_threshold
    s_thr = pick_cfg.S_proba_threshold
    blacklist = station_cfg.blacklist or []

    print(f"P_proba_threshold = {p_thr}")
    print(f"S_proba_threshold = {s_thr}")
    print(f"blacklist patterns = {blacklist}")
    print()

    con = duckdb.connect()

    total_manual = 0
    total_auto = 0
    stations = set()
    picks_per_station = {}         # (network, station) -> pick count
    manual_per_station = {}        # (network, station) -> manual pick count
    auto_per_station = {}          # (network, station) -> automatic pick count
    agencies_per_station = {}      # (network, station) -> set of agencies

    blacklist_re = [re.compile(b) for b in blacklist]

    for glob_pattern in glob_patterns:
        root_dir = Path(glob_pattern).parent.parent  # strip /**/*.parquet
        if not root_dir.exists():
            print(f"{root_dir.name}: MISSING, skipped")
            continue

        rows = con.execute(
            f"""
            SELECT station_id, phase_evaluation, agency, COUNT(*) AS n
            FROM read_parquet('{glob_pattern}', hive_partitioning=1)
            WHERE
                (phase_type IN ('P', 'Pg', 'Pn') AND phase_score >= {p_thr})
                OR
                (phase_type IN ('S', 'Sg', 'Sn') AND phase_score >= {s_thr})
            GROUP BY station_id, phase_evaluation, agency
            """
        ).fetchall()

        file_manual = 0
        file_auto = 0
        file_stations = set()

        for station_id, evaluation, agency, n in rows:
            if any(b.search(station_id) for b in blacklist_re):
                continue
            file_stations.add(station_id)
            parts = station_id.split(".")
            key = (parts[0], parts[1]) if len(parts) >= 2 else (None, station_id)
            picks_per_station[key] = picks_per_station.get(key, 0) + n
            if agency:
                agencies_per_station.setdefault(key, set()).add(agency)
            if evaluation == "manual":
                manual_per_station[key] = manual_per_station.get(key, 0) + n
                file_manual += n
            else:
                auto_per_station[key] = auto_per_station.get(key, 0) + n
                file_auto += n

        total_manual += file_manual
        total_auto += file_auto
        stations |= file_stations

        print(f"{root_dir.name}: manual={file_manual}, auto={file_auto}, stations={len(file_stations)}")

    # Apply rename rules (same as the pipeline) to normalise (network, station) keys
    if station_cfg.rename and picks_per_station:
        rename_input = pd.DataFrame(
            [(f"{net}.{sta}", "00.HHZ", pd.NaT) for net, sta in picks_per_station],
            columns=["station_id", "channel", "phase_time"],
        )
        renamed = rename_waveform_id(rename_input, station_cfg.rename)

        new_picks = {}
        new_manual = {}
        new_auto = {}
        new_agencies = {}
        for old_key, new_sid in zip(picks_per_station, renamed["station_id"]):
            new_key = tuple(new_sid.split(".")[:2])
            new_picks[new_key] = new_picks.get(new_key, 0) + picks_per_station[old_key]
            new_manual[new_key] = new_manual.get(new_key, 0) + manual_per_station.get(old_key, 0)
            new_auto[new_key] = new_auto.get(new_key, 0) + auto_per_station.get(old_key, 0)
            new_agencies.setdefault(new_key, set()).update(agencies_per_station.get(old_key, set()))
        picks_per_station = new_picks
        manual_per_station = new_manual
        auto_per_station = new_auto
        agencies_per_station = new_agencies
        stations = {f"{net}.{sta}" for net, sta in picks_per_station}

    print()
    print(f"Total manual picks:    {total_manual}")
    print(f"Total automatic picks: {total_auto}")
    print(f"Total picks:           {total_manual + total_auto}")
    print(f"Distinct stations:     {len(stations)}")

    if args.stations_csv or args.missing_stations_csv or args.suggest_rename_yaml:
        coords_df = extract_station_coords(
            inventory=station_cfg.inventory,
            fallback_df=station_cfg.fallback_df,
        )
        # Build (network, station) pairs from (renamed) pick station_ids
        station_ids = set(picks_per_station.keys())

        known_ids = set(zip(coords_df["network"], coords_df["station"]))
        missing_ids = sorted(station_ids - known_ids)

        if args.stations_csv:
            filtered = coords_df[
                coords_df.apply(lambda r: (r["network"], r["station"]) in station_ids, axis=1)
            ].copy()
            filtered["manual_picks"] = filtered.apply(
                lambda r: manual_per_station.get((r["network"], r["station"]), 0), axis=1
            )
            filtered["auto_picks"] = filtered.apply(
                lambda r: auto_per_station.get((r["network"], r["station"]), 0), axis=1
            )
            filtered = filtered.sort_values(["network", "station"]).reset_index(drop=True)
            out_path = Path(args.stations_csv)
            filtered.to_csv(out_path, index=False)
            print(f"\nStation coordinates written to {out_path} ({len(filtered)} stations)")

        # For each missing station, look for the same station name under a different network
        suggestions = {}  # (net, sta) -> list of candidate correct networks
        for net, sta in missing_ids:
            candidates = coords_df[coords_df["station"] == sta]["network"].tolist()
            other_nets = [n for n in candidates if n != net]
            if other_nets:
                suggestions[(net, sta)] = other_nets

        if missing_ids:
            print(f"\nStations in picks but missing from inventory/fallback: {len(missing_ids)}")

        if args.missing_stations_csv:
            missing_df = pd.DataFrame(
                [
                    (
                        net,
                        sta,
                        picks_per_station.get((net, sta), 0),
                        ",".join(sorted(agencies_per_station.get((net, sta), set()))),
                        ",".join(suggestions.get((net, sta), [])),
                    )
                    for net, sta in missing_ids
                ],
                columns=["network", "station", "pick_count", "agencies", "suggested_rename"],
            ).sort_values("pick_count", ascending=False)
            missing_path = Path(args.missing_stations_csv)
            missing_df.to_csv(missing_path, index=False)
            print(f"Missing stations written to {missing_path}")

        if args.suggest_rename_yaml and suggestions:
            lines = ["# Suggested rename rules (auto-generated by pick_stats_from_config.py)"]
            for (net, sta), correct_nets in sorted(suggestions.items()):
                correct_net = correct_nets[0]
                escaped_net = net.replace(".", r"\.")
                escaped_sta = sta.replace(".", r"\.")
                lines.append(f"- ^{escaped_net}\\.{escaped_sta}\\.: {correct_net}.{sta}.")
            yaml_path = Path(args.suggest_rename_yaml)
            yaml_path.write_text("\n".join(lines) + "\n")
            print(f"Suggested rename rules written to {yaml_path} ({len(suggestions)} rules)")


if __name__ == "__main__":
    main()
