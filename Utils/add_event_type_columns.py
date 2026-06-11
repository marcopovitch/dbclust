#!/usr/bin/env python3
"""Add event_type_<AGENCY> columns to an alceste CSV file.

For each configured agency, the column event_type_<AGENCY> is populated when:
  1. The agency name appears in the agency_names list of an alceste row.
  2. One of the event_ids in the agencies_list of that alceste row matches
     an event_id in the agency's source CSV.

A consensus column (event_type_consensus) and a final value
(event_type_final) are then derived across all agency columns.

Usage:
    add-event-types -c add_event_types.yml
    python Utils/add_event_type_columns.py -c add_event_types.yml
"""

import argparse
import ast
import sys
from pathlib import Path

import pandas as pd
import yaml


def yml_read_config(filename: str) -> dict:
    """
    Reads a YAML configuration file and returns its contents as a dictionary.

    :param filename: Path to the YAML configuration file.
    :return: Parsed configuration dictionary.
    """
    with open(filename, "r") as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
    return cfg


def parse_jsonish_list(raw) -> list:
    """Return list extracted from a serialized JSON/Python list column."""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return []

    text = str(raw).strip()
    if not text:
        return []

    try:
        parsed = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return []

    if isinstance(parsed, (list, tuple)):
        return [str(item).strip() for item in parsed if str(item).strip()]

    return []


def normalize_event_type(value, event_type_groups: dict) -> str:
    """Return the canonical group name for value, or value unchanged if ungrouped."""
    return event_type_groups.get(value, value)


def compute_consensus(row: pd.Series, et_cols: list, event_type_groups: dict) -> str:
    """
    Return a consensus status for a row across all event_type_<AGENCY> columns.

    event_type values are first normalized using event_type_groups, so that
    equivalent labels (e.g. "explosion" and "controlled explosion") don't
    count as a conflict.

    Values:
      - "no_data"   : no agency has an event_type for this event
      - "consensus" : at least one agency has a type, and all filled agencies agree
                       (after normalization)
      - "conflict"  : at least two agencies disagree on the event_type
                       (after normalization)
    """
    filled = row[et_cols].dropna()
    if filled.empty:
        return "no_data"
    normalized = filled.map(lambda v: normalize_event_type(v, event_type_groups))
    if normalized.nunique() == 1:
        return "consensus"
    return "conflict"


def resolve_fixed_event_type(agency_names_value, agency_name: str, fixed_value: str):
    """
    Return fixed_value if agency_name appears in agency_names, else None.
    Used for agencies that always classify events with the same event_type.
    """
    if agency_name in parse_jsonish_list(agency_names_value):
        return fixed_value
    return None


def build_event_type_map(agency_df: pd.DataFrame) -> dict:
    """Return a dict mapping event_id -> event_type for the agency source file."""
    return dict(zip(agency_df["event_id"], agency_df["event_type"]))


def resolve_event_type(
    agencies_list_value, agency_name: str, agency_names_value, event_type_map: dict
):
    """
    Return the event_type from the agency source file if:
      - agency_name is present in agency_names (the list of contributing agencies)
      - one of the ids in agencies_list matches an event_id in event_type_map

    Returns None if no match is found.
    """
    agency_names = parse_jsonish_list(agency_names_value)
    if agency_name not in agency_names:
        return None

    agencies_list = parse_jsonish_list(agencies_list_value)
    for eid in agencies_list:
        if eid in event_type_map:
            return event_type_map[eid]

    return None


def compute_final_event_type(row: pd.Series, et_cols: list, event_type_groups: dict):
    """
    Return a single event_type value derived from all agency columns:
      - consensus or single agency filled : return the canonical (normalized)
                                              value for that group
      - conflict or no_data              : return None
    """
    if row["event_type_consensus"] != "consensus":
        return None
    filled = row[et_cols].dropna()
    return normalize_event_type(filled.iloc[0], event_type_groups)


def add_event_type_columns(config_path: Path) -> None:
    config = yml_read_config(str(config_path))
    base_dir = config_path.parent

    input_path = base_dir / config["input_file"]
    output_path = base_dir / config["output_file"]
    event_type_groups = config.get("event_type_groups", {}) or {}

    # Validate config: each agency must have exactly one of 'file' or 'fixed_event_type'
    for agency_cfg in config["agencies"]:
        has_file = "file" in agency_cfg
        has_fixed = "fixed_event_type" in agency_cfg
        if has_file == has_fixed:
            raise ValueError(
                f"Agency '{agency_cfg['agency_name']}': must have exactly one of "
                f"'file' or 'fixed_event_type', not both or neither."
            )

    print(f"Reading input file: {input_path}")
    alceste = pd.read_csv(input_path, low_memory=False)
    print(f"  {len(alceste):,} rows loaded")

    for agency_cfg in config["agencies"]:
        agency_name = agency_cfg["agency_name"]
        col_name = f"event_type_{agency_name}"

        if "fixed_event_type" in agency_cfg:
            fixed_value = agency_cfg["fixed_event_type"]
            print(f"\nProcessing agency: {agency_name}")
            print(f"  Fixed event_type: {fixed_value}")
            alceste[col_name] = [
                resolve_fixed_event_type(an, agency_name, fixed_value)
                for an in alceste["agency_names"]
            ]
        else:
            agency_file = base_dir / agency_cfg["file"]
            print(f"\nProcessing agency: {agency_name}")
            print(f"  Source file: {agency_file}")
            agency_df = pd.read_csv(
                agency_file, usecols=["event_id", "event_type"], low_memory=False
            )
            print(f"  {len(agency_df):,} rows in source file")
            event_type_map = build_event_type_map(agency_df)
            alceste[col_name] = [
                resolve_event_type(al, agency_name, an, event_type_map)
                for al, an in zip(alceste["agencies_list"], alceste["agency_names"])
            ]

        filled = alceste[col_name].notna().sum()
        print(
            f"  Column '{col_name}' filled for {filled:,} rows "
            f"({100 * filled / len(alceste):.1f}%)"
        )

    et_cols = [f"event_type_{cfg['agency_name']}" for cfg in config["agencies"]]
    alceste["event_type_consensus"] = alceste.apply(
        compute_consensus, axis=1, et_cols=et_cols, event_type_groups=event_type_groups
    )
    counts = alceste["event_type_consensus"].value_counts()
    print("\nConsensus summary:")
    for label in ("consensus", "conflict", "no_data"):
        n = counts.get(label, 0)
        print(f"  {label:10s}: {n:>7,}  ({100 * n / len(alceste):.1f}%)")

    renass_phasenet_only = (
        alceste["agencies_list"]
        .apply(lambda v: parse_jsonish_list(v) == ["RENASS/PHASENET"])
        .sum()
    )
    print(
        f"\n  Only 'RENASS/PHASENET' in agencies_list: {renass_phasenet_only:,}"
        f"  ({100 * renass_phasenet_only / len(alceste):.1f}%)"
    )

    alceste["event_type_final"] = alceste.apply(
        compute_final_event_type, axis=1, et_cols=et_cols, event_type_groups=event_type_groups
    )
    filled_final = alceste["event_type_final"].notna().sum()
    print(
        f"\n  event_type_final filled: {filled_final:,}  ({100 * filled_final / len(alceste):.1f}%)"
    )
    print("  Distribution:")
    print(alceste["event_type_final"].value_counts().to_string(max_rows=10))

    print(f"\nWriting output file: {output_path}")
    alceste.to_csv(output_path, index=False)
    print("Done.")


def main():
    parser = argparse.ArgumentParser(
        description="Add per-agency event_type columns and a consensus to an alceste CSV file."
    )
    parser.add_argument(
        "-c",
        "--conf",
        required=True,
        dest="configfile",
        help="YAML configuration file path.",
        type=str,
    )
    args = parser.parse_args()

    config_path = Path(args.configfile)
    if not config_path.exists():
        print(f"Error: config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    add_event_type_columns(config_path)


if __name__ == "__main__":
    main()
