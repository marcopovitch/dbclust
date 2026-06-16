#!/usr/bin/env python3
"""Add event_type_<AGENCY> columns to an alceste CSV file.

For each configured agency, the column event_type_<AGENCY> is populated when:
  1. The agency name appears in the agency_names list of an alceste row.
  2. One of the event_ids in the agencies_list of that alceste row matches
     an event_id in the agency's source CSV.

A consensus column (event_type_consensus) and a final value
(event_type_final) are then derived across all agency columns.

If spectrocnn_column is configured, a trusted CNN-based prediction
(earthquake / quarry blast) takes priority over the agency consensus
for event_type_final, and event_type_final_source records which one
was used ("spectrocnn" or "consensus_agencies").

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


def compute_final_event_type(
    row: pd.Series, et_cols: list, event_type_groups: dict, priority_cols: list
):
    """
    Return a single event_type value derived from all agency columns:
      - consensus or single agency filled : return the canonical (normalized)
                                              value for that group
      - conflict : return the normalized value from the highest-priority
                    agency that has one (per priority_cols, an ordered list
                    of event_type_<AGENCY> column names)
      - no_data : return None
    """
    if row["event_type_consensus"] == "no_data":
        return None
    if row["event_type_consensus"] == "consensus":
        filled = row[et_cols].dropna()
        return normalize_event_type(filled.iloc[0], event_type_groups)
    # conflict: resolve by agency priority
    for col in priority_cols:
        value = row[col]
        if pd.notna(value):
            return normalize_event_type(value, event_type_groups)
    return None


def resolve_spectrocnn_event_type(value):
    """
    Return value if it is one of the trusted spectrocnn predictions
    (earthquake, quarry blast), else None. Other values (e.g. "unknown")
    are not informative enough to be used as a reference.
    """
    if value in ("earthquake", "quarry blast"):
        return value
    return None


def compute_final_event_type_and_source(
    row: pd.Series,
    et_cols: list,
    event_type_groups: dict,
    priority_cols: list,
    spectrocnn_col: str | None,
    spectrocnn_compatibility: dict,
):
    """
    Return (event_type_final, event_type_final_source) for a row.

    - The agency value is derived from event_type_consensus: on consensus,
      the shared (normalized) value; on conflict, the normalized value from
      the highest-priority agency (per priority_cols); on no_data, None.
    - If a trusted spectrocnn prediction is available, it takes priority
      over the agency value, unless the agency value is a more precise
      label compatible with the spectrocnn category (per
      spectrocnn_compatibility): in that case the agency value is used
      (source = "spectrocnn+agencies"). Otherwise the spectrocnn value is
      used as-is (source = "spectrocnn").
    - Otherwise, fall back to the agency value (source =
      "consensus_agencies" when a value is derived, else None).
    """
    agency_value = compute_final_event_type(row, et_cols, event_type_groups, priority_cols)

    if spectrocnn_col is not None:
        spectrocnn_value = row[spectrocnn_col]
        if pd.notna(spectrocnn_value):
            compatible = spectrocnn_compatibility.get(spectrocnn_value, [])
            if agency_value is not None and agency_value in compatible:
                return agency_value, "spectrocnn+agencies"
            return normalize_event_type(spectrocnn_value, event_type_groups), "spectrocnn"

    if agency_value is None:
        return None, None
    return agency_value, "consensus_agencies"


def add_event_type_columns(config_path: Path) -> None:
    config = yml_read_config(str(config_path))
    base_dir = config_path.parent

    input_path = base_dir / config["input_file"]
    output_path = base_dir / config["output_file"]
    event_type_groups = config.get("event_type_groups", {}) or {}
    spectrocnn_column = config.get("spectrocnn_column")
    spectrocnn_compatibility = config.get("spectrocnn_compatibility", {}) or {}

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

    spectrocnn_col = None
    if spectrocnn_column:
        if spectrocnn_column not in alceste.columns:
            raise ValueError(
                f"spectrocnn_column '{spectrocnn_column}' not found in input file columns."
            )
        spectrocnn_col = "event_type_SPECTROCNN"
        print(f"\nProcessing spectrocnn reference column: {spectrocnn_column}")
        alceste[spectrocnn_col] = alceste[spectrocnn_column].apply(
            resolve_spectrocnn_event_type
        )
        filled = alceste[spectrocnn_col].notna().sum()
        print(
            f"  Column '{spectrocnn_col}' filled for {filled:,} rows "
            f"({100 * filled / len(alceste):.1f}%)"
        )

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

    agency_priority = config.get("agency_priority", []) or []
    agency_names = [cfg["agency_name"] for cfg in config["agencies"]]
    for name in agency_priority:
        if name not in agency_names:
            raise ValueError(f"agency_priority: unknown agency '{name}'")
    ordered_names = agency_priority + [n for n in agency_names if n not in agency_priority]
    priority_cols = [f"event_type_{name}" for name in ordered_names]

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

    final_results = alceste.apply(
        compute_final_event_type_and_source,
        axis=1,
        et_cols=et_cols,
        event_type_groups=event_type_groups,
        priority_cols=priority_cols,
        spectrocnn_col=spectrocnn_col,
        spectrocnn_compatibility=spectrocnn_compatibility,
    )
    alceste["event_type_final"] = final_results.apply(lambda r: r[0])
    alceste["event_type_final_source"] = final_results.apply(lambda r: r[1])

    filled_final = alceste["event_type_final"].notna().sum()
    print(
        f"\n  event_type_final filled: {filled_final:,}  ({100 * filled_final / len(alceste):.1f}%)"
    )
    print("  Distribution:")
    print(alceste["event_type_final"].value_counts().to_string(max_rows=10))
    print("\n  Source distribution:")
    print(alceste["event_type_final_source"].value_counts(dropna=False).to_string(max_rows=10))

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
