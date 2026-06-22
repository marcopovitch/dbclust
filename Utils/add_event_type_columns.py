#!/usr/bin/env python3
"""Add event_type_<AGENCY> columns to an events CSV file.

For each configured agency, the column event_type_<AGENCY> is populated when:
  1. The agency name appears in the agency_names list of a catalog row.
  2. One of the event_ids in the agencies_list of that catalog row matches
     an event_id in the agency's source CSV.

Derived columns:
  event_type_SPECTROCNN   : trusted CNN prediction (earthquake / quarry blast)
                            from spectrocnn_predictions_file (predhdq50 0/1).
                            Can be a single path or a list of paths, in which
                            case predictions are concatenated and deduplicated
                            by event_id (first occurrence wins).
  event_type_consensus    : inter-agency agreement (consensus/conflict/no_data/NULL)
                            NULL when spectrocnn-only (no agency data)
  event_type_final        : canonical final value (agency value takes priority
                            when compatible with the spectrocnn category;
                            otherwise the spectrocnn value is used)
  event_type_final_source : provenance — one of:
                            spectrocnn, spectrocnn+consensus,
                            spectrocnn+single:<AGENCY>, spectrocnn+conflict:<AGENCY>,
                            consensus, single:<AGENCY>, conflict:<AGENCY>

Usage:
    add-event-types -c add_event_types.yml
    python Utils/add_event_type_columns.py -c add_event_types.yml
"""

import argparse
import ast
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
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
    Return (value, source_detail) derived from all agency columns:
      - no_data  : (None, None)
      - single   : (normalized value, "single:<AGENCY>")
      - consensus: (normalized value, "consensus")
      - conflict : (normalized value from highest-priority agency,
                    "conflict:<AGENCY>")
    """
    if row["event_type_consensus"] == "no_data":
        return None, None
    if row["event_type_consensus"] == "consensus":
        filled = row[et_cols].dropna()
        if len(filled) == 1:
            agency_name = filled.index[0].removeprefix("event_type_")
            return normalize_event_type(filled.iloc[0], event_type_groups), f"single:{agency_name}"
        return normalize_event_type(filled.iloc[0], event_type_groups), "consensus"
    # conflict: resolve by agency priority
    for col in priority_cols:
        value = row[col]
        if pd.notna(value):
            agency_name = col.removeprefix("event_type_")
            return normalize_event_type(value, event_type_groups), f"conflict:{agency_name}"
    return None, None



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

    Sources (event_type_final_source values):
      "spectrocnn"             — spectrocnn value used as-is, because no agency
                                 value is available or compatible with it
      "spectrocnn+<src>"       — agency-derived value used as-is (it is more
                                 precise and compatible with the spectrocnn
                                 category, which only confirms it); <src> is
                                 one of: consensus, single:<AGENCY>,
                                 conflict:<AGENCY>
      "consensus"              — multiple agencies agreed, no spectrocnn
      "single:<AGENCY>"        — only one agency had a value, no spectrocnn
      "conflict:<AGENCY>"      — agencies disagreed, <AGENCY> resolved it, no spectrocnn
      None                     — no information available
    """
    agency_value, agency_source = compute_final_event_type(
        row, et_cols, event_type_groups, priority_cols
    )

    if spectrocnn_col is not None:
        spectrocnn_value = row[spectrocnn_col]
        if pd.notna(spectrocnn_value):
            compatible = spectrocnn_compatibility.get(spectrocnn_value, [])
            if agency_value is not None and agency_value in compatible:
                return agency_value, f"spectrocnn+{agency_source}"
            return normalize_event_type(spectrocnn_value, event_type_groups), "spectrocnn"

    if agency_value is None:
        return None, None
    return agency_value, agency_source


# Explanation of event_type_final_source labels, shown as an annotation on
# the source distribution plot. NULL/no_data (no information available) is
# excluded from that plot, so it is not listed here.
SOURCE_EXPLANATION = (
    "single:<AGENCY>   : only one agency had a value, no spectrocnn<br>"
    "consensus         : multiple agencies agreed, no spectrocnn<br>"
    "conflict:<AGENCY> : agencies disagreed, <AGENCY> (by priority) resolved it<br>"
    "spectrocnn        : spectrocnn value used as-is (no compatible agency value)<br>"
    "spectrocnn+<src>  : agency value used as-is, confirmed compatible with the "
    "spectrocnn category<br>"
    "                    (<src> is single/consensus/conflict)"
)


def category_of_source_label(label: str) -> str:
    """
    Return the category a event_type_final_source label belongs to, for coloring.

    "spectrocnn+<src>" labels are sub-categorized by <src> (single/consensus/
    conflict), since they are primarily an agency-derived value confirmed by
    spectrocnn — only bare "spectrocnn" (no compatible agency refinement) gets
    its own category.
    """
    if label == "spectrocnn":
        return "spectrocnn"
    if label.startswith("spectrocnn+single:") or label.startswith("single:"):
        return "single"
    if label.startswith("spectrocnn+conflict:") or label.startswith("conflict:"):
        return "conflict"
    if label in ("spectrocnn+consensus", "consensus"):
        return "consensus"
    return "other"


def is_spectrocnn_confirmed(label: str) -> bool:
    """Return True for spectrocnn+<src> labels (spectrocnn-confirmed agency value)."""
    return label.startswith("spectrocnn+")


CATEGORY_COLORS = {
    "spectrocnn": "#1f77b4",
    "consensus": "#2ca02c",
    "single": "#ff7f0e",
    "conflict": "#d62728",
    "other": "#7f7f7f",
}

# Lighter tint used for spectrocnn+<src> bars, to show they are a spectrocnn-
# confirmed variant of their underlying single/consensus/conflict category.
CATEGORY_COLORS_SPECTROCNN_CONFIRMED = {
    "consensus": "#a8dba8",
    "single": "#ffc999",
    "conflict": "#f4a6a6",
}


def plot_distribution(
    distribution: pd.Series,
    title: str,
    x_title: str,
    output_path: Path,
    drop_null: bool = False,
    colorize: bool = False,
    explanation: str | None = None,
) -> None:
    """Write an interactive bar chart (HTML) of a value_counts() distribution."""
    if drop_null:
        distribution = distribution[distribution.index.notna()]

    labels = ["NULL" if pd.isna(v) else str(v) for v in distribution.index]

    if colorize:
        categories = [category_of_source_label(label) for label in labels]
        confirmed = [is_spectrocnn_confirmed(label) for label in labels]
        colors = [
            CATEGORY_COLORS_SPECTROCNN_CONFIRMED.get(cat, CATEGORY_COLORS[cat])
            if is_confirmed
            else CATEGORY_COLORS[cat]
            for cat, is_confirmed in zip(categories, confirmed)
        ]
        fig = go.Figure(
            data=[
                go.Bar(
                    x=labels,
                    y=distribution.values,
                    text=distribution.values,
                    marker_color=colors,
                )
            ]
        )
        # Legend: one dummy trace per (category, spectrocnn-confirmed) combination
        # actually present, in a stable, readable order.
        legend_keys = sorted(set(zip(categories, confirmed)), key=lambda k: (k[0], k[1]))
        for category, is_confirmed in legend_keys:
            color = (
                CATEGORY_COLORS_SPECTROCNN_CONFIRMED.get(category, CATEGORY_COLORS[category])
                if is_confirmed
                else CATEGORY_COLORS[category]
            )
            name = f"{category} (spectrocnn-confirmed)" if is_confirmed else category
            fig.add_trace(
                go.Bar(
                    x=[None],
                    y=[None],
                    name=name,
                    marker_color=color,
                    showlegend=True,
                )
            )
    else:
        fig = go.Figure(
            data=[go.Bar(x=labels, y=distribution.values, text=distribution.values)]
        )

    fig.update_traces(textposition="outside", selector=dict(type="bar"))
    fig.update_layout(title=title, xaxis_title=x_title, yaxis_title="count")

    if explanation:
        fig.add_annotation(
            text=explanation,
            showarrow=False,
            xref="paper",
            yref="paper",
            x=0.98,
            y=0.98,
            xanchor="right",
            yanchor="top",
            align="left",
            bordercolor="black",
            borderwidth=1,
            borderpad=6,
            bgcolor="rgba(255,255,255,0.9)",
            font=dict(size=11, family="monospace"),
        )
    fig.write_html(str(output_path))


def add_event_type_columns(config_path: Path) -> None:
    config = yml_read_config(str(config_path))
    base_dir = config_path.parent

    input_path = base_dir / config["input_file"]
    output_path = base_dir / config["output_file"]
    event_type_final_source_distribution_file = config.get(
        "event_type_final_source_distribution_file"
    )
    event_type_final_distribution_file = config.get(
        "event_type_final_distribution_file"
    )
    event_type_final_source_distribution_plot_file = config.get(
        "event_type_final_source_distribution_plot_file"
    )
    event_type_final_distribution_plot_file = config.get(
        "event_type_final_distribution_plot_file"
    )
    event_type_groups = config.get("event_type_groups", {}) or {}
    spectrocnn_predictions_file = config.get("spectrocnn_predictions_file")
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
    catalog = pd.read_csv(input_path, low_memory=False)
    print(f"  {len(catalog):,} rows loaded")

    spectrocnn_col = None
    if spectrocnn_predictions_file:
        pred_paths = (
            [spectrocnn_predictions_file]
            if isinstance(spectrocnn_predictions_file, str)
            else list(spectrocnn_predictions_file)
        )

        pred_dfs = []
        for raw_path in pred_paths:
            pred_path = Path(raw_path)
            if not pred_path.exists():
                raise ValueError(
                    f"spectrocnn_predictions_file not found: {pred_path}"
                )
            print(f"\nProcessing spectrocnn predictions file: {pred_path}")
            pred_df = pd.read_csv(
                pred_path, usecols=["event_id", "predhdq50"], low_memory=False
            )
            print(f"  {len(pred_df):,} predictions loaded")
            pred_dfs.append(pred_df)

        pred_df = pd.concat(pred_dfs, ignore_index=True)
        duplicated = pred_df["event_id"].duplicated().sum()
        if len(pred_dfs) > 1 and duplicated:
            nunique_per_id = pred_df.groupby("event_id")["predhdq50"].nunique()
            conflicting_ids = nunique_per_id[nunique_per_id > 1]
            if len(conflicting_ids):
                print(
                    f"\n  WARNING: {len(conflicting_ids):,} duplicate event_id "
                    f"have conflicting predhdq50 values across files "
                    f"(first file in the list takes priority)"
                )
        pred_df = pred_df.drop_duplicates(subset="event_id", keep="first")
        if len(pred_dfs) > 1:
            print(
                f"\n  Combined predictions: {len(pred_df):,} unique events "
                f"({duplicated:,} duplicate event_id rows dropped)"
            )

        # predhdq50: 0=earthquake, 1=quarry blast, other=unknown
        label_map = {0: "earthquake", 1: "quarry blast"}
        pred_df = pred_df[pred_df["predhdq50"].isin(label_map)]
        pred_map = dict(zip(pred_df["event_id"], pred_df["predhdq50"].map(label_map)))
        spectrocnn_col = "event_type_SPECTROCNN"
        catalog[spectrocnn_col] = catalog["event_id"].map(pred_map)
        filled = catalog[spectrocnn_col].notna().sum()
        print(
            f"  Column '{spectrocnn_col}' filled for {filled:,} rows "
            f"({100 * filled / len(catalog):.1f}%)"
        )

    for agency_cfg in config["agencies"]:
        agency_name = agency_cfg["agency_name"]
        col_name = f"event_type_{agency_name}"

        if "fixed_event_type" in agency_cfg:
            fixed_value = agency_cfg["fixed_event_type"]
            print(f"\nProcessing agency: {agency_name}")
            print(f"  Fixed event_type: {fixed_value}")
            catalog[col_name] = [
                resolve_fixed_event_type(an, agency_name, fixed_value)
                for an in catalog["agency_names"]
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
            catalog[col_name] = [
                resolve_event_type(al, agency_name, an, event_type_map)
                for al, an in zip(catalog["agencies_list"], catalog["agency_names"])
            ]

        filled = catalog[col_name].notna().sum()
        print(
            f"  Column '{col_name}' filled for {filled:,} rows "
            f"({100 * filled / len(catalog):.1f}%)"
        )

    et_cols = [f"event_type_{cfg['agency_name']}" for cfg in config["agencies"]]

    agency_priority = config.get("agency_priority", []) or []
    agency_names = [cfg["agency_name"] for cfg in config["agencies"]]
    for name in agency_priority:
        if name not in agency_names:
            raise ValueError(f"agency_priority: unknown agency '{name}'")
    ordered_names = agency_priority + [n for n in agency_names if n not in agency_priority]
    priority_cols = [f"event_type_{name}" for name in ordered_names]

    catalog["event_type_consensus"] = catalog.apply(
        compute_consensus, axis=1, et_cols=et_cols, event_type_groups=event_type_groups
    )
    # no_data + spectrocnn present → NULL (consensus is agency-only; spectrocnn
    # is not an agency, so the question of inter-agency agreement doesn't arise)
    if spectrocnn_col is not None:
        mask_no_data_with_spectrocnn = (
            (catalog["event_type_consensus"] == "no_data") &
            catalog[spectrocnn_col].notna()
        )
        catalog.loc[mask_no_data_with_spectrocnn, "event_type_consensus"] = None

    counts = catalog["event_type_consensus"].value_counts(dropna=False)
    print("\nConsensus summary:")
    for label in ("consensus", "conflict", "no_data", None):
        n = counts.get(label, 0)
        label_str = "NULL" if label is None else label
        print(f"  {label_str:10s}: {n:>7,}  ({100 * n / len(catalog):.1f}%)")

    renass_phasenet_only = (
        catalog["agencies_list"]
        .apply(lambda v: parse_jsonish_list(v) == ["RENASS/PHASENET"])
        .sum()
    )
    print(
        f"\n  Only 'RENASS/PHASENET' in agencies_list: {renass_phasenet_only:,}"
        f"  ({100 * renass_phasenet_only / len(catalog):.1f}%)"
    )

    final_results = catalog.apply(
        compute_final_event_type_and_source,
        axis=1,
        et_cols=et_cols,
        event_type_groups=event_type_groups,
        priority_cols=priority_cols,
        spectrocnn_col=spectrocnn_col,
        spectrocnn_compatibility=spectrocnn_compatibility,
    )
    catalog["event_type_final"] = final_results.apply(lambda r: r[0])
    catalog["event_type_final_source"] = final_results.apply(lambda r: r[1])

    filled_final = catalog["event_type_final"].notna().sum()
    print(
        f"\n  event_type_final filled: {filled_final:,}  ({100 * filled_final / len(catalog):.1f}%)"
    )

    event_type_final_distribution = catalog["event_type_final"].value_counts(dropna=False)
    print("  Distribution:")
    print(event_type_final_distribution.to_string(max_rows=10))

    event_type_final_source_distribution = catalog["event_type_final_source"].value_counts(
        dropna=False
    )
    print("\n  Source distribution:")
    print(event_type_final_source_distribution.to_string(max_rows=10))

    if event_type_final_distribution_file:
        dist_path = base_dir / event_type_final_distribution_file
        print(f"\nWriting event_type_final distribution file: {dist_path}")
        event_type_final_distribution.rename_axis("event_type_final").reset_index(
            name="count"
        ).to_csv(dist_path, index=False)

    if event_type_final_source_distribution_file:
        source_dist_path = base_dir / event_type_final_source_distribution_file
        print(f"Writing event_type_final_source distribution file: {source_dist_path}")
        event_type_final_source_distribution.rename_axis(
            "event_type_final_source"
        ).reset_index(name="count").to_csv(source_dist_path, index=False)

    if event_type_final_distribution_plot_file:
        plot_path = base_dir / event_type_final_distribution_plot_file
        print(f"Writing event_type_final distribution plot: {plot_path}")
        plot_distribution(
            event_type_final_distribution,
            title="event_type_final distribution",
            x_title="event_type_final",
            output_path=plot_path,
        )

    if event_type_final_source_distribution_plot_file:
        source_plot_path = base_dir / event_type_final_source_distribution_plot_file
        print(f"Writing event_type_final_source distribution plot: {source_plot_path}")
        plot_distribution(
            event_type_final_source_distribution,
            title="event_type_final_source distribution",
            x_title="event_type_final_source",
            output_path=source_plot_path,
            drop_null=True,
            colorize=True,
            explanation=SOURCE_EXPLANATION,
        )

    print(f"\nWriting output file: {output_path}")
    catalog.to_csv(output_path, index=False)
    print("Done.")


def main():
    parser = argparse.ArgumentParser(
        description="Add per-agency event_type columns and a consensus to a catalog CSV file."
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
