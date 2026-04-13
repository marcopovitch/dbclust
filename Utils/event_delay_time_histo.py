#!/usr/bin/env python
import argparse
import sys
from os.path import exists

import matplotlib.pyplot as plt
import pandas as pd
from dateutil.parser import parse
from matplotlib.dates import DateFormatter


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-file", required=True, help="CSV input file", type=str)
    parser.add_argument("-o", "--output-file", required=True, help="PNG output file", type=str)
    parser.add_argument("--from", dest="from_time", help="Start time filter (YYYY-MM-DD)", type=str)
    parser.add_argument("--to", dest="to_time", help="End time filter (YYYY-MM-DD)", type=str)
    parser.add_argument("-r", "--region", default="", help="Region name for the title", type=str)
    parser.add_argument(
        "--marker-size",
        dest="marker_size",
        default=1.0,
        type=float,
        help="Marker size for Δt points (default: 1.0)",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Print statistics on inter-event delays",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()

    if exists(args.output_file):
        print(f"Error: output file {args.output_file} already exists.")
        sys.exit(1)

    # Load and parse time column
    df = pd.read_csv(args.input_file)
    if "time" not in df.columns:
        print("CSV must contain a 'time' column.")
        sys.exit(1)

    df["time"] = pd.to_datetime(df["time"].apply(parse), utc=True)
    df = df.sort_values("time")

    # Optional time filtering
    if args.from_time:
        start_time = pd.to_datetime(parse(args.from_time)).tz_localize("UTC")
        df = df[df["time"] >= start_time]
    if args.to_time:
        end_time = pd.to_datetime(parse(args.to_time)).tz_localize("UTC")
        df = df[df["time"] <= end_time]

    # Compute time delta between consecutive events
    df["delta_seconds"] = df["time"].diff().dt.total_seconds()
    df_deltas = df.dropna(subset=["delta_seconds"])

    positive_deltas = df_deltas[df_deltas["delta_seconds"] > 0]["delta_seconds"]
    if len(positive_deltas) == 0:
        min_positive_delta = 1.0
    else:
        min_positive_delta = float(positive_deltas.min())

    # Optional statistics on inter-event delays
    if args.stats:
        import numpy as np

        s = positive_deltas
        percentiles = np.percentile(s, [25, 50, 75, 90, 95, 99])
        print(f"Inter-event delay statistics ({len(s)} intervals):")
        print(f"  min    : {s.min():.1f} s  ({s.min()/60:.2f} min)")
        print(f"  p25    : {percentiles[0]:.1f} s  ({percentiles[0]/60:.2f} min)")
        print(f"  median : {percentiles[1]:.1f} s  ({percentiles[1]/60:.2f} min)")
        print(f"  mean   : {s.mean():.1f} s  ({s.mean()/60:.2f} min)")
        print(f"  p75    : {percentiles[2]:.1f} s  ({percentiles[2]/60:.2f} min)")
        print(f"  p90    : {percentiles[3]:.1f} s  ({percentiles[3]/3600:.3f} h)")
        print(f"  p95    : {percentiles[4]:.1f} s  ({percentiles[4]/3600:.3f} h)")
        print(f"  p99    : {percentiles[5]:.1f} s  ({percentiles[5]/3600:.3f} h)")
        print(f"  max    : {s.max():.1f} s  ({s.max()/3600:.2f} h)")
        print(f"  < 1 min : {(s < 60).sum()} events ({100*(s < 60).mean():.1f}%)")
        print(f"  < 1 h   : {(s < 3600).sum()} events ({100*(s < 3600).mean():.1f}%)")
        print(f"  < 1 day : {(s < 86400).sum()} events ({100*(s < 86400).mean():.1f}%)")
        print(f"  < 1 week: {(s < 7*86400).sum()} events ({100*(s < 7*86400).mean():.1f}%)")

    # Compute daily histogram
    df["date"] = df["time"].dt.date
    daily_counts = df.groupby("date").size()
    daily_counts.index = pd.to_datetime(daily_counts.index)

    # Create subplots with shared x-axis
    fig, (ax1, ax2) = plt.subplots(
        nrows=2, ncols=1, figsize=(20, 10), sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [1, 2], "hspace": 0.1}
    )

    # Upper plot: daily histogram
    ax1.bar(daily_counts.index, daily_counts.values, width=1.0, color="blue", align="edge")
    ax1.set_ylabel("Events/day")
    ax1.set_title(f"Event activity and inter-event delay — {args.region}")
    ax1.grid(True, linestyle="--", alpha=0.3)

    # Lower plot: delta times (log scale)
    ax2.plot(
        df_deltas["time"],
        df_deltas["delta_seconds"],
        marker="o",
        linestyle="None",
        markersize=float(args.marker_size),
        label="Time delta",
    )
    ax2.set_yscale("log")
    ax2.set_ylim(bottom=min_positive_delta)
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Δt (seconds)")
    ax2.grid(True, linestyle="--", alpha=0.3)

    # Human-readable x-axis
    ax2.xaxis.set_major_formatter(DateFormatter("%Y-%m-%d"))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)

    # Reference lines
    reference_lines = {
        "1 min": 60,
        "1 h": 3600,
        "1 day": 86400,
        "1 month": 30 * 86400,
    }

    for label, seconds in reference_lines.items():
        ax2.axhline(y=seconds, color="red", linestyle="--", linewidth=1)
        y_text = seconds / 1.15
        if y_text <= min_positive_delta:
            y_text = min_positive_delta
        ax2.text(
            df_deltas["time"].min(), y_text, label,
            color="red", fontsize=9, va="top"
        )

    fig.savefig(args.output_file)
    print(f"Plot saved to {args.output_file}")


if __name__ == "__main__":
    main()
