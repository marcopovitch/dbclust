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

    # Compute daily histogram
    df["date"] = df["time"].dt.date
    daily_counts = df.groupby("date").size()
    daily_counts.index = pd.to_datetime(daily_counts.index)

    # Create subplots with shared x-axis
    fig, (ax1, ax2) = plt.subplots(
        nrows=2, ncols=1, figsize=(16, 8), sharex=True,
        gridspec_kw={"height_ratios": [1, 2], "hspace": 0.1}
    )

    # Upper plot: daily histogram
    ax1.bar(daily_counts.index, daily_counts.values, width=1.0, color="blue", align="center")
    ax1.set_ylabel("Events/day")
    ax1.set_title(f"Event activity and inter-event delay — {args.region}")
    ax1.grid(True, linestyle="--", alpha=0.3)

    # Lower plot: delta times (log scale)
    ax2.plot(df_deltas["time"], df_deltas["delta_seconds"], marker="o", linestyle="None", markersize=1, label="Time delta")
    ax2.set_yscale("log")
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
        ax2.text(
            df_deltas["time"].min(), seconds * 1.05, label,
            color="red", fontsize=9, va="bottom"
        )

    plt.tight_layout()
    fig.savefig(args.output_file)
    print(f"Plot saved to {args.output_file}")


if __name__ == "__main__":
    main()
