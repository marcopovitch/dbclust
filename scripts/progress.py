#!/usr/bin/env python
"""Monitor DBClust task progress from CSV profile file."""

import argparse
import sys
import time

import pandas as pd
from pathlib import Path
from tqdm import tqdm


def monitor_progress(csv_path: Path, poll_interval: float = 1.0) -> None:
    """Monitor progress from a CSV profile file.

    Args:
        csv_path: Path to the CSV profile file.
        poll_interval: Time in seconds between file reads.
    """
    pbar = None
    last_completed = 0

    print(f"Monitoring: {csv_path}")
    print("Waiting for tasks to start...")

    while True:
        if not csv_path.exists():
            time.sleep(poll_interval)
            continue

        try:
            df = pd.read_csv(csv_path)
        except pd.errors.EmptyDataError:
            time.sleep(poll_interval)
            continue

        if df.empty:
            time.sleep(poll_interval)
            continue

        last = df.iloc[-1]

        total = int(last["total_tasks"])
        completed = int(last["completed_count"])
        pct = float(last["progress_pct"])

        if pbar is None:
            pbar = tqdm(total=total, desc="Traitement", unit="task")

        delta = completed - last_completed
        if delta > 0:
            pbar.update(delta)
            last_completed = completed

        # Compute ETA based on average duration
        avg_duration = df["duration_sec"].mean()
        remaining_tasks = total - completed
        eta_sec = avg_duration * remaining_tasks

        pbar.set_postfix(
            pct=f"{pct:.1f}%",
            mem=f'{last["peak_memory_mb"]:.0f} MB',
            last_task=int(last["task_index"]),
            eta=f"{eta_sec / 60:.1f} min",
        )

        if completed >= total:
            pbar.close()
            print("\nAll tasks completed!")
            break

        time.sleep(poll_interval)


def main():
    parser = argparse.ArgumentParser(
        description="Monitor DBClust task progress from CSV profile file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s task_profiles.csv
  %(prog)s /path/to/task_profiles.csv --interval 2
        """,
    )
    parser.add_argument(
        "csv_file",
        type=Path,
        help="Path to the CSV profile file (e.g., task_profiles.csv)",
    )
    parser.add_argument(
        "-i",
        "--interval",
        type=float,
        default=1.0,
        help="Polling interval in seconds (default: 1.0)",
    )

    args = parser.parse_args()

    if not args.csv_file.suffix == ".csv":
        print(f"Warning: {args.csv_file} does not have .csv extension", file=sys.stderr)

    try:
        monitor_progress(args.csv_file, args.interval)
    except KeyboardInterrupt:
        print("\nMonitoring interrupted.")
        sys.exit(0)


if __name__ == "__main__":
    main()
