#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import logging
import pandas as pd
from contextlib import contextmanager
from sklearn.cluster import DBSCAN

logger = logging.getLogger("pick_preproc")
logger.setLevel(logging.INFO)


@contextmanager
def omp_single_thread():
    """Context manager to temporarily set OMP_NUM_THREADS=1."""
    old_val = os.environ.get("OMP_NUM_THREADS")
    os.environ["OMP_NUM_THREADS"] = "1"
    try:
        yield
    finally:
        # Restore previous value
        if old_val is None:
            os.environ.pop("OMP_NUM_THREADS", None)
        else:
            os.environ["OMP_NUM_THREADS"] = old_val


# This function is a wrapper to ensure that deduplication runs in a single thread.
# This is necessary to avoid issues with scikit-learn and libtiff when using multiple threads.
# At least on macOS
def safe_deduplicate_picks_by_time(*args, **kwargs):
    with omp_single_thread():
        return deduplicate_picks_by_time(*args, **kwargs)


def get_index(group: pd.DataFrame, debug=False) -> int:
    """
    Determines the index of the pick with the highest phase_score.
    If there are multiple picks with the highest score, it returns the index of the median phase_time,
    giving preference to manually evaluated picks.

    Args:
        group (GroupBy): A group of picks associated with a single station and phase (P|S).
        debug (bool, optional): If True, outputs debug information. Defaults to False.

    Returns:
        int: The index of the chosen pick.
    """
    max_value = group["phase_score"].max()
    max_value_indices = group[group["phase_score"] == max_value].index
    if len(max_value_indices) == 1:
        # only one absolute max
        max_index = max_value_indices[0]
        if debug:
            max_value = group.loc[max_index, "phase_score"]
            logger.debug(
                f"Preprocessing pick: nb picks {len(group)}, proba max {len(group)}: max_index={max_index} max_value={max_value}"
            )
        return max_index
    else:
        # multiple picks with the highest phase_score

        # check if there are manual picks
        if "phase_evaluation" in group.columns:
            manual_df = group[group["phase_evaluation"] == "manual"]
        else:
            manual_df = pd.DataFrame()

        # prioritize manual picks
        if len(manual_df) >= 1:
            # get the index of the phase_time median pick
            median_index = (
                manual_df["phase_time"].sort_values().index[len(manual_df) // 2]
            )
            if debug:
                median_value = manual_df.loc[median_index, "phase_time"]
                logger.debug(
                    f"Preprocessing pick: nb picks {len(group)}, median manual {len(manual_df)}: median_index={median_index} median_value={median_value}"
                )
            return median_index
        else:
            # no manual pick, get the index of the phase_time median pick
            median_index = group["phase_time"].sort_values().index[len(group) // 2]
            if debug:
                median_value = group.loc[median_index, "phase_time"]
                logger.debug(
                    f"cluster nb picks {len(group)}, median other {len(group)}: median_index={median_index} median_value={median_value}"
                )
            return median_index


def deduplicate_picks_by_time(
    df: pd.DataFrame, P_delta_time: float, S_delta_time: float
) -> pd.DataFrame:
    """
    Unloads picks that are too close in time to each other.
    The function uses DBSCAN to cluster the picks by station_id and phase_type (P|S).
    For each cluster, it keeps the pick with the highest phase_score.
    If there are multiple picks with the highest score, it returns the pick with the median phase_time,
    giving preference to manually evaluated picks.

    Args:
        df (pd.DataFrame): The DataFrame containing the picks.
        P_delta_time (float): The maximum time difference between P picks.
        S_delta_time (float): The maximum time difference between S picks.

    Returns:
        pd.DataFrame: The DataFrame containing the filtered picks.
    """

    # empty dataframe
    results = pd.DataFrame(
        columns=[
            "station_id",
            "channel",
            "phase_type",
            "phase_time",
            "phase_score",
            "phase_evaluation",
            "phase_method",
            "event_id",
            "agency",
        ]
    )

    df = df.sort_values(by=["station_id", "phase_type", "phase_time"])

    # run separately by phase type
    for phase in ("P", "S"):
        logger.debug(f"Working on {phase}.")
        phase_df = df[df["phase_type"].str.contains(phase)].copy()

        # skip if no picks for this phase
        if phase_df.empty:
            logger.debug(f"No {phase} picks found, skipping.")
            continue

        # create numeric_time for time distance computation
        min_timestamp = phase_df["phase_time"].min()
        phase_df["numeric_time"] = (
            phase_df["phase_time"] - pd.to_datetime(min_timestamp, utc=True)
        ).dt.total_seconds()

        # dbscan configuration
        if "P" in phase:
            max_distance = P_delta_time  # secondes
        else:
            max_distance = S_delta_time  # secondes

        min_samples = 1
        dbscan = DBSCAN(eps=max_distance, min_samples=min_samples, metric="euclidean")

        # loop over station_id
        for station_id in phase_df["station_id"].drop_duplicates():
            logger.debug(f"Working on {phase}/{station_id}")
            # filter by station and sort by phase, as the order is kept by groupby
            # The order should be Pn, Pg, P.
            tmp_df = (
                phase_df.loc[phase_df["station_id"] == station_id]
                .sort_values(by="phase_type", ascending=False)
                .copy()
            )
            before = len(tmp_df)

            # clusterize by station_id
            tmp_df["cluster"] = dbscan.fit_predict(tmp_df[["numeric_time"]])

            # keeps only the pick with the higher score
            # idx_tmp = tmp_df.groupby(["cluster"])["phase_score"].idxmax()

            # keeps the pick with the higher score or the median one
            # given preference to manually evaluated picks
            idx = tmp_df.groupby(["cluster"])[
                ["phase_score", "phase_evaluation", "phase_time"]
            ].apply(get_index)

            tmp_df = tmp_df.loc[idx]
            tmp_df = tmp_df.drop(columns=["numeric_time", "cluster"])

            after = len(tmp_df)
            logger.debug(f"length before: {before}, after: {after}")

            # Concatenate the results
            if results.empty:
                results = tmp_df.copy()
            else:
                results = pd.concat([results, tmp_df], ignore_index=True)

    results.sort_values(by=["phase_time", "station_id"], inplace=True)
    return results
