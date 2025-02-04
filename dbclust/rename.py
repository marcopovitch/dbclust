#!/usr/bin/env python
import logging
import re
import sys
from typing import Dict
from typing import List
from typing import Optional

import pandas as pd
from icecream import ic

from dbclust.config import RenameConfig

# default logger
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def apply_regex_to_row(row, patterns: Optional[List[Dict[str, str]]]) -> tuple:
    """Apply regex matches to station_id and channel columns.

    The function applies regex patterns to the waveform_id (concatenation of station_id
    and channel columns).

    Attributes:
        row: DataFrame row containing station_id and channel columns.
        patterns: List of dictionaries with regex patterns and replacements.

    Returns:
        Tuple with updated station_id and channel.

    """
    if not patterns:
        return row["station_id"], row["channel"]

    full_key = f"{row['station_id']}.{row['channel']}"
    for pattern_dict in patterns:
        for pattern_str, replacement in pattern_dict.items():
            pattern = re.compile(pattern_str)
            new_full_key = re.sub(pattern, replacement, full_key)
            if new_full_key != full_key:
                logger.info(f"match found: {full_key} -> {new_full_key}")
                net, sta, loc, chan = new_full_key.split(".")
                station_id = f"{net}.{sta}"
                channel = f"{loc}.{chan}"
                return station_id, channel

    # No match found
    return row["station_id"], row["channel"]


def rename_waveform_id(df: pd.DataFrame, rename_config: RenameConfig) -> pd.DataFrame:
    """Rename waveform identifiers according to specified rules.


    Attributes:
        df: DataFrame containing waveform identifiers.
        rename_config: Configuration object with regex patterns and time windows.

    Returns:
        DataFrame with updated station_id and channel columns.

    The rename_config object should contain the following attributes:
    - before: List of dictionaries (or []) with regex patterns to apply before everything else.
    - time_windows: List of dictionaries with time windows and regex patterns to apply.
    - after: List of dictionaries (or [] ) with regex patterns to apply after everything else.

    Each dictionary in the lists should have the following structure:
    {
        "regex_pattern": "replacement",
        ...
    }

    """

    def apply_regex_and_update(data, patterns, indices=None):
        """Apply regex to rows and update the DataFrame."""
        temp_result = data.apply(lambda row: apply_regex_to_row(row, patterns), axis=1)
        temp_df = pd.DataFrame(
            temp_result.tolist(), index=data.index, columns=["station_id", "channel"]
        )
        if indices is not None:
            df.loc[indices, ["station_id", "channel"]] = temp_df.values
        else:
            df.loc[:, ["station_id", "channel"]] = temp_df[["station_id", "channel"]].values

    # Apply "before" transformations
    if rename_config.before:
        logger.info("Applying 'before' transformations...")
        apply_regex_and_update(df, rename_config.before)

    # Apply transformations for each time window
    if rename_config.time_windows:
        for time_window in rename_config.time_windows:
            time_window_str = time_window.get("time_window")
            regex_patterns = time_window.get("regex", [])
            rows_to_update = pd.Series(
                True, index=df.index
            )  # Apply to all rows if no time window

            if time_window_str:
                start_time, end_time = [
                    pd.to_datetime(t).tz_localize(None) if t.lower() != "null" else None
                    for t in time_window_str.split("/")
                ]
                if start_time:
                    rows_to_update &= (
                        df["phase_time"].dt.tz_localize(None) >= start_time
                    )
                if end_time:
                    rows_to_update &= df["phase_time"].dt.tz_localize(None) <= end_time

            logger.info(
                f"Applying regex transformations for time window: {time_window_str}"
            )
            apply_regex_and_update(
                df.loc[rows_to_update],
                regex_patterns,
                indices=df.loc[rows_to_update].index,
            )

    # Apply "after" transformations
    if rename_config.after:
        logger.info("Applying 'after' transformations...")
        apply_regex_and_update(df, rename_config.after)

    return df
