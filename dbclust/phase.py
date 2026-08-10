#!/usr/bin/env python
import functools
import logging
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime
from typing import Dict
from typing import List
from typing import Literal
from typing import Optional
from typing import Union

import numpy as np
import pandas as pd
from dateutil import parser
from obspy import Inventory
from obspy import read_inventory
from obspy import UTCDateTime
from obspy.core.event import Comment
from obspy.core.event import CreationInfo
from obspy.core.event import Pick
from obspy.core.event import ResourceIdentifier
from obspy.core.event.base import WaveformStreamID

# default logger (uses hierarchical name for selective level control)
logger = logging.getLogger("dbclust.phase")

# Track stations already warned about missing coordinates (to avoid log spam)
_warned_missing_coords: set = set()


@dataclass
class Phase:
    network: str
    station: str
    location: str
    channel: str
    phase: str
    time: UTCDateTime
    time_uncertainty: Optional[float]
    proba: float
    info_sta: Union[Inventory, str]
    provenance: Optional[str] = None
    fallback_df: Optional[pd.DataFrame] = None
    evaluation: Optional[Literal["automatic", "manual"]] = None
    method: Optional[str] = None
    event_id: Optional[str] = None
    agency: Optional[str] = None
    coord: Optional[Dict[str, float]] = None

    """
    Represents a seismic phase with associated metadata and methods for processing.

    Attributes:
        network (str): Network code.
        station (str): Station code.
        location (str): Location code.
        channel (str): Channel code.
        phase (str): Phase type.
        time (UTCDateTime): Time of the phase.
        time_uncertainty (Optional[float]): Uncertainty in the phase time.
        proba (float): Probability associated with the phase.
        info_sta (Union[Inventory, str]): Station information, either as an Inventory object or FDSNWS URL.
        provenance: (Optional[str]): Provenance of the station information.
        fallback_df (Optional[pd.DataFrame]): Fallback station info dataframe.
        evaluation (Optional[Literal["automatic", "manual"]]): Evaluation mode of the phase.
        method (Optional[str]): Method used for phase determination.
        event_id (Optional[str]): Event identifier.
        agency (Optional[str]): Agency responsible for the phase.
        coord (Optional[Dict[str, float]]): Coordinates of the station.
    """

    def __post_init__(self) -> None:
        self.time = UTCDateTime(self.time)  # Ensure UTCDateTime type

        if not self.coord:
            self._fetch_coordinates()

    def is_p(self) -> bool:
        return bool(self.phase) and self.phase.upper().startswith("P")

    def is_s(self) -> bool:
        return bool(self.phase) and self.phase.upper().startswith("S")

    def _fetch_coordinates(self) -> None:
        """
        Fetch coordinates and channel information for the station associated with this phase.
        Uses either an Inventory object or FDSNWS service, with a fallback to a predefined DataFrame.

        Raises:
            ValueError: If coordinates cannot be fetched from any source.
        """
        # Determine the appropriate function and time range for fetching station info
        if isinstance(self.info_sta, Inventory):
            get_station_info = get_station_info_from_inventory
            time_search_begin = time_search_end = self.time
        else:
            get_station_info = get_station_info_from_fdsnws
            time_search_begin = UTCDateTime(
                self.time.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            )
            time_search_end = UTCDateTime(
                time_search_begin.replace(year=time_search_begin.year + 1)
            )

        # Fetch coordinates and channels from the primary source
        try:
            lat, lon, elev, loc, chans = get_station_info(
                self.network,
                self.station,
                str(time_search_begin),
                str(time_search_end),
                self.info_sta,
                loc=self.location,
                chan=self.channel,
            )
        except ValueError as e:
            logger.error(f"ValueError fetching coordinates: {e}")
            lat, lon, elev, loc, chans = None, None, None, None, None
        except Exception as e:
            logger.exception(f"Unexpected error during station info fetch: {e}")
            raise

        # If the primary source failed, use the fallback method
        if lat is None or lon is None:
            result = self._fallback_coordinates(
                self.network, self.station, self.time, self.fallback_df, chan=self.channel
            )
            if result is None:
                # No coordinates found, raise to skip this phase
                raise ValueError(f"No coordinates for {self.network}.{self.station}")
            lat, lon, elev, loc, chans = result
            self.provenance = "fallback"
        else:
            self.provenance = (
                "inventory" if isinstance(self.info_sta, Inventory) else "fdsnws"
            )

        # Update instance attributes with the fetched data
        self.coord = {"latitude": lat, "longitude": lon, "elevation": elev}
        self.location = loc or self.location
        if chans:
            new_chan = chans[-1] if "P" in self.phase.upper() else chans[0]
            if new_chan:  # Don't overwrite with empty string (e.g. from minimal CSV without channel column)
                self.channel = new_chan

    @staticmethod
    def _fallback_coordinates(
        network: str,
        station: str,
        time: UTCDateTime,
        fallback_df: Optional[pd.DataFrame] = None,
        chan: Optional[str] = None,
    ) -> Optional[tuple]:
        """
        Fetch fallback coordinates (latitude, longitude, elevation, location, channels)
        for a given station ID using a fallback DataFrame.

        Args:
            network (str): Network code of the station.
            station (str): Station code.
            time (UTCDateTime): Time of the query.
            fallback_df (Optional[pd.DataFrame]): Fallback DataFrame containing station information.
            chan (Optional[str]): Channel code hint; if provided, filters channels by the first
                two characters (e.g. "HHZ" → keeps only "HH?" channels).

        Returns:
            tuple: (latitude, longitude, elevation, location, channels) if found, None otherwise.
        """
        if fallback_df is not None:
            # Filter the DataFrame based on the station and time constraints
            df_filtered = fallback_df[
                (fallback_df["network"] == network)
                & (fallback_df["station"] == station)
                & (fallback_df["starttime"] <= time)
                & (fallback_df["endtime"] >= time)
            ]

            # Check if filtered DataFrame is not empty
            if not df_filtered.empty:
                # Extract required information from the first matching row
                lat, lon, elev, loc = df_filtered.iloc[0][
                    ["latitude", "longitude", "elevation", "location"]
                ]
                # Extract channels, optionally filtered by channel type prefix (first 2 chars)
                all_chans = df_filtered["channel"].tolist()
                if chan and len(chan) >= 2:
                    prefix = chan[:2]
                    filtered = [c for c in all_chans if c.startswith(prefix)]
                    all_chans = filtered if filtered else all_chans
                chans = sorted(all_chans)

                return lat, lon, elev, loc, chans

        # Log warning only once per station
        station_key = f"{network}.{station}"
        if station_key not in _warned_missing_coords:
            _warned_missing_coords.add(station_key)
            logger.warning(
                f"Cannot find coordinates for {station_key} in fallback dataframe [picks ignored]."
            )

        return None

    def to_pick(self) -> Pick:
        """Export the Phase object to an ObsPy Pick object."""
        waveform_id = WaveformStreamID(
            network_code=self.network,
            station_code=self.station,
            location_code=self.location,
            channel_code=self.channel,
        )

        # Build method_id only if method is a valid non-empty string
        method_id = None
        if self.method and isinstance(self.method, str) and self.method.strip():
            try:
                method_id = ResourceIdentifier(self.method)
            except (TypeError, ValueError) as e:
                logger.warning(
                    f"Invalid method_id '{self.method}' for {self.network}.{self.station}: {e}"
                )

        pick = Pick(
            time=self.time,
            waveform_id=waveform_id,
            phase_hint=self.phase,
            method_id=method_id,
        )
        if self.evaluation and isinstance(self.evaluation, str):
            pick.evaluation_mode = self.evaluation

        pick.time_errors.uncertainty = self.time_uncertainty
        pick.creation_info = CreationInfo(agency_id=self.agency, author=self.event_id)

        return pick

    def __eq__(self, obj: object) -> bool:
        return isinstance(obj, Phase) and hash(self) == hash(obj)

    def __hash__(self) -> int:
        return int(
            hash(
                (
                    self.network,
                    self.station,
                    self.phase,
                    self.time.datetime,
                )
            )
        )

    def __repr__(self) -> str:
        return f"{self.network}.{self.station}.{self.location}.{self.channel}: {self.phase} {self.time} {self.proba:.2f} {self.event_id}"

    def __lt__(self, obj: "Phase") -> bool:
        return self.time < obj.time

    def show_all(self) -> None:
        print(
            f"{self.network}.{self.station}.{self.location}.{self.channel}: "
            f"event_id={self.event_id or 'N/A'}, evaluation={self.evaluation or 'N/A'}, "
            f"method={self.method or 'N/A'}, agency={self.agency or 'N/A'}"
        )
        print(f"    Phase: {self.phase} at {self.time} with proba={self.proba:.3f}")
        if self.coord:
            print(
                f"    Coordinates: lat={self.coord['latitude']:.4f}, "
                f"lon={self.coord['longitude']:.4f}, elev={self.coord['elevation']:.1f}, "
                f"provenance={self.provenance}"
            )
        else:
            print("    No coordinates found.")


def inventory2df(inventory: Inventory) -> pd.DataFrame:
    """Convert inventory to dataframe

    Args:
        inventory (Inventory): inventory to convert

    Returns:
        pd.DataFrame: dataframe
    """
    channels_info = []
    for network in inventory:
        for station in network:
            for channel in station.channels:

                if channel.response and channel.response.instrument_sensitivity:
                    scale = channel.response.instrument_sensitivity.value
                    scale_freq = channel.response.instrument_sensitivity.frequency
                    scale_units = channel.response.instrument_sensitivity.input_units
                else:
                    scale = scale_freq = scale_units = None

                if channel.sensor:
                    sensor_description = channel.sensor.description
                else:
                    sensor_description = None

                channel_info = {
                    "Network": network.code,
                    "Station": station.code,
                    "Location": channel.location_code,
                    "Channel": channel.code,
                    "Latitude": station.latitude,
                    "Longitude": station.longitude,
                    "Elevation": station.elevation,
                    "Depth": channel.depth,
                    # "Azimuth": channel.azimuth,
                    # "Dip": channel.dip,
                    # "SensorDescription": sensor_description,
                    # "Scale": scale,
                    # "ScaleFreq": scale_freq,
                    # "ScaleUnits": scale_units,
                    "SampleRate": channel.sample_rate,
                    "StartTime": station.start_date,
                    "EndTime": station.end_date,
                }
                # Add channel information to the list
                channels_info.append(channel_info)

    # Create a pandas DataFrame from the list of dictionaries
    df = pd.DataFrame(channels_info, dtype=str)
    if df.empty:
        return df

    df["SampleRate"] = df["SampleRate"].apply(np.float32)
    df = df.fillna("")
    df["Latitude"] = df["Latitude"].apply(np.float32)
    df["Longitude"] = df["Longitude"].apply(np.float32)
    df["Elevation"] = df["Elevation"].apply(np.float32)
    df["Location"] = df["Location"].astype(str)
    df["Channel"] = df["Channel"].astype(str)

    df = df.drop_duplicates()

    return df


def extract_station_coords(
    inventory: Optional["Inventory"] = None,
    fallback_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Extract unique station coordinates from inventory and/or fallback CSV.

    This function combines station coordinates from an ObsPy Inventory object
    and a fallback DataFrame (from CSV files). Priority is given to inventory
    data when both sources contain the same station.

    Args:
        inventory: ObsPy Inventory object containing station metadata.
        fallback_df: DataFrame from fallback CSV with columns:
            network, station, latitude, longitude.

    Returns:
        DataFrame with columns: network, station, latitude, longitude.
        Each row represents a unique network.station combination.
    """
    coords_list = []

    # Extract from inventory if provided
    if inventory is not None:
        inv_df = inventory2df(inventory)
        if not inv_df.empty:
            # Keep only unique network.station with their coordinates
            inv_coords = (
                inv_df[["Network", "Station", "Latitude", "Longitude"]]
                .drop_duplicates(subset=["Network", "Station"])
                .rename(
                    columns={
                        "Network": "network",
                        "Station": "station",
                        "Latitude": "latitude",
                        "Longitude": "longitude",
                    }
                )
            )
            coords_list.append(inv_coords)

    # Extract from fallback DataFrame if provided
    if fallback_df is not None and not fallback_df.empty:
        fallback_coords = fallback_df[
            ["network", "station", "latitude", "longitude"]
        ].drop_duplicates(subset=["network", "station"])
        coords_list.append(fallback_coords)

    if not coords_list:
        return pd.DataFrame(columns=["network", "station", "latitude", "longitude"])

    # Combine all sources
    combined = pd.concat(coords_list, ignore_index=True)

    # Remove duplicates, keeping first occurrence (inventory has priority)
    combined = combined.drop_duplicates(subset=["network", "station"], keep="first")

    # Ensure numeric types for coordinates
    combined["latitude"] = pd.to_numeric(combined["latitude"], errors="coerce")
    combined["longitude"] = pd.to_numeric(combined["longitude"], errors="coerce")

    # Remove rows with invalid coordinates
    combined = combined.dropna(subset=["latitude", "longitude"])

    logger.info(f"Extracted coordinates for {len(combined)} unique stations")

    return combined.reset_index(drop=True)


def get_missing_info_from_df(
    df: pd.DataFrame, loc: Optional[str], chan: Optional[str]
) -> tuple:
    if loc is not None and chan is not None:
        df = df.loc[df["Location"] == loc, :]
        # channel is specified: use it to filter
        re_chan = f"^{chan}"
        df = df[df["Channel"].str.contains(re_chan, regex=True)]
    else:
        # Try to guess the channels choosing the highest sampling rate
        try:
            # sometimes SampleRate is not defined
            max_sample_rate = df["SampleRate"].max()
        except (KeyError, TypeError):
            pass
        else:
            df = df[df["SampleRate"] == max_sample_rate]
        df = df.sort_values(by="StartTime")[:3]

    df = df.sort_values(by=["StartTime", "Channel"])

    if len(df) == 0:
        return (None, None, None, None, None)
    elif len(df) < 3:
        rows = df["Channel"].iloc[0]
        new_chans = [rows]
    else:
        rows = df["Channel"].iloc[:3]
        new_chans = rows.tolist()

    new_loc = df["Location"].iloc[0]

    return (
        df.iloc[0]["Latitude"],
        df.iloc[0]["Longitude"],
        df.iloc[0]["Elevation"],
        new_loc,
        new_chans,
    )


def get_station_info_from_inventory(
    network: str,
    station: str,
    time_search_begin: str,
    time_search_end: str,
    inventory: Inventory,
    loc: Optional[str] = None,
    chan: Optional[str] = None,
) -> tuple:
    """
    Get station coordinates and channel information from an inventory object.

    Args:
        network (str): Station's network code.
        station (str): Station's name.
        time_search_begin (str): Start time of the search in ISO format.
        time_search_end (str): End time of the search in ISO format.
        inventory (Inventory): ObsPy inventory object containing station data.
        loc (Optional[str], optional): Location code. Defaults to None.
        chan (Optional[str], optional): Channel code. Defaults to None.

    Returns:
        List[Optional[float]]: A list containing:
            - Latitude (float)
            - Longitude (float)
            - Elevation (float)
            - Location code (str)
            - List of channel codes (List[str])
    """
    logger.debug(f"Fetching station info from inventory for {network}.{station}...")

    # Prepare regex pattern for channel and location for inventory.select()
    re_chan = chan[:2] + "?" if chan else "*"
    select_loc = loc if loc is not None else "*"

    try:
        # Select matching station entries from the inventory
        inv = inventory.select(
            network=network,
            station=station,
            location=select_loc,
            channel=re_chan,
            starttime=time_search_begin,
            # endtime=time_search_end,
        )
    except Exception as e:
        logger.error(f"Error selecting data from inventory: {e}")
        return (None, None, None, None, [])

    # Convert inventory to a DataFrame for easier processing
    df = inventory2df(inv)
    if df.empty:
        logger.debug(f"No matching data found in inventory for {network}.{station}.")
        return (None, None, None, None, [])

    # Use helper function to filter and extract required data
    return get_missing_info_from_df(df, loc, chan)


@functools.lru_cache(maxsize=None)
def get_station_info_from_fdsnws(
    network: str,
    station: str,
    time_search_begin: str,
    time_search_end: str,
    fdsnws_station_url: str,
    loc: Optional[str] = None,
    chan: Optional[str] = None,
) -> tuple:
    """Get station coordinates from fdsnws, find all channels

    Args:
        network (str): station's network
        station (str): stations's name
        time_search (Union[datetime, pd.Timestamp]): time
        fdsnws_station_url (str): station fdsnws url
        loc (Optional[str], optional): location code. Defaults to None.
        chan (Optional[str], optional): channel. Defaults to None.

    Returns:
        tuple: station's latitude, longitude, elevation, location, channels
    """
    logger.debug(f"Getting station info from fdsnws: {network}.{station}")
    # if not time_search:
    #     time_search = UTCDateTime.now()

    url = (
        f"{fdsnws_station_url}/fdsnws/station/1/query?"
        f"network={network}&"
        f"station={station}&"
        f"starttime={time_search_begin}&"
        f"endtime={time_search_end}&"
        f"format=text&"
        f"level=channel"
    )

    try:
        df = pd.read_csv(url, sep="|", skipinitialspace=True, dtype=str)
    except Exception as e:
        logger.warning(f"Failed to load station map from {url}: {e}", exc_info=True)
        return (None, None, None, None, None)

    # Network|Station|Location|Channel|Latitude|Longitude|Elevation|Depth|
    df.columns = df.columns.str.replace("#", "")
    df.columns = [x.strip() for x in df.columns]
    df = df.fillna("")
    df["Latitude"] = df["Latitude"].apply(np.float32)
    df["Longitude"] = df["Longitude"].apply(np.float32)
    df["Elevation"] = df["Elevation"].apply(np.float32)
    df["Location"] = df["Location"].astype(str)
    df["Channel"] = df["Channel"].astype(str)
    df["SampleRate"] = df["SampleRate"].apply(np.float32)

    return get_missing_info_from_df(df, loc, chan)


def import_phases(
    df: Optional[pd.DataFrame] = None,
    P_proba_threshold: float = 0,
    S_proba_threshold: float = 0,
    P_uncertainty: Optional[float] = 0.1,
    S_uncertainty: Optional[float] = 0.2,
    info_sta: Optional[Union[Inventory, str]] = None,
    fallback_df: Optional[pd.DataFrame] = None,
) -> List[Phase]:
    """
    Import phases from a DataFrame and filter them based on given thresholds.

    Parameters:
        df (pd.DataFrame, optional): DataFrame containing phase information. Default is None.
        P_proba_threshold (float, optional): Probability threshold for P phases. Default is 0.
        S_proba_threshold (float, optional): Probability threshold for S phases. Default is 0.
        P_uncertainty (Optional[float], optional): Uncertainty for P phases. Default is 0.1.
        S_uncertainty (Optional[float], optional): Uncertainty for S phases. Default is 0.2.
        info_sta (Optional[Union[Inventory, str]], optional): Station information. Default is None.
        fallback_df (Optional[pd.DataFrame], optional): Fallback station information DataFrame. Default is None.

    Returns:
        List[Phase]: List of Phase objects created from the DataFrame.
    """
    phases: List[Phase] = []

    # Validate DataFrame
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        logger.info("Input DataFrame is None, not a DataFrame, or empty.")
        return phases

    # Check required columns
    required_columns = ["station_id", "phase_type", "phase_time", "phase_score"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        logger.error(f"Missing required columns: {missing_columns}")
        logger.error(f"Available columns: {list(df.columns)}")
        return phases

    # Handle optional "channel" column
    if "channel" in df.columns:
        channel = df["channel"].fillna("").astype(str)

        # Append channel only if at least one non-empty value exists
        if (channel.str.strip() != "").any():
            df = df.copy()
            df["station_id"] = df["station_id"].astype(str) + "." + channel

    # Filter by phase score thresholds
    df_before = df.copy()
    df = df.loc[
        ~((df["phase_type"] == "P") & (df["phase_score"] < P_proba_threshold))
        & ~((df["phase_type"] == "S") & (df["phase_score"] < S_proba_threshold))
    ]

    # Iterate through filtered rows
    for row in df.itertuples(index=False):
        try:
            # Split station ID into components
            components = str(row.station_id).split(".")
            net = components[0] if len(components) > 0 else None
            sta = components[1] if len(components) > 1 else None
            loc = components[2] if len(components) > 2 else None
            chan = components[3] if len(components) > 3 else None
        except ValueError as e:
            logger.error(f"Error parsing station_id: {row.station_id}. Error: {e}")
            continue

        # Extract optional fields
        evaluation = getattr(row, "phase_evaluation", None)
        method = getattr(row, "phase_method", None)
        event_id = getattr(row, "event_id", None)
        if pd.isna(event_id):
            event_id = None
        agency = getattr(row, "agency", None)
        if pd.isna(agency):
            agency = None

        # Time uncertainty
        phase_type = str(row.phase_type).upper()
        time_uncertainty = (
            P_uncertainty if phase_type.startswith("P") else S_uncertainty
        )

        # Create Phase object
        try:
            myphase = Phase(
                network=net,
                station=sta,
                location=loc,
                channel=chan[:2] if chan else None,
                phase=row.phase_type,
                time=row.phase_time,
                time_uncertainty=time_uncertainty,
                proba=row.phase_score,
                evaluation=evaluation,
                method=method,
                event_id=event_id,
                agency=agency,
                info_sta=info_sta,
                fallback_df=fallback_df,
            )
        except ValueError as e:
            logger.debug(f"ValueError creating Phase object for row: {row}. Error: {e}")
            continue
        except Exception as e:
            logger.exception(f"Unexpected error creating Phase object: {e}")
            raise

        # Append phase outside try blocks
        phases.append(myphase)

        # Optionally show phase details
        if logger.level == logging.DEBUG:
            myphase.show_all()

    # Log cache information if applicable
    if isinstance(info_sta, str):
        logger.info(get_station_info_from_fdsnws.cache_info())

    return phases
