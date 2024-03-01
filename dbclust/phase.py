#!/usr/bin/env python
import functools
import logging
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import List
from typing import Literal
from typing import Optional
from typing import Union

import numpy as np
import pandas as pd
from dateutil import parser
from icecream import ic
from obspy import Inventory
from obspy import read_inventory
from obspy import UTCDateTime


# default logger
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger("phase")
logger.setLevel(logging.DEBUG)


@dataclass
class Phase:
    """_summary_

    Returns:
        _type_: _description_
    """

    network: str
    station: str
    location: str
    channel: str
    phase: str
    time: UTCDateTime
    proba: float
    info_sta: Union[Inventory, str]
    evaluation: Literal["automatic", "manual", None] = None
    method: str = None
    event_id: str = None
    agency: str = None
    coord: Optional[dict] = None

    def __post_init__(self) -> None:
        if self.coord:
            return

        self.time = UTCDateTime(self.time)

        if type(self.info_sta) == Inventory:
            get_station_info = get_station_info_from_inventory
            time_search_begin = UTCDateTime(self.time)
            time_search_end = UTCDateTime(self.time)
        else:
            get_station_info = get_station_info_from_fdsnws
            # Don't be too precise to benefit from lru_cache
            # time_search_begin and time_search_begin type must be string
            time_search = self.time
            time_search_begin = UTCDateTime(
                time_search.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            )
            time_search_end = UTCDateTime(
                # time_search_begin.replace(month=time_search_begin.month + 1, day=1)
                time_search_begin.replace(year=time_search_begin.year + 1)
            )
            time_search_begin = str(time_search_begin)
            time_search_end = str(time_search_end)

        (lat, lon, elev, chans) = get_station_info(
            self.network,
            self.station,
            time_search_begin,
            time_search_end,
            self.info_sta,
            loc=self.location,
            chan=self.channel,
        )
        if lat == None or lon == None:
            raise ValueError(
                f"Can't find coordinates for {self.network}.{self.station}.{self.location}.{self.channel} "
                f"from {time_search_begin} to {time_search_end}."
            )
        self.coord = {"latitude": lat, "longitude": lon, "elevation": elev}

        # chans order is in lexicographic order
        if "P" in self.phase:
            self.channel = chans[-1]
        else:
            self.channel = chans[0]

    def __eq__(self, obj: object) -> bool:
        if self.__hash__() == obj.__hash__():
            return True
        else:
            return False

    def __hash__(self) -> int:
        return hash(
            (self.network, self.station, self.phase, self.time.datetime, self.proba)
        )

    def __repr__(self) -> str:
        return f"{self.network}.{self.station}.{self.channel}: {self.phase} {self.time} {self.proba:.3f}"

    # needed for lru_cache
    def __lt__(self, obj: "Phase") -> bool:
        return (self.time) < (obj.time)

    def show_all(self) -> None:
        if self.event_id:
            print(
                f"{self.network}.{self.station}.{self.location}.{self.channel}: from {self.event_id}"
            )
        else:
            print(f"{self.network}.{self.station}.{self.location}.{self.channel}:")

        print(
            f"    evaluation is {self.evaluation}, method: {self.method}, agency: {self.agency}"
        )
        if (
            self.coord["latitude"] is not None
            and self.coord["longitude"] is not None
            and self.coord["elevation"] is not None
        ):
            print(
                f"    lat={self.coord['latitude']:.4f}, lon={self.coord['longitude']:.4f}, elev={self.coord['elevation']:.1f}"
            )
            print(f"    phase={self.phase}, time={self.time} proba={self.proba:.3f}")
        else:
            print("No coordinates found !")


def get_station_info_from_inventory(
    network: str,
    station: str,
    time_search_begin: str,
    time_search_end: str,
    inventory: Inventory,
    loc: Optional[str] = None,
    chan: Optional[str] = None,
) -> list:
    """Get station coordinates from inventory

    Args:
        network (str): station's network
        station (str): stations's name
        time_search (Union[datetime, pd.Timestamp]): time
        inventory (Inventory): stations inventory
        loc (Optional[str], optional): location code. Defaults to None.
        chan (Optional[str], optional): channel. Defaults to None.

    Returns:
        list: station's latitude, longitude, elevation
    """
    logger.debug(f"Getting station info from inventory: {network}.{station}")

    if chan:
        re_chan = chan[:2] + "?"
    else:
        re_chan = '*'

    inv = inventory.select(
        network=network,
        station=station,
        location=loc,
        channel=re_chan,
        starttime=time_search_begin,
        endtime=time_search_end,
    )
    channels = inv.get_contents()["channels"]
    if len(channels) == 0:
        return [None] * 4

    my_chan = sorted(set([c.split(".")[3] for c in channels]))
    coords = inv.get_coordinates(channels[0])
    # elevation is in meters
    return coords["latitude"], coords["longitude"], coords["elevation"], my_chan


@functools.lru_cache(maxsize=None)
def get_station_info_from_fdsnws(
    network: str,
    station: str,
    time_search_begin: str,
    time_search_end: str,
    fdsnws_station_url: str,
    loc: Optional[str] = None,
    chan: Optional[str] = None,
) -> list:
    """Get station coordinates from fdsnws, find all channels

    Args:
        network (str): station's network
        station (str): stations's name
        time_search (Union[datetime, pd.Timestamp]): time
        fdsnws_station_url (str): station fdsnws url
        loc (Optional[str], optional): location code. Defaults to None.
        chan (Optional[str], optional): channel. Defaults to None.

    Returns:
        list: station's latitude, longitude, elevation, channels
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
    # ic(url)
    # print(url)

    try:
        df = pd.read_csv(url, sep="|", skipinitialspace=True)
    except BaseException as e:
        # logger.error("The exception: {}".format(e))
        # logger.debug(url)
        return [None] * 4

    # Network|Station|Location|Channel|Latitude|Longitude|Elevation|Depth|
    df.columns = df.columns.str.replace("#", "")
    df.columns = [x.strip() for x in df.columns]
    df["Latitude"] = df["Latitude"].apply(np.float32)
    df["Longitude"] = df["Longitude"].apply(np.float32)
    df["Elevation"] = df["Elevation"].apply(np.float32)
    df["Channel"] = df["Channel"].astype(str)

    df["Location"] = df["Location"].fillna("")
    df.loc[df["Location"] == loc, :]

    # get high sample rate first
    # df.sort_values(by=["SampleRate", "StartTime"], ascending=False, inplace=True)

    if chan:
        re_chan = f"^{chan}"
        df = df[df["Channel"].str.contains(re_chan, regex=True)]
        if len(df) == 0:
            return [None] * 4
        elif len(df) < 3:
            rows = df["Channel"].iloc[0]
            new_chans = [rows]
        else:
            rows = df["Channel"].iloc[:3]
            new_chans = rows.tolist()
    else:
        new_chans = [chan] * 3

    return (
        df.iloc[0]["Latitude"],
        df.iloc[0]["Longitude"],
        df.iloc[0]["Elevation"],
        new_chans,
    )


def import_phases(
    df: pd.DataFrame = None,
    P_proba_threshold: float = 0,
    S_proba_threshold: float = 0,
    info_sta: Optional[Union[Inventory, str]] = None,
) -> List[Phase]:
    """Read phaseNet dataframe picks

    Args:
        df (pd.DataFrame, optional): _description_. Defaults to None.
        P_proba_threshold (float, optional): P filter threshold. Defaults to 0 ie. no filter.
        S_proba_threshold (float, optional): S filter threshold. Defaults to 0 ie. no filter.
        info_sta (Optional[Union[Inventory, str]], optional): How to get stations information.

    Returns:
        List[Phase]: returns a list of Phase objects
    """
    phases = []

    if df is None or not isinstance(df, pd.DataFrame) or not len(df):
        # ic(df)
        return None

    needed_columns = ["station_id", "phase_type", "phase_time", "phase_score"]
    for c in needed_columns:
        if c not in df.columns:
            logger.error("Missing columns in phase file !")
            logger.error(df.columns)
            return None

    if "channel" in df.columns and df["channel"] is not df.empty:
        df["station_id"] = df["station_id"].str.cat(df["channel"], sep=".")

    ic(df)

    # use phase score threshold filters
    df = df.loc[~((df["phase_type"] == "P") & (df["phase_score"] < P_proba_threshold))]
    df = df.loc[~((df["phase_type"] == "S") & (df["phase_score"] < S_proba_threshold))]

    for row in df.itertuples(index=False):
        # ic(row)
        if "phase_evaluation" in df.columns and type(row.phase_evaluation) is str:
            evaluation = row.phase_evaluation
        else:
            evaluation = None

        if "phase_method" in df.columns and type(row.phase_method) is str:
            method = row.phase_method
        else:
            method = None

        if "event_id" in df.columns and type(row.event_id) is str:
            event_id = row.event_id
        else:
            event_id = None

        if "agency" in df.columns and type(row.agency) is str:
            agency = row.agency
        else:
            agency = None

        ic(row.station_id)

        try:
            net, sta, loc, chan = row.station_id.split(".")[:4]
        except:
            net, sta = row.station_id.split(".")[:2]
            loc = None
            chan = None

        try:
            myphase = Phase(
                network=net,
                station=sta,
                location=loc,
                channel=chan,
                phase=row.phase_type,
                time=row.phase_time,
                proba=row.phase_score,
                evaluation=evaluation,
                method=method,
                event_id=event_id,
                agency=agency,
                info_sta=info_sta,
            )
        except ValueError as e:
            logger.error(e)
            continue
        except Exception as e:
            raise

        phases.append(myphase)
        if logger.level == logging.DEBUG:
            myphase.show_all()

    if type(info_sta) == str:
        logger.info(get_station_info_from_fdsnws.cache_info())
    return phases


def _test_import(picks_file, info_sta):
    logger.info(f"Opening {picks_file} file.")
    try:
        df = pd.read_csv(picks_file, parse_dates=["phase_time"])
    except Exception as e:
        logger.error(e)
        sys.exit()

    # fixme: samples format are outdated ...
    # df["channel"] = df["station_id"].map(lambda x: ".".join(x.split(".")[2:4]))
    # df["station_id"] = df["station_id"].map(lambda x: ".".join(x.split(".")[:2]))

    logger.info(f"Read {len(df)} phases.")

    phases = import_phases(
        df,
        P_proba_threshold=0.8,
        S_proba_threshold=0.5,
        info_sta=info_sta,
    )


if __name__ == "__main__":
    logger.setLevel(logging.DEBUG)
    _test_import("../samples/renass.csv", "http://10.0.1.36:8080")
    _test_import("../samples/phasenet.csv", "http://10.0.1.36:8080")

    inventory_files = [
        "/Users/marc/Data/DBClust/france.2016.01/inventory/all_from_renass.inv.xml",
        "/Users/marc/Data/DBClust/france.2016.01/inventory/inventory-RENASS-LDG.xml",
    ]
    inventory = Inventory()
    for f in inventory_files:
        logger.info(f"Reading inventory file {f}")
        inventory.extend(read_inventory(f))

    _test_import("../samples/renass.csv", inventory)
    _test_import("../samples/phasenet.csv", inventory)
