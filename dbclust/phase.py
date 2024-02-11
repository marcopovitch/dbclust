#!/usr/bin/env python
import functools
import logging
import sys
from datetime import datetime
from typing import List
from typing import Optional
from typing import Union

import numpy as np
import pandas as pd
from icecream import ic
from obspy import Inventory
from obspy import UTCDateTime
from tqdm import tqdm

# default logger
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger("phases")
logger.setLevel(logging.INFO)


class Phase(object):
    """Phase class"""

    def __init__(
        self,
        net: str = None,
        sta: str = None,
        coord: dict = None,
        time_search: Optional[Union[datetime, pd.Timestamp]] = None,
        info_sta: Optional[Union[Inventory, str]] = None,
    ):
        self.network = net
        self.station = sta
        self.phase = None
        self.time = None
        self.proba = None

        if coord:
            self.coord = coord
            return
        else:
            self.coord = {"latitude": None, "longitude": None, "elevation": None}

        if not self.network or not self.station:
            return

        if time_search:
            time_search = str(time_search)  # madatorry to use lru_cache

        if type(info_sta) == Inventory:
            get_station_info = self.get_station_info_from_inventory
        else:
            # url
            get_station_info = self.get_station_info_from_fdsnws
            # hack to get benefit from lru_cache
            time_search = None

        (lat, lon, elev) = get_station_info(
            self.network,
            self.station,
            time_search,
            info_sta,
        )

        if lat == None or lon == None:
            raise Exception(f"Can't find coordinates for {net}.{sta}")

        (
            self.coord["latitude"],
            self.coord["longitude"],
            self.coord["elevation"],
        ) = (lat, lon, elev)

    def __eq__(self, obj):
        if self.__hash__() == obj.__hash__():
            return True
        else:
            False

    def __hash__(self):
        return hash(
            (self.network, self.station, self.phase, self.time.datetime, self.proba)
        )

    def __repr__(self):
        return (
            f"{self.network}.{self.station} {self.phase} {self.time} {self.proba:.3f}"
        )

    # needed for lru_cache
    def __lt__(self, obj):
        return (self.time) < (obj.time)

    @staticmethod
    # @functools.lru_cache(maxsize=None)
    def get_station_info_from_inventory(
        network: str,
        station: str,
        time_search: Union[datetime, pd.Timestamp],
        inventory: Inventory,
    ) -> list:
        inv = inventory.select(
            network=network,
            station=station,
            starttime=UTCDateTime(str(time_search)[:10]),
        )
        channels = inv.get_contents()["channels"]
        if len(channels) == 0:
            logger.debug(
                f"Can't find coordinates for {network}.{station} at {time_search}"
            )
            return None, None, None

        coords = inv.get_coordinates(channels[0])
        # elevation is in meters
        return coords["latitude"], coords["longitude"], coords["elevation"]

    @staticmethod
    @functools.lru_cache(maxsize=None)
    def get_station_info_from_fdsnws(
        network: str,
        station: str,
        time_search: Union[datetime, pd.Timestamp],
        fdsnws_station_url: str,
    ) -> list:
        logger.debug(f"Getting station info from {network}.{station}")
        if not time_search:
            time_search = UTCDateTime.now()

        # time_search = str(time_search)[:10]
        url = (
            f"{fdsnws_station_url}/fdsnws/station/1/query?"
            + "network=%s&" % network
            + "station=%s&" % station
            + "format=text"
        )
        #   + "starttime=%sT00:00:00&" % time_search

        try:
            df = pd.read_csv(url, sep="|", skipinitialspace=True)
        except BaseException as e:
            # logger.error("The exception: {}".format(e))
            # logger.debug(url)
            return None, None, None

        df.columns = df.columns.str.replace("#", "")
        df.columns = [x.strip() for x in df.columns]
        df["Latitude"] = df["Latitude"].apply(np.float32)
        df["Longitude"] = df["Longitude"].apply(np.float32)
        df["Elevation"] = df["Elevation"].apply(np.float32)

        return df.iloc[0]["Latitude"], df.iloc[0]["Longitude"], df.iloc[0]["Elevation"]

    def set_phase_info(
        self,
        phase: str,
        time: str,
        proba: float,
        eventid: str = None,
        agency: str = None,
    ) -> None:
        self.phase = phase
        self.time = UTCDateTime(time)
        self.proba = float(proba)
        self.eventid = eventid
        self.agency = agency

    def show_all(self) -> None:
        if self.eventid:
            print(f"{self.network}.{self.station}: from {self.eventid}")
        else:
            print(f"{self.network}.{self.station}:")
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


def import_phases(
    df: pd.DataFrame = None,
    P_proba_threshold: float = 0,
    S_proba_threshold: float = 0,
    info_sta: Optional[Union[Inventory, str]] = None,
) -> List[Phase]:
    """
    Read phaseNet dataframe picks.
    Returns a list of Phase objects.
    """
    phases = []

    if df is None or not isinstance(df, pd.DataFrame) or not len(df):
        ic(df)
        return None

    if (
        "station_id" not in df.columns
        and "phase_type" not in df.columns
        and "phase_score" not in df.columns
        and "phase_time" not in df.columns
    ):
        logger.error("No phasenet header found")
        return None

    df = df.loc[~((df["phase_type"] == "P") & (df["phase_score"] < P_proba_threshold))]
    df = df.loc[~((df["phase_type"] == "S") & (df["phase_score"] < S_proba_threshold))]
    #df[["net", "sta"]] = df["station_id"].astype(str).str.split(".", n=1, expand=True)

    for row in df.itertuples(index=False):
        # ic(row)
        if "eventid" in df.columns:
            eventid = row.eventid
        else:
            eventid = None

        if "agency" in df.columns:
            agency = row.agency
        else:
            agency = None

        net, sta = row.station_id.split(".")[:2]

        try:
            myphase = Phase(
                net=net,
                sta=sta,
                time_search=row.phase_time,
                info_sta=info_sta,
            )
        except Exception as e:
            logger.warning(e)
            continue

        myphase.set_phase_info(
            row.phase_type,
            row.phase_time,
            row.phase_score,
            eventid=eventid,
            agency=agency,
        )
        phases.append(myphase)
        if logger.level == logging.DEBUG:
            myphase.show_all()
    if type(info_sta) == str:
        logger.info(Phase.get_station_info_from_fdsnws.cache_info())
    return phases


def import_eqt_phases(df=None, P_proba_threshold=0, S_proba_threshold=0):
    """
    Read EQT dataframe picks.
    Returns a list of Phase objects.
    """
    phases = []

    if (
        "station_lat" not in df.columns
        and "station_lon" not in df.columns
        and "station_elv" not in df.columns
        and "p_probability" not in df.columns
        and "s_probability" not in df.columns
        and "p_arrival_time" not in df.columns
        and "s_arrival_time" not in df.columns
    ):
        logger.error("No EQT header found")
        return None

    for i in tqdm(range(len(df))):
        net = df.iloc[i]["network"]
        sta = df.iloc[i]["station"]
        coord = {
            "latitude": df.iloc[i]["station_lat"],
            "longitude": df.iloc[i]["station_lon"],
            "elevation": df.iloc[i]["station_elv"],
        }
        p_proba = df.iloc[i]["p_probability"]
        s_proba = df.iloc[i]["s_probability"]

        if p_proba and p_proba >= P_proba_threshold:
            phase_type = "P"
            phase_time = UTCDateTime(df.iloc[i]["p_arrival_time"])
            myphase = Phase(net=net, sta=sta, coord=coord)
            myphase.set_phase_info(
                phase_type,
                phase_time,
                p_proba,
            )
            phases.append(myphase)
            if logger.level == logging.DEBUG:
                myphase.show_all()

        if s_proba and s_proba >= S_proba_threshold:
            phase_type = "S"
            phase_time = UTCDateTime(df.iloc[i]["s_arrival_time"])
            myphase = Phase(net=net, sta=sta, coord=coord)
            myphase.set_phase_info(
                phase_type,
                phase_time,
                s_proba,
            )
            phases.append(myphase)
            if logger.level == logging.DEBUG:
                myphase.show_all()
    return phases


def _test_phasenet_import():
    picks_file = "../test/picksSS.csv"
    logger.info(f"Opening {picks_file} file.")
    try:
        df = pd.read_csv(picks_file, parse_dates=["phase_time"])
    except Exception as e:
        logger.error(e)
        sys.exit()

    logger.info(f"Read {len(df)} phases.")

    phases = import_phases(
        df,
        P_proba_threshold=0.8,
        S_proba_threshold=0.5,
        info_sta="http://10.0.1.36:8080",
        # info_sta="http://ws.resif.fr",
    )


def _test_eqt_import():
    picks_file = "../test/EQT-2022-09-10.csv"
    logger.info(f"Opening {picks_file} file.")
    try:
        df = pd.read_csv(picks_file, parse_dates=["p_arrival_time", "s_arrival_time"])
    except Exception as e:
        logger.error(e)
        sys.exit()

    logger.info(f"Read {len(df)} phases.")

    phases = import_eqt_phases(
        df,
        P_proba_threshold=0.8,
        S_proba_threshold=0.5,
    )


if __name__ == "__main__":
    logger.setLevel(logging.INFO)
    _test_phasenet_import()
    _test_eqt_import()
