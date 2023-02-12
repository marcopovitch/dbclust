#!/usr/bin/env python
import sys
import logging
import functools
import numpy as np
import pandas as pd
from tqdm import tqdm
from obspy import UTCDateTime

# default logger
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger("phases")
logger.setLevel(logging.INFO)


class Phase(object):
    def __init__(
        self,
        net=None,
        sta=None,
        coord=None,
        fdsnws_time_search=None,
        fdsnws_station_url="http://10.0.1.36:8080",
    ):
        self.network = net
        self.station = sta
        self.phase = None
        self.time = None
        self.proba = None
        if coord:
            self.coord = coord
            return

        self.coord = {"latitude": None, "longitude": None, "elevation": None}
        if fdsnws_time_search:
            fdsnws_time_search = str(fdsnws_time_search)  # madatorry to use lru_cache

        if self.network and self.station:
            (
                self.coord["latitude"],
                self.coord["longitude"],
                self.coord["elevation"],
            ) = self.get_station_info(
                self.network,
                self.station,
                fdsnws_time_search,
                fdsnws_station_url,
            )

    def __repr__(self):
        return (
            f"{self.network}.{self.station} {self.phase} {self.time} {self.proba:.3f}"
        )

    # needed for lru_cache
    def __lt__(self, obj):
        return (self.time) < (obj.time)

    @staticmethod
    @functools.lru_cache(maxsize=None)
    def get_station_info(network, station, time_search, fdsnws_station_url):
        logger.debug(f"Getting station info from {network}.{station}")
        if not time_search:
            time_search = UTCDateTime.now()

        time_search = str(time_search)[:10]
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
            logger.error("The exception: {}".format(e))
            logger.debug(url)
            return None, None, None

        df.columns = df.columns.str.replace("#", "")
        df.columns = [x.strip() for x in df.columns]
        df["Latitude"] = df["Latitude"].apply(np.float32)
        df["Longitude"] = df["Longitude"].apply(np.float32)
        df["Elevation"] = df["Elevation"].apply(np.float32)

        return df.iloc[0]["Latitude"], df.iloc[0]["Longitude"], df.iloc[0]["Elevation"]

    def set_phase_info(self, phase, time, proba):
        self.phase = phase
        self.time = time
        self.proba = proba

    def show_all(self):
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
    df=None,
    P_proba_threshold=0,
    S_proba_threshold=0,
    fdsnws_station_url="http://10.0.1.36:8080",
):
    """
    Read phaseNet dataframe picks.
    Returns a list of Phase objects.
    """
    phases = []

    if (
        "station_id" not in df.columns
        and "phase_type" not in df.columns
        and "phase_score" not in df.columns
        and "phase_time" not in df.columns
    ):
        logger.error("No phasenet header found")
        return None

    for i in tqdm(range(len(df))):
        net, sta = df.iloc[i]["station_id"].split(".")[:2]
        phase_type = df.iloc[i]["phase_type"]
        phase_time = UTCDateTime(df.iloc[i]["phase_time"])
        proba = df.iloc[i]["phase_score"]
        if phase_type == "P" and proba < P_proba_threshold:
            continue
        if phase_type == "S" and proba < S_proba_threshold:
            continue

        myphase = Phase(
            net=net,
            sta=sta,
            # fdsnws_time_search=phase_time,  # query take
            fdsnws_time_search=None,
            fdsnws_station_url=fdsnws_station_url,
        )
        myphase.set_phase_info(
            phase_type,
            phase_time,
            proba,
        )
        phases.append(myphase)
        if logger.level == logging.DEBUG:
            myphase.show_all()
    logger.info(Phase.get_station_info.cache_info())
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
        fdsnws_station_url="http://10.0.1.36:8080",
        # fdsnws_station_url="http://ws.resif.fr",
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
