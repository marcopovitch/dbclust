#!/usr/bin/env python
import sys
import logging
import functools
import numpy as np
import pandas as pd
from obspy import UTCDateTime

# default logger
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger("phases")
logger.setLevel(logging.DEBUG)


class Phase(object):
    def __init__(
        self,
        net=None,
        sta=None,
        time_search=None,
        fdsnws_station_url="http://10.0.1.36:8080",
    ):
        self.network = net
        self.station = sta
        self.coord = {"latitude": None, "longitude": None, "elevation": None}
        time_search = str(time_search)  # madatorry to use lru_cache

        if self.network and self.station:
            (
                self.coord["latitude"],
                self.coord["longitude"],
                self.coord["elevation"],
            ) = self.get_station_info(
                self.network,
                self.station,
                time_search,
                fdsnws_station_url,
            )
        self.phase = None
        self.time = None
        self.proba = None

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
            + "starttime=%sT00:00:00&" % time_search
            + "format=text"
        )
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
        if self.coord['latitude'] and self.coord['longitude'] and self.coord['elevation']:
            print(
                f"    lat={self.coord['latitude']:.3f}, lon={self.coord['longitude']:.3f}, elev={self.coord['elevation']:.1f}"
            )
            print(f"    phase={self.phase}, time={self.time} proba={self.proba:.3f}")
        else:
            print("No coordinates found !")

    def oneline_show(self):
        print(
            f"{self.network}.{self.station} {self.phase} {self.time} {self.proba:.3f}"
        )


def import_phases(
    fname=None, proba_threshold=0, fdsnws_station_url="http://10.0.1.36:8080"
):
    """
    Read phaseNet csv picks file.
    Returns a list of Phase objects.
    """
    if not fname:
        logger.error("No file name defined !")
        return None
    try:
        df = pd.read_csv(fname)
    except Exception as e:
        logger.error(e)
        return None
    phases = []
    for i in range(len(df)):
        net, sta = df.iloc[i]["station_id"].split(".")[:2]
        phase_type = df.iloc[i]["phase_type"]
        phase_time = UTCDateTime(df.iloc[i]["phase_time"])
        proba = df.iloc[i]["phase_score"]
        if proba < proba_threshold:
            continue

        myphase = Phase(
            net=net,
            sta=sta,
            time_search=phase_time,
            fdsnws_station_url=fdsnws_station_url,
        )
        myphase.set_phase_info(
            phase_type,
            phase_time,
            proba,
        )
        phases.append(myphase)
        myphase.show_all()
    return phases


def test():
    phases = import_phases(
        fname="../test/picks.csv",
        proba_threshold=0.8,
        fdsnws_station_url="http://10.0.1.36:8080",
        #fdsnws_station_url="http://ws.resif.fr",
    )


if __name__ == "__main__":
    logger.setLevel(logging.DEBUG)
    test()
