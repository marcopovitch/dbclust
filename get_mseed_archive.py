#!/usr/bin/env python
import sys
import os
import logging
import argparse
from eventfetcher import EventFetcher

# default logger
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger("EventFetcher")
logger.setLevel(logging.INFO)


def get_mseed_archive(event_id, time_length=90):
    # webservice URL
    ws_base_url = "http://10.0.1.36"
    ws_event_url = "https://api.franceseisme.fr/fdsnws/event/1/"
    ws_station_url = "http://10.0.1.36:8080/fdsnws/station/1/"
    ws_dataselect_url = "http://10.0.1.36:8080/fdsnws/dataselect/1/"
    # ws_station_url = "http://ws.resif.fr/fdsnws/station/1/"
    # ws_dataselect_url = "http://ws.resif.fr/fdsnws/dataselect/1/"

    # starttime = UTCDateTime("2022-05-17T23:55:00")
    # endtime = UTCDateTime("2022-05-19T05:00:00")

    os.makedirs(event_id)

    # define black listed data
    bl_wfid = []
    for location in ["10"]:
        for channel in ["HHZ", "HHN", "HHE"]:
            bl_wfid.append(["FR", "STR", location, channel])

    # RUSF
    for location in ["01", "02", "03", "04", "05", "06"]:
        for channel in ["HHZ", "HHN", "HHE"]:
            bl_wfid.append(["FR", "RUSF", location, channel])

    # NIMR
    for location in ["10", "30"]:
        for channel in ["HHZ", "HHN", "HHE", "HH1", "HH2", "HH3"]:
            bl_wfid.append(["FR", "NIMR", location, channel])

    # MT.GUI
    for location in ["01", "02", "03"]:
        for channel in ["EHZ"]:
            bl_wfid.append(["MT", "GUI", location, channel])

    # XX.GP : grand pilier
    for location in ["00", "01", "02", "03"]:
        for channel in ["HHZ", "HHN", "HHE", "HH1", "HH2", "HH3"]:
            bl_wfid.append(["XX", "GPIL", location, channel])

    # get data
    mydata = EventFetcher(
        event_id,
        use_only_trace_with_weighted_arrival=False,
        station_max_dist_km=130,
        time_length=time_length,
        black_listed_waveforms_id=bl_wfid,
        base_url=ws_base_url,
        ws_event_url=ws_event_url,
        ws_station_url=ws_station_url,
        ws_dataselect_url=ws_dataselect_url,
        backup_dirname="Data/cache",
        enable_read_cache=True,
        enable_write_cache=True,
        fdsn_debug=False,
    )

    if not mydata.st:
        logger.info("No data associated to event %s", event_id)
    else:
        # for tr in mydata.st:
        #    print(tr, tr.stats['mseed']['encoding'])

        # subsample channel to 100 Hz
        for tr in mydata.st:
            if tr.stats.sampling_rate > 100.0:
                logger.debug(
                    f"Resampling {tr.id} from {tr.stats.sampling_rate} to 100 hz"
                )
                try:
                    tr.resample(sampling_rate=100.0)
                except Exception as e:
                    logger.error(f"Interpolation failed for {tr.id} ({e})")
            else:
                logger.debug(
                    f"{tr.id}, sample rate ({tr.stats.sampling_rate}) <= 100 hz: nothing to do !"
                )

        # logger.info(mydata.st.__str__(extended=True))

        # phaseNet complient mseed file
        net_sta = set()
        for tr in mydata.st:
            net, sta, loc, chan = tr.id.split(".")
            net_sta.add(f"{net}.{sta}")

        for i in net_sta:
            net, sta = i.split(".")
            filename = os.path.join(f"{event_id}", f"{i}.mseed")
            logger.info(f"Writting {filename}")
            mydata.st.select(network=net, station=sta).write(filename, format="MSEED")


if __name__ == "__main__":
    logger.setLevel(logging.DEBUG)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-e",
        "--eventid",
        default=None,
        dest="event_id",
        help="fdsnws-event event id.",
        type=str,
    )
    parser.add_argument(
        "-l",
        "--timelength",
        default=90,
        dest="time_lenght",
        help="trace time length",
        type=float,
    )   
    args = parser.parse_args()
    if not args.event_id:
        parser.print_help()
        sys.exit(255)

    get_mseed_archive(event_id=args.event_id, time_length=args.time_lenght)
