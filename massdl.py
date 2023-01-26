#!/usr/bin/env python
import obspy
from obspy.clients.fdsn import Client
from obspy.clients.fdsn.mass_downloader import (
    CircularDomain,
    Restrictions,
    MassDownloader,
)

origin_time = obspy.UTCDateTime(2022, 9, 10, 0, 0, 0)

# Circular domain around the epicenter. This will download all data between
# 70 and 90 degrees distance from the epicenter. This module also offers
# rectangular and global domains. More complex domains can be defined by
# inheriting from the Domain class.
domain = CircularDomain(
    latitude=47.67, longitude=7.47, minradius=0, maxradius=130 / 111.0
)

restrictions = Restrictions(
    # Get data from 5 minutes before the event to one hour after the
    # event. This defines the temporal bounds of the waveform data.
    starttime=origin_time - 5 * 60,
    endtime=origin_time + 86400 + 5 * 60,
    # You might not want to deal with gaps in the data. If this setting is
    # True, any trace with a gap/overlap will be discarded.
    reject_channels_with_gaps=False,
    # And you might only want waveforms that have data for at least 95 % of
    # the requested time span. Any trace that is shorter than 95 % of the
    # desired total duration will be discarded.
    minimum_length=0.0,
    # No two stations should be closer than 10 km to each other. This is
    # useful to for example filter out stations that are part of different
    # networks but at the same physical station. Settings this option to
    # zero or None will disable that filtering.
    minimum_interstation_distance_in_m=10e3,
    # Only HH or BH channels. If a station has HH channels, those will be
    # downloaded, otherwise the BH. Nothing will be downloaded if it has
    # neither. You can add more/less patterns if you like.
    # channel_priorities=["HH[ZNE]", "BH[ZNE]"],
    channel_priorities=["HH[ZNE123]"],
    # Location codes are arbitrary and there is no rule as to which
    # location is best. Same logic as for the previous setting.
    location_priorities=["", "00", "10"],
)

# No specified providers will result in all known ones being queried.

# BCSF-Renass webservice URL
ws_base_url = "http://10.0.1.36"
ws_station_url = "http://10.0.1.36:8080/fdsnws/station/1"
ws_dataselect_url = "http://10.0.1.36:8080/fdsnws/dataselect/1"

fdsn_debug= True

client = Client(
    debug=fdsn_debug,
    base_url=ws_base_url,
    service_mappings={
        "dataselect": ws_dataselect_url,
        "station": ws_station_url,
    },
)

mdl = MassDownloader(providers=[client])
# The data will be downloaded to the ``./waveforms/`` and ``./stations/``
# folders with automatically chosen file names.
mdl.download(
    domain, restrictions, threads_per_client=20, mseed_storage="waveforms", stationxml_storage="stations"
)
