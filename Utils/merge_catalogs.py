#!/usr/bin/env python
import pandas as pd
from obspy import read_events, Catalog


print("Reading merge catalogs information: events_merge_info.csv")
df = pd.read_csv("events_merge_info.csv")
df["agencies_list"].fillna("", inplace=True)

print("Reading merged catalog: france.2016.01.qml")
mycat = read_events("france.2016.01.qml")

files = [
    "../bulletins/LDG-2016.01.xml",
    "../bulletins/OCA.2016.01.xml",
    "../bulletins/OMP-2016.01.xml",
    "../bulletins/isterre.2016.01.qml",
    "../bulletins/renass201601.qml",
]
cat = Catalog()
for f in files:
    print(f"Reading contributing catalogs {f}")
    tmpcat = read_events(f)
    cat.extend(tmpcat)

print("starting merge ...")
for index, row in df.iterrows():
    myevent_id = row["event_id"]
    print(myevent_id)
    myevent = [e for e in mycat.events if e.resource_id.id == myevent_id].pop()
    agencies_event_list = row["agencies_list"].split(" ")

    events = [e for e in cat.events if e.resource_id.id in agencies_event_list]
    origins = [e.preferred_origin() for e in events]
    magnitudes = [e.preferred_magnitude() for e in events if e.preferred_magnitude()]
    picks = [p for e in events for p in e.picks]

    myevent.origins.extend(origins)
    myevent.magnitudes.extend(magnitudes)
    myevent.picks.extend(picks)

print("Writing QUAKEML")
mycat.write("cat_merge.2016.01.qml", format="QUAKEML")
print("Writing SC3ML")
mycat.write("cat_merge.2016.01.sc3ml", format="SC3ML")