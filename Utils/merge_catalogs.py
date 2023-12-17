import pandas as pd
from obspy import read_events, Catalog


df = pd.read_csv("events_merge_info.csv")
df["agencies_list"].fillna("", inplace=True)

print("france.2016.01.qml")
mycat = read_events("france.2016.01.qml")

files = [
    "../bulletins/LDG-2016.01.xml",
    "../bulletins/OCA.2016.01.xml",
    "../bulletins/OMP-2016.xml",
    "../bulletins/isterre.2016.01.qml",
    "../bulletins/renass201601.qml",
]
cat = Catalog()
for f in files:
    print(f)
    tmpcat = read_events(f)
    cat.extend(tmpcat)

for index, row in df.iterrows():
    myevent_id = row["event_id"]
    print(myevent_id)
    myevent = [e for e in mycat.events if e.resource_id.id == myevent_id].pop()
    agencies_event_list = row["agencies_list"].split(" ")
    # origins = [e.preferred_origin() for e in cat.events if e.resource_id.id in agencies_event_list]
    events = [e for e in cat.events if e.resource_id.id in agencies_event_list]
    origins = [e.preferred_origin() for e in events]
    picks = [p for e in events for p in e.picks]
    myevent.origins.extend(origins)
    myevent.picks.extend(picks)

mycat.write("cat_merge.qml", format="QUAKEML")
