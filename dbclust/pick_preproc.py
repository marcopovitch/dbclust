#!/usr/bin/env python
import pandas as pd
from datetime import timedelta
from sklearn.cluster import DBSCAN


def unload_too_close_picks_clustering(csv_file_in, csv_file_out, delta_time):
    print(f"Reading from {csv_file_out}.")
    df = pd.read_csv(csv_file_in)
    df.rename(
        columns={
            "seedid": "station_id",
            "phasename": "phase_type",
            "time": "phase_time",
            "probability": "phase_score",
        },
        inplace=True,
    )
    # Keeps only network_code.station_code
    df["station_id"] = df["station_id"].map(lambda x: ".".join(x.split(".")[:2]))
    df["phase_time"] = pd.to_datetime(df["phase_time"], utc=True)

    df = df.sort_values(by=["station_id", "phase_type", "phase_time"])

    # dbscan configuration
    max_distance = delta_time  # secondes
    min_samples = 1
    dbscan = DBSCAN(eps=max_distance, min_samples=min_samples, metric="euclidean")

    results = pd.DataFrame(
        columns=["station_id", "phase_type", "phase_time", "phase_score"]
    )
    # run separately by phase type
    for phase in ("P", "S"):
        print(f"Working on {phase}.")
        phase_df = df[df["phase_type"].str.contains(phase)]
        phase_df["phase_time"] = pd.to_datetime(phase_df["phase_time"], utc=True)
        # get min time from df
        min_timestamp = phase_df["phase_time"].min()
        # create numeric_time for time distance computation
        phase_df["numeric_time"] = (
            phase_df["phase_time"] - pd.to_datetime(min_timestamp, utc=True)
        ).dt.total_seconds()

        # loop over station_id
        for station_id in phase_df["station_id"].drop_duplicates():
            print(f"Working on {phase}/{station_id}")
            tmp_df = phase_df.loc[phase_df["station_id"] == station_id]
            before = len(tmp_df)
            # clusterize by station_id
            tmp_df["cluster"] = dbscan.fit_predict(tmp_df[["numeric_time"]])
            # keeps only the pick with the higher score
            idx = tmp_df.groupby(["cluster"])["phase_score"].idxmax()
            tmp_df = tmp_df.loc[idx]
            tmp_df = tmp_df.drop(columns=["numeric_time", "cluster"])
            after = len(tmp_df)
            print(f"length before: {before}, after: {after}")
            results = pd.concat([results, tmp_df], ignore_index=True)
            
    results["phase_time"] = pd.to_datetime(results["phase_time"]).dt.strftime(
        "%Y-%m-%dT%H:%M:%S.%fZ"
    )
    print(results)
    print(f"Writing to {csv_file_out}.")
    results.to_csv(csv_file_out, index=False)


if __name__ == "__main__":
    # input file is :
    # 
    csv_file_in = "/Users/marc/Data/DBClust/france.2016.01/picks/france-2016.01.picks"
    csv_file_out = (
        "/Users/marc/Data/DBClust/france.2016.01/picks/france-2016.01-filtered.picks"
    )
    delta_time = 0.1
    unload_too_close_picks_clustering(csv_file_in, csv_file_out, delta_time)
