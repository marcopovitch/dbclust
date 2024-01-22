#!/usr/bin/env python
import sys
import os
import logging
import argparse
import pandas as pd
import dask
from dask.distributed import Client, LocalCluster
import dask.dataframe as dd
from dask import delayed
from icecream import ic


def ymljoin(loader, node):
    seq = loader.construct_sequence(node)
    return "".join([str(i) for i in seq])


def yml_read_config(filename):
    with open(filename, "r") as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
    return cfg


def dbclust_partition(df, start, end):
    return start, end, len(df)


if __name__ == "__main__":
    # default logger
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    logger = logging.getLogger("dbclust")
    logger.setLevel(logging.INFO)

    # parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     "-c",
    #     "--conf",
    #     default=None,
    #     dest="configfile",
    #     help="yaml configuration file.",
    #     type=str,
    # )
    # parser.add_argument(
    #     "-n",
    #     "--ncpu",
    #     default=4,
    #     dest="ncpu",
    #     help="number of cpu to use",
    #     type=int,
    # )
    # args = parser.parse_args()
    # if not args.configfile:
    #     parser.print_help()
    #     sys.exit(255)

    # # check configfile and csv_file

    # yaml.add_constructor("!join", ymljoin)
    # cfg = yml_read_config(args.configfile)
    
    #csv_file = "/Users/marc/Data/DBClust/selestat/picks/renass.picks.csv"
    #time_column = "time"
    
    csv_file = "/Users/marc/Data/DBClust/selestat/picks/test.csv"
    time_column = "phase_time"

    cluster = LocalCluster()
    client = Client(cluster)
    
    ic(client)

    ddf = dd.read_csv(
        csv_file,
        dtype={
            "station_id": "string",
            "phase_type": "string",
            "phase_time": "string",
            "phase_score": "float64",
            "eventid": "string",
            "agency": "string",
        },
    )

    ddf[time_column] = dd.to_datetime(ddf[time_column], utc=True)
    ddf[time_column] = ddf[time_column].dt.round("0.0001S")

    partition_duration = pd.Timedelta(hours=24)
    overlap_duration = pd.Timedelta(seconds=30)

    min_time = ddf[time_column].min().compute()
    max_time = ddf[time_column].max().compute()
    
    ic(min_time, max_time)

    time_divisions = pd.date_range(
        start=min_time, end=max_time, freq=partition_duration
    )

    adjusted_time_divisions = [
        (start - overlap_duration, end + overlap_duration)
        for start, end in zip(time_divisions, time_divisions[1:])
    ]

    # Need to get 
    dd_adjusted_time_divisions = dd.from_pandas(
        pd.DataFrame(adjusted_time_divisions, columns=["start", "end"]), npartitions=1
    )

    # Create tasks for each data segment between start and end indices
    delayed_tasks = [
        dask.delayed(dbclust_partition)(
            #ddf[(ddf[time_column] >= start) & (ddf[time_column] < end)].compute(),
            ddf[(ddf[time_column] >= start) & (ddf[time_column] < end)],
            start,
            end,
        )
        for start, end in dd_adjusted_time_divisions.itertuples(index=False)
    ]
    
    # execute tasks
    results = dask.compute(*delayed_tasks)

    # check that everything has gone smoothly
    ic(results)

    client.close()
