#!/usr/bin/env python
from typing import Optional

import duckdb


def duckdb_init(filename: str, type: str):
    if type == "parquet":
        conn = duckdb_init_parquet(filename)
        return conn
    else:
        conn = duckdb_init_csv(filename)
    return conn


def duckdb_init_parquet(parquet_filename: str, threads: int = 4):
    config = {"threads": threads}

    duckdb_con = duckdb.connect(
        database=":memory:",
        config=config,
    )
    try:
        duckdb_con.execute(
            f"CREATE VIEW PICKS AS SELECT * "
            f"FROM parquet_scan('{parquet_filename}');"
        )

    except Exception as e:
        raise e

    return duckdb_con


def duckdb_init_csv(filename: str, threads: int = 1):
    config = {"threads": threads}

    duckdb_con = duckdb.connect(
        database=":memory:",
        config=config,
    )
    try:
        duckdb_con.execute(f"CREATE VIEW PICKS AS SELECT * FROM '{filename}';")

    except Exception as e:
        raise e

    return duckdb_con
