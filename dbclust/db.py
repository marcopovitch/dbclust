#!/usr/bin/env python
import duckdb


def duckdb_init(parquet_filename: str):
    duckdb_con = duckdb.connect(database=":memory:")
    try:
        duckdb_con.execute(
            f"CREATE VIEW PICKS AS SELECT * "
            f"FROM parquet_scan('{parquet_filename}');"
        )
    except Exception as e:
        raise e

    return duckdb_con
