#!/usr/bin/env python
from typing import List
from typing import Optional

import duckdb


def duckdb_init(filenames: List[str], type: str):
    if type == "parquet":
        conn = duckdb_init_parquet(filenames)
        return conn
    else:
        conn = duckdb_init_csv(filenames)
    return conn


def duckdb_init_parquet(parquet_filenames: List[str], threads: int = 4):
    config = {"threads": threads}

    files_str = ""
    for f in parquet_filenames:
        files_str += f"'{f}', "
    files_str = "[" + files_str[:-2] + "]"

    rqt = f"""
        CREATE VIEW PICKS AS SELECT *
        FROM read_parquet({files_str});
    """

    duckdb_con = duckdb.connect(
        database=":memory:",
        config=config,
    )
    try:
        duckdb_con.execute(rqt)
    except Exception as e:
        raise e

    return duckdb_con


def duckdb_init_csv(csv_filenames: str, threads: int = 1):
    config = {"threads": threads}

    files_str = ""
    for f in csv_filenames:
        files_str += f"'{f}', "
    files_str = "[" + files_str[:-2] + "]"

    rqt = f"""
        CREATE VIEW PICKS AS SELECT * FROM '{files_str}';")
    """

    duckdb_con = duckdb.connect(
        database=":memory:",
        config=config,
    )
    try:
        duckdb_con.execute(rqt)
    except Exception as e:
        raise e

    return duckdb_con
