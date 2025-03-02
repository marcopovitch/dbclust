#!/usr/bin/env python
import json
import sqlite3
from typing import Optional

from export_csv import generate_csv_response
from export_quakeml import generate_quake_response
from fastapi import FastAPI
from fastapi import HTTPException
from fastapi import Query
from fastapi import Request
from fastapi.responses import PlainTextResponse
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from tabulate import tabulate

# Path to SQLite database
DB_PATH = (
    "/Users/marc/Data/DBClust/france.2016.01/quakeml_2010-2018_run2/MTE_2010-2018.db"
)

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


def get_db_connection():
    # Connect to the SQLite database in read-only mode
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True, check_same_thread=False)
    conn.enable_load_extension(True)
    conn.load_extension("mod_spatialite")
    conn.row_factory = sqlite3.Row  # Allows getting a dictionary instead of a tuple
    return conn


@app.get("/")
def redirect_to_builder():
    return RedirectResponse(url="/fdsnws/event/1/builder")


@app.get("/fdsnws/event/1/builder")
def event_builder(request: Request):
    return templates.TemplateResponse("builder.html", {"request": request})


@app.get("/fdsnws/event/1/query")
async def query_events(
    request: Request,
    starttime: Optional[str] = Query(None),
    endtime: Optional[str] = Query(None),
    minlatitude: Optional[float] = Query(None),
    maxlatitude: Optional[float] = Query(None),
    minlongitude: Optional[float] = Query(None),
    maxlongitude: Optional[float] = Query(None),
    latitude: Optional[float] = Query(None),
    longitude: Optional[float] = Query(None),
    minradius: Optional[float] = Query(0),
    maxradius: Optional[float] = Query(None),
    mindepth: Optional[float] = Query(None),
    maxdepth: Optional[float] = Query(None),
    eventtype: Optional[str] = Query(None),
    includeallorigins: Optional[bool] = Query(False),
    includeallmagnitudes: Optional[bool] = Query(False),
    includearrivals: Optional[bool] = Query(False),
    includepicks: Optional[bool] = Query(False),
    eventid: Optional[str] = Query(None),
    format: Optional[str] = Query("json"),
    nodata: int = Query(404),
):
    try:
        with get_db_connection() as conn:
            cur = conn.cursor()

            # Retrieve columns from the table
            cur.execute("PRAGMA table_info(event_coordinates);")
            columns = cur.fetchall()
            column_names = [column["name"] for column in columns]

            # Select columns based on output format
            columns_to_select = (
                ["event_id"]
                if format == "quakeml"
                else [col for col in column_names if col != "geometry"]
            )

            # Start of the SQL query
            query = f"SELECT {', '.join(columns_to_select)} FROM event_coordinates WHERE 1=1"
            params = {}

            # Temporal filters
            if starttime:
                query += " AND time >= :starttime"
                params["starttime"] = starttime
            if endtime:
                query += " AND time <= :endtime"
                params["endtime"] = endtime

            # Bounding box filters
            if minlatitude is not None:
                query += " AND latitude >= :minlatitude"
                params["minlatitude"] = minlatitude
            if maxlatitude is not None:
                query += " AND latitude <= :maxlatitude"
                params["maxlatitude"] = maxlatitude
            if minlongitude is not None:
                query += " AND longitude >= :minlongitude"
                params["minlongitude"] = minlongitude
            if maxlongitude is not None:
                query += " AND longitude <= :maxlongitude"
                params["maxlongitude"] = maxlongitude

            # Circular search
            if latitude is not None and longitude is not None and maxradius is not None:
                query += """
                    AND ST_Intersects(
                        geometry,
                        ST_Buffer(
                            MakePoint(:longitude, :latitude, 4326),
                            :maxradius
                        )
                    )
                    AND ST_Distance(
                        geometry,
                        MakePoint(:longitude, :latitude, 4326)
                    ) >= :minradius
                """
                params.update(
                    {
                        "latitude": latitude,
                        "longitude": longitude,
                        "maxradius": maxradius,
                        "minradius": minradius or 0,
                    }
                )

            # Depth filters
            if mindepth is not None:
                query += " AND depth_km >= :mindepth"
                params["mindepth"] = mindepth
            if maxdepth is not None:
                query += " AND depth_km <= :maxdepth"
                params["maxdepth"] = maxdepth

            # Event type filter
            if eventtype:
                query += " AND event_type = :eventtype"
                params["eventtype"] = eventtype

            # Event ID filter
            if eventid:
                query += " AND event_id = :eventid"
                params["eventid"] = eventid

            # Execute the query
            try:
                cur.execute(query, params)
                events = cur.fetchall()
            except sqlite3.Error as e:
                raise HTTPException(status_code=500, detail=f"SQL Error: {e}")

            if not events:
                return PlainTextResponse("No events found", status_code=nodata)

            # Transform results into dictionaries
            results = [dict(event) for event in events]

    except sqlite3.Error as e:
        raise HTTPException(status_code=500, detail=f"Database connection error: {e}")

    # Handle output formats
    if format == "text":
        return generate_csv_response(results, delimiter=",")
    elif format == "csv":
        return generate_csv_response(results, delimiter=",")
    elif format == "html":
        return templates.TemplateResponse(
            "table.html", {"request": request, "events": results}
        )
    elif format == "json":
        return results
    elif format == "geojson":
        return {"type": "FeatureCollection", "features": results}
    elif format == "quakeml":
        # new db connection
        with sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True, check_same_thread=False) as conn:
            return await generate_quake_response(results, conn)
    else:
        raise HTTPException(
            status_code=501, detail=f"Format '{format}' not implemented"
        )
