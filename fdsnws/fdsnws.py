#!/usr/bin/env python
import json
import sqlite3
from typing import Optional

from fastapi import FastAPI
from fastapi import HTTPException
from fastapi import Query
from fastapi.requests import Request
from fastapi.responses import PlainTextResponse
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fdsnws_csv import generate_csv_response
from tabulate import tabulate


DB_PATH = "/Users/marc/Data/DBClust/france.2016.01/quakeml_2010-2018_run2/MTE_2010-2018.db"  # Chemin vers ta base de données SQLite

app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.enable_load_extension(True)
    conn.load_extension("mod_spatialite")
    conn.row_factory = sqlite3.Row  # Permet d'obtenir un dictionnaire au lieu d'un tuple
    return conn

@app.get("/")
def redirect_to_builder():
    return RedirectResponse(url="/fdsnws/event/1/builder")

@app.get("/fdsnws/event/1/builder")
def event_builder(request: Request):
    return templates.TemplateResponse("builder.html", {"request": request})

@app.get("/fdsnws/event/1/query")
def query_events(
    starttime: Optional[str] = Query(None),
    endtime: Optional[str] = Query(None),
    minlatitude: Optional[str] = Query(None),  # Changed to Optional[str]
    maxlatitude: Optional[str] = Query(None),  # Changed to Optional[str]
    minlongitude: Optional[str] = Query(None),  # Changed to Optional[str]
    maxlongitude: Optional[str] = Query(None),  # Changed to Optional[str]
    latitude: Optional[float] = Query(None),
    longitude: Optional[float] = Query(None),
    minradius: Optional[float] = Query(0),  # Défaut à 0 si non défini
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
    request: Request = None,  # Ajout de la requête pour Jinja2
):
    try:
        conn = get_db_connection()
    except sqlite3.Error as e:
        raise HTTPException(status_code=500, detail=f"Database error: {e}")
    cur = conn.cursor()

    # Récupérer les informations sur les colonnes de la table
    cur.execute("PRAGMA table_info(event_coordinates);")
    columns = cur.fetchall()
    column_names = [column[1] for column in columns]
    # Exclure 'geometry' des colonnes sélectionnées
    columns_to_select = [col for col in column_names if col != "geometry"]

    # Convert empty string values to None for optional numeric params
    minlatitude = float(minlatitude) if minlatitude not in (None, "") else None
    maxlatitude = float(maxlatitude) if maxlatitude not in (None, "") else None
    minlongitude = float(minlongitude) if minlongitude not in (None, "") else None
    maxlongitude = float(maxlongitude) if maxlongitude not in (None, "") else None

    # Prepare the basic SQL query
    query = "SELECT " + ", ".join(columns_to_select) + " FROM event_coordinates WHERE 1=1"
    params = {}

    # Time
    if starttime:
        query += " AND time >= :starttime"
        params["starttime"] = starttime
    if endtime:
        query += " AND time <= :endtime"
        params["endtime"] = endtime

    # Bounding Box
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

    # Circular Search
    if latitude is not None and longitude is not None:
        query = f"""
            SELECT {', '.join(columns_to_select)}
            FROM event_coordinates
            WHERE ST_Intersects(
                geometry,
                ST_Buffer(
                    MakePoint(:longitude, :latitude, 4326),
                    :maxradius
                )
            )
            AND ST_Distance(
                geometry,
                MakePoint(:longitude, :latitude, 4326)
            ) BETWEEN :minradius AND :maxradius
        """
        params["latitude"] = latitude
        params["longitude"] = longitude
        if minradius is not None:
            params["minradius"] = minradius
        if maxradius is not None:
            params["maxradius"] = maxradius
        elif minradius == 0:  # Special handling for minradius == 0
            query = query.replace("BETWEEN :minradius AND :maxradius", ">= 0")
            params["minradius"] = 0

    # Depth filters
    if mindepth is not None:
        query += " AND depth_km >= :mindepth"
        params["mindepth"] = mindepth
    if maxdepth is not None:
        query += " AND depth_km <= :maxdepth"
        params["maxdepth"] = maxdepth

    if eventtype:
        query += " AND event_type = :eventtype"
        params["eventtype"] = eventtype

    if eventid:
        query += " AND event_id = :eventid"
        params["eventid"] = eventid

    # Optional Filters
    if includeallorigins or includeallmagnitudes or includearrivals or includepicks:
        # These filters are all defaulted to True, so we don’t need to add any specific conditions.
        pass

    # Execute query
    try:
        cur.execute(query, params)
    except sqlite3.Error as e:
        raise HTTPException(status_code=500, detail=f"SQL execution error: {e}")

    events = cur.fetchall()

    if not events:
        raise HTTPException(status_code=nodata, detail="No data found")

    results = [dict(event) for event in events]

    # if format == "text":
    #     text_output = "\n".join([json.dumps(event) for event in results])
    #     return PlainTextResponse(text_output)

    if format == "text":
        return generate_csv_response(results, delimiter=",")
    elif format == "csv":
        return generate_csv_response(results, delimiter=",")
    elif format == "html":
        return templates.TemplateResponse("table.html", {"request": request, "events": results})
    elif format == "json":
        return results
    elif format == "geojson":
        return {"type": "FeatureCollection", "features": results}
    elif format == "quakeml":
        raise HTTPException(status_code=501, detail="QuakeML output not implemented")

    return results
