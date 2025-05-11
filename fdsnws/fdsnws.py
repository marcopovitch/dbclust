#!/usr/bin/env python3
import argparse
import logging
import re
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import uvicorn
from export_csv import generate_csv_response
from export_geojson import generate_geojson_response
from export_quakeml import generate_quake_response
from export_text import generate_text_response
from fastapi import FastAPI
from fastapi import HTTPException
from fastapi import Query
from fastapi import Request
from fastapi.responses import FileResponse
from fastapi.responses import JSONResponse
from fastapi.responses import PlainTextResponse
from fastapi.responses import RedirectResponse
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.base import BaseHTTPMiddleware


class NormalizeSlashesMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Normalize the path in-place (no redirect)
        url_path = request.scope.get('path', '')
        normalized_path = re.sub(r'/+', '/', url_path)
        if normalized_path != url_path:
            request.scope['path'] = normalized_path
        return await call_next(request)

def iso_to_sqlite(dtstr):
    # Gère les formats ISO 8601 avec ou sans 'Z'
    dt = datetime.fromisoformat(dtstr.replace('Z', '+00:00'))
    # Retourne au format SQLite classique
    return dt.strftime('%Y-%m-%d %H:%M:%S.%f')

def create_app(db_path: str, debug=False):
    app = FastAPI(
        debug=debug,
        title="FDSN Web Service (Spatialite)",
        description="FDSN-compliant web service for querying seismic event data (Spatialite)",
        version="1.2.0",
    )

    app.add_middleware(NormalizeSlashesMiddleware)

    templates = Jinja2Templates(directory="templates")
    app.mount("/static", StaticFiles(directory="static"), name="static")

    # Dependency: get a read-only connection and load spatialite
    def get_db_connection():
        conn = sqlite3.connect(f'file:{db_path}?mode=ro', uri=True, check_same_thread=False)
        try:
            conn.enable_load_extension(True)
            conn.execute("SELECT load_extension('mod_spatialite');")
        except Exception as e:
            logging.error(f"Could not load spatialite extension: {e}")
            raise HTTPException(status_code=500, detail="Spatialite extension not loaded")
        return conn

    @app.get("/")
    def root_redirect(request: Request):
        return RedirectResponse(url="/static/builder.html")

    @app.get("/fdsnws/event/1/query")
    def query_events(
        request: Request,
        starttime: Optional[str] = Query(None),
        endtime: Optional[str] = Query(None),
        minlatitude: Optional[float] = Query(None),
        maxlatitude: Optional[float] = Query(None),
        minlongitude: Optional[float] = Query(None),
        maxlongitude: Optional[float] = Query(None),
        latitude: Optional[float] = Query(None),
        longitude: Optional[float] = Query(None),
        minradius: Optional[float] = Query(None),
        maxradius: Optional[float] = Query(None),
        mindepth: Optional[float] = Query(None),
        maxdepth: Optional[float] = Query(None),
        minmagnitude: Optional[float] = Query(None),
        maxmagnitude: Optional[float] = Query(None),
        magnitudetype: Optional[str] = Query(None),
        eventtype: Optional[str] = Query(None),
        eventid: Optional[str] = Query(None),
        format: str = Query("xml", pattern="^(xml|json|text|csv|quakeml|geojson|html)$"),
        limit: Optional[int] = Query(None),
        offset: Optional[int] = Query(None),
        orderby: Optional[str] = Query(None, pattern="^(time|magnitude)(-asc|-desc)?$"),
        catalog: Optional[str] = Query(None),
        contributor: Optional[str] = Query(None),
        updatedafter: Optional[str] = Query(None),
        includeallorigins: Optional[bool] = Query(False),
        includeallmagnitudes: Optional[bool] = Query(False),
        includearrivals: Optional[bool] = Query(False),
        includepicks: Optional[bool] = Query(False),
    ):
        conn = get_db_connection()
        try:
            where = ["1=1"]
            params = []
            # Time constraints
            if starttime:
                where.append("time >= ?")
                params.append(iso_to_sqlite(starttime))
            if endtime:
                where.append("time <= ?")
                params.append(iso_to_sqlite(endtime))
            # Spatial constraints
            if minlatitude is not None:
                where.append("latitude >= ?")
                params.append(minlatitude)
            if maxlatitude is not None:
                where.append("latitude <= ?")
                params.append(maxlatitude)
            if minlongitude is not None:
                where.append("longitude >= ?")
                params.append(minlongitude)
            if maxlongitude is not None:
                where.append("longitude <= ?")
                params.append(maxlongitude)
            if mindepth is not None:
                where.append("depth_km >= ?")
                params.append(mindepth)
            if maxdepth is not None:
                where.append("depth_km <= ?")
                params.append(maxdepth)
            if minmagnitude is not None:
                where.append("magnitude >= ?")
                params.append(minmagnitude)
            if maxmagnitude is not None:
                where.append("magnitude <= ?")
                params.append(maxmagnitude)
            if magnitudetype:
                where.append("magnitude_type = ?")
                params.append(magnitudetype)
            if eventtype:
                where.append("event_type = ?")
                params.append(eventtype)
            if eventid:
                where.append("event_id = ?")
                params.append(eventid)

            # Spatial circular search (if lat/lon/maxradius are provided)
            if latitude is not None and longitude is not None and maxradius is not None:
                where.append(
                    "ST_Distance(geometry, ST_GeomFromText(?, 4326)) <= ?"
                )
                point_wkt = f"POINT({longitude} {latitude})"
                params.extend([point_wkt, maxradius])
                if minradius and minradius > 0:
                    where.append(
                        "ST_Distance(geometry, ST_GeomFromText(?, 4326)) >= ?"
                    )
                    params.extend([point_wkt, minradius])

            sql = f"SELECT * FROM event_coordinates WHERE {' AND '.join(where)}"
            if orderby == 'time-asc':
                sql += " ORDER BY time ASC"
            elif orderby == 'time':
                sql += " ORDER BY time DESC"
            else:
                sql += " ORDER BY time DESC"

            if limit:
                sql += " LIMIT ?"
                params.append(limit)
            if offset:
                sql += " OFFSET ?"
                params.append(offset)

            logging.debug(f"SQL: {sql} | Params: {params}")
            print(sql)
            print(params)
            cursor = conn.execute(sql, params)
            columns = [desc[0] for desc in cursor.description]
            results = [dict(zip(columns, row)) for row in cursor.fetchall()]
            # get rid of geometry column
            for row in results:
                del row["geometry"]

            # Format-specific responses
            if format.lower() == "json":
                timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
                filename = f"events-{timestamp}.json"
                from fastapi.responses import JSONResponse
                headers = {"Content-Disposition": f"attachment; filename={filename}"}
                return JSONResponse({"events": results}, headers=headers)
            elif format.lower() == "csv":
                return generate_csv_response(results)
            elif format.lower() == "text":
                return generate_text_response(results)
            elif format.lower() == "geojson":
                return generate_geojson_response(results)
            elif format.lower() == "quakeml" or format.lower() == "xml":
                # For QuakeML, pass event_ids and db_path to the generator
                event_ids = [r["event_id"] for r in results if "event_id" in r]
                return generate_quake_response(event_ids, db_path)
            elif format.lower() == "html":
                return templates.TemplateResponse("table.html", {"request": request, "events": results})
            else:
                raise HTTPException(status_code=400, detail=f"Unsupported format: {format}")
        finally:
            conn.close()

    @app.get("/fdsnws/event/1/version")
    def get_event_version():
        return {"version": "1.2.0"}

    @app.get("/fdsnws/event/1/catalogs")
    def get_event_catalogs():
        return {"catalogs": ["LOCAL"]}

    @app.get("/fdsnws/event/1/contributors")
    def get_event_contributors():
        return {"contributors": ["LOCAL"]}

    return app

if __name__ == "__main__":
    # default values
    DB_PATH = ""
    PORT = 8000
    HOST = "localhost"
    DEBUG = False

    # parse command line arguments
    parser = argparse.ArgumentParser(
        description="Run the DBClust API with a specified database."
    )
    parser.add_argument(
        "--db",
        required=False,
        default=DB_PATH,
        help="Path to the SQLite database file.",
    )
    parser.add_argument(
        "--port", type=int, default=PORT, help="Port to run the API on."
    )
    parser.add_argument("--host", default=HOST, help="Host to run the API on.")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode.")
    args = parser.parse_args()

    # Validate database path
    db_path = Path(args.db)
    if not db_path.exists():
        logging.error(f"Database path does not exist: {args.db}")
        sys.exit(1)

    # Validate SQLite database connection
    try:
        with sqlite3.connect(args.db) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM event_coordinates LIMIT 1")
            if not cursor.fetchone():
                logging.warning("Table 'event_coordinates' does not exist in the database.")
            else:
                logging.info("SQLite database connected successfully and contains 'event_coordinates' table.")
    except sqlite3.Error as e:
        logging.error(f"Failed to connect to SQLite database: {e}")
        sys.exit(1)

    # Set logging level
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)

    logging.info(f"Using database: {args.db}")
    logging.info(f"Running API on {args.host}:{args.port}...")

    app = create_app(db_path=args.db, debug=args.debug)

    # Start the server with proper log level for uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="debug" if args.debug else "info")
