#!/usr/bin/env python3
import argparse
import logging
import re
import sqlite3
import sys
from pathlib import Path
from typing import Optional

import uvicorn
from export_csv import generate_csv_response
from export_quakeml import generate_quake_response
from export_text import generate_text_response
from fastapi import FastAPI
from fastapi import HTTPException
from fastapi import Query
from fastapi import Request
from fastapi.responses import FileResponse
from fastapi.responses import PlainTextResponse
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.base import BaseHTTPMiddleware
# import aiosqlite

logging.basicConfig(level=logging.INFO)


class NormalizeSlashesMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Retrieve the full URL
        url_path = request.url.path

        # Check if there are multiple slashes
        if "//" in url_path:
            # Normalize the URL by replacing multiple consecutive / with a single /
            normalized_path = re.sub(r"/+", "/", url_path)

            # Rebuild the full URL for redirection
            query_string = request.url.query
            normalized_url = normalized_path
            if query_string:
                normalized_url = f"{normalized_path}?{query_string}"

            # Redirect to the normalized URL
            return RedirectResponse(url=normalized_url, status_code=301)

        # Continue normal processing if the URL does not contain consecutive //
        return await call_next(request)


def create_app(db_path: str, debug=False) -> FastAPI:
    app = FastAPI(debug=debug)
    app.add_middleware(NormalizeSlashesMiddleware)

    templates = Jinja2Templates(directory="templates")
    app.mount("/static", StaticFiles(directory="static"), name="static")

    def static_xml(path: Path) -> FileResponse:
        return FileResponse(path, media_type="application/xml")

    @app.get("/")
    async def redirect_to_builder():
        return RedirectResponse(url="/fdsnws/event/1/builder")

    @app.get("/fdsnws/event/1/builder")
    async def event_builder(request: Request):
        return templates.TemplateResponse("builder.html", {"request": request})

    @app.get("/fdsnws/event/1/application.wadl")
    async def get_event_application_wadl():
        return static_xml(Path("config/wadl/event/application.wadl"))

    @app.get("/fdsnws/event/1/catalogs")
    async def get_event_catalogs():
        return static_xml(Path("config/wadl/event/catalogs"))

    @app.get("/fdsnws/event/1/contributors")
    async def get_event_contributors():
        return static_xml(Path("config/wadl/event/contributors"))

    @app.get("/fdsnws/event/1/version")
    async def get_event_version():
        return "1.2.0"

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
        eventtype: Optional[str] = Query(None),
        eventid: Optional[str] = Query(None),
        mindepth: Optional[float] = Query(None),
        maxdepth: Optional[float] = Query(None),
        minmagnitude: Optional[float] = Query(None),
        maxmagnitude: Optional[float] = Query(None),
        magnitudetype: Optional[str] = Query(None),
        format: Optional[str] = Query("quakeml"),
        limit: Optional[int] = Query(0),
        offset: Optional[int] = Query(0),
        orderby: Optional[str] = Query(None),
        nodata: int = Query(404),
        includeallorigins: Optional[bool] = Query(False),
        includeallmagnitudes: Optional[bool] = Query(False),
        includearrivals: Optional[bool] = Query(False),
        includepicks: Optional[bool] = Query(False),
    ):
        try:
            conn = sqlite3.connect(
                f"file:{db_path}?mode=ro", uri=True, check_same_thread=False
            )
            conn.enable_load_extension(True)
            conn.load_extension("mod_spatialite")
            conn.row_factory = sqlite3.Row

            cursor = conn.execute("PRAGMA table_info(event_coordinates);")
            column_names = [row["name"] for row in cursor]

            columns_to_select = (
                ["event_id"]
                if format == "quakeml"
                else [c for c in column_names if c != "geometry"]
            )

            query = f"SELECT {', '.join(columns_to_select)} FROM event_coordinates WHERE 1=1"
            params = {}

            if starttime:
                query += " AND time >= :starttime"
                params["starttime"] = starttime
            if endtime:
                query += " AND time <= :endtime"
                params["endtime"] = endtime
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
            if minmagnitude is not None:
                query += " AND magnitude >= :minmagnitude"
                params["minmagnitude"] = minmagnitude
            if maxmagnitude is not None:
                query += " AND magnitude <= :maxmagnitude"
                params["maxmagnitude"] = maxmagnitude
            if magnitudetype:
                query += " AND magnitude_type = :magnitudetype"
                params["magnitudetype"] = magnitudetype

            if orderby in ("time", "time-asc", "magnitude", "magnitude-asc"):
                direction = "ASC" if "asc" in orderby else "DESC"
                field = orderby.split("-")[0]
                query += f" ORDER BY {field} {direction}"

            if limit:
                query += " LIMIT :limit"
                params["limit"] = limit
                if offset:
                    query += " OFFSET :offset"
                    params["offset"] = offset

            if logging.getLogger().isEnabledFor(logging.DEBUG):
                logging.debug(f"SQL Query: {query} | Params: {params}")

            cursor = conn.execute(query, params)
            results = [dict(row) for row in cursor]

            if not results:
                conn.close()
                return PlainTextResponse("No events found", status_code=nodata)

            if format == "quakeml":
                return generate_quake_response(results, conn)
            elif format == "text":
                conn.close()
                return generate_text_response(results, delimiter="|")
            elif format == "csv":
                conn.close()
                return generate_csv_response(results, delimiter=",")
            elif format == "html":
                conn.close()
                return templates.TemplateResponse(
                    "table.html", {"request": request, "events": results}
                )
            elif format == "json":
                conn.close()
                return results
            elif format == "geojson":
                conn.close()
                return {"type": "FeatureCollection", "features": results}
            else:
                conn.close()
                raise HTTPException(
                    status_code=501, detail=f"Format '{format}' not implemented"
                )

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Internal error: {e}")

    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the DBClust API with a specified database."
    )
    parser.add_argument("--db", required=True, help="Path to the SQLite database file.")
    parser.add_argument(
        "--port", type=int, default=8000, help="Port to run the API on."
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug mode.")
    args = parser.parse_args()

    if not Path(args.db).exists():
        logging.error(f"Database path does not exist: {args.db}")
        sys.exit(1)

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    else:
        logging.getLogger().setLevel(logging.INFO)

    logging.info(f"Using database: {args.db}")
    logging.info(f"Running API on port {args.port}...")

    app = create_app(args.db)
    uvicorn.run(app, host="localhost", port=args.port)
