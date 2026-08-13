#!/usr/bin/env python3
import argparse
import io
import json
import logging
import os
import re
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional
from urllib.parse import quote

import uvicorn
from fastapi import FastAPI
from fastapi import HTTPException
from fastapi import Query
from fastapi import Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from fastapi.responses import JSONResponse
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.base import BaseHTTPMiddleware

from dbclust.inject_spatialite import create_safe_connection, load_spatialite

from fdsnws.export_csv import generate_csv_response
from fdsnws.export_geojson import generate_geojson_response
from fdsnws.export_quakeml import generate_quake_response
from fdsnws.export_text import generate_text_response


class NormalizeSlashesMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Normalize the path in-place (no redirect)
        url_path = request.scope.get("path", "")
        normalized_path = re.sub(r"/+", "/", url_path)
        if normalized_path != url_path:
            request.scope["path"] = normalized_path
        return await call_next(request)


def iso_to_sqlite(dtstr):
    # Handles ISO 8601 formats with or without 'Z'
    dt = datetime.fromisoformat(dtstr.replace("Z", "+00:00"))
    # Return in standard SQLite format
    return dt.strftime("%Y-%m-%d %H:%M:%S.%f")


# (freqmin, freqmax) per allowed /api/waveforms filter key; "none" is the passthrough.
WAVEFORM_FILTERS = {
    "none": None,
    "4-20": (4.0, 20.0),
    "8-32": (8.0, 32.0),
}


def _decode_and_filter_waveforms(mseed_content: bytes, filter: str, window_s: float) -> list:
    """Parse a spectrocnn miniSEED payload into the JSON shape /api/waveforms
    returns: optionally bandpass-filter, truncate to window_s, and build one
    dict per trace. Pure CPU/numpy work - kept synchronous so the caller can
    run it via run_in_threadpool instead of blocking the event loop.

    A trace whose id doesn't split into the expected NET.STA.LOC.CHAN is
    skipped (logged) rather than aborting the whole response - one malformed
    station shouldn't take down every other station's waveforms.
    """
    import obspy

    st = obspy.read(io.BytesIO(mseed_content))

    band = WAVEFORM_FILTERS[filter]
    if band is not None:
        st.detrend("demean")
        st.taper(max_percentage=0.05)
        st.filter("bandpass", freqmin=band[0], freqmax=band[1], corners=4, zerophase=True)

    traces = []
    for tr in st:
        try:
            # NET.STA.LOC.BANDINST (band+instrument code, no component
            # letter) - matches spectrocnn's discriminate trace_id format
            # (e.g. "FR.CREF.00.EH"), not the full 4-char SEED channel.
            net, sta, loc, chan = tr.id.split(".")
            if not chan:
                raise ValueError(f"empty channel code in trace id {tr.id!r}")
            sampling_rate = tr.stats.sampling_rate
            if not sampling_rate:
                raise ValueError(f"zero/invalid sampling_rate for trace {tr.id!r}")

            n_keep = min(tr.stats.npts, max(1, int(window_s * sampling_rate)))
            data = tr.data[:n_keep].astype(float)
            if band is None:
                # Remove DC offset using the pre-event portion of the trace
                # (the offset_before_P noise window) so unfiltered waveforms
                # are centered like the previous JSON endpoint's output.
                # Filtered traces are already demeaned above.
                noise_samples = min(n_keep, max(1, int(15 * sampling_rate)))
                data -= data[:noise_samples].mean()

            traces.append(
                {
                    "trace_id": f"{net}.{sta}.{loc}.{chan[:2]}",
                    "component": chan[-1],
                    "data": data.tolist(),
                    "sampling_rate": sampling_rate,
                    "start_time": tr.stats.starttime.isoformat() + "Z",
                    "duration": n_keep / sampling_rate,
                    "n_samples": n_keep,
                }
            )
        except Exception as e:
            logging.warning(f"Skipping malformed trace {tr.id!r} in waveforms response: {e}")
            continue

    return traces


def create_app(
    db_path: str,
    debug=False,
    spectrocnn_url: Optional[str] = None,
    waveforms_window_s: float = 60.0,
):
    app = FastAPI(
        debug=debug,
        title="FDSN Web Service (Spatialite)",
        description="FDSN-compliant web service for querying seismic event data (Spatialite)",
        version="1.2.0",
    )

    # Store the database path in the app's state
    app.state.db_name = os.path.basename(db_path)
    app.state.spectrocnn_url = spectrocnn_url
    app.state.waveforms_window_s = waveforms_window_s

    app.add_middleware(NormalizeSlashesMiddleware)

    # Get the directory containing this file
    base_dir = Path(__file__).parent
    templates_dir = base_dir / "templates"
    static_dir = base_dir / "static"

    templates = Jinja2Templates(directory=str(templates_dir))
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    # Dependency: get a read-only connection and load spatialite
    # def get_db_connection():
    #     conn = sqlite3.connect(
    #         f"file:{db_path}?mode=ro", uri=True, check_same_thread=False
    #     )
    #     try:
    #         conn.enable_load_extension(True)
    #         spatialite_paths = [
    #             "/opt/homebrew/lib/mod_spatialite.dylib",
    #             "/usr/local/lib/mod_spatialite.dylib",
    #             "mod_spatialite",
    #         ]

    #         loaded = False
    #         for path in spatialite_paths:
    #             try:
    #                 conn.load_extension(path)
    #                 loaded = True
    #                 break
    #             except Exception as e:
    #                 logging.debug(f"Could not load spatialite from {path}: {e}")

    #         if not loaded:
    #             error_msg = """
    #             Could not load Spatialite extension. This extension is required for spatial operations.

    #             Installation instructions:

    #             On macOS (using Homebrew):
    #                 brew install libspatialite

    #             On Debian/Ubuntu:
    #                 sudo apt-get install libsqlite3-mod-spatialite

    #             For more details, see the project's documentation or README.
    #             """
    #             logging.error(error_msg)
    #             raise HTTPException(status_code=500, detail=error_msg)

    #     except Exception as e:
    #         error_msg = f"""
    #         Failed to load Spatialite extension: {e}

    #         This usually means the Spatialite system library is not installed or not in the library path.
    #         Please check the installation instructions above or in the project's documentation.
    #         """
    #         logging.error(error_msg)
    #         raise HTTPException(status_code=500, detail=error_msg)
    #     return conn

    @app.get("/")
    def root_redirect(request: Request):
        return RedirectResponse(url="/static/builder.html")

    wadl_dir = base_dir / "config" / "wadl" / "event"

    def _wadl_file_response(filename: str):
        file_path = wadl_dir / filename
        if not file_path.is_file():
            raise HTTPException(
                status_code=404, detail=f"Resource not found: {filename}"
            )
        return FileResponse(file_path, media_type="application/xml")

    @app.get("/fdsnws/event/1/application.wadl")
    async def get_event_application_wadl():
        return _wadl_file_response("application.wadl")

    @app.get("/fdsnws/event/1/catalogs")
    async def get_event_catalogs():
        return _wadl_file_response("catalogs")

    @app.get("/fdsnws/event/1/contributors")
    async def get_event_contributors():
        return _wadl_file_response("contributors")

    @app.get("/fdsnws/event/1/version")
    async def get_event_version():
        return "1.2.0"

    @app.get("/api/db-info")
    async def get_db_info():
        return {"db_name": app.state.db_name, "spectrocnn_url": app.state.spectrocnn_url}

    @app.get("/api/picks")
    def get_picks(eventid: str = Query(...)):
        """P/S arrival picks for one event, keyed by NET.STA.

        Used by the Waveforms modal to mark phase arrivals on the seismogram
        plot. station_name in `picks` is "NET.STA" (e.g. "FR.WALT"), so the
        client matches on the NET.STA prefix of spectrocnn's NET.STA.LOC.CHAN
        trace_id. Only arrivals from the event's preferred origin are
        returned - an event can have several origins (e.g. re-locations),
        each with its own arrival set, and mixing them produces duplicate/
        inconsistent picks for the same station.

        Declared as a plain (sync) function rather than async def, like
        query_events below: create_safe_connection/conn.execute are blocking
        SQLite calls, and a sync def route lets Starlette run them in its
        threadpool instead of stalling the single asyncio event loop that
        every other request (including unrelated ones) shares.
        """
        conn = create_safe_connection(f"file:{db_path}?mode=ro", uri=True)
        try:
            cursor = conn.execute(
                """
                SELECT p.station_name, a.name AS phase, p.pick_time
                FROM picks p
                JOIN arrivals a ON a.pick_id = p.id
                JOIN origins o ON o.id = a.origin_id
                WHERE p.event_id = ? AND o.preferred = 1 AND p.pick_time IS NOT NULL
                """,
                (eventid,),
            )
            # pick_time is stored as naive UTC (SQLite convention, see
            # iso_to_sqlite() above); append "Z" so clients parse it as UTC
            # rather than local time.
            picks = [
                {
                    "station": row[0],
                    "phase": row[1],
                    "time": row[2].replace(" ", "T") + "Z",
                }
                for row in cursor.fetchall()
            ]
            return {"picks": picks}
        finally:
            conn.close()

    @app.get("/api/waveforms")
    async def get_waveforms(
        eventid: str = Query(...),
        window_s: Optional[float] = Query(None, gt=0, le=120),
        filter: str = Query("none", pattern="^(none|4-20|8-32)$"),
    ):
        """Raw per-trace waveforms for one event, decoded from spectrocnn's
        miniSEED export rather than its JSON /seismogram endpoint.

        The JSON endpoint is meant for spectrocnn's own CNN/spectrogram
        pipeline and returns samples that are not reliably aligned with
        origin_time/picks (observed several-second offsets between the
        JSON's reported start_time and where the actual signal sits). The
        miniSEED export doesn't have that problem - its start_time matches
        origin_time - offset_before_P as expected - so this proxies that
        instead and decodes it server-side with obspy (already a dbclust
        dependency).

        window_s caps how much of the (up to 120s) miniSEED is returned,
        keeping the response light for the waveforms modal's canvas plot.
        Defaults to the --waveforms-window CLI setting when not given.
        filter optionally applies a zero-phase Butterworth bandpass (on the
        full trace, before truncating to window_s, to avoid filter edge
        transients showing up inside the displayed window).
        """
        effective_window_s: float = window_s if window_s is not None else app.state.waveforms_window_s

        if not app.state.spectrocnn_url:
            raise HTTPException(status_code=503, detail="spectrocnn webservice not configured")

        import requests

        mseed_url = f"{app.state.spectrocnn_url}/api/v1/events/{quote(eventid, safe='')}/waveforms.mseed"
        try:
            resp = await run_in_threadpool(requests.get, mseed_url, timeout=60)
            resp.raise_for_status()
        except requests.RequestException as e:
            raise HTTPException(status_code=502, detail=f"Could not fetch waveforms from spectrocnn: {e}")

        try:
            # obspy.read/detrend/taper/filter and the per-trace slicing below
            # are CPU-bound (a zero-phase Butterworth bandpass over a
            # multi-station, up-to-120s miniSEED stream is tens to hundreds
            # of ms of pure numpy/scipy work) - threadpooled like the fetch
            # above so it doesn't stall the event loop for concurrent
            # requests (including unrelated ones like /api/db-info).
            traces = await run_in_threadpool(
                _decode_and_filter_waveforms, resp.content, filter, effective_window_s
            )
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Could not parse miniSEED from spectrocnn: {e}")

        return {"traces": traces}

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
        format: str = Query(
            "xml", pattern="^(xml|json|text|csv|quakeml|geojson|html)$"
        ),
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
        if debug:
            # force level    to debug
            logging.getLogger().setLevel(logging.DEBUG)

        #conn = get_db_connection()
        conn = create_safe_connection(f"file:{db_path}?mode=ro", uri=True)
        try:
            load_spatialite(conn)
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
                where.append("ST_Distance(geometry, ST_GeomFromText(?, 4326)) <= ?")
                point_wkt = f"POINT({longitude} {latitude})"
                params.extend([point_wkt, maxradius])
                if minradius and minradius > 0:
                    where.append("ST_Distance(geometry, ST_GeomFromText(?, 4326)) >= ?")
                    params.extend([point_wkt, minradius])

            sql = f"SELECT * FROM event_coordinates WHERE {' AND '.join(where)}"
            if orderby == "time-asc":
                sql += " ORDER BY time ASC"
            elif orderby == "magnitude":
                sql += " ORDER BY magnitude DESC"
            elif orderby == "magnitude-asc":
                sql += " ORDER BY magnitude ASC"
            elif orderby == "magnitude-desc":
                sql += " ORDER BY magnitude DESC"
            else:
                # Default and "time"/"time-desc"
                sql += " ORDER BY time DESC"

            if limit:
                sql += " LIMIT ?"
                params.append(limit)
            elif offset:
                sql += " LIMIT -1"  # SQLite requires LIMIT before OFFSET; -1 = unlimited
            if offset:
                sql += " OFFSET ?"
                params.append(offset)

            logging.debug(f"SQL: {sql} | Params: {params}")
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
                headers = {"Content-Disposition": f"attachment; filename={filename}"}
                return JSONResponse({"events": results}, headers=headers)
            elif format.lower() == "csv":
                return generate_csv_response(results)
            elif format.lower() == "text":
                return generate_text_response(results, catalog=app.state.db_name)
            elif format.lower() == "geojson":
                return generate_geojson_response(results)
            elif format.lower() == "quakeml" or format.lower() == "xml":
                # For QuakeML, pass event_ids and db_path to the generator
                event_ids = [r["event_id"] for r in results if "event_id" in r]
                return generate_quake_response(event_ids, db_path)
            elif format.lower() == "html":
                # Decode JSON-encoded list columns so the template can embed
                # them as proper JS arrays instead of raw JSON strings.
                html_results = []
                for row in results:
                    html_row = dict(row)
                    for key in ("agencies_list", "agency_names"):
                        raw_value = html_row.get(key)
                        if raw_value:
                            try:
                                html_row[key] = json.loads(raw_value)
                            except (TypeError, ValueError):
                                html_row[key] = []
                        else:
                            html_row[key] = []
                    # event_type_agencies_json is a dict {agency: event_type}, not a list
                    raw_eta = html_row.get("event_type_agencies_json")
                    if raw_eta:
                        try:
                            html_row["event_type_agencies_json"] = json.loads(raw_eta)
                        except (TypeError, ValueError):
                            html_row["event_type_agencies_json"] = {}
                    else:
                        html_row["event_type_agencies_json"] = {}
                    html_results.append(html_row)
                return templates.TemplateResponse(
                    "table.html", {"request": request, "events": html_results}
                )
            else:
                raise HTTPException(
                    status_code=400, detail=f"Unsupported format: {format}"
                )
        finally:
            conn.close()

    return app


def main():
    """main function to start the FDSN Web Service server.

    Usage example:
        dbclust --db /path/to/my_database.db --port 9998 --host localhost
    """
    PORT = 8000
    HOST = "localhost"

    # Command-line argument configuration
    parser = argparse.ArgumentParser(
        description="FDSN Web Service server for seismic data (Spatialite)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required arguments
    parser.add_argument(
        "--db",
        required=True,
        help="Path to the SQLite/Spatialite database",
        type=str,
    )

    # Optional arguments
    parser.add_argument("--port", type=int, default=PORT, help="Server listening port")

    parser.add_argument(
        "--host", default=HOST, help="Server listening IP address or hostname"
    )

    parser.add_argument(
        "--debug", action="store_true", help="Enable debug mode (more logs)"
    )

    parser.add_argument(
        "--spectrocnn-url",
        default="http://127.0.0.1:9000",
        help="Base URL of the spectrocnn webservice used to display waveforms "
        "(set to an empty string to disable the Waveforms button)",
    )

    parser.add_argument(
        "--waveforms-window",
        type=float,
        default=60.0,
        help="Duration in seconds of the waveform window shown in the Waveforms modal",
    )

    # Parse arguments
    args = parser.parse_args()

    # Check database path
    db_path = Path(args.db).expanduser()  # Handles ~ in paths
    if not db_path.exists():
        logging.error(f"ERROR: Database file does not exist: {db_path}")
        sys.exit(1)

    # Validate database
    try:
        with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
            cursor = conn.cursor()
            # Check for the presence of the event_coordinates table
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='view' AND name='event_coordinates'"
            )
            if not cursor.fetchone():
                logging.warning(
                    "WARNING: The view 'event_coordinates' does not exist in the database."
                )
            else:
                logging.info("Successfully connected to the SQLite database.")
    except sqlite3.Error as e:
        logging.error(f"ERROR: Unable to connect to the database: {e}")
        sys.exit(1)

    # Logging configuration
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    logging.info(f"Using database: {db_path}")
    logging.info(f"Starting server on {args.host}:{args.port}...")

    # Create and start the application
    app = create_app(
        db_path=str(db_path),
        debug=args.debug,
        spectrocnn_url=args.spectrocnn_url or None,
        waveforms_window_s=args.waveforms_window,
    )

    if args.debug:
        logging.warning(
            "--debug requests uvicorn auto-reload, but the app is built from "
            "CLI args via a factory (create_app), which reload's import-string "
            "mechanism cannot re-invoke. Auto-reload is disabled; restart the "
            "server manually to pick up code changes."
        )

    # Uvicorn server configuration
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="debug" if args.debug else "info",
        reload=False,
    )


if __name__ == "__main__":
    main()
