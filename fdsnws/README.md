# DBClust FDSN Web Service

This directory provides a FastAPI-based implementation of an FDSN-compatible web service for serving earthquake event data from an SQLite database. The service supports multiple output formats (CSV, GeoJSON, QuakeML, plain text) and is designed for efficient querying and robust error handling.

## References
- Official FDSNWS documentation: https://www.fdsn.org/webservices/

## Features
- Query earthquake event data from an SQLite database
- Export results in CSV, GeoJSON, QuakeML, and plain text formats
- Standards-compliant HTTP error responses
- Efficient handling of large event sets
- Clean database connection management

## Requirements
- Python 3.9+
- FastAPI
- Uvicorn
- SQLite3 with the `mod_spatialite` extension (for spatial queries)
- (Optional) Jinja2 for templating static pages

Install dependencies with:
```bash
pip install fastapi uvicorn jinja2
```

## Usage
To launch the webservice, run:

```bash
fdsnws-server --db /path/to/db --port 51243 --debug
```

- `--db /path/to/db` : Path to your SQLite database file
- `--port 51243`     : Port to serve the API on (default: 8000)
- `--debug`          : Enable debug mode for verbose logging

`fdsnws-server` is the console script installed by this package (see
`pyproject.toml`'s `[project.scripts]`, backed by `fdsnws.server:main`).

## Endpoints
- `/query` : Query events with flexible parameters
- `/static/` : Static files (e.g., builder.html)

## Notes
- The service expects your database to have appropriate tables (e.g., `event_coordinates`, `quakeml`).
- For QuakeML export, the `mod_spatialite` extension must be available.
- Error handling uses FastAPI's `HTTPException` for clear responses.

## Development
- See `server.py` for the main application logic.
- Output formatters are in `export_csv.py`, `export_geojson.py`, `export_quakeml.py`, and `export_text.py`.
