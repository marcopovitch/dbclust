#!/usr/bin/env python
import io
import sqlite3

from fastapi.responses import StreamingResponse

from dbclust.inject_spatialite import process_quakeml_row

QUAKEML_HEADER = """<?xml version="1.0" encoding="UTF-8"?>
<q:quakeml xmlns:q="http://quakeml.org/xmlns/quakeml/1.2" xmlns="http://quakeml.org/xmlns/bed/1.2">
    <q:eventParameters>
"""

QUAKEML_FOOTER = """
    </q:eventParameters>
</q:quakeml>
"""

# Define the namespace used in the QuakeML XML
NAMESPACE = {
    "": "http://quakeml.org/xmlns/bed/1.2",
}


async def generate_quake_response(results, conn):
    """
    Stream QuakeML events from the SQLite database.

    Arguments:
        results (list): List of dictionaries containing event_id.
        db_path (str): Path to the SQLite database.

    Returns:
        StreamingResponse with the generated QuakeML.
    """

    async def quake_generator():
        event_ids = [result["event_id"] for result in results if "event_id" in result]
        if not event_ids:
            yield QUAKEML_HEADER
            yield "    <!-- No seismic events found -->\n"
            yield QUAKEML_FOOTER
            return

        cursor = conn.cursor()

        # Always send the QuakeML header at the beginning
        yield QUAKEML_HEADER

        for event_id in event_ids:
            # Fetch the QuakeML data from the database
            cursor.execute(
                "SELECT data FROM quakeml WHERE event_id = ?", (event_id,)
            )
            row = cursor.fetchone()

            if row:
                # Process and stream each event individually, data is binary (zlib compressed)
                output_buffer = io.BytesIO()
                process_quakeml_row(row, output_buffer, NAMESPACE)
                yield output_buffer.getvalue().decode("utf-8") + "\n"
            else:
                yield f"    <!-- No QuakeML found for event_id: {event_id} -->\n"

        # Always send the QuakeML footer
        yield QUAKEML_FOOTER

    return StreamingResponse(
        quake_generator(),
        media_type="application/xml",
        headers={"Content-Disposition": 'attachment; filename="events.xml"'},
    )
