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

NAMESPACE = {
    "": "http://quakeml.org/xmlns/bed/1.2",
}

def generate_quake_response(results, conn: sqlite3.Connection):
    """
    Generate a streaming QuakeML XML response using a synchronous sqlite3 connection.

    Args:
        results (list of dict): List of event results with 'event_id' keys.
        conn (sqlite3.Connection): Open synchronous sqlite3 connection.

    Returns:
        StreamingResponse: Response streaming XML data.
    """
    def quake_generator():
        event_ids = [result["event_id"] for result in results if "event_id" in result]

        if not event_ids:
            yield QUAKEML_HEADER
            yield "    <!-- No seismic events found -->\n"
            yield QUAKEML_FOOTER
            return

        yield QUAKEML_HEADER
        cursor = conn.cursor()
        for event_id in event_ids:
            cursor.execute("SELECT data FROM quakeml WHERE event_id = ?", (event_id,))
            row = cursor.fetchone()

            if row:
                output_buffer = io.BytesIO()
                process_quakeml_row(row, output_buffer, NAMESPACE)
                yield output_buffer.getvalue().decode("utf-8") + "\n"
            else:
                yield f"    <!-- No QuakeML found for event_id: {event_id} -->\n"

        yield QUAKEML_FOOTER

    return StreamingResponse(
        quake_generator(),
        media_type="application/xml",
        headers={"Content-Disposition": 'attachment; filename="events.xml"'},
    )
