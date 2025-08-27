#!/usr/bin/env python
import io
import sqlite3
from datetime import datetime

from fastapi.responses import StreamingResponse

from dbclust.inject_spatialite import process_quakeml_row, create_safe_connection

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


MAX_EVENT_BUFF = 500


def generate_quake_response(event_ids, db_path):
    def quake_generator():
        yield QUAKEML_HEADER
        try:
            conn = create_safe_connection(f"file:{db_path}?mode=ro", uri=True)
            # Process event_ids in chunks
            for i in range(0, len(event_ids), MAX_EVENT_BUFF):
                chunk = event_ids[i : i + MAX_EVENT_BUFF]
                placeholders = ",".join(["?"] * len(chunk))
                sql = f"SELECT event_id, data FROM quakeml WHERE event_id IN ({placeholders})"
                cursor = conn.cursor()
                cursor.execute(sql, chunk)
                rows = cursor.fetchall()
                found_ids = set()
                for event_id, compressed_data in rows:
                    found_ids.add(event_id)
                    # Handle compressed_data as before
                    if isinstance(compressed_data, memoryview):
                        compressed_data = compressed_data.tobytes()
                    elif isinstance(compressed_data, bytes):
                        pass
                    elif compressed_data is None:
                        continue  # skip
                    else:
                        raise TypeError(
                            f"Unexpected type for quakeml.data: {type(compressed_data)}"
                        )
                    output_buffer = io.BytesIO()
                    process_quakeml_row((compressed_data,), output_buffer, NAMESPACE)
                    yield output_buffer.getvalue().decode("utf-8") + "\n"
                # For event_ids not found in this batch, yield a comment
                missing_ids = set(chunk) - found_ids
                for missing_id in missing_ids:
                    yield f"    <!-- No QuakeML found for event_id: {missing_id} -->\n"
        finally:
            conn.close()
        yield QUAKEML_FOOTER

    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    filename = f"events-{timestamp}.xml"
    return StreamingResponse(
        quake_generator(),
        media_type="application/xml",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
