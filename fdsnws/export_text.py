#!/usr/bin/env python
import csv
import io
from datetime import datetime

from fastapi.responses import PlainTextResponse
from fastapi.responses import StreamingResponse

def generate_text_response(results: list[dict]):
    """
    Export results to text format.

    Args:
        results (list of dict): List of event results as dictionaries.

    Returns:
        StreamingResponse: Response streaming text data.
    """
    file_extension = "txt"

    output = io.StringIO()
    writer = csv.writer(
        output, delimiter="|", quotechar='"', quoting=csv.QUOTE_MINIMAL
    )

    # FDSNWS-Event header
    writer.writerow(
        [
            "#EventID",
            "Time",
            "Latitude",
            "Longitude",
            "Depth/km",
            "Author",
            "Catalog",
            "Contributor",
            "ContributorID",
            "MagType",
            "Magnitude",
            "MagAuthor",
            "EventLocationName",
            "EventType",
        ]
    )

    # Write data
    for event in results:
        writer.writerow(
            [
                event.get("event_id", ""),
                event.get("time", ""),
                event.get("latitude", ""),
                event.get("longitude", ""),
                event.get("depth_km", ""),
                event.get("author", ""),
                event.get("catalog", ""),
                event.get("contributor", ""),
                event.get("contributor_id", ""),
                event.get("mag_type", ""),
                event.get("magnitude", ""),
                event.get("mag_author", ""),
                event.get("event_location_name", ""),
                event.get("event_type", ""),
            ]
        )

    output.seek(0)
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    filename = f"events-{timestamp}.{file_extension}"
    return StreamingResponse(
        output,
        media_type="text/plain",
        headers={
            "Content-Disposition": f"attachment; filename={filename}"
        }
    )
