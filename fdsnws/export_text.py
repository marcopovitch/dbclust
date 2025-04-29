#!/usr/bin/env python
import csv
import io

from fastapi.responses import PlainTextResponse
from fastapi.responses import StreamingResponse


def generate_text_response(data, delimiter="|"):
    if not data:
        return PlainTextResponse("No data available", status_code=404)
    file_extension = "txt"

    output = io.StringIO()
    writer = csv.writer(
        output, delimiter=delimiter, quotechar='"', quoting=csv.QUOTE_MINIMAL
    )

    # En-tête FDSNWS-Event
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

    # Écriture des données
    for event in data:
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
    return StreamingResponse(
        output,
        media_type="text/csv",
        headers={
            "Content-Disposition": f"attachment; filename=events.{file_extension}"
        },
    )
