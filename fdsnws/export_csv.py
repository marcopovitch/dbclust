#!/usr/bin/env python
import csv
import io
from datetime import datetime

from fastapi.responses import PlainTextResponse
from fastapi.responses import StreamingResponse


def generate_csv_response(results: list[dict]):
    """
    Export results to CSV format.

    Args:
        results (list of dict): List of event results as dictionaries.

    Returns:
        StreamingResponse: Response streaming CSV data.
    """
    file_extension = "csv"

    output = io.StringIO()
    # Use DictWriter for robust CSV output with column names
    fieldnames = list(results[0].keys())
    writer = csv.DictWriter(
        output,
        fieldnames=fieldnames,
        delimiter=",",
        quotechar='"',
        quoting=csv.QUOTE_MINIMAL,
    )
    writer.writeheader()
    for event in results:
        writer.writerow(event)

    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    filename = f"events-{timestamp}.csv"
    output.seek(0)
    return StreamingResponse(
        output,
        media_type="text/csv",
        headers={
            "Content-Disposition": f"attachment; filename={filename}"
        },
    )
