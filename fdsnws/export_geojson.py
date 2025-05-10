#!/usr/bin/env python
from datetime import datetime
from datetime import timezone

from fastapi.responses import JSONResponse


def generate_geojson_response(results: list[dict]):
    """
    Generate a GeoJSON response from event data.

    Args:
        results (list of dict): List of event dictionaries with latitude and longitude.

    Returns:
        JSONResponse: Response containing GeoJSON data.
    """
    if not results:
        return JSONResponse(
            content={"type": "FeatureCollection", "features": []}, status_code=200
        )

    # Convert to GeoJSON
    features = []
    for event in results:
        # Skip events without coordinates
        if "latitude" not in event or "longitude" not in event:
            continue

        # Convert only 'time' to ISO 8601 UTC string if it's a datetime
        properties = {}
        for k, v in event.items():
            if k == "time" and isinstance(v, datetime):
                properties[k] = v.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
            else:
                properties[k] = v

        # Create GeoJSON feature
        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [
                    event.get("longitude"),
                    event.get("latitude"),
                    event.get("depth_km", 0) if "depth_km" in event else None,
                ],
            },
            "properties": {
                k: v
                for k, v in properties.items()
                if k not in ("latitude", "longitude", "depth_km")
            },
        }

        # Remove null values from coordinates
        feature["geometry"]["coordinates"] = [
            coord for coord in feature["geometry"]["coordinates"] if coord is not None
        ]

        features.append(feature)

    # Create GeoJSON FeatureCollection
    geojson = {
        "type": "FeatureCollection",
        "features": features,
        "metadata": {"generated": "FDSN Web Service", "count": len(features)},
    }

    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    filename = f"events-{timestamp}.geojson"
    headers = {"Content-Disposition": f"attachment; filename={filename}"}
    return JSONResponse(
        content=geojson,
        media_type="application/geo+json",
        headers=headers
    )
