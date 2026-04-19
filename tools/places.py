import requests
from config import GOOGLE_PLACES_API_KEY


def search_places(query: str, location: str, type: str = None,
                  max_results: int = 5) -> dict:
    """Search for places via Google Places API (New) Text Search."""
    url = "https://places.googleapis.com/v1/places:searchText"

    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": GOOGLE_PLACES_API_KEY,
        "X-Goog-FieldMask": "places.displayName,places.formattedAddress,places.rating,places.userRatingCount,places.priceLevel,places.googleMapsUri,places.types,places.editorialSummary",
    }

    body = {
        "textQuery": f"{query} in {location}",
        "maxResultCount": min(max_results, 10),
    }

    if type:
        body["includedType"] = type

    resp = requests.post(url, headers=headers, json=body, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    places = []
    for place in data.get("places", [])[:max_results]:
        places.append({
            "name": place.get("displayName", {}).get("text", "Unknown"),
            "address": place.get("formattedAddress", ""),
            "rating": place.get("rating", None),
            "num_reviews": place.get("userRatingCount", 0),
            "price_level": place.get("priceLevel", "N/A"),
            "google_maps_url": place.get("googleMapsUri", ""),
            "types": place.get("types", []),
            "description": place.get("editorialSummary", {}).get("text", ""),
        })

    return {"places": places, "total_found": len(places)}


# Place types that indicate a "city / region" rather than a POI.
_LOCALITY_TYPES = {
    "locality", "administrative_area_level_1", "administrative_area_level_2",
    "administrative_area_level_3", "country", "political",
    "sublocality", "neighborhood", "postal_town",
}


def resolve_location(query: str, max_candidates: int = 5) -> dict:
    """Resolve a free-text destination (e.g., 'Durham') to canonical city candidates.

    Returns up to `max_candidates` matches that look like cities/regions, each with
    a formatted address and lat/lng. Used by the UI to disambiguate ambiguous
    destinations BEFORE running the agent.
    """
    url = "https://places.googleapis.com/v1/places:searchText"

    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": GOOGLE_PLACES_API_KEY,
        "X-Goog-FieldMask": "places.displayName,places.formattedAddress,places.location,places.types",
    }

    body = {"textQuery": query, "maxResultCount": 10}

    resp = requests.post(url, headers=headers, json=body, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    candidates = []
    seen_addresses = set()
    for place in data.get("places", []):
        types = set(place.get("types", []))
        # Keep only locality-like results, drop POIs (restaurants, hotels, etc.)
        if not (types & _LOCALITY_TYPES):
            continue
        addr = place.get("formattedAddress", "")
        if not addr or addr in seen_addresses:
            continue
        seen_addresses.add(addr)
        loc = place.get("location", {})
        candidates.append({
            "name": place.get("displayName", {}).get("text", ""),
            "formatted_address": addr,
            "latitude": loc.get("latitude"),
            "longitude": loc.get("longitude"),
            "types": list(types),
        })
        if len(candidates) >= max_candidates:
            break

    return {"candidates": candidates}
