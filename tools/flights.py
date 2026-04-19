from urllib.parse import urlencode

# Rough price estimate ranges by route type (for budget planning only)
PRICE_ESTIMATES = {
    "domestic_short": {"min": 80, "max": 250, "label": "Domestic short-haul (< 500 mi)"},
    "domestic_long": {"min": 150, "max": 450, "label": "Domestic long-haul (500+ mi)"},
    "international_short": {"min": 250, "max": 700, "label": "International short-haul"},
    "international_long": {"min": 400, "max": 1500, "label": "International long-haul"},
    "intercontinental": {"min": 600, "max": 2500, "label": "Intercontinental"},
}

US_AIRPORTS = {"JFK", "LAX", "ORD", "ATL", "DFW", "SFO", "SEA", "MIA", "BOS", "DEN", "IAH", "EWR", "LGA", "PHX", "RDU", "CLT", "MSP", "DTW", "PHL", "IAD", "DCA", "SAN", "TPA", "MCO", "LAS", "SLC", "PDX", "STL", "BNA", "AUS", "HNL"}
EUROPE_AIRPORTS = {"CDG", "LHR", "FRA", "AMS", "MAD", "BCN", "FCO", "MUC", "ZRH", "VIE", "CPH", "OSL", "ARN", "HEL", "DUB", "LIS", "ATH", "IST", "BRU", "MXP", "ORY", "LGW", "STN", "EDI", "MAN"}
ASIA_AIRPORTS = {"NRT", "HND", "PEK", "PVG", "HKG", "SIN", "BKK", "ICN", "KIX", "TPE", "DEL", "BOM", "KUL", "CGK", "MNL"}


def _classify_route(origin: str, destination: str) -> str:
    """Classify a route for price estimation."""
    o = origin.upper()
    d = destination.upper()
    o_us = o in US_AIRPORTS
    d_us = d in US_AIRPORTS
    o_eu = o in EUROPE_AIRPORTS
    d_eu = d in EUROPE_AIRPORTS
    o_asia = o in ASIA_AIRPORTS
    d_asia = d in ASIA_AIRPORTS

    if o_us and d_us:
        long_haul_pairs = {("JFK", "LAX"), ("LAX", "JFK"), ("JFK", "SFO"), ("SFO", "JFK"),
                           ("BOS", "LAX"), ("LAX", "BOS"), ("MIA", "SEA"), ("SEA", "MIA"),
                           ("JFK", "SEA"), ("SEA", "JFK")}
        if (o, d) in long_haul_pairs or (d, o) in long_haul_pairs:
            return "domestic_long"
        return "domestic_short"

    if (o_us and d_eu) or (o_eu and d_us):
        return "international_long"

    if (o_us and d_asia) or (o_asia and d_us):
        return "intercontinental"
    if (o_eu and d_asia) or (o_asia and d_eu):
        return "international_long"

    if o_eu and d_eu:
        return "international_short"

    if o_asia and d_asia:
        return "international_short"

    return "international_long"


def search_flights(origin: str, destination: str, departure_date: str,
                   num_passengers: int, return_date: str = None) -> dict:
    """Generate Google Flights search URL and price estimates."""

    query_parts = f"Flights from {origin} to {destination} on {departure_date}"
    if return_date:
        query_parts += f" return {return_date}"

    google_flights_url = f"https://www.google.com/travel/flights?q={query_parts.replace(' ', '+')}"

    route_type = _classify_route(origin, destination)
    estimate = PRICE_ESTIMATES.get(route_type, PRICE_ESTIMATES["international_long"])

    price_per_person_min = estimate["min"]
    price_per_person_max = estimate["max"]

    return {
        "google_flights_url": google_flights_url,
        "origin": origin,
        "destination": destination,
        "departure_date": departure_date,
        "return_date": return_date,
        "num_passengers": num_passengers,
        "route_type": estimate["label"],
        "estimated_price_per_person": f"${price_per_person_min} - ${price_per_person_max} USD",
        "estimated_total": f"${price_per_person_min * num_passengers} - ${price_per_person_max * num_passengers} USD",
        "note": "These are rough estimates for budget planning. Click the Google Flights link for real-time prices and booking.",
    }
