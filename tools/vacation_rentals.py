"""Vacation-rental search-link generator. Pure Python, no API.

Why this is routed through Google search instead of linking directly to Airbnb:
in 2026, Airbnb / Booking.com / Vrbo all aggressively gate direct deep-links
from external sites with client-side JS anti-bot pages ("We'll be right back",
captchas, etc.). The URLs themselves are valid server-side (curl returns real
HTML), but their JS bundle refuses to render the listings unless the user has
the right cookies/session state. There is no fix from outside their site.

The reliable workaround — used by Kayak, Hopper, and most travel meta-search
apps — is to route the user through a Google search. Google never gates
external clicks, and it shows its native hotels-and-rentals widget at the top
of the results with the dates and guest count already applied.
"""
from urllib.parse import quote_plus
from datetime import datetime


def find_vacation_rentals(location: str, checkin: str, checkout: str, adults: int,
                          children: int = 0, min_price: int = None,
                          max_price: int = None, property_type: str = None) -> dict:
    """Generate a Google search URL pre-filled with vacation rental criteria.

    Returns the Google URL (primary, always works) plus the direct Airbnb URL
    (secondary — may or may not load depending on the user's browser session).
    """
    try:
        d1 = datetime.strptime(checkin, "%Y-%m-%d")
        d2 = datetime.strptime(checkout, "%Y-%m-%d")
        num_nights = max((d2 - d1).days, 1)
        date_str = f"{d1.strftime('%b %-d')} to {d2.strftime('%b %-d, %Y')}"
    except ValueError:
        num_nights = 1
        date_str = f"{checkin} to {checkout}"

    # Primary: Google search URL — always works, surfaces Google's native
    # hotels/rentals widget plus links to Airbnb, Vrbo, Booking, etc.
    total_guests = adults + (children or 0)
    query_terms = [
        "vacation rental",
        location,
        date_str,
        f"{total_guests} guests",
    ]
    if property_type == "entire_home":
        query_terms.insert(0, "entire home")
    elif property_type == "private_room":
        query_terms.insert(0, "private room")
    if max_price:
        query_terms.append(f"under ${max_price} per night")

    google_query = " ".join(query_terms)
    google_search_url = f"https://www.google.com/search?q={quote_plus(google_query)}"

    # Secondary: direct Airbnb deep link (kept as a fallback). Use city name
    # only for the slug — Airbnb's geocoder resolves it to the right place.
    city_only = location.split(",")[0].strip().replace(" ", "-") or "homes"
    airbnb_params = [f"checkin={checkin}", f"checkout={checkout}", f"adults={adults}"]
    if children:
        airbnb_params.append(f"children={children}")
    if min_price is not None:
        airbnb_params.append(f"price_min={min_price}")
    if max_price is not None:
        airbnb_params.append(f"price_max={max_price}")
    airbnb_direct_url = f"https://www.airbnb.com/s/{quote_plus(city_only)}/homes?" + "&".join(airbnb_params)

    result = {
        "vacation_rentals_url": google_search_url,
        "airbnb_direct_url": airbnb_direct_url,
        "location": location,
        "checkin": checkin,
        "checkout": checkout,
        "guests": total_guests,
        "num_nights": num_nights,
        "filters_applied": [],
        "note": (
            "The primary link routes through Google search (always works) and shows "
            "Google's native hotels & rentals widget with the dates and guests "
            "pre-applied. The airbnb_direct_url is a fallback — Airbnb sometimes "
            "gates external deep-links with a 'We'll be right back' page depending "
            "on the user's browser session."
        ),
    }

    if min_price or max_price:
        if min_price and max_price:
            price_str = f"${min_price}-${max_price}/night"
        elif max_price:
            price_str = f"Up to ${max_price}/night"
        else:
            price_str = f"From ${min_price}/night"
        result["price_filter"] = price_str
        result["filters_applied"].append(f"Price: {price_str}")

    if property_type:
        result["property_type"] = property_type.replace("_", " ").title()
        result["filters_applied"].append(f"Type: {result['property_type']}")

    return result
