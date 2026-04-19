from tools.places import search_places
from tools.weather import get_weather
from tools.flights import search_flights
from tools.vacation_rentals import find_vacation_rentals

TOOL_MAP = {
    "search_places": search_places,
    "get_weather": get_weather,
    "search_flights": search_flights,
    "find_vacation_rentals": find_vacation_rentals,
}

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "search_places",
            "description": "Search for hotels, restaurants, attractions, activities, or car rental places at a destination using Google Places. This is your primary tool for finding real places with ratings, addresses, price levels, and Google Maps links. Use it for: hotels ('hotels in Paris'), restaurants ('romantic restaurants in Tokyo'), attractions ('museums in London'), car rentals ('car rental near LAX'), nightlife ('bars in Barcelona'), etc.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query (e.g., 'boutique hotels in Paris', 'romantic restaurants in Tokyo', 'car rental near Los Angeles airport')",
                    },
                    "location": {
                        "type": "string",
                        "description": "City or area name for the search",
                    },
                    "type": {
                        "type": "string",
                        "description": "Optional place type filter",
                        "enum": ["restaurant", "tourist_attraction", "museum", "car_rental", "amusement_park", "night_club", "spa", "park", "shopping_mall", "cafe", "bar", "lodging", "hotel"],
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Number of results to return (default 5, max 10)",
                    },
                },
                "required": ["query", "location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather forecast for a location and date range. Returns daily temperature, precipitation probability, and weather conditions. Use this EARLY in planning to inform activity choices (e.g., indoor activities on rainy days, outdoor activities on sunny days).",
            "parameters": {
                "type": "object",
                "properties": {
                    "latitude": {"type": "number", "description": "Latitude of the location"},
                    "longitude": {"type": "number", "description": "Longitude of the location"},
                    "start_date": {"type": "string", "description": "Start date in YYYY-MM-DD format"},
                    "end_date": {"type": "string", "description": "End date in YYYY-MM-DD format"},
                },
                "required": ["latitude", "longitude", "start_date", "end_date"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_flights",
            "description": "Generate a Google Flights search link and provide estimated price ranges for flights between two cities. Use this when the user needs air transportation. Returns a clickable Google Flights URL where users can see real-time prices and book. Do NOT call this if the user chose road trip / driving as transportation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "origin": {
                        "type": "string",
                        "description": "IATA airport code of departure city (e.g., 'JFK', 'LAX', 'RDU')",
                    },
                    "destination": {
                        "type": "string",
                        "description": "IATA airport code of destination city (e.g., 'CDG', 'NRT', 'LHR')",
                    },
                    "departure_date": {
                        "type": "string",
                        "description": "Departure date in YYYY-MM-DD format",
                    },
                    "return_date": {
                        "type": "string",
                        "description": "Return date in YYYY-MM-DD format (optional for one-way)",
                    },
                    "num_passengers": {
                        "type": "integer",
                        "description": "Number of adult passengers",
                    },
                },
                "required": ["origin", "destination", "departure_date", "num_passengers"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_vacation_rentals",
            "description": "Generate a vacation-rental search link for a destination, with dates, guests, and price filters pre-applied. Returns a clickable Google search URL that surfaces Google's native hotels-and-rentals widget plus listings from Airbnb, Vrbo, Booking, etc. Use this ALONGSIDE search_places (for hotels) so users get BOTH hotel and short-term-rental options for accommodation. The link routes through Google because direct deep-links to Airbnb / Booking / Vrbo are gated by their anti-bot pages.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "Destination city or area (e.g., 'Paris, France', 'Tokyo, Japan')",
                    },
                    "checkin": {"type": "string", "description": "Check-in date YYYY-MM-DD"},
                    "checkout": {"type": "string", "description": "Check-out date YYYY-MM-DD"},
                    "adults": {"type": "integer", "description": "Number of adult guests"},
                    "children": {
                        "type": "integer",
                        "description": "Number of children (optional, default 0)",
                    },
                    "min_price": {
                        "type": "integer",
                        "description": "Minimum price per night in USD (optional)",
                    },
                    "max_price": {
                        "type": "integer",
                        "description": "Maximum price per night in USD (optional)",
                    },
                    "property_type": {
                        "type": "string",
                        "description": "Type of property (optional)",
                        "enum": ["entire_home", "private_room", "shared_room"],
                    },
                },
                "required": ["location", "checkin", "checkout", "adults"],
            },
        },
    },
]
