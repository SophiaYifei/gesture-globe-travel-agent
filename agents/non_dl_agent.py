"""
Non-DL Agent
Perception : Classical CV (color histograms, spatial coordinate lookup)
Planning   : Rule-based template system with lookup tables
Control    : Structured itinerary renderer
"""

import numpy as np
import json, os
from PIL import Image
import io

# ── Region database ────────────────────────────────────────────────────────────
REGIONS = [
    # (name, lat_min, lat_max, lng_min, lng_max, climate, biome, emoji)
    ("Iceland",          63,  67,  -24,  -13, "subarctic",   "tundra",        "🧊"),
    ("Norway",           57,  71,    4,   31, "subarctic",   "fjord",         "🏔️"),
    ("Scotland",         55,  59,   -8,    0, "temperate",   "highland",      "🏴󠁧󠁢󠁳󠁣󠁴󠁿"),
    ("France",           42,  51,   -5,    8, "temperate",   "mixed",         "🗼"),
    ("Italy",            36,  47,    6,   18, "mediterranean","coastal",      "🍕"),
    ("Greece",           35,  42,   20,   27, "mediterranean","coastal",      "🏛️"),
    ("Spain",            36,  44,  -10,    4, "mediterranean","coastal",      "🌞"),
    ("Morocco",          28,  36,  -13,    2, "arid",        "desert",        "🏜️"),
    ("Egypt",            22,  32,   25,   37, "arid",        "desert",        "🛕"),
    ("Kenya",            -5,   5,   34,   42, "tropical",    "savanna",       "🦁"),
    ("South Africa",    -35, -22,   16,   33, "temperate",   "diverse",       "🦓"),
    ("Japan",            31,  45,  130,  145, "temperate",   "island",        "🗾"),
    ("Thailand",          5,  21,   97,  106, "tropical",    "coastal",       "🌴"),
    ("Bali",             -9,  -8,  115,  116, "tropical",    "island",        "🌺"),
    ("India",             8,  37,   68,   97, "tropical",    "diverse",       "🕌"),
    ("Nepal",            26,  30,   80,   89, "alpine",      "mountain",      "🏔️"),
    ("Australia",       -39, -10,  114,  154, "diverse",     "outback",       "🦘"),
    ("New Zealand",     -47, -34,  166,  178, "temperate",   "island",        "🥝"),
    ("Peru",            -18,   0,  -81,  -68, "diverse",     "mountain",      "🦙"),
    ("Brazil",          -34,   5,  -74,  -34, "tropical",    "rainforest",    "🌿"),
    ("Mexico",           15,  33,  -118, -86, "diverse",     "coastal",       "🌮"),
    ("Cuba",             19,  24,  -85,  -74, "tropical",    "island",        "🎶"),
    ("USA Southwest",    30,  42,  -120, -102,"arid",        "desert",        "🌵"),
    ("USA Northeast",    40,  48,   -80,  -67,"temperate",   "forest",        "🍁"),
    ("Canada Rockies",   49,  57,  -120, -110,"alpine",      "mountain",      "🏔️"),
    ("Patagonia",       -55, -40,   -76,  -63,"subarctic",   "wilderness",    "🦅"),
    ("Maldives",         -1,   8,   72,   74, "tropical",    "atoll",         "🏝️"),
    ("Caribbean",        10,  25,  -85,  -60, "tropical",    "island",        "🌊"),
    ("Scandinavia",      55,  70,    5,   30, "subarctic",   "nordic",        "🌌"),
    ("Turkey",           36,  42,   26,   45, "mediterranean","diverse",      "🕌"),
    ("Vietnam",           8,  23,  102,  110, "tropical",    "coastal",       "🍜"),
    ("China",            20,  53,   73,  135, "diverse",     "diverse",       "🐉"),
    ("Portugal",         36,  42,  -10,   -6, "mediterranean","coastal",      "🎭"),
    ("Croatia",          42,  46,   13,   19, "mediterranean","coastal",      "⛵"),
    ("Hawaii",           18,  22, -161, -154, "tropical",    "island",        "🌺"),
    ("Pacific Ocean",   -60,  60,  -180, -100,"oceanic",     "ocean",         "🌊"),
    ("Atlantic Ocean",  -60,  60,   -60,   20,"oceanic",     "ocean",         "🌊"),
    ("Indian Ocean",    -60,  30,    20,  100,"oceanic",     "ocean",         "🌊"),
]

# ── Activity / tip templates per biome/climate ────────────────────────────────
TEMPLATES = {
    "tropical": {
        "activities": [
            "Snorkeling and diving at coral reefs",
            "Jungle trekking and wildlife spotting",
            "Beach relaxation and water sports",
            "Visit local night markets for street food",
            "Sunrise hike to a viewpoint",
            "Kayaking through mangroves",
            "Cultural temple or village tour",
        ],
        "food": ["Fresh seafood", "Tropical fruits", "Coconut-based dishes", "Street food tours"],
        "tips": [
            "Best visited Nov–Apr to avoid monsoon season",
            "Pack light, breathable clothing and strong sunscreen",
            "Stay hydrated — heat and humidity are intense",
            "Mosquito repellent is essential",
        ],
        "best_season": "November – April",
        "budget_per_day": "$60–$150 USD",
    },
    "mediterranean": {
        "activities": [
            "Explore ancient ruins and historical sites",
            "Coastal boat tours and swimming coves",
            "Wine tasting at local vineyards",
            "Wander old town alleyways and markets",
            "Sunset watching from a hilltop village",
            "Day trip to a nearby island",
            "Cooking class with local cuisine",
        ],
        "food": ["Tapas / mezze", "Fresh seafood", "Olive oil and local cheese", "Local wine"],
        "tips": [
            "July–August is peak season and very crowded — consider May or September",
            "Many sites close midday — plan morning visits",
            "Dress modestly when visiting religious sites",
        ],
        "best_season": "May – June, September – October",
        "budget_per_day": "$80–$200 USD",
    },
    "alpine": {
        "activities": [
            "Trekking and mountaineering",
            "Cable car or gondola rides for panoramic views",
            "Visit alpine lakes and waterfalls",
            "Skiing or snowboarding (winter)",
            "Mountain biking on trail networks",
            "Visit a high-altitude monastery or shrine",
            "Wildlife watching (ibex, eagles, marmots)",
        ],
        "food": ["Hearty mountain stews", "Yak or local meat dishes", "Cheese fondue / raclette", "Hot tea and local bread"],
        "tips": [
            "Acclimatize gradually to avoid altitude sickness",
            "Weather changes fast — always pack a rain layer",
            "Book mountain huts well in advance in summer",
        ],
        "best_season": "June – September (trekking), December – March (skiing)",
        "budget_per_day": "$50–$180 USD",
    },
    "arid": {
        "activities": [
            "Sunrise and sunset desert walks",
            "Camel trekking through sand dunes",
            "Stargazing (minimal light pollution)",
            "Visit ancient ruins or canyon landscapes",
            "4WD off-road adventure",
            "Explore oasis towns and souks",
            "Hot air balloon ride over the desert",
        ],
        "food": ["Slow-cooked tagine", "Dates and dried fruits", "Flatbreads and dips", "Mint tea"],
        "tips": [
            "Travel Oct–Mar to avoid extreme heat",
            "Cover up — sun protection and cultural respect",
            "Carry more water than you think you need",
        ],
        "best_season": "October – March",
        "budget_per_day": "$40–$130 USD",
    },
    "temperate": {
        "activities": [
            "Explore the city centre, museums and galleries",
            "Hiking or cycling through countryside",
            "Visit castles, cathedrals, or historic sites",
            "Day trip to a nearby town or village",
            "Farm-to-table dining experience",
            "Attend a local festival or market",
            "Boat or river cruise",
        ],
        "food": ["Seasonal local cuisine", "Pub / brasserie culture", "Fresh pastries and coffee", "Local craft beer or wine"],
        "tips": [
            "Weather is unpredictable — pack layers",
            "Spring and autumn are ideal for fewer crowds",
            "Book popular attractions in advance",
        ],
        "best_season": "April – June, September – October",
        "budget_per_day": "$100–$250 USD",
    },
    "subarctic": {
        "activities": [
            "Chase the Northern Lights (aurora borealis)",
            "Glacier hiking and ice cave exploration",
            "Whale watching boat tour",
            "Geothermal pools and natural hot springs",
            "Dog sledding or snowmobile tour",
            "Midnight sun experience (summer)",
            "Birdwatching for puffins and Arctic species",
        ],
        "food": ["Fresh Arctic fish (salmon, cod, arctic char)", "Lamb or reindeer dishes", "Skyr / Nordic dairy", "Craft beer"],
        "tips": [
            "Pack extreme cold-weather gear even in summer",
            "Northern Lights: best September – March",
            "Summer offers 24-hour daylight — bring eye masks",
        ],
        "best_season": "Jun–Aug (midnight sun), Sep–Mar (Northern Lights)",
        "budget_per_day": "$150–$350 USD",
    },
    "island": {
        "activities": [
            "Island hopping by ferry or small plane",
            "Snorkeling over coral gardens",
            "Visit remote fishing villages",
            "Cycling around the island perimeter",
            "Surf lesson or stand-up paddleboarding",
            "Beachside yoga and wellness retreat",
        ],
        "food": ["Grilled catch-of-the-day", "Tropical cocktails", "Local island specialties", "Fresh coconut"],
        "tips": [
            "Ferry schedules can be unreliable — build in buffer days",
            "Book accommodation early during peak season",
            "Respect marine protected areas",
        ],
        "best_season": "Varies by island — check local weather patterns",
        "budget_per_day": "$70–$200 USD",
    },
    "savanna": {
        "activities": [
            "Game drive safari at dawn and dusk",
            "Walking safari with armed ranger",
            "Hot air balloon over the savanna",
            "Visit a local Maasai or tribal village",
            "Birdwatching (500+ species common)",
            "Night game drive for nocturnal animals",
        ],
        "food": ["Nyama choma (grilled meat)", "Ugali with stew", "Fresh tropical fruit", "Bush dinner under the stars"],
        "tips": [
            "Book safari lodges 6–12 months in advance",
            "Neutral/khaki colours are best — avoid bright clothing",
            "Yellow fever vaccination often required",
            "Best wildlife: Great Migration Jul–Oct in East Africa",
        ],
        "best_season": "July – October",
        "budget_per_day": "$100–$500 USD",
    },
    "rainforest": {
        "activities": [
            "Guided Amazon or jungle river boat tour",
            "Canopy walkway and tree-top experience",
            "Indigenous community cultural visit",
            "Night walk for frogs, insects, and nocturnal creatures",
            "Piranha fishing",
            "Visit a wildlife rescue or rehabilitation centre",
        ],
        "food": ["Exotic river fish (pacu, tambaqui)", "Açaí and local fruits", "Traditional stews", "Cachaça cocktails"],
        "tips": [
            "Malaria prophylaxis strongly recommended",
            "Dry season (Jun–Nov) best for trails and wildlife",
            "Hire a local guide — the jungle is disorienting",
        ],
        "best_season": "June – November",
        "budget_per_day": "$60–$180 USD",
    },
    "mountain": {
        "activities": [
            "Multi-day trekking through mountain passes",
            "Visit ancient monasteries and spiritual sites",
            "Photography of dramatic landscapes",
            "Interact with local mountain communities",
            "Acclimatisation day hikes",
        ],
        "food": ["Dal bhat (lentil rice)", "Momos (dumplings)", "Sherpa stew", "Butter tea"],
        "tips": [
            "Altitude sickness is real — ascend slowly",
            "Hire a local guide and porter — supports the community",
            "Trekking permits required in many regions",
        ],
        "best_season": "March – May, September – November",
        "budget_per_day": "$40–$100 USD",
    },
    "ocean": {
        "activities": [
            "Deep sea fishing charter",
            "Cruise or sailing voyage",
            "Whale and dolphin watching",
            "Diving at underwater sea mounts",
        ],
        "food": ["Fresh seafood", "Island cuisine at ports of call"],
        "tips": [
            "You appear to have clicked on open ocean — try clicking a land destination!",
            "Consider nearby island or coastal destinations",
        ],
        "best_season": "Varies by ocean basin",
        "budget_per_day": "$200–$800 USD (cruise)",
    },
    "default": {
        "activities": [
            "Explore the local area on foot",
            "Visit nearby natural landmarks",
            "Try authentic local cuisine",
            "Connect with local guides for hidden gems",
        ],
        "food": ["Local cuisine", "Street food", "Regional specialties"],
        "tips": ["Research visa requirements", "Get travel insurance", "Learn a few local phrases"],
        "best_season": "Spring or Autumn",
        "budget_per_day": "$80–$200 USD",
    },
}

DAY_STRUCTURES = [
    {"label": "Day 1", "theme": "Arrival & Orientation", "slot": "activities", "idx": 0},
    {"label": "Day 2", "theme": "Signature Experience",  "slot": "activities", "idx": 1},
    {"label": "Day 3", "theme": "Cultural Deep Dive",    "slot": "activities", "idx": 2},
    {"label": "Day 4", "theme": "Adventure Day",         "slot": "activities", "idx": 3},
    {"label": "Day 5", "theme": "Hidden Gems & Leisure", "slot": "activities", "idx": 4},
    {"label": "Day 6", "theme": "Local Immersion",       "slot": "activities", "idx": 5},
    {"label": "Day 7", "theme": "Farewell & Reflection", "slot": "activities", "idx": 6},
]


class NonDLAgent:
    """
    Perception  : Coordinate-based geographic lookup + PIL color histogram analysis
    Planning    : Rule-based template + lookup table system
    Control     : Structured itinerary assembly
    """

    # ── PERCEPTION ─────────────────────────────────────────────────────────────
    def perceive(self, lat, lng, img_bytes=None):
        region_info = self._lookup_region(lat, lng)
        terrain_hint = self._analyze_image(img_bytes) if img_bytes else None
        if terrain_hint and region_info["biome"] == "diverse":
            region_info["biome"] = terrain_hint
        return region_info

    def _lookup_region(self, lat, lng):
        best = None
        for row in REGIONS:
            name, lat_min, lat_max, lng_min, lng_max, climate, biome, emoji = row
            if lat_min <= lat <= lat_max and lng_min <= lng <= lng_max:
                best = {"name": name, "climate": climate, "biome": biome, "emoji": emoji}
                break
        if best is None:
            # fallback: nearest by centroid distance
            best = {"name": "Unknown Region", "climate": "temperate", "biome": "default", "emoji": "🌍"}
        return best

    def _analyze_image(self, img_bytes):
        """Classical CV: analyze dominant color to hint at terrain type."""
        try:
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB").resize((64, 64))
            arr = np.array(img).reshape(-1, 3).astype(float)
            mean_color = arr.mean(axis=0)  # [R, G, B]
            r, g, b = mean_color

            # Simple color-to-terrain heuristic
            if b > r + 20 and b > g + 10:
                return "ocean"
            if g > r + 15 and g > b + 10:
                return "rainforest"
            if r > 160 and g > 120 and b < 80:
                return "arid"
            if r > 200 and g > 200 and b > 200:
                return "alpine"  # snow/ice
            return None
        except Exception:
            return None

    # ── PLANNING ───────────────────────────────────────────────────────────────
    def plan(self, region_info, preferences=None):
        preferences = preferences or {}
        biome = region_info.get("biome", "default")
        climate = region_info.get("climate", "temperate")

        template = TEMPLATES.get(biome) or TEMPLATES.get(climate) or TEMPLATES["default"]
        activities = template["activities"]
        duration = preferences.get("duration", 7)

        itinerary = []
        for i in range(min(duration, len(DAY_STRUCTURES))):
            day = DAY_STRUCTURES[i]
            act_idx = day["idx"] % len(activities)
            itinerary.append({
                "day": day["label"],
                "theme": day["theme"],
                "activity": activities[act_idx],
                "tip": template["tips"][i % len(template["tips"])],
            })

        return {
            "itinerary": itinerary,
            "food": template["food"],
            "best_season": template["best_season"],
            "budget": template["budget_per_day"],
            "tips": template["tips"],
        }

    # ── CONTROL ────────────────────────────────────────────────────────────────
    def run(self, lat, lng, img_bytes=None, preferences=None):
        region_info = self.perceive(lat, lng, img_bytes)
        plan = self.plan(region_info, preferences)
        return {
            "agent": "non_dl",
            "region": region_info,
            "plan": plan,
            "perception_method": "Color histogram + coordinate lookup",
        }
