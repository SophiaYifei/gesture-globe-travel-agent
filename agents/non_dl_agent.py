"""Non-DL Agent — rule-based travel planner.

Perception : Classical CV (color histogram + coordinate bbox lookup). UNCHANGED.
Planning   : Google Places POI search + deterministic scoring + slot filling.
             No LLM involved — every decision (which POI fills which day-slot)
             comes from a fixed scoring formula and a fixed slot schedule.
Control    : Structured + Markdown output, shape-compatible with DLAgent.

This replaces Tiffany's original biome-keyed template strings. We still
reuse her REGIONS table for coordinate-to-country lookup and her biome-keyed
TEMPLATES for trip-tier metadata (best_season, budget_per_day, tips) —
those are climate facts, not location-specific recommendations. The actual
itinerary now names real restaurants / museums / hotels pulled from the
same Google Places API the DL ReAct agent uses.

Scoring formula per POI (rule-based, no learning):
    score = rating × log(reviews + 1) × style_bonus
where style_bonus = 1.5 if the POI's Google `type` matches one of the
user's selected travel styles, else 1.0.
"""
from __future__ import annotations

import io
import math
from datetime import date
from typing import Optional

import numpy as np
from PIL import Image

from tools.places import search_places


# ── Region database (used by perception _lookup_region and the DL agent's
# geo-fallback too — don't remove without updating dl_agent.py). ───────────
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


# ── Biome-keyed metadata. We only keep best_season, which is a climate
# fact grounded in the biome label (tropical monsoon windows, alpine
# trekking vs. skiing windows, Mediterranean shoulder seasons, etc.).
#
# The per-biome "biome_tips" were dropped — they were generic strings
# keyed on coarse biomes ("arid" → "cover up for cultural respect") and
# fired on destinations where they made no sense (LA is in the "USA
# Southwest" biome=arid bbox, but cultural-modesty dress advice and
# "extreme heat" framing are simply wrong for Los Angeles). If we can't
# stand behind a tip destination-by-destination, we don't print it. Same
# policy we applied earlier to the made-up "$80-$200 USD" daily budgets.
#
# The per-biome "budget_per_day" was removed even earlier for the same
# reason — we echo the user's own budget if they typed one, else nothing.
TEMPLATES = {
    "tropical":      {"best_season": "November – April"},
    "mediterranean": {"best_season": "May – June, September – October"},
    "alpine":        {"best_season": "June – September (trekking), Dec – March (skiing)"},
    "arid":          {"best_season": "October – March"},
    "temperate":     {"best_season": "April – June, September – October"},
    "subarctic":     {"best_season": "Jun–Aug (midnight sun), Sep–Mar (Northern Lights)"},
    "island":        {"best_season": "Varies by island — check local weather patterns"},
    "savanna":       {"best_season": "July – October"},
    "rainforest":    {"best_season": "June – November"},
    "mountain":      {"best_season": "March – May, September – November"},
    "ocean":         {"best_season": "Varies by ocean basin"},
    "default":       {"best_season": ""},
}

# Generic travel-logistics tips — only appended when the trip actually
# crosses a border. Visa / insurance / language advice is universally
# true for international travel and wrong for domestic, which is exactly
# the signal we key on. A same-country trip gets no tips at all.
INTERNATIONAL_TIPS = [
    "Research visa / entry requirements for the destination country.",
    "Consider travel insurance that covers medical care abroad.",
    "Learn a few local phrases — it always helps.",
]

# Theme for each day — kept from Tiffany, purely cosmetic labelling.
DAY_STRUCTURES = [
    {"label": "Day 1", "theme": "Arrival & Orientation"},
    {"label": "Day 2", "theme": "Signature Experience"},
    {"label": "Day 3", "theme": "Cultural Deep Dive"},
    {"label": "Day 4", "theme": "Adventure Day"},
    {"label": "Day 5", "theme": "Hidden Gems & Leisure"},
    {"label": "Day 6", "theme": "Local Immersion"},
    {"label": "Day 7", "theme": "Farewell & Reflection"},
]


# ── Rule-based planner configuration ───────────────────────────────────────

# POI searches we issue to Google Places for every trip. The first field is
# the search query (free text), the second is the Places `includedType`.
POI_CATEGORIES = [
    ("tourist attractions",  "tourist_attraction"),
    ("museums",              "museum"),
    ("local restaurants",    "restaurant"),
    ("cafes",                "cafe"),
    ("hotels",               "lodging"),
    ("bars",                 "bar"),
    ("nightlife",            "night_club"),
    ("parks",                "park"),
]

# travel_style → Google Places types that get a 1.5× scoring boost.
STYLE_BONUS = {
    "foodie":     {"restaurant", "cafe"},
    "culture":    {"museum", "tourist_attraction"},
    "adventure":  {"park", "amusement_park", "tourist_attraction"},
    "nightlife":  {"bar", "night_club"},
    "relaxation": {"spa", "park"},
    "nature":     {"park", "tourist_attraction"},
    "shopping":   {"shopping_mall"},
}

# Which POI types are eligible for which day-slot. `EVENING_TYPES` includes
# restaurant as a fallback if the city has no bars / clubs in Places data.
ATTRACTION_TYPES = {"tourist_attraction", "museum", "park", "amusement_park"}
FOOD_TYPES       = {"restaurant", "cafe"}
EVENING_TYPES    = {"bar", "night_club", "restaurant"}


def _days_between(a: str, b: str) -> int:
    """Inclusive day count between two ISO dates. Falls back to 3 if invalid."""
    try:
        d1 = date.fromisoformat(a)
        d2 = date.fromisoformat(b)
        return max((d2 - d1).days + 1, 1)
    except Exception:
        return 3


# Common country-name aliases so "USA" and "United States" compare equal.
_COUNTRY_ALIASES = {
    "usa": "united states",
    "us": "united states",
    "u.s.a.": "united states",
    "u.s.": "united states",
    "uk": "united kingdom",
    "britain": "united kingdom",
    "great britain": "united kingdom",
}


def _country_of(name: str) -> str:
    """Pull a country tag off the end of a reverse-geocoded name like
    "Asheville, North Carolina, United States" → 'united states'. Tolerates
    both spellings and a few common aliases."""
    if not name:
        return ""
    last = name.split(",")[-1].strip().lower()
    return _COUNTRY_ALIASES.get(last, last)


def _is_cross_border(trip: dict) -> bool:
    """True when origin and destination are in different countries (or we
    can't tell for one side). Used to decide whether to include visa /
    insurance / language tips — nonsense on a same-country trip."""
    origin = _country_of(trip.get("origin_name", ""))
    dest = _country_of(trip.get("destination_name", ""))
    if not origin or not dest:
        return False  # missing data → assume domestic, skip the generic tips
    return origin != dest


def _style_bonus(place_type: str, styles: list) -> float:
    """1.5 if any of the user's styles claims this POI type, else 1.0."""
    for s in styles or []:
        if place_type in STYLE_BONUS.get(s, set()):
            return 1.5
    return 1.0


def _score_poi(poi: dict, place_type: str, styles: list) -> float:
    rating = float(poi.get("rating") or 0)
    reviews = int(poi.get("num_reviews") or 0)
    return rating * math.log(reviews + 1) * _style_bonus(place_type, styles)


def _render_poi_line(poi: Optional[dict], fallback: str) -> str:
    """Turn one POI into a markdown fragment for the itinerary. Picks a verb
    appropriate to the Google Places type (restaurant → 'Enjoy local cuisine
    at', museum → 'Visit', bar → 'Experience the nightlife at', etc.)."""
    if not poi:
        return fallback
    name = poi.get("name") or "Unknown place"
    url = poi.get("google_maps_url") or ""
    rating = poi.get("rating")
    reviews = poi.get("num_reviews") or 0
    place_type = poi.get("place_type", "")

    if place_type == "tourist_attraction":
        verb = "Explore"
    elif place_type == "museum":
        verb = "Visit"
    elif place_type == "restaurant":
        verb = "Enjoy local cuisine at"
    elif place_type == "cafe":
        verb = "Grab coffee at"
    elif place_type in ("bar", "night_club"):
        verb = "Experience the nightlife at"
    elif place_type == "park":
        verb = "Relax at"
    elif place_type == "amusement_park":
        verb = "Have fun at"
    else:
        verb = "Visit"

    link = f"[{name}]({url})" if url else f"**{name}**"
    tail_bits = []
    if rating:
        tail_bits.append(f"{rating}⭐")
    if reviews:
        tail_bits.append(f"{reviews} reviews")
    tail = f" ({', '.join(tail_bits)})" if tail_bits else ""
    return f"{verb} {link}{tail}"


class NonDLAgent:
    """Rule-based itinerary planner. Same output contract as DLAgent."""

    # ── PERCEPTION (unchanged) ────────────────────────────────────────────
    def perceive(self, lat, lng, img_bytes=None) -> dict:
        region_info = self._lookup_region(lat, lng)
        terrain_hint = self._analyze_image(img_bytes) if img_bytes else None
        if terrain_hint and region_info["biome"] == "diverse":
            region_info["biome"] = terrain_hint
        region_info["perception_method"] = "Color histogram + coordinate lookup"
        return region_info

    def _lookup_region(self, lat, lng) -> dict:
        if lat is None or lng is None:
            return {"name": "Unknown Region", "climate": "temperate",
                    "biome": "default", "emoji": "🌍"}
        for row in REGIONS:
            name, lat_min, lat_max, lng_min, lng_max, climate, biome, emoji = row
            if lat_min <= lat <= lat_max and lng_min <= lng <= lng_max:
                return {"name": name, "climate": climate,
                        "biome": biome, "emoji": emoji}
        return {"name": "Unknown Region", "climate": "temperate",
                "biome": "default", "emoji": "🌍"}

    def _analyze_image(self, img_bytes) -> Optional[str]:
        """Classical-CV terrain hint from dominant colour of a small crop."""
        try:
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB").resize((64, 64))
            arr = np.array(img).reshape(-1, 3).astype(float)
            r, g, b = arr.mean(axis=0)
            if b > r + 20 and b > g + 10:
                return "ocean"
            if g > r + 15 and g > b + 10:
                return "rainforest"
            if r > 160 and g > 120 and b < 80:
                return "arid"
            if r > 200 and g > 200 and b > 200:
                return "alpine"
            return None
        except Exception:
            return None

    # ── PLANNING (rule-based via Google Places) ────────────────────────────
    def plan_rule_based(self, region_info: dict, trip: dict) -> dict:
        destination = (trip.get("destination_name")
                       or region_info.get("name")
                       or "the destination")
        styles = trip.get("travel_style") or []
        duration = _days_between(trip.get("start_date"), trip.get("end_date"))

        # Step 1: Fetch POI candidates from Google Places
        pois_by_type: dict = {}
        for query, place_type in POI_CATEGORIES:
            try:
                result = search_places(query=query, location=destination,
                                       type=place_type, max_results=10)
                pois_by_type[place_type] = result.get("places", [])
            except Exception as e:
                print(f"[Non-DL] search_places({place_type}) failed: {e}")
                pois_by_type[place_type] = []

        # Step 2: Score every POI — purely multiplicative formula, no ML.
        scored: list = []
        for place_type, places in pois_by_type.items():
            for p in places:
                scored.append({
                    **p,
                    "place_type": place_type,
                    "score": _score_poi(p, place_type, styles),
                })
        scored.sort(key=lambda p: -p["score"])

        # Step 3: Fill day-slots by descending score, no POI repeats.
        used_names: set = set()

        def take(eligible: set) -> Optional[dict]:
            for cand in scored:
                if cand["name"] in used_names:
                    continue
                if cand["place_type"] in eligible:
                    used_names.add(cand["name"])
                    return cand
            return None

        # Look up biome metadata (best_season + biome_tips). Fall back to the
        # region's climate when the biome label doesn't have its own entry
        # (e.g. REGIONS["France"] has biome="mixed" which isn't keyed in
        # TEMPLATES, but its climate "temperate" is — and "temperate"
        # best-season advice is what we actually want for a Paris trip).
        biome = region_info.get("biome") or ""
        climate = region_info.get("climate") or ""
        meta = (TEMPLATES.get(biome)
                or TEMPLATES.get(climate)
                or TEMPLATES["default"])

        # Tips: only the international-travel trio, and only for cross-border
        # trips. Biome-keyed tips were dropped — they were generic chapter-
        # header advice that often contradicted the real destination (LA
        # getting "cover up for cultural respect" from the arid biome, etc.)
        tips_pool: list = (
            list(INTERNATIONAL_TIPS) if _is_cross_border(trip) else []
        )

        itinerary: list = []
        for i in range(min(max(duration, 1), 14)):
            theme = DAY_STRUCTURES[i % len(DAY_STRUCTURES)]["theme"]
            morning   = take(ATTRACTION_TYPES)
            lunch     = take(FOOD_TYPES)
            afternoon = take(ATTRACTION_TYPES)
            evening   = take(EVENING_TYPES)
            # Rotate through the tip pool; skip the daily tip entirely if
            # there are no tips to show (e.g. domestic trip in an unknown-
            # biome region like rural NC).
            daily_tip = tips_pool[i % len(tips_pool)] if tips_pool else ""
            itinerary.append({
                "day":       f"Day {i + 1}",
                "theme":     theme,
                "morning":   _render_poi_line(morning,   "(no morning attraction available)"),
                "lunch":     _render_poi_line(lunch,     "(no lunch spot available)"),
                "afternoon": _render_poi_line(afternoon, "(no afternoon attraction available)"),
                "evening":   _render_poi_line(evening,   "(no evening venue available)"),
                "tip":       daily_tip,
            })

        # Step 4: Render markdown that mirrors the DL agent's output shape.
        hotels = sorted(pois_by_type.get("lodging", []),
                        key=lambda p: -_score_poi(p, "lodging", styles))[:3]
        markdown = self._render_markdown(destination, trip, duration,
                                         meta, itinerary, hotels, tips_pool)

        return {
            "markdown":       markdown,
            "iterations":     0,
            "num_tool_calls": sum(1 for _, _ in POI_CATEGORIES),
            "tools_used":     ["search_places"] * len(POI_CATEGORIES),
            "itinerary":      itinerary,
            "food":           [p["name"] for p in pois_by_type.get("restaurant", [])[:5]],
            "tips":           tips_pool,  # biome tips + (intl tips iff cross-border)
            # dict.fromkeys preserves insertion order while deduping — Google
            # Places returns some POIs (e.g. the Louvre) under both
            # tourist_attraction AND museum, so they show up twice in `scored`.
            "highlights":     list(dict.fromkeys(p["name"] for p in scored))[:5],
            "best_season":    meta["best_season"],
            # Budget echoes the user's own input (if any) — the previous
            # biome-keyed "$X-$Y USD" numbers were made-up.
            "budget":         trip.get("budget") or "",
        }

    @staticmethod
    def _render_markdown(destination: str, trip: dict, duration: int,
                         meta: dict, itinerary: list, hotels: list,
                         tips: list) -> str:
        styles = trip.get("travel_style") or []
        style_str = " + ".join(s.capitalize() for s in styles) if styles else ""
        group = trip.get("group_type", "")
        start = trip.get("start_date", "")
        end = trip.get("end_date", "")
        num_people = trip.get("num_people", "")
        origin = trip.get("origin_name", "")
        user_budget = trip.get("budget") or ""

        lines: list = []

        lines.append("### Trip Overview")
        bits = [f"**{destination}**"]
        if start and end:
            bits.append(f"**{start} → {end}** ({duration} days)")
        if num_people:
            bits.append(f"**{num_people} traveler{'s' if num_people != 1 else ''}**")
        if group:
            bits.append(f"**{group}** trip")
        if style_str:
            bits.append(f"**{style_str}** focus")
        lines.append(" · ".join(bits))
        if origin and origin != "N/A":
            lines.append("")
            lines.append(f"_Departing from {origin}._")
        lines.append("")
        lines.append(
            "_Plan generated by the rule-based non-DL agent — POIs pulled live "
            "from Google Places, ranked by rating × log(reviews) with a style "
            "bonus, filled into fixed day-slots. No LLM involved._"
        )
        lines.append("")

        if hotels:
            lines.append("### Accommodation")
            for h in hotels:
                name = h.get("name", "?")
                url = h.get("google_maps_url", "")
                rating = h.get("rating") or "?"
                reviews = h.get("num_reviews") or 0
                link = f"[{name}]({url})" if url else f"**{name}**"
                lines.append(f"- {link} — {rating}⭐ ({reviews} reviews)")
            lines.append("")

        lines.append("### Day-by-Day Itinerary")
        lines.append("")
        for d in itinerary:
            lines.append(f"**{d['day']}: {d['theme']}**")
            lines.append(f"- **Morning:** {d['morning']}")
            lines.append(f"- **Lunch:** {d['lunch']}")
            lines.append(f"- **Afternoon:** {d['afternoon']}")
            lines.append(f"- **Evening:** {d['evening']}")
            if d.get("tip"):
                lines.append(f"> *Tip: {d['tip']}*")
            lines.append("")

        # Practical Info — render only fields we can stand behind:
        #   * best_season from the biome template (real climate fact), or
        #     nothing if the destination didn't match any known biome
        #   * user-provided total budget echoed back, if present
        # When both are empty, the whole section is dropped.
        best_season = meta.get("best_season") or ""
        if best_season or user_budget:
            lines.append("### Practical Info")
            if best_season:
                lines.append(f"- **Best season:** {best_season}")
            if user_budget:
                lines.append(f"- **Your total budget:** ${user_budget} USD")
            lines.append("")

        # Only emit the Travel Tips section if we actually have tips. Tips
        # are empty when: no biome match (e.g. unknown rural region) AND
        # it's a domestic trip (so no international-logistics tips apply).
        if tips:
            lines.append("### Travel Tips")
            for tip in tips:
                lines.append(f"- {tip}")

        return "\n".join(lines)

    # ── CONTROL — new trip-dict entrypoint (shape matches DLAgent.run_trip) ──
    def run_trip(self, trip: dict, img_bytes=None) -> dict:
        region_info = self.perceive(
            trip.get("destination_lat"),
            trip.get("destination_lng"),
            img_bytes,
        )
        # Prefer the user-confirmed (reverse-geocoded) city name over the
        # coarse lat/lng-bbox country lookup. That's what Google Places uses
        # for its search query.
        if trip.get("destination_name"):
            region_info["name"] = trip["destination_name"]

        try:
            plan_data = self.plan_rule_based(region_info, trip)
        except Exception as e:
            print(f"[Non-DL run_trip error] {e}")
            plan_data = {
                "markdown": f"### Planning failed\n\nRule-based planner errored: `{e}`",
                "iterations": 0, "num_tool_calls": 0, "tools_used": [],
                "itinerary": [], "food": [], "tips": [],
                "highlights": [], "best_season": "", "budget": "",
            }

        return {
            "agent": "non_dl",
            "region": region_info,
            "plan": plan_data,
            "conversation_history": [],
            "perception_method": region_info.get(
                "perception_method", "Color histogram + coordinate lookup"
            ),
            "trip": trip,
        }

    # ── CONTROL — legacy run() kept so evaluation/evaluator.py still works ───
    def run(self, lat, lng, img_bytes=None, preferences=None):
        """Compatibility shim: builds a minimal trip dict from legacy args and
        forwards to run_trip(). Used by `evaluation/evaluator.py`, which
        tests each of 12 geographic coordinates without a full form state."""
        preferences = preferences or {}
        style = preferences.get("style")
        trip = {
            "destination_lat":  lat,
            "destination_lng":  lng,
            "destination_name": "",
            "origin_name":      "N/A",
            "start_date":       "",
            "end_date":         "",
            "num_people":       preferences.get("num_people", 2),
            "group_type":       preferences.get("group_type", "solo"),
            "transport":        "flight",
            "travel_style":     [style] if style and style != "balanced" else [],
            "budget":           None,
            "notes":            "",
        }
        return self.run_trip(trip, img_bytes=img_bytes)
