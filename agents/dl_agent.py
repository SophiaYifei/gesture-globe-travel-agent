"""
DL Agent
Perception : CLIP zero-shot classification of map screenshot (Tiffany)
Planning   : ReAct agent via OpenRouter + OpenAI SDK (Yifei — from Project 1)
Control    : Dynamic itinerary with feedback refinement

Note (yifei branch):
- Tiffany's original Anthropic-SDK planning call is preserved below in
  `_plan_anthropic_legacy` for reference — kept commented-out, NOT deleted.
- Active planner is now `agents.react_agent.run_agent`, which drives a
  tool-using ReAct loop (search_places / get_weather / search_flights /
  find_vacation_rentals) and returns a full markdown itinerary.
"""

import base64, io, json, os, requests
import numpy as np

from agents.react_agent import run_agent as react_run_agent
from prompts import SYSTEM_PROMPT as REACT_SYSTEM_PROMPT

# Kept for reference only — planner no longer uses Anthropic SDK directly.
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

# CLIP candidate labels for zero-shot region classification
CLIP_LABELS = [
    "tropical beach and ocean coastline",
    "snowy alpine mountain peaks",
    "dry desert and sand dunes",
    "dense tropical rainforest",
    "green temperate countryside",
    "arctic tundra and glaciers",
    "mediterranean coastal cliffs",
    "open ocean and sea",
    "savanna grassland with acacia trees",
    "volcanic island landscape",
    "urban city and buildings",
    "river delta and wetlands",
]

LABEL_TO_BIOME = {
    "tropical beach and ocean coastline":   "tropical",
    "snowy alpine mountain peaks":          "alpine",
    "dry desert and sand dunes":            "arid",
    "dense tropical rainforest":            "rainforest",
    "green temperate countryside":          "temperate",
    "arctic tundra and glaciers":           "subarctic",
    "mediterranean coastal cliffs":         "mediterranean",
    "open ocean and sea":                   "ocean",
    "savanna grassland with acacia trees":  "savanna",
    "volcanic island landscape":            "island",
    "urban city and buildings":             "temperate",
    "river delta and wetlands":             "rainforest",
}


class DLAgent:
    """
    Perception  : CLIP zero-shot image classification (torch + transformers)
    Planning    : Anthropic Claude API for rich natural language itinerary
    Control     : Structured output with refinement loop support
    """

    def __init__(self):
        self.clip_model = None
        self.clip_processor = None
        self._clip_loaded = False

    def _load_clip(self):
        if self._clip_loaded:
            return self.clip_model is not None
        self._clip_loaded = True
        try:
            from transformers import CLIPProcessor, CLIPModel
            import torch
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self._torch = torch
            return True
        except Exception as e:
            print(f"[DL Agent] CLIP not available ({e}), falling back to coordinate lookup")
            return False

    # ── PERCEPTION ─────────────────────────────────────────────────────────────
    def perceive(self, lat, lng, img_bytes=None):
        clip_result = None
        if img_bytes and self._load_clip():
            clip_result = self._clip_classify(img_bytes)

        geo_result = self._geo_lookup(lat, lng)

        if clip_result:
            # Merge: CLIP biome + geo name
            return {
                "name": geo_result["name"],
                "emoji": geo_result["emoji"],
                "biome": clip_result["biome"],
                "climate": geo_result["climate"],
                "clip_label": clip_result["label"],
                "clip_confidence": clip_result["confidence"],
                "perception_method": "CLIP zero-shot classification",
            }
        else:
            return {**geo_result, "perception_method": "Coordinate lookup (CLIP unavailable)"}

    def _clip_classify(self, img_bytes):
        try:
            from PIL import Image
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            inputs = self.clip_processor(
                text=CLIP_LABELS, images=img, return_tensors="pt", padding=True
            )
            with self._torch.no_grad():
                outputs = self.clip_model(**inputs)
            probs = outputs.logits_per_image.softmax(dim=1)[0].numpy()
            best_idx = int(np.argmax(probs))
            best_label = CLIP_LABELS[best_idx]
            return {
                "label": best_label,
                "biome": LABEL_TO_BIOME[best_label],
                "confidence": float(probs[best_idx]),
                "all_scores": {CLIP_LABELS[i]: float(probs[i]) for i in range(len(CLIP_LABELS))},
            }
        except Exception as e:
            print(f"[CLIP classify error] {e}")
            return None

    def _geo_lookup(self, lat, lng):
        """Reuse the coordinate lookup from non-DL agent as fallback/supplement."""
        from agents.non_dl_agent import REGIONS
        for row in REGIONS:
            name, lat_min, lat_max, lng_min, lng_max, climate, biome, emoji = row
            if lat_min <= lat <= lat_max and lng_min <= lng <= lng_max:
                return {"name": name, "climate": climate, "biome": biome, "emoji": emoji}
        return {"name": "Unknown Region", "climate": "temperate", "biome": "default", "emoji": "🌍"}

    # ── PLANNING ───────────────────────────────────────────────────────────────
    def plan(self, region_info, preferences=None, conversation_history=None):
        """Drive the ReAct agent (Project 1) to produce a real, tool-backed itinerary.

        The ReAct agent calls search_places / get_weather / search_flights /
        find_vacation_rentals as needed and returns a markdown itinerary with
        live Google Maps links, rated places, and real weather. We keep the
        top-level response shape backward-compatible with Tiffany's frontend
        (region + plan + conversation_history) — the full itinerary text is
        returned under plan.markdown; the previous structured keys (itinerary,
        food, tips, ...) are kept as empty lists so the current UI doesn't
        crash. Frontend rendering will be updated in a follow-up.
        """
        preferences = preferences or {}
        duration = preferences.get("duration", 3)
        style = preferences.get("style", "balanced")
        budget_level = preferences.get("budget", "mid-range")

        user_message = (
            f"Plan a trip with the following details:\n"
            f"- Destination: {region_info.get('name', 'Unknown')} "
            f"(biome: {region_info.get('biome', 'n/a')}, climate: {region_info.get('climate', 'n/a')})\n"
            f"- Duration: {duration} days (assume starting ~2 weeks from today)\n"
            f"- Travel style: {style}\n"
            f"- Budget level: {budget_level}\n"
            f"- Transportation: flight unless the destination is clearly a road trip\n"
            f"- Context: Destination was selected by clicking on a 2D world map; "
            f"treat '{region_info.get('name')}' as the top-level region and pick a "
            f"concrete home base city within it."
        )

        try:
            result = react_run_agent(user_message, REACT_SYSTEM_PROMPT)
            markdown = result.get("response", "")
            plan_data = {
                "markdown": markdown,
                "iterations": result.get("iterations", 0),
                "num_tool_calls": len(result.get("tool_calls", [])),
                "tools_used": [c.get("tool") for c in result.get("tool_calls", [])],
                # Kept empty so Tiffany's frontend doesn't crash; UI will be
                # updated to render `markdown` in a follow-up task.
                "itinerary": [],
                "food": [],
                "tips": [],
                "highlights": [],
                "best_season": "",
                "budget": "",
            }
            updated_history = (conversation_history or []) + [
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": markdown},
            ]
            return plan_data, updated_history
        except Exception as e:
            print(f"[DL Agent ReAct planning error] {e}")
            # Graceful fallback to non-DL planning
            from agents.non_dl_agent import NonDLAgent
            fallback = NonDLAgent()
            fb_plan = fallback.plan(region_info, preferences)
            return fb_plan, conversation_history or []

    # ── Legacy planner (Tiffany) — kept for reference, NOT called. ─────────────
    # def _plan_anthropic_legacy(self, region_info, preferences=None, conversation_history=None):
    #     preferences = preferences or {}
    #     duration = preferences.get("duration", 3)
    #     style = preferences.get("style", "balanced")
    #     budget_level = preferences.get("budget", "mid-range")
    #
    #     system_prompt = (
    #         "You are a travel planner. Respond with valid JSON only, no markdown, no backticks. "
    #         "Keep each activity description under 10 words. "
    #         "Format: {\"itinerary\": [{\"day\": \"Day 1\", \"theme\": \"...\", \"morning\": \"...\", "
    #         "\"afternoon\": \"...\", \"evening\": \"...\", \"tip\": \"...\"}], "
    #         "\"food\": [\"dish1\", \"dish2\", \"dish3\"], "
    #         "\"best_season\": \"...\", \"budget\": \"...\", \"tips\": [\"tip1\", \"tip2\"], "
    #         "\"highlights\": [\"h1\", \"h2\"]}"
    #     )
    #
    #     user_message = (
    #         f"Create a {duration}-day {style} itinerary for {region_info['name']} "
    #         f"({region_info['biome']}, {region_info['climate']}). Budget: {budget_level}. "
    #         f"Be specific with local activities and food."
    #     )
    #
    #     messages = (conversation_history or []) + [{"role": "user", "content": user_message}]
    #
    #     try:
    #         response = requests.post(
    #             "https://api.anthropic.com/v1/messages",
    #             headers={
    #                 "x-api-key": ANTHROPIC_API_KEY,
    #                 "anthropic-version": "2023-06-01",
    #                 "content-type": "application/json",
    #             },
    #             json={
    #                 "model": "claude-sonnet-4-6",
    #                 "max_tokens": 2500,
    #                 "system": system_prompt,
    #                 "messages": messages,
    #             },
    #             timeout=60,
    #         )
    #         data = response.json()
    #         raw = data["content"][0]["text"]
    #         plan_data = json.loads(raw)
    #         return plan_data, messages + [{"role": "assistant", "content": raw}]
    #     except Exception as e:
    #         print(f"[DL Agent planning error] {e}")
    #         from agents.non_dl_agent import NonDLAgent
    #         fallback = NonDLAgent()
    #         fb_plan = fallback.plan(region_info, preferences)
    #         return fb_plan, messages

    # ── CONTROL ────────────────────────────────────────────────────────────────
    def run(self, lat, lng, img_bytes=None, preferences=None, conversation_history=None):
        """Legacy entrypoint — simple (lat, lng, preferences) call. Preserved
        so Tiffany's old /plan path and any direct callers still work."""
        region_info = self.perceive(lat, lng, img_bytes)
        plan, updated_history = self.plan(region_info, preferences, conversation_history)
        return {
            "agent": "dl",
            "region": region_info,
            "plan": plan,
            "conversation_history": updated_history,
            "perception_method": region_info.get("perception_method", "CLIP"),
        }

    # ── CONTROL (yifei) — trip-dict entrypoint for the new form flow ───────────
    def run_trip(self, trip: dict, img_bytes=None):
        """Drive a full itinerary from a trip dict (origin, destination, dates,
        preferences). Mirrors the Project 1 Streamlit flow's `_build_user_message`
        format so the ReAct agent sees exactly the shape it was tuned for.
        """
        region_info = self.perceive(trip.get("destination_lat"),
                                    trip.get("destination_lng"),
                                    img_bytes)
        # Prefer the user-confirmed reverse-geocoded name over the biome-DB name —
        # it's more specific (e.g. "Paris, Île-de-France, France" vs "France").
        if trip.get("destination_name"):
            region_info["name"] = trip["destination_name"]

        user_message = self._build_user_message(trip)

        try:
            result = react_run_agent(user_message, REACT_SYSTEM_PROMPT)
            markdown = result.get("response", "")
            plan_data = {
                "markdown": markdown,
                "iterations": result.get("iterations", 0),
                "num_tool_calls": len(result.get("tool_calls", [])),
                "tools_used": [c.get("tool") for c in result.get("tool_calls", [])],
                # Kept empty — frontend now renders `markdown` directly via marked.js.
                "itinerary": [], "food": [], "tips": [], "highlights": [],
                "best_season": "", "budget": "",
            }
        except Exception as e:
            print(f"[DL Agent run_trip error] {e}")
            plan_data = {
                "markdown": f"### Planning failed\n\nThe ReAct agent errored out: `{e}`",
                "iterations": 0,
                "num_tool_calls": 0,
                "tools_used": [],
                "itinerary": [], "food": [], "tips": [], "highlights": [],
                "best_season": "", "budget": "",
            }

        return {
            "agent": "dl",
            "region": region_info,
            "plan": plan_data,
            "conversation_history": [],
            "perception_method": region_info.get("perception_method", "CLIP"),
            "trip": trip,
        }

    @staticmethod
    def _build_user_message(trip: dict) -> str:
        """Mirrors Project 1's main.py `_build_user_message` so the system prompt
        sees the exact field shape it expects."""
        style = trip.get("travel_style") or []
        style_str = ", ".join(style) if style else "general"
        budget = trip.get("budget") or "Flexible - suggest what it would cost"
        return (
            "Plan a trip with the following details:\n"
            f"- Destination: {trip.get('destination_name', 'Unknown')}\n"
            f"- Dates: {trip.get('start_date', '')} to {trip.get('end_date', '')}\n"
            f"- Number of travelers: {trip.get('num_people', 2)}\n"
            f"- Group type: {trip.get('group_type', 'solo')}\n"
            f"- Travel style: {style_str}\n"
            f"- Transportation preference: {trip.get('transport', 'flight')}\n"
            f"- Budget: {budget}\n"
            f"- Origin city (for flights): {trip.get('origin_name', 'N/A')}\n"
            f"- Additional requests: {trip.get('notes') or 'None'}\n"
        )
