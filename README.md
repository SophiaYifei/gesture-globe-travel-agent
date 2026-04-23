# GestureGlobe — A Gesture-Controlled Travel Planning Agent
### Project 2 · Perception · Planning · Control

An interactive vacation planner driven by **hand gestures over a 3D globe**. Point a webcam at yourself, pinch to rotate the Earth, dwell to drop a pin, and let the agent build you a real day-by-day itinerary with actual restaurants, hotels, flights, and vacation rentals.

The system is implemented twice — a **DL track** (MediaPipe hand landmarks + LLM ReAct agent) and a **Non-DL track** (classical OpenCV skin segmentation + rule-based planner) — and you can flip between them at runtime from the UI to compare.

---

## 🧭 What the three stages are

| Stage | What it means here |
|-------|-------------------|
| **Perception** | Reading the user's hand from a webcam frame — classify the gesture and extract the fingertip landmark |
| **Planning** | Turning `(origin, destination, dates, preferences)` into a day-by-day itinerary grounded in real POIs |
| **Control** | The 3D Cesium globe + gesture state machine that lets the user express intent (pick origin → confirm → pick destination → confirm → fill form → plan) |

---

## 📁 Project Structure

```
gesture-globe-travel-agent/
├── app.py                      # Flask server — routes /plan and /gesture
├── config.py                   # OpenRouter + Google Places keys, model names
├── prompts.py                  # ReAct system prompt for the DL planner
├── requirements.txt
├── .env.example                # API-key template
│
├── perception/                 # Hand-gesture recognition (dual track)
│   ├── mediapipe_hands.py      # DL: MediaPipe HandLandmarker (Tasks API)
│   └── opencv_hands.py         # Non-DL: HSV+YCrCb skin mask + contour analysis
│
├── agents/                     # Travel planner (dual track)
│   ├── dl_agent.py             # DL: ReAct agent over OpenRouter + tool calls
│   ├── non_dl_agent.py         # Non-DL: Google Places POI scoring + slot filler
│   └── react_agent.py          # ReAct loop (shared by DL track)
│
├── tools/                      # Tool functions exposed to the ReAct agent
│   ├── places.py               # Google Places (New) Text Search
│   ├── weather.py              # Open-Meteo forecast
│   ├── flights.py              # Google Flights deep-link builder
│   └── vacation_rentals.py     # Airbnb/Vrbo/Booking search-link builder
│
├── evaluation/
│   ├── evaluator.py            # Side-by-side DL vs Non-DL with LLM-as-judge
│   ├── react_test_cases.json   # Test cases for the planner
│   └── eval_results.json       # Last run's output
│
└── templates/
    └── index.html              # Cesium globe + gesture HUD + trip form + itinerary view
```

---

## 🚀 Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

The big ones: **Flask**, **OpenAI SDK** (used as the OpenRouter client), **MediaPipe**, **OpenCV**, **Pillow**, **numpy**.

### 2. API keys

Copy `.env.example` → `.env` and fill in:

```env
OPENROUTER_API_KEY=sk-or-v1-...   # required for the DL planner + eval judge
GOOGLE_PLACES_API_KEY=AIza...     # required for real POI data (both planners)
```

Both tracks of the **planner** call Google Places for real POIs; only the **DL** track calls OpenRouter. The Non-DL track runs without any LLM key.

### 3. (DL-perception only) MediaPipe model

The first time you run with DL gestures, `perception/mediapipe_hands.py` will download `hand_landmarker.task` (~7 MB) from Google's MediaPipe CDN into the module folder. No action needed.

### 4. Run

```bash
python app.py
```

Open **http://localhost:5000**. Allow camera access when the browser prompts.

---

## 🎮 How the UI works

The front-end is a **Cesium 3D globe** (OpenStreetMap tiles, `requestRenderMode` on) plus a floating HUD that shows the live webcam feed and the detected gesture.

### Gesture vocabulary (final design)

| Gesture | Icon | Meaning |
|---------|------|---------|
| Pinch | 🤏 | Drag to rotate the globe (sensitivity scales with zoom altitude) |
| Open palm | 🖐️ | Zoom — `palm_area` maps exponentially to globe altitude |
| Point | ☝️ | Project a laser pin onto the globe; hold still for 2 s to select |
| Peace | ✌️ | Confirm the current origin / destination |
| Fist | ✊ | Lock view (subsequent open-palm needs a 2 s hold to unlock) |

Mouse fallback is present throughout — you don't need a camera to use the app.

### State machine

```
PICK_ORIGIN  →  CONFIRM_ORIGIN  →  PICK_DEST  →  CONFIRM_DEST  →  FORM  →  PLANNING  →  RESULT
```

Each step has its own HUD prompt. Reverse-geocoding uses Nominatim with an Overpass fallback, preferring *city* over *town* when within a 2.5× distance ratio.

### Toggles

- **Perception backend** — DL (MediaPipe) ↔ Non-DL (OpenCV), switches at runtime; Flask lazy-inits detectors per mode.
- **Planner backend** — DL (ReAct + LLM) ↔ Non-DL (rule-based), passed as `agent` in the `/plan` POST body.

---

## 🧠 Architecture

### Perception — hand gesture recognition

Both detectors expose the same `detect(img_bytes) → {gesture, confidence, landmarks, handedness, bbox, palm_area, index_tip}` interface, so the frontend doesn't care which is running.

**DL track — `perception/mediapipe_hands.py`**
- MediaPipe **HandLandmarker** via the Tasks API (the `mp.solutions.hands` API was removed in 0.10.22+).
- 21 3D landmarks per hand at ~30 fps on-device.
- Gesture classifier is a small rule layer on top of landmark geometry (finger-extension checks + angle/distance thresholds) — there's no separate gesture model. The *landmark* model is the learned component.

**Non-DL track — `perception/opencv_hands.py`**
- Skin segmentation: **HSV ∩ YCrCb** masks, morphological open/close, Haar frontal-face cascade to subtract the face from the skin region.
- Largest remaining contour is the hand. Features extracted: convex hull, convexity defects, solidity, `top_width_ratio`, aspect ratio, inner-contour holes (pinch detection via `RETR_CCOMP`).
- Same 5-gesture taxonomy, decided by feature thresholds. Pinch uses inner-hole area; peace uses `aspect > 1.35 AND defects ≤ 2 AND 0.35 < top_width_ratio < 0.72`.

### Planning — itinerary generation

Both planners take the same trip dict (`origin`, `destination`, `start_date`, `end_date`, `num_people`, `group_type`, `transport`, `travel_style`, `budget`, `notes`) and return the same shape: `{markdown, itinerary, meta}`.

**DL track — `agents/dl_agent.py` + `agents/react_agent.py`**
- Runs a **ReAct** loop via the OpenAI SDK pointed at **OpenRouter**.
- Model: `openai/gpt-5.4-nano` (agent), `openai/gpt-5.4` (eval judge — stronger than the agent to reduce self-bias).
- Four tools: `search_places` (Google Places), `get_weather` (Open-Meteo), `search_flights` (Google Flights link), `find_vacation_rentals` (Google → Airbnb/Vrbo/Booking link).
- The model plans tool use itself — no hard-coded schedule of calls.

**Non-DL track — `agents/non_dl_agent.py`**
- **No LLM involved.** Deterministic scoring + slot filling.
- Issues ~8 Google Places searches per trip (attractions, restaurants, cafés, hotels, etc.) at the destination.
- Ranks each POI by `rating × log(reviews + 1) × style_bonus` (1.5× if the POI type matches one of the user's selected travel styles).
- Fills four day-slots per day (morning / lunch / afternoon / evening) from the ranked pool.
- "Practical info" is strictly what we can back up: user's own budget echoed back, `best_season` from a biome table (climate facts — tropical monsoon windows, Mediterranean shoulder seasons, etc.), and international-travel tips **only** on cross-border trips. No fabricated daily-budget ranges, no generic biome tips.

### Control — the Cesium globe

- **CesiumJS 1.122** with `UrlTemplateImageryProvider` for OSM tiles (synchronous init avoids the black-globe flicker from `fromProviderAsync`).
- `requestRenderMode: true` — the globe only redraws when state changes, keeping CPU low while the gesture loop runs at 10 fps.
- All floating UI elements use `position: fixed` so they stay viewport-locked during globe drags.
- Pinch rotation: `BASE × (altitude/MAX)^0.75` with a 1.2 px accumulated deadband.
- Open-palm zoom: `altitude = MIN × (MAX/MIN)^(1−t)` with `palm_area → t ∈ [0.05, 0.55]`.

---

## 📊 Evaluation

`evaluation/evaluator.py` runs both agents over a battery of test cases and scores them on:

1. **Perception accuracy** — did the agent identify the region + biome correctly?
2. **Itinerary relevance** — keyword match against known-good activities *and* an LLM-as-judge score.
3. **Notes adherence** — if the test case specifies "wheelchair accessible" or "vegetarian", does the itinerary reflect it?
4. **Response time** (ms) and **estimated cost extraction**.
5. **Task completion** — did the agent return a non-empty, well-formed plan?

Run from the UI (bottom-left button) or directly:

```bash
python -c "from evaluation.evaluator import run_evaluation; import json; print(json.dumps(run_evaluation(), indent=2))"
```

Results are written to `evaluation/eval_results.json`.

---

## 🔌 HTTP API

### `POST /gesture`
```json
{"frame": "<dataURL or raw base64>", "mode": "dl" | "non_dl"}
```
Returns `{gesture, confidence, landmarks, handedness, bbox, palm_area, index_tip, mode}`. Client throttles to 10 fps.

### `POST /plan`
```json
{
  "agent": "dl" | "non_dl",
  "origin":      {"name": "Durham, NC", "lat": 35.99, "lng": -78.90},
  "destination": {"name": "Paris, France", "lat": 48.85, "lng": 2.35},
  "start_date": "2026-05-10",
  "end_date":   "2026-05-15",
  "num_people": 2,
  "group_type": "couple",
  "transport":  "flight",
  "travel_style": ["food", "culture"],
  "budget": "2000",
  "notes": "vegetarian"
}
```
Returns `{markdown, itinerary, meta}` — the frontend renders either agent's response through the same view.

### `POST /evaluate`
Runs the full evaluation suite. Returns the scored comparison JSON.

---

## 🛠 Tech stack

- **Backend**: Flask (`threaded=True` so the 10 fps gesture loop doesn't block `/plan`)
- **Perception — DL**: MediaPipe 0.10.33+ (HandLandmarker Tasks API)
- **Perception — Non-DL**: OpenCV (HSV + YCrCb skin mask, Haar face cascade, contour analysis)
- **Planning — DL**: OpenAI SDK → OpenRouter → `gpt-5.4-nano`, ReAct tool-use loop
- **Planning — Non-DL**: Pure-Python rule-based scorer over Google Places results
- **Tools**: Google Places API (New), Open-Meteo, Google Flights, Airbnb/Vrbo/Booking link builders
- **Frontend**: CesiumJS 1.122, OpenStreetMap tiles, vanilla JS + Nominatim/Overpass reverse-geocode

---

## 📹 Demo tips

- Start zoomed-out; use **pinch** to rotate to a continent, **open palm** to zoom in, **point + 2 s hold** to drop the origin pin, **peace** to confirm.
- Repeat for destination.
- Fill the form, hit "Plan my trip", compare the DL and Non-DL outputs on the same trip.
- Run **Evaluate** at the end to show the side-by-side metrics.
