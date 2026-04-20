from flask import Flask, render_template, request, jsonify
import base64, io, json, os
from datetime import date
from dotenv import load_dotenv
load_dotenv()
from agents.non_dl_agent import NonDLAgent
from agents.dl_agent import DLAgent

app = Flask(__name__)

non_dl_agent = NonDLAgent()
dl_agent = DLAgent()

# Lazy-init gesture detector — loading the MediaPipe model adds ~1s to startup,
# and we don't want that cost if the user is only using the map flow.
_gesture_detector = None


def _get_gesture_detector():
    global _gesture_detector
    if _gesture_detector is None:
        from perception.mediapipe_hands import MediaPipeHandDetector
        _gesture_detector = MediaPipeHandDetector()
    return _gesture_detector


@app.route('/')
def index():
    return render_template('index.html')


def _days_between(a: str, b: str) -> int:
    """Inclusive day count between two ISO dates. Falls back to 5 if invalid."""
    try:
        d1 = date.fromisoformat(a)
        d2 = date.fromisoformat(b)
        return max((d2 - d1).days + 1, 1)
    except Exception:
        return 5


# ── /plan (yifei) ──────────────────────────────────────────────────────────
# New payload shape:
#   {
#     "agent": "dl" | "non_dl",
#     "origin":      {"name": ..., "lat": ..., "lng": ...},
#     "destination": {"name": ..., "lat": ..., "lng": ...},
#     "start_date": "YYYY-MM-DD", "end_date": "YYYY-MM-DD",
#     "num_people": int, "group_type": str, "transport": str,
#     "travel_style": [str, ...], "budget": str|None, "notes": str
#   }
# Legacy payload (top-level lat/lng) is accepted for back-compat.
@app.route('/plan', methods=['POST'])
def plan():
    data = request.json or {}
    agent_type = data.get('agent', 'dl')

    origin = data.get('origin') or {}
    destination = data.get('destination') or {}

    # Back-compat: old shape sent only top-level lat/lng
    if not destination.get('lat') and data.get('lat') is not None:
        destination = {
            'name': f"{data.get('lat'):.2f}, {data.get('lng'):.2f}",
            'lat': data.get('lat'),
            'lng': data.get('lng'),
        }

    trip = {
        'origin_name':      origin.get('name') or 'N/A',
        'origin_lat':       origin.get('lat'),
        'origin_lng':       origin.get('lng'),
        'destination_name': destination.get('name') or 'Unknown',
        'destination_lat':  destination.get('lat'),
        'destination_lng':  destination.get('lng'),
        'start_date':       data.get('start_date', ''),
        'end_date':         data.get('end_date', ''),
        'num_people':       data.get('num_people', 2),
        'group_type':       data.get('group_type', 'solo'),
        'transport':        data.get('transport', 'flight'),
        'travel_style':     data.get('travel_style', []) or [],
        'budget':           data.get('budget'),
        'notes':            data.get('notes', ''),
    }

    map_image_b64 = data.get('map_image')
    img_bytes = base64.b64decode(map_image_b64.split(',')[-1]) if map_image_b64 else None

    if agent_type == 'non_dl':
        preferences = {
            'duration': _days_between(trip['start_date'], trip['end_date']),
            'style': (trip['travel_style'][0] if trip['travel_style'] else 'balanced'),
            'budget': 'mid-range',
        }
        result = non_dl_agent.run(trip['destination_lat'], trip['destination_lng'], img_bytes, preferences)
        result['trip'] = trip
    else:
        result = dl_agent.run_trip(trip, img_bytes=img_bytes)

    return jsonify(result)


# ── /plan (Tiffany, legacy) — kept for reference. ──────────────────────────
# @app.route('/plan', methods=['POST'])
# def plan_legacy():
#     data = request.json
#     lat = data.get('lat')
#     lng = data.get('lng')
#     map_image_b64 = data.get('map_image')
#     agent_type = data.get('agent', 'dl')
#     preferences = data.get('preferences', {})
#
#     img_bytes = base64.b64decode(map_image_b64.split(',')[-1]) if map_image_b64 else None
#
#     if agent_type == 'non_dl':
#         result = non_dl_agent.run(lat, lng, img_bytes, preferences)
#     else:
#         result = dl_agent.run(lat, lng, img_bytes, preferences)
#
#     return jsonify(result)


@app.route('/evaluate', methods=['POST'])
def evaluate():
    from evaluation.evaluator import run_evaluation
    results = run_evaluation()
    return jsonify(results)


# ── /gesture (yifei) ───────────────────────────────────────────────────────
# Accepts a single camera frame and returns the detected gesture + landmarks.
# Using POST (not WebSocket) for simplicity — client throttles to 5-10 fps.
# Payload: {"frame": "data:image/jpeg;base64,..."} or {"frame": "<base64>"}
# Response: {"gesture": "open_palm"|"pinch"|"point"|"fist"|"none",
#            "confidence": 0..1, "landmarks": [[x,y,z]..21], "handedness": "Left"|"Right"|null}
@app.route('/gesture', methods=['POST'])
def gesture():
    data = request.json or {}
    frame_b64 = data.get('frame')
    if not frame_b64:
        return jsonify({"gesture": "none", "confidence": 0.0,
                        "landmarks": [], "handedness": None,
                        "error": "missing 'frame' in payload"}), 400
    try:
        # Strip data URL prefix if present
        if ',' in frame_b64:
            frame_b64 = frame_b64.split(',', 1)[1]
        img_bytes = base64.b64decode(frame_b64)
    except Exception as e:
        return jsonify({"gesture": "none", "confidence": 0.0,
                        "landmarks": [], "handedness": None,
                        "error": f"bad_base64: {e}"}), 400

    try:
        detector = _get_gesture_detector()
        result = detector.detect(img_bytes)
    except Exception as e:
        return jsonify({"gesture": "none", "confidence": 0.0,
                        "landmarks": [], "handedness": None,
                        "error": f"detect_failed: {e}"}), 500
    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True, port=5000)
