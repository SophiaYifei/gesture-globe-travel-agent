"""DL perception — MediaPipe Hand Landmarker + geometric gesture classifier.

Gesture taxonomy (future globe control on the right):
  rotate_left   → spin globe left    (open palm held vertical, fingers left 🫲)
  rotate_right  → spin globe right   (open palm held vertical, fingers right 🫱)
  zoom          → pan zoom           (open palm flat, fingers up 🖐️; palm_area
                                      is the zoom signal — bigger = zoom in)
  point         → aim (laser dot)    (only the index finger is extended)
  fist          → confirm / stop     (no fingers extended)
  none          → no hand detected, or ambiguous pose

Note: "select" is NOT a classifier label — it's a state transition the client
synthesizes when gesture goes from `point` → `fist`. The backend just emits
the two raw states; the frontend pins a green marker at the last `index_tip`
seen during `point` when it sees the edge into `fist`.

Palm orientation is derived from the wrist → MIDDLE_MCP vector after the
frontend mirrors the frame horizontally before POSTing — so image X matches
the user's own left/right, and a 🫲 gesture produces dx<0 (rotate_left) as a
user naturally expects.

For the laser-pointer UI, detect() also returns `index_tip` — the normalized
(x, y) of landmark 8 — but only when the index is the ONLY extended finger
(i.e. the `point` pose). That keeps the red dot off during open_palm / fist.

Uses MediaPipe's Tasks API (`mediapipe.tasks.python.vision.HandLandmarker`)
since the old `mp.solutions.hands` module was removed in mediapipe 0.10.22+.
The detector itself is a DL model (21-point landmark regressor); classification
is pure geometry on top of the landmarks — the "DL" label applies to the
perception stage, which is what the project rubric cares about. A sibling
non_dl_hands.py will reimplement the perception stage with classical CV.
"""
from __future__ import annotations

import io
import math
import os
import threading
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision

MODEL_PATH = Path(__file__).resolve().parent / "models" / "hand_landmarker.task"

# Landmark indices in MediaPipe Hands order.
WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20


class MediaPipeHandDetector:
    """Thread-safe wrapper around MediaPipe HandLandmarker.

    `detect(img_bytes)` returns:
      {
        "gesture": "rotate_left"|"rotate_right"|"zoom"|"point"|"fist"|"none",
        "confidence": float,       # 0..1, coarse classifier margin
        "landmarks": [[x,y,z],..], # 21 points in normalized [0,1] image coords,
                                   # empty if no hand detected
        "handedness": "Left" | "Right" | null,
        "bbox": [xmin,ymin,xmax,ymax],  # normalized [0,1]; [] if no hand
        "palm_area": float,        # bbox area as fraction of frame, drives zoom
        "index_tip": [x, y] | null, # normalized tip coords during `point`
                                   #   (same gate as the laser-pointer dot)
      }
    """

    def __init__(self, model_path: Optional[str | Path] = None):
        path = Path(model_path) if model_path else MODEL_PATH
        if not path.exists():
            raise FileNotFoundError(
                f"Hand landmarker model not found at {path}. "
                "Download from https://storage.googleapis.com/mediapipe-models/"
                "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
            )
        base_opts = mp_tasks.BaseOptions(model_asset_path=str(path))
        options = mp_vision.HandLandmarkerOptions(
            base_options=base_opts,
            running_mode=mp_vision.RunningMode.IMAGE,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._landmarker = mp_vision.HandLandmarker.create_from_options(options)
        # HandLandmarker is not documented as thread-safe — guard with a lock.
        self._lock = threading.Lock()

    def detect(self, img_bytes: bytes) -> dict:
        # Decode JPEG/PNG bytes → RGB numpy array
        try:
            pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        except Exception as e:
            return {"gesture": "none", "confidence": 0.0, "landmarks": [],
                    "handedness": None, "error": f"decode_failed: {e}"}

        arr = np.array(pil)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=arr)

        with self._lock:
            result = self._landmarker.detect(mp_image)

        if not result.hand_landmarks:
            return {"gesture": "none", "confidence": 0.0,
                    "landmarks": [], "handedness": None,
                    "bbox": [], "palm_area": 0.0, "index_tip": None}

        # One hand max (num_hands=1)
        hand = result.hand_landmarks[0]
        handedness = None
        if result.handedness and result.handedness[0]:
            handedness = result.handedness[0][0].category_name  # "Left" / "Right"

        landmarks = [[lm.x, lm.y, lm.z] for lm in hand]
        gesture, confidence = classify_gesture(landmarks)
        bbox, palm_area = _compute_bbox(landmarks)

        return {
            "gesture": gesture,
            "confidence": round(float(confidence), 3),
            "landmarks": landmarks,
            "handedness": handedness,
            "bbox": [round(v, 4) for v in bbox],
            "palm_area": round(float(palm_area), 4),
            "index_tip": _index_tip_if_only_index(landmarks),
        }

    def close(self):
        with self._lock:
            self._landmarker.close()


# ── Gesture classification (pure geometry, no DL) ──────────────────────────

def _dist(a, b) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def _finger_extended(landmarks, tip_idx: int, pip_idx: int) -> bool:
    """A finger is considered extended when its tip is farther from the wrist
    than its PIP joint. This is robust to orientation, since both points move
    together under rotation/translation. Not used for the thumb.
    """
    tip = landmarks[tip_idx]
    pip = landmarks[pip_idx]
    wrist = landmarks[WRIST]
    tip_d = _dist(tip, wrist)
    pip_d = _dist(pip, wrist)
    return tip_d > pip_d * 1.05  # small margin to avoid flicker


def _thumb_extended(landmarks) -> bool:
    """Thumb geometry is different — it folds sideways, not curls. Use the
    distance from wrist of the tip vs the IP joint, with a tighter margin."""
    return _dist(landmarks[THUMB_TIP], landmarks[WRIST]) > \
           _dist(landmarks[THUMB_IP], landmarks[WRIST]) * 1.05


def _hand_scale(landmarks) -> float:
    """Rough hand-size proxy: distance from index MCP (5) to pinky MCP (17)."""
    return _dist(landmarks[INDEX_MCP], landmarks[PINKY_MCP]) or 1e-6


def classify_gesture(landmarks: list[list[float]]) -> tuple[str, float]:
    """Return (label, confidence) for a set of 21 normalized landmarks.

    Order: point → open_palm (→ rotate_left / rotate_right / zoom) → fist.
    The frontend mirrors the webcam frame before POST, so image X matches the
    user's own left/right. "select" is not a classifier label — the frontend
    synthesizes it from a `point → fist` edge.
    """
    if len(landmarks) != 21:
        return "none", 0.0

    index = _finger_extended(landmarks, INDEX_TIP, INDEX_PIP)
    middle = _finger_extended(landmarks, MIDDLE_TIP, MIDDLE_PIP)
    ring = _finger_extended(landmarks, RING_TIP, RING_PIP)
    pinky = _finger_extended(landmarks, PINKY_TIP, PINKY_PIP)

    # 1. Point — only the index finger is extended. No area/orientation gate:
    #    the laser-pointer dot tracks the tip continuously while the user is
    #    "aiming"; the frontend fires a select pin when the hand then closes
    #    into a fist.
    if index and not middle and not ring and not pinky:
        return "point", 0.9

    # 2. Open palm — 4 non-thumb fingers extended. Direction of the hand axis
    #    (wrist → MIDDLE_MCP) decides rotate_left / rotate_right / zoom.
    #    In image coords y+ is down, so:
    #      fingers up    → dy < 0, dx ≈ 0  → angle near −90°  → zoom
    #      fingers right → dx > 0, dy ≈ 0  → angle near   0°  → rotate_right
    #      fingers left  → dx < 0, dy ≈ 0  → angle near ±180° → rotate_left
    #    Pointing down (angle 45–135°) is unused → "none".
    if index and middle and ring and pinky:
        wrist = landmarks[WRIST]
        mmcp = landmarks[MIDDLE_MCP]
        dx = mmcp[0] - wrist[0]
        dy = mmcp[1] - wrist[1]
        angle = math.degrees(math.atan2(dy, dx))
        if -135 <= angle <= -45:
            return "zoom", 0.9
        if -45 < angle <= 45:
            return "rotate_right", 0.9
        if angle > 135 or angle < -135:
            return "rotate_left", 0.9
        return "none", 0.0

    # 3. Fist — none of the four non-thumb fingers extended. Thumb is ignored
    #    because it varies a lot in a real fist (can curl in front of the
    #    fingers or stick out sideways) and insisting on a strict thumb state
    #    caused point→fist transitions to stall on an intermediate frame that
    #    classified as "none" instead of "fist".
    if not index and not middle and not ring and not pinky:
        return "fist", 0.85

    return "none", 0.0


def _index_tip_if_only_index(landmarks: list[list[float]]) -> Optional[list[float]]:
    """Return the index fingertip [x, y] in normalized coords IFF the index is
    the only extended finger. Used by the frontend to render a laser-pointer
    dot that tracks the tip in "pointing mode" only (not during open_palm)."""
    if len(landmarks) != 21:
        return None
    if not _finger_extended(landmarks, INDEX_TIP, INDEX_PIP):
        return None
    if _finger_extended(landmarks, MIDDLE_TIP, MIDDLE_PIP):
        return None
    if _finger_extended(landmarks, RING_TIP, RING_PIP):
        return None
    if _finger_extended(landmarks, PINKY_TIP, PINKY_PIP):
        return None
    tip = landmarks[INDEX_TIP]
    return [round(tip[0], 4), round(tip[1], 4)]


def _compute_bbox(landmarks: list[list[float]]) -> tuple[list[float], float]:
    """Axis-aligned bounding box over the 21 landmarks, in normalized [0,1].

    Returns ([xmin, ymin, xmax, ymax], area). Area is the fraction of the frame
    the hand occupies and is the primary zoom-control signal: larger → zoom in.
    """
    if not landmarks:
        return [0.0, 0.0, 0.0, 0.0], 0.0
    xs = [p[0] for p in landmarks]
    ys = [p[1] for p in landmarks]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    # Clamp to [0,1] because MediaPipe can produce slightly-negative coords
    # for landmarks near the frame edge.
    xmin, ymin = max(0.0, xmin), max(0.0, ymin)
    xmax, ymax = min(1.0, xmax), min(1.0, ymax)
    area = max(0.0, (xmax - xmin) * (ymax - ymin))
    return [xmin, ymin, xmax, ymax], area
