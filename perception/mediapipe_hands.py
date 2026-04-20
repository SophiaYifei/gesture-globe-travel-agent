"""DL perception — MediaPipe Hand Landmarker + geometric gesture classifier.

4 gesture classes (mapped to future globe controls):
  open_palm  → rotate + zoom   (4-5 fingers extended; zoom level = palm_area)
  point      → select          (only index extended)
  fist       → stop             (no fingers extended)
  none       → (no hand detected) or ambiguous pose

Zoom is NOT its own gesture: while open_palm is held, the hand's bounding-box
area (as a fraction of the frame) drives zoom level — larger palm area zooms
in, smaller zooms out. This is why every response carries `bbox` and
`palm_area` alongside the gesture label.

Uses MediaPipe's new Tasks API (`mediapipe.tasks.python.vision.HandLandmarker`)
since the old `mp.solutions.hands` module was removed in mediapipe 0.10.22+.
The detector itself is a DL model (21-point landmark regressor); classification
is a pure geometric rule on top of the landmarks — the "DL" label applies to
the perception stage, which is what the project rubric cares about. A separate
non_dl_hands.py will reimplement the perception stage with classical CV only.
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
        "gesture": "open_palm" | "point" | "fist" | "none",
        "confidence": float,       # 0..1, coarse classifier margin
        "landmarks": [[x,y,z],..], # 21 points in normalized [0,1] image coords,
                                   # empty if no hand detected
        "handedness": "Left" | "Right" | null,
        "bbox": [xmin,ymin,xmax,ymax],  # normalized [0,1]; [] if no hand
        "palm_area": float,        # bbox area as fraction of frame, drives zoom
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
                    "bbox": [], "palm_area": 0.0}

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

    Order: open_palm → point → fist → none.
    Pinch is no longer a class — zoom is driven by the bbox area of open_palm.
    """
    if len(landmarks) != 21:
        return "none", 0.0

    thumb = _thumb_extended(landmarks)
    index = _finger_extended(landmarks, INDEX_TIP, INDEX_PIP)
    middle = _finger_extended(landmarks, MIDDLE_TIP, MIDDLE_PIP)
    ring = _finger_extended(landmarks, RING_TIP, RING_PIP)
    pinky = _finger_extended(landmarks, PINKY_TIP, PINKY_PIP)

    n_ext = sum([thumb, index, middle, ring, pinky])

    # 1. Open palm — the four non-thumb fingers are extended; thumb optional.
    #    Zoom signal = palm_area returned alongside the gesture label.
    if index and middle and ring and pinky:
        return "open_palm", 0.9

    # 2. Point — only the index finger is extended (thumb may or may not be).
    if index and not middle and not ring and not pinky:
        return "point", 0.85

    # 3. Fist — no fingers extended (thumb may wrap across palm).
    if n_ext == 0 or (n_ext == 1 and thumb):
        return "fist", 0.85

    return "none", 0.0


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
