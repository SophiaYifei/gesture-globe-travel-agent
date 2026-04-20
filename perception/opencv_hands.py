"""Non-DL perception — classical CV hand gesture detector.

Returns the SAME dict shape as MediaPipeHandDetector so the /gesture endpoint
can swap between detectors at runtime based on a `mode` parameter. No deep
learning involved. Pipeline:

    HSV skin threshold (two hue ranges for wraparound)
        ↓
    Morphological open + close (denoise + fill palm gaps)
        ↓
    findContours → take the largest blob as the hand
        ↓
    Bounding box + centroid from image moments
        ↓
    Convex hull + convexityDefects → count finger valleys
        ↓
    Classify (finger count + bbox aspect + centroid offset) → gesture label

Gesture output matches the DL classifier's interface:
    rotate_left / rotate_right / zoom / point / fist / none

Known limitations (inherent to classical skin-CV):
  * Sensitive to lighting and skin tone — the HSV range is a one-size default.
  * Skin-colored background (wood, walls, warm lighting) causes false positives.
  * Cannot resolve gestures whose defining feature is out-of-plane depth —
    e.g. "finger pointing at camera" has no long 2D contour, so `point` here
    requires the finger to be visible in-plane (typically held up/out).
  * No handedness; no 3D landmarks.

These are the honest baselines against which the DL detector is compared.
"""
from __future__ import annotations

import io
import math
from collections import deque, Counter
from typing import Optional

import cv2
import numpy as np
from PIL import Image


# Skin detection uses BOTH HSV and YCrCb color spaces, intersected. HSV alone
# trips on warm wood/walls; YCrCb alone trips on anything in the right Cr/Cb
# band regardless of hue. Demanding both agree is the classical recipe from
# skin-detection literature (Kakumanu et al. 2007, etc.) and dramatically
# reduces background false positives.
#
# Generous hue & value bounds because the user's skin tone and lighting vary
# a lot between sessions. Saturation capped at 220 to reject red/orange
# objects (lipstick red, traffic cones, brand logos) that look like deep skin.
_HSV_LOWER_1 = np.array([0,   25,  45],  dtype=np.uint8)
_HSV_UPPER_1 = np.array([25, 220, 255],  dtype=np.uint8)
_HSV_LOWER_2 = np.array([160, 25,  45],  dtype=np.uint8)
_HSV_UPPER_2 = np.array([180, 220, 255], dtype=np.uint8)

# YCrCb classical skin range (Hsu et al. 2002).
_YCRCB_LOWER = np.array([0,   135, 85],  dtype=np.uint8)
_YCRCB_UPPER = np.array([255, 180, 135], dtype=np.uint8)

# 3x3 kernel, fewer iterations — 5x5 was smearing out finger valleys and
# making every open palm look like one solid convex blob.
_MORPH_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

_MIN_CONTOUR_AREA_PX = 2000
_MAX_CONTOUR_AREA_FRAC = 0.70

# Defect depth is stored as pixels × 256. The filter in the DL-less finger
# counter requires a valley to be deeper than DEFECT_DEPTH_FRAC of the hand's
# bbox height. Scales naturally with hand size instead of failing on
# fingers-close-together-when-near-camera.
_DEFECT_DEPTH_FRAC = 0.12
_DEFECT_ANGLE_MAX_RAD = math.radians(100)  # was π/2 (90°), now wider

# Temporal smoothing — return the majority label of the last N raw frames.
# Eats 1-frame classifier flickers without adding noticeable latency
# (at 10 fps this is 300 ms of memory).
_SMOOTH_WINDOW = 3

# Haar face cascade — classical (Viola-Jones, 2001), ships with opencv-contrib.
# Used purely to punch the user's face out of the skin mask so it doesn't
# become the "largest skin blob" the pipeline classifies as a hand.
_FACE_CASCADE_XML = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"


class OpenCVHandDetector:
    """Classical-CV hand detector. Same dict shape as MediaPipeHandDetector."""

    def __init__(self):
        # Haar cascade is stateless at detect time but loading the XML once
        # here saves ~1-2ms per frame.
        self._face_cascade = cv2.CascadeClassifier(_FACE_CASCADE_XML)
        if self._face_cascade.empty():
            # Don't fail hard — just skip face subtraction and print a notice.
            print("[opencv_hands] Warning: face cascade XML not loaded; "
                  "hand detection will be noisier when a face is in frame.")
            self._face_cascade = None
        # Rolling history of raw gesture labels for temporal smoothing.
        self._history: deque[str] = deque(maxlen=_SMOOTH_WINDOW)

    def detect(self, img_bytes: bytes) -> dict:
        try:
            pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        except Exception as e:
            return self._no_hand(error=f"decode_failed: {e}")

        rgb = np.array(pil)
        h, w = rgb.shape[:2]
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)

        # Build a skin mask in HSV (two hue windows for wraparound), and a
        # separate one in YCrCb. Intersect them — a pixel must be classified
        # as skin by BOTH color spaces, which cuts a huge amount of noise
        # from warm-toned backgrounds that HSV alone would keep.
        hsv_a = cv2.inRange(hsv, _HSV_LOWER_1, _HSV_UPPER_1)
        hsv_b = cv2.inRange(hsv, _HSV_LOWER_2, _HSV_UPPER_2)
        hsv_mask = cv2.bitwise_or(hsv_a, hsv_b)
        ycrcb_mask = cv2.inRange(ycrcb, _YCRCB_LOWER, _YCRCB_UPPER)
        mask = cv2.bitwise_and(hsv_mask, ycrcb_mask)

        # Morph open removes salt noise, close fills tiny pinholes without
        # closing the valleys between fingers. Kernel deliberately small (3x3).
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _MORPH_KERNEL, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _MORPH_KERNEL, iterations=1)

        # Punch any detected frontal face out of the skin mask. Without this
        # the face is the largest skin blob — its bbox aspect (~1.2-1.4) and
        # smooth shape alternately satisfy the `point` and `fist` thresholds,
        # so the classifier bounces and fires spurious select events.
        if self._face_cascade is not None:
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            faces = self._face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=4, minSize=(60, 60)
            )
            for (fx, fy, fw, fh) in faces:
                # Pad to also remove hair/neck, which are often skin-ish.
                pad_w = int(fw * 0.25)
                pad_h = int(fh * 0.30)
                x1 = max(0, fx - pad_w)
                y1 = max(0, fy - pad_h)
                x2 = min(w, fx + fw + pad_w)
                y2 = min(h, fy + fh + pad_h)
                mask[y1:y2, x1:x2] = 0

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return self._no_hand()

        hand = max(contours, key=cv2.contourArea)
        area_px = cv2.contourArea(hand)
        frame_px = w * h
        if area_px < _MIN_CONTOUR_AREA_PX or area_px / frame_px > _MAX_CONTOUR_AREA_FRAC:
            return self._no_hand()

        x, y, bw, bh = cv2.boundingRect(hand)
        bbox_norm = [x / w, y / h, (x + bw) / w, (y + bh) / h]
        palm_area = (bw * bh) / frame_px

        m = cv2.moments(hand)
        if m["m00"] == 0:
            return self._no_hand()
        cx = m["m10"] / m["m00"]
        cy = m["m01"] / m["m00"]

        # Adaptive defect-depth threshold. Scale with the larger bbox side so
        # the "valleys between fingers" test works whether the hand is close
        # (big in frame) or far. Previous fixed 40-px threshold rejected real
        # finger valleys whenever the hand wasn't huge.
        hand_scale_px = max(bw, bh)
        min_defect_depth_fixed = int(hand_scale_px * _DEFECT_DEPTH_FRAC * 256)
        fingers = _count_fingers_via_defects(hand, min_defect_depth_fixed)

        # Solidity = contour area / convex hull area. Key discriminator between
        # a FIST (compact blob, solidity ≈ 0.94-0.98) and an OPEN PALM with
        # fingers together (elongated, still some concavity from thumb/edges,
        # solidity ≈ 0.80-0.90). Without this feature we can't distinguish
        # those two cases because they both produce 0 convexity defects.
        hull_points = cv2.convexHull(hand)
        hull_area = cv2.contourArea(hull_points)
        solidity = float(area_px) / max(hull_area, 1.0)

        # Width of the contour in its TOP 25% slice, as a fraction of bbox
        # width. Used to distinguish "palm + one raised finger" (narrow top,
        # wide bottom) from "palm with fingers together/spread" (similar
        # width throughout). The whole-shape width_ratio can't tell these
        # apart because in both cases the palm widens the overall bbox.
        pts_flat = hand.reshape(-1, 2)
        top_y_cutoff = y + bh * 0.25
        top_pts = pts_flat[pts_flat[:, 1] <= top_y_cutoff]
        if len(top_pts) > 0:
            top_width = float(top_pts[:, 0].max() - top_pts[:, 0].min())
            top_width_ratio = top_width / max(bw, 1)
        else:
            top_width_ratio = 1.0

        raw_gesture, confidence = _classify(
            fingers, bw, bh, cx, x, solidity, top_width_ratio
        )

        # Temporal smoothing: return the majority of the last N raw labels.
        # Eats 1-frame classifier flickers. The frontend's POINT_COMMIT_FRAMES
        # gate then adds another layer of sustained-pose confirmation.
        self._history.append(raw_gesture)
        vote = Counter(self._history).most_common(1)[0]
        # Only accept the smoothed label if it's a strict majority. Otherwise
        # return 'none' rather than picking arbitrarily.
        gesture = vote[0] if vote[1] * 2 > len(self._history) else "none"

        # Fingertip for the laser pointer — computed on the CURRENT frame's
        # raw analysis (not smoothed), so the dot tracks immediately. We only
        # expose it when the raw classification agrees that the blob looks
        # like a pointing pose, so the red dot doesn't wander on palm/fist.
        index_tip = None
        if raw_gesture == "point":
            pts = hand.reshape(-1, 2).astype(np.float32)
            d2 = (pts[:, 0] - cx) ** 2 + (pts[:, 1] - cy) ** 2
            idx = int(np.argmax(d2))
            tx, ty = pts[idx]
            index_tip = [round(float(tx) / w, 4), round(float(ty) / h, 4)]

        return {
            "gesture": gesture,
            "confidence": round(float(confidence), 3),
            "landmarks": [],        # OpenCV doesn't produce 21-point landmarks
            "handedness": None,     # Not inferrable from a silhouette
            "bbox": [round(v, 4) for v in bbox_norm],
            "palm_area": round(float(palm_area), 4),
            "index_tip": index_tip,
            # Debug payload — useful when diagnosing "why did it say X" in
            # the browser DevTools Network tab. Cheap to include.
            "debug": {
                "raw_gesture": raw_gesture,
                "defects": fingers,
                "solidity": round(solidity, 3),
                "aspect_h_over_w": round(bh / max(bw, 1), 3),
                "top_width_ratio": round(top_width_ratio, 3),
                "hand_scale_px": int(hand_scale_px),
            },
        }

    def close(self):
        pass

    @staticmethod
    def _no_hand(error: Optional[str] = None) -> dict:
        out = {
            "gesture": "none", "confidence": 0.0,
            "landmarks": [], "handedness": None,
            "bbox": [], "palm_area": 0.0, "index_tip": None,
        }
        if error:
            out["error"] = error
        return out


# ── Helpers ────────────────────────────────────────────────────────────────

def _count_fingers_via_defects(contour, min_depth_fixed: int) -> int:
    """Count convexity-defect valleys (spaces between extended fingers).

    `min_depth_fixed` is in OpenCV's 1/256-px fixed-point units, supplied by
    the caller so the threshold can scale with the hand's size in the frame.
    Also widens the angle gate from 90° to 100° — fingers that are extended
    but held fairly close together leave valleys whose interior angle exceeds
    90° and were being silently dropped.
    """
    try:
        hull = cv2.convexHull(contour, returnPoints=False)
        defects = cv2.convexityDefects(contour, hull)
    except cv2.error:
        return 0
    if defects is None:
        return 0

    count = 0
    for i in range(defects.shape[0]):
        s, e, f, d = defects[i, 0]
        if d < min_depth_fixed:
            continue
        start = contour[s][0]
        end = contour[e][0]
        far = contour[f][0]
        a = math.hypot(end[0] - start[0], end[1] - start[1])
        b = math.hypot(far[0] - start[0], far[1] - start[1])
        c = math.hypot(end[0] - far[0], end[1] - far[1])
        if b == 0 or c == 0:
            continue
        cos_angle = (b * b + c * c - a * a) / (2 * b * c)
        cos_angle = max(-1.0, min(1.0, cos_angle))
        angle = math.acos(cos_angle)
        if angle < _DEFECT_ANGLE_MAX_RAD:
            count += 1
    return count


def _classify(defects_count: int, bw: int, bh: int,
              cx: float, bbox_x: int,
              solidity: float,
              top_width_ratio: float) -> tuple[str, float]:
    """Map (defect count, bbox shape, centroid offset, solidity,
    top-slice width) → gesture.

    Aspect-first framework; solidity and defects are confidence/safety gates:

        aspect = bh / bw
        width_ratio = bw / bh
        top_width_ratio = width(top 25% of contour) / bw
        ┌─────────────────────────────────────────────────────────┐
        │ aspect > 1.5 AND (w/h < 0.30 OR top_width_ratio < 0.40) │
        │                                             → point     │
        │ aspect > 1.3                                → zoom       │
        │ aspect < 0.80                               → rotate_*   │
        │ 0.80 ≤ aspect ≤ 1.25 AND sol > 0.85         → fist       │
        │ otherwise                                   → none       │
        └─────────────────────────────────────────────────────────┘

    `top_width_ratio` is the fix for "☝️ palm + raised index finger" —
    overall width_ratio stays high (palm widens the bbox), but only the
    finger occupies the top quarter so its width is narrow. A palm with
    fingers together/spread has ~uniform width top-to-bottom, so the top
    slice spans most of the bbox and stays in the zoom branch.
    """
    aspect = bh / max(bw, 1)
    bbox_cx = bbox_x + bw / 2
    width_ratio = bw / max(bh, 1)
    finger_valleys_visible = defects_count >= 2

    # 1. Point — either the whole blob is narrow (isolated finger) OR just
    #    the top slice is narrow (finger protruding from a wider palm).
    if aspect > 1.5 and (width_ratio < 0.30 or top_width_ratio < 0.40):
        return "point", 0.70

    # 2. Zoom — palm held vertical, fingers up. Same test regardless of
    #    whether fingers are spread or together; defects just ups confidence.
    if aspect > 1.3:
        return "zoom", 0.85 if finger_valleys_visible else 0.60

    # 3. Rotate — palm held horizontal. Direction from centroid offset:
    #    palm is the dense half of the silhouette, so centroid biases toward
    #    the palm. Centroid right of bbox center ⇒ fingertips point LEFT.
    if aspect < 0.80:
        label = "rotate_left" if cx > bbox_cx else "rotate_right"
        return label, 0.85 if finger_valleys_visible else 0.60

    # 4. Fist — near-square, solid, compact. Solidity floor is what stops a
    #    near-square palm shadow from being read as a fist.
    if 0.80 <= aspect <= 1.25 and solidity > 0.85:
        return "fist", 0.80

    return "none", 0.0
