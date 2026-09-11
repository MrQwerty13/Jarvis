"""Lightweight local face and raised-finger detectors for camAI."""

from dataclasses import dataclass
from pathlib import Path
from functools import lru_cache

import cv2
import numpy as np


@dataclass
class Face:
    box: tuple
    score: float = 1.0


@lru_cache(maxsize=1)
def _face_cascade():
    return cv2.CascadeClassifier(
        str(Path(cv2.data.haarcascades) / 'haarcascade_frontalface_default.xml')
    )


def detect_faces(frame):
    """Detect visible faces; this identifies presence, not a person's identity."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    cascade = _face_cascade()
    if cascade.empty():
        return []
    gray = cv2.equalizeHist(gray)
    found = cascade.detectMultiScale(
        gray, scaleFactor=1.08, minNeighbors=5, minSize=(48, 48)
    )
    return [Face(tuple(map(int, box))) for box in found]


def _skin_mask(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    hsv_mask = cv2.inRange(hsv, np.array((0, 20, 45)), np.array((25, 190, 255)))
    ycrcb_mask = cv2.inRange(ycrcb, np.array((0, 130, 70)), np.array((255, 180, 135)))
    mask = cv2.bitwise_and(hsv_mask, ycrcb_mask)
    kernel = np.ones((5, 5), np.uint8)
    return cv2.morphologyEx(cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel), cv2.MORPH_CLOSE, kernel)


def count_fingers(frame):
    """Return (count, box), or (None, None) when no convincing hand is visible.

    Convexity defects work well for an open hand against a contrasting background;
    conservative geometry prevents faces and random blobs from becoming numbers.
    """
    mask = _skin_mask(frame)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None
    contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(contour)
    if area < frame.shape[0] * frame.shape[1] * 0.015:
        return None, None
    x, y, w, h = cv2.boundingRect(contour)
    if h < w * 0.65 or h < 70:
        return None, None
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    if hull_area <= 0 or area / hull_area < 0.35:
        return None, None
    defects = cv2.convexityDefects(contour, cv2.convexHull(contour, returnPoints=False))
    gaps = 0
    if defects is not None:
        for start, end, far, depth in defects[:, 0]:
            a, b, c = contour[start][0], contour[end][0], contour[far][0]
            sides = [np.linalg.norm(a - c), np.linalg.norm(b - c), np.linalg.norm(a - b)]
            angle = np.degrees(np.arccos(np.clip((sides[0]**2 + sides[1]**2 - sides[2]**2) /
                                                  max(2 * sides[0] * sides[1], 1), -1, 1)))
            if angle < 85 and depth / 256.0 > max(w, h) * 0.035:
                gaps += 1
    if gaps == 0:
        count = 1 if h > w * 1.15 else None
    else:
        count = min(5, gaps + 1)
    return count, (int(x), int(y), int(w), int(h)) if count is not None else None
