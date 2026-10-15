"""Face crop + mask resize to 128x128 (paper Sec. IV preprocessing).

Paper uses dlib HOG detector + Kazemi–Sullivan 68-landmark face mask.
Without dlib we use AFLW FaceRect (when provided) or OpenCV Haar, then an
elliptical face mask — see DEVIATIONS.md.
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np


def _elliptical_mask(h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(mask, (w // 2, h // 2), (int(w * 0.42), int(h * 0.48)), 0, 0, 360, 255, -1)
    return mask


def face_mask_128(
    bgr: np.ndarray,
    rect: Optional[Tuple[int, int, int, int]] = None,
) -> np.ndarray:
    """Return 128x128 float grayscale face mask in [0, 255]."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
    h, w = gray.shape[:2]
    if rect is None or any(v is None for v in rect):
        cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces = cascade.detectMultiScale(gray, 1.1, 4, minSize=(32, 32))
        if len(faces) == 0:
            x, y, rw, rh = 0, 0, w, h
        else:
            # largest face
            x, y, rw, rh = max(faces, key=lambda f: f[2] * f[3])
    else:
        x, y, rw, rh = [int(v) for v in rect]
        x = max(0, x)
        y = max(0, y)
        rw = max(1, min(rw, w - x))
        rh = max(1, min(rh, h - y))

    # expand slightly like a loose face hull
    pad = int(0.05 * max(rw, rh))
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(w, x + rw + pad)
    y1 = min(h, y + rh + pad)
    crop = gray[y0:y1, x0:x1]
    if crop.size == 0:
        crop = gray
    ch, cw = crop.shape
    mask = _elliptical_mask(ch, cw)
    masked = cv2.bitwise_and(crop, crop, mask=mask)
    out = cv2.resize(masked, (128, 128), interpolation=cv2.INTER_AREA)
    return out.astype(np.float64)
