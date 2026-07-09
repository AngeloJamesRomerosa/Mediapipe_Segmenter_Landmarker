"""Background layer builder: blur, solid colors, and custom image upload."""

from typing import Any

import cv2
import numpy as np

from .config import BLUR_RADIUS

_bg_frame_cache: dict[Any, Any] = {}


def build_bg(frame_bgr: np.ndarray, session: Any, h: int, w: int) -> np.ndarray | None:
    """Return the background layer for this session frame, or None (no preset)."""
    preset = session.preset

    if preset == "blur":
        return cv2.GaussianBlur(frame_bgr, (BLUR_RADIUS, BLUR_RADIUS), 0)

    if preset == "black":
        return np.zeros_like(frame_bgr)

    if preset == "white":
        return np.full_like(frame_bgr, 255)

    if preset == "green":
        bg = np.zeros_like(frame_bgr)
        bg[:, :, 1] = 177  # BGR green-screen colour
        return bg

    if preset == "image" and session.bg_image is not None:
        key = (id(session.bg_image), h, w)
        if key not in _bg_frame_cache:
            src = session.bg_image
            if src.shape[0] != h or src.shape[1] != w:
                src = cv2.resize(src, (w, h))
            _bg_frame_cache[key] = src  # already BGR from cv2.imdecode
        return _bg_frame_cache[key]

    return None
