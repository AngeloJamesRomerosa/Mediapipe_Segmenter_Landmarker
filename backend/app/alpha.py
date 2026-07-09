# NOTE: This file is unused in the current client-side inference build.
# Used by composite.py for server-side alpha mask construction. Requires numpy.

"""Alpha mask builders: segmenter smoothing, hand mask, face+hair mask."""

from typing import Any

import cv2
import numpy as np

from .config import FACE_OVAL_IDX, HAND_CONNECTIONS

_HAND_DILATE_K = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
_FACE_DILATE_K = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))


def mp_alpha(mp_mask: np.ndarray) -> np.ndarray:
    """Threshold + blur a raw segmenter mask into a soft alpha channel."""
    mask = np.clip((mp_mask - 0.1) / 0.9, 0.0, 1.0).astype(np.float32)
    return cv2.GaussianBlur(mask, (9, 9), 0)


_FINGER_CHAINS = [
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16],
    [17, 18, 19, 20],
]
_PALM_IDX = [0, 1, 5, 9, 13, 17]  # wrist + all MCP joints


def _finger_wide_pts(chain_pts: np.ndarray, half_w: float) -> np.ndarray:
    """Return the expanded point set for one finger (fully vectorized)."""
    pts = chain_pts.astype(np.float64)
    segs = pts[1:] - pts[:-1]
    lens = np.linalg.norm(segs, axis=1, keepdims=True)
    perp = np.zeros_like(segs)
    safe = lens[:, 0] > 1
    perp[safe] = np.column_stack([-segs[safe, 1], segs[safe, 0]]) / lens[safe]
    off = perp * half_w
    return np.round(
        np.vstack([
            chain_pts,
            pts[:-1] + off, pts[1:] + off,
            pts[:-1] - off, pts[1:] - off,
        ])
    ).astype(np.int32)


def build_hand_alpha(hand_lms_list: Any, h: int, w: int) -> np.ndarray:
    """Palm hull + widened per-finger hull hand mask."""
    mask = np.zeros((h, w), dtype=np.uint8)
    for hand_lms in hand_lms_list:
        pts = np.array(
            [(int(np.clip(lm.x * w, 0, w - 1)), int(np.clip(lm.y * h, 0, h - 1)))
             for lm in hand_lms],
            dtype=np.int32,
        )
        palm_pts = pts[_PALM_IDX]
        cv2.fillPoly(mask, [cv2.convexHull(palm_pts)], 255)

        mcp_span = float(np.linalg.norm(pts[5].astype(np.float64) - pts[17].astype(np.float64)))
        half_w_val = max(5.0, mcp_span / 9.0)

        for chain in _FINGER_CHAINS:
            wide = _finger_wide_pts(pts[chain], half_w_val)
            cv2.fillPoly(mask, [cv2.convexHull(wide)], 255)

        for a, b in HAND_CONNECTIONS:
            cv2.line(mask, tuple(pts[a]), tuple(pts[b]), 255, thickness=8)
        for pt in pts:
            cv2.circle(mask, tuple(pt), 6, 255, -1)

    mask = cv2.dilate(mask, _HAND_DILATE_K)
    return cv2.GaussianBlur(mask, (5, 5), 0).astype(np.float32) / 255.0


def build_face_alpha(face_lms: Any, h: int, w: int) -> np.ndarray | None:
    """Build the face+hair mask at display resolution from face landmarks."""
    oval = [
        (int(np.clip(face_lms[i].x * w, 0, w - 1)),
         int(np.clip(face_lms[i].y * h, 0, h - 1)))
        for i in FACE_OVAL_IDX
        if i < len(face_lms)
    ]
    if len(oval) < 3:
        return None
    arr = np.array(oval)
    top_y = int(arr[:, 1].min())
    chin_y = int(arr[:, 1].max())
    center_x = int(arr[:, 0].mean())
    hair_top = (center_x, max(0, top_y - int((chin_y - top_y) * 0.40)))
    all_pts = oval + [hair_top]
    mask = np.zeros((h, w), dtype=np.uint8)
    hull = cv2.convexHull(np.array(all_pts, dtype=np.int32))
    cv2.fillPoly(mask, [hull], 255)
    mask = cv2.dilate(mask, _FACE_DILATE_K)
    return cv2.GaussianBlur(mask, (11, 11), 0).astype(np.float32) / 255.0
