"""Composite math and the main per-frame entry point (apply_bg).

All models (segmenter, pose, hand, face) run in per-session background threads
(loops.py). apply_bg never blocks — it reads the latest available results and
returns immediately. CPU-only; no GPU path (targeting Render free tier).
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from .build import build_bg
from .config import ERODE_KERNEL, INFER_MAX_W
from .session import Session

log = logging.getLogger(__name__)

_CLOSE_KERNELS: dict[int, np.ndarray] = {}


def _composite_cpu(alpha_2d: np.ndarray, frame_bgr: np.ndarray, bg_bgr: np.ndarray) -> np.ndarray:
    alpha = alpha_2d[:, :, np.newaxis]
    return np.clip(
        frame_bgr.astype(np.float32) * alpha + bg_bgr.astype(np.float32) * (1.0 - alpha),
        0, 255,
    ).astype(np.uint8)


def _merge_alphas(
    alpha_seg: np.ndarray,
    pose_alpha: np.ndarray | None,
    lower_alpha: np.ndarray | None,
    body_zones: np.ndarray | None,
    hand_alpha: np.ndarray | None,
    face_alpha: np.ndarray | None,
    prev_alpha: np.ndarray | None,
    old_weight: float,
) -> np.ndarray:
    final = alpha_seg.copy()
    if pose_alpha is not None:
        np.maximum(final, pose_alpha, out=final)
    if lower_alpha is not None:
        np.maximum(final, lower_alpha, out=final)
    if body_zones is not None:
        gate = np.clip(alpha_seg * 3.0, 0.0, 1.0)
        np.maximum(final, body_zones * gate, out=final)
    if prev_alpha is not None and old_weight > 0.01:
        final = old_weight * prev_alpha + (1.0 - old_weight) * final
    if hand_alpha is not None:
        np.maximum(final, hand_alpha, out=final)
    if face_alpha is not None:
        np.maximum(final, face_alpha, out=final)
    return final


def apply_bg(frame_bgr: np.ndarray, session: Session) -> np.ndarray:
    """Apply background removal/replacement to one frame.

    Non-blocking: submits the frame to background inference threads, reads
    the latest available results, and composites immediately.  Returns the
    original frame until the first segmenter result arrives.
    """
    session.bg_apply_count += 1
    h, w = frame_bgr.shape[:2]

    # Downscale + convert for inference (saves ~3x pixels at 1080p)
    if w > INFER_MAX_W:
        scale = INFER_MAX_W / w
        infer_rgb = cv2.cvtColor(
            cv2.resize(frame_bgr, (INFER_MAX_W, int(h * scale))),
            cv2.COLOR_BGR2RGB,
        )
    else:
        infer_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    # Lazy-init per-session loops
    if session.loops is None:
        from .loops import BgModelLoops
        session.loops = BgModelLoops(session)

    needs_seg = session.preset != "none" or session.outline_on
    session.loops.submit(infer_rgb, h, w, needs_seg=needs_seg)

    pose_alpha, lower_alpha, body_zones, hand_alpha, alpha_seg, seg_mask_avg, face_alpha = (
        session.loops.results()
    )

    if alpha_seg is None:
        return frame_bgr  # still warming up — pass through

    if alpha_seg.shape != (h, w):
        alpha_seg = cv2.resize(alpha_seg, (w, h), interpolation=cv2.INTER_LINEAR)

    session.bg_mask_avg = seg_mask_avg

    bg_bgr = build_bg(frame_bgr, session, h, w)
    if bg_bgr is None and not session.outline_on:
        return frame_bgr  # preset is "none" and no glow — pass through
    if bg_bgr is None:
        bg_bgr = frame_bgr  # glow-only: composite is a no-op, but outline still applies

    prev_alpha = (
        session.prev_alpha
        if (session.prev_alpha is not None and session.prev_alpha.shape == (h, w))
        else None
    )

    final_alpha = _merge_alphas(
        alpha_seg, pose_alpha, lower_alpha, body_zones, hand_alpha, face_alpha,
        prev_alpha, old_weight=0.0,
    )

    # Morphological close: fill small holes (cheaper at INFER_MAX_W)
    if final_alpha.max() > 0.3:
        if w > INFER_MAX_W:
            cw, ch = INFER_MAX_W, int(h * INFER_MAX_W / w)
            a_small = cv2.resize(final_alpha, (cw, ch), interpolation=cv2.INTER_LINEAR)
            close_size = max(15, cw // 25)
            close_k = _CLOSE_KERNELS.setdefault(
                close_size, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size))
            )
            alpha_u8 = (a_small * 255).clip(0, 255).astype(np.uint8)
            a_small = cv2.morphologyEx(alpha_u8, cv2.MORPH_CLOSE, close_k).astype(np.float32) / 255.0
            final_alpha = cv2.resize(a_small, (w, h), interpolation=cv2.INTER_LINEAR)
        else:
            close_size = max(15, w // 25)
            close_k = _CLOSE_KERNELS.setdefault(
                close_size, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size))
            )
            alpha_u8 = (final_alpha * 255).clip(0, 255).astype(np.uint8)
            final_alpha = cv2.morphologyEx(alpha_u8, cv2.MORPH_CLOSE, close_k).astype(np.float32) / 255.0

    comp = _composite_cpu(final_alpha, frame_bgr, bg_bgr)
    session.prev_alpha = final_alpha

    if session.outline_on:
        dilated = cv2.dilate(final_alpha, ERODE_KERNEL, iterations=6)
        edge_ring = cv2.GaussianBlur(
            np.clip(dilated - final_alpha, 0.0, 1.0), (11, 11), 0
        )[:, :, np.newaxis]
        # Additive cyan glow — BGR equivalent of accent #7ec8e3
        glow_bgr = np.array([[[227, 200, 126]]], dtype=np.float32)
        comp = np.clip(
            comp.astype(np.float32) + edge_ring * glow_bgr * session.outline_strength * 3.0,
            0, 255,
        ).astype(np.uint8)

    return comp
