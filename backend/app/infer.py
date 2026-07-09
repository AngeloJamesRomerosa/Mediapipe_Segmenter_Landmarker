# NOTE: This file is unused in the current client-side inference build.
# To restore server-side inference, these result processors are called from
# loops.py callbacks. Requires mediapipe and numpy.

"""Per-model result processors for LIVE_STREAM mode.

detect_async() is called from the loop thread; results arrive via a callback
on MediaPipe's internal thread. These helpers take the raw result object and
return alpha masks + landmark lists — no model calls happen here.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from .alpha import build_face_alpha, build_hand_alpha, mp_alpha


def pose_process_result(
    result: Any, h: int, w: int
) -> tuple[Any, np.ndarray | None, np.ndarray | None]:
    """PoseLandmarkerResult → (pose_lms, pose_alpha, lower_alpha)."""
    pose_lms = result.pose_landmarks[0] if result.pose_landmarks else None
    pose_alpha: np.ndarray | None = None
    lower_alpha: np.ndarray | None = None

    if result.segmentation_masks:
        pose_seg = np.squeeze(result.segmentation_masks[0].numpy_view()).copy()
        if pose_seg.shape != (h, w):
            pose_seg = cv2.resize(pose_seg, (w, h), interpolation=cv2.INTER_LINEAR)
        pose_alpha = mp_alpha(pose_seg)
        if pose_lms is not None and len(pose_lms) > 24:
            hip_y = int(max(
                np.clip(pose_lms[23].y * h, 0, h - 1),
                np.clip(pose_lms[24].y * h, 0, h - 1),
            ))
            if hip_y < h:
                lm_ = np.clip((pose_seg - 0.05) / 0.95, 0.0, 1.0).astype(np.float32)
                lower_alpha = np.zeros((h, w), dtype=np.float32)
                lower_alpha[hip_y:, :] = lm_[hip_y:, :]
                lower_alpha = cv2.GaussianBlur(lower_alpha, (11, 11), 0)

    return pose_lms, pose_alpha, lower_alpha


def hand_process_result(
    result: Any, h: int, w: int
) -> tuple[np.ndarray | None, Any]:
    """HandLandmarkerResult → (hand_alpha, hand_lms) or (None, None)."""
    if result.hand_landmarks:
        return build_hand_alpha(result.hand_landmarks, h, w), result.hand_landmarks
    return None, None


def face_process_result(
    result: Any, h: int, w: int
) -> tuple[np.ndarray | None, Any]:
    """FaceLandmarkerResult → (face_alpha, face_lms) or (None, None)."""
    if result.face_landmarks:
        lms = result.face_landmarks[0]
        return build_face_alpha(lms, h, w), lms
    return None, None
