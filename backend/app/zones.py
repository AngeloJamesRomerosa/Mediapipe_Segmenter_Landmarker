# NOTE: This file is unused in the current client-side inference build.
# Used by composite.py to build per-body-part zone masks. Requires numpy and opencv.

"""Body-zone mask computation: per-part smoothed masks merged into one alpha."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from .config import PART_CONFIG, PART_CONN, PART_NAMES
from .models import get_part_executor

_MORPH_KERNELS: dict[int, Any] = {}


def _get_morph_kernel(size: int) -> Any:
    k = _MORPH_KERNELS.get(size)
    if k is None:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        _MORPH_KERNELS[size] = k
    return k


def _compute_one_part(
    name: str,
    lm_set: set[int],
    dilate: int,
    sw: float,
    ppts: list[Any],
    pose_lms: Any,
    h: int,
    w: int,
    part_smooth: dict[str, np.ndarray | None],
    lk_override: dict[int, Any] | None = None,
) -> np.ndarray | None:
    vis_threshold = (
        0.1 if name in ("left_foot", "right_foot")
        else 0.2 if name in ("left_leg", "right_leg")
        else 0.4
    )
    vis_set = {
        i for i in lm_set
        if i < len(ppts)
        and (pose_lms[i].visibility > vis_threshold or (lk_override and i in lk_override))
    }

    if not vis_set:
        prev = part_smooth.get(name)
        if prev is not None:
            decayed = prev * 0.5
            if decayed.max() > 0.01:
                part_smooth[name] = decayed
                return decayed
            part_smooth[name] = None
        return None

    pts: list[Any] = [ppts[i] for i in vis_set]
    synth_pts: list[Any] = []
    hip_idx = None
    conns = PART_CONN.get(name)

    if name in ("left_arm", "right_arm"):
        distal = [i for i in vis_set if i in {13, 14, 15, 16}]
        if distal and all(pose_lms[i].z > 0.1 for i in distal):
            dilate = max(10, dilate // 2)

    if name == "head" and len(ppts) >= 9:
        l_ear = np.array(ppts[7], dtype=float)
        r_ear = np.array(ppts[8], dtype=float)
        ear_mid = (l_ear + r_ear) / 2
        ear_dist = float(np.linalg.norm(r_ear - l_ear))
        pts.append((int(ear_mid[0]), int(ear_mid[1] - ear_dist * 0.85)))

    if name in ("left_leg", "right_leg"):
        hip_idx = 23 if name == "left_leg" else 24
        sho_idx = 11 if name == "left_leg" else 12
        real_lm = [i for i in lm_set if i < len(ppts) and pose_lms[i].visibility > 0.2]
        if len(real_lm) < 3 and hip_idx < len(ppts):
            hip = np.array(ppts[hip_idx], dtype=float)
            if sho_idx < len(ppts):
                torso_vec = hip - np.array(ppts[sho_idx], dtype=float)
                if float(np.linalg.norm(torso_vec)) > 10:
                    knee_est = hip + torso_vec * 0.8
                    sp = (int(np.clip(knee_est[0], 0, w - 1)), int(np.clip(knee_est[1], 0, h - 1)))
                    pts.append(sp)
                    synth_pts.append(sp)
            else:
                sp = (ppts[hip_idx][0], min(ppts[hip_idx][1] + 150, h - 1))
                pts.append(sp)
                synth_pts.append(sp)

    mask = np.zeros((h, w), dtype=np.uint8)
    drew_something = False

    if conns is not None:
        for a, b in conns:
            if a in vis_set and b in vis_set:
                cv2.line(mask, ppts[a], ppts[b], 255, thickness=dilate)
                drew_something = True
        for i in vis_set:
            cv2.circle(mask, ppts[i], dilate, 255, -1)
            drew_something = True
        for sp in synth_pts:
            cv2.circle(mask, sp, dilate, 255, -1)
            drew_something = True
            if hip_idx is not None and hip_idx < len(ppts):
                cv2.line(mask, ppts[hip_idx], sp, 255, thickness=dilate)
        if drew_something:
            mask = cv2.dilate(mask, _get_morph_kernel(dilate + 1))
    else:
        if len(pts) >= 3:
            hull = cv2.convexHull(np.array(pts, dtype=np.int32))
            cv2.fillPoly(mask, [hull], 255)
            drew_something = True
        elif len(pts) == 2:
            cv2.line(mask, pts[0], pts[1], 255, thickness=dilate * 2)
            drew_something = True
        elif len(pts) == 1:
            cv2.circle(mask, pts[0], dilate * 2, 255, -1)
            drew_something = True
        if drew_something:
            mask = cv2.dilate(mask, _get_morph_kernel(dilate * 2 + 1))

    if not drew_something:
        return part_smooth.get(name)

    raw = cv2.GaussianBlur(mask, (11, 11), 0).astype(np.float32) / 255.0
    prev = part_smooth.get(name)
    part_smooth[name] = (
        raw if (prev is None or prev.shape != raw.shape)
        else (1 - sw) * prev + sw * raw
    )
    return part_smooth[name]


def body_part_zones(
    pose_lms: Any,
    h: int,
    w: int,
    part_smooth: dict[str, np.ndarray | None],
    lk_override: dict[int, Any] | None = None,
) -> np.ndarray | None:
    """Compute all 8 body-part masks in parallel, merge into one alpha."""
    if pose_lms is None:
        parts: list[np.ndarray] = []
        for name in PART_NAMES:
            prev = part_smooth.get(name)
            if prev is not None:
                decayed = prev * 0.3
                if decayed.max() > 0.01:
                    part_smooth[name] = decayed
                    parts.append(decayed)
                else:
                    part_smooth[name] = None
        if not parts:
            return None
        out = parts[0].copy()
        for p in parts[1:]:
            np.maximum(out, p, out=out)
        return out

    ppts = [
        (int(np.clip(lm.x * w, 0, w - 1)), int(np.clip(lm.y * h, 0, h - 1)))
        for lm in pose_lms
    ]
    if lk_override:
        ppts = list(ppts)
        for idx, pt in lk_override.items():
            if idx < len(ppts):
                ppts[idx] = pt

    part_executor = get_part_executor()
    futures = [
        part_executor.submit(
            _compute_one_part, name, lm_set, dilate, sw,
            ppts, pose_lms, h, w, part_smooth, lk_override,
        )
        for name, lm_set, dilate, sw in PART_CONFIG
    ]

    merged = np.zeros((h, w), dtype=np.float32)
    for fut in futures:
        part_mask = fut.result()
        if part_mask is not None:
            np.maximum(merged, part_mask, out=merged)
    return merged
