"""Per-session background inference loops: segmenter, pose, hand, and face.

Pose / hand / face landmarkers run in LIVE_STREAM mode:
  - The loop thread calls detect_async() and returns immediately (non-blocking).
  - MediaPipe fires the result callback on its own internal thread as soon as
    inference finishes — no waiting for the round-trip.
  - LIVE_STREAM callbacks are always serial (one at a time per instance), so
    per-loop state (optical flow, cached zones) needs no extra locking.

  apply_bg ─submit(frame)─► [seg_q]  ─► seg_loop  ─► _seg_alpha
            ─submit(frame)─► [pose_q] ─► pose_loop ─► _pose_alpha / _body_zones
            ─submit(frame)─► [hand_q] ─► hand_loop ─► _hand_alpha
            ─submit(frame)─► [face_q] ─► face_loop ─► _face_alpha
            ◄─results()──────────────── (latest, non-blocking)
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import queue
import threading
import time
from typing import TYPE_CHECKING, Any

import cv2
import mediapipe as mp
import numpy as np

from .alpha import mp_alpha
from .config import PART_NAMES
from .infer import face_process_result, hand_process_result, pose_process_result
from .models import (
    get_segmenter,
    create_pose_landmarker_live,
    create_hand_landmarker_live,
    create_face_landmarker_live,
)
from .zones import body_part_zones

if TYPE_CHECKING:
    from .session import Session

log = logging.getLogger(__name__)

_LK_PARAMS = dict(
    winSize=(35, 35),
    maxLevel=4,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
)


class BgModelLoops:
    """Manages per-session background inference threads."""

    def __init__(self, session: Session) -> None:
        self._s = session
        self._running = True

        # Input queues: maxsize=1 so old frames are dropped when inference is busy
        self._seg_q: queue.Queue[Any] = queue.Queue(maxsize=1)
        self._pose_q: queue.Queue[Any] = queue.Queue(maxsize=1)
        self._hand_q: queue.Queue[Any] = queue.Queue(maxsize=1)
        self._face_q: queue.Queue[Any] = queue.Queue(maxsize=1)

        # Latest results (lock-protected, written by loop threads, read by composite)
        self._seg_alpha: np.ndarray | None = None
        self._seg_mask_avg: float = -1.0
        self._pose_alpha: np.ndarray | None = None
        self._lower_alpha: np.ndarray | None = None
        self._body_zones: np.ndarray | None = None
        self._pose_lms_raw: Any = None
        self._hand_alpha: np.ndarray | None = None
        self._hand_lms_raw: Any = None
        self._face_alpha: np.ndarray | None = None
        self._face_lms_raw: Any = None
        self._lock = threading.Lock()

        # Optical flow state — pose callback only, no lock needed (callbacks are serial)
        self._of_prev_gray: np.ndarray | None = None
        self._of_prev_pts: Any = None
        self._of_velocity: dict[int, Any] = {}
        self._cached_zones: np.ndarray | None = None

        sid = session.sid
        self._seg_thread  = threading.Thread(target=self._seg_loop,  daemon=True, name=f"seg-{sid}")
        self._pose_thread = threading.Thread(target=self._pose_loop, daemon=True, name=f"pose-{sid}")
        self._hand_thread = threading.Thread(target=self._hand_loop, daemon=True, name=f"hand-{sid}")
        self._face_thread = threading.Thread(target=self._face_loop, daemon=True, name=f"face-{sid}")

        for t in (self._seg_thread, self._pose_thread, self._hand_thread, self._face_thread):
            t.start()

    # ── internal helpers ──────────────────────────────────────────────────────

    def _push_landmarks(self) -> None:
        """Push current landmark state directly to the browser WebSocket.

        Called by each inference callback as soon as new results are stored,
        bypassing the frame round-trip so Panel 2 updates without waiting for
        the next composited frame response.
        """
        ws   = self._s.ws_ref
        loop = self._s.asyncio_loop
        if ws is None or loop is None:
            return
        with self._lock:
            pose_lms = self._pose_lms_raw
            hand_lms = self._hand_lms_raw
            face_lms = self._face_lms_raw
        msg: dict = {"type": "landmarks"}
        if pose_lms:
            msg["pose"] = [
                {"x": p.x, "y": p.y, "z": p.z, "v": float(p.visibility)}
                for p in pose_lms
            ]
        if hand_lms:
            msg["hands"] = [
                [{"x": p.x, "y": p.y, "z": p.z} for p in hand]
                for hand in hand_lms
            ]
        if face_lms:
            msg["face"] = [{"x": p.x, "y": p.y, "z": p.z} for p in face_lms]
        try:
            asyncio.run_coroutine_threadsafe(ws.send_text(json.dumps(msg)), loop)
        except Exception:
            pass

    @staticmethod
    def _q_put(q: queue.Queue[Any], item: Any) -> None:
        try:
            q.put_nowait(item)
        except queue.Full:
            with contextlib.suppress(queue.Empty):
                q.get_nowait()
            with contextlib.suppress(queue.Full):
                q.put_nowait(item)

    # ── background loops ──────────────────────────────────────────────────────

    def _seg_loop(self) -> None:
        while self._running:
            try:
                infer_rgb, h, w = self._seg_q.get(timeout=0.1)
            except queue.Empty:
                continue
            segmenter = get_segmenter()
            if segmenter is None:
                continue
            try:
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=infer_rgb)
                seg_result = segmenter.segment(mp_img)
                if not seg_result.confidence_masks:
                    continue
                raw_mask = np.squeeze(seg_result.confidence_masks[0].numpy_view()).copy()
                if raw_mask.shape != (h, w):
                    raw_mask = cv2.resize(raw_mask, (w, h), interpolation=cv2.INTER_LINEAR)
                alpha_seg = mp_alpha(raw_mask)
                with self._lock:
                    self._seg_alpha = alpha_seg
                    self._seg_mask_avg = float(raw_mask.mean())
            except Exception as exc:
                log.warning("[seg_loop] sid=%s: %s", self._s.sid, exc)

    def _pose_loop(self) -> None:
        # _pending maps timestamp → (full-res infer_rgb, h, w) so the callback
        # can do optical flow and zone computation with the original frame data.
        _pending: dict[int, tuple[np.ndarray, int, int]] = {}

        def _on_pose(result: Any, _output_image: Any, timestamp_ms: int) -> None:
            frame_data = _pending.pop(timestamp_ms, None)
            if frame_data is None or not self._s.pose_on:
                return
            infer_rgb, h, w = frame_data
            ih, iw = infer_rgb.shape[:2]

            curr_gray = cv2.cvtColor(infer_rgb, cv2.COLOR_RGB2GRAY)
            if (self._of_prev_gray is not None
                    and self._of_prev_gray.shape != curr_gray.shape):
                self._of_prev_gray = None
                self._of_prev_pts = None
                self._of_velocity = {}
                self._cached_zones = None

            motion_score = (
                float(np.abs(
                    curr_gray.astype(np.int16) - self._of_prev_gray.astype(np.int16)
                ).mean())
                if self._of_prev_gray is not None else 0.0
            )

            pose_lms, pose_alpha, lower_alpha = pose_process_result(result, h, w)

            if not self._s.part_smooth:
                self._s.part_smooth = {name: None for name in PART_NAMES}

            lk_override: dict[int, Any] = {}
            if pose_lms is not None:
                ppts_infer = [
                    (int(np.clip(lm.x * iw, 0, iw - 1)), int(np.clip(lm.y * ih, 0, ih - 1)))
                    for lm in pose_lms
                ]
                low_vis = {
                    i for i, lm in enumerate(pose_lms)
                    if lm.visibility <= (
                        0.1 if i in {27, 28, 29, 30, 31, 32}
                        else 0.2 if i in {23, 24, 25, 26}
                        else 0.4
                    )
                }
                if low_vis and self._of_prev_gray is not None and self._of_prev_pts is not None:
                    new_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                        self._of_prev_gray, curr_gray, self._of_prev_pts, None, **_LK_PARAMS
                    )
                    if new_pts is not None:
                        for i in range(len(status)):
                            if status[i][0] != 1:
                                continue
                            ox, oy = self._of_prev_pts[i][0]
                            nx, ny = new_pts[i][0]
                            dx, dy = nx - ox, ny - oy
                            pvx, pvy = self._of_velocity.get(i, (0.0, 0.0))
                            self._of_velocity[i] = (pvx * 0.5 + dx * 0.5, pvy * 0.5 + dy * 0.5)
                            if i in low_vis:
                                if i in {27, 28, 29, 30, 31, 32}:
                                    vx, vy = self._of_velocity[i]
                                    px = int(np.clip(nx + vx, 0, iw - 1))
                                    py = int(np.clip(ny + vy, 0, ih - 1))
                                else:
                                    px = int(np.clip(nx, 0, iw - 1))
                                    py = int(np.clip(ny, 0, ih - 1))
                                lk_override[i] = (int(px * w / iw), int(py * h / ih))
                self._of_prev_pts = np.array(ppts_infer, dtype=np.float32).reshape(-1, 1, 2)

            self._of_prev_gray = curr_gray

            if (motion_score < 2.0 and not lk_override
                    and self._cached_zones is not None
                    and self._cached_zones.shape == (h, w)):
                body_zones = self._cached_zones
            else:
                body_zones = body_part_zones(pose_lms, h, w, self._s.part_smooth, lk_override)
                if body_zones is not None:
                    self._cached_zones = body_zones

            with self._lock:
                self._pose_alpha = pose_alpha
                self._lower_alpha = lower_alpha
                self._body_zones = body_zones
                self._pose_lms_raw = pose_lms
            self._push_landmarks()

        lmker = create_pose_landmarker_live(_on_pose)
        ts_ms = 0
        last_run = 0.0

        while self._running:
            try:
                infer_rgb, h, w = self._pose_q.get(timeout=0.1)
            except queue.Empty:
                continue

            if not self._s.pose_on:
                with self._lock:
                    self._pose_alpha = self._lower_alpha = self._body_zones = None
                self._of_prev_gray = None
                self._of_prev_pts = None
                self._of_velocity = {}
                self._cached_zones = None
                continue

            now = time.monotonic()
            fps_cap = self._s.fps_cap
            if fps_cap > 0 and now - last_run < 1.0 / fps_cap:
                continue
            last_run = now

            if lmker is None:
                lmker = create_pose_landmarker_live(_on_pose)
            if lmker is None:
                continue

            now_ms = int(now * 1000)
            ts_ms = max(ts_ms + 1, now_ms)

            infer_half = cv2.resize(infer_rgb, (w // 2, h // 2), interpolation=cv2.INTER_LINEAR)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=infer_half)
            _pending[ts_ms] = (infer_rgb, h, w)
            # Safety valve — shouldn't accumulate but prevents unbounded growth
            if len(_pending) > 10:
                _pending.clear()
            try:
                lmker.detect_async(mp_img, ts_ms)
            except Exception as exc:
                log.warning("[pose_loop] sid=%s: %s", self._s.sid, exc)
                _pending.pop(ts_ms, None)

    def _hand_loop(self) -> None:
        _pending: dict[int, tuple[int, int]] = {}
        consecutive_lost = 0

        def _on_hand(result: Any, _output_image: Any, timestamp_ms: int) -> None:
            nonlocal consecutive_lost
            hw = _pending.pop(timestamp_ms, None)
            if hw is None or not self._s.hand_on:
                return
            h, w = hw
            hand_alpha, hand_lms = hand_process_result(result, h, w)
            with self._lock:
                if hand_alpha is not None:
                    consecutive_lost = 0
                    self._hand_alpha = hand_alpha
                    self._hand_lms_raw = hand_lms
                else:
                    self._hand_lms_raw = None
                    if self._hand_alpha is not None:
                        consecutive_lost += 1
                        if consecutive_lost > 3:
                            decayed = self._hand_alpha * 0.2
                            self._hand_alpha = decayed if decayed.max() > 0.05 else None
            self._push_landmarks()

        lmker = create_hand_landmarker_live(_on_hand)
        ts_ms = 0
        last_run = 0.0

        while self._running:
            try:
                infer_rgb, h, w = self._hand_q.get(timeout=0.1)
            except queue.Empty:
                continue

            if not self._s.hand_on:
                with self._lock:
                    self._hand_alpha = None
                    self._hand_lms_raw = None
                continue

            now = time.monotonic()
            fps_cap = self._s.fps_cap
            if fps_cap > 0 and now - last_run < 1.0 / fps_cap:
                continue
            last_run = now

            if lmker is None:
                lmker = create_hand_landmarker_live(_on_hand)
            if lmker is None:
                continue

            now_ms = int(now * 1000)
            ts_ms = max(ts_ms + 1, now_ms)

            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=infer_rgb)
            _pending[ts_ms] = (h, w)
            if len(_pending) > 10:
                _pending.clear()
            try:
                lmker.detect_async(mp_img, ts_ms)
            except Exception as exc:
                log.warning("[hand_loop] sid=%s: %s", self._s.sid, exc)
                _pending.pop(ts_ms, None)

    def _face_loop(self) -> None:
        _pending: dict[int, tuple[int, int]] = {}

        def _on_face(result: Any, _output_image: Any, timestamp_ms: int) -> None:
            hw = _pending.pop(timestamp_ms, None)
            if hw is None or not self._s.face_on:
                return
            h, w = hw
            face_alpha, face_lms = face_process_result(result, h, w)
            with self._lock:
                self._face_alpha = face_alpha
                self._face_lms_raw = face_lms
            self._push_landmarks()

        lmker = create_face_landmarker_live(_on_face)  # also pre-warms download
        ts_ms = 0
        last_run = 0.0

        while self._running:
            try:
                infer_rgb, h, w = self._face_q.get(timeout=0.1)
            except queue.Empty:
                continue

            if not self._s.face_on:
                with self._lock:
                    self._face_alpha = None
                    self._face_lms_raw = None
                continue

            now = time.monotonic()
            fps_cap = self._s.fps_cap
            if fps_cap > 0 and now - last_run < 1.0 / fps_cap:
                continue
            last_run = now

            if lmker is None:
                lmker = create_face_landmarker_live(_on_face)
            if lmker is None:
                continue

            now_ms = int(now * 1000)
            ts_ms = max(ts_ms + 1, now_ms)

            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=infer_rgb)
            _pending[ts_ms] = (h, w)
            if len(_pending) > 10:
                _pending.clear()
            try:
                lmker.detect_async(mp_img, ts_ms)
            except Exception as exc:
                log.warning("[face_loop] sid=%s: %s", self._s.sid, exc)
                _pending.pop(ts_ms, None)

    # ── public API ────────────────────────────────────────────────────────────

    def submit(self, infer_rgb: np.ndarray, h: int, w: int, needs_seg: bool = True) -> None:
        """Send a frame to background threads (non-blocking, drops old).

        needs_seg=False skips the segmenter when preset is 'none' and outline
        is off — the segmenter result would be thrown away anyway, so skipping
        it frees a significant share of CPU for the landmark models.
        """
        if needs_seg:
            self._q_put(self._seg_q, (infer_rgb, h, w))
        self._q_put(self._pose_q, (infer_rgb, h, w))
        self._q_put(self._hand_q, (infer_rgb, h, w))
        self._q_put(self._face_q, (infer_rgb, h, w))

    def results(self) -> tuple[
        np.ndarray | None,  # pose_alpha
        np.ndarray | None,  # lower_alpha
        np.ndarray | None,  # body_zones
        np.ndarray | None,  # hand_alpha
        np.ndarray | None,  # seg_alpha
        float,              # seg_mask_avg
        np.ndarray | None,  # face_alpha
    ]:
        """Return latest results from all threads (non-blocking)."""
        with self._lock:
            return (
                self._pose_alpha, self._lower_alpha, self._body_zones,
                self._hand_alpha, self._seg_alpha, self._seg_mask_avg,
                self._face_alpha,
            )

    def debug_results(self) -> tuple[Any, Any, Any]:
        """Return raw landmarks for frontend overlay (non-blocking)."""
        with self._lock:
            return (self._pose_lms_raw, self._hand_lms_raw, self._face_lms_raw)

    def stop(self) -> None:
        """Signal all background threads to exit."""
        self._running = False
