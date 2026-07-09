# NOTE: This file is unused in the current client-side inference build.
# To restore server-side inference, re-import this in main.py and wire it
# back into loops.py. Requires mediapipe and numpy in requirements.txt.

"""MediaPipe model singletons and thread pools.

All models use IMAGE running mode (stateless, thread-safe singletons).
Models are downloaded lazily on first use if not present on disk.
No GPU path — Render free tier is CPU-only.
"""

import logging
import os
import threading
import urllib.request
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

log = logging.getLogger(__name__)

try:
    from mediapipe.tasks import python as _mp_tasks
    from mediapipe.tasks.python import vision as _mp_vision

    MP_AVAILABLE = True
except ImportError:
    MP_AVAILABLE = False
    _mp_tasks = None  # type: ignore[assignment]
    _mp_vision = None  # type: ignore[assignment]

from .config import FACE_PATH, FACE_URL, HAND_PATH, HAND_URL, POSE_PATH, POSE_URL, SEG_PATH, SEG_URL

# ── Model singletons ──────────────────────────────────────────────────────────

_MODEL_FAILED = object()  # sentinel — distinct from None (not yet initialised)
_segmenter: Any = None
_pose_lmker: Any = None
_hand_lmker: Any = None
_face_lmker: Any = None

_seg_lock = threading.Lock()
_pose_lock = threading.Lock()
_hand_lock = threading.Lock()
_face_lock_obj = threading.Lock()


def _init_model(url: str, path: str, label: str, create_fn: Callable[[], Any]) -> Any:
    """Download model file if missing, then create it via create_fn."""
    if not MP_AVAILABLE:
        return None
    try:
        if not os.path.exists(path):
            log.info("Downloading %s...", label)
            urllib.request.urlretrieve(url, path)
        model = create_fn()
        log.info("%s ready.", label)
        return model
    except Exception as exc:
        log.warning("%s init failed: %s", label, exc)
        return None


def _ensure_model_file(url: str, path: str, label: str) -> bool:
    """Download model file if missing. Returns True when the file is available."""
    if not MP_AVAILABLE:
        return False
    try:
        if not os.path.exists(path):
            log.info("Downloading %s...", label)
            urllib.request.urlretrieve(url, path)
        return True
    except Exception as exc:
        log.warning("%s download failed: %s", label, exc)
        return False


# ── Per-session LIVE_STREAM-mode factory functions ────────────────────────────
# LIVE_STREAM mode: detect_async() returns immediately; results arrive via
# callback on MediaPipe's internal thread. Callbacks are always serial (one at
# a time), so optical-flow state shared within a single loop is safe without
# extra locking. Each call creates a NEW instance — state is per-session.

def create_pose_landmarker_live(callback: Callable) -> Any:
    if not _ensure_model_file(POSE_URL, POSE_PATH, "Pose landmarker"):
        return None
    try:
        return _mp_vision.PoseLandmarker.create_from_options(
            _mp_vision.PoseLandmarkerOptions(
                base_options=_mp_tasks.BaseOptions(
                    model_asset_path=POSE_PATH,
                    delegate=_mp_tasks.BaseOptions.Delegate.CPU,
                ),
                running_mode=_mp_vision.RunningMode.LIVE_STREAM,
                num_poses=1,
                min_pose_detection_confidence=0.5,
                min_pose_presence_confidence=0.5,
                min_tracking_confidence=0.5,
                output_segmentation_masks=True,
                result_callback=callback,
            )
        )
    except Exception as exc:
        log.warning("Pose landmarker (LIVE_STREAM) creation failed: %s", exc)
        return None


def create_hand_landmarker_live(callback: Callable) -> Any:
    if not _ensure_model_file(HAND_URL, HAND_PATH, "Hand landmarker"):
        return None
    try:
        return _mp_vision.HandLandmarker.create_from_options(
            _mp_vision.HandLandmarkerOptions(
                base_options=_mp_tasks.BaseOptions(
                    model_asset_path=HAND_PATH,
                    delegate=_mp_tasks.BaseOptions.Delegate.CPU,
                ),
                running_mode=_mp_vision.RunningMode.LIVE_STREAM,
                num_hands=2,
                min_hand_detection_confidence=0.5,
                min_hand_presence_confidence=0.5,
                min_tracking_confidence=0.5,
                result_callback=callback,
            )
        )
    except Exception as exc:
        log.warning("Hand landmarker (LIVE_STREAM) creation failed: %s", exc)
        return None


def create_face_landmarker_live(callback: Callable) -> Any:
    if not _ensure_model_file(FACE_URL, FACE_PATH, "Face landmarker"):
        return None
    try:
        return _mp_vision.FaceLandmarker.create_from_options(
            _mp_vision.FaceLandmarkerOptions(
                base_options=_mp_tasks.BaseOptions(
                    model_asset_path=FACE_PATH,
                    delegate=_mp_tasks.BaseOptions.Delegate.CPU,
                ),
                running_mode=_mp_vision.RunningMode.LIVE_STREAM,
                num_faces=1,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
                min_tracking_confidence=0.5,
                result_callback=callback,
            )
        )
    except Exception as exc:
        log.warning("Face landmarker (LIVE_STREAM) creation failed: %s", exc)
        return None


def get_segmenter() -> Any:
    global _segmenter
    if _segmenter is not None:
        return None if _segmenter is _MODEL_FAILED else _segmenter
    with _seg_lock:
        if _segmenter is None:
            result = _init_model(
                SEG_URL, SEG_PATH, "Selfie segmenter",
                lambda: _mp_vision.ImageSegmenter.create_from_options(
                    _mp_vision.ImageSegmenterOptions(
                        base_options=_mp_tasks.BaseOptions(
                            model_asset_path=SEG_PATH,
                            delegate=_mp_tasks.BaseOptions.Delegate.CPU,
                        ),
                        running_mode=_mp_vision.RunningMode.IMAGE,
                        output_confidence_masks=True,
                    )
                ),
            )
            _segmenter = result if result is not None else _MODEL_FAILED
    return None if _segmenter is _MODEL_FAILED else _segmenter


def get_pose_landmarker() -> Any:
    global _pose_lmker
    if _pose_lmker is not None:
        return None if _pose_lmker is _MODEL_FAILED else _pose_lmker
    with _pose_lock:
        if _pose_lmker is None:
            result = _init_model(
                POSE_URL, POSE_PATH, "Pose landmarker",
                lambda: _mp_vision.PoseLandmarker.create_from_options(
                    _mp_vision.PoseLandmarkerOptions(
                        base_options=_mp_tasks.BaseOptions(model_asset_path=POSE_PATH),
                        running_mode=_mp_vision.RunningMode.IMAGE,
                        num_poses=1,
                        min_pose_detection_confidence=0.5,
                        min_tracking_confidence=0.3,
                        output_segmentation_masks=True,
                    )
                ),
            )
            _pose_lmker = result if result is not None else _MODEL_FAILED
    return None if _pose_lmker is _MODEL_FAILED else _pose_lmker


def get_hand_landmarker() -> Any:
    global _hand_lmker
    if _hand_lmker is not None:
        return None if _hand_lmker is _MODEL_FAILED else _hand_lmker
    with _hand_lock:
        if _hand_lmker is None:
            result = _init_model(
                HAND_URL, HAND_PATH, "Hand landmarker",
                lambda: _mp_vision.HandLandmarker.create_from_options(
                    _mp_vision.HandLandmarkerOptions(
                        base_options=_mp_tasks.BaseOptions(model_asset_path=HAND_PATH),
                        running_mode=_mp_vision.RunningMode.IMAGE,
                        num_hands=2,
                        min_hand_detection_confidence=0.5,
                        min_tracking_confidence=0.3,
                    )
                ),
            )
            _hand_lmker = result if result is not None else _MODEL_FAILED
    return None if _hand_lmker is _MODEL_FAILED else _hand_lmker


def get_face_landmarker() -> Any:
    global _face_lmker
    if _face_lmker is not None:
        return None if _face_lmker is _MODEL_FAILED else _face_lmker
    with _face_lock_obj:
        if _face_lmker is None:
            result = _init_model(
                FACE_URL, FACE_PATH, "Face landmarker",
                lambda: _mp_vision.FaceLandmarker.create_from_options(
                    _mp_vision.FaceLandmarkerOptions(
                        base_options=_mp_tasks.BaseOptions(model_asset_path=FACE_PATH),
                        running_mode=_mp_vision.RunningMode.IMAGE,
                        num_faces=1,
                        min_face_detection_confidence=0.5,
                        min_tracking_confidence=0.3,
                    )
                ),
            )
            _face_lmker = result if result is not None else _MODEL_FAILED
    return None if _face_lmker is _MODEL_FAILED else _face_lmker


# ── Thread pools ──────────────────────────────────────────────────────────────

_cpu = os.cpu_count() or 2
_infer_executor: ThreadPoolExecutor | None = None
_part_executor: ThreadPoolExecutor | None = None


def get_infer_executor() -> ThreadPoolExecutor:
    global _infer_executor
    if _infer_executor is None:
        _infer_executor = ThreadPoolExecutor(max_workers=max(2, _cpu // 2))
    return _infer_executor


def get_part_executor() -> ThreadPoolExecutor:
    global _part_executor
    if _part_executor is None:
        _part_executor = ThreadPoolExecutor(max_workers=max(2, min(8, _cpu)))
    return _part_executor


def shutdown_executors() -> None:
    global _infer_executor, _part_executor
    if _infer_executor is not None:
        _infer_executor.shutdown(wait=False)
        _infer_executor = None
    if _part_executor is not None:
        _part_executor.shutdown(wait=False)
        _part_executor = None


def models_status() -> dict[str, Any]:
    return {
        "available": MP_AVAILABLE,
        "segmenter": _segmenter is not None and _segmenter is not _MODEL_FAILED,
        "pose":      os.path.exists(POSE_PATH),
        "hand":      os.path.exists(HAND_PATH),
        "face":      os.path.exists(FACE_PATH),
    }
