"""FastAPI app: WebSocket frame processing + static frontend serving.

WebSocket protocol (ws://<host>/ws):
  Client → Server  binary  : raw JPEG camera frame
  Client → Server  text    : JSON config message  {"type": "...", ...}
  Server → Client  binary  : processed JPEG frame (composited background)
  Server → Client  text    : JSON {"type": "landmarks", pose:[...], hands:[...], face:[...]}
                   text    : JSON {"type": "ack"} after config messages
                   text    : JSON {"type": "status", ...} on request
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import uuid
from contextlib import asynccontextmanager
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .composite import apply_bg
from .models import get_infer_executor, get_part_executor, models_status, shutdown_executors
from .session import Session

log = logging.getLogger(__name__)

os.environ.setdefault("GLOG_minloglevel", "3")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

_FRONTEND_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "frontend")
)


@asynccontextmanager
async def _lifespan(app: FastAPI):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    get_infer_executor()
    get_part_executor()
    log.info("MediaPipe BG server ready.")
    yield
    shutdown_executors()


_NO_CACHE = {"Cache-Control": "no-store"}

app = FastAPI(title="MediaPipe Background & Landmarks", lifespan=_lifespan)


# ── No-cache routes for frontend files ───────────────────────────────────────
# Explicit routes registered before the StaticFiles catch-all so these always
# win and the browser never caches the JS/CSS between edits.

@app.get("/")
async def serve_index() -> FileResponse:
    return FileResponse(os.path.join(_FRONTEND_DIR, "index.html"), headers=_NO_CACHE)

@app.get("/js/app.js")
async def serve_js() -> FileResponse:
    return FileResponse(os.path.join(_FRONTEND_DIR, "js", "app.js"), headers=_NO_CACHE)

@app.get("/css/style.css")
async def serve_css() -> FileResponse:
    return FileResponse(os.path.join(_FRONTEND_DIR, "css", "style.css"), headers=_NO_CACHE)


# ── Health / status ───────────────────────────────────────────────────────────


@app.get("/api/status")
def api_status() -> dict[str, Any]:
    return models_status()


# ── WebSocket endpoint ────────────────────────────────────────────────────────


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket) -> None:
    await ws.accept()
    sid = str(uuid.uuid4())[:8]
    session = Session(sid=sid)
    loop = asyncio.get_running_loop()
    session.asyncio_loop = loop
    session.ws_ref = ws
    executor = get_infer_executor()
    log.info("WS connected sid=%s", sid)

    try:
        while True:
            msg = await ws.receive()

            if msg.get("type") == "websocket.disconnect":
                break

            raw_bytes: bytes | None = msg.get("bytes")
            raw_text: str | None = msg.get("text")

            if raw_bytes:
                # Camera frame — process in thread pool (blocking NumPy/cv2 work)
                result = await loop.run_in_executor(
                    executor, _process_frame, session, raw_bytes
                )
                if result is not None:
                    await ws.send_bytes(result)

            elif raw_text:
                try:
                    cfg = json.loads(raw_text)
                    _apply_config(session, cfg)
                    await ws.send_text(json.dumps({"type": "ack"}))
                except (json.JSONDecodeError, KeyError, ValueError) as exc:
                    await ws.send_text(json.dumps({"type": "error", "msg": str(exc)}))

    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        log.info("WS disconnected sid=%s", sid)
        session.ws_ref = None        # stop background threads from pushing
        session.asyncio_loop = None
        if session.loops is not None:
            session.loops.stop()


# ── Frame processing (runs in thread pool) ────────────────────────────────────


def _process_frame(session: Session, jpeg_bytes: bytes) -> bytes | None:
    """Decode, composite, and re-encode one camera frame.

    Uses a non-blocking lock so that when inference is slower than the
    incoming frame rate we drop the new frame instead of building a backlog.
    Landmark data is now pushed directly by the inference threads via
    _push_landmarks() as soon as each model finishes — no longer bundled here.
    """
    if not session.bg_lock.acquire(blocking=False):
        return None
    try:
        arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        frame_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame_bgr is None:
            return None

        needs_apply = (
            session.preset != "none" or
            session.pose_on or session.hand_on or session.face_on or
            session.outline_on
        )

        composited = apply_bg(frame_bgr, session) if needs_apply else frame_bgr

        ok, buf = cv2.imencode(".jpg", composited, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            return None
        return bytes(buf)
    finally:
        session.bg_lock.release()


# ── Config handler ────────────────────────────────────────────────────────────


def _apply_config(session: Session, cfg: dict[str, Any]) -> None:
    msg_type = cfg.get("type")

    if msg_type == "preset":
        preset = str(cfg.get("preset", "none"))
        session.preset = preset
        session.prev_alpha = None
        if preset != "image":
            session.bg_image = None

    elif msg_type == "toggle":
        feature = str(cfg.get("feature", ""))
        value = bool(cfg.get("value", False))
        if feature == "pose":
            session.pose_on = value
            if not value:
                session.part_smooth = {}
                session.prev_alpha = None
        elif feature == "hand":
            session.hand_on = value
        elif feature == "face":
            session.face_on = value
        elif feature == "outline":
            session.outline_on = value

    elif msg_type == "fps_cap":
        session.fps_cap = max(0, int(cfg.get("fps", 30)))

    elif msg_type == "outline_strength":
        session.outline_strength = max(0.0, min(1.0, float(cfg.get("value", 0.5))))

    elif msg_type == "bg_image":
        # Accepts a data-URL or raw base64 string
        data: str = str(cfg.get("data", ""))
        if "," in data:
            data = data.split(",", 1)[1]
        try:
            raw = base64.b64decode(data)
            arr = np.frombuffer(raw, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is not None:
                session.bg_image = img
                session.preset = "image"
                session.prev_alpha = None
        except Exception as exc:
            log.warning("bg_image decode failed: %s", exc)

    elif msg_type == "status":
        pass  # client polling — ack is sent by caller


# ── Static frontend (registered last so API routes take priority) ─────────────

if os.path.isdir(_FRONTEND_DIR):
    app.mount("/", StaticFiles(directory=_FRONTEND_DIR, html=True), name="static")
