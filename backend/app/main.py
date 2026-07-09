"""FastAPI app: static frontend serving only.
All MediaPipe inference runs client-side via the Tasks Vision JS API (WASM/WebGL).
"""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

log = logging.getLogger(__name__)

_FRONTEND_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "frontend")
)


@asynccontextmanager
async def _lifespan(app: FastAPI):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    log.info("MediaPipe BG server ready (client-side inference mode).")
    yield


_NO_CACHE = {"Cache-Control": "no-store"}

app = FastAPI(title="MediaPipe Background & Landmarks", lifespan=_lifespan)


@app.get("/")
async def serve_index() -> FileResponse:
    return FileResponse(os.path.join(_FRONTEND_DIR, "index.html"), headers=_NO_CACHE)

@app.get("/js/app.js")
async def serve_js() -> FileResponse:
    return FileResponse(os.path.join(_FRONTEND_DIR, "js", "app.js"), headers=_NO_CACHE)

@app.get("/css/style.css")
async def serve_css() -> FileResponse:
    return FileResponse(os.path.join(_FRONTEND_DIR, "css", "style.css"), headers=_NO_CACHE)

@app.get("/api/status")
def api_status() -> dict:
    return {"mode": "client-side", "inference": "browser-wasm"}


if os.path.isdir(_FRONTEND_DIR):
    app.mount("/", StaticFiles(directory=_FRONTEND_DIR, html=True), name="static")
