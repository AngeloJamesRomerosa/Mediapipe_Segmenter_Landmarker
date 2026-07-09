"""Per-connection session state."""

from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from .loops import BgModelLoops


@dataclass
class Session:
    """Mutable per-WebSocket-connection state for background removal."""

    sid: str = ""

    # Background preset: none | blur | black | white | green | image
    preset: str = "none"
    bg_image: np.ndarray | None = None  # custom upload, stored as BGR

    # Feature toggles
    pose_on: bool = True
    hand_on: bool = True
    face_on: bool = False
    outline_on: bool = False
    outline_strength: float = 0.5

    # Pipeline FPS cap: 0 = uncapped (MAX), otherwise loops run at most this rate
    fps_cap: int = 30

    # Composite state
    prev_alpha: np.ndarray | None = None
    part_smooth: dict[str, np.ndarray | None] = field(default_factory=dict)

    # Diagnostics
    bg_apply_count: int = 0
    bg_mask_avg: float = -1.0

    # Background inference daemon threads (pose + hand + face + seg).
    # Lazily created on first apply_bg call, stopped on disconnect.
    loops: Any = field(default=None, compare=False, repr=False)

    # Frame-drop guard: non-blocking acquire prevents frame backlog when
    # inference is slower than the incoming camera rate.
    bg_lock: threading.Lock = field(
        default_factory=threading.Lock, compare=False, repr=False
    )

    # Set by ws_endpoint so background threads can push landmarks directly
    # without waiting for the next frame round-trip.
    ws_ref: Any = field(default=None, compare=False, repr=False)
    asyncio_loop: asyncio.AbstractEventLoop | None = field(
        default=None, compare=False, repr=False
    )
