"use strict";

// ── Landmark connection tables ────────────────────────────────────────────────

const POSE_CONNECTIONS = [
  [11,12],[11,13],[13,15],[12,14],[14,16],
  [11,23],[12,24],[23,24],
  [23,25],[25,27],[24,26],[26,28],
  [27,29],[27,31],[29,31],[28,30],[28,32],[30,32],
  [0,11],[0,12],
];

const HAND_CONNECTIONS = [
  [0,1],[1,2],[2,3],[3,4],
  [0,5],[5,6],[6,7],[7,8],
  [5,9],[9,10],[10,11],[11,12],
  [9,13],[13,14],[14,15],[15,16],
  [13,17],[17,18],[18,19],[19,20],
  [0,17],
];

const FACE_OVAL_IDX = [
  10,338,297,332,284,251,389,356,454,323,361,288,
  397,365,379,378,400,377,152,148,176,149,150,136,
  172,58,132,93,234,127,162,21,54,103,67,109,
];

// Inner face contours (MediaPipe face mesh indices)
const FACE_RIGHT_EYE  = [33,246,161,160,159,158,157,173,133,155,154,153,145,144,163,7];
const FACE_LEFT_EYE   = [362,382,381,380,374,373,390,249,263,466,388,387,386,385,384,398];
const FACE_RIGHT_BROW = [46,53,52,65,55,70,63,105,66,107];
const FACE_LEFT_BROW  = [276,283,282,295,285,300,293,334,296,336];
const FACE_NOSE_RIDGE = [168,6,197,195,5,4];
const FACE_LIPS_OUTER = [61,146,91,181,84,17,314,405,321,375,291,308,324,318,402,317,14,87,178,88,95];

// ── Per-landmark smoothing sigmas ─────────────────────────────────────────────
// alpha = exp(-d²/sigma) per rAF tick. Smaller sigma → snappier (arms/feet).
// Larger sigma → more stable (face/shoulders/hips).

const POSE_SIGMA = (() => {
  const s = new Float32Array(33).fill(5e-4);
  for (let i = 0; i <= 10; i++) s[i] = 8e-4;   // face landmarks — stable
  s[11] = s[12] = 7e-4;                          // shoulders
  s[13] = s[14] = 3e-4;                          // elbows — responsive
  for (let i = 15; i <= 22; i++) s[i] = 2e-4;   // wrists + hand tips
  s[23] = s[24] = 7e-4;                          // hips — stable
  s[25] = s[26] = 5e-4;                          // knees
  s[27] = s[28] = 3e-4;                          // ankles
  for (let i = 29; i <= 32; i++) s[i] = 1.5e-4; // feet — most responsive
  return s;
})();
const HAND_SIGMA = 1.5e-4;  // hands snap to fresh detection quickly
const FACE_SIGMA = 8e-4;    // face mesh stays stable

// ── App ───────────────────────────────────────────────────────────────────────

class App {
  constructor() {
    // Panel 1 — raw video element
    this._video = document.getElementById("camera");

    // Panel 2 — camera frame + landmark skeleton (requestAnimationFrame)
    this._lmCanvas = document.getElementById("landmarks");
    this._lmCtx    = this._lmCanvas.getContext("2d");

    // Panel 3 — server-composited frame
    this._outCanvas = document.getElementById("output");
    this._outCtx    = this._outCanvas.getContext("2d");

    // Off-screen canvas for JPEG capture
    this._capture = document.createElement("canvas");
    this._capCtx  = this._capture.getContext("2d");

    this._ws           = null;
    this._connected    = false;
    this._running      = false;   // false = stopped, true = active
    this._captureTimer = null;
    this._rafActive    = false;   // tracks whether the rAF loop is alive

    // FPS counter (Panel 3)
    this._frameCount  = 0;
    this._fpsLastTime = Date.now();

    // Latency tracking: time when last frame binary was sent
    this._frameSentAt = null;
    this._latencyEma  = null;   // exponential moving average for smooth display

    // Smoothed landmarks for Panel 2 — interpolated toward server target each rAF tick
    this._displayLandmarks = null;  // what is actually drawn (smoothed)
    this._landmarks        = null;  // latest target from server

    this._settings = {
      preset:          "none",
      pose:            true,
      hand:            true,
      face:            false,
      outline:         false,
      outlineStrength: 0.5,
      fpsCap:          30,       // 0 = MAX (uncapped)
    };

    this._bindUI();
  }

  // ── Start / Stop ──────────────────────────────────────────────────────────────

  start() {
    if (this._running) return;
    this._running = true;
    this._setBtn("stop");
    this._initCamera();
  }

  stop() {
    if (!this._running) return;
    this._running = false;
    this._setBtn("start");
    this._setStatus("Stopped", false);
    document.getElementById("fps-display").textContent     = "— fps";
    document.getElementById("latency-display").textContent = "— ms";
    this._frameSentAt = null;
    this._latencyEma  = null;

    // Stop frame capture interval
    if (this._captureTimer) {
      clearInterval(this._captureTimer);
      this._captureTimer = null;
    }

    // Close WebSocket without auto-reconnect
    if (this._ws) {
      this._ws.onclose = null;
      this._ws.close();
      this._ws = null;
    }
    this._connected = false;

    // Turn off camera
    if (this._video.srcObject) {
      this._video.srcObject.getTracks().forEach(t => t.stop());
      this._video.srcObject = null;
    }

    // Clear canvases
    this._lmCtx.clearRect(0, 0, this._lmCanvas.width, this._lmCanvas.height);
    this._outCtx.clearRect(0, 0, this._outCanvas.width, this._outCanvas.height);
    this._landmarks        = null;
    this._displayLandmarks = null;
  }

  // ── Camera init ───────────────────────────────────────────────────────────────

  async _initCamera() {
    this._setStatus("Starting camera…", false);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" },
      });
      if (!this._running) { stream.getTracks().forEach(t => t.stop()); return; }

      this._video.srcObject = stream;
      await new Promise(r => { this._video.onloadedmetadata = r; });
      this._video.play();

      const W = this._video.videoWidth  || 640;
      const H = this._video.videoHeight || 480;

      this._capture.width   = W;
      this._capture.height  = H;
      this._lmCanvas.width  = W;
      this._lmCanvas.height = H;
      this._outCanvas.width  = W;
      this._outCanvas.height = H;

      this._connect();
      this._startCapture();
      this._startLandmarksLoop();
    } catch (err) {
      this._running = false;
      this._setBtn("start");
      this._setStatus(`Camera error: ${err.message}`, false);
    }
  }

  // ── WebSocket ─────────────────────────────────────────────────────────────────

  _connect() {
    const proto = location.protocol === "https:" ? "wss:" : "ws:";
    this._ws = new WebSocket(`${proto}//${location.host}/ws`);
    this._ws.binaryType = "arraybuffer";

    this._ws.onopen = () => {
      this._connected = true;
      this._setStatus("Connected", true);
      // Sync settings to the new session
      this._send({ type: "toggle", feature: "pose",    value: this._settings.pose });
      this._send({ type: "toggle", feature: "hand",    value: this._settings.hand });
      this._send({ type: "toggle", feature: "face",    value: this._settings.face });
      this._send({ type: "toggle", feature: "outline", value: this._settings.outline });
      this._send({ type: "outline_strength",           value: this._settings.outlineStrength });
      this._send({ type: "preset",  preset: this._settings.preset });
      this._send({ type: "fps_cap", fps:    this._settings.fpsCap });
    };

    this._ws.onclose = () => {
      this._connected = false;
      if (!this._running) return; // stopped by user — don't reconnect
      this._setStatus("Reconnecting…", false);
      setTimeout(() => { if (this._running) this._connect(); }, 3000);
    };

    this._ws.onerror = () => { /* onclose fires next */ };

    this._ws.onmessage = (evt) => {
      if (evt.data instanceof ArrayBuffer) {
        this._renderOutputFrame(evt.data);   // Panel 3
      } else {
        try {
          const msg = JSON.parse(evt.data);
          if (msg.type === "landmarks") this._landmarks = msg;
        } catch (_) { /* ignore */ }
      }
    };
  }

  // ── Frame capture & send ──────────────────────────────────────────────────────

  _startCapture() {
    const fps = this._settings.fpsCap;
    const ms  = fps === 0 ? 1000 / 60 : 1000 / fps;
    this._captureTimer = setInterval(() => {
      if (!this._connected || this._ws.bufferedAmount > 60_000) return;
      this._capCtx.drawImage(this._video, 0, 0);
      this._capture.toBlob((blob) => {
        if (!blob || !this._connected) return;
        blob.arrayBuffer().then((buf) => {
          if (this._ws?.readyState === WebSocket.OPEN) {
            this._frameSentAt = Date.now();
            this._ws.send(buf);
          }
        });
      }, "image/jpeg", 0.8);
    }, ms);
  }

  // ── Panel 2: landmark loop (requestAnimationFrame) ────────────────────────────

  _startLandmarksLoop() {
    if (this._rafActive) return;
    this._rafActive = true;
    const tick = () => {
      if (!this._running) { this._rafActive = false; return; }
      const W = this._lmCanvas.width;
      const H = this._lmCanvas.height;
      this._lmCtx.drawImage(this._video, 0, 0, W, H);
      if (this._landmarks) {
        this._displayLandmarks = this._smoothLandmarks(this._displayLandmarks, this._landmarks);
        this._drawLandmarks(this._lmCtx, W, H, this._displayLandmarks);
      }
      requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  }

  // ── Landmark smoothing (adaptive exponential blend) ───────────────────────────
  // Technique from LearnOpenCV: alpha = exp(-d²/sigma) so small displacements
  // heavily favour the smoothed (stable) position, large displacements favour
  // the fresh detection (responsive). Applied per-landmark per rAF tick.

  _smoothLandmarks(cur, tgt) {
    if (!cur) return tgt;

    const smoothPt = (c, t, sigma) => {
      if (!c) return t;
      const dx = t.x - c.x, dy = t.y - c.y;
      const alpha = Math.exp(-(dx * dx + dy * dy) / sigma);
      return {
        ...t,
        x: t.x + alpha * (c.x - t.x),
        y: t.y + alpha * (c.y - t.y),
        z: t.z + alpha * (c.z - t.z),
      };
    };

    const out = { type: "landmarks" };
    if (tgt.pose)  out.pose  = tgt.pose.map((p, i) => smoothPt(cur.pose?.[i],      p, POSE_SIGMA[i] ?? 5e-4));
    if (tgt.hands) out.hands = tgt.hands.map((h, hi) =>
                                 h.map((p, i) => smoothPt(cur.hands?.[hi]?.[i],   p, HAND_SIGMA)));
    if (tgt.face)  out.face  = tgt.face.map((p, i) => smoothPt(cur.face?.[i],      p, FACE_SIGMA));
    return out;
  }

  // ── Panel 3: server-composited frame ──────────────────────────────────────────

  _renderOutputFrame(buf) {
    const sentAt = this._frameSentAt;
    const blob = new Blob([buf], { type: "image/jpeg" });
    createImageBitmap(blob).then((bmp) => {
      this._outCtx.drawImage(bmp, 0, 0, this._outCanvas.width, this._outCanvas.height);
      bmp.close();
      this._countFps();
      if (sentAt !== null) {
        const lat = Date.now() - sentAt;
        this._latencyEma = this._latencyEma === null
          ? lat
          : this._latencyEma * 0.75 + lat * 0.25;
        document.getElementById("latency-display").textContent =
          `${Math.round(this._latencyEma)} ms`;
      }
    });
  }

  // ── Landmark drawing ──────────────────────────────────────────────────────────

  _drawLandmarks(ctx, W, H, lms) {

    // Pose — green
    if (lms.pose?.length) {
      ctx.strokeStyle = "#22c55e";
      ctx.lineWidth   = 2;
      for (const [a, b] of POSE_CONNECTIONS) {
        const la = lms.pose[a], lb = lms.pose[b];
        if (la && lb && la.v > 0.3 && lb.v > 0.3) {
          ctx.beginPath();
          ctx.moveTo(la.x * W, la.y * H);
          ctx.lineTo(lb.x * W, lb.y * H);
          ctx.stroke();
        }
      }
      ctx.fillStyle = "#16a34a";
      for (const lm of lms.pose) {
        if (lm.v > 0.3) {
          ctx.beginPath();
          ctx.arc(lm.x * W, lm.y * H, 3.5, 0, Math.PI * 2);
          ctx.fill();
        }
      }
    }

    // Hands — orange bones, blue knuckles
    if (lms.hands?.length) {
      for (const hand of lms.hands) {
        ctx.strokeStyle = "#f97316";
        ctx.lineWidth   = 2;
        for (const [a, b] of HAND_CONNECTIONS) {
          if (a < hand.length && b < hand.length) {
            ctx.beginPath();
            ctx.moveTo(hand[a].x * W, hand[a].y * H);
            ctx.lineTo(hand[b].x * W, hand[b].y * H);
            ctx.stroke();
          }
        }
        ctx.fillStyle = "#3b82f6";
        for (const lm of hand) {
          ctx.beginPath();
          ctx.arc(lm.x * W, lm.y * H, 3.5, 0, Math.PI * 2);
          ctx.fill();
        }
      }
    }

    // Face — oval + inner contours + landmark dots
    if (lms.face?.length) {
      const f = lms.face;
      ctx.strokeStyle = "rgba(126,200,227,0.85)";
      ctx.lineWidth   = 2;

      const drawPath = (indices, closed) => {
        ctx.beginPath();
        let first = true;
        for (const i of indices) {
          if (i >= f.length) continue;
          const p = f[i];
          if (first) { ctx.moveTo(p.x * W, p.y * H); first = false; }
          else          ctx.lineTo(p.x * W, p.y * H);
        }
        if (closed) ctx.closePath();
        ctx.stroke();
      };

      drawPath(FACE_OVAL_IDX,   true);
      drawPath(FACE_RIGHT_EYE,  true);
      drawPath(FACE_LEFT_EYE,   true);
      drawPath(FACE_RIGHT_BROW, false);
      drawPath(FACE_LEFT_BROW,  false);
      drawPath(FACE_NOSE_RIDGE, false);
      drawPath(FACE_LIPS_OUTER, true);

      // Small dot at every landmark so the full mesh is visible
      ctx.fillStyle = "rgba(126,200,227,0.7)";
      for (const lm of f) {
        ctx.beginPath();
        ctx.arc(lm.x * W, lm.y * H, 2, 0, Math.PI * 2);
        ctx.fill();
      }
    }
  }

  // ── FPS counter ───────────────────────────────────────────────────────────────

  _countFps() {
    this._frameCount++;
    const now = Date.now(), elapsed = now - this._fpsLastTime;
    if (elapsed >= 1000) {
      document.getElementById("fps-display").textContent =
        `${Math.round(this._frameCount * 1000 / elapsed)} fps`;
      this._frameCount  = 0;
      this._fpsLastTime = now;
    }
  }

  // ── Helpers ───────────────────────────────────────────────────────────────────

  _send(obj) {
    if (this._ws?.readyState === WebSocket.OPEN) this._ws.send(JSON.stringify(obj));
  }

  _setStatus(text, connected) {
    document.getElementById("status-text").textContent = text;
    document.getElementById("status-dot").classList.toggle("connected", connected);
  }

  _setBtn(state) {
    const btn = document.getElementById("start-stop-btn");
    btn.textContent = state === "stop" ? "Stop" : "Start";
    btn.className   = state;
  }

  // ── Public API (called from event handlers) ───────────────────────────────────

  setPreset(preset) {
    this._settings.preset = preset;
    this._send({ type: "preset", preset });
    // Toggle active on regular preset buttons
    document.querySelectorAll(".preset-btn[data-preset]").forEach((btn) => {
      btn.classList.toggle("active", btn.dataset.preset === preset);
    });
    // Toggle active on the Image label separately
    document.getElementById("img-preset-btn").classList.toggle("active", preset === "image");
  }

  toggleFeature(feature) {
    this._settings[feature] = !this._settings[feature];
    this._send({ type: "toggle", feature, value: this._settings[feature] });
    document.getElementById(`chip-${feature}`).classList.toggle("on", this._settings[feature]);
  }

  setFpsCap(fps) {
    this._settings.fpsCap = fps;
    this._send({ type: "fps_cap", fps });
    document.querySelectorAll(".fps-btn").forEach((btn) => {
      btn.classList.toggle("active", parseInt(btn.dataset.fps) === fps);
    });
    // Restart capture timer at new rate if running
    if (this._captureTimer) {
      clearInterval(this._captureTimer);
      this._captureTimer = null;
      this._startCapture();
    }
  }

  setOutlineStrength(val) {
    const v = parseFloat(val);
    this._settings.outlineStrength = v;
    this._send({ type: "outline_strength", value: v });
    document.getElementById("strength-val").textContent = v.toFixed(2);
  }

  uploadBgImage(file) {
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (e) => {
      this._send({ type: "bg_image", data: e.target.result });
      this.setPreset("image");
    };
    reader.readAsDataURL(file);
  }

  // ── UI binding ────────────────────────────────────────────────────────────────

  _bindUI() {
    // Start / Stop button
    document.getElementById("start-stop-btn").addEventListener("click", () => {
      this._running ? this.stop() : this.start();
    });

    // Preset buttons (data-preset delegates; Image label handles itself via file input)
    document.getElementById("preset-row").addEventListener("click", (e) => {
      const btn = e.target.closest(".preset-btn[data-preset]");
      if (btn) this.setPreset(btn.dataset.preset);
    });

    // Background image file picker
    document.getElementById("bg-file").addEventListener("change", (e) => {
      this.uploadBgImage(e.target.files?.[0]);
    });

    // FPS cap buttons
    document.getElementById("fps-row").addEventListener("click", (e) => {
      const btn = e.target.closest(".fps-btn[data-fps]");
      if (btn) this.setFpsCap(parseInt(btn.dataset.fps));
    });

    // Feature toggles
    // e.preventDefault() stops the browser from activating the wrapped checkbox,
    // which would re-fire a second click event and immediately undo the toggle.
    ["pose", "hand", "face", "outline"].forEach((feat) => {
      document.getElementById(`chip-${feat}`).addEventListener("click", (e) => {
        e.preventDefault();
        this.toggleFeature(feat);
      });
    });

    // Outline strength slider
    const slider = document.getElementById("outline-strength");
    slider.addEventListener("input", () => this.setOutlineStrength(slider.value));
  }
}

window.addEventListener("DOMContentLoaded", () => {
  window.app = new App();
});
