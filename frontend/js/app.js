"use strict";

import {
  FilesetResolver,
  ImageSegmenter,
  PoseLandmarker,
  HandLandmarker,
  FaceLandmarker,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";

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
const FACE_RIGHT_EYE  = [33,246,161,160,159,158,157,173,133,155,154,153,145,144,163,7];
const FACE_LEFT_EYE   = [362,382,381,380,374,373,390,249,263,466,388,387,386,385,384,398];
const FACE_RIGHT_BROW = [46,53,52,65,55,70,63,105,66,107];
const FACE_LEFT_BROW  = [276,283,282,295,285,300,293,334,296,336];
const FACE_NOSE_RIDGE = [168,6,197,195,5,4];
const FACE_LIPS_OUTER = [61,146,91,181,84,17,314,405,321,375,291,308,324,318,402,317,14,87,178,88,95];

// ── Per-landmark smoothing sigmas ─────────────────────────────────────────────

const POSE_SIGMA = (() => {
  const s = new Float32Array(33).fill(5e-4);
  for (let i = 0; i <= 10; i++) s[i] = 8e-4;
  s[11] = s[12] = 7e-4;
  s[13] = s[14] = 3e-4;
  for (let i = 15; i <= 22; i++) s[i] = 2e-4;
  s[23] = s[24] = 7e-4;
  s[25] = s[26] = 5e-4;
  s[27] = s[28] = 3e-4;
  for (let i = 29; i <= 32; i++) s[i] = 1.5e-4;
  return s;
})();
const HAND_SIGMA = 1.5e-4;
const FACE_SIGMA = 8e-4;

// ── Model CDN paths ───────────────────────────────────────────────────────────

const MP_BASE   = "https://storage.googleapis.com/mediapipe-models";
const WASM_PATH = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm";
const SEG_MODEL  = `${MP_BASE}/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite`;
const POSE_MODEL = `${MP_BASE}/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task`;
const HAND_MODEL = `${MP_BASE}/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task`;
const FACE_MODEL = `${MP_BASE}/face_landmarker/face_landmarker/float16/latest/face_landmarker.task`;

// ── App ───────────────────────────────────────────────────────────────────────

class App {
  constructor() {
    this._video     = document.getElementById("camera");
    this._lmCanvas  = document.getElementById("landmarks");
    this._lmCtx     = this._lmCanvas.getContext("2d");
    this._outCanvas = document.getElementById("output");
    this._outCtx    = this._outCanvas.getContext("2d");

    // Off-screen canvas for mask compositing — willReadFrequently for getImageData perf
    this._maskCanvas        = document.createElement("canvas");
    this._maskCtx           = this._maskCanvas.getContext("2d", { willReadFrequently: true });

    this._running     = false;
    this._modelsReady = false;
    this._loopActive  = false;

    // MediaPipe models
    this._segmenter      = null;
    this._poseLandmarker = null;
    this._handLandmarker = null;
    this._faceLandmarker = null;

    // Latest segmentation mask data (Float32Array, length = W*H)
    this._latestMask = null;

    // Landmark targets + smoothed display state
    this._landmarks        = null;
    this._displayLandmarks = null;

    // Background image (loaded from file picker)
    this._bgImage = null;

    // FPS counter
    this._frameCount  = 0;
    this._fpsLastTime = Date.now();

    this._settings = {
      preset:          "none",
      pose:            true,
      hand:            true,
      face:            false,
      outline:         false,
      outlineStrength: 0.5,
      fpsCap:          30,
    };

    this._bindUI();
  }

  // ── Start / Stop ──────────────────────────────────────────────────────────

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

    if (this._video.srcObject) {
      this._video.srcObject.getTracks().forEach(t => t.stop());
      this._video.srcObject = null;
    }

    this._lmCtx.clearRect(0, 0, this._lmCanvas.width, this._lmCanvas.height);
    this._outCtx.clearRect(0, 0, this._outCanvas.width, this._outCanvas.height);
    this._landmarks        = null;
    this._displayLandmarks = null;
    this._latestMask       = null;
  }

  // ── Camera init ───────────────────────────────────────────────────────────

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

      this._lmCanvas.width   = W;  this._lmCanvas.height  = H;
      this._outCanvas.width  = W;  this._outCanvas.height = H;
      this._maskCanvas.width = W;  this._maskCanvas.height = H;

      if (!this._modelsReady) await this._initModels();
      if (!this._running) return;

      this._startLoop();
    } catch (err) {
      this._running = false;
      this._setBtn("start");
      this._setStatus(`Camera error: ${err.message}`, false);
    }
  }

  // ── Model initialization ──────────────────────────────────────────────────

  async _initModels() {
    this._setStatus("Loading models…", false);
    try {
      const vision = await FilesetResolver.forVisionTasks(WASM_PATH);

      [this._segmenter, this._poseLandmarker, this._handLandmarker, this._faceLandmarker] =
        await Promise.all([
          ImageSegmenter.createFromOptions(vision, {
            baseOptions: { modelAssetPath: SEG_MODEL,  delegate: "GPU" },
            runningMode: "VIDEO",
            outputConfidenceMasks: true,
          }),
          PoseLandmarker.createFromOptions(vision, {
            baseOptions: { modelAssetPath: POSE_MODEL, delegate: "GPU" },
            runningMode: "VIDEO",
            numPoses: 1,
            minPoseDetectionConfidence: 0.5,
            minTrackingConfidence: 0.5,
          }),
          HandLandmarker.createFromOptions(vision, {
            baseOptions: { modelAssetPath: HAND_MODEL, delegate: "GPU" },
            runningMode: "VIDEO",
            numHands: 2,
            minHandDetectionConfidence: 0.5,
            minTrackingConfidence: 0.5,
          }),
          FaceLandmarker.createFromOptions(vision, {
            baseOptions: { modelAssetPath: FACE_MODEL, delegate: "GPU" },
            runningMode: "VIDEO",
            numFaces: 1,
            minFaceDetectionConfidence: 0.5,
            minTrackingConfidence: 0.5,
          }),
        ]);

      this._modelsReady = true;
      this._setStatus("Ready", true);
    } catch (err) {
      this._setStatus(`Model load failed: ${err.message}`, false);
      throw err;
    }
  }

  // ── Main loop ─────────────────────────────────────────────────────────────
  // Panel 2 (landmarks) redraws every rAF tick for smooth display.
  // Inference + Panel 3 compositing runs at the capped FPS.

  _startLoop() {
    if (this._loopActive) return;
    this._loopActive = true;
    let lastInferTime = 0;

    const tick = (now) => {
      if (!this._running) { this._loopActive = false; return; }

      const W = this._lmCanvas.width;
      const H = this._lmCanvas.height;

      // Panel 2: video frame + smoothed landmarks every rAF tick
      this._lmCtx.drawImage(this._video, 0, 0, W, H);
      if (this._landmarks) {
        this._displayLandmarks = this._smoothLandmarks(this._displayLandmarks, this._landmarks);
        this._drawLandmarks(this._lmCtx, W, H, this._displayLandmarks);
      }

      // Inference + Panel 3 at capped FPS
      const interval = this._settings.fpsCap === 0 ? 0 : 1000 / this._settings.fpsCap;
      if (this._modelsReady && this._video.readyState >= 2 && now - lastInferTime >= interval) {
        lastInferTime = now;
        const t0 = performance.now();
        this._runInference(now);
        const inferMs = Math.round(performance.now() - t0);
        document.getElementById("latency-display").textContent = `${inferMs}ms`;
        this._countFps();
      }

      requestAnimationFrame(tick);
    };

    requestAnimationFrame(tick);
  }

  // ── Inference (runs all models synchronously on each inference tick) ───────

  _runInference(timestamp) {
    const needsSeg = this._settings.preset !== "none" || this._settings.outline;

    // Segmenter
    if (needsSeg) {
      const r    = this._segmenter.segmentForVideo(this._video, timestamp);
      const mask = r.confidenceMasks?.[0];
      // Copy Float32Array — the underlying buffer may be reused on next inference call
      this._latestMask = mask ? new Float32Array(mask.getAsFloat32Array()) : null;
    } else {
      this._latestMask = null;
    }

    // Pose
    const pose = this._settings.pose
      ? (this._poseLandmarker.detectForVideo(this._video, timestamp).landmarks?.[0] ?? null)
      : null;

    // Hand
    const hands = this._settings.hand
      ? (this._handLandmarker.detectForVideo(this._video, timestamp).landmarks ?? [])
      : [];

    // Face
    const face = this._settings.face
      ? (this._faceLandmarker.detectForVideo(this._video, timestamp).faceLandmarks?.[0] ?? null)
      : null;

    this._landmarks = { pose, hands, face };

    // Panel 3: composite background using segmentation mask
    this._compositeBg();
  }

  // ── Background compositing (Panel 3) ──────────────────────────────────────

  _compositeBg() {
    const W      = this._outCanvas.width;
    const H      = this._outCanvas.height;
    const ctx    = this._outCtx;
    const preset = this._settings.preset;

    // No background effect — just mirror the raw video
    if (preset === "none" && !this._settings.outline) {
      ctx.drawImage(this._video, 0, 0, W, H);
      return;
    }

    // Draw background layer
    if (preset === "blur") {
      ctx.filter = "blur(20px) saturate(1.3)";
      ctx.drawImage(this._video, -20, -20, W + 40, H + 40);
      ctx.filter = "none";
    } else if (preset === "black") {
      ctx.fillStyle = "#000";
      ctx.fillRect(0, 0, W, H);
    } else if (preset === "white") {
      ctx.fillStyle = "#fff";
      ctx.fillRect(0, 0, W, H);
    } else if (preset === "green") {
      ctx.fillStyle = "#00b140";
      ctx.fillRect(0, 0, W, H);
    } else if (preset === "image" && this._bgImage) {
      ctx.drawImage(this._bgImage, 0, 0, W, H);
    } else {
      // Fallback: no mask yet, show plain video
      ctx.drawImage(this._video, 0, 0, W, H);
      return;
    }

    const mask = this._latestMask;
    if (!mask || mask.length !== W * H) return;

    // Draw current video frame to mask canvas, then set alpha from segmentation mask
    this._maskCtx.drawImage(this._video, 0, 0, W, H);
    const imgData = this._maskCtx.getImageData(0, 0, W, H);
    const px = imgData.data;
    for (let i = 0; i < W * H; i++) {
      px[i * 4 + 3] = mask[i] * 255 | 0;
    }
    this._maskCtx.putImageData(imgData, 0, 0);

    // Optional glow outline — draw blurred person silhouette before sharp composite
    if (this._settings.outline) {
      const blur = Math.round(this._settings.outlineStrength * 16 + 4);
      ctx.save();
      ctx.filter      = `blur(${blur}px)`;
      ctx.globalAlpha = this._settings.outlineStrength * 0.9;
      ctx.drawImage(this._maskCanvas, 0, 0);
      ctx.restore();
    }

    // Composite sharp person over background
    ctx.drawImage(this._maskCanvas, 0, 0);
  }

  // ── Landmark smoothing ────────────────────────────────────────────────────

  _smoothLandmarks(cur, tgt) {
    if (!cur) return tgt;

    const smoothPt = (c, t, sigma) => {
      if (!c) return t;
      const dx = t.x - c.x, dy = t.y - c.y;
      const alpha = Math.exp(-(dx * dx + dy * dy) / sigma);
      return { ...t, x: t.x + alpha * (c.x - t.x), y: t.y + alpha * (c.y - t.y), z: t.z + alpha * (c.z - t.z) };
    };

    const out = {};
    if (tgt.pose)  out.pose  = tgt.pose.map((p, i) => smoothPt(cur.pose?.[i],       p, POSE_SIGMA[i] ?? 5e-4));
    if (tgt.hands) out.hands = tgt.hands.map((h, hi) =>
                                 h.map((p, i)  => smoothPt(cur.hands?.[hi]?.[i],    p, HAND_SIGMA)));
    if (tgt.face)  out.face  = tgt.face.map((p, i) => smoothPt(cur.face?.[i],       p, FACE_SIGMA));
    return out;
  }

  // ── Landmark drawing ──────────────────────────────────────────────────────
  // MediaPipe JS uses lm.visibility (float 0-1). Falls back to 1 if absent (hands/face).

  _drawLandmarks(ctx, W, H, lms) {

    // Pose — green
    if (lms.pose?.length) {
      ctx.strokeStyle = "#22c55e";
      ctx.lineWidth   = 2;
      for (const [a, b] of POSE_CONNECTIONS) {
        const la = lms.pose[a], lb = lms.pose[b];
        if (la && lb && (la.visibility ?? 1) > 0.3 && (lb.visibility ?? 1) > 0.3) {
          ctx.beginPath();
          ctx.moveTo(la.x * W, la.y * H);
          ctx.lineTo(lb.x * W, lb.y * H);
          ctx.stroke();
        }
      }
      ctx.fillStyle = "#16a34a";
      for (const lm of lms.pose) {
        if ((lm.visibility ?? 1) > 0.3) {
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

    // Face — oval + inner contours + mesh dots
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

      ctx.fillStyle = "rgba(126,200,227,0.7)";
      for (const lm of f) {
        ctx.beginPath();
        ctx.arc(lm.x * W, lm.y * H, 2, 0, Math.PI * 2);
        ctx.fill();
      }
    }
  }

  // ── FPS counter ───────────────────────────────────────────────────────────

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

  // ── Helpers ───────────────────────────────────────────────────────────────

  _setStatus(text, ready) {
    document.getElementById("status-text").textContent = text;
    document.getElementById("status-dot").classList.toggle("connected", ready);
  }

  _setBtn(state) {
    const btn = document.getElementById("start-stop-btn");
    btn.textContent = state === "stop" ? "Stop" : "Start";
    btn.className   = state;
  }

  // ── Public API ────────────────────────────────────────────────────────────

  setPreset(preset) {
    this._settings.preset = preset;
    if (preset !== "image") this._bgImage = null;
    document.querySelectorAll(".preset-btn[data-preset]").forEach(btn => {
      btn.classList.toggle("active", btn.dataset.preset === preset);
    });
    document.getElementById("img-preset-btn").classList.toggle("active", preset === "image");
  }

  toggleFeature(feature) {
    this._settings[feature] = !this._settings[feature];
    document.getElementById(`chip-${feature}`).classList.toggle("on", this._settings[feature]);
  }

  setFpsCap(fps) {
    this._settings.fpsCap = fps;
    document.querySelectorAll(".fps-btn").forEach(btn => {
      btn.classList.toggle("active", parseInt(btn.dataset.fps) === fps);
    });
  }

  setOutlineStrength(val) {
    const v = parseFloat(val);
    this._settings.outlineStrength = v;
    document.getElementById("strength-val").textContent = v.toFixed(2);
  }

  uploadBgImage(file) {
    if (!file) return;
    const url = URL.createObjectURL(file);
    const img = new Image();
    img.onload = () => {
      this._bgImage = img;
      URL.revokeObjectURL(url);
      this.setPreset("image");
    };
    img.src = url;
  }

  // ── UI binding ────────────────────────────────────────────────────────────

  _bindUI() {
    document.getElementById("start-stop-btn").addEventListener("click", () => {
      this._running ? this.stop() : this.start();
    });

    document.getElementById("preset-row").addEventListener("click", (e) => {
      const btn = e.target.closest(".preset-btn[data-preset]");
      if (btn) this.setPreset(btn.dataset.preset);
    });

    document.getElementById("bg-file").addEventListener("change", (e) => {
      this.uploadBgImage(e.target.files?.[0]);
    });

    document.getElementById("fps-row").addEventListener("click", (e) => {
      const btn = e.target.closest(".fps-btn[data-fps]");
      if (btn) this.setFpsCap(parseInt(btn.dataset.fps));
    });

    ["pose", "hand", "face", "outline"].forEach((feat) => {
      document.getElementById(`chip-${feat}`).addEventListener("click", (e) => {
        e.preventDefault();
        this.toggleFeature(feat);
      });
    });

    const slider = document.getElementById("outline-strength");
    slider.addEventListener("input", () => this.setOutlineStrength(slider.value));
  }
}

window.addEventListener("DOMContentLoaded", () => {
  window.app = new App();
});
