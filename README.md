# MediaPipe Background & Landmarks

Real-time background removal and body landmark overlay running entirely in the browser via MediaPipe Tasks Vision JS (WebAssembly + WebGL). The server only serves static files — all inference runs on the user's local device CPU/GPU.

## Three-panel view

| Panel | What it shows |
|-------|---------------|
| **1 — Original** | Raw camera feed, no processing |
| **2 — Landmarks** | Camera frame with pose skeleton, hand bones, and face mesh drawn live at ~60 fps with per-body-part adaptive smoothing |
| **3 — Background + Crop** | Background replaced client-side using the segmentation mask; optional glow outline around the person |

## Features

- **Background removal** — selfie segmentation mask composited client-side on canvas
- **Background presets** — None, Blur, Black, White, Green Screen, or Image (upload any photo)
- **Full face mesh** — 468 landmark dots + face oval, eye contours, eyebrows, nose ridge, and lip outline
- **Glow outline** — blurred silhouette layer composited before the sharp person cutout; adjustable strength slider
- **Per-feature toggles** — enable/disable Pose, Hand, Face, and Glow independently
- **Pipeline FPS control** — 5 / 10 / 15 / 20 / 30 / MAX
- **Inference time display** — ms per inference cycle shown in the status bar
- **No server CPU used** — backend is a static file server; all MediaPipe runs in the browser

## How it works

```
Browser (user's machine)
  ↓ webcam frame (stays in browser — never sent to server)
  │
  ├─ ImageSegmenter    → confidence mask (Float32Array W×H)
  ├─ PoseLandmarker   → 33 normalized landmarks
  ├─ HandLandmarker   → 21 landmarks × 2 hands
  └─ FaceLandmarker   → 478 face landmarks
         │
         ├─ Panel 2: video + per-body-part smoothed skeleton drawn via rAF at 60 fps
         └─ Panel 3: canvas composite — background layer + masked person + optional glow

FastAPI server
  └─ serves index.html, app.js, style.css, background images
     (no WebSocket, no inference, no mediapipe installed)
```

Models are loaded from Google's MediaPipe CDN on the first **Start** click, downloaded once by the browser, then cached. GPU delegate (WebGL) is used automatically where available.

## Landmark smoothing

Per-body-part adaptive exponential blend applied every rAF tick:

```
α = exp(−d² / σ)
```

Small displacement → favour the stable smoothed position. Large displacement → favour the fresh detection. Sigma is tuned per landmark group:

| Body part | Sigma | Behaviour |
|-----------|-------|-----------|
| Feet (29–32) | `1.5e-4` | Snappiest |
| Wrists / hand tips (15–22) | `2e-4` | Very responsive |
| Elbows / ankles (13–14, 27–28) | `3e-4` | Responsive |
| Knees (25–26) | `5e-4` | Medium |
| Shoulders / hips (11–12, 23–24) | `7e-4` | Stable |
| Face / head (0–10) | `8e-4` | Most stable |
| Hands | `1.5e-4` | Very snappy |
| Face mesh | `8e-4` | Stable |

Panel 2 redraws at 60 fps regardless of inference rate, so motion stays smooth at any pipeline FPS setting.

## Project structure

```
.
├── backend/
│   ├── app/
│   │   └── main.py        # FastAPI — serves static files only
│   └── requirements.txt   # fastapi + uvicorn only (no mediapipe/opencv)
├── frontend/
│   ├── index.html          # 3-panel layout + controls
│   ├── css/style.css
│   └── js/app.js           # all inference, compositing, and drawing (ES module)
├── Dockerfile              # python:3.11-slim-bookworm + GL libs for Render
├── render.yaml             # Render free tier deploy config (Docker runtime)
└── TROUBLESHOOTING.txt     # port conflicts, kill commands, caching fixes
```

## Running locally

**Requirements:** Python 3.9+, a webcam, Chrome or Edge (best WebGL support)

```powershell
# Clone
git clone https://github.com/AngeloJamesRomerosa/Mediapipe_Segmenter_Landmarker.git
cd Mediapipe_Segmenter_Landmarker

# Install (only fastapi + uvicorn — no mediapipe needed)
pip install -r backend/requirements.txt

# Start
uvicorn backend.app.main:app --reload
```

Open `http://localhost:8000` and click **Start**. On first run the browser downloads the MediaPipe WASM runtime and model files from Google's CDN (~30 MB total). Subsequent starts are instant.

### Stopping the server

Press **Ctrl+C**, or if the process is stuck:

```powershell
Get-Process -Name "python*" -ErrorAction SilentlyContinue | Stop-Process -Force
```

## Deploying to Render

1. Push this repo to GitHub.
2. On [render.com](https://render.com) → **New → Web Service** → connect the repo → set Language to **Docker**.
3. Render picks up `render.yaml` automatically.
4. The Docker build installs the GL libraries MediaPipe's WASM runtime needs. Since inference is client-side, the server itself uses near-zero CPU — the free tier is sufficient.

> **Free tier note:** The service sleeps after 15 minutes of inactivity. First request after sleep takes ~20 s to wake up. No model downloads happen on the server — models load in the browser on first use.

## Models used (loaded in browser from Google CDN)

| Model | Task | Size |
|-------|------|------|
| `selfie_segmenter.tflite` | Person alpha mask | ~1 MB |
| `pose_landmarker_lite.task` | 33 body landmarks | ~5 MB |
| `hand_landmarker.task` | 21 landmarks × 2 hands | ~9 MB |
| `face_landmarker.task` | 478 face landmarks | ~14 MB |

All four load in parallel via `Promise.all` when you click **Start** for the first time.
