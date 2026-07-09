# MediaPipe Background & Landmarks

Real-time background removal and body landmark overlay using MediaPipe, served as a single FastAPI web service. Designed to run on Render's free tier.

## Three-panel view

| Panel | What it shows |
|-------|---------------|
| **1 — Original** | Raw camera feed, no processing |
| **2 — Landmarks** | Camera frame with pose skeleton, hand bones, and full face mesh (oval + eyes + eyebrows + nose + lips + 468 dots) drawn live at ~60 fps with client-side adaptive smoothing |
| **3 — Background + Crop** | Server-composited frame — background replaced, with pose/hand/face landmark masks used to sharpen the person cutout; optional cyan glow outline around the person |

## Features

- **Background removal** — selfie segmentation composited with pose, hand, and face alpha masks
- **Background presets** — None, Blur, Black, White, Green Screen, or Image (click to upload any photo)
- **Full face mesh** — 468 landmark dots + face oval, eye contours, eyebrows, nose ridge, and lip outline
- **Glow outline** — additive cyan glow around the person cutout; works with or without a background preset; adjustable strength slider
- **Landmark overlay** — Panel 2 shows skeleton/mesh at display frame rate with smoothing; Panel 3 shows the composited result
- **Per-feature toggles** — enable/disable Pose, Hand, Face, and Glow independently
- **Pipeline FPS control** — choose 5 / 10 / 15 / 20 / 30 / MAX to cap the entire capture + inference pipeline
- **Live latency display** — end-to-end round-trip time shown in the status bar alongside FPS
- **Single service** — backend serves the frontend as static files; no separate CDN needed

## How it works

```
Browser camera
    ↓ JPEG frames at selected pipeline FPS (WebSocket binary)
FastAPI /ws
    ↓
4 daemon threads per session (VIDEO mode, capped at pipeline FPS):
  seg_loop   → selfie segmenter alpha mask           (IMAGE mode, shared)
  pose_loop  → 33 body landmarks + zone masks        (VIDEO mode, per-session)
               + Lucas-Kanade optical flow for low-visibility joints
  hand_loop  → 21 landmarks × 2 hands + finger masks (VIDEO mode, per-session)
  face_loop  → 468 face landmarks + face oval mask   (VIDEO mode, per-session)
    ↓ each thread pushes landmark JSON directly to browser via asyncio (no frame wait)
apply_bg(): merge all alphas → morph close → composite with background
    ↓ processed JPEG (binary) back to browser

Panel 1 — raw <video>  (no server involved)
Panel 2 — camera frame + landmark JSON smoothed and drawn client-side via rAF at 60 fps
Panel 3 — composited JPEG from server
```

Models are downloaded automatically on first use from Google's MediaPipe CDN and cached in `models/` for the lifetime of the process.

## Landmark smoothing — three layers

Jitter and lag are reduced at three independent levels:

| Layer | Where | What it does |
|-------|-------|-------------|
| **VIDEO mode** | MediaPipe model (server) | Temporal tracking between frames — model reuses previous detection instead of re-detecting every frame; built-in landmark smoothing |
| **Optical flow** | `loops.py` (server) | Lucas-Kanade flow for low-visibility pose joints (e.g. feet, wrists) to keep them stable when partially occluded |
| **Adaptive exp blend** | `app.js` (browser) | Per-landmark `α = exp(−d²/σ)` applied every rAF tick — tiny movements heavily favour the smoothed position (stable), large movements snap to fresh detection (responsive) |

The client-side blend runs at 60 fps regardless of server update rate, so motion looks smooth even at low pipeline FPS settings.

## Threading model

```
asyncio event loop (main thread)
  │  receives frame binary
  └─► run_in_executor ──► _process_frame (frame worker thread pool)
                                │  apply_bg() → loops.submit(frame)
                                │                    │
                                │             4 daemon threads
                                │             (inference runs here, VIDEO mode)
                                │                    │
                                │             _push_landmarks()
                                │             asyncio.run_coroutine_threadsafe
                                │             → sends JSON directly to browser
                                │             as soon as inference finishes
                                │
                                └─► ws.send_bytes(composited JPEG)
```

Landmark JSON is pushed **immediately** when each inference thread finishes — it does not wait for the next composited frame response. This minimises the visible lag between body movement and skeleton movement.

## Project structure

```
.
├── backend/
│   ├── app/
│   │   ├── main.py        # FastAPI app, WebSocket endpoint
│   │   ├── composite.py   # apply_bg() — alpha merge and pixel composite
│   │   ├── loops.py       # per-session daemon inference threads (VIDEO mode)
│   │   ├── infer.py       # per-model inference wrappers (detect_for_video)
│   │   ├── alpha.py       # mask builders (hand, face, segmenter)
│   │   ├── zones.py       # 8 body-part zone masks (parallel)
│   │   ├── build.py       # background layer builder
│   │   ├── session.py     # per-connection state dataclass
│   │   ├── models.py      # segmenter singleton + per-session VIDEO-mode factories
│   │   └── config.py      # model URLs/paths, landmark tables, constants
│   └── requirements.txt
├── frontend/
│   ├── index.html          # 3-panel layout + FPS cap + latency controls
│   ├── css/style.css
│   └── js/app.js           # camera capture, WS comms, adaptive landmark smoothing
└── render.yaml             # Render free tier deploy config
                            # models/ is created at runtime and should be gitignored
```

## Running locally

### Requirements

- **Python 3.11 or 3.12** (MediaPipe does not yet support Python 3.13)
- A webcam
- A modern browser (Chrome or Edge recommended for best WebRTC support)

### Step-by-step setup

**1. Clone the repo**

```bash
git clone <your-repo-url>
cd Mediapipe_Background_Change_Landmarks
```

**2. Create and activate a virtual environment**

```bash
# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# macOS / Linux
python -m venv .venv
source .venv/bin/activate
```

> If PowerShell blocks the activation script, run this once first:
> `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

**3. Install dependencies**

```bash
pip install -r backend/requirements.txt
```

This installs FastAPI, MediaPipe, OpenCV, uvicorn, and everything else needed. It may take a minute.

**4. Start the server**

```bash
uvicorn backend.app.main:app --reload
```

You should see output like:

```
INFO     MediaPipe BG server ready.
INFO     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```

**5. Open in browser**

Go to `http://localhost:8000`, then click **Start** and allow camera access when prompted.

The first time each model is used it downloads from Google's MediaPipe CDN (~30 MB total) and caches in a local `models/` folder. After that, startup is instant.

### Stopping the server

Press **Ctrl+C** in the terminal.

### Model download notes

| Model | Running mode | Downloaded when | Size |
|-------|-------------|-----------------|------|
| Selfie segmenter | IMAGE (shared singleton) | First camera frame | ~1 MB |
| Pose landmarker | VIDEO (per-session) | Session start | ~5 MB |
| Hand landmarker | VIDEO (per-session) | Session start | ~9 MB |
| Face landmarker | VIDEO (per-session) | Session start (pre-warms) | ~14 MB |

Pose, Hand, and Face landmarkers use **VIDEO mode** — each session creates its own model instance so the temporal tracking state is not shared between users. The face model pre-warms in the background as soon as the camera starts.

## Deploying to Render

1. Push this repo to GitHub.
2. On [render.com](https://render.com) → **New → Web Service** → connect the repo.
3. Render picks up `render.yaml` automatically — no manual config needed.
4. First deploy takes a minute to install mediapipe. Cold starts re-download models (~30 MB, takes ~20 s).

> **Free tier note:** The service sleeps after 15 minutes of inactivity. The first request after sleep takes ~30 s to wake up and re-download models. There is no persistent disk — everything is in-memory per session.

## WebSocket protocol

All communication happens over a single `ws://<host>/ws` connection per browser tab.

| Direction | Type | Description |
|-----------|------|-------------|
| Client → Server | binary | Raw JPEG camera frame |
| Client → Server | text JSON | Config message (see below) |
| Server → Client | binary | Processed JPEG frame |
| Server → Client | text JSON | `{"type":"landmarks", pose:[...], hands:[...], face:[...]}` pushed directly by inference threads |

### Config messages

```jsonc
// Set background preset
{"type": "preset", "preset": "none|blur|black|white|green|image"}

// Toggle a feature
{"type": "toggle", "feature": "pose|hand|face|outline", "value": true}

// Outline glow strength (0.0 – 1.0)
{"type": "outline_strength", "value": 0.5}

// Pipeline FPS cap (0 = MAX / uncapped)
{"type": "fps_cap", "fps": 30}

// Upload a custom background image (data-URL or raw base64)
{"type": "bg_image", "data": "data:image/jpeg;base64,..."}
```

## Models used

| Model | Task | Mode | Size |
|-------|------|------|------|
| `selfie_segmenter.tflite` | Person alpha mask | IMAGE — shared singleton | ~1 MB |
| `pose_landmarker_lite.task` | 33 body landmarks + segmentation | VIDEO — per-session | ~5 MB |
| `hand_landmarker.task` | 21 landmarks × 2 hands | VIDEO — per-session | ~9 MB |
| `face_landmarker.task` | 468 face landmarks | VIDEO — per-session | ~14 MB |

## Configuration

| Constant | File | Default | Description |
|----------|------|---------|-------------|
| `INFER_MAX_W` | `config.py` | `480` | Max width for inference (frame is downscaled before sending to models) |
| `BLUR_RADIUS` | `config.py` | `21` | Gaussian blur kernel size for blur background preset |
| `SIGMA` | `app.js` | `5e-4` | Adaptive smoothing sensitivity — lower = snappier, higher = smoother |

## Gitignore suggestion

Add this to `.gitignore` to avoid committing the downloaded models:

```
models/
__pycache__/
*.pyc
.venv/
```
