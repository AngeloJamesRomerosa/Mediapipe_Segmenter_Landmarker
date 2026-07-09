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
4 daemon threads per session (LIVE_STREAM mode, capped at pipeline FPS):
  seg_loop   → selfie segmenter alpha mask           (IMAGE mode, shared singleton)
               skipped entirely when preset="none" and outline=off
  pose_loop  → 33 body landmarks + zone masks        (LIVE_STREAM, per-session)
               + Lucas-Kanade optical flow for low-visibility joints
  hand_loop  → 21 landmarks × 2 hands + finger masks (LIVE_STREAM, per-session)
  face_loop  → 468 face landmarks + face oval mask   (LIVE_STREAM, per-session)
    ↓ detect_async() — non-blocking; callback fires on MediaPipe's thread
      as soon as inference finishes, pushing landmark JSON directly to browser
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
| **LIVE_STREAM mode** | MediaPipe model (server) | Temporal tracking between frames — model reuses previous detection instead of re-detecting every frame; built-in landmark smoothing |
| **Optical flow** | `loops.py` (server) | Lucas-Kanade flow for low-visibility pose joints (e.g. feet, wrists) to keep them stable when partially occluded |
| **Per-body-part adaptive exp blend** | `app.js` (browser) | Per-landmark `α = exp(−d²/σ)` applied every rAF tick — sigma tuned per body part so fast-moving parts (hands, feet) snap to fresh detections while stable parts (head, torso) hold steady |

The client-side blend runs at 60 fps regardless of server update rate, so motion looks smooth even at low pipeline FPS settings.

### Per-body-part sigma values

| Body part | Sigma | Behaviour |
|-----------|-------|-----------|
| Feet (29–32) | `1.5e-4` | Snappiest — feet move fast |
| Wrists / hand tips (15–22) | `2e-4` | Very responsive |
| Elbows / ankles (13–14, 27–28) | `3e-4` | Responsive |
| Knees (25–26) | `5e-4` | Medium |
| Shoulders / hips (11–12, 23–24) | `7e-4` | Stable |
| Face landmarks / head (0–10) | `8e-4` | Most stable |
| Hands | `1.5e-4` | Very snappy |
| Face mesh | `8e-4` | Stable |

## Threading model

```
asyncio event loop (main thread)
  │  receives frame binary
  └─► run_in_executor ──► _process_frame (frame worker thread pool)
                                │  apply_bg() → loops.submit(frame)
                                │                    │
                                │             4 daemon threads
                                │             (LIVE_STREAM mode)
                                │                    │
                                │             detect_async() ← non-blocking
                                │             returns immediately
                                │                    │
                                │             MediaPipe callback thread
                                │             fires when inference finishes
                                │             → _push_landmarks()
                                │             asyncio.run_coroutine_threadsafe
                                │             → sends JSON directly to browser
                                │
                                └─► ws.send_bytes(composited JPEG)
```

Landmark JSON is pushed **immediately** when each inference callback fires — it does not wait for the next composited frame response. This minimises the visible lag between body movement and skeleton movement.

### Segmenter CPU optimisation

The selfie segmenter is skipped entirely when the background preset is "None" and the glow outline is off — in that mode its result would be thrown away anyway. This frees significant CPU for the landmark models, reducing lag when no background replacement is active.

## Project structure

```
.
├── backend/
│   ├── app/
│   │   ├── main.py        # FastAPI app, WebSocket endpoint
│   │   ├── composite.py   # apply_bg() — alpha merge and pixel composite
│   │   ├── loops.py       # per-session daemon inference threads (LIVE_STREAM)
│   │   ├── infer.py       # per-model result processors (called from callbacks)
│   │   ├── alpha.py       # mask builders (hand, face, segmenter)
│   │   ├── zones.py       # 8 body-part zone masks (parallel)
│   │   ├── build.py       # background layer builder
│   │   ├── session.py     # per-connection state dataclass
│   │   ├── models.py      # segmenter singleton + LIVE_STREAM per-session factories
│   │   └── config.py      # model URLs/paths, landmark tables, constants
│   └── requirements.txt
├── frontend/
│   ├── index.html          # 3-panel layout + FPS cap + latency controls
│   ├── css/style.css
│   └── js/app.js           # camera capture, WS comms, per-body-part adaptive smoothing
├── TROUBLESHOOTING.txt     # common issues: caching, port conflicts, kill commands
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
git clone https://github.com/AngeloJamesRomerosa/Mediapipe_Segmenter_Landmarker.git
cd Mediapipe_Segmenter_Landmarker
```

**2. Create and activate a virtual environment**

```powershell
# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

> If PowerShell blocks the activation script, run this once first:
> `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

**3. Install dependencies**

```powershell
pip install -r backend/requirements.txt
```

**4. Start the server**

```powershell
uvicorn backend.app.main:app --reload
```

**5. Open in browser**

Go to `http://localhost:8000`, then click **Start** and allow camera access when prompted.

The first time each model is used it downloads from Google's MediaPipe CDN (~30 MB total) and caches in a local `models/` folder. After that, startup is instant.

### Stopping the server

Press **Ctrl+C** in the terminal, or if the process is stuck run this in any PowerShell window:

```powershell
Get-Process -Name "python*" -ErrorAction SilentlyContinue | Stop-Process -Force
```

### Model download notes

| Model | Running mode | Downloaded when | Size |
|-------|-------------|-----------------|------|
| Selfie segmenter | IMAGE — shared singleton | First frame with background/outline active | ~1 MB |
| Pose landmarker | LIVE_STREAM — per-session | Session start | ~5 MB |
| Hand landmarker | LIVE_STREAM — per-session | Session start | ~9 MB |
| Face landmarker | LIVE_STREAM — per-session | Session start (pre-warms) | ~14 MB |

Pose, Hand, and Face landmarkers use **LIVE_STREAM mode** — each session creates its own model instance. `detect_async()` is non-blocking; results arrive via a callback on MediaPipe's internal thread the moment inference finishes, decoupling the inference wait from the loop thread.

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
| Server → Client | text JSON | `{"type":"landmarks", pose:[...], hands:[...], face:[...]}` pushed directly by inference callbacks |

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
| `pose_landmarker_lite.task` | 33 body landmarks + segmentation | LIVE_STREAM — per-session | ~5 MB |
| `hand_landmarker.task` | 21 landmarks × 2 hands | LIVE_STREAM — per-session | ~9 MB |
| `face_landmarker.task` | 468 face landmarks | LIVE_STREAM — per-session | ~14 MB |

## Configuration

| Constant | File | Default | Description |
|----------|------|---------|-------------|
| `INFER_MAX_W` | `config.py` | `480` | Max width for inference (frame is downscaled before sending to models) |
| `BLUR_RADIUS` | `config.py` | `21` | Gaussian blur kernel size for blur background preset |
| `POSE_SIGMA` | `app.js` | see table above | Per-landmark adaptive smoothing — smaller = snappier |
| `HAND_SIGMA` | `app.js` | `1.5e-4` | Hand landmark smoothing — very responsive |
| `FACE_SIGMA` | `app.js` | `8e-4` | Face mesh smoothing — stable |

## Gitignore

The following are excluded from the repo and created at runtime:

```
models/
__pycache__/
*.pyc
.venv/
```
