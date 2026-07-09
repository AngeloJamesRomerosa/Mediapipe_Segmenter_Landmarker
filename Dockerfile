FROM python:3.11-slim-bookworm

# MediaPipe's C++ runtime hard-links against libGLESv2 even on CPU-only mode
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgles2-mesa \
    libgl1 \
    libgomp1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["sh", "-c", "uvicorn backend.app.main:app --host 0.0.0.0 --port ${PORT:-10000}"]
