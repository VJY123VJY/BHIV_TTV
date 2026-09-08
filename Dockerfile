# ==============================================================================
# BHIV TTV (Text-to-Video) Production Dockerfile
# ==============================================================================
FROM python:3.11-slim

# Install system dependencies:
# - FFmpeg for video/audio assembly & rendering
# - OpenCV GUI/runtime dependencies (libsm6, libxext6, libgl1, libglib2.0-0)
# - curl for Docker container health check
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install Python dependencies
COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

# Copy backend application, frontend static assets, and supporting modules
COPY backend ./backend
COPY frontend ./frontend
COPY models ./models
COPY training ./training

# Ensure persistent directories exist for generated artifacts
RUN mkdir -p generated/videos generated/temp generated/images generated/scenes generated/audio

# Environment configurations
ENV PYTHONPATH=/app/backend:/app
ENV PYTHONUNBUFFERED=1

# Expose microservice port
EXPOSE 8000

# Container Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Launch FastAPI application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
