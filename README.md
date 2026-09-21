# Unified Text-to-Video (TTV) AI Studio

A consolidated, production-ready, enterprise-grade Text-to-Video system intelligently engineered by auditing and merging 19 individual repositories into a single coherent architecture.

---

## 📖 Table of Contents

1. [Project Overview](#project-overview)
2. [Unified Architecture](#unified-architecture)
3. [The 14-Stage Text-to-Video Pipeline](#the-14-stage-text-to-video-pipeline)
4. [Text-to-Vision Integration](#text-to-vision-integration)
5. [AI Provider & Adapter System](#ai-provider--adapter-system)
6. [Repository Consolidation Report (All 19 Repositories)](#repository-consolidation-report)
7. [Installation & Setup](#installation--setup)
8. [Environment Variables](#environment-variables)
9. [Docker & Docker Compose](#docker--docker-compose)
10. [REST API Reference](#rest-api-reference)
11. [Interactive Frontend](#interactive-frontend)
12. [Testing & Verification](#testing--verification)
13. [Troubleshooting](#troubleshooting)

---

## 1. Project Overview

This project provides an end-to-end AI platform that transforms natural language text prompts into complete, multi-scene, cinematic videos with synchronized speech narration, ambient musical scores, dynamic camera motion, and validated H.264/MP4 encoding.

**Core Workflow:**
```
Prompt / Script
  → Video Format (16:9 or 9:16)
  → Video Quality (Standard 720p / High 1080p / Ultra 4K)
  → Reference Upload or Reference URL
  → Visual Style
  → Language
  → Generate Voice + Visuals
  → Scene Video
  → Automatic Lip Sync
  → FFmpeg Assembly
  → Final MP4
```

---

## 2. Unified Architecture

```
                                USER / FRONTEND
                                      │
                                      ▼
                           REST API (FastAPI Router)
                         POST /api/v1/generate
                         GET  /api/v1/jobs/{job_id}
                         GET  /api/v1/videos/{video_id}
                         GET  /health
                                      │
                                      ▼
                        TextToVideoPipeline Orchestrator
                                      │
         ┌────────────────────────────┼────────────────────────────┐
         ▼                            ▼                            ▼
   Validation & Auth          Governance Guard             Telemetry Emitter
                                      │
                                      ▼
                              Prompt Processing
                                      │
                                      ▼
                        LLM / Story Engine (OpenAI / Gemini / Local)
                                      │
                                      ▼
                        Scene Breakdown & Shot Planning
                        (CinematicComposer: camera, pacing)
                                      │
                                      ▼
                        Visual Consistency Engine
                        (Character anchor, style lock, color grade)
                                      │
                                      ▼
                         Visual Generation Stage
                    ┌─────────────────┴─────────────────┐
                    ▼                                   ▼
               Text-to-Image                       Text-to-Video
             (Keyframe Synthesis)              (Kinematic Motion Engine)
                    └─────────────────┬─────────────────┘
                                      ▼
                               Scene Video Clips
                                      │
                                      ▼
                             Speech Synthesis (TTS)
                                      │
                                      ▼
                                Audio Mixing
                         (Voice + Ambient Music + Ducking)
                                      │
                                      ▼
                               FFmpeg Composer
                    (Concatenation + H.264 Transcoding + SRT)
                                      │
                                      ▼
                               Production MP4
                                      │
                                      ▼
                           Artifact & Sidecar Storage
                          (/generated/videos + metadata.json)
```

---

## 3. The 14-Stage Text-to-Video Pipeline

The system is coordinated by `TextToVideoPipeline` located in `backend/app/pipelines/text_to_video.py`:

1. **Validate settings & prompt**: Aspect ratio, quality, language, FPS, duration, and prompt.
2. **Process reference**: Upload ID or public URL → validated retrieval → still frame.
3. **Understand prompt**: Subject, setting, lighting, mood.
4. **Generate story & script**: Multi-act narrative.
5. **Generate scenes**: Timed shots with camera motion.
6. **Localize narration**: Scene dialogue is rewritten in the selected language before TTS.
7. **Visual prompts**: Style lock, aspect-aware framing, reference identity.
8. **Image generation**: Keyframes at the selected `width x height`.
9. **Scene video generation**: Motion clips at the same resolution and FPS.
10. **Language-specific TTS**: Edge TTS / gTTS for `en`, `hi`, `mr`, `gu`, `fr`, `es`, `de`.
11. **Automatic lip sync**: Viseme mouth animation (optional Wav2Lip checkpoint). Scenes without a visible face are skipped, not failed.
12. **Audio mix + FFmpeg assembly**: Pad/scale without stretching, enforce output dimensions.
13. **Validate & store**: Confirm MP4 dimensions, persist sidecar metadata.
14. **Return result**: Job metadata including selected settings.

---

## 4. Text-to-Vision Integration

Text-to-Vision functionality is directly integrated as the **Visual Generation Stage** of the pipeline rather than a detached tool. It provides:
- **Style Presets**: `realistic`, `cartoon`, `cinematic`, `3d`, `anime`, `fantasy`, `cyberpunk`.
- **Subject Anchoring**: Cross-scene prompt tags preserving subject appearance, wardrobe, and features.
- **Lighting & Color Palette Locks**: Synchronizes ambient color gradients across scene transitions.
- **Image Validation**: Probes generated keyframes before passing them to the video motion synthesizer.

---

## 5. AI Provider & Adapter System

The system avoids vendor lock-in through pluggable abstract adapters in `backend/app/adapters/`:

| Subsystem | Available Providers | Config Key |
|---|---|---|
| **LLM** | `local` (offline NLP engine), `openai` (GPT-4o), `gemini` (Gemini 1.5) | `LLM_PROVIDER` |
| **Image** | `local` (procedural / PIL / OpenCV), `openai` (DALL-E 3) | `IMAGE_PROVIDER` |
| **Video** | `opencv` (kinematic motion synthesis), `external` (API router) | `VIDEO_PROVIDER` |
| **TTS** | `local` (EdgeTTS / gTTS; English-only offline formant fallback) | `TTS_PROVIDER` |
| **Vision** | `standard` (consistency & continuity engine) | `VISION_PROVIDER` |
| **Reference** | Direct HTTP(S) media + `yt-dlp` social sources | — |
| **Lip-sync** | `viseme` (offline mouth animation) or `wav2lip` | `LIPSYNC_PROVIDER` |

---

## 6. Repository Consolidation Report

| # | Repository | Retained Functionality | Reason |
|---|---|---|---|
| 1 | `VJY123VJY/tttv_adapter` | Request hashing, execution tracing, contract validation | Provides deterministic trace safety and schema validation. |
| 2 | `VJY123VJY/ttv10` | Style configurations, CPU fallback patch, artifact metadata | Hardware patch ensures CPU execution without PyTorch CUDA crashes. |
| 3 | `VJY123VJY/ttv_converegence` | Cinematic Composer (shot planning, camera sequencing, pacing) | Establishes cinematic shot planning and camera choreography. |
| 4 | `VJY123VJY/ttv_adapter14` | Modular visual processing pipeline (scene representation, temporal smoothing) | Clean breakdown of visual stages from scene to renderable representation. |
| 5 | `VJY123VJY/adapter_ttv15` | Dual engine router (local vs external) and evaluation logging | Enables seamless switching between local and cloud providers. |
| 6 | `VJY123VJY/ttv_adapter1` | Gateway security, authentication token validation, error payloads | Guarantees non-bypassable governance boundary discipline. |
| 7 | `VJY123VJY/ttv12` | Clean Pydantic schemas, standard response builder, prompt purity | Provides clean typed input/output contracts. |
| 8 | `VJY123VJY/ttv11` | 3-Layer Universal Adapter pattern (Normalize -> Execute -> Response) | Architectural discipline separating validation from execution. |
| 9 | `VJY123VJY/ttv9` | Formal execution lifecycle states and structured session telemetry | Complete observability across every phase of execution. |
| 10 | `VJY123VJY/ttv8` | FastAPI REST API routing (`/generate`, `/status`, `/health`) | Proven REST endpoint conventions and validation bounds. |
| 11 | `VJY123VJY/ttv7` | Governance lock mechanism (`GOVERNANCE_LOCK=ON`) | Prevents direct uncontrolled access to raw generators. |
| 12 | `VJY123VJY/ttv6` | Storage abstraction (`BucketAdapter`), fail-closed exception handling | Isolates file system storage from business logic. |
| 13 | `VJY123VJY/ttv5` | Probe-based loader with graceful fallbacks | Prevents service crashes when optional ML packages are missing. |
| 14 | `VJY123VJY/ttv3` | Metadata sidecar generation (.json paired with .mp4) | Guarantees every video has an auditable metadata manifest. |
| 15 | `VJY123VJY/text_to_video6` | Pipeline wrapper and storage integration | Merged with `ttv6` to unify storage and execution wrappers. |
| 16 | `VJY123VJY/Text_to_vision` | Categorized `FailureHandler` and recovery strategies | Structured failure categorization and recovery. |
| 17 | `VJY123VJY/text_to_video2` | OpenCV video synthesis (`cv2.VideoWriter`) and frame rendering | Generates real, playable MP4 video files locally. |
| 18 | `VJY123VJY/Text_to_vision2` | Token-locked execution gateway and configuration schema | Clean configuration loading and execution tokens. |
| 19 | `VJY123VJY/Text_to_video` | System coordinator and telemetry emitter architecture | Consolidated into the unified core logging system. |

---

## 7. Installation & Setup

### Prerequisites
- Python 3.10+
- FFmpeg installed and available in system PATH

### Quick Start
```bash
# 1. Clone or navigate to the repository
cd text_to_video_final

# 2. Install dependencies
pip install -r backend/requirements.txt

# 3. Configure environment
cp .env.example .env

# 4. Start the server
python scripts/run_dev.py
```
Visit `http://localhost:8000` in your web browser.

---

## 8. Environment Variables

Configured in `.env`:

```ini
# Server
HOST=0.0.0.0
PORT=8000
ENVIRONMENT=production
LOG_LEVEL=INFO

# Governance & Security
GOVERNANCE_LOCK=ON
WRAPPER_TOKEN=TTV_SECURE_GOVERNED_TOKEN_2026

# Provider Selection (local | openai | gemini)
LLM_PROVIDER=local
IMAGE_PROVIDER=local
VIDEO_PROVIDER=opencv
TTS_PROVIDER=local
VISION_PROVIDER=standard

# Optional Cloud API Keys
OPENAI_API_KEY=
GEMINI_API_KEY=

# Storage Paths
OUTPUT_DIR=generated/videos
TEMP_DIR=generated/temp
IMAGES_DIR=generated/images
SCENES_DIR=generated/scenes
AUDIO_DIR=generated/audio
```

---

## 9. Docker & Docker Compose

Run the entire platform inside a single self-contained container:

```bash
# Build and start the container
docker compose up --build -d

# Check service health
docker compose ps

# View logs
docker compose logs -f
```

The container automatically installs FFmpeg, sets up directories, and serves both the API and the interactive UI on port `8000`.

---

## 10. REST API Reference

### 1. Initiate Generation
`POST /api/v1/generate`
```json
{
  "prompt": "A small robot explores a futuristic city at sunset.",
  "duration": 15,
  "style": "cinematic",
  "voice": true,
  "resolution": "1280x720"
}
```
**Response (200 OK):**
```json
{
  "job_id": "exec_20260907_120000_a1b2c3d4",
  "status": "queued",
  "message": "Video generation job queued successfully"
}
```

### 2. Poll Job Progress
`GET /api/v1/jobs/{job_id}`
```json
{
  "job_id": "exec_20260907_120000_a1b2c3d4",
  "status": "processing",
  "progress": 68,
  "stage": "rendering_scene_videos",
  "created_at": "2026-09-07T12:00:00Z",
  "updated_at": "2026-09-07T12:00:05Z"
}
```

### 3. Retrieve Video Artifact
`GET /api/v1/videos/{video_id}`
Returns metadata sidecar JSON. Append `?stream=true` to stream the raw MP4 file directly.

### 4. System Health
`GET /health`
```json
{
  "status": "healthy",
  "service": "unified-text-to-video",
  "version": "1.0.0",
  "providers": {
    "llm": "local",
    "image": "local",
    "video": "opencv",
    "tts": "local",
    "vision": "standard"
  },
  "ffmpeg_available": true
}
```

---

## 11. Interactive Frontend

The built-in web studio is accessible at `http://localhost:8000`:
- **Prompt Input**: Enter natural language descriptions.
- **Duration Slider**: Adjust duration from 5s to 60s.
- **Style Selector**: Choose Cinematic, Anime, Cyberpunk, Realistic, or Fantasy.
- **Voice Narration Toggle**: Enable/disable audio speech synthesis.
- **Live Pipeline Tracker**: Visual progress bar tracking each stage.
- **Integrated Video Player**: Preview and download the generated MP4.

---

## 12. Testing & Verification

### Run Automated Unit & Integration Tests
```bash
pytest backend/tests/ -v
```

### Run Full End-to-End Test
```bash
python scripts/test_e2e.py
```
This runs the full 14-stage pipeline for *"A small robot explores a futuristic city at sunset."*, generating and validating the final playable MP4 artifact.

---

## 13. Troubleshooting

- **`ffmpeg: command not found`**: Ensure FFmpeg is installed and added to system PATH (`ffmpeg -version`).
- **Missing API keys**: If `OPENAI_API_KEY` or `GEMINI_API_KEY` are not provided, set `LLM_PROVIDER=local` and `IMAGE_PROVIDER=local`. The local engines generate full videos without external dependencies.
- **Port 8000 already in use**: Change `PORT=8080` in `.env`.
