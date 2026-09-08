# BHIV TTV – Production Infrastructure & Deployment Guide

This document specifies the deployment infrastructure, container orchestration, CI/CD pipeline, and operations manual for the **BHIV Text-to-Video (TTV)** microservice.

---

## 1. System Architecture & Topology

- **Service Model**: Single consolidated FastAPI / Uvicorn container serving both API endpoints and the frontend user interface (`index.html`, `app.js`, `style.css`).
- **Generation Pipeline**:
  `Text Prompt` → `LLM / Story Generation` → `Scene Planning` → `Image Synthesis` → `Video Engine (OpenCV)` → `TTS (EdgeTTS/gTTS/Local)` → `Audio Mixing` → `FFmpeg Rendering` → `MP4 + Sidecar Metadata`.
- **Port Allocation**:
  - Internal Port: `8000/TCP`
  - Host Mapping: `8019:8000` (Port `8019` allocated on production VM)
  - Binding: `0.0.0.0:8000`
  - No additional external database or frontend port required.
- **Storage & Volume Mounts**:
  - Host persistent volume: `./generated:/app/generated`
  - Subdirectories managed automatically:
    - `generated/videos` (final MP4 videos and metadata JSONs)
    - `generated/temp` (temporary processing artifacts)
    - `generated/images` (scene visual frames)
    - `generated/scenes` (scene intermediate assets)
    - `generated/audio` (speech synthesis and background audio)

---

## 2. Infrastructure Components

| File | Purpose |
|---|---|
| `Dockerfile` | Multi-dependency production image based on `python:3.11-slim`, with FFmpeg, OpenCV runtime libraries, curl, Python requirements, and healthcheck. |
| `.dockerignore` | Build optimization filter ignoring `.git`, `__pycache__`, virtual environments, and generated test files. |
| `docker-compose.yml` | Local and staging container orchestration configuration. |
| `docker-compose.production.template.yml` | Production deployment template utilized by GitHub Actions CI/CD to bind commit SHAs. |
| `.github/workflows/cicd.yml` | Automated 4-stage CI/CD pipeline (Validate, Build, Deploy, Rollback) deploying to remote VM over SSH. |
| `scripts/deploy.sh` | One-touch local and VM deployment script. |
| `scripts/healthcheck.sh` | Automated service readiness and FFmpeg validation script. |

---

## 3. Environment Configuration

All runtime variables are driven via `.env`. A complete template is available in `.env.example`.

### Key Parameters

| Variable | Default Value | Description |
|---|---|---|
| `HOST` | `0.0.0.0` | Bind address for FastAPI/Uvicorn |
| `PORT` | `8000` | Port for the service |
| `ENVIRONMENT` | `production` | Environment mode (`development` / `production`) |
| `LOG_LEVEL` | `INFO` | Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |
| `GOVERNANCE_LOCK` | `ON` | System governance enforcement flag |
| `WRAPPER_TOKEN` | `<secure-token>` | Authentication / bearer token |
| `LLM_PROVIDER` | `local` | Provider for story generation (`local`, `openai`, `gemini`) |
| `IMAGE_PROVIDER` | `local` | Provider for image synthesis (`local`, `openai`) |
| `VIDEO_PROVIDER` | `opencv` | Video generation engine (`opencv`, `external`) |
| `TTS_PROVIDER` | `local` | Text-to-speech engine (`local`, `edgetts`, `gtts`) |
| `VISION_PROVIDER` | `standard` | Visual consistency provider (`standard`, `enhanced`) |
| `OPENAI_API_KEY` | `""` | Optional OpenAI API Key (required if provider is `openai`) |
| `GEMINI_API_KEY` | `""` | Optional Gemini API Key (required if provider is `gemini`) |

---

## 4. GitHub Actions CI/CD Pipeline

The workflow defined in [`.github/workflows/cicd.yml`](.github/workflows/cicd.yml) runs on pushes to `main` or manual trigger via `workflow_dispatch`.

```mermaid
flowchart LR
    A[Push to main] --> B[Validate]
    B --> C[Build & Push]
    C --> D[Deploy to VM]
    D -- Failure --> E[Automatic Rollback]
    D -- Success --> F[Release Registered]
```

### Pipeline Stages

1. **`validate`**:
   - Generates test `docker-compose.production.yml` with commit SHA.
   - Validates compose schema and configurations via `docker compose config`.
   - Uploads verified deployment artifacts.
2. **`build`**:
   - Sets up Docker Buildx.
   - Authenticates with Docker Hub.
   - Builds `bhiv/ttv-service:<commit_sha>` and `bhiv/ttv-service:latest`.
   - Pushes images to Docker Hub.
3. **`deploy`**:
   - Downloads verified deployment artifact.
   - Injects `.env` securely from repository secrets (`TTV_BACKEND_ENV` or `BACKEND_ENV_FILE`).
   - Packages `deployment.tar.gz` and transfers to VM via SCP (`sshpass`).
   - Pulls container image on VM and executes zero-downtime rolling restart (`docker compose up -d`).
   - Executes polling health checks against `http://localhost:8019/health`.
   - Records successful release in `docs/RELEASE_HISTORY.md` and backs up to `/var/tmp/BHIV_TTV/`.
   - Prunes unused Docker images older than 7 days.
4. **`rollback`**:
   - Automatically triggered if health verification fails.
   - Reads `docs/RELEASE_HISTORY.md` on the VM to extract the last known healthy commit SHA.
   - Re-deploys the previous healthy version and logs `ROLLBACK_SUCCESS`.

### Required GitHub Repository Secrets

Configure the following secrets under **Settings > Secrets and variables > Actions**:

| Secret Name | Description | Example |
|---|---|---|
| `DOCKER_USERNAME` | Docker Hub username | `bhiv` |
| `DOCKER_PASSWORD` | Docker Hub access token or password | `dckr_pat_...` |
| `TTV_BACKEND_ENV` | Production `.env` file contents | *(Full .env content)* |
| `VM_IP` | Production VM IPv4 address | `192.168.1.100` |
| `VM_PORT` | SSH port of production VM | `22` |
| `VM_USERNAME` | SSH username on VM | `ubuntu` |
| `VM_PASSWORD` | SSH password for deployment user | `...` |

---

## 5. Local Quickstart & Verification

### Using Docker Compose

1. Prepare your local `.env` file:
   ```bash
   cp .env.example .env
   ```
2. Build and run containers:
   ```bash
   docker compose up --build -d
   ```
3. Inspect container status:
   ```bash
   docker compose ps
   docker compose logs -f
   ```
4. Verify health endpoint:
   ```bash
   curl -s http://localhost:8000/health
   ```
   Expected response:
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
5. Access the Web UI in your browser:
   `http://localhost:8000/`

### Using Helper Scripts

```bash
# Start deployment and run healthcheck
bash scripts/deploy.sh

# Run healthcheck independently
bash scripts/healthcheck.sh
```
