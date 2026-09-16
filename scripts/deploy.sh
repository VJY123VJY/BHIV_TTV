#!/usr/bin/env bash
# ==============================================================================
# BHIV TTV Local / VM Deployment Script
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${ROOT_DIR}"

echo "====================================================="
echo " Starting BHIV TTV Service Deployment"
echo " Working directory: ${ROOT_DIR}"
echo "====================================================="

# Check prerequisites
command -v docker >/dev/null 2>&1 || { echo "❌ Docker is required but not installed. Aborting."; exit 1; }
command -v docker compose >/dev/null 2>&1 || { echo "❌ Docker Compose is required but not installed. Aborting."; exit 1; }

# Check for .env file
if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        echo "⚠️ .env not found. Creating .env from .env.example..."
        cp .env.example .env
    else
        echo "❌ Neither .env nor .env.example found. Aborting."
        exit 1
    fi
fi

# Check for frontend.env file
if [ ! -f "frontend.env" ]; then
    if [ -f "frontend.env.example" ]; then
        echo "⚠️ frontend.env not found. Creating frontend.env from frontend.env.example..."
        cp frontend.env.example frontend.env
    else
        echo "VIDEO_PROVIDER=opencv" > frontend.env
    fi
fi

# Ensure persistent directories exist
mkdir -p generated/videos generated/temp generated/images generated/scenes generated/audio

echo "📦 Building and starting containers..."
docker compose build
docker compose up -d --remove-orphans

echo "🔍 Validating service health..."
bash "${SCRIPT_DIR}/healthcheck.sh"

echo "====================================================="
echo "✅ BHIV TTV services deployed successfully!"
echo "   Web UI (Frontend):  http://localhost:8021/"
echo "   API Health (Backend): http://localhost:8019/health"
echo "   API Documentation:   http://localhost:8019/docs"
echo "====================================================="

