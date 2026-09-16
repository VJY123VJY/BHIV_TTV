#!/usr/bin/env bash
# ==============================================================================
# BHIV TTV Healthcheck Verification Script
# ==============================================================================
set -euo pipefail

BACKEND_PORT="${BACKEND_PORT:-8019}"
FRONTEND_PORT="${FRONTEND_PORT:-8021}"
HOST="${HOST:-localhost}"

BACKEND_HEALTH_URL="http://${HOST}:${BACKEND_PORT}/health"
FRONTEND_URL="http://${HOST}:${FRONTEND_PORT}/"
MAX_RETRIES=15
DELAY_SECONDS=3

echo "Pinging BHIV TTV Backend health endpoint at ${BACKEND_HEALTH_URL}..."
echo "Pinging BHIV TTV Frontend endpoint at ${FRONTEND_URL}..."

BACKEND_OK=0
FRONTEND_OK=0

for ((i=1; i<=MAX_RETRIES; i++)); do
    # Check Backend
    if [ "${BACKEND_OK}" -eq 0 ]; then
        RESPONSE=$(curl -sf "${BACKEND_HEALTH_URL}" 2>/dev/null || curl -sf "http://${HOST}:8000/health" 2>/dev/null || echo "")
        if [ -n "${RESPONSE}" ]; then
            STATUS=$(echo "${RESPONSE}" | grep -o '"status":"[^"]*"' | cut -d'"' -f4 || echo "")
            if [ "${STATUS}" = "healthy" ]; then
                BACKEND_OK=1
                echo "✅ Backend Healthcheck PASSED"
            fi
        fi
    fi

    # Check Frontend
    if [ "${FRONTEND_OK}" -eq 0 ]; then
        if curl -sf "${FRONTEND_URL}" >/dev/null 2>&1 || curl -sf "http://${HOST}:8000/" >/dev/null 2>&1; then
            FRONTEND_OK=1
            echo "✅ Frontend Healthcheck PASSED"
        fi
    fi

    if [ "${BACKEND_OK}" -eq 1 ] && [ "${FRONTEND_OK}" -eq 1 ]; then
        echo "🎉 All BHIV TTV services are HEALTHY and ready!"
        exit 0
    fi

    echo "⏳ Attempt ${i}/${MAX_RETRIES}: Services starting up (backend=${BACKEND_OK}, frontend=${FRONTEND_OK})... retrying in ${DELAY_SECONDS}s"
    sleep "${DELAY_SECONDS}"
done

echo "❌ Healthcheck FAILED after ${MAX_RETRIES} attempts."
echo "Container status & logs:"
docker compose ps
docker compose logs --tail=30
exit 1

