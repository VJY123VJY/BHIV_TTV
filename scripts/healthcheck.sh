#!/usr/bin/env bash
# ==============================================================================
# BHIV TTV Healthcheck Verification Script
# ==============================================================================
set -euo pipefail

PORT="${PORT:-8000}"
HOST="${HOST:-localhost}"
HEALTH_URL="http://${HOST}:${PORT}/health"
MAX_RETRIES=15
DELAY_SECONDS=3

echo "Pinging BHIV TTV health endpoint at ${HEALTH_URL}..."

for ((i=1; i<=MAX_RETRIES; i++)); do
    RESPONSE=$(curl -sf "${HEALTH_URL}" 2>/dev/null || echo "")
    if [ -n "${RESPONSE}" ]; then
        STATUS=$(echo "${RESPONSE}" | grep -o '"status":"[^"]*"' | cut -d'"' -f4 || echo "")
        FFMPEG=$(echo "${RESPONSE}" | grep -o '"ffmpeg_available":[^,}]*' | cut -d':' -f2 || echo "")
        
        if [ "${STATUS}" = "healthy" ]; then
            echo "✅ Healthcheck PASSED (attempt ${i}/${MAX_RETRIES})"
            echo "   Status: ${STATUS}"
            echo "   FFmpeg Available: ${FFMPEG}"
            exit 0
        fi
    fi
    echo "⏳ Attempt ${i}/${MAX_RETRIES}: Service starting up... retrying in ${DELAY_SECONDS}s"
    sleep "${DELAY_SECONDS}"
done

echo "❌ Healthcheck FAILED after ${MAX_RETRIES} attempts."
echo "Container logs:"
docker compose logs --tail=30
exit 1
