import asyncio
from fastapi import APIRouter, HTTPException, BackgroundTasks
from app.schemas.request import GenerationRequest
from app.schemas.response import GenerationResponse
from app.core.queue import job_manager
from app.pipelines.text_to_video import pipeline
from app.utils.hashing import generate_execution_id
from app.core.config import settings
from app.core.logging import telemetry

router = APIRouter()

async def _run_generation_task(job_id: str, request: GenerationRequest):
    """Background execution runner with progress updates."""
    try:
        async def _progress_callback(stage: str, pct: int):
            await job_manager.update_progress(job_id, stage, pct)

        result = await pipeline.execute(
            prompt=request.prompt,
            duration=request.duration or 15,
            style=request.style or "cinematic",
            voice=request.voice if request.voice is not None else True,
            token=request.token,
            job_id=job_id,
            progress_callback=_progress_callback
        )
        await job_manager.complete_job(job_id, result.get("metadata", {}))
    except Exception as e:
        telemetry.emit("job_execution_failed", job_id, {"error": str(e)}, level="error")
        await job_manager.fail_job(job_id, str(e))

@router.post("/generate", response_model=GenerationResponse)
async def generate_video(request: GenerationRequest):
    """
    Initiate video generation workflow.
    Validates payload, initializes queue job, and dispatches processing asynchronously.
    """
    job_id = generate_execution_id()
    telemetry.emit("request_received", job_id, {
        "received_prompt": request.prompt,
        "generation_request_id": job_id,
        "models_used": {
            "llm": settings.LLM_PROVIDER,
            "image": settings.IMAGE_PROVIDER,
            "video": settings.VIDEO_PROVIDER,
            "tts": settings.TTS_PROVIDER
        },
        "generation_parameters": {
            "duration": request.duration or 15,
            "style": request.style or "cinematic",
            "voice": request.voice if request.voice is not None else True,
            "fps": settings.DEFAULT_FPS,
            "resolution": settings.DEFAULT_RESOLUTION
        }
    })
    await job_manager.create_job(job_id, request.model_dump())

    # Launch generation task in background
    asyncio.create_task(_run_generation_task(job_id, request))

    return GenerationResponse(
        job_id=job_id,
        status="queued",
        message="Video generation job queued successfully"
    )
