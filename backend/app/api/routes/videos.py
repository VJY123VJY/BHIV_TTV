from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse
from app.services.storage_service import storage_service

router = APIRouter()

@router.get("/videos/{video_id}")
async def get_video(video_id: str, stream: bool = Query(default=False, description="Stream the MP4 file directly")):
    """Retrieve video metadata or stream the raw MP4 media file."""
    try:
        if stream:
            video_path = storage_service.get_video_path(video_id)
            return FileResponse(
                path=str(video_path),
                media_type="video/mp4",
                filename=f"{video_id}.mp4"
            )
        metadata = storage_service.get_video_metadata(video_id)
        return metadata
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Video artifact '{video_id}' not found.")
